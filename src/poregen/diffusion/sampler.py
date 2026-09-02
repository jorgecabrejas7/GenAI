"""DDPM patch sampler and full-volume generator for PoreGen LDM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit

logger = logging.getLogger(__name__)

_NEIGHBOR_DIRS = [
    ( 1, 0, 0), (-1, 0, 0),
    ( 0, 1, 0), ( 0,-1, 0),
    ( 0, 0, 1), ( 0, 0,-1),
]

# Availability states (must match conditioning.py)
_NB_OOB     = 0
_NB_EXISTS  = 1
_NB_UNKNOWN = 2

# Global porosity conditioning is clamped to the training distribution
# range (EDA ground truth: min 0.002, max 0.107) to avoid OOD extrapolation.
_POR_MIN = 0.002
_POR_MAX = 0.107


class DDPMSampler:
    """Patch-level DDPM reverse diffusion sampler.

    Parameters
    ----------
    model : UNet3DDenoiser
    schedule : DDPMSchedule
    device : torch.device
    """

    def __init__(self, model: torch.nn.Module, schedule: Any, device: torch.device) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device

    @torch.no_grad()
    def sample_patch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: float,
        local_por: float,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run the full T-step DDPM reverse process for one patch.

        Parameters
        ----------
        nb_latents          : (6, C, D, H, W) — known neighbor latents (zeros for unavail)
        nb_avail            : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        pos_frac            : (3,) float — normalised patch position in volume
        global_por          : float
        local_por           : float
        return_intermediates: if True, return (final, list[Tensor]) where the list contains
                              the latent after each denoising step, on CPU float32

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent, or
        ((C, D, H, W), list[(C, D, H, W)]) when return_intermediates=True
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        C = nb_latents.shape[1]
        D = nb_latents.shape[2]

        # Add batch dim
        nb_l   = nb_latents.unsqueeze(0).to(self.device)     # (1,6,C,D,D,D)
        nb_a   = nb_avail.unsqueeze(0).to(self.device)       # (1,6)
        pos    = pos_frac.unsqueeze(0).to(self.device)       # (1,3)
        g_por  = torch.tensor([global_por], dtype=torch.float32, device=self.device)
        l_por  = torch.tensor([local_por],  dtype=torch.float32, device=self.device)

        x = torch.randn(1, C, D, D, D, device=self.device)
        intermediates: list[torch.Tensor] = [] if return_intermediates else None  # type: ignore[assignment]

        for t_idx in reversed(range(schedule.T)):
            t = torch.tensor([t_idx], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_l, nb_a, pos, g_por, l_por)
            x = schedule.p_sample(x, t, eps_pred)
            if return_intermediates:
                intermediates.append(x.squeeze(0).float().cpu())

        result = x.squeeze(0).float()
        if return_intermediates:
            return result, intermediates
        return result

    @torch.no_grad()
    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Run the full T-step DDPM reverse process for a batch of patches.

        Parameters
        ----------
        nb_latents : (B, 6, C, D, D, D) — neighbor latents, already on device
        nb_avail   : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN, already on device
        pos_frac   : (B, 3) float — normalised patch positions, already on device
        global_por : (B,) float — global porosity, already on device
        local_por  : (B,) float — local porosity, already on device

        Returns
        -------
        (B, C, D, D, D) float32 on device
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        B = nb_latents.shape[0]
        C = nb_latents.shape[2]
        D = nb_latents.shape[3]

        x = torch.randn(B, C, D, D, D, device=self.device)
        for t_idx in reversed(range(schedule.T)):
            t = torch.full((B,), t_idx, dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_latents, nb_avail, pos_frac, global_por, local_por)
            x = schedule.p_sample(x, t, eps_pred)
        return x.float()


class DDIMSampler:
    """DDIM patch sampler — deterministic inference in n_steps < T steps.

    Parameters
    ----------
    model    : UNet3DDenoiser
    schedule : DDPMSchedule
    device   : torch.device
    n_steps  : int — number of denoising steps (default 50)
    s_por    : float — porosity guidance scale; 1.0 = un-guided (default)
    s_nb     : float — neighbour guidance scale; 1.0 = un-guided (default)

    When both scales are 1.0 the standard single-pass full-conditional denoiser is
    used.  Any other combination activates the 3-pass nested CFG decomposition:

        eps_uncond = model(z_t, t, nb, ALL_UNK, pos, por, drop_por=True)
        eps_por    = model(z_t, t, nb, ALL_UNK, pos, por, drop_por=False)
        eps_full   = model(z_t, t, nb, REAL,    pos, por, drop_por=False)
        eps = eps_uncond + s_por*(eps_por - eps_uncond) + s_nb*(eps_full - eps_por)

    At s_por=s_nb=1 this telescopes to eps_full — exact un-guided equality.
    Position is always on (never dropped) in all three passes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        schedule: Any,
        device: torch.device,
        n_steps: int = 50,
        s_por: float = 1.0,
        s_nb: float = 1.0,
    ) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.n_steps  = n_steps
        self.s_por    = s_por
        self.s_nb     = s_nb
        self.guided   = not (s_por == 1.0 and s_nb == 1.0)
        T = schedule.T
        ts = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long)
        # Store as Python ints for torch.compile compatibility (no dynamic shapes)
        self._timesteps: list[int] = ts.flip(0).tolist()   # [T-1, ..., 0]

    def _guided_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
        autocast_dtype: torch.dtype,
    ) -> torch.Tensor:
        """3-pass nested CFG decomposition.

        Position is on in all three calls.  The ALL_UNKNOWN passes rely on
        UNet3DDenoiser._build_nb_spatial masking latents by NB_EXISTS, so they
        are structurally independent of nb_latents values.
        """
        B = x.shape[0]
        all_unk  = torch.full_like(nb_avail, _NB_UNKNOWN)
        drop_all = torch.ones(B, dtype=torch.bool, device=self.device)

        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            eps_uncond = self.model(x, t, nb_latents, all_unk,  pos_frac,
                                    global_por, local_por, drop_all)
            eps_por    = self.model(x, t, nb_latents, all_unk,  pos_frac,
                                    global_por, local_por, None)
            eps_full   = self.model(x, t, nb_latents, nb_avail, pos_frac,
                                    global_por, local_por, None)

        return (
            eps_uncond
            + self.s_por * (eps_por  - eps_uncond)
            + self.s_nb  * (eps_full - eps_por)
        )

    @torch.no_grad()
    def sample_patch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: float,
        local_por: float,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run DDIM reverse process for one patch.

        Parameters
        ----------
        nb_latents          : (6, C, D, H, W) — known neighbor latents (zeros for unavail)
        nb_avail            : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        pos_frac            : (3,) float — normalised patch position in volume
        global_por          : float
        local_por           : float
        return_intermediates: if True, return (final, list[Tensor]) where the list contains
                              the latent after each denoising step, on CPU float32

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent, or
        ((C, D, H, W), list[(C, D, H, W)]) when return_intermediates=True
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        C = nb_latents.shape[1]
        D = nb_latents.shape[2]

        nb_l  = nb_latents.unsqueeze(0).to(self.device)
        nb_a  = nb_avail.unsqueeze(0).to(self.device)
        pos   = pos_frac.unsqueeze(0).to(self.device)
        g_por = torch.tensor([global_por], dtype=torch.float32, device=self.device)
        l_por = torch.tensor([local_por],  dtype=torch.float32, device=self.device)

        x = torch.randn(1, C, D, D, D, device=self.device)
        intermediates: list[torch.Tensor] = [] if return_intermediates else None  # type: ignore[assignment]

        for i, t_val in enumerate(self._timesteps[:-1]):
            t_prev_val = self._timesteps[i + 1]
            t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
            t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
            if self.guided:
                eps_pred = self._guided_eps(x, t, nb_l, nb_a, pos, g_por, l_por,
                                            autocast_dtype)
            else:
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    eps_pred = self.model(x, t, nb_l, nb_a, pos, g_por, l_por)
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
            if return_intermediates:
                intermediates.append(x.squeeze(0).float().cpu())

        result = x.squeeze(0).float()
        if return_intermediates:
            return result, intermediates
        return result

    @torch.no_grad()
    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_x0_saturation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        """Run DDIM reverse process for a batch of patches.

        Parameters
        ----------
        nb_latents : (B, 6, C, D, D, D) — neighbor latents, already on device
        nb_avail   : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN, already on device
        pos_frac   : (B, 3) float — normalised patch positions, already on device
        global_por : (B,) float — global porosity, already on device
        local_por  : (B,) float — local porosity, already on device
        return_x0_saturation : if True, also return the fraction of x0-prediction
            elements hitting the ±10 clamp inside ddim_step, averaged over all
            denoising steps — an off-manifold diagnostic.

        Returns
        -------
        (B, C, D, D, D) float32 on device — or (samples, sat_frac) when
        return_x0_saturation=True.
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        B = nb_latents.shape[0]
        C = nb_latents.shape[2]
        D = nb_latents.shape[3]

        x = torch.randn(B, C, D, D, D, device=self.device)
        sat_sum = torch.zeros((), device=self.device)
        n_steps = 0
        for i, t_val in enumerate(self._timesteps[:-1]):
            t_prev_val = self._timesteps[i + 1]
            t      = torch.full((B,), t_val,      dtype=torch.long, device=self.device)
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=self.device)
            if self.guided:
                eps_pred = self._guided_eps(x, t, nb_latents, nb_avail, pos_frac,
                                            global_por, local_por, autocast_dtype)
            else:
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    eps_pred = self.model(x, t, nb_latents, nb_avail, pos_frac,
                                         global_por, local_por)
            if return_x0_saturation:
                x0_pred = schedule.predict_x0(x, t, eps_pred)
                sat_sum += (x0_pred.abs() >= 10.0).float().mean()
                n_steps += 1
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
        if return_x0_saturation:
            return x.float(), (sat_sum / max(n_steps, 1)).item()
        return x.float()


class VolumeGenerator:
    """Generate a full synthetic 3D volume using a two-phase checkerboard schedule.

    Patches are classified by checkerboard parity ``(iz + iy + ix) % 2``,
    matching the checkerboard parity convention used across the LDM pipeline:

    - Phase 1 (parity 0, "anchors"): every in-bounds neighbor is UNKNOWN
      (zero latent) and every out-of-bounds neighbor is OOB. Neighbors of a
      parity-0 patch are always parity-1, hence never yet generated — this
      holds structurally, with no need to track generation state.
    - Phase 2 (parity 1, "non-anchors"): every in-bounds neighbor is EXISTS
      (the real, now-completed latent) and every out-of-bounds neighbor is
      OOB. Neighbors of a parity-1 patch are always parity-0, hence always
      completed by the end of phase 1.

    This reproduces the training-time conditioning distribution exactly —
    no patch ever sees a mix of EXISTS and UNKNOWN neighbors.

    Assembly uses **cosine-taper overlap-add blending**: each decoded 64³ patch
    is multiplied by a 3D periodic-Hann window
    ``w3d[i,j,k] = sin²(πi/P)·sin²(πj/P)·sin²(πk/P)``
    and accumulated into floating-point sum and weight buffers.  After all
    patches the output is ``sum_vol / weight_vol`` (epsilon-guarded).  For
    50%-overlap (stride 32) the 1D windows form a perfect partition of unity in
    the interior, so blended interior values equal the unweighted patch values.
    The three zero-weight boundary planes (index 0 on each axis) resolve to 0.

    Parameters
    ----------
    sampler       : DDPMSampler or DDIMSampler
    vae           : VAE model with .decoder, .xct_head, .mask_head attributes
    device        : torch.device
    patch_size    : int — voxel side length of each patch (default 64)
    patch_stride  : int — stride between patch origins; should match the
                    stride used during LDM training (default 32)
    latent_size   : int — spatial side length of the latent (default 16)
    latent_mean   : float | (C,1,1,1) tensor — per-channel normalisation mean;
                    generated latents are denormalised ``z*std + mean`` before decoding
    latent_std    : float | (C,1,1,1) tensor — per-channel normalisation std
    voxel_size_mm : float — physical voxel size in millimetres (default 0.025 = 25 µm)
    """

    def __init__(
        self,
        sampler: Any,
        vae: torch.nn.Module,
        device: torch.device,
        patch_size: int = 64,
        patch_stride: int = 32,
        latent_size: int = 16,
        latent_mean: torch.Tensor | float = 0.0,
        latent_std: torch.Tensor | float = 1.0,
        voxel_size_mm: float = 0.025,
    ) -> None:
        self.sampler       = sampler
        self.vae           = vae
        self.device        = device
        self.patch_size    = patch_size
        self.patch_stride  = patch_stride
        self.latent_size   = latent_size
        self.latent_mean   = latent_mean
        self.latent_std    = latent_std
        self.voxel_size_mm = voxel_size_mm
        self.z_channels    = sampler.model.cfg.z_channels

    @staticmethod
    def _parity(gi: tuple[int, int, int]) -> int:
        return (gi[0] + gi[1] + gi[2]) % 2

    def _generate_latents(
        self,
        volume_shape: tuple[int, int, int],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
        gen_batch_size: int = 32,
    ) -> tuple[dict[tuple[int, int, int], torch.Tensor], dict[tuple[int, int, int], tuple[int, int, int]]]:
        """Run the two-phase checkerboard schedule and return per-patch latents.

        Parameters
        ----------
        volume_shape    : (D, H, W) in voxels
        target_porosity : global VVF target for conditioning (None = 0.05 fallback)
        autocast_dtype  : AMP dtype for the denoiser
        local_por_map   : optional dict mapping (iz, iy, ix) grid indices to per-patch
                          local porosity; overrides target_porosity for local_por only
        gen_batch_size  : number of patches to sample in parallel through the UNet

        Returns
        -------
        (generated, grid_origins) — ``generated`` maps grid index → generated
        latent tensor (on CPU), ``grid_origins`` maps grid index → voxel origin (z0, y0, x0).
        """
        vol_d, vol_h, vol_w = volume_shape
        stride = self.patch_stride
        P      = self.patch_size
        por    = float(np.clip(target_porosity, _POR_MIN, _POR_MAX)) if target_porosity is not None else 0.05

        # Build grid of patch origins
        zs = list(range(0, vol_d - P + 1, stride))
        ys = list(range(0, vol_h - P + 1, stride))
        xs = list(range(0, vol_w - P + 1, stride))
        if not zs or not ys or not xs:
            raise ValueError(
                f"Volume {volume_shape} too small for patch_size={P}, "
                f"patch_stride={stride}."
            )

        # Grid index → origin voxel
        grid_origins = {(iz, iy, ix): (zs[iz], ys[iy], xs[ix])
                        for iz in range(len(zs))
                        for iy in range(len(ys))
                        for ix in range(len(xs))}

        # Store generated latents keyed by grid index (iz, iy, ix), held on CPU
        generated: dict[tuple[int, int, int], torch.Tensor] = {}

        # Two-phase checkerboard schedule: phase 1 = parity-0 anchors (every
        # in-bounds neighbor UNKNOWN), phase 2 = parity-1 non-anchors (every
        # in-bounds neighbor EXISTS). Order within a phase doesn't matter —
        # there are no intra-phase dependencies — so we just sort for determinism.
        phase0 = sorted(gi for gi in grid_origins if self._parity(gi) == 0)
        phase1 = sorted(gi for gi in grid_origins if self._parity(gi) == 1)

        total = len(grid_origins)
        logger.info(
            "VolumeGenerator: %d total patches (%d phase-1 anchors, %d phase-2 non-anchors), grid %d×%d×%d",
            total, len(phase0), len(phase1), len(zs), len(ys), len(xs),
        )

        zero_latent = torch.zeros(
            self.z_channels, self.latent_size, self.latent_size,
            self.latent_size, dtype=torch.float32,
        )

        n_done = 0
        log_interval = max(1, total // 20)

        for phase_num, (phase_patches, nb_state_in_bounds) in enumerate(
            ((phase0, _NB_UNKNOWN), (phase1, _NB_EXISTS)), start=1
        ):
            n_phase = len(phase_patches)
            for chunk_start in range(0, n_phase, gen_batch_size):
                chunk_gis = phase_patches[chunk_start : chunk_start + gen_batch_size]

                # Build inputs for this chunk
                chunk_nbl:  list[torch.Tensor] = []
                chunk_nba:  list[torch.Tensor] = []
                chunk_pos:  list[torch.Tensor] = []
                chunk_gpor: list[float]         = []
                chunk_lpor: list[float]         = []

                for gi in chunk_gis:
                    iz, iy, ix = gi
                    z0, y0, x0 = grid_origins[gi]

                    nb_latents_list: list[torch.Tensor] = []
                    nb_avail_list:   list[int]          = []
                    for dz, dy, dx in _NEIGHBOR_DIRS:
                        ngi = (iz + dz, iy + dy, ix + dx)
                        if ngi not in grid_origins:
                            nb_latents_list.append(zero_latent)
                            nb_avail_list.append(_NB_OOB)
                        elif nb_state_in_bounds == _NB_EXISTS:
                            nb_latents_list.append(generated[ngi])   # CPU tensor
                            nb_avail_list.append(_NB_EXISTS)
                        else:
                            nb_latents_list.append(zero_latent)
                            nb_avail_list.append(_NB_UNKNOWN)

                    chunk_nbl.append(torch.stack(nb_latents_list, dim=0))          # (6,C,D,D,D)
                    chunk_nba.append(torch.tensor(nb_avail_list, dtype=torch.long)) # (6,)
                    chunk_pos.append(torch.tensor([
                        z0 / max(vol_d - 1, 1),
                        y0 / max(vol_h - 1, 1),
                        x0 / max(vol_w - 1, 1),
                    ], dtype=torch.float32).clamp(0.0, 1.0))

                    local_por = (
                        local_por_map[gi]
                        if (local_por_map is not None and gi in local_por_map)
                        else por
                    )
                    chunk_gpor.append(por)
                    chunk_lpor.append(local_por)

                # Move to device and sample
                nb_latents_t = torch.stack(chunk_nbl).to(self.device)                             # (B,6,C,D,D,D)
                nb_avail_t   = torch.stack(chunk_nba).to(self.device)                             # (B,6)
                pos_frac_t   = torch.stack(chunk_pos).to(self.device)                             # (B,3)
                g_por_t      = torch.tensor(chunk_gpor, dtype=torch.float32, device=self.device)  # (B,)
                l_por_t      = torch.tensor(chunk_lpor,  dtype=torch.float32, device=self.device) # (B,)

                z_batch = self.sampler.sample_batch(
                    nb_latents_t, nb_avail_t, pos_frac_t, g_por_t, l_por_t,
                    autocast_dtype=autocast_dtype,
                )  # (B, C, D, D, D) float32 on device

                prev_done = n_done
                for i, gi in enumerate(chunk_gis):
                    generated[gi] = z_batch[i].cpu()
                n_done += len(chunk_gis)

                if patch_pbar is not None:
                    patch_pbar.update(len(chunk_gis))
                elif n_done // log_interval > prev_done // log_interval:
                    logger.info(
                        "Generated %d / %d patches (phase %d/2)", n_done, total, phase_num
                    )

        return generated, grid_origins

    def generate(
        self,
        volume_size_mm: tuple[float, float, float],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
        gen_batch_size: int = 32,
        decode_batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a full volume via the two-phase checkerboard schedule.

        Parameters
        ----------
        volume_size_mm    : (D, H, W) physical size in millimetres; each dimension
                            is snapped down to the nearest multiple of patch_size
        target_porosity   : global VVF target for conditioning (None = 0.05 fallback)
        autocast_dtype    : AMP dtype for the denoiser and VAE
        local_por_map     : optional dict mapping (iz, iy, ix) grid indices to per-patch
                            local porosity; overrides target_porosity for local_por only
        gen_batch_size    : patches sampled in parallel through the UNet per step
        decode_batch_size : latents decoded in one VAE forward pass

        Returns
        -------
        (xct_uint8, mask_uint8) — uint8 ndarrays of shape volume_shape
        """
        P   = self.patch_size
        vsz = self.voxel_size_mm

        # Snap each physical dimension to the nearest patch_size multiple (in voxels)
        volume_shape = tuple(
            (round(d / vsz) // P) * P for d in volume_size_mm
        )
        for d_mm, snapped_vox in zip(volume_size_mm, volume_shape):
            raw_vox = round(d_mm / vsz)
            if raw_vox - snapped_vox > 1:
                logger.warning(
                    "Volume axis %.3f mm: snapped %d vox → %d vox (dropped %d vox)",
                    d_mm, raw_vox, snapped_vox, raw_vox - snapped_vox,
                )

        vol_d, vol_h, vol_w = volume_shape
        stride = self.patch_stride

        nz = len(range(0, vol_d - P + 1, stride))
        ny = len(range(0, vol_h - P + 1, stride))
        nx = len(range(0, vol_w - P + 1, stride))

        patch_size_mm = P * vsz
        logger.info(
            "Volume %.1f×%.1f×%.1f mm  grid %d×%d×%d  patch %.2f mm",
            *volume_size_mm, nz, ny, nx, patch_size_mm,
        )

        generated, grid_origins = self._generate_latents(
            volume_shape=volume_shape,
            target_porosity=target_porosity,
            autocast_dtype=autocast_dtype,
            local_por_map=local_por_map,
            patch_pbar=patch_pbar,
            gen_batch_size=gen_batch_size,
        )

        # ── Cosine-taper overlap-add assembly ─────────────────────────────────
        # Each decoded 64³ patch is multiplied by a 3D periodic-Hann window and
        # accumulated into sum and weight buffers.  After all patches the output
        # is sum / weight (epsilon-guarded).  For 50%-overlap (stride 32) the
        # window forms a partition of unity in the interior; the three min-face
        # planes (index 0 on each axis) have zero weight and resolve to 0.
        w1d = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(P) / P))).astype(np.float32)
        w3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]   # (P,P,P)

        xct_sum    = np.zeros(volume_shape, dtype=np.float32)
        mask_sum   = np.zeros(volume_shape, dtype=np.float32)
        weight_vol = np.zeros(volume_shape, dtype=np.float32)

        all_items = list(generated.items())

        self.vae.eval()
        with torch.no_grad():
            for chunk_start in range(0, len(all_items), decode_batch_size):
                chunk = all_items[chunk_start : chunk_start + decode_batch_size]

                z_stacked = torch.stack(
                    [(z_gen * self.latent_std + self.latent_mean) for _, z_gen in chunk], dim=0
                ).to(self.device)   # (B, C, ls, ls, ls) — denormalised latents

                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    dec         = self.vae.decoder(z_stacked)
                    xct_logits  = self.vae.xct_head(dec)
                    mask_logits = self.vae.mask_head(dec)

                # squeeze(1): drop channel dim (size 1) while keeping batch dim.
                # Accumulate raw logits — blending happens in logit space so that
                # spatially-offset pore predictions in adjacent patches do not smear
                # in probability space (which inflates apparent pore diameter).
                xct_patches  = xct_logits.squeeze(1).float().cpu().numpy()   # (B,P,P,P) logits
                mask_patches = mask_logits.squeeze(1).float().cpu().numpy()  # (B,P,P,P) logits

                for i, (gi, _) in enumerate(chunk):
                    z0, y0, x0 = grid_origins[gi]
                    sl = (slice(z0, z0 + P), slice(y0, y0 + P), slice(x0, x0 + P))

                    xct_sum[sl]    += xct_patches[i]  * w3d
                    mask_sum[sl]   += mask_patches[i] * w3d
                    weight_vol[sl] += w3d

        # Normalise; epsilon guard prevents divide-by-zero on zero-weight faces.
        # sigmoid (expit) is applied once after blending in logit space; threshold
        # for the mask moves from >0.5 (probability) to >0.0 (logit).  Zero-weight
        # min-face planes are forced back to 0 for the XCT output so that they
        # remain black (consistent with the old behaviour and the weight comment above).
        w = np.maximum(weight_vol, 1e-8)
        xct_out  = np.clip(expit(xct_sum / w) * 255.0, 0, 255)
        xct_out[weight_vol <= 1e-8] = 0.0            # keep zero-weight boundary faces black
        xct_out  = xct_out.astype(np.uint8)
        mask_out = ((mask_sum / w) > 0.0).astype(np.uint8) * 255

        return xct_out, mask_out

    @staticmethod
    def save_tiff(
        xct:  np.ndarray,
        mask: np.ndarray,
        path_xct:  str | Path,
        path_mask: str | Path,
    ) -> None:
        """Write XCT and mask volumes to TIFF files."""
        import tifffile
        path_xct  = Path(path_xct)
        path_mask = Path(path_mask)
        path_xct.parent.mkdir(parents=True, exist_ok=True)
        path_mask.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path_xct),  xct)
        tifffile.imwrite(str(path_mask), mask)
        logger.info("Saved XCT  → %s", path_xct)
        logger.info("Saved mask → %s", path_mask)

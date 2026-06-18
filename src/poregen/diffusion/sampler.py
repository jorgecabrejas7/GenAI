"""DDPM patch sampler and full-volume generator for PoreGen LDM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


class DDIMSampler:
    """DDIM patch sampler — deterministic inference in n_steps < T steps.

    Parameters
    ----------
    model    : UNet3DDenoiser
    schedule : DDPMSchedule
    device   : torch.device
    n_steps  : int — number of denoising steps (default 50)
    """

    def __init__(self, model: torch.nn.Module, schedule: Any, device: torch.device, n_steps: int = 50) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.n_steps  = n_steps
        T = schedule.T
        ts = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long)
        # Store as Python ints for torch.compile compatibility (no dynamic shapes)
        self._timesteps: list[int] = ts.flip(0).tolist()   # [T-1, ..., 0]

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
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_l, nb_a, pos, g_por, l_por)
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
            if return_intermediates:
                intermediates.append(x.squeeze(0).float().cpu())

        result = x.squeeze(0).float()
        if return_intermediates:
            return result, intermediates
        return result


class VolumeGenerator:
    """Generate a full synthetic 3D volume using a two-phase checkerboard schedule.

    Patches are classified by checkerboard parity ``(iz + iy + ix) % 2``,
    exactly as in ``latent_dataset.py`` / ``encode_latents.py``:

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

    Parameters
    ----------
    sampler     : DDPMSampler
    vae         : VAE model with .decoder, .xct_head, .mask_head attributes
    device      : torch.device
    patch_size  : int — voxel side length of each patch (default 64)
    patch_stride: int — stride between patch origins (default 32)
    """

    def __init__(
        self,
        sampler: Any,
        vae: torch.nn.Module,
        device: torch.device,
        patch_size: int = 64,
        patch_stride: int = 32,
        latent_size: int = 16,
        latent_std: float = 1.0,
    ) -> None:
        self.sampler      = sampler
        self.vae          = vae
        self.device       = device
        self.patch_size   = patch_size
        self.patch_stride = patch_stride
        self.latent_size  = latent_size
        self.latent_std   = latent_std
        self.z_channels   = sampler.model.cfg.z_channels

    def _make_hann_weights(self) -> np.ndarray:
        """3D Hann window for patch blending — reduces seam artefacts."""
        w1d = np.hanning(self.patch_size).astype(np.float32)
        w3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]
        return w3d  # (P, P, P)

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
    ) -> tuple[dict[tuple[int, int, int], torch.Tensor], dict[tuple[int, int, int], tuple[int, int, int]]]:
        """Run the two-phase checkerboard schedule and return per-patch latents.

        Parameters
        ----------
        volume_shape    : (D, H, W) in voxels
        target_porosity : global VVF target for conditioning (None = 0.05 fallback)
        autocast_dtype  : AMP dtype for the denoiser
        local_por_map   : optional dict mapping (iz, iy, ix) grid indices to per-patch
                          local porosity; overrides target_porosity for local_por only

        Returns
        -------
        (generated, grid_origins) — ``generated`` maps grid index → generated
        latent tensor, ``grid_origins`` maps grid index → voxel origin (z0, y0, x0).
        """
        vol_d, vol_h, vol_w = volume_shape
        stride = self.patch_stride
        por = float(target_porosity) if target_porosity is not None else 0.05

        # Build grid of patch origins
        zs = list(range(0, vol_d - self.patch_size + 1, stride))
        ys = list(range(0, vol_h - self.patch_size + 1, stride))
        xs = list(range(0, vol_w - self.patch_size + 1, stride))
        if not zs or not ys or not xs:
            raise ValueError(
                f"Volume {volume_shape} too small for patch_size={self.patch_size}, "
                f"patch_stride={stride}."
            )

        # Grid index → origin voxel
        grid_origins = {(iz, iy, ix): (zs[iz], ys[iy], xs[ix])
                        for iz in range(len(zs))
                        for iy in range(len(ys))
                        for ix in range(len(xs))}

        # Store generated latents keyed by grid index (iz, iy, ix)
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
        for phase_num, (phase_patches, nb_state_in_bounds) in enumerate(
            ((phase0, _NB_UNKNOWN), (phase1, _NB_EXISTS)), start=1
        ):
            for gi in phase_patches:
                iz, iy, ix = gi
                z0, y0, x0 = grid_origins[gi]

                # Build neighbor context. In-bounds neighbors always belong to
                # the opposite parity, so their state is fixed by the phase:
                # phase 1 (anchors) → UNKNOWN (not generated yet), phase 2
                # (non-anchors) → EXISTS (completed in phase 1).
                nb_latents_list = []
                nb_avail_list   = []
                for dz, dy, dx in _NEIGHBOR_DIRS:
                    ngi = (iz + dz, iy + dy, ix + dx)
                    if ngi not in grid_origins:
                        nb_latents_list.append(zero_latent)
                        nb_avail_list.append(_NB_OOB)
                    elif nb_state_in_bounds == _NB_EXISTS:
                        nb_latents_list.append(generated[ngi].cpu())
                        nb_avail_list.append(_NB_EXISTS)
                    else:
                        nb_latents_list.append(zero_latent)
                        nb_avail_list.append(_NB_UNKNOWN)

                nb_latents_t = torch.stack(nb_latents_list, dim=0)               # (6, C, D, D, D)
                nb_avail_t   = torch.tensor(nb_avail_list, dtype=torch.long)     # (6,)

                pos_frac = torch.tensor([
                    z0 / max(vol_d - 1, 1),
                    y0 / max(vol_h - 1, 1),
                    x0 / max(vol_w - 1, 1),
                ], dtype=torch.float32).clamp(0.0, 1.0)

                local_por = local_por_map[gi] if (local_por_map is not None and gi in local_por_map) else por
                z_gen = self.sampler.sample_patch(
                    nb_latents_t, nb_avail_t, pos_frac, por, local_por,
                    autocast_dtype=autocast_dtype,
                )
                generated[gi] = z_gen
                n_done += 1

                if patch_pbar is not None:
                    patch_pbar.update(1)
                elif n_done % max(1, total // 20) == 0:
                    logger.info("Generated %d / %d patches (phase %d/2)", n_done, total, phase_num)

        return generated, grid_origins

    def generate(
        self,
        volume_shape: tuple[int, int, int],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a full volume via the two-phase checkerboard schedule.

        Parameters
        ----------
        volume_shape    : (D, H, W) in voxels
        target_porosity : global VVF target for conditioning (None = 0.05 fallback)
        autocast_dtype  : AMP dtype for the denoiser
        local_por_map   : optional dict mapping (iz, iy, ix) grid indices to per-patch
                          local porosity; overrides target_porosity for local_por only

        Returns
        -------
        (xct_uint8, mask_uint8) — uint8 ndarrays of shape volume_shape
        """
        generated, grid_origins = self._generate_latents(
            volume_shape=volume_shape,
            target_porosity=target_porosity,
            autocast_dtype=autocast_dtype,
            local_por_map=local_por_map,
            patch_pbar=patch_pbar,
        )

        # ── Decode all latents and blend into volume ──────────────────────────
        # Hann blending only helps when patches overlap; with stride ≥ patch_size
        # (non-overlapping tiles) the Hann endpoints are zero and create a
        # visible grid of dead voxels.  Use a box window (all-ones) instead.
        if self.patch_stride >= self.patch_size:
            hann = np.ones((self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
        else:
            hann = self._make_hann_weights()                    # (P, P, P)
        xct_acc  = np.zeros(volume_shape, dtype=np.float32)
        mask_acc = np.zeros(volume_shape, dtype=np.float32)
        wgt_acc  = np.zeros(volume_shape, dtype=np.float32)

        self.vae.eval()
        with torch.no_grad():
            for gi, z_gen in generated.items():
                z0, y0, x0 = grid_origins[gi]
                ze, ye, xe = z0 + self.patch_size, y0 + self.patch_size, x0 + self.patch_size

                z_batch = (z_gen * self.latent_std).unsqueeze(0).to(self.device)   # (1, C, ls, ls, ls)
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    dec = self.vae.decoder(z_batch)
                    xct_logits  = self.vae.xct_head(dec)
                    mask_logits = self.vae.mask_head(dec)

                xct_patch  = torch.sigmoid(xct_logits).squeeze().float().cpu().numpy()   # (P,P,P)
                mask_patch = torch.sigmoid(mask_logits).squeeze().float().cpu().numpy()  # (P,P,P)

                xct_acc [z0:ze, y0:ye, x0:xe] += xct_patch  * hann
                mask_acc[z0:ze, y0:ye, x0:xe] += mask_patch * hann
                wgt_acc [z0:ze, y0:ye, x0:xe] += hann

        eps = 1e-8
        xct_vol  = np.clip(xct_acc  / (wgt_acc + eps), 0.0, 1.0)
        mask_vol = np.clip(mask_acc / (wgt_acc + eps), 0.0, 1.0)

        xct_uint8  = (xct_vol  * 255).astype(np.uint8)
        mask_uint8 = (mask_vol > 0.5).astype(np.uint8) * 255

        return xct_uint8, mask_uint8

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

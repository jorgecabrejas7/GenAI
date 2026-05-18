"""DDPM patch sampler and full-volume generator for PoreGen LDM."""

from __future__ import annotations

import logging
from collections import deque
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
    ) -> torch.Tensor:
        """Run the full T-step DDPM reverse process for one patch.

        Parameters
        ----------
        nb_latents : (6, C, D, H, W) — known neighbor latents (zeros for unavail)
        nb_avail   : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        pos_frac   : (3,) float — normalised patch position in volume
        global_por : float
        local_por  : float

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent
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

        for t_idx in reversed(range(schedule.T)):
            t = torch.tensor([t_idx], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_l, nb_a, pos, g_por, l_por)
            x = schedule.p_sample(x, t, eps_pred)

        return x.squeeze(0).float()


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
    ) -> torch.Tensor:
        """Run DDIM reverse process for one patch.

        Parameters
        ----------
        nb_latents : (6, C, D, H, W) — known neighbor latents (zeros for unavail)
        nb_avail   : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        pos_frac   : (3,) float — normalised patch position in volume
        global_por : float
        local_por  : float

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent
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

        for i, t_val in enumerate(self._timesteps[:-1]):
            t_prev_val = self._timesteps[i + 1]
            t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
            t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_l, nb_a, pos, g_por, l_por)
            x = schedule.ddim_step(x, t, t_prev, eps_pred)

        return x.squeeze(0).float()


class VolumeGenerator:
    """Generate a full synthetic 3D volume using anchor-first BFS patch ordering.

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

    def generate(
        self,
        volume_shape: tuple[int, int, int],
        target_porosity: float | None = None,
        anchor_gap: int = 3,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a full volume via anchor-first BFS.

        Parameters
        ----------
        volume_shape    : (D, H, W) in voxels
        target_porosity : VVF target for conditioning (None = 0.05 fallback)
        anchor_gap      : grid spacing between anchor patches
        autocast_dtype  : AMP dtype for the denoiser

        Returns
        -------
        (xct_uint8, mask_uint8) — uint8 ndarrays of shape volume_shape
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
        queue: deque[tuple[int, int, int]] = deque()
        queued: set[tuple[int, int, int]] = set()

        # Checkerboard anchors: every non-anchor's 6 face-neighbors are anchors,
        # so the first BFS wave fills in all remaining patches with full context.
        anchors = {
            (iz, iy, ix)
            for (iz, iy, ix) in grid_origins
            if (iz + iy + ix) % 2 == 0
        }
        for a in sorted(anchors):
            queue.append(a)
            queued.add(a)

        total = len(grid_origins)
        logger.info(
            "VolumeGenerator: %d total patches, %d anchors, grid %d×%d×%d",
            total, len(anchors), len(zs), len(ys), len(xs),
        )

        n_done = 0
        while queue:
            gi = queue.popleft()
            iz, iy, ix = gi
            z0, y0, x0 = grid_origins[gi]

            # Build neighbor context
            nb_latents_list = []
            nb_avail_list   = []
            for dz, dy, dx in _NEIGHBOR_DIRS:
                ngi = (iz + dz, iy + dy, ix + dx)
                if ngi not in grid_origins:
                    nb_latents_list.append(torch.zeros(
                        self.z_channels, self.latent_size, self.latent_size,
                        self.latent_size, dtype=torch.float32))
                    nb_avail_list.append(_NB_OOB)
                elif ngi in generated:
                    nb_latents_list.append(generated[ngi].cpu())
                    nb_avail_list.append(_NB_EXISTS)
                else:
                    nb_latents_list.append(torch.zeros(
                        self.z_channels, self.latent_size, self.latent_size,
                        self.latent_size, dtype=torch.float32))
                    nb_avail_list.append(_NB_UNKNOWN)

            nb_latents_t = torch.stack(nb_latents_list, dim=0)               # (6, C, D, D, D)
            nb_avail_t   = torch.tensor(nb_avail_list, dtype=torch.long)     # (6,)

            pos_frac = torch.tensor([
                z0 / max(vol_d - 1, 1),
                y0 / max(vol_h - 1, 1),
                x0 / max(vol_w - 1, 1),
            ], dtype=torch.float32).clamp(0.0, 1.0)

            z_gen = self.sampler.sample_patch(
                nb_latents_t, nb_avail_t, pos_frac, por, por,
                autocast_dtype=autocast_dtype,
            )
            generated[gi] = z_gen
            n_done += 1

            if n_done % max(1, total // 20) == 0:
                logger.info("Generated %d / %d patches", n_done, total)

            # Enqueue unqueued grid neighbours that are in-bounds
            for dz, dy, dx in _NEIGHBOR_DIRS:
                ngi = (iz + dz, iy + dy, ix + dx)
                if ngi in grid_origins and ngi not in queued:
                    queue.append(ngi)
                    queued.add(ngi)

        # ── Decode all latents and blend into volume ──────────────────────────
        hann = self._make_hann_weights()                        # (P, P, P)
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

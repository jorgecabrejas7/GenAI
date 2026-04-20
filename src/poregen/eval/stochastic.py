"""Stochastic and deterministic volume reconstruction for eval.

Three reconstruction modes
--------------------------
All three modes are computed in a single tiling pass over the volume so that
each 64³ patch is loaded from disk only once.

``stoch_mean``
    N forward passes per patch with z = μ + σε (reparameterisation).
    The decoded XCT logits and mask probabilities are averaged across passes.
    This is the canonical reconstruction used for quality metrics.

``stoch_single``
    The first of the N stochastic passes (pass index 0).  Identical random
    seed, so it is reproducible.  Useful for comparing a single realisation
    against the mean.

``mu``
    Deterministic pass: z is set to μ directly, bypassing the reparameterisation.
    No random noise is involved.  Useful for checking whether posterior variance
    contributes meaningfully to reconstruction quality.

Reproducibility
---------------
``torch.manual_seed(seed)`` is called **once before the patch loop** for each
volume.  This ensures that the N samples for every patch in the same volume are
drawn from a deterministic but spatially-varying RNG stream.  The same
checkpoint + seed will always produce identical reconstructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PATCH_SIZE = 64


def _encode(model: nn.Module, xct_t: torch.Tensor) -> torch.Tensor:
    """Run the encoder portion of a VAE and return the pre-bottleneck feature map.

    Handles both single-branch models (``model.encoder``) and dual-branch
    models (``model.encoder_a`` + ``model.encoder_b`` + ``model.fusion``).
    """
    if hasattr(model, "encoder"):
        return model.encoder(xct_t)
    # Dual-branch: encode twice and fuse
    h_a = model.encoder_a(xct_t)
    h_b = model.encoder_b(xct_t)
    return model.fusion(torch.cat([h_a, h_b], dim=1))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VolumeReconstruction:
    """All reconstruction modes plus GT arrays for one full volume.

    All spatial arrays have shape ``(Dz, Dy, Dx)`` where the dimensions are
    the *tiled* extent (remainder voxels after integer division by 64 are
    dropped to keep patches non-overlapping and boundary-aligned).

    Attributes
    ----------
    volume_id : str
        The volume identifier from the dataset index (e.g.
        ``"MedidasDB__SomeCylinder"``).
    xct_gt : ndarray, float32, [0, 1]
        Ground-truth XCT intensity normalised to [0, 1].
    mask_gt : ndarray, float32, {0, 1}
        Ground-truth pore mask (binary).
    xct_stoch_mean : ndarray, float32, [0, 1]
        Mean XCT reconstruction across ``n_samples`` stochastic passes.
    mask_stoch_mean : ndarray, float32, [0, 1]
        Mean mask sigmoid probability across ``n_samples`` stochastic passes.
    xct_stoch_std : ndarray, float32, ≥ 0
        Voxel-wise standard deviation of XCT reconstructions across passes.
        Low values indicate a tight posterior; high values indicate uncertainty.
    mask_stoch_std : ndarray, float32, ≥ 0
        Voxel-wise std of mask sigmoid probabilities across passes.
    xct_stoch_single : ndarray, float32, [0, 1]
        XCT reconstruction from pass index 0 (first stochastic sample).
    mask_stoch_single : ndarray, float32, [0, 1]
        Mask sigmoid probability from pass index 0.
    xct_mu : ndarray, float32, [0, 1]
        Deterministic reconstruction: z = μ, no noise added.
    mask_mu : ndarray, float32, [0, 1]
        Deterministic mask sigmoid probability using z = μ.
    shape_original : tuple of int
        Original (D, H, W) of the full TIFF / Zarr volume before tiling.
    shape_tiled : tuple of int
        (Dz, Dy, Dx) after dropping remainder voxels.
    """

    volume_id: str
    xct_gt: np.ndarray
    mask_gt: np.ndarray
    xct_stoch_mean: np.ndarray
    mask_stoch_mean: np.ndarray
    xct_stoch_std: np.ndarray
    mask_stoch_std: np.ndarray
    xct_stoch_single: np.ndarray
    mask_stoch_single: np.ndarray
    xct_mu: np.ndarray
    mask_mu: np.ndarray
    shape_original: tuple[int, int, int]
    shape_tiled: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_patch_three_modes(
    model: nn.Module,
    xct_patch_u8: np.ndarray,
    device: torch.device,
    *,
    n_samples: int = 50,
    seed_offset: int = 0,
) -> tuple[
    np.ndarray, np.ndarray,   # xct_stoch_mean, mask_stoch_mean
    np.ndarray, np.ndarray,   # xct_stoch_std,  mask_stoch_std
    np.ndarray, np.ndarray,   # xct_stoch_single, mask_stoch_single
    np.ndarray, np.ndarray,   # xct_mu,          mask_mu
    np.ndarray, np.ndarray,   # mu,               logvar
]:
    """Encode and decode one 64³ patch in all three reconstruction modes.

    The caller is responsible for setting ``torch.manual_seed`` once before
    starting the patch loop for a volume.  This function does **not** reset
    the seed, so successive calls consume different parts of the RNG stream,
    producing spatially-varying but reproducible noise patterns across patches.

    Parameters
    ----------
    model : nn.Module
        ConvVAE3DNoAttnV2 (or compatible).  Must expose ``encoder``,
        ``to_mu``, ``to_logvar``, ``decoder``, ``xct_head``, ``mask_head``.
        The model must already be in ``eval()`` mode.
    xct_patch_u8 : ndarray, uint8, shape (64, 64, 64)
        Raw XCT patch.  Normalised to [0, 1] internally.
    device : torch.device
    n_samples : int
        Number of stochastic forward passes for mean / std computation.
    seed_offset : int
        Unused (kept for a possible future per-patch seeding extension).

    Returns
    -------
    xct_stoch_mean  : float32, (64, 64, 64), [0, 1]
    mask_stoch_mean : float32, (64, 64, 64), [0, 1]
    xct_stoch_std   : float32, (64, 64, 64), ≥ 0
    mask_stoch_std  : float32, (64, 64, 64), ≥ 0
    xct_stoch_single  : float32, (64, 64, 64) — pass 0
    mask_stoch_single : float32, (64, 64, 64) — pass 0
    xct_mu  : float32, (64, 64, 64) — z = μ, no noise
    mask_mu : float32, (64, 64, 64) — z = μ, no noise
    mu      : float32, (C, d, h, w) — latent mean
    logvar  : float32, (C, d, h, w) — latent log-variance
    """
    # Normalise and move to device
    xct_f = xct_patch_u8.astype(np.float32) / 255.0
    xct_t = torch.from_numpy(xct_f).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64,64)

    # Encode once — μ and logvar are deterministic
    h = _encode(model, xct_t)
    mu_t = model.to_mu(h)
    logvar_t = model.to_logvar(h)

    # --- deterministic pass: z = μ -----------------------------------------
    dec_mu = model.decoder(mu_t)
    xct_mu_t   = model.xct_head(dec_mu).clamp(0.0, 1.0)
    mask_mu_t  = torch.sigmoid(model.mask_head(dec_mu))
    xct_mu_np  = xct_mu_t .squeeze().cpu().numpy().astype(np.float32)
    mask_mu_np = mask_mu_t.squeeze().cpu().numpy().astype(np.float32)

    # --- stochastic passes: z = μ + σε -------------------------------------
    std_t = torch.exp(0.5 * logvar_t)  # σ = exp(0.5 * logvar)
    xct_samples:  list[np.ndarray] = []
    mask_samples: list[np.ndarray] = []

    for _ in range(n_samples):
        eps = torch.randn_like(mu_t)
        z   = mu_t + std_t * eps
        dec = model.decoder(z)
        xct_out  = model.xct_head(dec).clamp(0.0, 1.0).squeeze().cpu().numpy().astype(np.float32)
        mask_out = torch.sigmoid(model.mask_head(dec)).squeeze().cpu().numpy().astype(np.float32)
        xct_samples.append(xct_out)
        mask_samples.append(mask_out)

    xct_arr  = np.stack(xct_samples,  axis=0)  # (N, 64, 64, 64)
    mask_arr = np.stack(mask_samples, axis=0)

    xct_mean   = xct_arr .mean(axis=0).astype(np.float32)
    mask_mean  = mask_arr.mean(axis=0).astype(np.float32)
    xct_std    = xct_arr .std(axis=0).astype(np.float32)
    mask_std   = mask_arr.std(axis=0).astype(np.float32)

    return (
        xct_mean, mask_mean,
        xct_std,  mask_std,
        xct_samples[0], mask_samples[0],  # single = pass 0
        xct_mu_np, mask_mu_np,
        mu_t    .squeeze().cpu().numpy().astype(np.float32),
        logvar_t.squeeze().cpu().numpy().astype(np.float32),
    )


def reconstruct_volume_three_modes(
    model: nn.Module,
    volume_id: str,
    xct_raw_u8: np.ndarray,
    mask_raw_u8: np.ndarray,
    device: torch.device,
    *,
    n_samples: int = 50,
    seed: int = 42,
) -> VolumeReconstruction:
    """Reconstruct a full volume from non-overlapping 64³ patches in all three modes.

    Tiling strategy
    ---------------
    Patches are extracted with stride = patch_size = 64 (non-overlapping).
    Any voxels in the remainder (D % 64, H % 64, W % 64) are discarded, so
    the GT arrays are cropped to the same tiled extent.

    Seed handling
    -------------
    ``torch.manual_seed(seed)`` is called once before the patch loop.
    This makes the full set of N×n_patches random samples reproducible given
    the same seed and checkpoint.

    Parameters
    ----------
    model : nn.Module
        VAE model in eval mode.
    volume_id : str
        Identifier used to populate ``VolumeReconstruction.volume_id``.
    xct_raw_u8 : ndarray, uint8, shape (D, H, W)
        Full-volume XCT intensity.
    mask_raw_u8 : ndarray, uint8 {0, 1}, shape (D, H, W)
        Full-volume binary pore mask.
    device : torch.device
    n_samples : int
        Number of stochastic passes per patch.
    seed : int
        Seed passed to ``torch.manual_seed``.

    Returns
    -------
    VolumeReconstruction
    """
    D, H, W = xct_raw_u8.shape
    dz = (D // PATCH_SIZE) * PATCH_SIZE
    dy = (H // PATCH_SIZE) * PATCH_SIZE
    dx = (W // PATCH_SIZE) * PATCH_SIZE

    xct_stoch_mean   = np.zeros((dz, dy, dx), dtype=np.float32)
    mask_stoch_mean  = np.zeros((dz, dy, dx), dtype=np.float32)
    xct_stoch_std    = np.zeros((dz, dy, dx), dtype=np.float32)
    mask_stoch_std   = np.zeros((dz, dy, dx), dtype=np.float32)
    xct_stoch_single = np.zeros((dz, dy, dx), dtype=np.float32)
    mask_stoch_single= np.zeros((dz, dy, dx), dtype=np.float32)
    xct_mu_vol  = np.zeros((dz, dy, dx), dtype=np.float32)
    mask_mu_vol = np.zeros((dz, dy, dx), dtype=np.float32)

    model.eval()
    torch.manual_seed(seed)

    n_patches = (dz // PATCH_SIZE) * (dy // PATCH_SIZE) * (dx // PATCH_SIZE)
    done = 0
    log_every = max(1, n_patches // 10)

    for z0 in range(0, dz, PATCH_SIZE):
        for y0 in range(0, dy, PATCH_SIZE):
            for x0 in range(0, dx, PATCH_SIZE):
                patch = xct_raw_u8[
                    z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE
                ]
                sl = np.s_[z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

                (
                    xct_stoch_mean  [sl],  mask_stoch_mean  [sl],
                    xct_stoch_std   [sl],  mask_stoch_std   [sl],
                    xct_stoch_single[sl],  mask_stoch_single[sl],
                    xct_mu_vol      [sl],  mask_mu_vol      [sl],
                    _, _,   # mu, logvar — not stored at volume level
                ) = encode_patch_three_modes(
                    model, patch, device, n_samples=n_samples
                )

                done += 1
                if done % log_every == 0:
                    logger.info("    %d / %d patches …", done, n_patches)

    xct_gt  = xct_raw_u8 [:dz, :dy, :dx].astype(np.float32) / 255.0
    mask_gt = mask_raw_u8[:dz, :dy, :dx].astype(np.float32)

    return VolumeReconstruction(
        volume_id=volume_id,
        xct_gt=xct_gt,
        mask_gt=mask_gt,
        xct_stoch_mean=xct_stoch_mean,
        mask_stoch_mean=mask_stoch_mean,
        xct_stoch_std=xct_stoch_std,
        mask_stoch_std=mask_stoch_std,
        xct_stoch_single=xct_stoch_single,
        mask_stoch_single=mask_stoch_single,
        xct_mu=xct_mu_vol,
        mask_mu=mask_mu_vol,
        shape_original=(D, H, W),
        shape_tiled=(dz, dy, dx),
    )

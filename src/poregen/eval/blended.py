"""Overlapping-patch volume reconstruction with Tukey-window blending.

Logits are blended *before* the final sigmoid / clamp so that patch boundaries
do not pull predictions toward 0.5 through the nonlinearity.

Algorithm
---------
1. Reflection-pad the volume so patches cover every voxel.
2. Build a 3-D Tukey (alpha=0.5) weight window once.
3. For each stochastic pass ``s`` in ``[0, n_samples)``:
   a. ``torch.manual_seed(seed + s)`` for reproducibility.
   b. Sweep all patches in batches; accumulate ``W3d × logit`` into
      per-pass weighted-sum buffers.
   c. Divide by the weight accumulator to get the blended logit volume.
   d. Apply ``clip(·, 0, 1)`` for XCT and ``sigmoid`` for mask.
4. Compute per-voxel mean and std across the ``n_samples`` pass-volumes.
5. Crop back to the original (D, H, W).

Returns
-------
dict
    recon_xct           : float32 (D, H, W) in [0, 1]
    recon_mask_binary   : uint8   (D, H, W) in {0, 1}
    recon_mask_prob     : float32 (D, H, W) in [0, 1]
    recon_xct_std_mean  : float   — mean over voxels of per-voxel XCT std
    recon_mask_std_mean : float   — mean over voxels of per-voxel mask std
    n_patches           : int
    padded_shape        : tuple (Pd, Ph, Pw)
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers (also imported by tests)
# ---------------------------------------------------------------------------

def _tukey_window_3d(patch_size: int, alpha: float = 0.5) -> np.ndarray:
    """3-D Tukey window as the outer product of three 1-D windows."""
    from scipy.signal.windows import tukey as _tukey
    w = _tukey(patch_size, alpha=alpha).astype(np.float32)
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


def _pad_size(n: int, patch_size: int, stride: int) -> int:
    """Smallest padded length so stride-spaced patches cover all *n* voxels."""
    if n <= patch_size:
        return patch_size
    n_patches = math.ceil((n - patch_size) / stride) + 1
    return (n_patches - 1) * stride + patch_size


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode(model: nn.Module, xct_t: torch.Tensor) -> torch.Tensor:
    """Architecture-agnostic encoder: single-branch or dual-branch."""
    if hasattr(model, "encoder"):
        return model.encoder(xct_t)
    h_a = model.encoder_a(xct_t)
    h_b = model.encoder_b(xct_t)
    return model.fusion(torch.cat([h_a, h_b], dim=1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_volume(
    volume_tiff_path,
    gt_mask_tiff_path,
    model: nn.Module,
    patch_size: int = 64,
    stride: int = 48,
    n_samples: int = 5,
    device: str | torch.device = "cuda",
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """Reconstruct a full volume with overlapping cosine-taper blending.

    Parameters
    ----------
    volume_tiff_path : path-like
        XCT TIFF (uint8, D×H×W).
    gt_mask_tiff_path : path-like
        Binary mask TIFF (uint8 {0,1}, D×H×W).  Loaded but not used for
        inference — returned in the dict for downstream metric computation.
    model : nn.Module
        VAE in eval mode.  Must expose ``encoder`` (or ``encoder_a`` /
        ``encoder_b`` / ``fusion``), ``to_mu``, ``to_logvar``, ``decoder``,
        ``xct_head``, ``mask_head``.
    patch_size : int
        Cubic patch side-length in voxels.  Default 64.
    stride : int
        Patch step.  ``stride < patch_size`` gives overlap.  Default 48.
    n_samples : int
        Number of stochastic passes (z = μ + σε) per patch.  Default 5.
    device : str or torch.device
    batch_size : int
        Number of patches per GPU batch.  Default 32.
    seed : int
        Pass ``s`` uses ``torch.manual_seed(seed + s)``.  Default 42.

    Returns
    -------
    dict with keys:
        recon_xct           float32 (D, H, W)
        recon_mask_binary   uint8   (D, H, W)
        recon_mask_prob     float32 (D, H, W)
        recon_xct_std_mean  float
        recon_mask_std_mean float
        n_patches           int
        padded_shape        tuple[int, int, int]
    """
    import tifffile

    if isinstance(device, str):
        device = torch.device(device)

    # ── Load ────────────────────────────────────────────────────────────────
    xct_u8   = tifffile.imread(str(volume_tiff_path))
    mask_u8  = tifffile.imread(str(gt_mask_tiff_path))
    D, H, W  = xct_u8.shape
    xct_f    = xct_u8.astype(np.float32) / 255.0

    # ── Padding ──────────────────────────────────────────────────────────────
    # Symmetric pre-padding of alpha/2 * patch_size voxels ensures that the
    # original boundary voxels fall in the flat (weight ≈ 1) region of the
    # first / last Tukey patch, not in the taper (weight → 0) region.
    alpha = 0.5
    taper_half = max(1, int(patch_size * alpha / 2))  # 16 for size=64, alpha=0.5

    pre = taper_half
    D_pre = D + 2 * pre
    H_pre = H + 2 * pre
    W_pre = W + 2 * pre

    # First, add symmetric pre-padding
    padded_pre = np.pad(xct_f,
                        ((pre, pre), (pre, pre), (pre, pre)),
                        mode="reflect")

    # Then, add trailing padding to align with patch grid
    pad_d = _pad_size(D_pre, patch_size, stride)
    pad_h = _pad_size(H_pre, patch_size, stride)
    pad_w = _pad_size(W_pre, patch_size, stride)
    padded_shape = (pad_d, pad_h, pad_w)

    padded = np.pad(
        padded_pre,
        ((0, pad_d - D_pre), (0, pad_h - H_pre), (0, pad_w - W_pre)),
        mode="reflect",
    )

    # ── Tukey window ─────────────────────────────────────────────────────────
    W3d = _tukey_window_3d(patch_size)

    # ── Patch coordinates ─────────────────────────────────────────────────────
    coords = [
        (z, y, x)
        for z in range(0, pad_d - patch_size + 1, stride)
        for y in range(0, pad_h - patch_size + 1, stride)
        for x in range(0, pad_w - patch_size + 1, stride)
    ]
    n_patches_total = len(coords)

    logger.info(
        "reconstruct_volume: vol=(%d,%d,%d) padded=%s patches=%d stride=%d N=%d",
        D, H, W, padded_shape, n_patches_total, stride, n_samples,
    )

    # ── Weight accumulator (geometry-only, same for every pass) ───────────────
    weight_acc = np.zeros(padded_shape, dtype=np.float32)
    for z0, y0, x0 in coords:
        weight_acc[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += W3d
    safe_w = np.where(weight_acc > 0, weight_acc, 1.0)

    # ── N stochastic passes ───────────────────────────────────────────────────
    model.eval()
    xct_passes:  list[np.ndarray] = []
    mask_passes: list[np.ndarray] = []

    for s in range(n_samples):
        torch.manual_seed(seed + s)

        xct_wsum  = np.zeros(padded_shape, dtype=np.float32)
        mask_wsum = np.zeros(padded_shape, dtype=np.float32)

        for b0 in range(0, n_patches_total, batch_size):
            b_coords = coords[b0 : b0 + batch_size]

            patches_np = np.stack(
                [padded[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size]
                 for z0, y0, x0 in b_coords]
            )  # (B, P, P, P)

            xct_t = torch.from_numpy(patches_np).unsqueeze(1).to(device)  # (B,1,P,P,P)

            h      = _encode(model, xct_t)
            mu     = model.to_mu(h)
            logvar = model.to_logvar(h)
            std_t  = torch.exp(0.5 * logvar)

            eps = torch.randn_like(mu)
            z_t = mu + std_t * eps
            dec = model.decoder(z_t)

            # Raw logits before activation — blend these
            xct_logit  = model.xct_head(dec)   # (B, 1, P, P, P)
            mask_logit = model.mask_head(dec)   # (B, 1, P, P, P)

            xct_np  = xct_logit .squeeze(1).cpu().numpy().astype(np.float32)
            mask_np = mask_logit.squeeze(1).cpu().numpy().astype(np.float32)

            for i, (z0, y0, x0) in enumerate(b_coords):
                sl = np.s_[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size]
                xct_wsum [sl] += W3d * xct_np [i]
                mask_wsum[sl] += W3d * mask_np[i]

        # Divide by weight, then activate
        xct_blended  = np.clip(xct_wsum / safe_w, 0.0, 1.0)
        mask_blended = (1.0 / (1.0 + np.exp(-(mask_wsum / safe_w)))).astype(np.float32)

        xct_passes .append(xct_blended .astype(np.float32))
        mask_passes.append(mask_blended)

        logger.info("  Pass %d/%d done", s + 1, n_samples)

    # ── Aggregate across passes ───────────────────────────────────────────────
    xct_arr  = np.stack(xct_passes,  axis=0)  # (N, Pd, Ph, Pw)
    mask_arr = np.stack(mask_passes, axis=0)

    recon_xct       = xct_arr .mean(axis=0).astype(np.float32)
    recon_mask_prob = mask_arr.mean(axis=0).astype(np.float32)
    xct_std         = xct_arr .std (axis=0).astype(np.float32)
    mask_std        = mask_arr.std (axis=0).astype(np.float32)

    # Crop back to original region (skip the symmetric pre-padding)
    recon_xct       = recon_xct      [pre:pre+D, pre:pre+H, pre:pre+W]
    recon_mask_prob = recon_mask_prob [pre:pre+D, pre:pre+H, pre:pre+W]
    xct_std         = xct_std        [pre:pre+D, pre:pre+H, pre:pre+W]
    mask_std        = mask_std        [pre:pre+D, pre:pre+H, pre:pre+W]

    return {
        "recon_xct":           recon_xct,
        "recon_mask_binary":   (recon_mask_prob >= 0.5).astype(np.uint8),
        "recon_mask_prob":     recon_mask_prob,
        "recon_xct_std_mean":  float(xct_std.mean()),
        "recon_mask_std_mean": float(mask_std.mean()),
        "n_patches":           n_patches_total,
        "padded_shape":        padded_shape,
    }

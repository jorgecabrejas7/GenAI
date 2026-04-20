"""Tests for overlapping-patch Tukey-window volume reconstruction."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal VAE fixture
# ---------------------------------------------------------------------------

class _TinyVAE(nn.Module):
    """Minimal VAE for testing: 64³ → 16³ latent → 64³."""

    def __init__(self, z_ch: int = 4, fixed_logvar: float = -100.0):
        super().__init__()
        self._z_ch = z_ch
        self._fixed_logvar = fixed_logvar
        # stride-4 conv: 64 → 16
        self._enc   = nn.Conv3d(1,    z_ch, 3, stride=4, padding=1)
        self._mu    = nn.Conv3d(z_ch, z_ch, 1)
        # stride-4 transposed conv: 16 → 64
        self._dec   = nn.ConvTranspose3d(z_ch, z_ch, 4, stride=4, padding=0)
        self._xct   = nn.Conv3d(z_ch, 1, 1)
        self._mask  = nn.Conv3d(z_ch, 1, 1)

    # The _encode helper checks hasattr(model, "encoder") — provide it as a method.
    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self._enc(x)

    def to_mu(self, h: torch.Tensor) -> torch.Tensor:
        return self._mu(h)

    def to_logvar(self, h: torch.Tensor) -> torch.Tensor:
        return torch.full_like(h, self._fixed_logvar)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        return self._dec(z)

    def xct_head(self, d: torch.Tensor) -> torch.Tensor:
        return self._xct(d)

    def mask_head(self, d: torch.Tensor) -> torch.Tensor:
        return self._mask(d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiffs(tmp_path, shape):
    """Write random uint8 XCT and binary mask TIFFs; return their paths."""
    import tifffile
    rng = np.random.default_rng(0)
    xct  = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = rng.integers(0, 2,   shape, dtype=np.uint8)
    xct_p  = tmp_path / "xct.tif"
    mask_p = tmp_path / "mask.tif"
    tifffile.imwrite(str(xct_p),  xct)
    tifffile.imwrite(str(mask_p), mask)
    return xct_p, mask_p


# ---------------------------------------------------------------------------
# Test 1 — output shape matches input volume shape
# ---------------------------------------------------------------------------

def test_shape(tmp_path):
    """recon_xct / mask shapes must equal the input volume shape after cropping."""
    from poregen.eval.blended import reconstruct_volume

    shape = (128, 96, 80)
    xct_p, mask_p = _write_tiffs(tmp_path, shape)
    model = _TinyVAE()
    model.eval()

    out = reconstruct_volume(
        xct_p, mask_p, model,
        patch_size=64, stride=48,
        n_samples=1, device="cpu", batch_size=8, seed=0,
    )

    assert out["recon_xct"].shape        == shape, out["recon_xct"].shape
    assert out["recon_mask_binary"].shape == shape
    assert out["recon_mask_prob"].shape   == shape
    assert out["recon_xct"].dtype         == np.float32
    assert out["recon_mask_binary"].dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 2 — determinism: near-zero σ → different seeds give identical output
# ---------------------------------------------------------------------------

def test_determinism(tmp_path):
    """With logvar=-100 (σ≈0), stochastic pass is deterministic regardless of seed."""
    from poregen.eval.blended import reconstruct_volume

    shape = (80, 80, 80)
    xct_p, mask_p = _write_tiffs(tmp_path, shape)
    model = _TinyVAE(fixed_logvar=-100.0)
    model.eval()

    common = dict(
        patch_size=64, stride=48, n_samples=1,
        device="cpu", batch_size=16,
    )
    out_a = reconstruct_volume(xct_p, mask_p, model, seed=0,  **common)
    out_b = reconstruct_volume(xct_p, mask_p, model, seed=999, **common)

    np.testing.assert_allclose(
        out_a["recon_xct"], out_b["recon_xct"],
        atol=1e-5,
        err_msg="Different seeds should give identical output when σ≈0",
    )


# ---------------------------------------------------------------------------
# Test 3 — weight accumulator covers every voxel in the original region
# ---------------------------------------------------------------------------

def test_weight_coverage():
    """Every voxel in the original (D,H,W) region must receive positive weight.

    Mirrors the symmetric pre-padding strategy used in reconstruct_volume:
    each dimension is pre-padded by taper_half = alpha/2 * patch_size so that
    original boundary voxels land in the flat (weight~1) region of the Tukey
    window, not in the taper (weight=0) region.
    """
    from poregen.eval.blended import _pad_size, _tukey_window_3d

    D, H, W = 128, 96, 80
    patch_size, stride, alpha = 64, 48, 0.5
    taper_half = max(1, int(patch_size * alpha / 2))  # 16

    pre = taper_half
    D_pre = D + 2 * pre
    H_pre = H + 2 * pre
    W_pre = W + 2 * pre

    pad_d = _pad_size(D_pre, patch_size, stride)
    pad_h = _pad_size(H_pre, patch_size, stride)
    pad_w = _pad_size(W_pre, patch_size, stride)
    padded_shape = (pad_d, pad_h, pad_w)

    W3d = _tukey_window_3d(patch_size)

    coords = [
        (z, y, x)
        for z in range(0, pad_d - patch_size + 1, stride)
        for y in range(0, pad_h - patch_size + 1, stride)
        for x in range(0, pad_w - patch_size + 1, stride)
    ]

    weight_acc = np.zeros(padded_shape, dtype=np.float64)
    for z0, y0, x0 in coords:
        weight_acc[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] += W3d

    # Crop to original region using the pre-padding offset
    cropped = weight_acc[pre:pre+D, pre:pre+H, pre:pre+W]
    assert cropped.min() > 0, (
        f"weight_acc has zero-weight voxels in original region "
        f"(min={cropped.min():.6f})"
    )

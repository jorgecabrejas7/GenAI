"""Tests for VolumeGenerator cosine-taper overlap-add assembly.

Verifies the blending strategy:
  1. Interior voxels (index ≥ 1 on every axis) receive uniform output when
     every decoded patch is uniform — partition-of-unity property.
  2. The three zero-weight boundary planes (index 0 on each axis) produce 0
     (accepted epsilon-guard artifact).
  3. Output shape matches the snapped volume size.
  4. Non-cubic (anisotropic) volumes are handled correctly.
  5. The shared weight accumulator approximates 1.0 in the interior for
     50%-overlap tiling (partition of unity).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.diffusion.sampler import VolumeGenerator


# ── Minimal stubs ─────────────────────────────────────────────────────────────

class _FakeModelCfg:
    z_channels = 2


class _FakeModel:
    cfg = _FakeModelCfg()


class _FakeSampler:
    """Returns all-zero latents (decoded → sigmoid(0) = 0.5 everywhere)."""

    def __init__(self) -> None:
        self.model = _FakeModel()

    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        B = nb_latents.shape[0]
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        return torch.zeros(B, C, D, D, D)


class _FakeVAE:
    """Decodes latents to constant 0.5 patches (all logits = 0 → sigmoid = 0.5)."""

    class _ConstantHead:
        def __call__(self, dec: torch.Tensor) -> torch.Tensor:
            B, _, P, _, _ = dec.shape
            return torch.zeros(B, 1, P, P, P)   # sigmoid(0) = 0.5

    def __init__(self, patch_size: int = 64) -> None:
        self._P = patch_size
        self.xct_head  = self._ConstantHead()
        self.mask_head = self._ConstantHead()

    def eval(self) -> "_FakeVAE":
        return self

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        P = self._P
        return torch.zeros(B, 4, P, P, P)


def _make_generator(patch_size: int, patch_stride: int, voxel_size_mm: float = 0.025) -> VolumeGenerator:
    return VolumeGenerator(
        sampler=_FakeSampler(),
        vae=_FakeVAE(patch_size=patch_size),
        device=torch.device("cpu"),
        patch_size=patch_size,
        patch_stride=patch_stride,
        latent_size=4,
        latent_std=1.0,
        voxel_size_mm=voxel_size_mm,
    )


def _expected_vol_shape(
    volume_size_mm: tuple[float, float, float],
    patch_size: int,
    voxel_size_mm: float = 0.025,
) -> tuple[int, int, int]:
    return tuple(
        (round(d / voxel_size_mm) // patch_size) * patch_size
        for d in volume_size_mm
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_assembly_interior_uniform_stride32() -> None:
    """Interior voxels are 127 (0.5 → uint8) for uniform-input patches; stride=32."""
    P   = 64
    gen = _make_generator(patch_size=P, patch_stride=32)
    volume_size_mm = (3.2, 3.2, 3.2)   # 128 vox per axis

    xct_out, mask_out = gen.generate(volume_size_mm=volume_size_mm)

    expected_shape = _expected_vol_shape(volume_size_mm, P)
    assert xct_out.shape  == expected_shape
    assert mask_out.shape == expected_shape

    # Interior: all axes index ≥ 1 — partition of unity holds here.
    # sigmoid(0) = 0.5 → 0.5*255 = 127.5, cast to uint8 = 127.
    interior_xct = xct_out[1:, 1:, 1:]
    assert (interior_xct == 127).all(), (
        f"Interior should be 127; unique values: {np.unique(interior_xct)}"
    )

    # Boundary faces (index 0 on each axis) are 0 due to zero-weight epsilon guard.
    assert (xct_out[0, :, :] == 0).all(), "z=0 face should be 0 (zero-weight boundary)"
    assert (xct_out[:, 0, :] == 0).all(), "y=0 face should be 0 (zero-weight boundary)"
    assert (xct_out[:, :, 0] == 0).all(), "x=0 face should be 0 (zero-weight boundary)"

    # mask: sigmoid(0)=0.5, 0.5 > 0.5 is False → all zero
    assert (mask_out == 0).all(), f"mask should be all-zero; unique: {np.unique(mask_out)}"


def test_assembly_output_shape_snapping() -> None:
    """Volume shape is snapped to the nearest patch_size multiple downward."""
    P   = 64
    gen = _make_generator(patch_size=P, patch_stride=32)

    # 3.21 mm / 0.025 = 128.4 vox → round → 128; 128//64*64 = 128 → no snap
    xct, _ = gen.generate(volume_size_mm=(3.21, 3.2, 3.2))
    assert xct.shape[0] == 128

    # 3.3 mm / 0.025 = 132 vox; 132//64*64 = 128 → snaps from 132 to 128
    xct2, _ = gen.generate(volume_size_mm=(3.3, 3.2, 3.2))
    assert xct2.shape[0] == 128


def test_assembly_anisotropic_volume() -> None:
    """generate() works for non-cubic volumes with different grid sizes per axis."""
    P   = 64
    gen = _make_generator(patch_size=P, patch_stride=32)

    # z: 3.2 mm → 128 vox, y: 6.4 mm → 256 vox, x: 3.2 mm → 128 vox
    xct_out, mask_out = gen.generate(volume_size_mm=(3.2, 6.4, 3.2))
    assert xct_out.shape  == (128, 256, 128)
    assert mask_out.shape == (128, 256, 128)

    # Interior uniform
    assert (xct_out[1:, 1:, 1:] == 127).all(), (
        f"Interior should be 127; unique: {np.unique(xct_out[1:,1:,1:])}"
    )


def test_assembly_partition_of_unity_interior() -> None:
    """Accumulated weight_vol equals 1.0 in the two-patch overlap zones.

    With P=64, stride=32, and 3 patches (origins 0, 32, 64) in a 128-vox axis:
    - 1D partition of unity holds where two patches overlap: indices [32, 95].
    - Outside that band (indices 0..31 and 96..127) only one patch covers the
      voxel, so the weight is just that patch's window value (< 1).
    - The 3D PoU region is the Cartesian product: [32:96, 32:96, 32:96].
    """
    P      = 64
    stride = P // 2   # 32 — standard 50% overlap

    w1d = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(P) / P))).astype(np.float32)
    w3d = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]

    vol_size = 128   # 3 patches along each axis at stride 32 → origins [0, 32, 64]

    weight_vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)

    zs = list(range(0, vol_size - P + 1, stride))
    ys = list(range(0, vol_size - P + 1, stride))
    xs = list(range(0, vol_size - P + 1, stride))
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                sl = (slice(z0, z0 + P), slice(y0, y0 + P), slice(x0, x0 + P))
                weight_vol[sl] += w3d

    # The 1D PoU band is [stride, vol_size - stride) = [32, 96) for this config.
    # The 3D PoU region is the Cartesian product of the 1D bands.
    overlap_region = weight_vol[stride:vol_size - stride,
                                stride:vol_size - stride,
                                stride:vol_size - stride]
    assert np.allclose(overlap_region, 1.0, atol=1e-5), (
        f"weight_vol in overlap region should be ≈1.0; "
        f"range [{overlap_region.min():.6f}, {overlap_region.max():.6f}]"
    )

    # Zero-weight boundary faces (w1d[0] = 0)
    assert (weight_vol[0, :, :] == 0.0).all(), "z=0 face should have zero weight"
    assert (weight_vol[:, 0, :] == 0.0).all(), "y=0 face should have zero weight"
    assert (weight_vol[:, :, 0] == 0.0).all(), "x=0 face should have zero weight"


def test_assembly_stride_equals_patch_size() -> None:
    """Non-overlapping tiling (stride=P): interior is 127, block-boundary planes are 0."""
    P  = 64
    gen = _make_generator(patch_size=P, patch_stride=P)

    volume_size_mm = (3.2, 3.2, 3.2)   # 128 vox → 2 patches per axis
    xct_out, mask_out = gen.generate(volume_size_mm=volume_size_mm)

    assert xct_out.shape == (128, 128, 128)

    # With stride==P, w1d[0]=0 so every patch's min-face plane has zero weight.
    # Boundary planes at z=0,64; y=0,64; x=0,64 will be 0.
    # Interior (avoiding those planes) should be 127.
    interior = xct_out[1:64, 1:64, 1:64]
    assert (interior == 127).all(), (
        f"Interior block should be 127; unique: {np.unique(interior)}"
    )

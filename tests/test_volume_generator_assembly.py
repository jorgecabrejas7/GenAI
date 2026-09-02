"""Tests for VolumeGenerator direct-tiling assembly.

``generation_stride == patch_size``, so every decoded patch owns its own block
of the output.  There is no overlap, therefore no cosine-taper window, no
weight buffer and no zero-weight boundary planes.  Verifies:

  1. Uniform decoded patches give a uniform volume — including index 0, which
     the old Hann window forced to 0.
  2. Output shape matches the snapped volume size.
  3. Non-cubic (anisotropic) volumes are handled correctly.
  4. Each patch writes exactly its own block: patch content is not mixed.
  5. generate() reports the seam diagnostic in its stats.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.diffusion.sampler import VolumeGenerator


# ── Minimal stubs ─────────────────────────────────────────────────────────────

class _FakeModelCfg:
    z_channels = 2
    use_por_cond = False
    use_orient_cond = False


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
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist: torch.Tensor,
        cond_orient: torch.Tensor | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        B = nb_latents.shape[0]
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        return torch.zeros(B, C, D, D, D)


class _PerPatchSampler:
    """Gives every patch a latent filled with its own depth-derived value."""

    def __init__(self) -> None:
        self.model = _FakeModel()

    def sample_batch(self, nb_latents, nb_avail, cond_por, cond_depth,
                     cond_dist, cond_orient=None, autocast_dtype=torch.bfloat16):
        B = nb_latents.shape[0]
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        out = torch.zeros(B, C, D, D, D)
        for i in range(B):
            out[i] = float(cond_depth[i])
        return out


class _FakeVAE:
    """Decodes latents to constant patches (logit = the latent's mean)."""

    class _MeanHead:
        def __init__(self, patch_size: int) -> None:
            self.P = patch_size

        def __call__(self, dec: torch.Tensor) -> torch.Tensor:
            B = dec.shape[0]
            per_sample = dec.mean(dim=(1, 2, 3, 4)).view(B, 1, 1, 1, 1)
            return per_sample.expand(B, 1, self.P, self.P, self.P)

    def __init__(self, patch_size: int = 64) -> None:
        self._P = patch_size
        self.xct_head  = self._MeanHead(patch_size)
        self.mask_head = self._MeanHead(patch_size)

    def eval(self) -> "_FakeVAE":
        return self

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        return z


def _make_generator(patch_size: int, sampler=None,
                    voxel_size_mm: float = 0.025) -> VolumeGenerator:
    return VolumeGenerator(
        sampler=sampler if sampler is not None else _FakeSampler(),
        vae=_FakeVAE(patch_size=patch_size),
        device=torch.device("cpu"),
        patch_size=patch_size,
        generation_stride=patch_size,
        neighbour_offset=patch_size,
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

def test_assembly_is_uniform_everywhere() -> None:
    """Uniform patches give a uniform volume, boundary planes included."""
    P   = 64
    gen = _make_generator(patch_size=P)
    volume_size_mm = (3.2, 3.2, 3.2)   # 128 vox per axis → 2 patches per axis

    xct_out, mask_out, _ = gen.generate(volume_size_mm=volume_size_mm)

    expected_shape = _expected_vol_shape(volume_size_mm, P)
    assert xct_out.shape  == expected_shape
    assert mask_out.shape == expected_shape

    # The XCT head regresses xct/255 directly (no activation), so a head that
    # outputs 0 decodes to grey level 0 — not sigmoid(0)*255 = 127.  No window means
    # index 0 on each axis is no longer a zero-weight artifact.
    assert (xct_out == 0).all(), f"unique values: {np.unique(xct_out)}"
    # mask: logit 0 is not > 0 → all zero
    assert (mask_out == 0).all(), f"unique: {np.unique(mask_out)}"


def test_assembly_output_shape_snapping() -> None:
    """Volume shape is snapped to the nearest patch_size multiple downward."""
    P   = 64
    gen = _make_generator(patch_size=P)

    # 3.21 mm / 0.025 = 128.4 vox → round → 128; 128//64*64 = 128 → no snap
    xct, _, _ = gen.generate(volume_size_mm=(3.21, 3.2, 3.2))
    assert xct.shape[0] == 128

    # 3.3 mm / 0.025 = 132 vox; 132//64*64 = 128 → snaps from 132 to 128
    xct2, _, _ = gen.generate(volume_size_mm=(3.3, 3.2, 3.2))
    assert xct2.shape[0] == 128


def test_assembly_anisotropic_volume() -> None:
    """generate() works for non-cubic volumes with different grid sizes per axis."""
    P   = 64
    gen = _make_generator(patch_size=P)

    # z: 3.2 mm → 128 vox, y: 6.4 mm → 256 vox, x: 3.2 mm → 128 vox
    xct_out, mask_out, _ = gen.generate(volume_size_mm=(3.2, 6.4, 3.2))
    assert xct_out.shape  == (128, 256, 128)
    assert mask_out.shape == (128, 256, 128)
    assert (xct_out == 0).all()


def test_each_patch_writes_exactly_its_own_block() -> None:
    """No averaging: a patch's decoded value appears verbatim in its block.

    Patches differ by cond_depth, which _PerPatchSampler bakes into the latent
    and _FakeVAE passes through as the decoded logit.  Every voxel of a block
    must carry that patch's value and nothing of its neighbours'.
    """
    from scipy.special import expit

    P   = 64
    gen = _make_generator(patch_size=P, sampler=_PerPatchSampler())
    xct, _, _ = gen.generate(volume_size_mm=(4.8, 1.6, 1.6))   # 192 x 64 x 64
    assert xct.shape == (192, 64, 64)

    for iz in range(3):
        z0 = iz * P
        depth = (z0 + P / 2.0) / 192.0            # matches _patch_position
        expected = int(round(float(np.clip(depth, 0.0, 1.0)) * 255.0))
        block = xct[z0:z0 + P]
        assert np.unique(block).size == 1, "a block must be uniform, not blended"
        assert abs(int(block.flat[0]) - expected) <= 1


def test_generate_reports_the_seam_diagnostic() -> None:
    """The assembly-quality metric reaches the stats dict generate() returns."""
    P   = 64
    gen = _make_generator(patch_size=P, sampler=_PerPatchSampler())
    _, _, stats = gen.generate(volume_size_mm=(4.8, 1.6, 1.6))

    for key in ("seam_xct_ratio", "seam_xct_mad", "seam_xct_interior_mad",
                "seam_xct_planes", "seam_mask_ratio"):
        assert key in stats
    for axis in ("z", "y", "x"):
        assert f"seam_xct_{axis}_ratio" in stats
    assert stats["seam_xct_z_planes"] == 2       # 3 blocks in z → 2 seams
    assert stats["seam_xct_y_planes"] == 0
    # Constant blocks with a step between them: interior is flat, seam is not,
    # so the ratio is infinite/NaN rather than a small number.
    assert stats["seam_xct_z_mad"] > 0.0
    assert stats["seam_xct_z_interior_mad"] == pytest.approx(0.0, abs=1e-6)


def test_stats_still_carry_the_porosity_self_audit() -> None:
    P   = 64
    gen = _make_generator(patch_size=P)
    _, mask, stats = gen.generate(volume_size_mm=(1.6, 1.6, 1.6),
                                  target_porosity=0.05)
    assert stats["actual_mask_porosity"] == pytest.approx(float((mask > 0).mean()))
    assert stats["target_porosity"] == pytest.approx(0.05)
    assert stats["conditioned_porosity"] == pytest.approx(0.05)

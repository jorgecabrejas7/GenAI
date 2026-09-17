"""The full-patch porosity a window is conditioned on.

`cond_por` is pore over the WHOLE patch, air included in the denominator,
while a request is material porosity — pore over specimen. Converting one to
the other is a per-cell product, not a product of two means, and the difference
is the covariance between the requested field and the specimen shape.

A window half exterior air at a high requested porosity and half material at a
low one is the case that separates them, and it is not exotic: it is what every
window on a specimen edge under a graded request looks like.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.diffusion.sampler import window_tile_cells, window_tile_mean

P, L = 64, 16
DS = P // L


def full_patch_phi_cellwise(field, material):
    """The definition: mean over cells of phi(x) * m(x)."""
    return float((np.asarray(field) * np.asarray(material)).mean())


def full_patch_phi_by_means(field, material):
    """What the sampler used to do."""
    return float(np.mean(field)) * float(np.mean(material))


class TestTheCellField:

    def test_a_cell_never_straddles_a_tile_boundary(self):
        """Which is what makes the sampling exact rather than an interpolation."""
        assert P % L == 0 and P % DS == 0

    def test_a_uniform_field_is_reproduced_everywhere(self):
        cells = window_tile_cells((0, 0, 0), P, L, {}, 0.037)
        assert cells.shape == (L, L, L)
        assert np.allclose(cells, 0.037)

    def test_each_cell_takes_the_value_of_the_tile_it_lies_in(self):
        field = {(0, 0, 0): 0.01, (0, 0, 1): 0.05}
        # A window at x-origin 32 straddles tiles 0 and 1 half and half.
        cells = window_tile_cells((0, 0, 32), P, L, field, 0.0)
        assert np.allclose(cells[:, :, : L // 2], 0.01)
        assert np.allclose(cells[:, :, L // 2 :], 0.05)

    def test_its_mean_agrees_with_the_scalar_helper(self):
        """The two must not disagree about the FIELD; they differ only in what
        is done with it afterwards."""
        rng = np.random.default_rng(0)
        field = {(z, y, x): float(rng.uniform(0.005, 0.09))
                 for z in range(3) for y in range(3) for x in range(3)}
        for origin in ((0, 0, 0), (0, 32, 32), (32, 96, 64), (0, 16, 48)):
            a = window_tile_cells(origin, P, L, field, 0.03).mean()
            b = window_tile_mean(origin, P, field, 0.03)
            assert a == pytest.approx(b, abs=1e-6), origin


class TestTheProductOfMeansIsWrong:

    def test_half_air_at_high_phi_against_half_material_at_low_phi(self):
        """The reproduction case, with the answer worked out by hand.

        Half the window is exterior air (m = 0) under a 0.10 request; the other
        half is solid specimen (m = 1) under a 0.01 request. No pore can exist
        in the air half, so the patch holds 0.01 x half a patch of pore:
        phi_full = 0.005.

        The product of means gives mean(phi) = 0.055 times mean(m) = 0.5, i.e.
        0.0275 — five and a half times the truth, and above it rather than
        below, so it asks the model for pores the request never wanted.
        """
        field = np.empty((L, L, L), np.float32)
        field[:, :, : L // 2] = 0.10      # the air half, high request
        field[:, :, L // 2 :] = 0.01      # the material half, low request
        material = np.zeros((L, L, L), np.float32)
        material[:, :, L // 2 :] = 1.0

        assert full_patch_phi_cellwise(field, material) == pytest.approx(0.005)
        assert full_patch_phi_by_means(field, material) == pytest.approx(0.0275)
        assert full_patch_phi_by_means(field, material) > \
               5 * full_patch_phi_cellwise(field, material)

    def test_they_agree_when_the_field_is_uniform(self):
        """Which is why every uniform-request campaign is unaffected."""
        rng = np.random.default_rng(1)
        material = rng.uniform(0, 1, (L, L, L)).astype(np.float32)
        field = np.full((L, L, L), 0.03, np.float32)
        assert full_patch_phi_cellwise(field, material) == pytest.approx(
            full_patch_phi_by_means(field, material), rel=1e-6)

    def test_they_agree_when_the_window_is_all_specimen(self):
        """And why an interior window under a graded request is unaffected too."""
        rng = np.random.default_rng(2)
        field = rng.uniform(0.005, 0.09, (L, L, L)).astype(np.float32)
        material = np.ones((L, L, L), np.float32)
        assert full_patch_phi_cellwise(field, material) == pytest.approx(
            full_patch_phi_by_means(field, material), rel=1e-6)

    def test_the_error_is_the_covariance_and_has_either_sign(self):
        """A field that is LOW where the specimen is thin under-asks instead."""
        field = np.empty((L, L, L), np.float32)
        field[:, :, : L // 2] = 0.01      # air half, low request
        field[:, :, L // 2 :] = 0.10      # material half, high request
        material = np.zeros((L, L, L), np.float32)
        material[:, :, L // 2 :] = 1.0
        assert full_patch_phi_cellwise(field, material) == pytest.approx(0.05)
        assert full_patch_phi_by_means(field, material) == pytest.approx(0.0275)
        # Here the old formula ASKS FOR TOO FEW pores, so the defect is not a
        # bias in one direction that could be calibrated away.
        assert full_patch_phi_by_means(field, material) < \
               full_patch_phi_cellwise(field, material)


class TestTheSamplerUsesIt:

    def test_the_sampler_computes_the_cellwise_product(self):
        """Reads the real code path, not a copy of the formula."""
        import inspect

        from poregen.diffusion.sampler import VolumeGenerator

        src = inspect.getsource(VolumeGenerator._window_conditioning)
        assert "window_tile_cells" in src
        assert "(cells * block).mean()" in src
        assert "phi *= float(block.mean())" not in src

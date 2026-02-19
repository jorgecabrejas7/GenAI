"""Test that integral-volume porosity matches brute-force per-patch mean."""

import numpy as np
import pytest

from poregen.dataset.patch_index import (
    compute_integral_volume,
    query_integral_volume,
)


def _brute_force_sum(mask, z0, y0, x0, ps):
    return int(mask[z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps].sum())


class TestIntegralPorosity:

    def test_uniform_ones(self):
        mask = np.ones((32, 32, 32), dtype=np.uint8)
        sat = compute_integral_volume(mask)
        assert query_integral_volume(sat, 0, 0, 0, 16) == 16 ** 3

    def test_uniform_zeros(self):
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        sat = compute_integral_volume(mask)
        assert query_integral_volume(sat, 0, 0, 0, 16) == 0

    def test_single_voxel(self):
        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[5, 7, 3] = 1
        sat = compute_integral_volume(mask)
        # Patch containing the voxel
        assert query_integral_volume(sat, 0, 0, 0, 16) == 1
        # Patch NOT containing the voxel
        assert query_integral_volume(sat, 6, 0, 0, 8) == 0

    def test_matches_brute_force_random(self):
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(64, 64, 64), dtype=np.uint8)
        sat = compute_integral_volume(mask)

        ps = 16
        for _ in range(50):
            z0 = rng.integers(0, 64 - ps + 1)
            y0 = rng.integers(0, 64 - ps + 1)
            x0 = rng.integers(0, 64 - ps + 1)
            expected = _brute_force_sum(mask, z0, y0, x0, ps)
            got = query_integral_volume(sat, z0, y0, x0, ps)
            assert got == expected, f"Mismatch at ({z0},{y0},{x0}): {got} != {expected}"

    def test_porosity_value(self):
        rng = np.random.default_rng(99)
        mask = rng.integers(0, 2, size=(32, 32, 32), dtype=np.uint8)
        sat = compute_integral_volume(mask)
        ps = 16
        s = query_integral_volume(sat, 0, 0, 0, ps)
        porosity = s / ps ** 3
        expected = mask[:ps, :ps, :ps].mean()
        assert abs(porosity - expected) < 1e-6

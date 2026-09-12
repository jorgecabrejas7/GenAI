"""The rim statistic, on volumes whose answer is planted.

`window_rim_test.py` exists to decide whether the model under-predicts pores in
the rim of a window that has a neighbour.  Its whole output is one profile, so
that profile has to separate a RIM effect from a porosity difference: a rim
effect moves the first shells and leaves the tail flat, a porosity difference
moves every shell together.  If the statistic cannot tell those apart it will
confirm whichever hypothesis it is pointed at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import window_rim_test as W  # noqa: E402

MATERIAL, PORE, AIR = 0, 1, 2


TILES = (4, 4, 4)          # small, and still has interior windows
SHAPE = W.shape_of(TILES)


def canvas(fill=MATERIAL) -> np.ndarray:
    return np.full(SHAPE, fill, np.uint8)


def per_tile(fn) -> np.ndarray:
    """Build a volume by writing the same 64-cubed pattern into every tile."""
    lab = canvas()
    n = W.TILE
    tile = fn(np.indices((n, n, n)))
    for tz in range(TILES[0]):
        for ty in range(TILES[1]):
            for tx in range(TILES[2]):
                lab[tz * n:(tz + 1) * n, ty * n:(ty + 1) * n, tx * n:(tx + 1) * n] = tile
    return lab


class TestRimProfile:
    """The tile grid is not cubic in production (3 x 10 x 10): no val volume
    holds a 384-deep block. These use (4, 4, 4) for speed; the statistic takes
    the tile counts as an argument for exactly that reason."""


    def test_uniform_porosity_is_flat_across_every_shell(self):
        """A porosity difference must NOT look like a rim effect."""
        rng = np.random.default_rng(0)
        lab = np.where(rng.random(SHAPE) < 0.25, PORE, MATERIAL).astype(np.uint8)
        prof = W.rim_profile(lab, TILES)
        vals = [v for v in prof.values() if v is not None]
        assert len(vals) == 4
        assert max(vals) - min(vals) < 0.01          # flat to a percent

    def test_a_planted_rim_shows_in_the_first_shells_only(self):
        """Pore-free in the outer 16 voxels of every tile, porous inside."""
        n = W.TILE

        def pattern(idx):
            d = np.minimum(np.minimum(idx[0], n - 1 - idx[0]),
                           np.minimum(np.minimum(idx[1], n - 1 - idx[1]),
                                      np.minimum(idx[2], n - 1 - idx[2])))
            return np.where(d < 16, MATERIAL, PORE).astype(np.uint8)

        prof = W.rim_profile(per_tile(pattern), TILES)
        assert prof["0"] == pytest.approx(0.0)
        assert prof["8"] == pytest.approx(0.0)
        assert prof["16"] == pytest.approx(1.0)
        assert prof["24"] == pytest.approx(1.0)

    def test_the_shells_partition_the_tile_exactly(self):
        """Every voxel is counted once, or the profile is not a decomposition."""
        lab = per_tile(lambda idx: np.full(idx[0].shape, PORE, np.uint8))
        prof = W.rim_profile(lab, TILES)
        assert all(v == pytest.approx(1.0) for v in prof.values())
        n = W.TILE
        idx = np.arange(n)
        d1 = np.minimum(idx, n - 1 - idx)
        d3 = np.minimum(np.minimum(d1[:, None, None], d1[None, :, None]), d1[None, None, :])
        counted = sum(int(((d3 >= d) & (d3 < d + W.SLAB)).sum())
                      for d in range(0, n // 2, W.SLAB))
        assert counted == n ** 3
        assert W.interior_phi(lab, TILES) == pytest.approx(1.0)
        assert W.n_interior(TILES) == (4 - 2) ** 3

    def test_only_the_interior_windows_are_pooled(self):
        """The boundary tiles have an OOB face and must not enter the profile.

        Filling ONLY the boundary tiles with pore has to leave the profile at
        zero: if it does not, windows without six real neighbours are being
        counted and the experiment answers a different question.
        """
        lab = canvas()
        n = W.TILE
        rz, ry, rx = W.interior_range(TILES)
        for tz in range(TILES[0]):
            for ty in range(TILES[1]):
                for tx in range(TILES[2]):
                    if tz in rz and ty in ry and tx in rx:
                        continue
                    lab[tz * n:(tz + 1) * n, ty * n:(ty + 1) * n, tx * n:(tx + 1) * n] = PORE
        assert W.rim_profile(lab, TILES) == {
            k: pytest.approx(0.0) for k in ("0", "8", "16", "24")}
        assert W.interior_phi(lab, TILES) == pytest.approx(0.0)

    def test_air_is_excluded_from_the_denominator(self):
        """phi is pore over SOLID; a tile that is half air must not read as half phi."""
        lab = per_tile(lambda idx: np.where(idx[0] < 32, AIR, PORE).astype(np.uint8))
        prof = W.rim_profile(lab, TILES)
        assert all(v == pytest.approx(1.0) for v in prof.values() if v is not None)

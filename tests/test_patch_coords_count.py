"""Test that patch coordinate generation produces the expected count."""

import pytest

from poregen.dataset.patch_index import generate_patch_coords


def _expected_count(shape, ps, stride):
    """Brute-force expected number of patches."""
    D, H, W = shape
    nz = max(0, (D - ps) // stride + 1)
    nh = max(0, (H - ps) // stride + 1)
    nw = max(0, (W - ps) // stride + 1)
    return nz * nh * nw


class TestPatchCoordsCount:

    @pytest.mark.parametrize(
        "shape,ps,stride",
        [
            ((128, 128, 128), 64, 64),
            ((128, 128, 128), 64, 32),
            ((100, 200, 300), 32, 16),
            ((64, 64, 64), 64, 64),
            ((63, 63, 63), 64, 64),    # no patches fit
            ((65, 65, 65), 64, 64),    # exactly 1 patch
        ],
    )
    def test_count(self, shape, ps, stride):
        coords = generate_patch_coords(shape, ps, stride)
        assert len(coords) == _expected_count(shape, ps, stride)

    def test_coords_within_bounds(self):
        shape = (100, 200, 150)
        ps = 32
        coords = generate_patch_coords(shape, ps, 16)
        for z0, y0, x0 in coords:
            assert 0 <= z0 <= shape[0] - ps
            assert 0 <= y0 <= shape[1] - ps
            assert 0 <= x0 <= shape[2] - ps

"""Specimen-box selection in ``poregen.dataset.segmentation.material_mask``.

The specimen bounding box is taken from the max-projection of the Otsu binary.
It must come from the LARGEST projected component, not from whichever component
``measure.label`` happened to number first: a bright dust speck near the top-left
corner gets label 1 and would otherwise become the specimen.

When two components are of comparable size there is no single specimen to pick,
so the scan is ambiguous and must fail loudly instead of being segmented against
an arbitrary half of itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.dataset.segmentation import (
    AMBIGUOUS_COMPONENT_RATIO,
    compute_sample_mask,
    material_mask,
)


def _volume(shape=(4, 64, 64)) -> np.ndarray:
    return np.zeros(shape, np.uint8)


class TestSpecimenSelection:

    def test_small_speck_does_not_become_the_specimen(self):
        """A speck with the lowest label id must not steal the bounding box."""
        xct = _volume()
        xct[:, 1:4, 1:4] = 255            # 9-px speck, labelled FIRST
        xct[:, 20:60, 20:60] = 200        # 1600-px specimen, labelled second

        sm = material_mask(xct)

        assert sm[:, 20:60, 20:60].all()          # specimen is the mask
        assert not sm[:, 1:4, 1:4].any()          # speck is outside the box
        assert int(sm.sum()) == 4 * 40 * 40

    def test_speck_in_every_corner_still_picks_the_specimen(self):
        """Label order depends on where the specks sit; the answer must not."""
        xct = _volume()
        xct[:, 0:3, 0:3] = 255
        xct[:, 0:3, 61:64] = 255
        xct[:, 61:64, 0:3] = 255
        xct[:, 24:56, 24:56] = 200

        sm = material_mask(xct)

        assert sm[:, 24:56, 24:56].all()
        assert int(sm.sum()) == 4 * 32 * 32

    def test_internal_voids_are_still_filled(self):
        """Picking the largest component must not change the fill behaviour."""
        xct = _volume()
        xct[:, 2:5, 2:5] = 255
        xct[:, 16:48, 16:48] = 200
        xct[1:3, 28:36, 28:36] = 5        # dark internal pore

        sm = material_mask(xct)

        assert sm[1, 30, 30]
        assert sm[:, 16:48, 16:48].all()

    def test_compute_sample_mask_ignores_a_speck(self):
        """The dataset-build entry point inherits the fix."""
        xct = np.zeros((16, 64, 64), np.uint8)
        xct[:, 1:4, 1:4] = 255
        xct[4:12, 20:52, 20:52] = 200

        sm = compute_sample_mask(xct)

        assert sm is not None
        assert sm[8, 36, 36]
        assert not sm[8, 2, 2]


class TestAmbiguousScan:

    def test_two_comparable_components_raise(self):
        """Half a specimen is not a specimen: refuse rather than guess."""
        xct = _volume()
        xct[:, 4:44, 4:24] = 200          # 800 px
        xct[:, 4:44, 34:54] = 210         # 800 px

        with pytest.raises(ValueError, match="ambiguous"):
            material_mask(xct)

    def test_error_names_both_component_areas(self):
        xct = _volume()
        xct[:, 4:44, 4:24] = 200          # 800 px
        xct[:, 10:30, 34:54] = 210        # 400 px

        with pytest.raises(ValueError) as excinfo:
            material_mask(xct)
        message = str(excinfo.value)
        assert "800" in message and "400" in message

    def test_second_component_at_the_ratio_is_accepted(self):
        """The gate is 'exceeds', so exactly 10 % passes."""
        assert AMBIGUOUS_COMPONENT_RATIO == 0.10
        xct = _volume((4, 64, 64))
        xct[:, 4:24, 4:54] = 200          # 20 x 50 = 1000 px
        xct[:, 40:50, 4:14] = 210         # 10 x 10 =  100 px  -> exactly 10 %

        sm = material_mask(xct)

        assert sm[:, 4:24, 4:54].all()
        assert not sm[:, 40:50, 4:14].any()

    def test_second_component_just_over_the_ratio_raises(self):
        xct = _volume((4, 64, 64))
        xct[:, 4:24, 4:54] = 200          # 1000 px
        xct[:, 40:51, 4:14] = 210         # 11 x 10 = 110 px  -> 11 %

        with pytest.raises(ValueError, match="ambiguous"):
            material_mask(xct)

    def test_compute_sample_mask_propagates_the_error(self):
        xct = np.zeros((8, 64, 64), np.uint8)
        xct[:, 4:44, 4:24] = 200
        xct[:, 4:44, 34:54] = 210

        with pytest.raises(ValueError, match="ambiguous"):
            compute_sample_mask(xct)


def test_single_component_volume_is_unchanged():
    """The common case — one specimen, nothing else — behaves as before."""
    xct = np.zeros((40, 40, 40), np.uint8)
    xct[8:32, 8:32, 8:32] = 200
    sm = material_mask(xct)
    assert sm[8:32, 8:32, 8:32].all()
    assert not sm[:, 2, 2].any()

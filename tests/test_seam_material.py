"""The material-restricted seam, against the situation it was written for.

Campaign 19 has two requests whose specimen boundary lands exactly on a chunk
plane — the L-bracket's 192-voxel leg against a 192-voxel chunk period, and the
two-coupon gap edge at x = 576 = 3 x 192. The plain seam statistic reads 1.85
and 1.95 there against a real-material floor of 0.906, purely because the
specimen ends at the plane. These tests build that situation deliberately and
check the restricted reading removes it while the plain one does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.eval_v4 import metrics as M

PERIOD = 32
SHAPE = (64, 64, 64)


def seamless(seed=0):
    """A volume with ordinary texture and no seam anywhere."""
    rng = np.random.default_rng(seed)
    return (0.5 + 0.02 * rng.standard_normal(SHAPE)).astype(np.float32)


class TestEquivalence:

    def test_all_material_reproduces_the_plain_metric(self):
        """The restricted reading must not be a different statistic — with
        nothing excluded it has to give the plain answer exactly."""
        v = seamless()
        full = np.ones(SHAPE, bool)
        a = M.seam_discontinuity(v, PERIOD, prefix="p")
        b = M.seam_discontinuity_material(v, full, PERIOD, prefix="p")
        for axis in ("z", "y", "x"):
            assert a[f"p_{axis}_ratio"] == pytest.approx(b[f"p_{axis}_ratio"], rel=1e-4)

    def test_a_seamless_volume_reads_about_one_either_way(self):
        v = seamless()
        full = np.ones(SHAPE, bool)
        b = M.seam_discontinuity_material(v, full, PERIOD, prefix="p")
        assert 0.8 < b["p_ratio"] < 1.25


class TestSpecimenBoundaryOnAPlane:
    """The defect the metric exists for."""

    def build(self):
        """Material ends at x = 32, which IS a chunk plane. No assembly seam."""
        v = seamless()
        mat = np.ones(SHAPE, bool)
        mat[:, :, PERIOD:] = False
        # Outside the specimen the decoder writes air: dark and flat. That step
        # at x = 32 is the specimen ending, not two chunk solves disagreeing.
        v[:, :, PERIOD:] = 0.02
        return v, mat

    def test_the_plain_seam_is_inflated_by_the_boundary(self):
        v, mat = self.build()
        a = M.seam_discontinuity(v, PERIOD, prefix="p")
        assert a["p_x_ratio"] > 5, (
            "a specimen edge on the plane should wreck the plain reading; "
            "if it does not, this test no longer tests anything")

    def test_the_material_restricted_seam_is_not(self):
        v, mat = self.build()
        b = M.seam_discontinuity_material(v, mat, PERIOD, prefix="p")
        # Every pair at x = 32 has NO shared material, so that plane carries no
        # reading at all and the axis falls back to the planes that do.
        assert np.isnan(b["p_x_ratio"]) or b["p_x_ratio"] < 2.0

    def test_a_boundary_NEAR_a_plane_is_still_removed(self):
        """The harder case: material ends at 33, so the pair at 32 straddles
        real material on both sides and the pair at 33 does not."""
        v = seamless()
        mat = np.ones(SHAPE, bool)
        mat[:, :, PERIOD + 1:] = False
        v[:, :, PERIOD + 1:] = 0.02
        plain = M.seam_discontinuity(v, PERIOD, prefix="p")["p_x_ratio"]
        restricted = M.seam_discontinuity_material(
            v, mat, PERIOD, prefix="p")["p_x_ratio"]
        assert restricted < plain
        assert restricted < 2.0


class TestRealSeamSurvives:
    """The metric must still SEE an assembly seam — removing the geometry must
    not remove the thing the statistic is for."""

    def test_a_genuine_step_at_the_plane_is_still_caught(self):
        v = seamless()
        v[:, :, PERIOD:] += 0.25          # a real discontinuity, all material
        mat = np.ones(SHAPE, bool)
        b = M.seam_discontinuity_material(v, mat, PERIOD, prefix="p")
        assert b["p_x_ratio"] > 5

    def test_a_genuine_step_is_caught_inside_a_shaped_specimen(self):
        """Both at once: a shaped specimen AND a real seam. The seam must
        survive the restriction."""
        v = seamless()
        mat = np.ones(SHAPE, bool)
        mat[:, :, 56:] = False            # specimen ends away from the plane
        v[:, :, 56:] = 0.02
        v[:, :, PERIOD:56] += 0.25        # real step at the plane, in material
        b = M.seam_discontinuity_material(v, mat, PERIOD, prefix="p")
        assert b["p_x_ratio"] > 5


class TestGuards:

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match="same shape"):
            M.seam_discontinuity_material(
                seamless(), np.ones((32, 32, 32), bool), PERIOD)

    def test_a_plane_with_too_little_shared_material_is_dropped(self):
        """Not reported as zero. A handful of voxels is noise with a number."""
        v = seamless()
        mat = np.zeros(SHAPE, bool)
        mat[:2, :2, :] = True             # 4 voxels per plane, under the floor
        b = M.seam_discontinuity_material(v, mat, PERIOD, prefix="p")
        assert np.isnan(b["p_x_ratio"])

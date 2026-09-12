"""The stress geometries, each against a closed form it cannot fake.

A voxel mask is only the shape it claims to be if something independent says
so. Every builder here has an analytic volume — a cylinder annulus, a spherical
shell, a linear taper — and the test checks the mask against that rather than
against a previously recorded number, which would only pin whatever the code
did the first time.

The small canvases are deliberate: the geometry is scale-free in every case
that has a closed form, and a 1024-cubed check would cost minutes of CPU while
the GPU is training.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.eval_v4 import stress_geometry as SG
from poregen.eval_v4.cases import STRESS_DDIM, STRESS_GEOMETRIES, build_cases

TILE = 64


class TestTube:

    def test_the_wall_volume_matches_the_annulus(self):
        shape = (128, 128, 256)
        m = SG.material_tube(shape, outer=50, wall=20)
        expected = np.pi * (50 ** 2 - 30 ** 2) / (128 * 128)
        assert m.mean() == pytest.approx(expected, rel=0.02)

    def test_it_is_hollow_and_uniform_along_its_axis(self):
        m = SG.material_tube((128, 128, 256), outer=50, wall=20)
        assert not m[64, 64, 128]                    # the bore is air
        assert m[64, 64 - 40, 128]                   # the wall is material
        # Every slice along x is the same annulus: the axis really is x.
        assert (m[:, :, 0] == m[:, :, -1]).all()

    def test_a_tube_that_does_not_fit_is_refused(self):
        with pytest.raises(ValueError, match="does not fit"):
            SG.material_tube((128, 128, 256), outer=70, wall=20)

    def test_a_wall_thicker_than_the_radius_is_refused(self):
        with pytest.raises(ValueError, match="wall"):
            SG.material_tube((128, 128, 256), outer=50, wall=60)


class TestLBracket:

    def test_both_legs_are_present_and_the_diagonal_is_not(self):
        m = SG.material_l_bracket((256, 256, 64), leg=200, thick=64,
                                  inner_radius=32, outer_radius=96)
        assert m[10, 150, 32]        # the leg along y
        assert m[150, 10, 32]        # the leg along z
        assert not m[200, 200, 32]   # the open quadrant

    def test_the_inner_corner_is_filleted_not_square(self):
        m = SG.material_l_bracket((256, 256, 64), leg=200, thick=64,
                                  inner_radius=32, outer_radius=96)
        # Just inside the square corner, on the fillet's diagonal, is air.
        assert not m[70, 70, 32]

    def test_it_is_a_constant_cross_section_along_x(self):
        m = SG.material_l_bracket((256, 256, 64))
        assert (m[:, :, 0] == m[:, :, -1]).all()


class TestTaper:

    def test_the_mean_thickness_is_the_average_of_the_ends(self):
        shape = (192, 256, 64)
        m = SG.material_taper(shape, t0=192, t1=80)
        assert m.mean() == pytest.approx(((192 + 80) / 2) / 192, rel=0.01)

    def test_the_thick_end_is_thicker_than_the_thin_end(self):
        m = SG.material_taper((192, 256, 64), t0=192, t1=80)
        assert m[:, 0, 32].sum() > m[:, -1, 32].sum()
        assert m[:, 0, 32].sum() == pytest.approx(192, abs=2)
        assert m[:, -1, 32].sum() == pytest.approx(80, abs=2)

    def test_it_is_centred_on_z(self):
        m = SG.material_taper((192, 256, 64))
        col = m[:, 128, 32]
        idx = np.flatnonzero(col)
        assert abs((idx.mean()) - (192 - 1) / 2) < 1.0


class TestTwoCoupons:

    def test_the_gap_is_air_and_the_plates_are_material(self):
        m = SG.material_two_coupons((64, 64, 256), gap=64)
        assert m[32, 32, 10] and m[32, 32, 245]
        assert not m[32, 32, 128]

    def test_the_material_fraction_is_one_minus_the_gap(self):
        m = SG.material_two_coupons((64, 64, 256), gap=64)
        assert m.mean() == pytest.approx((256 - 64) / 256, rel=0.01)

    def test_the_two_bodies_are_disconnected(self):
        """The point of the case: one request, two components."""
        from scipy import ndimage
        m = SG.material_two_coupons((32, 32, 256), gap=64)
        _, n = ndimage.label(m)
        assert n == 2


class TestHollowSphere:

    def test_the_shell_volume_matches_the_difference_of_spheres(self):
        shape = (256, 256, 256)
        m = SG.material_hollow_sphere(shape, outer=100, shell=30)
        expected = (4 / 3 * np.pi * (100 ** 3 - 70 ** 3)) / 256 ** 3
        assert m.mean() == pytest.approx(expected, rel=0.02)

    def test_the_interior_is_air_not_material(self):
        m = SG.material_hollow_sphere((256, 256, 256), outer=100, shell=30)
        assert not m[128, 128, 128]          # enclosed void
        assert m[128, 128, 128 - 85]         # the shell
        assert not m[128, 128, 10]           # outside


class TestGyroid:

    def test_it_is_near_half_dense_at_the_calibrated_thickness(self):
        """The check that the brief's own numbers failed.

        80-voxel struts at period 256 is 31 % of the period on each side of the
        surface: the walls meet and the result is a solid with pinholes. At
        period 512 the same strut gives the canonical near-half-dense gyroid.
        """
        m = SG.material_gyroid((256, 256, 256), period=512, thickness=80)
        assert 0.40 < m.mean() < 0.60

    def test_the_rejected_combination_really_is_solid(self):
        m = SG.material_gyroid((128, 128, 128), period=256, thickness=80)
        assert m.mean() > 0.95

    def test_thicker_struts_mean_more_material(self):
        f = [SG.material_gyroid((128, 128, 128), period=512, thickness=t).mean()
             for t in (20, 40, 80)]
        assert f[0] < f[1] < f[2]

    def test_every_face_of_the_canvas_cuts_material(self):
        """No outer surface anywhere — the first request with none."""
        m = SG.material_gyroid((128, 128, 128), period=512, thickness=80)
        for axis in range(3):
            assert np.take(m, 0, axis=axis).any()
            assert np.take(m, -1, axis=axis).any()


class TestLetters:

    def test_the_text_is_cut_through_the_whole_depth(self):
        m = SG.material_letters((8, 512, 512), text="Po", height=128)
        assert (m[0] == m[-1]).all()
        assert not m.all()                    # something was cut

    def test_the_glyphs_fit_inside_the_canvas(self):
        """At the requested size 'PoreGen' is wider than the canvas.

        It is WRAPPED rather than shrunk: the glyph height is the feature size
        the request is about, so it is what must be preserved.
        """
        m = SG.material_letters((4, 1024, 1024))
        ys, xs = np.where(~m[0])
        assert xs.min() >= 0 and xs.max() < 1024
        assert ys.min() >= 0 and ys.max() < 1024

    def test_it_cuts_air_and_not_the_whole_plate(self):
        m = SG.material_letters((4, 1024, 1024))
        assert 0.01 < (1 - m.mean()) < 0.35


class TestStressCaseList:

    def test_every_geometry_runs_at_both_step_counts(self):
        cases = build_cases("stress_geometry")
        assert len(cases) == len(STRESS_GEOMETRIES) * len(STRESS_DDIM)
        for name, *_ in STRESS_GEOMETRIES:
            got = sorted(c.ddim_steps for c in cases if c.name.startswith(name + "_"))
            assert got == sorted(STRESS_DDIM), name

    def test_every_axis_is_a_whole_number_of_tiles(self):
        """A canvas that is not tile-aligned cannot be generated at all."""
        for c in build_cases("stress_geometry"):
            assert all(v % TILE == 0 for v in c.volume_shape), c.name

    def test_every_case_declares_itself_exploratory(self):
        """A number from here must not be read as a gated result."""
        for c in build_cases("stress_geometry"):
            assert c.notes.get("exploratory") is True, c.name
            assert c.notes.get("off_gates_because"), c.name

    def test_every_feature_fits_the_canvas_it_was_given(self):
        """Checked without building the arrays: at full size they are 200 MB
        each, and the GPU is usually busy when this suite runs.

        Shrinking the canvas to make it cheap is what this test used to do, and
        it failed for the right reason — a quarter-size canvas with full-size
        features does not hold them. The feature sizes are the point, so the
        canvas is what has to be checked against them.
        """
        fits = {
            "tube": lambda sh: 2 * 200 < min(sh[0], sh[1]),      # axis is x
            # Two legs of 512 from a SHARED corner span 512, not 512 + 192:
            # the thickness is inside the leg length, not added to it.
            "lbracket": lambda sh: 512 <= sh[0] and 512 <= sh[1],
            "taper": lambda sh: 192 <= sh[0],
            "hollow_sphere": lambda sh: 2 * 224 < min(sh),
            "two_coupons": lambda sh: 128 < sh[2],
            "letters": lambda sh: 256 < min(sh[1], sh[2]),
            "gyroid": lambda sh: 512 <= max(sh) * 2,             # half a period
        }
        for name, shape, mat_fn, _field, _note in STRESS_GEOMETRIES:
            if mat_fn is None:
                continue
            assert name in fits, f"{name} has no fit rule"
            assert fits[name](shape), f"{name}: {shape} cannot hold its features"

    def test_the_painted_field_stays_inside_the_conditioning_clamp(self):
        """A request above POR_MAX would measure the clamp, not the model."""
        from poregen.diffusion.conditioning import POR_MAX

        f = SG.field_ramp_and_spots((3, 16, 16), 101)
        assert f.max() <= POR_MAX
        assert f.min() > 0
        # The ramp must actually ramp, and the spots must actually peak.
        assert f[:, -1, :].mean() > f[:, 0, :].mean() * 3
        assert f.max() > f[:, :, 0].mean() * 1.5

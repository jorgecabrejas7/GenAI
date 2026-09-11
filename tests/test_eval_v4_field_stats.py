"""eval v4 ``field_stats`` on fields whose spatial statistics are known in advance.

Nothing here loads a model or reads a real volume.  Each test builds a field
whose correlation length per axis is set by construction - Gaussian-smoothed
noise, whose autocovariance is ``exp(-r^2 / 4 sigma^2)`` and whose 1/e length is
therefore exactly ``2 sigma`` - so a failure names the estimator that is wrong
rather than the model.

The end-to-end tests go through a LABEL, because that is what the assessment
reads: a uint8 volume whose voxels are Bernoulli draws from a known porosity
field.  The window porosities that come back are checked against the exact
window means of that field, so the label-to-field step is measured separately
from the field-to-statistic step.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from poregen.eval_v4 import field_stats as FS
from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import MEASURE_ONLY
from poregen.eval_v4.cli import build_parser
from poregen.eval_v4.io import LABEL_PORE, TILE, read_results, save_case
from poregen.eval_v4.manifest import Manifest
from poregen.eval_v4.measure import MEASURERS, measure
from poregen.eval_v4.report import REPORTERS, report_one

COMMIT = "0" * 40

#: Voxels per cell of the coarse field a synthetic label is drawn from.  Eight
#: is far below every correlation length tested here, so the blockiness it
#: introduces is not what any assertion is about.
CELL = 8


# ---------------------------------------------------------------------------
# Synthetic fields with a known correlation length
# ---------------------------------------------------------------------------

def gaussian_field(shape, sigmas, seed: int) -> np.ndarray:
    """White noise smoothed per axis.  1/e correlation length is ``2 * sigma``.

    Smoothed on a domain twice the size and cropped, so the periodic wrap of the
    filter does not close the field back on itself inside the region measured.
    """
    rng = np.random.default_rng(seed)
    big = tuple(2 * s for s in shape)
    a = ndimage.gaussian_filter(rng.standard_normal(big), sigma=sigmas, mode="wrap")
    return a[tuple(slice(0, s) for s in shape)]


def label_from_field(sigmas_vox, shape, seed, mean_phi=0.03, cv=0.35):
    """A uint8 label whose local porosity follows a field of known scale.

    Returns the label and the exact window-mean porosity the label was drawn
    from, so the measured field can be checked against the truth and not only
    against itself.
    """
    cells = tuple(s // CELL for s in shape)
    a = gaussian_field(cells, [s / CELL for s in sigmas_vox], seed)
    p = mean_phi * (1.0 + cv * a / a.std())
    voxel_p = np.repeat(np.repeat(np.repeat(p.astype(np.float32), CELL, 0),
                                  CELL, 1), CELL, 2)
    rng = np.random.default_rng(seed + 9000)
    label = (rng.random(voxel_p.shape, dtype=np.float32) < voxel_p).astype(np.uint8)
    blocks = M.block_sum(voxel_p, FS.STRIDE) / FS.STRIDE ** 3
    k = FS.WINDOW // FS.STRIDE
    truth = FS._box_sum(blocks, k) / k ** 3
    return label, truth


# ---------------------------------------------------------------------------
# The 1/e crossing
# ---------------------------------------------------------------------------

class TestCorrelationLength:
    def test_the_crossing_is_interpolated_between_the_bracketing_lags(self):
        # 1/e = 0.3679 sits 60 % of the way from 0.50 down to 0.30, so the
        # crossing of a curve sampled every 32 voxels is at 64 + 0.6 * 32.
        lags = np.array([0.0, 32.0, 64.0, 96.0])
        corr = np.array([1.0, 0.80, 0.50, 0.30])
        got = M.correlation_length_1_over_e(lags, corr)
        assert got == pytest.approx(64.0 + 32.0 * (0.5 - np.exp(-1.0)) / 0.2, rel=1e-9)
        assert 64.0 < got < 96.0

    def test_a_curve_that_never_reaches_1_over_e_has_no_length(self):
        lags = np.array([0.0, 32.0, 64.0])
        assert M.correlation_length_1_over_e(lags, np.array([1.0, 0.9, 0.8])) is None

    def test_the_search_stops_where_the_curve_runs_out_of_pairs(self):
        # The NaN tail is "not measured", not "decorrelated": a crossing may not
        # be read off the other side of a gap.
        lags = np.array([0.0, 32.0, 64.0, 96.0])
        corr = np.array([1.0, 0.9, np.nan, 0.1])
        assert M.correlation_length_1_over_e(lags, corr) is None

    def test_lag_correlation_of_a_field_with_no_variance_is_not_a_number(self):
        flat = np.full((8, 8, 8), 0.03)
        curve = M.lag_correlation([flat], 0, 4)
        assert not np.isfinite(curve["r"][1])

    def test_lags_with_too_few_pairs_are_dropped(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((6, 2, 2))
        curve = M.lag_correlation([f], 0, 5, min_pairs=16)
        # lag 5 leaves one pair per line, four in all - far under the floor.
        assert curve["n_pairs"][5] == 4
        assert not np.isfinite(curve["r"][5])


# ---------------------------------------------------------------------------
# The estimator, on fields whose answer is set by construction
# ---------------------------------------------------------------------------

class TestAnisotropicRecovery:
    #: sigma per axis in grid steps; the 1/e length of the smoothed field is
    #: exactly twice each, so on a 32-voxel grid the answers are 96, 192 and 384
    #: voxels - an order of anisotropy like the real laminate's.
    SIGMAS = (1.5, 3.0, 6.0)
    EXPECTED_VOX = {"z": 96.0, "y": 192.0, "x": 384.0}

    @pytest.mark.parametrize("seed", (1, 2, 3))
    def test_each_axis_is_recovered_at_its_own_correlation_length(self, seed):
        f = gaussian_field((96, 96, 96), self.SIGMAS, seed)
        per_axis = FS.axis_correlations([f], stride=FS.STRIDE)
        for name, want in self.EXPECTED_VOX.items():
            got = per_axis[name]["corr_length_vox"]
            assert got is not None, f"{name} found no crossing inside the field"
            assert got == pytest.approx(want, rel=0.12), (
                f"{name}: {got:.1f} voxels, built to be {want:.1f}"
            )

    @pytest.mark.parametrize("seed", (1, 2, 3))
    def test_the_three_axes_are_told_apart_and_not_averaged(self, seed):
        f = gaussian_field((96, 96, 96), self.SIGMAS, seed)
        per_axis = FS.axis_correlations([f], stride=FS.STRIDE)
        lz, ly, lx = (per_axis[a]["corr_length_vox"] for a in ("z", "y", "x"))
        assert lz < ly < lx
        # Built at 1 : 2 : 4. An isotropic summary would put all three together.
        assert ly / lz == pytest.approx(2.0, rel=0.2)
        assert lx / lz == pytest.approx(4.0, rel=0.2)
        aniso = FS.anisotropy(per_axis)
        assert aniso["y_over_z"] == pytest.approx(ly / lz, rel=1e-9)
        assert aniso["x_over_z"] == pytest.approx(lx / lz, rel=1e-9)

    def test_an_isotropic_field_comes_back_isotropic(self):
        f = gaussian_field((96, 96, 96), (3.0, 3.0, 3.0), 4)
        per_axis = FS.axis_correlations([f], stride=FS.STRIDE)
        got = [per_axis[a]["corr_length_vox"] for a in ("z", "y", "x")]
        for v in got:
            assert v == pytest.approx(192.0, rel=0.12)

    def test_a_field_that_does_not_decorrelate_reports_no_length_not_the_crop(self):
        # A field that only trends: shifting it along any axis adds a constant,
        # so its correlation is exactly 1 at every lag and it never falls to
        # 1/e. Quoting the last lag instead would publish the size of the crop
        # as a property of the material.
        iz, iy, ix = np.indices((16, 16, 16))
        f = (iz + 2.0 * iy + 4.0 * ix).astype(float)
        per_axis = FS.axis_correlations([f], stride=FS.STRIDE)
        for name in ("z", "y", "x"):
            assert per_axis[name]["corr_length_vox"] is None
            assert per_axis[name]["reach_vox"] > 0
        assert FS.anisotropy(per_axis) == {"y_over_z": None, "x_over_z": None}

    @pytest.mark.parametrize("sigmas", ((1.5, 3.0, 6.0), (8.0, 8.0, 8.0),
                                        (0.5, 20.0, 40.0)))
    @pytest.mark.parametrize("n", (5, 17))
    def test_a_reported_length_never_runs_past_the_reach(self, sigmas, n):
        """The guard that keeps a crop size out of the answer.

        A short crop of a long-correlated field can still cross 1/e by chance -
        few pairs, few independent blobs - so the crossing alone is not proof
        that the length was measurable. What must always hold is that a reported
        length lies inside the lags the crop actually reaches, and that the reach
        is reported beside it; :func:`field_stats.corr_length_gap` then refuses
        the comparisons where it does not.
        """
        f = gaussian_field((n, n, n), sigmas, 6)
        per_axis = FS.axis_correlations([f], stride=FS.STRIDE)
        for name in ("z", "y", "x"):
            length = per_axis[name]["corr_length_vox"]
            reach = per_axis[name]["reach_vox"]
            assert reach <= (n - 1) * FS.STRIDE
            assert length is None or length <= reach


class TestOneEstimatorInTheSuite:
    """The surface floor reads its correlation length through the same pair.

    ``real_floor`` used to carry its own 2-D autocovariance with a whole-lag
    crossing. It now calls :func:`metrics.lag_correlation` and
    :func:`metrics.correlation_length_1_over_e`, so a roughness length and a
    porosity-field length are the same kind of number and only one estimator has
    to be right.
    """

    def test_the_surface_floor_recovers_a_known_lateral_scale(self):
        from poregen.eval_v4.real_floor import _correlation_length

        h = gaussian_field((256, 256), (4.0, 12.0), 2)
        got = _correlation_length(h, np.ones(h.shape, bool), max_lag=80)
        assert set(got) == {"x", "y", "mean"}
        assert got["y"] == pytest.approx(8.0, rel=0.15)    # axis 0, sigma 4
        assert got["x"] == pytest.approx(24.0, rel=0.15)   # axis 1, sigma 12
        assert got["mean"] == pytest.approx((got["x"] + got["y"]) / 2, rel=1e-9)

    def test_invalid_columns_are_excluded_rather_than_counted_as_zero(self):
        from poregen.eval_v4.real_floor import _correlation_length

        h = gaussian_field((256, 256), (4.0, 12.0), 2)
        valid = np.ones(h.shape, bool)
        valid[:, 200:] = False             # a drilled hole through the map
        got = _correlation_length(h, valid, max_lag=80)
        assert got["x"] == pytest.approx(24.0, rel=0.2)


# ---------------------------------------------------------------------------
# Label -> delivered field
# ---------------------------------------------------------------------------

class TestDeliveredField:
    def test_window_porosity_is_pore_over_material_exactly(self):
        label = np.zeros((128, 128, 128), np.uint8)
        label[:32, :32, :32] = LABEL_PORE
        field = FS.delivered_field(label, np.ones((128, 128, 128), bool))
        assert field.shape == (3, 3, 3)
        assert field[0, 0, 0] == pytest.approx(32 ** 3 / 64 ** 3)
        assert field[2, 2, 2] == pytest.approx(0.0)

    def test_the_denominator_is_material_and_not_the_window(self):
        label = np.zeros((64, 64, 64), np.uint8)
        material = np.zeros((64, 64, 64), bool)
        material[:, :, :48] = True           # three quarters of the window
        label[:, :, :12] = LABEL_PORE        # a quarter of that material
        field = FS.delivered_field(label, material)
        assert field.shape == (1, 1, 1)
        assert field[0, 0, 0] == pytest.approx(0.25)

    def test_a_window_that_is_mostly_air_is_dropped_not_measured(self):
        material = np.zeros((128, 64, 64), bool)
        material[:64] = True                 # the second window holds no material
        label = np.zeros((128, 64, 64), np.uint8)
        label[:64, :, :16] = LABEL_PORE
        field = FS.delivered_field(label, material)
        assert field.shape == (3, 1, 1)
        assert field[0, 0, 0] == pytest.approx(0.25)
        assert np.isnan(field[2, 0, 0])

    def test_a_window_must_be_a_whole_number_of_strides(self):
        with pytest.raises(ValueError, match="whole number of strides"):
            FS.delivered_field(np.zeros((64, 64, 64), np.uint8),
                               np.ones((64, 64, 64), bool), window=48, stride=32)

    def test_pores_outside_the_requested_material_do_not_count(self):
        # Every voxel is a pore, but only three quarters of the window was asked
        # for as specimen. Both sides of the ratio are taken inside that quarter,
        # so the answer is 1.0 - the pores in the air do not push it past it.
        material = np.zeros((64, 64, 64), bool)
        material[:, :, :48] = True
        label = np.full((64, 64, 64), LABEL_PORE, np.uint8)
        field = FS.delivered_field(label, material)
        assert field[0, 0, 0] == pytest.approx(1.0)


class TestLabelToStatistics:
    """The whole chain on a label drawn from a field whose windows are known."""

    SHAPE = (256, 256, 256)
    SIGMAS_VOX = (16.0, 24.0, 32.0)

    @pytest.mark.parametrize("seed", (101, 202))
    def test_the_measured_field_is_the_field_the_label_was_drawn_from(self, seed):
        label, truth = label_from_field(self.SIGMAS_VOX, self.SHAPE, seed)
        got = FS.delivered_field(label, np.ones(self.SHAPE, bool))
        assert got.shape == truth.shape == (7, 7, 7)
        # Binomial noise over a 64-cubed window at phi 0.03 has sd 3.3e-4; the
        # largest of 343 windows should stay inside five of those.
        assert np.abs(got - truth).max() < 0.002
        assert got.mean() == pytest.approx(truth.mean(), abs=2e-4)

    @pytest.mark.parametrize("seed", (101, 202))
    def test_the_statistics_survive_the_trip_through_the_label(self, seed):
        label, truth = label_from_field(self.SIGMAS_VOX, self.SHAPE, seed)
        got = FS.delivered_field(label, np.ones(self.SHAPE, bool))
        a = FS.axis_correlations([got])
        b = FS.axis_correlations([truth])
        for name in ("z", "y", "x"):
            la, lb = a[name]["corr_length_vox"], b[name]["corr_length_vox"]
            assert (la is None) == (lb is None)
            if la is not None:
                assert la == pytest.approx(lb, rel=0.05)
        assert FS.marginal_distance(got.ravel(), truth.ravel())["w1"] < 3e-4

    def test_a_192_cubed_volume_cannot_resolve_the_in_plane_length(self):
        """The limit this assessment has to report rather than paper over.

        Five windows an axis is 128 voxels of lag. The real in-plane
        correlation length is several hundred, so a 192-cubed case can only ever
        say 'longer than this crop' in y and x - and it must say that, not a
        number that is really the size of the crop.
        """
        label, _ = label_from_field((16.0, 160.0, 160.0), (192, 192, 192), 7)
        field = FS.delivered_field(label, np.ones((192, 192, 192), bool))
        assert field.shape == (5, 5, 5)
        per_axis = FS.axis_correlations([field])
        assert per_axis["y"]["corr_length_vox"] is None
        assert per_axis["x"]["corr_length_vox"] is None
        assert per_axis["y"]["reach_vox"] <= 128


# ---------------------------------------------------------------------------
# Comparing two fields
# ---------------------------------------------------------------------------

class TestComparison:
    def test_the_mean_normalised_distance_ignores_the_global_level(self):
        rng = np.random.default_rng(3)
        a = 0.03 * (1.0 + 0.3 * rng.standard_normal(4000))
        b = 2.0 * a                      # same shape, twice the porosity
        d = FS.marginal_distance(a, b)
        assert d["w1"] == pytest.approx(0.03, abs=2e-3)
        assert d["w1_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_a_different_spread_is_seen_even_at_the_same_mean(self):
        rng = np.random.default_rng(4)
        a = 0.03 * (1.0 + 0.1 * rng.standard_normal(4000))
        b = 0.03 * (1.0 + 0.6 * rng.standard_normal(4000))
        d = FS.marginal_distance(a, b)
        assert d["w1_ratio"] > 0.1
        assert d["ks"] > 0.2

    def test_two_lengths_found_over_different_reaches_are_not_compared(self):
        gen = {"z": {"corr_length_vox": 80.0, "reach_vox": 128},
               "y": {"corr_length_vox": None, "reach_vox": 128},
               "x": {"corr_length_vox": None, "reach_vox": 128}}
        real = {"z": {"corr_length_vox": 76.0, "reach_vox": 960},
                "y": {"corr_length_vox": 420.0, "reach_vox": 960},
                "x": {"corr_length_vox": 900.0, "reach_vox": 960}}
        gap = FS.corr_length_gap(gen, real)
        assert gap["z"]["difference_vox"] == pytest.approx(4.0)
        assert gap["z"]["common_reach_vox"] == 128
        # The real length is outside what the generated crop could have found,
        # so there is no difference to report - only the two lengths.
        assert gap["y"]["difference_vox"] is None
        assert gap["y"]["real_vox"] == 420.0
        assert gap["x"]["ratio"] is None


# ---------------------------------------------------------------------------
# Registration: an assessment with no reporter is dropped from findings.md
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_field_stats_has_a_measurer_and_a_reporter(self):
        assert "field_stats" in MEASURERS
        assert "field_stats" in REPORTERS

    def test_it_is_measurable_from_the_cli_but_not_generatable(self):
        assert "field_stats" in MEASURE_ONLY
        parser = build_parser()
        args = parser.parse_args(["measure", "field_stats", "--root", "x"])
        assert args.assessment == "field_stats"
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "field_stats", "--model", "m",
                               "--ckpt", "1", "--out", "o"])


# ---------------------------------------------------------------------------
# The measure and report stages over a campaign on disk
# ---------------------------------------------------------------------------

SHAPE = (192, 192, 192)


def _write_generated(root, name, seed, sigmas):
    label, _ = label_from_field(sigmas, SHAPE, seed)
    grid = tuple(s // TILE for s in SHAPE)
    # A request deliberately UNLIKE the label beside it: a hard ramp through the
    # thickness, three times the spread the label actually holds. A measurement
    # that read the request instead of the label would show the ramp.
    requested = np.repeat(
        np.array([0.01, 0.03, 0.05], np.float32).reshape(-1, 1, 1),
        grid[1], 1).repeat(grid[2], 2)
    man = Manifest(
        assessment="porosity_local", case=name, volume_shape=SHAPE, git_commit=COMMIT,
        model_run="runs/ldm/test", checkpoint_step=1000, weights="ema", ddim_steps=50,
        chunk_tiles=(3, 3, 3), window_stride=32, decode="overlapped", decode_overlap=32,
        s_por=1.0, s_nb=1.0, objective="v", cfg_rescale=0.0, seed=seed,
        requested_global_phi=0.03, requested_field="requested_field.npy",
        requested_material="full", notes={"field": "coherent", "layup": "A"},
    )
    save_case(root / "porosity_local" / "volumes" / name, man,
              np.zeros(SHAPE, np.uint8), label, requested_field=requested)


def _write_real(root, name, seed, sigmas):
    label, _ = label_from_field(sigmas, SHAPE, seed)
    man = Manifest(
        assessment="real_floor", case=name, volume_shape=SHAPE, git_commit=COMMIT,
        sampler="real", chunk_tiles=(3, 3, 3), window_stride=32,
        decode="overlapped", decode_overlap=32, requested_material="full",
        notes={"shape_tag": "small", "volume_id": name, "split": "test"},
    )
    save_case(root / "real_floor" / "volumes" / name, man,
              np.zeros(SHAPE, np.uint8), label)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    for i, seed in enumerate((11, 22)):
        _write_generated(root, f"coherent_ddim50_seed{seed}", seed, (16.0, 24.0, 32.0))
    for i, seed in enumerate((33, 44)):
        _write_real(root, f"vol{i}__small", seed, (16.0, 40.0, 48.0))
    return root


class TestMeasureAndReport:
    def test_the_measure_step_reads_both_sides_at_one_window(self, campaign):
        res = measure(campaign, "field_stats")
        assert res["geometry"]["window_vox"] == FS.WINDOW
        assert res["geometry"]["stride_vox"] == FS.STRIDE
        grids = {g["field_grids"][0][0]
                 for name, g in res["groups"].items() if not name.startswith("requested")}
        assert grids == {5}, "real and generated must be measured on one window grid"
        assert res["n_cases_measured"] == 4
        assert res["n_cases_expected"] == 4

    def test_a_measured_volume_is_not_held_to_the_end_of_the_pass(self, campaign):
        from poregen.eval_v4.io import Case

        case = Case.load(campaign / "real_floor" / "volumes" / "vol0__small")
        assert case.label.shape == SHAPE
        case.release()
        assert "label" not in case.__dict__
        assert case.label.shape == SHAPE      # and it reads again on demand

    def test_it_reports_the_generated_the_requested_and_the_real_apart(self, campaign):
        res = read_results(campaign, "field_stats")
        assert set(res["groups"]) == {
            "generated/porosity_local", "requested/porosity_local", "real/small"}
        # The delivered field is read from the LABEL, so its spread is the
        # label's and not the ramp that was requested beside it.
        delivered = res["groups"]["generated/porosity_local"]["marginal"]
        requested = res["groups"]["requested/porosity_local"]["marginal"]
        assert requested["sd"] == pytest.approx(0.0163, abs=1e-3)
        assert delivered["sd"] < 0.5 * requested["sd"]
        assert delivered["mean"] == pytest.approx(0.03, abs=0.005)

    def test_every_generated_group_is_compared_against_the_real_one(self, campaign):
        res = read_results(campaign, "field_stats")
        assert "generated/porosity_local vs real/small" in res["comparisons"]
        assert "requested/porosity_local vs real/small" in res["comparisons"]
        floor = res["comparisons"]["real/small vs real/small (real floor)"]
        assert floor["marginal"]["w1"] is not None

    def test_findings_state_the_window_and_never_invent_a_length(self, campaign):
        path = report_one(campaign, "field_stats")
        text = path.read_text()
        assert "64-voxel windows every 32 voxels" in text
        assert "L z (vox)" in text
        # The in-plane axes cannot decorrelate inside a 192-cubed crop, so the
        # table has to say 'longer than the reach' rather than print a number.
        assert "> 128" in text

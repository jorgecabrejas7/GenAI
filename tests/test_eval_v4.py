"""eval v4: the manifest contract, and every metric on data whose answer is known.

Nothing here loads a model.  Each test builds an array whose measurement can be
worked out by hand - a label with an exactly-set porosity per tile, a volume
with a step at a chosen plane, an angle sequence with chosen errors - so a
failure names the metric that is wrong rather than the model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import (
    build_cases,
    field_checkerboard,
    field_two_halves,
    material_notch_and_hole,
)
from poregen.eval_v4.generate import (
    latent_material_map,
    resolve_latent_store,
    theta_for_canvas,
)
from poregen.eval_v4.io import (
    LABEL_AIR,
    LABEL_MATERIAL,
    LABEL_PORE,
    TILE,
    Case,
    load_u8,
    save_case,
)
from poregen.eval_v4.manifest import Manifest, ManifestError, requires

COMMIT = "0" * 40


def make_manifest(**kw) -> Manifest:
    """A valid generated manifest; ``kw`` overrides any field."""
    base = dict(
        assessment="unit", case="c0", volume_shape=(192, 192, 192), git_commit=COMMIT,
        sampler="hybrid_chunked", model_run="runs/ldm/x", checkpoint_step=1000,
        weights="ema", ddim_steps=200, chunk_tiles=(3, 3, 3), window_stride=32,
        decode="overlapped", decode_overlap=32, s_por=1.0, s_nb=1.0, seed=101,
        objective="v", cfg_rescale=0.0,
        requested_global_phi=0.03, requested_material="full",
    )
    base.update(kw)
    return Manifest(**base)


def label_with_tile_porosity(phi: np.ndarray) -> np.ndarray:
    """A label whose every 64-cubed tile holds EXACTLY ``round(phi * 64^3)`` pores."""
    gz, gy, gx = phi.shape
    label = np.zeros((gz * TILE, gy * TILE, gx * TILE), np.uint8)
    for iz in range(gz):
        for iy in range(gy):
            for ix in range(gx):
                n = int(round(float(phi[iz, iy, ix]) * TILE ** 3))
                block = np.zeros(TILE ** 3, np.uint8)
                block[:n] = LABEL_PORE
                label[iz * TILE:(iz + 1) * TILE,
                      iy * TILE:(iy + 1) * TILE,
                      ix * TILE:(ix + 1) * TILE] = block.reshape(TILE, TILE, TILE)
    return label


# ---------------------------------------------------------------------------
# The manifest contract
# ---------------------------------------------------------------------------

class TestManifest:
    def test_a_valid_manifest_round_trips_through_json(self, tmp_path):
        m = make_manifest(chunk_tiles=[3, 3, 3], volume_shape=[192, 192, 192])
        m.write(tmp_path)
        back = Manifest.read(tmp_path)
        assert back == m
        assert back.chunk_tiles == (3, 3, 3)
        assert back.volume_shape == (192, 192, 192)

    def test_a_generated_volume_must_declare_its_sampler_settings(self):
        with pytest.raises(ManifestError, match="ddim_steps"):
            make_manifest(ddim_steps=None)

    def test_a_generated_volume_must_declare_a_porosity_request(self):
        with pytest.raises(ManifestError, match="porosity request"):
            make_manifest(requested_global_phi=None, requested_field=None)

    def test_a_field_request_alone_is_enough(self):
        m = make_manifest(requested_global_phi=None, requested_field="requested_field.npy")
        assert m.requested_field == "requested_field.npy"

    def test_decode_and_overlap_cannot_contradict_each_other(self):
        with pytest.raises(ManifestError, match="contradicts"):
            make_manifest(decode="tiled", decode_overlap=32)
        with pytest.raises(ManifestError, match="contradicts"):
            make_manifest(decode="overlapped", decode_overlap=0)
        assert make_manifest(decode="tiled", decode_overlap=0).decode == "tiled"

    def test_the_prediction_objective_is_recorded_and_checked(self):
        """Two checkpoints trained on different objectives are different models,
        so the objective is required and its value is constrained."""
        assert make_manifest(objective="eps").objective == "eps"
        with pytest.raises(ManifestError, match="objective must be one of"):
            make_manifest(objective="x0")
        with pytest.raises(ManifestError, match="objective"):
            make_manifest(objective=None)

    def test_cfg_rescale_is_recorded_and_must_not_be_negative(self):
        assert make_manifest(cfg_rescale=0.7).cfg_rescale == 0.7
        with pytest.raises(ManifestError, match="cfg_rescale"):
            make_manifest(cfg_rescale=-0.1)
        with pytest.raises(ManifestError, match="cfg_rescale"):
            make_manifest(cfg_rescale=None)

    def test_an_unknown_field_is_refused_rather_than_ignored(self):
        d = make_manifest().to_dict()
        d["chunk_size"] = 3
        with pytest.raises(ManifestError, match="unknown fields"):
            Manifest.from_dict(d)

    def test_a_real_crop_needs_no_model_and_no_request(self):
        m = Manifest(assessment="real_floor", case="v0", volume_shape=(192, 192, 192),
                     git_commit=COMMIT, sampler="real")
        assert m.is_real
        assert m.model_run is None

    def test_a_region_must_lie_inside_the_volume(self):
        with pytest.raises(ManifestError, match="leaves the volume"):
            make_manifest(volume_shape=(192, 192, 192),
                          region_offset=(64, 0, 0), region_shape=(192, 192, 192))

    def test_a_missing_manifest_is_an_error_not_an_empty_one(self, tmp_path):
        with pytest.raises(ManifestError, match="no manifest"):
            Manifest.read(tmp_path)

    def test_a_metric_refuses_a_volume_whose_shape_is_not_the_declared_one(self):
        m = make_manifest(volume_shape=(64, 64, 64))
        with pytest.raises(ManifestError, match="declares volume_shape"):
            M.phase_fractions(np.zeros((128, 64, 64), np.uint8),
                              np.ones((128, 64, 64), bool), manifest=m)

    def test_a_metric_refuses_a_manifest_that_does_not_carry_its_fields(self):
        real = Manifest(assessment="real_floor", case="v0", volume_shape=(64, 64, 64),
                        git_commit=COMMIT, sampler="real")
        label = np.zeros((64, 64, 64), np.uint8)
        with pytest.raises(ManifestError, match="requested_global_phi"):
            M.porosity_error(label, np.ones_like(label, bool), manifest=real)

    def test_the_declaration_is_readable_from_the_metric(self):
        assert M.porosity_error.requires == ("requested_global_phi",)
        assert "chunk_tiles" in M.seam_metrics.requires

    def test_a_keyword_array_on_another_grid_is_not_shape_checked(self):
        """The decorator checks positional arrays only - that is the convention
        auxiliary arrays like a tile-grid field depend on."""

        @requires()
        def metric(vol, *, manifest, aux=None):
            return aux.shape

        m = make_manifest(volume_shape=(64, 64, 64))
        assert metric(np.zeros((64, 64, 64)), manifest=m, aux=np.zeros((1, 1, 1))) == (1, 1, 1)


# ---------------------------------------------------------------------------
# Case IO
# ---------------------------------------------------------------------------

class TestCaseIO:
    def test_a_case_round_trips_and_derives_its_requests(self, tmp_path):
        shape = (64, 128, 128)
        field = np.array([[[0.01, 0.05], [0.05, 0.01]]], np.float32)
        material = np.ones(shape, bool)
        material[:, :32, :] = False
        m = make_manifest(volume_shape=shape, requested_field="requested_field.npy",
                          requested_material="requested_material.npy")
        save_case(
            tmp_path / "c", m,
            np.full(shape, 200, np.uint8), np.zeros(shape, np.uint8),
            requested_field=field, requested_material=latent_material_map(material),
        )
        case = Case.load(tmp_path / "c")
        assert case.xct.shape == shape
        np.testing.assert_allclose(case.requested_phi_per_tile(), field)
        # Half a latent cell of specimen is the cut-off, and the notch edge is
        # 64-voxel aligned here, so the voxel request comes back exactly.
        np.testing.assert_array_equal(case.material_voxels(), material)

    def test_a_float_volume_is_refused_rather_than_rescaled(self, tmp_path):
        import tifffile

        tifffile.imwrite(str(tmp_path / "v.tif"), np.zeros((8, 8, 8), np.float32))
        with pytest.raises(TypeError, match="not uint8"):
            load_u8(tmp_path / "v.tif")


# ---------------------------------------------------------------------------
# Local obedience
# ---------------------------------------------------------------------------

class TestLocalObedience:
    def test_a_perfectly_obedient_volume_has_slope_one_and_r2_one(self):
        req = field_two_halves((3, 3, 3), 0)
        label = label_with_tile_porosity(req)
        m = make_manifest()
        out = M.local_obedience(label, np.ones(label.shape, bool),
                                manifest=m, requested_tiles=req)
        assert out["within_volume_slope"] == pytest.approx(1.0, abs=2e-3)
        assert out["within_volume_r2"] == pytest.approx(1.0, abs=1e-4)
        assert out["per_cell"]["abs_error_mean"] < 1e-4

    def test_a_volume_that_ignores_the_field_has_slope_zero(self):
        req = field_checkerboard((3, 3, 3), 0)
        flat = np.full((3, 3, 3), float(req.mean()), np.float32)
        label = label_with_tile_porosity(flat)
        out = M.local_obedience(label, np.ones(label.shape, bool),
                                manifest=make_manifest(), requested_tiles=req)
        assert abs(out["within_volume_slope"]) < 1e-6
        # A perfectly flat delivered field has no variance to explain, so R2 is
        # undefined and is reported as such. The slope carries the answer.
        assert out["within_volume_r2"] is None
        assert out["delivered_cell_sd"] == pytest.approx(0.0, abs=1e-9)

    def test_a_half_obedient_volume_reports_the_gain_it_delivered(self):
        req = field_two_halves((3, 3, 3), 0)
        delivered = req.mean() + 0.5 * (req - req.mean())
        label = label_with_tile_porosity(delivered)
        out = M.local_obedience(label, np.ones(label.shape, bool),
                                manifest=make_manifest(), requested_tiles=req)
        assert out["within_volume_slope"] == pytest.approx(0.5, abs=5e-3)

    def test_pooling_two_volumes_inflates_r2_above_the_within_volume_answer(self):
        """The reason the pooled fit is never reported as local obedience.

        Two volumes that each ignore their field entirely, at two different
        global levels, pool to a near-perfect R-squared.  The swing asked for
        WITHIN a volume is small beside the range asked for ACROSS volumes,
        which is the situation the dose-response sweep actually creates.
        """
        outs, req_cells, got_cells = [], [], []
        for base in (0.01, 0.10):
            req = field_checkerboard((3, 3, 3), 0, lo=0.028, hi=0.032) + (base - 0.03)
            flat = np.full((3, 3, 3), float(req.mean()), np.float32)
            label = label_with_tile_porosity(flat)
            o = M.local_obedience(label, np.ones(label.shape, bool),
                                  manifest=make_manifest(), requested_tiles=req)
            outs.append(o)
            req_cells.append(np.asarray(o["cells_requested"]))
            got_cells.append(np.asarray(o["cells_delivered"]))
        pooled = M.pooled_dose_fit(req_cells, got_cells)
        assert all(abs(o["within_volume_slope"]) < 1e-6 for o in outs)
        assert pooled["pooled_r2_not_obedience"] > 0.8

    def test_a_real_volume_reports_only_its_cell_spread(self):
        rng = np.random.default_rng(0)
        phi = 0.02 + 0.004 * rng.standard_normal((3, 3, 3))
        label = label_with_tile_porosity(np.clip(phi, 0, 1))
        real = Manifest(assessment="real_floor", case="v", volume_shape=label.shape,
                        git_commit=COMMIT, sampler="real")
        out = M.local_obedience(label, np.ones(label.shape, bool),
                                manifest=real, requested_tiles=None)
        assert out["requested"] is False
        assert out["delivered_cell_sd"] == pytest.approx(float(np.std(phi, ddof=1)), rel=0.02)
        assert "within_volume_slope" not in out


# ---------------------------------------------------------------------------
# Seam plane selection
# ---------------------------------------------------------------------------

class TestSeamPlanes:
    def test_the_window_and_chunk_planes_are_the_ones_the_geometry_implies(self):
        m = make_manifest(volume_shape=(192, 192, 384), chunk_tiles=(3, 3, 3))
        out = M.seam_metrics(np.zeros((192, 192, 384), np.uint8), manifest=m)
        assert M.chunk_period(m) == (192, 192, 192)
        # Window planes at every multiple of 64 inside each axis.
        assert out["seam_xct_z_planes"] == 2      # 64, 128
        assert out["seam_xct_x_planes"] == 5      # 64 .. 320
        # One chunk per axis in z and y, two in x: only x has a chunk plane.
        assert out["seam_chunk_xct_z_planes"] == 0
        assert out["seam_chunk_xct_x_planes"] == 1
        assert out["pore_logit_available"] is False

    @staticmethod
    def _textured(shape, seed=0):
        """A volume with ordinary interior texture, so the seam ratio has a
        non-zero baseline to be a ratio against."""
        rng = np.random.default_rng(seed)
        return rng.integers(98, 103, size=shape, dtype=np.uint8)

    def test_a_step_at_the_chunk_plane_is_found_by_the_chunk_period(self):
        """Every chunk plane is also a window plane, which is why both periods
        are reported: the window metric averages one bad plane over five and
        dilutes it, the chunk metric looks only where two canvases met."""
        shape = (64, 64, 384)
        clean = self._textured(shape)
        vol = clean.copy()
        vol[:, :, 192:] += 60                      # one step, at the chunk plane
        m = make_manifest(volume_shape=shape, chunk_tiles=(3, 3, 3))
        out = M.seam_metrics(vol, manifest=m)
        base = M.seam_metrics(clean, manifest=m)
        assert out["seam_chunk_xct_x_ratio"] > 20
        assert out["seam_chunk_xct_x_ratio"] > 3 * out["seam_xct_x_ratio"]
        assert base["seam_xct_x_ratio"] == pytest.approx(1.0, abs=0.15)
        assert base["seam_chunk_xct_x_ratio"] == pytest.approx(1.0, abs=0.2)

    def test_a_step_at_every_window_plane_shows_at_the_window_period(self):
        shape = (64, 64, 384)
        vol = self._textured(shape)
        vol[:, :, ::TILE] += 40
        m = make_manifest(volume_shape=shape, chunk_tiles=(3, 3, 3))
        out = M.seam_metrics(vol, manifest=m)
        assert out["seam_xct_x_ratio"] > 10

    def test_the_pore_logit_seam_is_measured_when_the_field_is_there(self):
        shape = (64, 64, 128)
        m = make_manifest(volume_shape=shape, chunk_tiles=(3, 3, 3))
        out = M.seam_metrics(np.zeros(shape, np.uint8), manifest=m,
                             pore_logit=np.zeros(shape, np.float32))
        assert out["pore_logit_available"] is True
        assert "seam_pore_ratio" in out

    def test_the_chunk_slab_covers_the_chunk_planes_only(self):
        slab = M.chunk_plane_slab((64, 64, 384), (192, 192, 192), half_width=8)
        assert slab[:, :, 184:200].all()
        assert not slab[:, :, :100].any()
        assert not M.chunk_plane_slab((192, 192, 192), (192, 192, 192)).any()


# ---------------------------------------------------------------------------
# Layup reader scoring
# ---------------------------------------------------------------------------

def _stub_readers(angles: dict, window: int = 8):
    """The real T-I scoring maths with synthetic reader output substituted in.

    Only ``measure_volume`` is replaced: ``ply_edges``, ``requested_sequence``
    and ``score_recovery`` are the repo's own, so the test scores synthetic
    angles through the code that will score real ones.
    """
    lr = pytest.importorskip(
        "layup_roundtrip",
        reason="scripts/analysis is not importable in this environment",
    )
    return types.SimpleNamespace(
        ti=types.SimpleNamespace(WINDOW=window),
        ply_edges=lr.ply_edges,
        requested_sequence=lr.requested_sequence,
        score_recovery=lr.score_recovery,
        measure_volume=lambda xct, mask, edges, pitch: {
            name: {"angles": np.asarray(a, float),
                   "weights": np.ones(len(a), float)}
            for name, a in angles.items()
        },
    )


@pytest.fixture(autouse=True, scope="module")
def _analysis_on_path():
    from poregen.eval_v4.io import repo_root

    import sys

    analysis = repo_root() / "scripts" / "analysis"
    if analysis.exists() and str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))


class TestLayupScoring:
    LAYUP = (45, -45, 90, 0)
    PITCH = 10.0
    SHAPE = (40, 16, 16)

    def _run(self, monkeypatch, angles):
        stub = _stub_readers(angles)
        monkeypatch.setattr(M, "_ti_readers", lambda repo: stub)
        m = make_manifest(volume_shape=self.SHAPE, requested_layup=self.LAYUP,
                          requested_ply_thickness_vox=self.PITCH)
        return M.layup_recovery(
            np.zeros(self.SHAPE, np.uint8), np.zeros(self.SHAPE, np.uint8), manifest=m
        )

    def test_an_exact_read_scores_zero_error_and_every_ply_hit(self, monkeypatch):
        out = self._run(monkeypatch, {"fft_slice": [45, 135, 90, 0]})
        r = out["readers"]["fft_slice"]
        assert out["requested_deg"] == [45.0, 135.0, 90.0, 0.0]
        assert r["median_abs_error_deg"] == pytest.approx(0.0)
        assert r["strict_class_accuracy"] == pytest.approx(1.0)
        assert r["per_ply_hit"] == [True] * 4
        assert r["recovered_ply_count"] == 4 == out["requested_ply_count"]

    def test_errors_are_wrapped_into_plus_or_minus_ninety(self, monkeypatch):
        # 179 against a requested 0 is a 1-degree error on an axial angle.
        out = self._run(monkeypatch, {"fft_slice": [45, 135, 90, 179]})
        r = out["readers"]["fft_slice"]
        assert abs(r["errors_deg"][3]) == pytest.approx(1.0)
        assert r["strict_class_accuracy"] == pytest.approx(1.0)

    def test_a_wrong_ply_is_counted_as_a_miss(self, monkeypatch):
        out = self._run(monkeypatch, {"fft_slice": [45, 135, 0, 0]})
        r = out["readers"]["fft_slice"]
        assert r["per_ply_hit"] == [True, True, False, True]
        assert r["strict_class_accuracy"] == pytest.approx(0.75)

    def test_a_reader_that_flattens_the_stack_recovers_fewer_plies(self, monkeypatch):
        out = self._run(monkeypatch, {"fft_slice": [45, 45, 90, 90]})
        assert out["readers"]["fft_slice"]["recovered_ply_count"] == 2
        assert out["requested_ply_count"] == 4

    def test_a_reader_that_did_not_run_is_reported_absent_not_perfect(self, monkeypatch):
        out = self._run(monkeypatch, {"fft_slice": [45, 135, 90, 0]})
        assert out["readers"]["pore_axes"] == {"available": False}

    def test_a_volume_narrower_than_the_reader_window_is_refused(self, monkeypatch):
        stub = _stub_readers({"fft_slice": [0, 0, 0, 0]}, window=1024)
        monkeypatch.setattr(M, "_ti_readers", lambda repo: stub)
        m = make_manifest(volume_shape=self.SHAPE, requested_layup=self.LAYUP,
                          requested_ply_thickness_vox=self.PITCH)
        with pytest.raises(ValueError, match="in-plane window"):
            M.layup_recovery(np.zeros(self.SHAPE, np.uint8),
                             np.zeros(self.SHAPE, np.uint8), manifest=m)

    def test_the_ply_count_is_one_more_than_the_class_changes(self):
        assert M._run_count(np.array([0, 0, 1, 1, 2])) == 3
        assert M._run_count(np.array([0, 1, 0, 1])) == 4
        assert M._run_count(np.array([3, 3, 3])) == 1


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    SHAPE = (64, 512, 512)

    def _manifest(self):
        return make_manifest(volume_shape=self.SHAPE,
                             requested_material="requested_material.npy")

    def test_a_label_that_matches_the_request_scores_dice_one(self):
        material = material_notch_and_hole(self.SHAPE)
        label = np.where(material, LABEL_MATERIAL, LABEL_AIR).astype(np.uint8)
        out = M.geometry_agreement(label, material, manifest=self._manifest())
        assert out["dice_air"] == pytest.approx(1.0)
        assert out["air_fraction_inside_material"] == pytest.approx(0.0)
        assert out["air_fraction_outside_material"] == pytest.approx(1.0)

    def test_a_label_that_ignores_the_request_scores_dice_zero(self):
        material = material_notch_and_hole(self.SHAPE)
        label = np.zeros(self.SHAPE, np.uint8)
        out = M.geometry_agreement(label, material, manifest=self._manifest())
        assert out["dice_air"] == pytest.approx(0.0)
        assert out["recall_air"] == pytest.approx(0.0)

    def test_a_half_filled_request_scores_the_dice_the_overlap_implies(self):
        material = np.ones(self.SHAPE, bool)
        material[:, :100, :] = False              # 100 rows of requested air
        label = np.zeros(self.SHAPE, np.uint8)
        label[:, :50, :] = LABEL_AIR              # half of them delivered
        out = M.geometry_agreement(label, material, manifest=self._manifest())
        assert out["dice_air"] == pytest.approx(2 * 50 / (50 + 100))
        assert out["precision_air"] == pytest.approx(1.0)
        assert out["recall_air"] == pytest.approx(0.5)

    def test_a_case_that_asked_for_no_geometry_is_refused(self):
        material = np.ones(self.SHAPE, bool)
        with pytest.raises(ManifestError, match="painted material map"):
            M.geometry_agreement(np.zeros(self.SHAPE, np.uint8), material,
                                 manifest=make_manifest(volume_shape=self.SHAPE))

    def test_the_notch_and_the_hole_are_both_present_and_do_not_touch(self):
        from scipy import ndimage

        material = material_notch_and_hole(self.SHAPE)
        _, n = ndimage.label(~material[0])
        assert n == 2
        assert (~material).mean() > 0.01


# ---------------------------------------------------------------------------
# Phases, failure and the other small metrics
# ---------------------------------------------------------------------------

class TestPhasesAndFailure:
    def test_fractions_are_taken_inside_the_requested_material(self):
        shape = (128, 128, 128)
        material = np.ones(shape, bool)
        material[:, :64, :] = False
        label = np.where(material, LABEL_MATERIAL, LABEL_AIR).astype(np.uint8)
        label[:, 64:, :32] = LABEL_PORE
        m = make_manifest(volume_shape=shape, requested_material="requested_material.npy")
        out = M.phase_fractions(label, material, manifest=m)
        assert out["phi_pore"] == pytest.approx(0.25)
        assert out["phi_pore_all"] == pytest.approx(0.125)
        assert out["air_fraction"] == pytest.approx(0.0)
        assert out["air_fraction_all"] == pytest.approx(0.5)

    def test_the_three_failure_conditions_each_fire_on_their_own(self):
        shape = (64, 64, 64)
        mat = np.ones(shape, bool)
        m = make_manifest(volume_shape=shape)
        assert M.failure_flags(np.zeros(shape, np.uint8), mat, manifest=m)["phi_collapsed"]
        assert M.failure_flags(np.full(shape, LABEL_PORE, np.uint8), mat,
                               manifest=m)["phi_saturated"]
        air = np.zeros(shape, np.uint8)
        air[:32] = LABEL_AIR
        assert M.failure_flags(air, mat, manifest=m)["air_in_material"]

    def test_the_interior_excludes_a_thirty_two_voxel_shell(self):
        m = M.interior_mask((192, 192, 192))
        assert m.sum() == 128 ** 3
        assert not m[31].any() and m[32].any()
        with pytest.raises(ValueError, match="no interior"):
            M.interior_mask((64, 64, 64))

    def test_degenerate_cells_count_the_tiles_outside_the_failure_band(self):
        phi = np.array([[[0.0, 0.03], [0.6, 0.03]]], np.float32)
        label = label_with_tile_porosity(phi)
        out = M.degenerate_cells(label, np.ones(label.shape, bool),
                                 manifest=make_manifest(volume_shape=label.shape))
        assert out["degenerate_cell_fraction"] == pytest.approx(0.5)

    def test_pore_dice_of_a_volume_with_itself_is_one(self):
        rng = np.random.default_rng(1)
        a = (rng.random((32, 32, 32)) < 0.1).astype(np.uint8)
        assert M.pore_dice(a, a) == pytest.approx(1.0)
        assert np.isnan(M.pore_dice(np.zeros((8, 8, 8), np.uint8),
                                    np.zeros((8, 8, 8), np.uint8)))


# ---------------------------------------------------------------------------
# The requests the cases build
# ---------------------------------------------------------------------------

#: The layup requests are read from the dataset, not typed into source, so the
#: case builders need it.  A worktree has no `data/` until it is symlinked in -
#: see docs/DEVELOPMENT.md - and a missing dataset is a skip, not a failure.
LAYUP_TRUTH = Path(__file__).resolve().parents[1] / "data" / "layup_ground_truth.json"
needs_layup_truth = pytest.mark.skipif(
    not LAYUP_TRUTH.exists(), reason=f"{LAYUP_TRUTH} is not present"
)


class TestRequests:
    @needs_layup_truth
    def test_every_assessment_builds_cases_with_unique_names(self):
        for name in ("sampler", "porosity_global", "porosity_local", "cfg",
                     "layup", "assembly", "geometry"):
            specs = build_cases(name)
            assert specs, name
            assert len({s.name for s in specs}) == len(specs), name
            assert all(s.assessment == name for s in specs)
            for s in specs:
                assert all(v % TILE == 0 for v in s.volume_shape), s.name

    @needs_layup_truth
    def test_the_s_nb_arm_runs_on_a_volume_that_has_a_chunk_plane(self):
        nb = [s for s in build_cases("cfg") if s.notes.get("arm") == "s_nb"]
        assert nb
        for s in nb:
            period = tuple(TILE * c for c in s.chunk_tiles)
            assert any(p < d for p, d in zip(period, s.volume_shape)), s.name

    @needs_layup_truth
    def test_the_layup_c_request_is_a_permutation_of_a(self):
        from poregen.eval_v4.cases import load_layups

        layups = load_layups()
        assert sorted(layups["C"]["plies"]) == sorted(layups["A"]["plies"])
        assert layups["C"]["plies"] != layups["A"]["plies"]
        assert len(layups["B16"]["plies"]) == 16
        assert layups["B16"]["ply_vox"] == pytest.approx(10.0)

    @needs_layup_truth
    def test_the_assembly_offsets_separate_chunk_alignment_from_window_phase(self):
        """One offset confounds the two things an offset can move.

        32 voxels is a WHOLE window stride, so it keeps the window phase and
        moves only the chunk alignment; 16 is half a stride, so it moves the
        phase as well.  Both are whole latent cells, because the noise frame
        the sampler rolls lives on the latent grid.
        """
        from poregen.eval_v4.cases import (
            ASSEMBLY_OFFSETS,
            ASSEMBLY_REGION,
            LATENT_DOWNSAMPLE,
            WINDOW_STRIDE,
        )

        assert ASSEMBLY_OFFSETS[0] == 0            # the reference
        assert set(ASSEMBLY_OFFSETS) == {0, 16, 32}
        assert 32 % WINDOW_STRIDE == 0             # same phase, new chunk plane
        assert 16 % WINDOW_STRIDE == WINDOW_STRIDE // 2      # half-window phase
        specs = build_cases("assembly")
        by_offset = {s.notes["offset"] for s in specs}
        assert by_offset == set(ASSEMBLY_OFFSETS)
        for s in specs:
            off = s.notes["offset"]
            assert s.request_offset == (off,) * 3
            assert s.specimen_box == ((off,) * 3, tuple(off + r for r in ASSEMBLY_REGION))
            assert s.region_offset == (off,) * 3
            assert all(o % LATENT_DOWNSAMPLE == 0 for o in s.request_offset), s.name
            assert all(o + r <= v for o, r, v in
                       zip(s.region_offset, s.region_shape, s.volume_shape)), s.name

    def test_the_orientation_profile_moves_with_the_request(self):
        layup, pitch = (45, -45, 90, 0), 10.0
        base = theta_for_canvas(64, layup, pitch, 0)
        shifted = theta_for_canvas(96, layup, pitch, 32)
        np.testing.assert_array_equal(base, shifted[32:96])

    def test_the_unshifted_profile_is_the_samplers_own(self):
        from poregen.diffusion.sampler import theta_from_layup

        layup, pitch = (45, -45, 90, 0), 19.6
        np.testing.assert_array_equal(
            theta_for_canvas(192, layup, pitch, 0),
            theta_from_layup(192, list(layup), pitch),
        )

    def test_the_latent_material_map_is_the_block_mean_not_a_sample(self):
        vox = np.zeros((4, 4, 8), bool)
        vox[:, :, :4] = True                       # one whole cell, one empty
        m = latent_material_map(vox)
        assert m.shape == (1, 1, 2)
        assert m[0, 0, 0] == pytest.approx(1.0)
        assert m[0, 0, 1] == pytest.approx(0.0)
        half = np.zeros((4, 4, 4), bool)
        half[:2] = True
        assert latent_material_map(half)[0, 0, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Which latent store a run is evaluated against
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, *, name: str, z: int, vae_ckpt: Path) -> Path:
    """A latent store that is only its metadata - no arrays are needed here."""
    root = tmp_path / "data" / name
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({
        "vae_checkpoint": str(vae_ckpt),
        "latent_shape": [z, 16, 16, 16],
        "normalization": {
            "per_channel_mean": [0.0] * z,
            "per_channel_std": [1.0] * z,
        },
        "conditioning": {"por_standardisation": {"mean": -3.5, "std": 0.5}},
    }))
    return root


def _run_cfg(*, store: str, z: int, vae_ckpt: str) -> dict:
    return {
        "model": {"type": "unet3d", "z_channels": z, "channel_mult": [1, 2, 4]},
        "data": {"latents_root": store},
        "vae": {"checkpoint": vae_ckpt},
    }


class TestLatentStoreResolution:
    """The store comes from the run, and has to agree with it.

    A run trained on one rung and scored against another produces a volume, not
    an error: both stores are valid files and the sampler never learns which one
    the weights belong to.  These are the two disagreements that cannot be seen
    in the output afterwards.

    Both the eval suite and ``scripts/generate_volumes.py`` go through this one
    function, so an explicit override has to face the same two checks.
    """

    def test_there_is_no_default_store(self):
        from poregen.eval_v4 import generate as G

        assert not hasattr(G, "DEFAULT_LATENTS_ROOT")

    def test_matching_store_resolves_relative_to_the_repo(self, tmp_path):
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=ckpt)
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        root, meta = resolve_latent_store(cfg, tmp_path)

        assert root == (tmp_path / "data" / "latents_z8").resolve()
        assert meta["latent_shape"][0] == 8

    def test_latent_width_mismatch_raises(self, tmp_path):
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        _store(tmp_path, name="latents_z4", z=4, vae_ckpt=ckpt)
        # The run trained on z=8; the store it is pointed at holds z=4.
        cfg = _run_cfg(store="data/latents_z4", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        with pytest.raises(ValueError, match="Latent width mismatch") as e:
            resolve_latent_store(cfg, tmp_path)
        assert "8" in str(e.value) and "4" in str(e.value)

    def test_vae_checkpoint_mismatch_raises(self, tmp_path):
        store_ckpt = tmp_path / "runs" / "vae" / "r08-other" / "best.ckpt"
        store_ckpt.parent.mkdir(parents=True)
        store_ckpt.touch()
        run_ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        run_ckpt.parent.mkdir(parents=True)
        run_ckpt.touch()
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=store_ckpt)
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        with pytest.raises(ValueError, match="VAE checkpoint mismatch") as e:
            resolve_latent_store(cfg, tmp_path)
        assert str(run_ckpt) in str(e.value)
        assert str(store_ckpt) in str(e.value)

    def test_a_run_that_names_no_store_raises(self, tmp_path):
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")
        cfg["data"] = {}

        with pytest.raises(KeyError, match="data.latents_root"):
            resolve_latent_store(cfg, tmp_path)

    def test_a_missing_store_names_what_the_run_asked_for(self, tmp_path):
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        with pytest.raises(FileNotFoundError, match="data/latents_z8"):
            resolve_latent_store(cfg, tmp_path)

    def test_two_spellings_of_one_checkpoint_are_not_a_mismatch(self, tmp_path):
        """The check is on the FILE, never on how the path was spelled.

        ``build_latent_dataset`` writes ``vae_checkpoint`` as an absolute path;
        a run config carries the repo-relative one.  Comparing the two strings
        rejects every correctly matched store there is, which is worse than no
        check at all: the one escape hatch left is the override, and it would be
        used to get past a false alarm.
        """
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        # The store names the checkpoint absolutely, and through a redundant
        # ``.`` / ``..`` hop for good measure.  The run names it relatively.
        store_spelling = tmp_path / "runs" / "vae" / "r08" / ".." / "r08" / "best.ckpt"
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=store_spelling)
        cfg = _run_cfg(store="./data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")
        assert str(store_spelling) != cfg["vae"]["checkpoint"]

        root, meta = resolve_latent_store(cfg, tmp_path)

        assert root == (tmp_path / "data" / "latents_z8").resolve()
        assert meta["latent_shape"][0] == 8

    def test_an_absolute_store_reference_resolves_unchanged(self, tmp_path):
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        root = _store(tmp_path, name="latents_z8", z=8, vae_ckpt=ckpt)
        cfg = _run_cfg(store=str(root), z=8, vae_ckpt=str(ckpt))

        assert resolve_latent_store(cfg, tmp_path)[0] == root.resolve()

    def test_an_override_replaces_the_store_the_run_names(self, tmp_path):
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=ckpt)
        _store(tmp_path, name="latents_z8_probe", z=8, vae_ckpt=ckpt)
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        root, _ = resolve_latent_store(cfg, tmp_path, override="data/latents_z8_probe")

        assert root == (tmp_path / "data" / "latents_z8_probe").resolve()

    def test_an_override_of_the_wrong_width_raises(self, tmp_path):
        ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.touch()
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=ckpt)
        _store(tmp_path, name="latents_z4", z=4, vae_ckpt=ckpt)
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        with pytest.raises(ValueError, match="Latent width mismatch"):
            resolve_latent_store(cfg, tmp_path, override="data/latents_z4")

    def test_an_override_with_the_wrong_vae_raises(self, tmp_path):
        run_ckpt = tmp_path / "runs" / "vae" / "r08" / "best.ckpt"
        run_ckpt.parent.mkdir(parents=True)
        run_ckpt.touch()
        other_ckpt = tmp_path / "runs" / "vae" / "r08-other" / "best.ckpt"
        other_ckpt.parent.mkdir(parents=True)
        other_ckpt.touch()
        _store(tmp_path, name="latents_z8", z=8, vae_ckpt=run_ckpt)
        _store(tmp_path, name="latents_other", z=8, vae_ckpt=other_ckpt)
        cfg = _run_cfg(store="data/latents_z8", z=8, vae_ckpt="runs/vae/r08/best.ckpt")

        with pytest.raises(ValueError, match="VAE checkpoint mismatch"):
            resolve_latent_store(cfg, tmp_path, override="data/latents_other")


class TestGenerateVolumesUsesTheRunsStore:
    """``scripts/generate_volumes.py`` shares the check, and has no default."""

    @staticmethod
    def _script():
        path = Path(__file__).resolve().parents[1] / "scripts" / "generate_volumes.py"
        spec = importlib.util.spec_from_file_location("generate_volumes_store", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_latents_root_has_no_default(self):
        mod = self._script()

        args = mod._build_parser().parse_args(["--checkpoint", "runs/ldm/x/checkpoints/b.ckpt"])

        assert args.latents_root is None

    def test_the_script_shares_the_one_resolver(self):
        from poregen.eval_v4.generate import resolve_latent_store as canonical

        assert self._script().resolve_latent_store is canonical


# ---------------------------------------------------------------------------
# Smoke test on real campaign-05 volumes
# ---------------------------------------------------------------------------

CAMPAIGN_05 = (
    Path(__file__).resolve().parents[1]
    / "runs" / "campaigns" / "05-eval-v3-fixed-decode" / "volumes" / "ddim_probe"
)


@pytest.mark.skipif(not CAMPAIGN_05.exists(),
                    reason="campaign 05 volumes are not present")
def test_the_metrics_run_on_a_campaign_05_volume(tmp_path):
    """The metric code against a real generated volume, not a synthetic one.

    Campaign 05 predates the 3-class head, so its ``mask.tif`` is a binary pore
    mask; it is lifted to the label contract here (no air class) purely to give
    the metrics a real array to run on.
    """
    import tifffile

    src = sorted(CAMPAIGN_05.iterdir())[0]
    xct = load_u8(src / "volume.tif")
    mask = tifffile.imread(str(src / "mask.tif"))
    label = (mask > 0).astype(np.uint8) * LABEL_PORE

    m = make_manifest(assessment="unit", case=src.name, volume_shape=xct.shape,
                      requested_global_phi=float(json.loads(
                          (src / "stats.json").read_text()).get("target_porosity", 0.03)))
    material = np.ones(xct.shape, bool)

    phases = M.phase_fractions(label, material, manifest=m)
    seams = M.seam_metrics(xct, manifest=m)
    local = M.local_obedience(label, material, manifest=m, requested_tiles=None)

    assert 0.0 <= phases["phi_pore"] <= 1.0
    assert phases["air_fraction"] == 0.0            # no air class in a v3 volume
    assert np.isfinite(seams["seam_xct_ratio"])
    assert seams["seam_xct_planes"] > 0
    assert local["n_cells"] == 27
    assert M.porosity_error(label, material, manifest=m)["delivered_phi"] == \
        pytest.approx(phases["phi_pore"])


def test_every_assessment_has_cases_a_measurer_and_a_reporter():
    """A missing reporter is SILENT: `report` builds its list from REPORTERS.

    multichunk shipped with cases and a measurer and no reporter, so its
    findings.md would simply not have been written — no error, no warning, just
    an assessment missing from the report. This is the check that would have
    caught it.
    """
    from poregen.eval_v4.cases import ASSESSMENTS
    from poregen.eval_v4.measure import MEASURERS
    from poregen.eval_v4.report import REPORTERS

    assert not set(ASSESSMENTS) - set(MEASURERS), "assessment without a measurer"
    assert not set(ASSESSMENTS) - set(REPORTERS), "assessment without a reporter"


# --------------------------------------------------------------------------- #
# the cross-plane pore Dice is only readable against its own interior
# --------------------------------------------------------------------------- #
class TestPoreDiceAcrossPlanes:
    """The bare number is uninterpretable and must never be quoted alone.

    It is the Dice between two ADJACENT SLICES. Adjacent slices of any porous
    medium disagree, because a pore is finite along the axis, and the value
    falls with pore fraction whether or not anything is broken: real 1024-wide
    test material scores 0.25-0.34 at every plane, chunk or not. Only the ratio
    to the same volume's interior says anything.

    Several tests pass a per-axis period with zeros to isolate one axis. A
    structure that is broken along z is untouched along y and x, and averaging
    all three would hide the very thing being asserted.
    """

    def _box(self, shape=(192, 192, 192)):
        """Pore everywhere: adjacent slices are identical on every axis, so
        every plane scores 1 and nothing about the geometry can confound it.
        A pore box with faces INSIDE the volume would put those faces in the
        interior set and pull the baseline off 1 for a reason unrelated to the
        thing under test."""
        return np.full(shape, M.LABEL_PORE, np.uint8)

    def test_a_volume_continuous_along_every_axis_scores_one_everywhere(self):
        out = M.pore_dice_across_planes(self._box(), 64)
        assert out["mean"] == pytest.approx(1.0)
        assert out["interior_mean"] == pytest.approx(1.0)
        assert out["ratio_to_interior"] == pytest.approx(1.0)

    def test_structure_that_does_not_continue_drops_the_ratio(self):
        """Pore stripes that shift sideways exactly at the chunk plane.

        Either side is ordinary striped material — the interior scores 1 — but
        the two sides do not line up where they meet, which is what an
        assembly failure looks like.
        """
        # 128 deep, so period 64 gives exactly ONE chunk plane and the mean is
        # that plane rather than an average over an untouched second one.
        lab = np.zeros((128, 64, 64), np.uint8)
        y = np.arange(64)
        a = ((y // 8) % 2 == 0)[None, :, None]       # stripes
        b = ((y // 8) % 2 == 1)[None, :, None]       # the complement
        lab[:64] = np.where(a, M.LABEL_PORE, M.LABEL_MATERIAL)
        lab[64:] = np.where(b, M.LABEL_PORE, M.LABEL_MATERIAL)
        out = M.pore_dice_across_planes(lab, (64, 0, 0))
        assert out["mean"] == pytest.approx(0.0)     # no overlap at the plane
        assert out["interior_mean"] == pytest.approx(1.0)
        assert out["ratio_to_interior"] == pytest.approx(0.0)

    def test_a_plane_with_no_pores_on_either_side_is_dropped_not_scored_zero(self):
        """A KNOWN LIMITATION, pinned so it is not mistaken for a pass.

        Dice is undefined when both slices are empty, so the plane leaves the
        average instead of scoring 0. A seam the model left completely
        pore-free is therefore INVISIBLE to this metric — which is exactly the
        direction the ldm06 chunk planes fail in, and why the porosity profile
        beside it is not optional.
        """
        lab = self._box((192, 64, 64))
        lab[62:66, :, :] = M.LABEL_MATERIAL          # clear a band over plane 64
        out = M.pore_dice_across_planes(lab, (64, 0, 0))
        assert out["n_planes"] == 1                  # plane 128 only
        assert out["per_axis"]["axis0"] == [pytest.approx(1.0)]

    def test_a_low_mean_with_an_equally_low_interior_is_NOT_a_seam_defect(self):
        """The real-material case: everything disagrees, planes included.

        Alternating pore slices make every adjacent pair along z disagree
        completely, at the chunk planes and between them alike. A reader
        quoting the mean alone would call this a total assembly failure; the
        ratio says correctly that the planes are no worse than the interior.
        """
        lab = np.zeros((192, 192, 192), np.uint8)
        lab[::2, 8:184, 8:184] = M.LABEL_PORE
        out = M.pore_dice_across_planes(lab, (64, 0, 0))
        assert out["mean"] == pytest.approx(0.0)
        assert out["interior_mean"] == pytest.approx(0.0)
        assert out["ratio_to_interior"] is None       # 0/0 is not a verdict

    def test_the_interior_excludes_the_window_planes_as_well(self):
        """Otherwise the baseline is another seam, not ordinary material."""
        lab = self._box((384, 64, 64))
        out = M.pore_dice_across_planes(lab, (192, 0, 0), window_period=64)
        assert out["n_planes"] == 1                      # the chunk plane at 192
        assert out["n_interior_planes"] > 0
        # 64, 128, 256 and 320 are window planes and must not be in the baseline.
        chosen = {int(i) for i in range(1, 384)
                  if i % 192 and i % 64 == 0}
        assert chosen and out["n_interior_planes"] <= 384 - 1 - len(chosen) - 1

    def test_the_definition_says_the_bare_number_is_not_readable(self):
        out = M.pore_dice_across_planes(self._box(), 64)
        assert "ratio_to_interior" in out["definition"]

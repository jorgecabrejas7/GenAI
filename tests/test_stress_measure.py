"""The stress-geometry measurer, end to end on a small fake campaign.

An assessment that is registered but never run through `measure` fails the
first time it meets a real volume, which on this campaign is after hours of
GPU. The shapes here are small on purpose; what is checked is that every block
is produced, that the blocks that CANNOT be produced say so with a reason
instead of raising, and that the band metric reads a band that was put there
deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.eval_v4 import stress_geometry as SG
from poregen.eval_v4.io import FIELD_NPY, MATERIAL_NPY
from poregen.eval_v4.io import case_dir as case_path
from poregen.eval_v4.io import repo_root, save_case
from poregen.eval_v4.manifest import Manifest

COMMIT = "0" * 40
LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2
SHAPE = (192, 384, 384)          # two chunks on y and x at the 192 period
PERIOD = 192


def _label(shape, material, phi=0.04, *, deplete_band=False, seed=0):
    """Pores at `phi` inside the material, air outside it.

    With ``deplete_band`` the 8 voxels before each chunk end and after each
    chunk start on x get no pores at all — the defect the band metric exists to
    read, put in by hand so the metric can be checked against a known answer.
    """
    rng = np.random.default_rng(seed)
    lab = np.where(rng.random(shape) < phi, LABEL_PORE, LABEL_MATERIAL).astype(np.uint8)
    if deplete_band:
        for axis in (1, 2):
            for plane in range(PERIOD, shape[axis], PERIOD):
                sl = [slice(None)] * 3
                sl[axis] = slice(plane - 8, plane + 8)
                lab[tuple(sl)] = LABEL_MATERIAL
    lab[~material] = LABEL_AIR
    return lab


def _write(root, name, *, material=None, field=None, deplete_band=False,
           layup=(45, -45, 90, 0), seed=101):
    rng = np.random.default_rng(seed)
    xct = np.clip(120 + 6 * rng.standard_normal(SHAPE), 0, 255).astype(np.uint8)
    mat = np.ones(SHAPE, bool) if material is None else material
    lab = _label(SHAPE, mat, deplete_band=deplete_band, seed=seed)
    cell = None if material is None else (
        mat.reshape(SHAPE[0] // 4, 4, SHAPE[1] // 4, 4, SHAPE[2] // 4, 4)
        .mean(axis=(1, 3, 5)).astype(np.float32))
    m = Manifest(
        assessment="stress_geometry", case=name, volume_shape=SHAPE,
        git_commit=COMMIT, model_run="runs/ldm/x", checkpoint_step=1,
        weights="ema", ddim_steps=50, chunk_tiles=(3, 3, 3), window_stride=32,
        decode="overlapped", decode_overlap=32, s_por=1.0, s_nb=1.0, seed=seed,
        objective="v", cfg_rescale=0.0, requested_global_phi=0.04,
        requested_layup=layup, requested_ply_thickness_vox=19.6,
        requested_material="full" if material is None else MATERIAL_NPY,
        requested_field=None if field is None else FIELD_NPY,
        wall_time_s=12.5, peak_gpu_memory_bytes=1 << 30,
        notes={"request": name.split("_ddim")[0], "exploratory": True,
               "off_gates_because": "test", "geometry": "a test shape"},
    )
    save_case(case_path(root, "stress_geometry", name), m, xct, lab,
              requested_field=field, requested_material=cell)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Three requests — full material, a painted shape, a painted field — built
    and measured ONCE.  Per-test rebuilding cost four minutes of CPU that the
    training run on this machine needed more."""
    tmp_path = tmp_path_factory.mktemp("stress")
    _write(tmp_path, "cube1024_ddim50", deplete_band=True)
    _write(tmp_path, "two_coupons_ddim50",
           material=SG.material_two_coupons(SHAPE, gap=64))
    ramp = np.linspace(0.01, 0.08, SHAPE[1] // 64, dtype=np.float32)
    field = np.broadcast_to(ramp[None, :, None],
                            tuple(s // 64 for s in SHAPE)).copy()
    _write(tmp_path, "gradient_spots_ddim50", field=field)
    from poregen.eval_v4.measure import measure_stress_geometry
    return measure_stress_geometry(tmp_path, repo_root())


class TestMeasurer:

    def test_every_block_the_notebook_reads_is_present(self, measured):
        res = measured
        assert res["exploratory"] is True and res["off_gates_because"]
        assert len(res["per_case"]) == 3
        for row in res["per_case"]:
            for key in ("phase_fractions", "geometry_agreement",
                        "surface_agreement", "layup_recovery",
                        "local_obedience", "seam_metrics", "chunk_band",
                        "failure_flags", "wall_time_s",
                        "peak_gpu_memory_bytes"):
                assert key in row, f"{row['case']} is missing {key}"

    def test_a_full_material_request_says_why_it_has_no_geometry_score(self, measured):
        row = next(r for r in measured["per_case"]
                   if r["request"] == "cube1024")
        assert row["geometry_agreement"]["available"] is False
        assert "full" in row["geometry_agreement"]["reason"]

    def test_a_painted_request_is_scored_against_its_own_shape(self, measured):
        row = next(r for r in measured["per_case"]
                   if r["request"] == "two_coupons")
        # The label was built FROM the request, so the agreement is near perfect
        # and any real failure shows as a departure from this.
        assert row["geometry_agreement"]["dice_air"] > 0.95

    def test_the_band_metric_finds_a_band_that_was_put_there(self, measured):
        row = next(r for r in measured["per_case"]
                   if r["request"] == "cube1024")
        band = row["chunk_band"]
        # 384 on two axes is two chunks, so both planes are terminal: the
        # non-terminal reading is empty and says so rather than reporting 0.
        assert band["n_trailing_planes"] == 0
        assert band["ratio_-8_terminal"] == pytest.approx(0.0, abs=1e-9)
        assert band["ratio_+0_terminal"] == pytest.approx(0.0, abs=1e-9)
        assert band["phi_volume"] == pytest.approx(0.04, abs=0.005)

    def test_a_narrow_volume_reports_the_layup_window_instead_of_a_number(self, measured):
        for row in measured["per_case"]:
            lr = row["layup_recovery"]
            assert lr["available"] is False
            assert "1024" in lr["reason"]
            assert lr["in_plane"] == [384, 384]

    def test_the_ramp_is_fitted_only_where_a_field_was_painted(self, measured):
        rows = {r["request"]: r for r in measured["per_case"]}
        assert rows["cube1024"]["ramp"] is None
        ramp = rows["gradient_spots"]["ramp"]
        assert ramp["requested_slope_per_tile"] > 0
        # The fake label has a FLAT porosity, so the delivered ramp is near
        # zero: the metric must report that rather than inheriting the request.
        assert abs(ramp["slope_ratio"]) < 0.1
        assert rows["gradient_spots"]["local_obedience"]["requested"] is True

    def test_the_surface_error_drops_the_columns_the_request_left_empty(self, measured):
        row = next(r for r in measured["per_case"]
                   if r["request"] == "two_coupons")
        lower = row["surface_agreement"]["lower"]
        assert lower["columns_scored"] > 0
        assert np.isfinite(lower["error_abs_mean"])

    def test_the_summary_has_one_entry_per_request(self, measured):
        res = measured
        assert set(res["summary"]) == {"cube1024", "two_coupons", "gradient_spots"}
        assert res["summary"]["cube1024"]["n"] == 1

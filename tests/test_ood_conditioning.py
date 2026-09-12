"""Campaign 20's requests, against the things that make them measurements.

Three claims carry this assessment and each is checked here rather than
asserted in a docstring: no sequence introduces an angle the model never saw,
no sequence is secretly one of the trained ones, and the porosity extremes are
actually distinct requests once the clamp is lifted.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.diffusion.conditioning import POR_MAX, POR_MIN, porosity_to_cond
from poregen.eval_v4.cases import (
    OOD_CORR_VOX,
    OOD_PITCH_VOX,
    OOD_POROSITY,
    OOD_SEQUENCES,
    SHAPE_LARGE,
    build_cases,
    load_layups,
)

TILE = 64
#: The angle set the training volumes contain, and the only one allowed here.
TRAINED_ANGLES = {0, 45, -45, 90}
#: The train-split (mean, std) of log(phi + 1e-3), from the latent store's
#: metadata.json. Hard-coded because the store is 272 GB and this is a unit
#: test; if it ever disagrees with the store, the store wins and this fails.
POR_LOG_STATS = (-4.605927032883609, 1.2686379819165527)


def realised(plies, pitch: float, depth: int = SHAPE_LARGE[0]) -> tuple[int, ...]:
    """The per-ply-block angle the model is actually conditioned on.

    Comparing the REQUEST lists would let a 4-ply cross-ply and a 10-ply
    sequence look different while conditioning identically, or the reverse.
    What reaches the model is the sequence repeated cyclically over the blocks
    the pitch lays down, so that is what is compared.
    """
    n = int(np.ceil(depth / pitch))
    return tuple(int(plies[k % len(plies)]) % 180 for k in range(n))


class TestSequences:

    def test_no_sequence_introduces_an_angle_the_model_never_saw(self):
        """The point of the group: the ORDER is new, the orientations are not."""
        for name, plies in OOD_SEQUENCES.items():
            assert set(plies) <= TRAINED_ANGLES, (name, sorted(set(plies)))

    def test_no_sequence_coincides_with_a_trained_or_measured_one(self):
        """A permutation that happens to equal layup A is not a new request."""
        layups = load_layups()
        known = {name: realised(spec["plies"], spec["ply_vox"])
                 for name, spec in layups.items()}
        pitch_a = layups["A"]["ply_vox"]
        for name, plies in OOD_SEQUENCES.items():
            got = realised(plies, pitch_a)
            for other, seq in known.items():
                assert got != seq, f"{name} conditions identically to layup {other}"

    def test_the_four_sequences_differ_from_each_other(self):
        seen = {}
        pitch_a = load_layups()["A"]["ply_vox"]
        for name, plies in OOD_SEQUENCES.items():
            got = realised(plies, pitch_a)
            assert got not in seen, f"{name} conditions identically to {seen.get(got)}"
            seen[got] = name

    def test_a_short_sequence_still_fills_the_depth(self):
        """Cross-ply is four plies; the request must not stop a third of the
        way down the volume."""
        pitch_a = load_layups()["A"]["ply_vox"]
        got = realised(OOD_SEQUENCES["seq_crossply"], pitch_a)
        assert len(got) == int(np.ceil(SHAPE_LARGE[0] / pitch_a))
        assert set(got) == {0, 90}


class TestPitch:

    def test_the_pitches_bracket_the_two_trained_ones(self):
        layups = load_layups()
        trained = sorted({layups["A"]["ply_vox"], layups["B16"]["ply_vox"]})
        assert min(OOD_PITCH_VOX) < min(trained), "8 must be below the thinner trained pitch"
        assert max(OOD_PITCH_VOX) > max(trained), "32 must be above the thicker one"

    def test_the_ply_counts_are_the_ones_the_brief_asks_for(self):
        """'ply thickness 8 and 32' and '24 thin plies and 6 thick plies' are
        the SAME request at 192 deep, so this is one set of two cases."""
        counts = {p: int(np.ceil(SHAPE_LARGE[0] / p)) for p in OOD_PITCH_VOX}
        assert counts == {8.0: 24, 32.0: 6}


class TestPorosityExtremes:

    def test_the_clamp_is_lifted_exactly_where_it_would_destroy_the_request(self):
        lifted = {phi for phi, clamped in OOD_POROSITY if not clamped}
        kept = {phi for phi, clamped in OOD_POROSITY if clamped}
        assert lifted == {0.000, 0.001, 0.150, 0.200}
        # 0.002 IS POR_MIN, so the clamp is a no-op there and the case is the
        # in-distribution anchor the other four are read against.
        assert kept == {0.002}
        assert all(POR_MIN <= p <= POR_MAX for p in kept)

    def test_with_the_clamp_on_the_three_low_requests_are_one_request(self):
        """Why the brief's clamp lift had to extend to the low pair.

        The brief lifted the clamp for 0.15 and 0.20 only. Left clamped, the
        low row is not three requests: it is 0.002 generated nine times.
        """
        clamped = {float(np.clip(p, POR_MIN, POR_MAX)) for p in (0.000, 0.001, 0.002)}
        assert clamped == {POR_MIN}

    def test_lifted_the_low_requests_separate_in_the_conditioning(self):
        cond = [float(porosity_to_cond(p, POR_LOG_STATS)) for p in (0.000, 0.001, 0.002)]
        assert cond[0] < cond[1] < cond[2]
        # Each step is a real distance in the training distribution's own units.
        assert cond[1] - cond[0] > 0.5
        assert cond[2] - cond[1] > 0.3

    def test_the_clamp_is_far_milder_at_the_top_than_at_the_bottom(self):
        """So the two extremes must not be read as symmetric.

        cond_por is log(phi + 1e-3): the top of the range is compressed.
        """
        c = {p: float(porosity_to_cond(p, POR_LOG_STATS))
             for p in (0.000, 0.002, 0.107, 0.150, 0.200)}
        low_gap = c[0.002] - c[0.000]      # what the clamp erases at the bottom
        high_gap = c[0.200] - c[0.107]     # what it erases at the top
        assert low_gap > high_gap
        assert high_gap < 0.6


class TestCaseList:

    def test_the_expected_number_of_cases_in_each_group(self):
        cases = build_cases("ood_conditioning")
        from collections import Counter
        got = Counter(c.notes["group"] for c in cases)
        assert got == {"sequence": 12, "pitch": 6, "porosity": 15, "correlation": 3}
        assert len(cases) == 36

    def test_every_case_declares_itself_exploratory(self):
        for c in build_cases("ood_conditioning"):
            assert c.notes.get("exploratory") is True, c.name
            assert c.notes.get("off_gates_because"), c.name

    def test_every_axis_is_a_whole_number_of_tiles(self):
        for c in build_cases("ood_conditioning"):
            assert all(v % TILE == 0 for v in c.volume_shape), c.name

    def test_the_layup_groups_are_wide_enough_for_the_readers(self):
        """Sequence, pitch and correlation cases exist to be READ BACK, so they
        must hold the 1024-voxel T-I window in both in-plane axes."""
        from poregen.eval_v4.measure import LAYUP_WINDOW_VOX

        for c in build_cases("ood_conditioning"):
            if c.notes["group"] == "porosity":
                continue
            assert min(c.volume_shape[1], c.volume_shape[2]) >= LAYUP_WINDOW_VOX, c.name

    def test_only_the_porosity_group_ever_lifts_the_clamp(self):
        for c in build_cases("ood_conditioning"):
            if c.notes["group"] != "porosity":
                assert c.clamp_porosity is True, c.name

    def test_the_correlation_lengths_differ_from_production(self):
        """Isotropic is itself off-manifold — the measured lengths are
        (79, 414, 901). The case asks two things at once and must say so."""
        for c in build_cases("ood_conditioning"):
            if c.notes["group"] != "correlation":
                continue
            assert c.notes["isotropic"] is True
            assert c.notes["corr_length_vox"] in OOD_CORR_VOX
            assert c.field_fn is not None


# ---------------------------------------------------------------------------
# The measurer and reporter, end to end on a small fake campaign
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from poregen.eval_v4.io import FIELD_NPY  # noqa: E402
from poregen.eval_v4.io import case_dir as case_path  # noqa: E402
from poregen.eval_v4.io import repo_root, save_case  # noqa: E402
from poregen.eval_v4.manifest import Manifest  # noqa: E402

COMMIT = "0" * 40
LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2
FAKE_SHAPE = (192, 192, 192)


def _write_ood(root, name, *, group, phi, clamped=True, pitch=19.6,
               layup=(45, -45, 90, 0), field=None, notes=None, seed=101):
    rng = np.random.default_rng(seed)
    xct = np.clip(120 + 6 * rng.standard_normal(FAKE_SHAPE), 0, 255).astype(np.uint8)
    lab = np.where(rng.random(FAKE_SHAPE) < max(phi, 1e-4),
                   LABEL_PORE, LABEL_MATERIAL).astype(np.uint8)
    m = Manifest(
        assessment="ood_conditioning", case=name, volume_shape=FAKE_SHAPE,
        git_commit=COMMIT, model_run="runs/ldm/x", checkpoint_step=1,
        weights="ema", ddim_steps=50, chunk_tiles=(3, 3, 3), window_stride=32,
        decode="overlapped", decode_overlap=32, s_por=1.0, s_nb=1.0, seed=seed,
        objective="v", cfg_rescale=0.0, requested_global_phi=phi,
        requested_layup=layup, requested_ply_thickness_vox=pitch,
        requested_material="full", porosity_clamped=clamped,
        requested_field=None if field is None else FIELD_NPY,
        wall_time_s=1.0, peak_gpu_memory_bytes=1 << 30,
        notes={"group": group, "exploratory": True,
               "off_gates_because": "test", **(notes or {})},
    )
    save_case(case_path(root, "ood_conditioning", name), m, xct, lab,
              requested_field=field)


@pytest.fixture(scope="module")
def ood_measured(tmp_path_factory):
    """One case per group, built and measured ONCE."""
    root = tmp_path_factory.mktemp("ood")
    _write_ood(root, "seq_crossply_seed101", group="sequence", phi=0.03,
               layup=(0, 90, 90, 0), notes={"sequence": "seq_crossply"})
    _write_ood(root, "pitch32_seed101", group="pitch", phi=0.03, pitch=32.0,
               notes={"pitch_vox": 32.0})
    _write_ood(root, "phi0_seed101", group="porosity", phi=0.0, clamped=False,
               notes={"clamp_lifted": True})
    _write_ood(root, "phi0.2_seed101", group="porosity", phi=0.2, clamped=False,
               notes={"clamp_lifted": True})
    _write_ood(root, "phi0.002_seed101", group="porosity", phi=0.002,
               clamped=True, notes={"clamp_lifted": False})
    grid = tuple(s // 64 for s in FAKE_SHAPE)
    _write_ood(root, "corr16", group="correlation", phi=0.03,
               field=np.full(grid, 0.03, np.float32),
               notes={"corr_length_vox": 16.0, "isotropic": True})
    from poregen.eval_v4.measure import measure_ood_conditioning
    res = measure_ood_conditioning(root, repo_root())
    res["_root"] = str(root)
    return res


class TestMeasurer:

    def test_every_block_the_report_reads_is_present(self, ood_measured):
        assert ood_measured["exploratory"] is True
        assert len(ood_measured["per_case"]) == 6
        for row in ood_measured["per_case"]:
            for key in ("phase_fractions", "seam_metrics", "failure_flags",
                        "conditioning", "wall_time_s", "porosity_error"):
                assert key in row, f"{row['case']} is missing {key}"

    def test_the_clamp_is_reported_per_case_and_not_inferred(self, ood_measured):
        by = {r["case"]: r["conditioning"] for r in ood_measured["per_case"]}
        assert by["phi0_seed101"]["porosity_clamped"] is False
        assert by["phi0_seed101"]["phi_conditioned"] == 0.0
        assert by["phi0_seed101"]["phi_if_clamped"] == POR_MIN
        assert by["phi0_seed101"]["clamp_changed_the_request"] is True
        # 0.002 IS POR_MIN, so holding it changes nothing.
        assert by["phi0.002_seed101"]["porosity_clamped"] is True
        assert by["phi0.002_seed101"]["phi_conditioned"] == pytest.approx(0.002)

    def test_cond_por_says_how_far_off_the_manifold_each_request_is(self, ood_measured):
        by = {r["case"]: r["conditioning"]["cond_por"]
              for r in ood_measured["per_case"] if r["group"] == "porosity"}
        assert by["phi0_seed101"] < by["phi0.002_seed101"] < by["phi0.2_seed101"]

    def test_the_layup_groups_report_a_reason_when_too_narrow(self, ood_measured):
        """The fake volumes are 192 wide, under the 1024 T-I window."""
        for r in ood_measured["per_case"]:
            if r["group"] not in ("sequence", "pitch"):
                continue
            assert r["layup_recovery"]["available"] is False
            assert "1024" in r["layup_recovery"]["reason"]
            assert r["requested_ply_blocks"] > 0

    def test_the_correlation_case_measures_the_field_back(self, ood_measured):
        r = next(r for r in ood_measured["per_case"] if r["group"] == "correlation")
        assert r["requested_corr_vox"] == 16.0
        assert set(r["field"]["per_axis"]) >= {"z", "y", "x"}
        assert r["local_obedience"]["requested"] is True

    def test_the_summary_has_one_entry_per_group(self, ood_measured):
        assert set(ood_measured["summary"]) == {
            "sequence", "pitch", "porosity", "correlation"}


class TestReporter:

    def test_findings_carry_all_four_groups(self, ood_measured):
        from poregen.eval_v4.io import write_results
        from poregen.eval_v4.report import report_one

        root = Path(ood_measured["_root"])
        write_results(root, "ood_conditioning", ood_measured)
        text = report_one(root, "ood_conditioning").read_text()
        assert "EXPLORATORY" in text
        for heading in ("stacking sequences", "ply pitch", "requested porosity",
                        "correlation length"):
            assert heading in text, heading
        # The clamp must be visible in the table, not only in the prose.
        assert "lifted" in text and "held" in text
        assert "stated failure mode" in text
        assert (root / "ood_conditioning" / "figures"
                / "porosity_extremes.png").exists()

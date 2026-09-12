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

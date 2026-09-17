"""Campaign 24: the conditioning ablations, and the trap in the swap rows.

An ablation arm is only a measurement if the input it disturbs ACTUALLY
REACHES THE MODEL DIFFERENTLY. Two ways that quietly fails here, and both are
tested rather than assumed:

  * a swapped specimen box that the canvas falls OUTSIDE of makes every
    distance saturate to the value the unswapped box already gives — the arm
    then measures nothing and looks like a result;
  * a zeroed input that was already zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.diffusion.sampler import ABLATABLE, VolumeGenerator
from poregen.eval_v4.cases import (
    SHAPE_SMALL,
    _shifted_box,
    build_cases,
    load_layups,
)
from poregen.eval_v4.generate import theta_for_canvas


def gen(ablate=(), theta=None):
    """A VolumeGenerator with only the fields the conditioning helpers read."""
    g = object.__new__(VolumeGenerator)
    g.patch_size, g.latent_size = 64, 16
    g.ablate = frozenset(ablate)
    g.theta_deg = None if theta is None else np.asarray(theta)
    return g


class TestTheCaseList:

    def test_every_arm_has_three_seeds(self):
        cases = build_cases("ablation")
        arms: dict[str, int] = {}
        for c in cases:
            arms[c.notes["arm"]] = arms.get(c.notes["arm"], 0) + 1
        assert arms and all(n == 3 for n in arms.values()), arms

    def test_every_case_is_ddim_50(self):
        """The result is a DIFFERENCE against campaign 18, and both sides of a
        difference must be sampled the same way."""
        assert {c.ddim_steps for c in build_cases("ablation")} == {50}

    def test_every_case_names_the_row_it_is_a_difference_from(self):
        for c in build_cases("ablation"):
            assert c.notes["compare_against"].startswith("18-eval-v4-final/")
            assert c.notes["kind"] in ("swap", "zero")
            assert c.notes["exploratory"] is True

    def test_each_case_changes_exactly_one_thing(self):
        """An arm that moved two inputs could not attribute what it measured."""
        for c in build_cases("ablation"):
            changed = sum([
                c.cond_layup is not None,
                c.cond_specimen_box is not None,
                bool(c.ablate),
                c.notes["arm"].startswith("material_full"),
                c.notes["arm"].startswith("field_global"),
            ])
            assert changed == 1, f"{c.name} changes {changed} inputs"

    def test_only_known_inputs_are_ablated(self):
        for c in build_cases("ablation"):
            assert set(c.ablate) <= set(ABLATABLE), c.name


class TestThePlySwap:

    def test_it_feeds_c_while_scoring_a(self):
        layups = load_layups()
        c = next(x for x in build_cases("ablation") if x.name.startswith("ply_swap"))
        assert c.layup == layups["A"]["plies"]
        assert c.cond_layup == layups["C"]["plies"]

    def test_the_two_layups_really_give_different_profiles(self):
        """C is a PERMUTATION of A, so this is not obvious and must be checked."""
        layups = load_layups()
        pitch = layups["A"]["ply_vox"]
        ta = theta_for_canvas(192, layups["A"]["plies"], pitch, 0)
        tc = theta_for_canvas(192, layups["C"]["plies"], pitch, 0)
        assert not np.array_equal(ta, tc)
        assert not torch.equal(VolumeGenerator._window_orient(gen(theta=ta), 0),
                               VolumeGenerator._window_orient(gen(theta=tc), 0))


class TestTheZeroedRows:

    def test_zeroing_orientation_gives_exactly_zero(self):
        t = theta_for_canvas(192, load_layups()["A"]["plies"], 19.6, 0)
        assert float(VolumeGenerator._window_orient(gen(("orient",), t), 0).abs().max()) == 0.0

    def test_the_unablated_profile_is_on_the_unit_circle(self):
        """Which is WHY (0, 0) is an impossible input and not an unusual one."""
        t = theta_for_canvas(192, load_layups()["A"]["plies"], 19.6, 0)
        o = VolumeGenerator._window_orient(gen(theta=t), 0)
        assert float((o ** 2).sum(0).mean()) == pytest.approx(1.0, abs=0.06)

    def test_zeroing_position_gives_exactly_zero(self):
        d, d6 = VolumeGenerator._window_position(
            gen(("depth_dist6",)), (64, 64, 64), (0, 0, 0), SHAPE_SMALL)
        assert d == 0.0 and not d6.any()

    def test_an_unknown_ablation_name_is_refused(self):
        with pytest.raises(ValueError, match="ablate must be a subset"):
            VolumeGenerator(sampler=None, vae=None, device=torch.device("cpu"),
                            ablate=("orientation",))


class TestTheSwappedBox:
    """The trap: a swap that silently equals the row it is compared against."""

    def test_the_canvas_stays_inside_the_swapped_box(self):
        lo, hi = _shifted_box(SHAPE_SMALL)
        assert all(l <= 0 for l in lo)
        assert all(h >= s for h, s in zip(hi, SHAPE_SMALL))

    def test_it_changes_both_the_depth_and_the_distances(self):
        true_box = ((0, 0, 0), SHAPE_SMALL)
        swap_box = _shifted_box(SHAPE_SMALL)
        a = VolumeGenerator._window_position(gen(), (0, 64, 64), *true_box)
        b = VolumeGenerator._window_position(gen(), (0, 64, 64), *swap_box)
        assert a[0] != b[0]
        assert not np.array_equal(a[1], b[1])

    def test_the_swapped_depth_still_varies_across_the_canvas(self):
        """A constant depth would be a degenerate input, not a different one."""
        box = _shifted_box(SHAPE_SMALL)
        d = {VolumeGenerator._window_position(gen(), (z, 64, 64), *box)[0]
             for z in (0, 64, 128)}
        assert len(d) == 3

    def test_the_swap_is_not_secretly_the_zero_row(self):
        box = _shifted_box(SHAPE_SMALL)
        depth, d6 = VolumeGenerator._window_position(gen(), (0, 64, 64), *box)
        assert depth != 0.0 or d6.any()

    def test_the_naive_shift_would_have_been_degenerate(self):
        """Guarding the fix: moving a same-sized box puts the canvas OUTSIDE it.

        Windows then sit beyond a face and cond_depth CLIPS TO 0.0 — which is
        the value the zeroed row feeds. The swap row would have collided with
        the row it is supposed to contrast against, on a plausible-looking
        number. This is what `_shifted_box` used to do.
        """
        shape = SHAPE_SMALL
        dz, dy, dx = shape[0] // 2, 256, 256
        naive = ((dz, dy, dx), (shape[0] + dz, shape[1] + dy, shape[2] + dx))
        naive_depths = [VolumeGenerator._window_position(gen(), (z, 64, 64), *naive)[0]
                        for z in (0, 64, 128)]
        assert naive_depths.count(0.0) >= 2, (
            "the naive shift should clip most windows' depth to the zeroed "
            f"row's own value, got {naive_depths}")
        good = [VolumeGenerator._window_position(gen(), (z, 64, 64),
                                                 *_shifted_box(shape))[0]
                for z in (0, 64, 128)]
        assert 0.0 not in good, f"the fixed box must clip nothing, got {good}"

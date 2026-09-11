"""Downstream-utility campaign: are the three arms actually comparable?

The finding this campaign produces is a DIFFERENCE between three numbers.  If
the arms differ in anything but the data mix — a patch more, a step more, a
channel more — the difference measures that instead, and nothing in the output
would say so.  So the tests here are mostly not about training: they are about
the invariants that make the three numbers mean what the README will claim.

Everything runs on CPU on hand-sized tensors.  Nothing reads the real patch
memmaps and nothing touches a GPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / "scripts" / "analysis" / "downstream_utility.py"
    spec = importlib.util.spec_from_file_location("downstream_utility", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


du = _load_module()


# ---------------------------------------------------------------------------
# Fixtures: a fake real distribution and a synthetic pool, as histograms
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pool():
    """Porosity and air per patch for a fake real train split spanning every bin."""
    rng = np.random.default_rng(7)
    n = 4000
    por = np.concatenate([
        rng.uniform(0.000, 0.010, int(n * 0.56)),
        rng.uniform(0.010, 0.030, int(n * 0.22)),
        rng.uniform(0.030, 0.060, int(n * 0.12)),
        rng.uniform(0.060, 0.200, n - int(n * 0.56) - int(n * 0.22) - int(n * 0.12)),
    ])
    air = np.where(rng.random(n) < 0.36, rng.uniform(0.01, 0.9, n), 0.0)
    rng.shuffle(por)
    return por, air


@pytest.fixture(scope="module")
def full_pool_keys():
    """A synthetic pool with plenty of patches in EVERY stratum."""
    return np.repeat(np.arange(du.N_STRATA), 500)


@pytest.fixture(scope="module")
def budget():
    return du.Budget(steps=13, batch_size=4, patch_count=800, test_patches=64,
                     seeds=(101, 202))


def _plans(budget, real_pool, pool_keys, seed):
    por, air = real_pool
    keys = du.stratum_keys(por, air)
    perm = du.seed_permutation(len(por), budget, seed)
    return perm, keys, {
        arm.name: du.plan_arm(arm, budget, seed, perm, keys, pool_keys)
        for arm in du.ARMS
    }


# ---------------------------------------------------------------------------
# The comparability invariants
# ---------------------------------------------------------------------------

def test_every_arm_trains_on_exactly_the_same_number_of_patches(
    budget, real_pool, full_pool_keys
):
    for seed in budget.seeds:
        _, _, plans = _plans(budget, real_pool, full_pool_keys, seed)
        assert {p.total for p in plans.values()} == {budget.patch_count}
        for name, p in plans.items():
            n_real, n_synth = du.arm_patch_counts(du.arm_by_name(name), budget)
            assert p.real_rows.size == n_real
            assert p.synthetic_rows.size == n_synth


def test_arm_patch_counts_always_sum_to_the_budget():
    # Odd totals and awkward fractions are exactly where a second rounding
    # would leave one arm a patch short of another.
    for total in (1, 2, 3, 999, 16001):
        b = du.Budget(patch_count=total)
        for arm in du.ARMS:
            n_real, n_synth = du.arm_patch_counts(arm, b)
            assert n_real + n_synth == total
            assert n_real >= 0 and n_synth >= 0


def test_every_arm_targets_the_same_real_request_distribution(
    budget, real_pool, full_pool_keys
):
    """The point of the stratified draw: the arms differ in provenance only.

    With a pool that covers every stratum, all three arms hold the SAME joint
    (porosity bin x has-air) histogram — the one the real draw of that seed has.
    """
    for seed in budget.seeds:
        perm, keys, plans = _plans(budget, real_pool, full_pool_keys, seed)
        reference = np.bincount(keys[perm], minlength=du.N_STRATA)
        for name, p in plans.items():
            got = np.bincount(keys[p.real_rows], minlength=du.N_STRATA)
            if p.synthetic_rows.size:
                got = got + np.bincount(
                    full_pool_keys[p.synthetic_rows], minlength=du.N_STRATA
                )
            assert np.array_equal(got, reference), name


def test_mixed_arm_reuses_the_real_arms_own_patches(budget, real_pool, full_pool_keys):
    """Arm (c)'s real half is literally a subset of arm (a)'s patches.

    Not cosmetic: it means the mixed arm's real material is the same material,
    so a difference cannot be a lucky or unlucky real draw.
    """
    for seed in budget.seeds:
        _, _, plans = _plans(budget, real_pool, full_pool_keys, seed)
        real_rows = set(plans["real"].real_rows.tolist())
        mixed_rows = set(plans["real_plus_synthetic"].real_rows.tolist())
        assert mixed_rows and mixed_rows <= real_rows
        assert not set(plans["synthetic"].real_rows.tolist())


def test_run_specs_differ_only_in_the_data_mix(budget, real_pool, full_pool_keys):
    """Same steps, same batch size, same architecture width, same totals."""
    for seed in budget.seeds:
        _, _, plans = _plans(budget, real_pool, full_pool_keys, seed)
        specs = {
            name: du.arm_run_spec(du.arm_by_name(name), budget, seed, plans[name])
            for name in plans
        }
        shared = [
            {k: v for k, v in s.items() if k not in du.ARM_VARYING_KEYS}
            for s in specs.values()
        ]
        assert all(s == shared[0] for s in shared[1:]), shared
        # and the three headline budget knobs, named explicitly
        for key in ("steps", "batch_size", "base_channels", "total_patches",
                    "test_patches", "lr", "seed"):
            assert len({s[key] for s in specs.values()}) == 1, key


def test_an_arm_cannot_carry_a_training_setting():
    """``Arm`` holds the data mix and nothing else, by construction.

    A ``steps`` or ``lr`` field on ``Arm`` is all it would take to make the
    campaign meaningless; this test fails the moment one appears.
    """
    assert [f.name for f in dataclasses.fields(du.Arm)] == ["name", "real_fraction"]
    assert du.Arm.__dataclass_params__.frozen
    assert du.Budget.__dataclass_params__.frozen
    assert [a.name for a in du.ARMS] == ["real", "synthetic", "real_plus_synthetic"]
    assert [a.real_fraction for a in du.ARMS] == [1.0, 0.0, 0.5]
    assert du.REFERENCE_ARM == "real"


def test_the_model_cannot_depend_on_the_arm(budget):
    """``build_model`` takes the budget and nothing else, and is deterministic."""
    assert list(inspect.signature(du.build_model).parameters) == ["budget"]

    states = []
    for _ in du.ARMS:
        torch.manual_seed(0)
        states.append(du.build_model(budget).state_dict())
    first = states[0]
    for other in states[1:]:
        assert first.keys() == other.keys()
        for k in first:
            assert torch.equal(first[k], other[k]), k
    n_params = sum(v.numel() for v in first.values())
    assert n_params > 0


def test_the_eval_set_is_fixed_and_arm_independent():
    """Every arm and seed is scored on the same real test patches."""
    b = du.Budget(test_patches=50)
    draws = []
    for _ in range(3):
        rng = np.random.default_rng(du.EVAL_SEED)
        draws.append(np.sort(rng.permutation(1000)[: b.test_patches]))
    assert all(np.array_equal(draws[0], d) for d in draws[1:])


# ---------------------------------------------------------------------------
# The stratified draw
# ---------------------------------------------------------------------------

def test_stratum_keys_use_the_reported_porosity_bins():
    por = np.array([0.0, 0.009, 0.01, 0.029, 0.03, 0.059, 0.06, 0.5])
    air = np.zeros_like(por)
    assert du.stratum_keys(por, air).tolist() == [
        k * du.N_AIR_BINS for k in (0, 0, 1, 1, 2, 2, 3, 3)
    ]
    # air is a presence flag, not a magnitude
    assert du.stratum_keys(np.array([0.02, 0.02, 0.02]),
                           np.array([0.0, 1e-9, 0.9])).tolist() == [2, 3, 3]


def test_match_reproduces_the_target_histogram_exactly(full_pool_keys):
    rng = np.random.default_rng(0)
    target = np.repeat([0, 2, 5, 7], [40, 10, 25, 5])
    idx, report = du.match_strata(target, full_pool_keys, rng)
    assert idx.size == target.size
    assert np.array_equal(
        np.bincount(full_pool_keys[idx], minlength=du.N_STRATA),
        np.bincount(target, minlength=du.N_STRATA),
    )
    assert report["unfilled_strata"] == []
    assert report["redistributed_patches"] == 0
    assert report["max_stratum_reuse"] == 1.0


def test_match_records_reuse_when_a_stratum_is_thin():
    """A pool too small for a stratum repeats patches, and SAYS how badly."""
    pool = np.array([0] * 100 + [3] * 5)
    target = np.repeat([0, 3], [50, 50])
    idx, report = du.match_strata(target, pool, np.random.default_rng(1))
    assert idx.size == 100
    assert np.bincount(pool[idx], minlength=du.N_STRATA)[3] == 50
    assert report["max_stratum_reuse"] == pytest.approx(10.0)
    assert report["strata"][du.stratum_label(3)]["pool"] == 5
    assert report["unfilled_strata"] == []


def test_match_redistributes_a_missing_stratum_and_names_it():
    """A stratum the pool cannot supply is reported, never silently dropped."""
    pool = np.array([0] * 100 + [2] * 100)
    target = np.repeat([0, 2, 5], [30, 30, 40])
    idx, report = du.match_strata(target, pool, np.random.default_rng(2))
    assert idx.size == target.size, "the patch count survives the redistribution"
    counts = np.bincount(pool[idx], minlength=du.N_STRATA)
    assert counts[5] == 0
    assert counts[0] + counts[2] == 100
    assert report["unfilled_strata"] == [du.stratum_label(5)]
    assert report["redistributed_patches"] == 40
    # the redistribution goes to the strata that exist, in their own proportion
    assert counts[0] == counts[2] == 50


def test_match_refuses_a_pool_that_shares_no_stratum():
    with pytest.raises(ValueError, match="nothing to train on"):
        du.match_strata(np.array([1, 1, 1]), np.array([0, 0]),
                        np.random.default_rng(3))


def test_largest_remainder_never_feeds_a_zero_weight_stratum():
    out = du._largest_remainder(np.array([0.0, 0.0, 1.0, 1.0]), 7)
    assert out.sum() == 7
    assert out[0] == out[1] == 0


# ---------------------------------------------------------------------------
# The synthetic arm refuses an incomplete campaign
# ---------------------------------------------------------------------------

def _fake_campaign(root: Path, cases) -> None:
    for assessment, name in cases:
        d = root / assessment / "volumes" / name
        d.mkdir(parents=True, exist_ok=True)
        for f in ("manifest.json", "volume.tif", "label.tif"):
            (d / f).touch()


def test_required_cases_exclude_the_off_manifold_request():
    cases = du.required_synthetic_cases(REPO_ROOT)
    assert cases, "the required case list must not be empty"
    assert {a for a, _ in cases} == set(du.SYNTHETIC_ASSESSMENTS)
    assert not [n for _, n in cases if "0.15" in n]
    assert any(n.startswith("flat_") or n.startswith("rough_") for _, n in cases), \
        "the surface assessment is the only source of AIR in the synthetic set"


def test_synthetic_arm_refuses_to_run_before_the_volumes_exist(tmp_path):
    with pytest.raises(du.MissingSyntheticVolumes) as e:
        du.check_synthetic_volumes(tmp_path, REPO_ROOT)
    msg = str(e.value)
    assert "eval_v4 generate" in msg, "the message must say how to produce them"
    assert str(tmp_path) in msg


def test_a_half_generated_case_does_not_count_as_present(tmp_path):
    cases = du.required_synthetic_cases(REPO_ROOT)
    _fake_campaign(tmp_path, cases)
    assert len(du.check_synthetic_volumes(tmp_path, REPO_ROOT)) == len(cases)

    assessment, name = cases[len(cases) // 2]
    (tmp_path / assessment / "volumes" / name / "manifest.json").unlink()
    with pytest.raises(du.MissingSyntheticVolumes) as e:
        du.check_synthetic_volumes(tmp_path, REPO_ROOT)
    assert f"{assessment}/{name}" in str(e.value)


def test_planning_a_synthetic_arm_without_a_pool_raises(budget, real_pool):
    por, air = real_pool
    keys = du.stratum_keys(por, air)
    perm = du.seed_permutation(len(por), budget, 101)
    with pytest.raises(du.MissingSyntheticVolumes):
        du.plan_arm(du.arm_by_name("synthetic"), budget, 101, perm, keys, None)
    # the real-only arm needs no pool at all
    plan = du.plan_arm(du.arm_by_name("real"), budget, 101, perm, keys, None)
    assert plan.total == budget.patch_count


# ---------------------------------------------------------------------------
# Synthetic patch assembly
# ---------------------------------------------------------------------------

def _write_case(d: Path, volume: np.ndarray, label: np.ndarray) -> None:
    import tifffile
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(d / "volume.tif"), np.ascontiguousarray(volume, np.uint8))
    tifffile.imwrite(str(d / "label.tif"), np.ascontiguousarray(label, np.uint8))
    (d / "manifest.json").write_text("{}")


def test_synthetic_patches_are_cut_and_read_at_the_right_box(tmp_path):
    """The pool's coordinates and the reader's crop must agree.

    An off-by-one here silently trains on patches whose porosity is not the
    porosity the stratified draw selected them for.
    """
    rng = np.random.default_rng(11)
    shape = (96, 96, 96)
    volume = rng.integers(0, 256, shape, dtype=np.uint8)
    label = np.zeros(shape, np.uint8)
    label[:32] = 2                         # air slab
    label[64:, :, :10] = 1                 # a pore block
    case = tmp_path / "sampler" / "volumes" / "c0"
    _write_case(case, volume, label)

    pool = du.scan_synthetic_pool([case], verbose=False)
    assert len(pool) == 2 ** 3             # (96 - 64) / 32 + 1 = 2 per axis
    assert pool.coords[:, 0].tolist() == [0] * len(pool)

    # the summarised fractions are the fractions of the box that is read back
    ds = du.SyntheticPatchSubset(pool, np.arange(len(pool)))
    for i in range(len(pool)):
        xct, lab = ds[i]
        assert xct.shape == (1, du.PATCH, du.PATCH, du.PATCH)
        assert lab.shape == (du.PATCH,) * 3
        assert xct.dtype == torch.float32 and lab.dtype == torch.int64
        assert 0.0 <= float(xct.min()) and float(xct.max()) <= 1.0
        assert float((lab == 1).float().mean()) == pytest.approx(pool.porosity[i], abs=1e-6)
        assert float((lab == 2).float().mean()) == pytest.approx(pool.air[i], abs=1e-6)
        _, z0, y0, x0 = (int(v) for v in pool.coords[i])
        expect = volume[z0:z0 + du.PATCH, y0:y0 + du.PATCH, x0:x0 + du.PATCH]
        # grey is scaled by 1/255, the same normalisation the real memmap
        # loader applies, so a synthetic and a real patch are the same object
        assert torch.equal(
            (xct[0] * 255.0).round().to(torch.uint8), torch.from_numpy(expect)
        )


def test_the_synthetic_pool_spans_both_air_states(tmp_path):
    """A pool with no air at all would make the air-Dice row meaningless."""
    rng = np.random.default_rng(12)
    shape = (96, 96, 96)
    interior = np.zeros(shape, np.uint8)
    interior[::4] = 1
    surface = np.zeros(shape, np.uint8)
    surface[:20] = 2
    c1 = tmp_path / "a" / "volumes" / "interior"
    c2 = tmp_path / "a" / "volumes" / "surface"
    _write_case(c1, rng.integers(0, 256, shape, dtype=np.uint8), interior)
    _write_case(c2, rng.integers(0, 256, shape, dtype=np.uint8), surface)
    pool = du.scan_synthetic_pool([c1, c2], verbose=False)
    air_bins = pool.keys % du.N_AIR_BINS
    assert set(air_bins.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_flips_move_the_image_and_the_label_together():
    """``xct`` has a channel axis and ``label`` has not; both flip on SPATIAL axes."""
    xct = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    label = torch.arange(8, dtype=torch.long).reshape(2, 2, 2)
    x2, l2 = du.apply_flips(xct, label, (True, False, True))
    assert torch.equal(x2, torch.flip(xct, dims=[1, 3]))
    assert torch.equal(l2, torch.flip(label, dims=[0, 2]))
    assert torch.equal(x2[0].long(), l2), "image and label must still correspond"

    same_x, same_l = du.apply_flips(xct, label, (False, False, False))
    assert torch.equal(same_x, xct) and torch.equal(same_l, label)


def test_flip_sampling_is_seeded_and_covers_every_combination():
    torch.manual_seed(0)
    a = [du.sample_flips() for _ in range(200)]
    torch.manual_seed(0)
    b = [du.sample_flips() for _ in range(200)]
    assert a == b
    assert len(set(a)) == 8


def test_augmentation_wrapper_preserves_shapes_and_labels():
    base = [(torch.rand(1, 4, 4, 4), torch.randint(0, 3, (4, 4, 4)))]
    ds = du.FlipAugmented(base)
    torch.manual_seed(3)
    xct, label = ds[0]
    assert xct.shape == (1, 4, 4, 4) and label.shape == (4, 4, 4)
    assert sorted(label.unique().tolist()) == sorted(base[0][1].unique().tolist())


# ---------------------------------------------------------------------------
# The metric, against a hand-computed answer
# ---------------------------------------------------------------------------

def _onehot_logits(pred: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    """(B, C, D, H, W) logits whose argmax is exactly *pred*."""
    return torch.nn.functional.one_hot(pred, n_classes).permute(0, 4, 1, 2, 3).float() * 10.0


@pytest.fixture
def known_answer_case():
    """Two 2x2x2 patches whose Dice is worked out by hand.

    Patch A: label = 4 pore, 4 material.  Prediction = 3 pore, of which 2 are
    right.  pore Dice = 2*2 / (3 + 4) = 4/7.  Air is absent from both, which the
    repo's convention scores as a perfect 1.
    Patch B: label = 8 air, predicted 8 air.  Every class Dice is 1.
    """
    label = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    label[0].view(-1)[:4] = 1
    label[1] = 2
    pred = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    pred[0].view(-1)[[0, 1, 4]] = 1
    pred[1] = 2
    return _onehot_logits(pred), label


def test_dice_and_porosity_match_the_hand_computation(known_answer_case):
    logits, label = known_answer_case
    score = du.SegmentationScore()
    score.update(logits, label)
    r = score.result()

    assert r["dice_pore"] == pytest.approx((4 / 7 + 1.0) / 2)
    assert r["dice_air"] == pytest.approx(1.0)
    assert r["dice_material"] == pytest.approx((2 / 3 + 1.0) / 2)
    # predicted pore fraction 3/8 against a true 4/8, and 0 against 0
    assert r["porosity_mae"] == pytest.approx((0.125 + 0.0) / 2)
    assert r["air_mae"] == pytest.approx(0.0)
    assert r["n_patches"] == 2.0

    # per-bin: patch A's GT porosity 0.5 lands in the top bin, patch B's 0 in the first
    assert r["porosity_n_bin_0"] == 1.0 and r["porosity_mae_bin_0"] == pytest.approx(0.0)
    assert r["porosity_n_bin_3"] == 1.0 and r["porosity_mae_bin_3"] == pytest.approx(0.125)
    assert r["porosity_n_bin_1"] == 0.0 and np.isnan(r["porosity_mae_bin_1"])


def test_the_score_is_a_per_patch_mean_not_a_mean_of_batches(known_answer_case):
    """Split the same data into uneven batches; the answer must not move."""
    logits, label = known_answer_case
    logits = torch.cat([logits, logits, logits])
    label = torch.cat([label, label, label])

    one = du.SegmentationScore()
    one.update(logits, label)

    split = du.SegmentationScore()
    split.update(logits[:5], label[:5])
    split.update(logits[5:], label[5:])

    a, b = one.result(), split.result()
    for k in a:
        if np.isnan(a[k]):
            assert np.isnan(b[k])
        else:
            assert b[k] == pytest.approx(a[k]), k


def test_scoring_before_any_update_is_an_error():
    with pytest.raises(RuntimeError):
        du.SegmentationScore().result()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def test_unet_returns_three_class_logits_at_full_resolution():
    model = du.SegUNet3D(base_channels=4)
    x = torch.rand(2, 1, 16, 16, 16)
    out = model(x)
    assert out.shape == (2, 3, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_unet_trains_a_step_on_cpu():
    """One optimiser step with the campaign's real loss, on a tiny tensor."""
    from poregen.losses.mask import combined_class_loss

    torch.manual_seed(0)
    model = du.SegUNet3D(base_channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(2, 1, 16, 16, 16)
    y = torch.randint(0, 3, (2, 16, 16, 16))
    before = [p.detach().clone() for p in model.parameters()]
    loss = combined_class_loss(model(x), y)["class_total"]
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_class_weights_are_the_real_train_splits_and_shared_by_every_arm():
    w = du.load_class_weights()
    assert len(w) == 3
    assert all(x > 0 for x in w)
    assert w[1] > w[0] and w[2] > w[0], "pore and air must outweigh material"

"""Test that volume-level splits are deterministic, exact, and non-overlapping."""

import pytest

from poregen.dataset.splits import assign_volume_splits


def _make_ids(n: int) -> list[str]:
    return [f"vol_{i:03d}" for i in range(n)]


class TestVolumeSplitCounts:
    """Verify exact counts, no overlap, and determinism."""

    def test_exact_counts(self):
        ids = _make_ids(20)
        splits = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=42)

        train = [k for k, v in splits.items() if v == "train"]
        val = [k for k, v in splits.items() if v == "val"]
        test = [k for k, v in splits.items() if v == "test"]

        assert len(train) == 14
        assert len(val) == 3
        assert len(test) == 3

    def test_no_overlap(self):
        ids = _make_ids(20)
        splits = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=42)

        train = {k for k, v in splits.items() if v == "train"}
        val = {k for k, v in splits.items() if v == "val"}
        test = {k for k, v in splits.items() if v == "test"}

        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_all_assigned_when_exact(self):
        ids = _make_ids(10)
        splits = assign_volume_splits(ids, n_train=6, n_val=2, n_test=2, seed=99)
        assert len(splits) == 10

    def test_unused_volumes(self):
        ids = _make_ids(15)
        splits = assign_volume_splits(ids, n_train=8, n_val=2, n_test=2, seed=7)
        # Only 12 assigned, 3 unused
        assert len(splits) == 12

    def test_deterministic(self):
        ids = _make_ids(20)
        s1 = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=42)
        s2 = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=42)
        assert s1 == s2

    def test_different_seed_gives_different_split(self):
        ids = _make_ids(20)
        s1 = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=42)
        s2 = assign_volume_splits(ids, n_train=14, n_val=3, n_test=3, seed=99)
        # Extremely unlikely (but theoretically possible) to be identical
        assert s1 != s2

    def test_too_many_volumes_raises(self):
        ids = _make_ids(5)
        with pytest.raises(ValueError, match="only 5 available"):
            assign_volume_splits(ids, n_train=4, n_val=3, n_test=3, seed=42)

"""The planted-copy validation, and the patch assembly it depends on.

A detection rate is only meaningful if the planted query really is the shifted
patch it claims to be. A wrong assembly would quietly produce a query that
matches nothing, and the script would then report a LOW detection rate and call
it a detection limit — the most misleading way this could fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from poregen.eval_v4.memorisation import (
    BANK_COVERAGE_NOTE,
    PLANTED_SHIFTS,
    RATIO_THRESHOLD,
    Top2,
    detection_rate,
)

spec = importlib.util.spec_from_file_location(
    "memsens", Path(__file__).resolve().parents[1] / "scripts" / "analysis"
    / "memorisation_sensitivity.py")
memsens = importlib.util.module_from_spec(spec)
sys.modules["memsens"] = memsens
spec.loader.exec_module(memsens)

PATCH = 8          # a toy patch size; the logic is size-independent
STRIDE = 4         # "stride 32" in miniature: half a patch


class FakeStore:
    """A volume cut into overlapping patches, with the store's own accessors."""

    def __init__(self, volume: np.ndarray, patch=PATCH, stride=STRIDE):
        self.volume = volume
        origins = [(z, y, x)
                   for z in range(0, volume.shape[0] - patch + 1, stride)
                   for y in range(0, volume.shape[1] - patch + 1, stride)
                   for x in range(0, volume.shape[2] - patch + 1, stride)]
        self.origins = origins
        self._patches = np.stack([
            volume[z:z + patch, y:y + patch, x:x + patch].ravel() for z, y, x in origins
        ])
        # The BANK is the stride-2*stride subset, as in the real store.
        self.rows = np.array([i for i, (z, y, x) in enumerate(origins)
                              if z % (2 * stride) == 0 and y % (2 * stride) == 0
                              and x % (2 * stride) == 0], np.int64)

    def patch_memmap(self):
        return (self._patches * 255).astype(np.uint8)

    def grey_at(self, pos):
        return self._patches[self.rows[np.asarray(pos)]].astype(np.float32)

    def __len__(self):
        return len(self.rows)


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    vol = rng.uniform(0, 1, (24, 24, 24)).astype(np.float32)
    # Quantise to uint8 levels so the assembly is bit-comparable with the
    # store's own uint8 -> float conversion.
    vol = (np.round(vol * 255) / 255).astype(np.float32)
    store = FakeStore(vol)
    index = {
        "z0": np.array([o[0] for o in store.origins]),
        "y0": np.array([o[1] for o in store.origins]),
        "x0": np.array([o[2] for o in store.origins]),
        "volume_id": np.array(["v"] * len(store.origins)),
        "source_row": np.arange(len(store.origins)),
    }
    return vol, store, index


class TestTheAssembly:

    @pytest.mark.parametrize("shift", [0, 1, 2, 4])
    def test_the_assembled_patch_is_the_volume_at_that_offset(self, setup, shift):
        """The whole validation rests on this being the actual shifted patch."""
        vol, store, index = setup
        pos = 0
        z0, y0, x0 = store.origins[store.rows[pos]]
        got = memsens.assemble_shifted(store, pos, (shift,) * 3, index, PATCH)
        assert got is not None
        want = vol[z0 + shift:z0 + shift + PATCH,
                   y0 + shift:y0 + shift + PATCH,
                   x0 + shift:x0 + shift + PATCH].ravel()
        assert np.allclose(got, want, atol=1e-6)

    def test_a_shift_that_runs_off_the_volume_returns_none(self, setup):
        _, store, index = setup
        pos = len(store) - 1
        assert memsens.assemble_shifted(store, pos, (999,) * 3, index, PATCH) is None

    def test_an_unshifted_assembly_equals_the_stored_patch(self, setup):
        _, store, index = setup
        got = memsens.assemble_shifted(store, 3, (0, 0, 0), index, PATCH)
        assert np.allclose(got, store.grey_at([3])[0], atol=1e-6)


class TestTheDetectionStatistic:

    def test_an_exact_copy_is_detected(self):
        """d1 = 0 with a distinct second neighbour is ratio 0, the clearest
        possible memorisation verdict."""
        acc = Top2(4)
        acc.d1 = np.zeros(4)
        acc.d2 = np.ones(4)
        r = detection_rate(acc)
        assert r["detection_rate"] == 1.0
        assert r["nn_distance_median"] == 0.0

    def test_an_unrelated_query_is_not_detected(self):
        acc = Top2(4)
        acc.d1 = np.full(4, 0.90)
        acc.d2 = np.full(4, 0.95)
        assert detection_rate(acc)["detection_rate"] == 0.0

    def test_the_rate_is_the_fraction_below_the_threshold(self):
        acc = Top2(4)
        acc.d1 = np.array([0.0, 0.1, 0.9, 0.95])
        acc.d2 = np.array([1.0, 1.0, 1.0, 1.0])
        assert detection_rate(acc)["detection_rate"] == 0.5
        assert RATIO_THRESHOLD == pytest.approx(1 / 3)

    def test_absolute_distances_are_reported_beside_the_ratio(self):
        """A ratio cannot tell 'identical' from 'closest of many far rows'."""
        acc = Top2(2)
        acc.d1 = np.array([0.0, 3.0])
        acc.d2 = np.array([1.0, 30.0])
        r = detection_rate(acc)
        # Both are 'memorised' by the ratio; only the distances separate them.
        assert r["detection_rate"] == 1.0
        assert r["nn_distance_max"] == 3.0


class TestTheDocumentedLimit:

    def test_the_shifts_include_the_sanity_floor_and_the_store_stride(self):
        assert 0 in PLANTED_SHIFTS      # must be detected, or the search is broken
        assert 32 in PLANTED_SHIFTS     # in the store, not in the bank

    def test_the_bank_coverage_limit_is_written_down(self):
        assert "stride-64" in BANK_COVERAGE_NOTE
        assert "detection limit" in BANK_COVERAGE_NOTE.lower()

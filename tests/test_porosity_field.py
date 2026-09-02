"""Tests for the coherent porosity-field builder (D32 §4).

Uses the real in-repo T-E sampler and T-D correlation-length artefacts under
``runs/campaigns/01-conditioning-design/``.  All tests run on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from poregen.diffusion.porosity_field import (
    DEFAULT_TD_RESULTS,
    DEFAULT_TE_RESULTS,
    PHI_MAX,
    PHI_MIN,
    build_porosity_field,
    load_corr_lengths_voxels,
    load_sampler,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sampler():
    return load_sampler(REPO_ROOT / DEFAULT_TE_RESULTS)


@pytest.fixture(scope="module")
def corr_lengths():
    return load_corr_lengths_voxels(REPO_ROOT / DEFAULT_TD_RESULTS)


def test_corr_lengths_are_td_volume_mean_removed(corr_lengths):
    cz, cy, cx = corr_lengths
    assert cz == pytest.approx(79.447, abs=0.01)
    assert cy == pytest.approx(413.582, abs=0.01)
    assert cx == pytest.approx(900.948, abs=0.01)


@pytest.mark.parametrize("target", [0.01, 0.03, 0.05])
def test_mean_matches_target(sampler, corr_lengths, target):
    field = build_porosity_field((8, 8, 16), target, sampler, corr_lengths, seed=3)
    assert field.shape == (8, 8, 16)
    assert abs(field.mean() - target) < 1e-6


def test_values_within_training_phi_range(sampler, corr_lengths):
    for target in [0.005, 0.03, 0.10]:
        field = build_porosity_field((8, 8, 16), target, sampler, corr_lengths, seed=5)
        assert field.min() >= PHI_MIN
        assert field.max() <= PHI_MAX


def test_deterministic_per_seed(sampler, corr_lengths):
    a = build_porosity_field((4, 6, 8), 0.03, sampler, corr_lengths, seed=7)
    b = build_porosity_field((4, 6, 8), 0.03, sampler, corr_lengths, seed=7)
    c = build_porosity_field((4, 6, 8), 0.03, sampler, corr_lengths, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def _lag1_corr(field: np.ndarray, axis: int) -> float:
    centred = field - field.mean()
    n = field.shape[axis]
    head = np.take(centred, range(n - 1), axis=axis).ravel()
    tail = np.take(centred, range(1, n), axis=axis).ravel()
    return float(np.corrcoef(head, tail)[0, 1])


def test_anisotropy_x_more_correlated_than_z(sampler, corr_lengths):
    """T-D lengths are strongly anisotropic (x 901 vox vs z 79 vox), so the
    lag-1 correlation on the patch grid must be higher along x than z."""
    field = build_porosity_field((8, 8, 16), 0.03, sampler, corr_lengths, seed=42)
    assert _lag1_corr(field, axis=2) > _lag1_corr(field, axis=0)


def test_marginal_spread_with_tiny_smoothing(sampler):
    """With near-zero correlation lengths the field is the raw T-E marginal
    draw (rescaled/clamped): values must span a wide range, not collapse to
    the target."""
    target = 0.02
    field = build_porosity_field(
        (20, 20, 20), target, sampler, corr_lengths_voxels=(1e-9, 1e-9, 1e-9), seed=1
    )
    assert field.std() > 0.25 * target
    assert field.min() < 0.5 * target
    assert field.max() > 2.0 * target


def _load_generate_volumes_module():
    path = REPO_ROOT / "scripts" / "generate_volumes.py"
    spec = importlib.util.spec_from_file_location("generate_volumes_porfield", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_generate_volumes_coherent_distribution():
    mod = _load_generate_volumes_module()
    gz, gy, gx = 2, 3, 4
    por_map = mod._build_local_por_map(gz, gy, gx, 0.03, "coherent")
    assert set(por_map) == {
        (iz, iy, ix) for iz in range(gz) for iy in range(gy) for ix in range(gx)
    }
    vals = np.array(list(por_map.values()))
    assert vals.mean() == pytest.approx(0.03, abs=1e-6)
    assert vals.min() >= PHI_MIN and vals.max() <= PHI_MAX
    # Deterministic: same porosity level → identical map.
    again = mod._build_local_por_map(gz, gy, gx, 0.03, "coherent")
    assert por_map == again

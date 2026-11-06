"""Known-answer checks for the improved precision/recall in scripts/analysis/precision_recall.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "precision_recall", Path(__file__).resolve().parents[1] / "scripts/analysis/precision_recall.py")
pr = importlib.util.module_from_spec(spec)
sys.modules["precision_recall"] = pr
spec.loader.exec_module(pr)


def test_same_distribution_is_symmetric_and_high():
    # k-NN balls of a finite sample never cover the whole density: two i.i.d.
    # 16-d Gaussian samples of 2 000 points score ~0.78 each way with k = 3.
    # What identifies the metric is symmetry and the gap to a shifted sample.
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2000, 16)), rng.normal(size=(2000, 16))
    r = pr.precision_recall(a, b, k=3)
    assert r["precision"] > 0.7 and r["recall"] > 0.7
    assert abs(r["precision"] - r["recall"]) < 0.05


def test_disjoint_sets_score_zero():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(1000, 16)), rng.normal(size=(1000, 16)) + 50.0
    r = pr.precision_recall(a, b, k=3)
    assert r["precision"] == 0.0 and r["recall"] == 0.0


def test_mode_drop_lowers_recall_not_precision():
    rng = np.random.default_rng(2)
    real = np.concatenate([rng.normal(size=(1000, 16)), rng.normal(size=(1000, 16)) + 20.0])
    gen = rng.normal(size=(2000, 16))          # covers one mode only
    r = pr.precision_recall(real, gen, k=3)
    same = pr.precision_recall(rng.normal(size=(2000, 16)), gen, k=3)
    assert r["precision"] > 0.7                       # every generated point is real-like
    assert abs(r["recall"] - same["recall"] / 2) < 0.08   # half the real modes are covered

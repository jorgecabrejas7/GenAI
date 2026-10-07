"""The author's plateau rule for the rf-8 continuation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("w", REPO / "scripts" / "vae_plateau_watch.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def row(step, kl, coll, std, mu):
    return {"step": step, "kl_per_channel": kl, "kl_collapsed_fraction": coll,
            "std_mean": std, "mu_std": mu}


def test_quantities_are_the_three_the_author_named():
    q = w.quantities(row(1, 0.33, 0.25, 0.759, 0.528))
    assert q["channels_under_floor"] == 2.0
    assert abs(q["std_over_mu_spread"] - 1.4375) < 1e-9


def test_flat_needs_all_three_under_tol_over_the_window():
    rows = [row(30000, 0.20, 0.25, 0.90, 0.40), row(35000, 0.201, 0.25, 0.905, 0.40)]
    assert w.verdict(rows, 5000, 0.02)["flat"]
    rows[-1] = row(35000, 0.201, 0.30, 0.905, 0.40)        # floor count moved 20 %
    assert not w.verdict(rows, 5000, 0.02)["flat"]


def test_no_verdict_until_a_row_is_a_window_older():
    assert w.verdict([row(30000, .2, .25, .9, .4), row(34000, .2, .25, .9, .4)], 5000, 0.02) is None

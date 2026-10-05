"""The convergence gate in scripts/diag_ldm_samples.py, on fake records."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _diag():
    spec = importlib.util.spec_from_file_location(
        "diag_ldm_samples", REPO / "scripts" / "diag_ldm_samples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(degen_unrequested=0.0, direction_ok=True):
    bucket = {"degen": 0.125, "degen_unrequested": degen_unrequested}
    ks = {"asked_lo": 0.005, "asked_hi": 0.05,
          "raw": {"por_lo": 0.005, "por_hi": 0.05, "direction_ok": True},
          "ema": {"por_lo": 0.005, "por_hi": 0.05 if direction_ok else 0.004,
                  "direction_ok": direction_ok}}
    return {"variants": {"ema_ddim50": {"buckets": {"3": dict(bucket), "6": dict(bucket)}}},
            "killswitch": ks}


def test_an_obeyed_pore_free_request_is_not_a_failure():
    mod = _diag()
    # one cell asked phi=0 and delivered nothing; one asked 1e-4 (the floor)
    por = np.array([0.0, 0.0, 0.02, 0.01])
    phi = np.array([0.0, 1e-4, 0.02, 0.012])
    assert mod.degen_unrequested(por, phi) == 0.0
    assert mod.fatal_failures(_entry()) == []


def test_an_empty_cell_against_a_porous_request_fails():
    mod = _diag()
    por = np.array([0.0, 0.02])
    phi = np.array([0.03, 0.02])
    assert mod.degen_unrequested(por, phi) == 0.5
    fails = mod.fatal_failures(_entry(degen_unrequested=0.5))
    assert len(fails) == 2 and all("degen_unrequested" in f for f in fails)


def test_a_saturated_cell_fails_whatever_was_asked():
    mod = _diag()
    assert mod.degen_unrequested(np.array([0.6]), np.array([0.6])) == 1.0


def test_a_reversed_kill_switch_fails():
    mod = _diag()
    fails = mod.fatal_failures(_entry(direction_ok=False))
    assert len(fails) == 1 and "kill switch (ema)" in fails[0]

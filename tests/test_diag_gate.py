"""The convergence report and gate in scripts/diag_ldm_samples.py, on fake records.

The script REPORTS every cell and the kill switch, pass or fail. It exits
non-zero only for a reversed kill switch or a non-finite value; an unrequested
degenerate cell is reported loudly but does not stop the chain.
"""
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


def _entry(mod, por=(0.0, 0.02, 0.01), phi=(0.0, 0.02, 0.012), direction_ok=True,
           killswitch=True):
    bucket = {"degen": 0.125, "std_ratio": 1.0}
    ks = {"asked_lo": 0.005, "asked_hi": 0.05, "bucket": "6", "n": 4,
          "raw": {"por_lo": 0.005, "por_hi": 0.05, "separation": 0.045,
                  "direction_ok": True},
          "ema": {"por_lo": 0.005, "por_hi": 0.05 if direction_ok else 0.004,
                  "separation": 0.045 if direction_ok else -0.001,
                  "direction_ok": direction_ok}}
    cells = mod.cell_rows("ema_ddim50", "6", np.array(por), np.full(len(por), 0.01),
                          np.array(phi))
    return {"step": 1000,
            "variants": {"ema_ddim50": {"buckets": {"6": dict(bucket)}}},
            "killswitch": ks if killswitch else None, "cells": cells}


def test_an_obeyed_pore_free_request_is_not_a_failure():
    mod = _diag()
    # one cell asked phi=0 and delivered nothing; one asked 1e-4 (the floor)
    por = np.array([0.0, 0.0, 0.02, 0.01])
    phi = np.array([0.0, 1e-4, 0.02, 0.012])
    assert mod.degen_unrequested(por, phi) == 0.0
    e = _entry(mod, por=por, phi=phi)
    assert mod.fatal_failures(e) == []
    assert mod.unrequested_cells(e) == []
    assert [c["requested_pore_free"] for c in e["cells"]] == [True, True, False, False]
    assert [c["degenerate_why"] for c in e["cells"]] == ["pore < 1e-4", "pore < 1e-4",
                                                         None, None]


def test_an_unrequested_degenerate_cell_is_reported_but_not_fatal():
    mod = _diag()
    e = _entry(mod, por=(0.0, 0.6, 0.02), phi=(0.03, 0.6, 0.02))
    unreq = mod.unrequested_cells(e)
    assert [(c["cell"], c["degenerate_why"]) for c in unreq] == [
        (0, "pore < 1e-4"), (1, "pore > 0.5")]
    assert mod.fatal_failures(e) == []
    md = mod.findings_markdown(e, "run-x")
    assert "**2 are not explained by the request**" in md
    assert "UNREQUESTED degenerate cells" in md


def test_a_reversed_kill_switch_fails():
    mod = _diag()
    fails = mod.fatal_failures(_entry(mod, direction_ok=False))
    assert len(fails) == 1 and "kill switch (ema) reversed" in fails[0]


def test_a_non_finite_value_fails():
    mod = _diag()
    e = _entry(mod)
    e["variants"]["ema_ddim50"]["buckets"]["6"]["std_ratio"] = float("nan")
    fails = mod.fatal_failures(e)
    assert len(fails) == 1 and "non-finite" in fails[0] and "std_ratio" in fails[0]
    e = _entry(mod, por=(float("inf"),), phi=(0.02,))
    assert any("cells[0].delivered_phi" in f for f in mod.fatal_failures(e))


def test_the_findings_always_carry_the_kill_switch_and_every_cell():
    mod = _diag()
    e = _entry(mod)
    md = mod.findings_markdown(e, "run-x")
    assert "Verdict: **PASS**" in md
    assert "## Kill switch" in md and "+0.0450" in md
    assert md.count("| ema_ddim50 | 6 |") == 3
    md = mod.findings_markdown(_entry(mod, direction_ok=False), "run-x")
    assert "Verdict: **FATAL**" in md and "-0.0010" in md
    assert "Not run" in mod.findings_markdown(_entry(mod, killswitch=False), "run-x")

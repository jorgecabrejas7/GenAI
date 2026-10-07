#!/usr/bin/env python
"""Stop a VAE run when its latent has stopped moving. (CPU, polls metrics.jsonl.)

The author's rule for continuing rf-8 on split_v4 (2026-10-07): stop early
only when three latent quantities have all been flat — less than ``--tol``
relative change over ``--window`` steps:

* KL per channel                 (``kl_per_channel``, nats per latent channel)
* channels under the free-bits floor  (8 × ``kl_collapsed_fraction``)
* posterior std over mean spread (``std_mean / mu_std``)

Every new row of the chosen split is logged to ``<run>/plateau_watch.jsonl``
with the three values and the verdict. When the rule holds at step s, the
watcher waits for the next ``--save-every`` checkpoint after s to land and
then sends SIGTERM to the trainer, so the run ends on a saved checkpoint.

Rows: ``val_full`` (whole val split, every full_val_every steps) or ``val``
(``val_batches`` batches every eval_every). A flat test compares the row at s
with the newest row at least ``--window`` steps older.

    python scripts/vae_plateau_watch.py --run runs/vae/<run> --pid <trainer pid>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

N_CHANNELS = 8


def quantities(row: dict) -> dict[str, float]:
    return {
        "kl_per_channel": float(row["kl_per_channel"]),
        "channels_under_floor": N_CHANNELS * float(row["kl_collapsed_fraction"]),
        "std_over_mu_spread": float(row["std_mean"]) / float(row["mu_std"]),
    }


def flat(now: dict[str, float], then: dict[str, float], tol: float) -> dict[str, float]:
    """Relative change of each quantity; the rule holds when all are < tol."""
    return {k: abs(now[k] - then[k]) / max(abs(then[k]), 1e-12) for k in now}


def verdict(rows: list[dict], window: int, tol: float) -> dict | None:
    """The rule at the newest row, or None when no row is ``window`` older."""
    s = rows[-1]["step"]
    older = [r for r in rows if r["step"] <= s - window]
    if not older:
        return None
    then = older[-1]
    change = flat(quantities(rows[-1]), quantities(then), tol)
    return {"step": s, "against_step": then["step"], "change": change,
            "flat": all(c < tol for c in change.values())}


def read_rows(metrics: Path, split: str, min_step: int) -> list[dict]:
    rows = []
    with open(metrics) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("split") == split and r["step"] >= min_step:
                    rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--rows", choices=("val_full", "val"), default="val_full")
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--from-step", type=int, default=0,
                    help="ignore rows before this step (the resume point)")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args()

    log = a.run / "plateau_watch.jsonl"
    # Every validation row is LOGGED (both splits, so the trend is visible at
    # eval_every); only rows of --rows DECIDE.
    logged = {"val": 0, "val_full": 0}
    seen = 0
    while True:
        try:
            os.kill(a.pid, 0)
        except ProcessLookupError:
            return 0
        other = "val" if a.rows == "val_full" else "val_full"
        orows = read_rows(a.run / "metrics.jsonl", other, a.from_step)
        with open(log, "a") as fh:
            for r in orows[logged[other]:]:
                fh.write(json.dumps({"step": r["step"], "rows": other,
                                     **quantities(r), "verdict": "not deciding"}) + "\n")
        logged[other] = len(orows)
        rows = read_rows(a.run / "metrics.jsonl", a.rows, a.from_step)
        for i in range(seen, len(rows)):
            v = verdict(rows[: i + 1], a.window, a.tol)
            entry = {"step": rows[i]["step"], "rows": a.rows, **quantities(rows[i]),
                     "verdict": v}
            with open(log, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            if v is not None and v["flat"]:
                stop_at = (v["step"] // a.save_every + 1) * a.save_every
                ckpt = a.run / f"{a.run.name}_step{stop_at:08d}.ckpt"
                while not ckpt.exists():
                    time.sleep(10)
                time.sleep(30)          # the file is written; let latest.ckpt land too
                with open(log, "a") as fh:
                    fh.write(json.dumps({"stop": True, "flat_at": v["step"],
                                         "checkpoint": str(ckpt),
                                         "signal": "SIGTERM", "pid": a.pid}) + "\n")
                os.kill(a.pid, signal.SIGTERM)
                return 0
        seen = len(rows)
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())

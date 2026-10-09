#!/usr/bin/env python
"""Keep a VAE training while its collapsed channels keep reviving. (CPU.)

The author's rule (2026-10-09): "if channels under the floor keep dropping we
keep training, if other metrics don't worsen."

Phase 1 — the trainer (``--pid``) runs with its own early stopping. When it
exits on an ``early_stopping`` event, the run is resumed from latest.ckpt with
``--no-early-stopping --total-steps <cap>``. Any other exit ends the watch.

Phase 2 — every full-validation row after the resume is read, against ALL the
run's full-validation rows so far:

* STOP when channels under the free-bits floor (z_channels x
  ``kl_collapsed_fraction``) have not dropped over the last two full
  validations (now >= the row two before);
* STOP when, in two consecutive full rows, ``dice_pore`` or
  ``sharpness_recon_over_gt`` is more than 0.01 below its best so far, or
  ``xct_loss`` more than 0.001 above its best so far;
* otherwise continue (to the cap).

A stop waits for the next ``--save-every`` checkpoint and SIGTERMs the trainer,
so the run ends on a saved checkpoint. Every row and decision goes to
``<run>/floor_watch.jsonl``, with KL/channel and std_mean/mu_std beside them.

    python scripts/vae_floor_watch.py --run runs/vae/<run> --pid <trainer pid>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

DICE_TOL = 0.01
SHARP_TOL = 0.01
XCT_TOL = 0.001


def full_rows(metrics: Path) -> list[dict]:
    rows = []
    with open(metrics) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("split") == "val_full":
                    rows.append(r)
    return rows


def quantities(r: dict, z: int) -> dict:
    return {"channels_under_floor": z * float(r["kl_collapsed_fraction"]),
            "kl_per_channel": float(r["kl_per_channel"]),
            "std_over_mu_spread": float(r["std_mean"]) / float(r["mu_std"]),
            "dice_pore": float(r["dice_pore"]),
            "xct_loss": float(r["xct_loss"]),
            "sharpness": float(r["sharpness_recon_over_gt"])}


def worse(q: dict, best: dict) -> list[str]:
    """Which of the three guarded metrics is past its tolerance vs the best."""
    out = []
    if q["dice_pore"] < best["dice_pore"] - DICE_TOL:
        out.append("dice_pore")
    if q["sharpness"] < best["sharpness"] - SHARP_TOL:
        out.append("sharpness")
    if q["xct_loss"] > best["xct_loss"] + XCT_TOL:
        out.append("xct_loss")
    return out


def decide(rows: list[dict], z: int) -> dict:
    """The rule at the newest full row, given every full row of the run."""
    qs = [quantities(r, z) for r in rows]
    q = qs[-1]
    reasons = []
    if len(qs) >= 3 and q["channels_under_floor"] >= qs[-3]["channels_under_floor"]:
        reasons.append("channels under the floor did not drop over the last two full "
                       f"validations ({qs[-3]['channels_under_floor']:.2f} -> "
                       f"{q['channels_under_floor']:.2f})")
    if len(qs) >= 2:
        best_before = {"dice_pore": max(x["dice_pore"] for x in qs[:-2] or qs[:1]),
                       "sharpness": max(x["sharpness"] for x in qs[:-2] or qs[:1]),
                       "xct_loss": min(x["xct_loss"] for x in qs[:-2] or qs[:1])}
        a, b = worse(qs[-2], best_before), worse(q, best_before)
        both = sorted(set(a) & set(b))
        if both:
            reasons.append(f"worse than the best so far in two consecutive full rows: {both}")
    return {"step": rows[-1]["step"], **q, "stop": bool(reasons), "reasons": reasons}


def wait_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--cap", type=int, default=71690)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--poll", type=float, default=120.0)
    a = ap.parse_args()
    run = a.run.resolve()
    repo = Path(__file__).resolve().parents[1]
    log = run / "floor_watch.jsonl"
    z = int(yaml.safe_load((run / "resolved_config.yaml").read_text())["model"]["z_channels"])

    def note(**kw):
        with open(log, "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}) + "\n")

    # ── phase 1: until the trainer's own early stop ─────────────────────────
    note(phase=1, watching_pid=a.pid, z_channels=z)
    wait_pid(a.pid)
    events = [json.loads(l) for l in open(run / "metrics.jsonl")
              if '"split": "event"' in l]
    if not events or events[-1].get("event") != "early_stopping":
        note(phase=1, ended="the trainer exited without an early_stopping event; no resume")
        return 0
    stop_step = int(events[-1]["step"])
    cmd = [sys.executable, "scripts/train_vae.py", "resume", str(run), "latest.ckpt",
           "--no-early-stopping", "--total-steps", str(a.cap)]
    out = open(run.parent.parent / "campaigns" / "chain_logs" / f"floor_resume_{run.name[:12]}.log", "a") \
        if (run.parent.parent / "campaigns" / "chain_logs").is_dir() else open(run / "floor_resume.log", "a")
    proc = subprocess.Popen(["choom", "-n", "1000", "--", *cmd], cwd=repo, stdout=out,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "POREGEN_SPLIT": os.environ.get("POREGEN_SPLIT", "split_v4")})
    subprocess.Popen(["bash", "scripts/analysis/host_pressure_log.sh", str(proc.pid), str(run)],
                     cwd=repo, stdout=open(run / "floor_resume_pressure.tsv", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    note(phase=2, early_stopped_at=stop_step, resumed_pid=proc.pid, cmd=" ".join(cmd))

    # ── phase 2: every full validation after the resume ─────────────────────
    seen = len(full_rows(run / "metrics.jsonl"))
    while proc.poll() is None:
        rows = full_rows(run / "metrics.jsonl")
        for i in range(seen, len(rows)):
            if rows[i]["step"] <= stop_step:
                continue
            d = decide(rows[: i + 1], z)
            note(phase=2, **d)
            if d["stop"]:
                target = (d["step"] // a.save_every + 1) * a.save_every
                ckpt = run / f"{run.name}_step{target:08d}.ckpt"
                while not ckpt.exists() and proc.poll() is None:
                    time.sleep(10)
                time.sleep(30)
                proc.send_signal(signal.SIGTERM)
                note(phase=2, stopped=True, checkpoint=str(ckpt), signal="SIGTERM", pid=proc.pid)
                return 0
        seen = len(rows)
        time.sleep(a.poll)
    note(phase=2, ended=f"the resumed trainer exited rc={proc.returncode} (cap or fault)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

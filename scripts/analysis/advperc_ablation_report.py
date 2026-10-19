#!/usr/bin/env python
"""Reduction-factor ablation table of the advperc-ldm rungs (campaign 33). (CPU, no model.)

Reads only artefacts that the trainer and the queue checks already wrote:
  - <run>/metrics.jsonl: the trainer's whole-split evaluations. Every rung has a
    val_full at step 12658 (discriminator off, the last full eval before
    disc_start 15000) and a val_full + test_full at its last step (disc on).
    These are the like-for-like comparison: same steps, same code, every rung.
  - <run>/log.jsonl: train rows, for the discriminator state over the last
    1000 train steps (median disc_loss, logit margin, accuracies, lambda).
  - <run>/summary.json: stop step.
  - campaign 09 rung report and calibration probe on best.ckpt (whole val and
    test; the checkpoint step differs between rungs, see the table).
  - chain_logs/check_<tag>_l1.json: vae_val_l1 harness on best.ckpt
    (640 val patches, texture correlation, sharpness).

Writes <out>/table.md, table.csv and four figures:
  fig_metrics_vs_rf.png   whole-split metrics against the reduction factor,
                          disc off (val, 12658) vs disc on (val and test, last step)
  fig_change_on_off.png   relative change disc on vs off, val, per rung and metric
  fig_porosity_bins.png   porosity MAE per true-porosity bin, val, off vs on
  fig_disc_state.png      discriminator loss and margin over training per rung

Headline (2026-10-19, campaign 33 table.md): the disc phase RAISES (worsens) val grey L1
on every rung (rf-32 0.0276 -> 0.0518) and raises val total on rf-2, rf-16, rf-32.
The disc left chance only on rf-32 and rf-64.

Usage:
    python scripts/analysis/advperc_ablation_report.py --out runs/campaigns/33-advperc-ldm/ablation \\
        --run "16=runs/vae/<run>" --run "8=runs/remote/.../<run>" ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "runs/campaigns/09-r08-latent-sweep"
CHAIN = REPO / "runs/campaigns/chain_logs"
OFF_STEP = 12658
# metric key, label, lower-is-better
METRICS = [("total", "val total loss", True), ("xct_loss", "grey L1", True),
           ("lpips", "LPIPS", True), ("dice_pore", "pore Dice", False),
           ("dice_air", "air Dice", False), ("porosity_mae", "porosity MAE", True),
           ("sharpness_recon_over_gt", "sharpness recon/GT", False)]
BINS = ["<1%", "1-3%", "3-6%", ">=6%"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def rung(rf: int, run: Path) -> dict:
    rows = read_jsonl(run / "metrics.jsonl")
    full = {(r["step"], r["split"]): r for r in rows if r["split"].endswith("_full")}
    last = max(s for s, _ in full)
    off, on_val, on_test = full[(OFF_STEP, "val_full")], full[(last, "val_full")], full[(last, "test_full")]
    train = [r for r in read_jsonl(run / "log.jsonl") if r.get("split") == "train"]
    tail = [r for r in train if r["step"] > train[-1]["step"] - 1000 and "disc_loss" in r]
    med = lambda k: float(np.median([r[k] for r in tail if k in r])) if tail else float("nan")
    tag = f"advperc-ldm-rf{rf}"
    rep = json.loads((SWEEP / f"r08_{tag}-dsv4/results.json").read_text())
    cal = json.loads((SWEEP / f"calibration_probe_r08_{tag}-dsv4/results.json").read_text())
    l1 = json.loads((CHAIN / f"check_{tag}_l1.json").read_text())
    curve = [(r["step"], r.get("disc_loss"), r.get("disc_margin")) for r in train if "disc_loss" in r]
    return dict(rf=rf, run=run, stop=last - 1, off=off, on_val=on_val, on_test=on_test,
                disc=dict(loss=med("disc_loss"), margin=med("disc_margin"), acc_real=med("disc_acc_real"),
                          acc_fake=med("disc_acc_fake"), lam=med("disc_lambda")),
                rep=rep, cal=cal, l1=l1, curve=curve)


def fmt(v: float) -> str:
    if v != v:
        return "–"
    a = abs(v)
    return f"{v:.5f}" if a < 0.01 else f"{v:.4f}" if a < 1 else f"{v:.3f}"


def write_tables(R: list[dict], out: Path) -> None:
    L = ["# advperc-ldm reduction-factor ablation (campaign 33)", "",
         "Source of every number in tables A–C: the trainer's whole-split evaluations in each run's metrics.jsonl. "
         f"OFF = val_full at step {OFF_STEP} (before disc_start 15000). ON = val_full / test_full at the last step. "
         "No test_full exists for OFF.", ""]
    L += ["## A. Whole val split, disc OFF (step 12658) → disc ON (last step)", "",
          "| rung | latent | stop step | " + " | ".join(m[1] for m in METRICS) + " |",
          "|---|---|---|" + "---|" * len(METRICS)]
    for r in R:
        cells = [f"{fmt(r['off'][k])} → {fmt(r['on_val'][k])}" for k, _, _ in METRICS]
        L.append(f"| rf-{r['rf']} | 16³×{64 // r['rf']} | {r['stop']} | " + " | ".join(cells) + " |")
    L += ["", "## B. Whole test split, disc ON (last step)", "",
          "| rung | " + " | ".join(m[1] for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for r in R:
        L.append(f"| rf-{r['rf']} | " + " | ".join(fmt(r['on_test'][k]) for k, _, _ in METRICS) + " |")
    L += ["", "## C. Porosity MAE per true-porosity bin, whole val, OFF → ON", "",
          "| rung | " + " | ".join(BINS) + " |", "|---|" + "---|" * 4]
    for r in R:
        L.append(f"| rf-{r['rf']} | " + " | ".join(
            f"{fmt(r['off'][f'porosity_mae_bin_{i}'])} → {fmt(r['on_val'][f'porosity_mae_bin_{i}'])}" for i in range(4)) + " |")
    L += ["", "## D. Discriminator state, median of the last 1000 train steps (log.jsonl)", "",
          "Hinge loss at chance = 1.0, margin at chance = 0.", "",
          "| rung | disc loss | logit margin (real − fake) | acc real | acc fake | adaptive λ |", "|---|---|---|---|---|---|"]
    for r in R:
        d = r["disc"]
        L.append(f"| rf-{r['rf']} | {d['loss']:.3f} | {d['margin']:+.4f} | {d['acc_real']:.2f} | {d['acc_fake']:.2f} | {d['lam']:.3f} |")
    L += ["", "## E. Queue checks on best.ckpt (step differs per rung)", "",
          "Rung report (r08_rung_report.py, whole val and test, argmax), calibration probe "
          "(r08_calibration_probe.py, 6204 stratified patches, argmax), vae_val_l1 harness (640 val patches).", "",
          "| rung | best.ckpt step | disc | val pore Dice | test pore Dice | val φ MAE | test φ MAE | calib worst-bin MAE | harness L1 | texture corr | sharpness |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        v, t = r["rep"]["splits"]["val"], r["rep"]["splits"]["test"]
        worst = fmt(r["cal"]["summary"]["argmax"]["worst_bin_mae"])
        st = r["rep"]["step"]
        L.append(f"| rf-{r['rf']} | {st} | {'on' if st > 15000 else 'off'} | {v['dice_pore']:.4f} | {t['dice_pore']:.4f} | "
                 f"{fmt(v['porosity_mae'])} | {fmt(t['porosity_mae'])} | {worst} | {fmt(r['l1']['l1'])} | "
                 f"{r['l1']['texture_corr']:.3f} | {r['l1']['sharpness']:.3f} |")
    (out / "table.md").write_text("\n".join(L) + "\n")
    with open(out / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "phase", "split", "step"] + [k for k, _, _ in METRICS] + [f"porosity_mae_bin_{i}" for i in range(4)])
        for r in R:
            for ph, sp, row in (("off", "val", r["off"]), ("on", "val", r["on_val"]), ("on", "test", r["on_test"])):
                w.writerow([r["rf"], ph, sp, row["step"]] + [row[k] for k, _, _ in METRICS] + [row[f"porosity_mae_bin_{i}"] for i in range(4)])


def plots(R: list[dict], out: Path) -> None:
    rf = [r["rf"] for r in R]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5)); ax = ax.ravel()
    for a, (k, lab, low) in zip(ax, METRICS):
        a.plot(rf, [r["off"][k] for r in R], "o-", c="tab:blue", label=f"OFF val (step {OFF_STEP})")
        a.plot(rf, [r["on_val"][k] for r in R], "s-", c="tab:red", label="ON val (last step)")
        a.plot(rf, [r["on_test"][k] for r in R], "^--", c="tab:red", alpha=.5, label="ON test (last step)")
        a.set_xscale("log", base=2); a.set_xticks(rf, [f"rf-{x}" for x in rf]); a.set_title(f"{lab} ({'lower' if low else 'higher'} is better)")
        a.grid(alpha=.3)
        if k == "porosity_mae":
            a.set_yscale("log")
    ax[-1].axis("off"); ax[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("advperc-ldm: whole-split metrics vs reduction factor (metrics.jsonl full evals)")
    fig.tight_layout(); fig.savefig(out / "fig_metrics_vs_rf.png", dpi=130); plt.close(fig)

    fig, a = plt.subplots(figsize=(12, 4.8))
    w = 0.8 / len(METRICS)
    for i, (k, lab, low) in enumerate(METRICS):
        # positive = better, so every bar reads the same way
        ch = [(r["on_val"][k] - r["off"][k]) / abs(r["off"][k]) * 100 * (-1 if low else 1) for r in R]
        a.bar(np.arange(len(R)) + (i - len(METRICS) / 2 + .5) * w, ch, w, label=lab)
    a.axhline(0, c="k", lw=.8); a.set_xticks(range(len(R)), [f"rf-{x}" for x in rf])
    a.set_ylabel("% change ON vs OFF, val\n(positive = better)"); a.legend(ncol=4, fontsize=8); a.grid(axis="y", alpha=.3)
    a.set_title(f"Effect of the discriminator phase per rung (val_full last step vs val_full step {OFF_STEP})")
    fig.tight_layout(); fig.savefig(out / "fig_change_on_off.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=False)
    for i, a in enumerate(ax):
        a.plot(rf, [r["off"][f"porosity_mae_bin_{i}"] for r in R], "o-", c="tab:blue", label="OFF val")
        a.plot(rf, [r["on_val"][f"porosity_mae_bin_{i}"] for r in R], "s-", c="tab:red", label="ON val")
        a.axhline(0.005, c="k", ls=":", lw=1, label="gate 0.005")
        n = int(R[0]["off"][f"porosity_n_bin_{i}"])
        a.set_xscale("log", base=2); a.set_xticks(rf, [f"rf-{x}" for x in rf], fontsize=7); a.set_yscale("log")
        a.set_title(f"true φ {BINS[i]} (n={n})"); a.grid(alpha=.3)
    ax[0].set_ylabel("porosity MAE"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_porosity_bins.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for r in R:
        s = np.array([c[0] for c in r["curve"]]); l = np.array([c[1] for c in r["curve"]], float)
        m = np.array([c[2] if c[2] is not None else np.nan for c in r["curve"]], float)
        k = 50
        sm = lambda y: np.convolve(y, np.ones(k) / k, "valid")
        ax[0].plot(s[k - 1:], sm(l), label=f"rf-{r['rf']}"); ax[1].plot(s[k - 1:], sm(m), label=f"rf-{r['rf']}")
    ax[0].axhline(1, c="k", ls=":"); ax[1].axhline(0, c="k", ls=":")
    ax[0].set_title("disc hinge loss (1.0 = chance), 50-step mean"); ax[1].set_title("logit margin real − fake (0 = chance), 50-step mean")
    for a in ax:
        a.set_xlabel("train step"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_disc_state.png", dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"rf=run_dir"')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = sorted((rung(int(s.split("=", 1)[0]), REPO / s.split("=", 1)[1]) for s in a.run), key=lambda r: r["rf"])
    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True)
    write_tables(R, out); plots(R, out)
    print((out / "table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

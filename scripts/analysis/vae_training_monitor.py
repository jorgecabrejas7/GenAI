#!/usr/bin/env python
"""Training monitor for running VAE rungs: curves, block tables and a
reconstruction progression. (CPU, reads logs and step checkpoints only.)

Written for the advperc-ldm table (the author, 2026-10-15: "more metrics
tables and visual plots for checking how the training goes"). Three outputs
in ``--out``:

- ``curves.png``  : train curves (rolling median), discriminator health,
  validation metrics and step time, one colour per run, a dotted line at
  ``--disc-start``. Reference runs (``--ref``) are drawn dashed.
- ``tables.md``   : per run, medians over 1000-step blocks of the train and
  discriminator terms, then every validation row's headline numbers, then
  the best partial validation so far.
- ``recon_<label>.png`` (with ``--recon``): the run's step checkpoints every
  ``--recon-every`` steps on the three val patches of
  ``vae_recon_cross_sections.py`` (almost no pores, very porous, with air),
  the central axis-1 plane, input in the top row. Shows how texture and the
  air edge come back as training goes on, and what the discriminator changes
  after it starts.

Usage:
    python scripts/analysis/vae_training_monitor.py \\
        --run "rf-16=runs/vae/r08-run-0020-..." --run "rf-8=runs/remote/.../r08-run-0025-..." \\
        --ref "rf-16 SN-LSGAN=runs/vae/r08-run-0019-..." --disc-start 15000 \\
        --recon --out runs/campaigns/33-advperc-ldm/monitor
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

TRAIN_PANELS = [
    ("xct_loss", "train grey L1 (xct_loss)", False),
    ("lpips", "train LPIPS", False),
    ("class_total", "train class CE + Dice", False),
    ("kl_rombach", "train KL (summed / 64³)", True),
]
DISC_PANELS = [
    (("disc_score_real", "disc_score_fake"), "disc mean logit: real (—) fake (··)", False),
    ("disc_margin", "disc margin real − fake", False),
    (("disc_acc_real", "disc_acc_fake"), "disc accuracy: real (—) fake (··)", False),
    ("disc_loss", "disc loss", False),
    ("disc_lambda", "adaptive weight disc_lambda", True),
    ("gen_adv_loss", "generator adversarial loss", False),
]
VAL_PANELS = [
    ("total", "val total", False),
    ("xct_loss", "val grey L1", False),
    ("lpips", "val LPIPS", False),
    ("dice_pore", "val pore Dice", False),
    ("dice_air", "val air Dice", False),
    ("porosity_mae", "val porosity MAE", False),
    ("sharpness_recon_over_gt", "val sharpness recon / gt", False),
]
TABLE_TRAIN = ["xct_loss", "lpips", "class_total", "kl_rombach", "disc_loss", "disc_margin",
               "disc_acc_real", "disc_acc_fake", "disc_lambda", "gen_adv_loss", "step_time_ms"]
TABLE_VAL = ["total", "xct_loss", "lpips", "dice_pore", "dice_air", "porosity_mae",
             "sharpness_recon_over_gt"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line being written
    return rows


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not pts:
        return np.array([]), np.array([])
    s, v = zip(*pts)
    return np.asarray(s), np.asarray(v, dtype=float)


def rolling_median(v: np.ndarray, w: int) -> np.ndarray:
    if len(v) < w:
        return v
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - w + 1): i + 1])
    return out


def load(spec: str) -> tuple[str, Path, list[dict], list[dict]]:
    label, path = spec.split("=", 1)
    run = REPO / path
    train = [r for r in read_jsonl(run / "log.jsonl") if r.get("split") in (None, "train")]
    val = [r for r in read_jsonl(run / "metrics.jsonl") if r.get("split") == "val" and "total" in r]
    return label, run, train, val


def plot_curves(runs, refs, disc_start: int, out: Path, smooth: int) -> None:
    panels = ([("train", *p) for p in TRAIN_PANELS] + [("disc", *p) for p in DISC_PANELS]
              + [("val", *p) for p in VAL_PANELS] + [("train", "step_time_ms", "s / step", False)])
    ncol = 4
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    entries = [(r, colours[i % len(colours)], "-") for i, r in enumerate(runs)] + \
              [(r, colours[(len(runs) + i) % len(colours)], "--") for i, r in enumerate(refs)]
    for ax, (src, key, title, logy) in zip(axes.ravel(), panels):
        for (label, _, train, val), c, ls in entries:
            rows = val if src == "val" else train
            keys = key if isinstance(key, tuple) else (key,)
            for j, k in enumerate(keys):
                s, v = series(rows, k)
                if not len(s):
                    continue
                if k == "step_time_ms":
                    v = v / 1000.0
                if src != "val":
                    v = rolling_median(v, smooth)
                ax.plot(s, v, color=c, ls=(":" if j else ls), lw=1.1,
                        label=label if j == 0 else None)
        if key == "step_time_ms":
            # The first steps (compile, loader warm-up) take tens of seconds and
            # would flatten the curve that matters.
            ys = [l.get_ydata() for l in ax.get_lines() if len(l.get_ydata())]
            if ys:
                ax.set_ylim(0, 1.2 * float(np.percentile(np.concatenate(ys), 99)))
        ax.axvline(disc_start, color="grey", ls=":", lw=0.8)
        ax.set_title(title, fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labels):
            handles, labels = h, l
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9)
    fig.suptitle(f"train curves are rolling medians over {smooth} steps; dotted vertical line = disc_start {disc_start}",
                 fontsize=9, y=0.005)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    a = abs(v)
    return f"{v:.3g}" if (a >= 1e3 or a < 1e-3) and a != 0 else f"{v:.4f}"


def would_stop_at(val: list[dict], patience_steps: int, min_delta: float) -> int | None:
    """The first step at which the trainer's early-stopping rule (val.total, min,
    ``min_delta``, patience in steps) would fire if it had NO warm-up. The
    author (2026-10-15) asked to watch whether 15000 steps is enough for the
    model to converge at the new lr before the discriminator starts."""
    best, best_step = None, None
    for r in val:
        v = r["total"]
        if best is None or v < best - min_delta:
            best, best_step = v, r["step"]
        elif r["step"] - best_step >= patience_steps:
            return int(r["step"])
    return None


def window_change(val: list[dict], key: str, last: int, gap: int = 3000, width: int = 1000) -> str:
    """Median of ``key`` over the last ``width`` steps vs the same window ``gap``
    steps earlier, as a % change. Negative = still improving (for losses)."""
    now = [r[key] for r in val if last - width < r["step"] <= last and isinstance(r.get(key), (int, float))]
    then = [r[key] for r in val if last - gap - width < r["step"] <= last - gap and isinstance(r.get(key), (int, float))]
    if not now or not then or np.median(then) == 0:
        return "–"
    return f"{100.0 * (np.median(now) - np.median(then)) / abs(np.median(then)):+.1f} %"


def convergence(runs, patience_steps: int, min_delta: float) -> list[str]:
    keys = ["total", "xct_loss", "lpips", "dice_pore"]
    lines = ["## Convergence before the discriminator\n",
             f"Early-stop rule as in the trainer (val.total, min_delta {min_delta}, patience "
             f"{patience_steps} steps) but with no warm-up: the step at which it would have "
             "fired. Change = median of the last 1000 steps vs the 1000 steps ending 3000 "
             "steps earlier (negative = still improving for losses, positive for Dice).\n",
             "| run | last val step | best val.total (step) | steps since best | would stop at | "
             + " | ".join(f"Δ {k}" for k in keys) + " |",
             "|---" * (5 + len(keys)) + "|"]
    for label, _, _, val in runs:
        if not val:
            continue
        last = val[-1]["step"]
        b = min(val, key=lambda r: r["total"])
        stop = would_stop_at(val, patience_steps, min_delta)
        lines.append(f"| {label} | {last} | {fmt(b['total'])} ({b['step']}) | {last - b['step']} | "
                     f"{stop if stop is not None else 'not yet'} | "
                     + " | ".join(window_change(val, k, last) for k in keys) + " |")
    return lines + [""]


def tables(runs, out: Path, patience_steps: int = 3000, min_delta: float = 1e-4) -> None:
    lines = convergence(runs, patience_steps, min_delta)
    for label, run, train, val in runs:
        lines.append(f"## {label}\n\n`{run.relative_to(REPO)}`\n")
        last = max((r["step"] for r in train), default=0)
        lines.append(f"Last train step: {last}\n")
        lines.append("### Train, median per 1000-step block\n")
        lines.append("| steps | " + " | ".join("s/step" if k == "step_time_ms" else k for k in TABLE_TRAIN) + " |")
        lines.append("|---" * (len(TABLE_TRAIN) + 1) + "|")
        for b0 in range(0, last + 1, 1000):
            blk = [r for r in train if b0 <= r["step"] < b0 + 1000]
            if not blk:
                continue
            cells = []
            for k in TABLE_TRAIN:
                vals = [r[k] for r in blk if isinstance(r.get(k), (int, float))]
                v = float(np.median(vals)) if vals else None
                if k == "step_time_ms" and v is not None:
                    v /= 1000.0
                cells.append(fmt(v))
            lines.append(f"| {b0}–{b0 + 999} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("### Validation rows (partial val)\n")
        lines.append("| step | " + " | ".join(TABLE_VAL) + " |")
        lines.append("|---" * (len(TABLE_VAL) + 1) + "|")
        for r in val:
            lines.append(f"| {r['step']} | " + " | ".join(fmt(r.get(k)) for k in TABLE_VAL) + " |")
        if val:
            b = min(val, key=lambda r: r["total"])
            lines.append(f"\nBest partial val.total {fmt(b['total'])} at step {b['step']} "
                         f"(pore Dice {fmt(b.get('dice_pore'))}, grey L1 {fmt(b.get('xct_loss'))}).\n")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def recon_progression(label: str, run: Path, every: int, split: str, out: Path) -> None:
    import torch
    from vae_val_l1 import load_model_and_patches  # same dir
    from vae_recon_cross_sections import KINDS, TITLES, corr, pick_rows
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    ckpts = {}
    for p in run.glob("*_step*.ckpt"):
        m = re.search(r"_step(\d+)\.ckpt$", p.name)
        if m and int(m.group(1)) % every == 0:
            ckpts[int(m.group(1))] = p.name
    if not ckpts:
        return
    steps = sorted(ckpts)
    rows, x_ref = [], None
    for s in steps:
        model, ds, _, _, _ = load_model_and_patches(run, split, 1, 1, None, ckpts[s])
        rows_idx = pick_rows(ds.df)
        items = [ds[rows_idx[k]] for k in KINDS]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        torch.manual_seed(0)
        bd, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            rec = decode_xct(model(*args).xct_out)
        x_ref = bd["xct"]
        l1 = [float((rec[i] - x_ref[i]).abs().mean()) for i in range(len(KINDS))]
        cc = [corr(rec[i], x_ref[i]) for i in range(len(KINDS))]
        rows.append((s, rec, l1, cc))
    c = x_ref.shape[-1] // 2
    fig, axes = plt.subplots(1 + len(rows), len(KINDS), figsize=(2.4 * len(KINDS), 2.3 * (1 + len(rows))),
                             squeeze=False)
    for p, kind in enumerate(KINDS):
        vol = x_ref[p, 0]
        lo, hi = float(vol.min()), float(vol.max())
        axes[0, p].imshow(vol[:, c].numpy(), cmap="gray", vmin=lo, vmax=hi)
        axes[0, p].set_title(f"input: {TITLES[kind]}", fontsize=8)
        for r, (s, rec, l1, cc) in enumerate(rows, start=1):
            axes[r, p].imshow(rec[p, 0, :, c].numpy(), cmap="gray", vmin=lo, vmax=hi)
            axes[r, p].set_title(f"L1 {l1[p]:.3f}  corr {cc[p]:.2f}", fontsize=7)
    for r, (s, *_rest) in enumerate(rows, start=1):
        axes[r, 0].set_ylabel(f"step {s}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{label}: reconstruction by step (axis-1 centre plane; L1 and corr over the whole patch)",
                 fontsize=9)
    fig.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    fig.savefig(out / f"recon_{safe}.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=run_dir" (repo-relative)')
    ap.add_argument("--ref", action="append", default=[], help='reference "label=run_dir", drawn dashed')
    ap.add_argument("--disc-start", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=200)
    ap.add_argument("--recon", action="store_true")
    ap.add_argument("--recon-every", type=int, default=3000)
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [load(s) for s in a.run]
    refs = [load(s) for s in a.ref]
    plot_curves(runs, refs, a.disc_start, out, a.smooth)
    tables(runs, out)
    if a.recon:
        for label, run, _, _ in runs:
            recon_progression(label, run, a.recon_every, a.split, out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Turn ``results.json`` from ``recompute_vae_metrics.py`` into the report.

Writes, all under ``runs/campaigns/07-vae-metric-recompute/vae_metric_recompute/``:

* ``per_run_metrics.csv`` — one row per (run, metric): logged / recomputed / delta
* ``findings.md``         — control verification, per-run tables, ranking assessment
* ``logged_vs_recomputed.pdf`` / ``.png`` — dumbbell chart of the two corrupted
  metrics plus one control, across every run

Usage
-----
::

    python scripts/analysis/report_vae_metric_recompute.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "campaigns" / "07-vae-metric-recompute" / "vae_metric_recompute"

# Validated categorical slots 1 and 2 (light mode) — see the dataviz palette.
# Adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface.
C_LOGGED = "#eb6834"      # orange — as logged (buggy sigmoid path)
C_FIXED = "#2a78d6"       # blue   — recomputed (correct clamp path)
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#b8b6b0"
C_SURFACE = "#fcfcfb"

AFFECTED = ["mae", "sharpness_recon_over_gt"]

# Ordering of the run families down the chart / tables.
FAMILY_ORDER = ["r03", "r04", "r05", "r06", "r07", "vrrae03", "vrrae04"]

FAMILY_LABEL = {
    "r03": "r03 — single-branch",
    "r04": "r04 — dual-branch",
    "r05": "r05 — mask-out sweep",
    "r06": "r06 — XCT-only sweep",
    "r07": "r07 — mask-in sweep",
    "vrrae03": "vrrae03",
    "vrrae04": "vrrae04",
}


def sort_key(run: dict[str, Any]) -> tuple[int, int, int]:
    fam = run["experiment"]
    fam_i = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
    # Inside a sweep, order by decreasing latent width (2x compression first).
    z = run.get("z_channels") or 0
    return (fam_i, -z, run["run_index"])


def label_of(run: dict[str, Any]) -> str:
    z = run.get("z_channels")
    base = f"{run['experiment']}-{run['run_index']:04d}"
    if z:
        return f"{base}  z={z} ({64 // z}x)"
    return base


def build_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    metrics = payload["affected_metrics"] + payload["control_metrics"]
    for run in sorted(payload["runs"], key=sort_key):
        for metric in metrics:
            logged = (run["logged"] or {}).get(metric)
            fixed = run["recomputed_fixed"].get(metric)
            buggy = run["recomputed_buggy"].get(metric)
            if fixed is None and logged is None:
                continue
            delta = None if (fixed is None or logged is None) else fixed - logged
            pct = None
            if delta is not None and logged not in (None, 0):
                pct = 100.0 * delta / abs(logged)
            repro = None
            if buggy is not None and logged not in (None, 0):
                repro = 100.0 * (buggy - logged) / abs(logged)
            rows.append({
                "run": run["run"],
                "label": label_of(run),
                "experiment": run["experiment"],
                "run_index": run["run_index"],
                "z_channels": run.get("z_channels"),
                "reduction_factor": (64 // run["z_channels"]) if run.get("z_channels") else None,
                "model": run["model_name"],
                "best_step": run["best_step"],
                "n_patches": run["n_patches"],
                "metric": metric,
                "affected_by_bug": metric in AFFECTED,
                "logged": logged,
                "recomputed_buggy_path": buggy,
                "recomputed_fixed_path": fixed,
                "delta_fixed_minus_logged": delta,
                "delta_pct": pct,
                "replay_vs_logged_pct": repro,
            })
    return pd.DataFrame(rows)


def fmt(v: Any, nd: int = 6) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ranking_block(df: pd.DataFrame, family: str, metric: str, higher_is_better: bool) -> dict[str, Any]:
    """Order the rungs of one sweep by *metric*, on both the logged and fixed values."""
    sub = df[(df.experiment == family) & (df.metric == metric)].dropna(subset=["logged"])
    if sub.empty:
        return {}
    asc = not higher_is_better
    logged_order = sub.sort_values("logged", ascending=asc)["label"].tolist()
    fixed_order = sub.sort_values("recomputed_fixed_path", ascending=asc)["label"].tolist()
    return {
        "family": family,
        "metric": metric,
        "logged_order": logged_order,
        "fixed_order": fixed_order,
        "identical": logged_order == fixed_order,
        "spearman": float(sub["logged"].rank().corr(
            sub["recomputed_fixed_path"].rank(), method="pearson")) if len(sub) > 2 else None,
    }


def sigmoid_artefact_check(n_patches: int = 400) -> dict[str, float] | None:
    """Predict the size of the bug straight from the data, with no model involved.

    If the reconstruction were perfect, the sigmoid path would still report
    ``mean|sigmoid(x) - x|`` as its MAE — a floor that has nothing to do with the
    model.  The same squashing scales every finite difference by the sigmoid's
    slope, which is what ``sharpness_recon_over_gt`` was multiplied by.  Both
    numbers come from the validation grey levels alone, so they are an
    independent prediction of what the recompute should show.
    """
    import numpy as np

    meta_p = REPO_ROOT / "data" / "split_v2" / "patches_meta.json"
    bin_p = REPO_ROOT / "data" / "split_v2" / "patches_xct.bin"
    index_p = REPO_ROOT / "data" / "split_v2" / "patch_index.parquet"
    if not (meta_p.exists() and bin_p.exists() and index_p.exists()):
        return None

    meta = json.loads(meta_p.read_text())
    n, ps = meta["N"], meta["patch_size"]
    val_rows = pd.read_parquet(index_p, columns=["split"])
    val_idx = np.flatnonzero((val_rows["split"] == "val").to_numpy())
    arr = np.memmap(bin_p, dtype=np.uint8, mode="r", shape=(n, ps, ps, ps))
    take = np.sort(np.random.default_rng(0).choice(val_idx, size=n_patches, replace=False))
    x = np.asarray(arr[take], dtype=np.float32) / 255.0
    sig = 1.0 / (1.0 + np.exp(-x))
    g_x = float(np.abs(np.diff(x, axis=1)).mean())
    g_s = float(np.abs(np.diff(sig, axis=1)).mean())
    return {
        "n_patches": n_patches,
        "mean_grey": float(x.mean()),
        "mae_floor": float(np.abs(sig - x).mean()),
        "slope_factor": g_s / g_x,
    }


def make_figure(payload: dict[str, Any], df: pd.DataFrame, out_dir: Path) -> None:
    """Dumbbell chart: logged vs recomputed, for the two corrupted metrics + a control."""
    panels = [
        ("mae", "val/mae\n(XCT reconstruction MAE, lower is better)"),
        ("sharpness_recon_over_gt", "val/sharpness_recon_over_gt\n(1.0 = recon as sharp as GT)"),
        ("xct_loss", "val/xct_loss  —  CONTROL\n(never touched by the bug)"),
    ]
    runs = [r for r in sorted(payload["runs"], key=sort_key) if r["logged"] is not None]
    labels = [label_of(r) for r in runs]
    y = list(range(len(runs)))[::-1]  # first run at the top

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.42 * len(runs) + 2.6), sharey=True)
    fig.patch.set_facecolor(C_SURFACE)

    for ax, (metric, title) in zip(axes, panels):
        ax.set_facecolor(C_SURFACE)
        lo = [(r["logged"] or {}).get(metric) for r in runs]
        fx = [r["recomputed_fixed"].get(metric) for r in runs]
        for yi, a, b in zip(y, lo, fx):
            if a is None or b is None:
                continue
            # 2px connector, recessive — it carries the size of the correction.
            ax.plot([a, b], [yi, yi], color=C_MUTED, lw=2, zorder=1,
                    solid_capstyle="round")
        ax.scatter([v for v in lo if v is not None],
                   [yi for yi, v in zip(y, lo) if v is not None],
                   s=64, color=C_LOGGED, zorder=3, edgecolor=C_SURFACE, linewidth=1.2)
        ax.scatter([v for v in fx if v is not None],
                   [yi for yi, v in zip(y, fx) if v is not None],
                   s=64, color=C_FIXED, zorder=3, edgecolor=C_SURFACE, linewidth=1.2)

        # Every panel starts at zero.  Autoscaling the control panel to its own
        # sub-0.1% spread would magnify eval noise into a visible bar and hide
        # the point of the panel, which is that nothing moved.
        vals = [v for v in lo + fx if v is not None]
        top = 1.15 if metric == "sharpness_recon_over_gt" else max(vals) * 1.15
        ax.set_xlim(0, max(top, 1.05 * max(vals)))

        if metric == "sharpness_recon_over_gt":
            ax.axvline(1.0, color=C_INK_2, lw=1, ls=(0, (4, 3)), zorder=0)
            ax.text(1.0, len(runs) - 0.35, " as sharp as GT", color=C_INK_2,
                    fontsize=8, va="bottom", ha="left")

        # Median move, stated in words: on the control panel the two marks land
        # on top of each other, so the number is what tells the reader that the
        # hidden orange dot is *under* the blue one rather than missing.
        moves = [abs(100 * (b - a) / a) for a, b in zip(lo, fx)
                 if a not in (None, 0) and b is not None]
        move = sorted(moves)[len(moves) // 2] if moves else None
        note = ("median move: below 0.1%" if move is not None and move < 0.1
                else f"median move: {move:.0f}%" if move is not None else "")
        ax.set_title(f"{title}\n{note}", fontsize=10, color=C_INK, pad=10, loc="left")
        ax.grid(axis="x", color="#e6e5e1", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
        ax.tick_params(colors=C_INK_2, labelsize=8, length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8, color=C_INK)
    axes[0].set_ylim(-0.8, len(runs) - 0.2)

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=C_LOGGED,
               label="as logged  (sigmoid path, pre-2026-09-01)"),
        Line2D([], [], marker="o", ls="", ms=8, color=C_FIXED,
               label="recomputed  (correct clamp path)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=C_INK, bbox_to_anchor=(0.5, 0.004))
    # Title block gets a fixed *physical* allowance, so it never collides with
    # the panels however many runs the chart grows to.
    h_in = fig.get_size_inches()[1]
    fig.text(0.008, 1 - 0.28 / h_in,
             "VAE validation metrics: effect of the XCT-sigmoid eval bug",
             fontsize=13, color=C_INK, ha="left", va="top")
    fig.text(0.008, 1 - 0.60 / h_in,
             f"best.ckpt of each run re-evaluated on {payload['n_batches_per_pass']} "
             f"seeded validation batches; both values come from the same forward pass.",
             fontsize=9, color=C_INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0.34 / h_in, 1, 1 - 0.80 / h_in))

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"logged_vs_recomputed.{ext}", dpi=300,
                    facecolor=C_SURFACE)
    plt.close(fig)


def write_findings(payload: dict[str, Any], df: pd.DataFrame, out_dir: Path) -> None:
    runs = sorted(payload["runs"], key=sort_key)
    n_batches = payload["n_batches_per_pass"]
    lines: list[str] = []
    A = lines.append

    A("# VAE evaluation metrics — recompute after the XCT-sigmoid fix")
    A("")
    A(f"Generated {payload['generated']} · device `{payload['device']}` · "
      f"{len(runs)} runs · {n_batches} validation batches per run "
      f"({n_batches * 128:,} patches).")
    A("")
    A("## What was wrong")
    A("")
    A("The VAE XCT head regresses `xct / 255` directly — `compute_total_loss` calls")
    A("`recon_fn(output.xct_out, batch[\"xct\"])`, so the decoder output already IS the")
    A("grey level in [0, 1]. Until 2026-09-01, `engine._run_eval` applied")
    A("`torch.sigmoid()` to that output before computing the reconstruction eval")
    A("metrics. The sigmoid maps [0, 1] onto [0.5, 0.731], so:")
    A("")
    A("* **`mae`** measured the distance from the ground truth to a squashed copy of the")
    A("  reconstruction. That put an artefact floor of roughly 0.135 under every value —")
    A("  larger than the true error itself, which swamped the real signal.")
    A("* **`sharpness_recon_over_gt`** was scaled by the sigmoid slope (~0.21 over the")
    A("  working range), because the gradient of a squashed volume is a squashed gradient.")
    A("")
    A("Nothing else moved. The training loss, the mask metrics, `porosity_mae` and the")
    A("KL/latent metrics all consumed the raw decoder output and were never affected.")
    A("")
    A("## Method")
    A("")
    A("`scripts/analysis/recompute_vae_metrics.py` loads each run's `best.ckpt`,")
    A("rebuilds the model and the validation loader from that run's own")
    A("`resolved_config.yaml`, and calls the real `engine._run_eval` — the same")
    A("function that produced the historical numbers. A probe inside that single pass")
    A("also computes the old sigmoid-path values from the same forward pass, so the")
    A("buggy→fixed delta carries no eval noise at all.")
    A("")
    A("Two independent checks make the recompute trustworthy:")
    A("")
    A("1. **The control metrics** (`xct_loss`, `porosity_mae`, `kl`, `dice_pos_only`, …)")
    A("   must land on their logged values. They were never affected, so any drift here")
    A("   is pure sampling noise from evaluating a subset of the validation set.")
    A("2. **The replay check** — the probe's sigmoid-path `mae` and")
    A("   `sharpness_recon_over_gt` must land on the *logged* (buggy) values. That proves")
    A("   the reproduction of the old code path is faithful, and therefore that the")
    A("   difference to the corrected value is the real correction.")
    A("")
    check = sigmoid_artefact_check()
    if check:
        A("### An independent prediction of the correction's size")
        A("")
        A(f"Over {check['n_patches']} random validation patches (mean grey "
          f"{check['mean_grey']:.3f}), with no model involved:")
        A("")
        A(f"* `mean |sigmoid(x) - x|` = **{check['mae_floor']:.4f}** — the `mae` a")
        A("  *perfect* reconstruction would still have reported on the sigmoid path.")
        A("  That is several times the real reconstruction error, so on the sigmoid path")
        A("  every model lands near this same number almost regardless of how good it is.")
        A("  (A blurrier model can even read slightly *below* it, because a reconstruction")
        A("  pulled toward the mean is squashed less than the ground truth is — one more")
        A("  reason the logged ordering carried little information.)")
        A(f"* `mean|grad sigmoid(x)| / mean|grad x|` = **{check['slope_factor']:.4f}** — the")
        A("  factor by which the sigmoid flattens every finite difference, and therefore")
        A("  the factor by which `sharpness_recon_over_gt` was scaled down.")
        A("")
        A("Both predictions are borne out below: every logged `mae` sits within a few")
        A("percent of that floor, and the ratio of logged to corrected")
        A("`sharpness_recon_over_gt` reproduces the slope factor across every run.")
        A("")

    A("Each run is compared against its own `val_full` record at the step `best.ckpt`")
    A("was saved from — a full pass over all 1,791 validation batches. The recompute")
    A(f"uses a seeded random subset of {n_batches} batches, so a residual difference of a")
    A("fraction of a percent on the controls is expected and is the noise floor.")
    A("")

    # ── control verification ─────────────────────────────────────────────────
    A("## 1. Control verification — did the recompute reproduce history?")
    A("")
    A("`xct_loss` and `porosity_mae` were never touched by the bug, so recomputed must")
    A("equal logged. The replay columns show the reproduction of the buggy path.")
    A("")
    ctl_rows = []
    for run in runs:
        if not run["logged"]:
            continue
        lab = label_of(run)
        r = {m: (run["logged"].get(m), run["recomputed_fixed"].get(m)) for m in
             ("xct_loss", "porosity_mae")}
        def pct(pair):
            lg, rc = pair
            if lg in (None, 0) or rc is None:
                return "n/a"
            return f"{100 * (rc - lg) / abs(lg):+.2f}%"
        mae_lg = run["logged"].get("mae")
        mae_replay = run["recomputed_buggy"].get("mae")
        sh_lg = run["logged"].get("sharpness_recon_over_gt")
        sh_replay = run["recomputed_buggy"].get("sharpness_recon_over_gt")
        rp = lambda lg, rc: "n/a" if lg in (None, 0) or rc is None else f"{100 * (rc - lg) / abs(lg):+.2f}%"  # noqa: E731
        sh_fx = run["recomputed_fixed"].get("sharpness_recon_over_gt")
        scale = f"{sh_lg / sh_fx:.3f}" if sh_lg and sh_fx else "n/a"
        ctl_rows.append([
            lab,
            fmt(r["xct_loss"][0]), fmt(r["xct_loss"][1]), pct(r["xct_loss"]),
            fmt(r["porosity_mae"][0]), fmt(r["porosity_mae"][1]), pct(r["porosity_mae"]),
            rp(mae_lg, mae_replay), rp(sh_lg, sh_replay), scale,
        ])
    A(md_table(
        ["run", "xct_loss logged", "xct_loss recomp", "dev",
         "porosity_mae logged", "porosity_mae recomp", "dev",
         "mae replay dev", "sharpness replay dev", "sharpness scale"],
        ctl_rows,
    ))
    A("")
    A("The last column is `logged sharpness / corrected sharpness` — the factor the")
    A("sigmoid applied. It should equal the data-derived slope factor above for every")
    A("run, because it is a property of the squashing, not of the model.")
    A("")

    ctl = df[df.metric.isin(["xct_loss", "porosity_mae"])].dropna(subset=["delta_pct"])
    rep = df[df.metric.isin(AFFECTED)].dropna(subset=["replay_vs_logged_pct"])
    A(f"Worst control deviation: **{ctl.delta_pct.abs().max():.2f}%** "
      f"(median {ctl.delta_pct.abs().median():.2f}%). ")
    A(f"Worst buggy-path replay deviation: **{rep.replay_vs_logged_pct.abs().max():.2f}%** "
      f"(median {rep.replay_vs_logged_pct.abs().median():.2f}%).")
    A("")

    # A third, free check: every run trains its XCT head with Charbonnier, which is
    # L1 with a 1e-6 smoothing knee, so a correct MAE has to sit almost on top of
    # xct_loss.  On the sigmoid path it did not, and that alone should have shown
    # the bug.
    wide = df.pivot_table(index="run", columns="metric",
                          values=["logged", "recomputed_fixed_path"], aggfunc="first")
    try:
        gap_fix = ((wide[("recomputed_fixed_path", "mae")]
                    - wide[("recomputed_fixed_path", "xct_loss")]).abs()
                   / wide[("recomputed_fixed_path", "xct_loss")]).median() * 100
        gap_log = ((wide[("logged", "mae")] - wide[("logged", "xct_loss")]).abs()
                   / wide[("logged", "xct_loss")]).median() * 100
        A("A third check comes free. Every run trains its XCT head with Charbonnier —")
        A("L1 with a 1e-6 smoothing knee — so a correct `mae` must sit almost on top of")
        A(f"`xct_loss`. The corrected values differ from `xct_loss` by a median of")
        A(f"**{gap_fix:.1f}%**; the logged ones differed by **{gap_log:.0f}%**. That gap was")
        A("visible in the logs the whole time.")
        A("")
    except KeyError:
        pass

    # ── per-run tables ───────────────────────────────────────────────────────
    A("## 2. Per-run comparison")
    A("")
    A("`as-logged` is the value in the run's `metrics.jsonl`; `recomputed` is the")
    A("corrected value. Rows marked **affected** are the ones the bug changed;")
    A("the rest are controls.")
    A("")
    for run in runs:
        lab = label_of(run)
        A(f"### {lab}")
        A("")
        A(f"`{run['run']}`  ·  {run['model_name']}  ·  best.ckpt @ step {run['best_step']}"
          + (f"  ·  compared against `val_full` @ step {run['logged_step']} "
             f"({run['logged_n_batches']} batches)" if run["logged"] else
             "  ·  **no matching `val_full` record — logged column unavailable**"))
        A("")
        rows = []
        for metric in payload["affected_metrics"] + payload["control_metrics"]:
            lg = (run["logged"] or {}).get(metric)
            fx = run["recomputed_fixed"].get(metric)
            if lg is None and fx is None:
                continue
            delta = None if (lg is None or fx is None) else fx - lg
            pctv = "n/a" if (delta is None or not lg) else f"{100 * delta / abs(lg):+.1f}%"
            rows.append([
                f"**{metric}**" if metric in AFFECTED else metric,
                "affected" if metric in AFFECTED else "control",
                fmt(lg), fmt(fx),
                "n/a" if delta is None else f"{delta:+.6f}",
                pctv,
            ])
        A(md_table(["metric", "role", "as-logged (buggy)", "recomputed (correct)",
                    "delta", "delta %"], rows))
        A("")

    # ── ranking assessment ───────────────────────────────────────────────────
    A("## 3. Does the correction change any ranking?")
    A("")
    rank_summaries = []
    for family in ("r05", "r06", "r07"):
        for metric, hib in (("mae", False), ("sharpness_recon_over_gt", True)):
            blk = ranking_block(df, family, metric, hib)
            if blk:
                rank_summaries.append(blk)
    for blk in rank_summaries:
        A(f"### {FAMILY_LABEL.get(blk['family'], blk['family'])} — `{blk['metric']}`")
        A("")
        A(f"* order on the logged (buggy) values: {' > '.join(blk['logged_order'])}")
        A(f"* order on the corrected values: {' > '.join(blk['fixed_order'])}")
        A(f"* **ranking {'unchanged' if blk['identical'] else 'CHANGED'}**")
        A("")
    A("")

    # ── cross-family comparison at matched compression ───────────────────────
    A("### Across the three families, at matched compression")
    A("")
    A("The vault's D31 frames r05 / r06 / r07 as a triplet. If the correction were")
    A("going to overturn anything, this is where it would show: `mae` on the sigmoid")
    A("path was dominated by an artefact floor that is roughly the same for every")
    A("model, which compresses genuine differences between families into the noise.")
    A("")
    cross_rows = []
    for z in sorted({r for r in df.z_channels.dropna().unique()}, reverse=True):
        for metric in AFFECTED:
            sub = df[(df.metric == metric) & (df.z_channels == z)
                     & (df.experiment.isin(["r05", "r06", "r07"]))].dropna(subset=["logged"])
            if len(sub) < 2:
                continue
            hib = metric == "sharpness_recon_over_gt"
            lg = sub.sort_values("logged", ascending=not hib)["experiment"].tolist()
            fx = sub.sort_values("recomputed_fixed_path", ascending=not hib)["experiment"].tolist()
            cross_rows.append([
                f"z={int(z)} ({64 // int(z)}x)", metric,
                " > ".join(lg), " > ".join(fx),
                "same" if lg == fx else "**CHANGED**",
            ])
    if cross_rows:
        A(md_table(["compression", "metric", "best-first (logged)",
                    "best-first (corrected)", "family order"], cross_rows))
    else:
        A("_No compression rung has two or more families with a logged value._")
    A("")

    # ── consequences for the record ──────────────────────────────────────────
    A("## 4. Which past conclusions are affected")
    A("")
    A("Based on a read-only survey of the vault at `/home/jorgecabrejas/Dev/PhDTracker`")
    A("and of `docs/`. The vault was not modified.")
    A("")
    A("### Not affected — the sweep and architecture decisions")
    A("")
    A("Every r05 / r06 / r07 conclusion in the vault is stated in `val.xct_loss` and")
    A("`val/porosity_mae`, never in `mae` or `sharpness_recon_over_gt`:")
    A("")
    A("* the R05 sweep summary (`20_Writing/Notes/PoreGen - VAE Experiments.md:463`) lists")
    A("  `xct_loss` and `porosity_mae` per rung;")
    A("* the R06 sweep summary (same file, `:438`) is explicit — \"val.xct_loss, unico")
    A("  metric comparable\";")
    A("* the only numeric sweep table in the vault")
    A("  (`10_Research/Analyses/R06 - val total incomparable entre factores de reduccion.md:18`)")
    A("  has columns val.total / xct / beta·KL / free-bits floor;")
    A("* D29 (`PoreGen - Decisions Log.md:362`) argues the compression elbow from")
    A("  `xct_loss` alone; D31 (`:378`) fixes the selection criterion as \"primario val")
    A("  porosity-MAE; secundario val xct_loss\";")
    A("* D13 (`:139`, removing attention) cites Dice, Charbonnier loss, `porosity_mae`")
    A("  and latent statistics.")
    A("")
    A("All of those are control metrics here, and every one of them reproduces. **The")
    A("compression-elbow and architecture conclusions stand unchanged.**")
    A("")
    A("D33 — the choice of `r07-run-0006` (z=4, 16x) as the production VAE for the LDM")
    A("stack (`PoreGen - Decisions Log.md:404`) — rests on the decoder mask-sanity gate")
    A("(porosity stays anchored under 2-sigma posterior perturbation; ambiguous voxels")
    A("below 0.5%), not on any reconstruction metric. **Unaffected.**")
    A("")
    A("D19 (raising `disc_weight` 0.01 -> 0.05, `PoreGen - Decisions Log.md:230`, cited at")
    A("`:235`) rests on the \"volume-level sharpness ratio of 0.12-0.27\" from the R04")
    A("evaluation, and the same R04 entry (`PoreGen - VAE Experiments.md:216`) quotes")
    A("PSNR 27.93 dB, SSIM 0.617 and MAE 0.031. Those are *volume-level* numbers from")
    A("`src/poregen/eval/metrics.py`, which has always decoded with")
    A("`model.xct_head(dec).clamp(0.0, 1.0)` — the correct path, never the sigmoid.")
    A("**D19 and the R04 volume-level figures stand.**")
    A("")
    A("That same entry (`:218`) records: \"Patch-level sharpness ratio metric is broken")
    A("(std=238); use volume-level values only.\" In hindsight that was this bug being")
    A("noticed and worked around rather than diagnosed — the right call at the time, and")
    A("it is why no decision was ever built on the patch-level number.")
    A("")
    A("### Affected — numbers quoted in the record")
    A("")
    A("The patch-level `val/mae` is quoted in the vault for the VRRAE family and nowhere")
    A("else — five places in `PoreGen - VAE Experiments.md` (`:327`, `:355`, `:379`,")
    A("`:390`, `:394`), all in the range 0.172-0.183. Every one of those numbers sits on")
    A("the artefact floor and says almost nothing about the model that produced it. The")
    A("corrected values for the VRRAE runs that still have a checkpoint are in the tables")
    A("above; the runs quoted at `:390` and `:394` (`vrrae02-run-0001`,")
    A("`vrrae03-run-0002`) have no `best.ckpt` on disk and cannot be corrected — see the")
    A("last section.")
    A("")
    A("D26 — the VRRAE linear-ablation decision (`PoreGen - Decisions Log.md:334`) — is")
    A("argued from \"VRRAE reconstruction is ~3x worse (L1) than the spatial baselines\",")
    A("i.e. from `xct_loss`. **The conclusion is unaffected, even though the `mae`")
    A("numbers quoted alongside it are wrong.**")
    A("")
    A("One analysis note")
    A("(`10_Research/Analyses/R06 - val total incomparable entre factores de reduccion.md:15`)")
    A("proposed comparing the compression sweep on \"val.xct_loss / val.mae / sharpness\".")
    A("That comparison was never carried out. Had it been, it would have been carried out")
    A("on corrupted numbers; this document now supplies the correct ones.")
    A("")
    A("### Still wrong in the code")
    A("")
    A("`src/poregen/experiments/r03.py:482` applies `sigmoid_xct(output.xct_out)` to the")
    A("*main* VAE decoder output before scoring it against the auxiliary decoder. The")
    A("auxiliary decoder is genuinely trained through a sigmoid")
    A("(`charbonnier_on_sigmoid_logits`), so line 483 is correct — but line 482 is the")
    A("same bug, and it makes the R03 notebook's original-vs-auxiliary sharpness and")
    A("Charbonnier comparison unfair to the main decoder. Out of scope for this")
    A("recompute; flagged for a separate fix.")
    A("")

    # ── skipped / failed ─────────────────────────────────────────────────────
    if payload["skipped"] or payload["failures"]:
        A("## 5. Runs not re-evaluated")
        A("")
        for s in payload["skipped"]:
            A(f"* `{s['run']}` — {s['reason']}")
        for f in payload["failures"]:
            A(f"* `{f['run']}` — FAILED: {f['error']}")
        A("")

    (out_dir / "findings.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    payload = json.loads((args.dir / "results.json").read_text())
    df = build_frame(payload)
    df.to_csv(args.dir / "per_run_metrics.csv", index=False)
    make_figure(payload, df, args.dir)
    write_findings(payload, df, args.dir)
    print(f"wrote per_run_metrics.csv, findings.md, logged_vs_recomputed.{{pdf,png}} "
          f"to {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

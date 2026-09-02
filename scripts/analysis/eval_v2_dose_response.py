"""Dose-response v2 — three arms, corrected conditioning semantics.

Arms (see scripts/analysis/_eval_v2.py): seq (sequential, s_por 1.0),
joint_legacy (joint, legacy semantics, s_por 1.5), joint_oob (joint,
specimen semantics with honest OOB edges, s_por 1.5).

Protocol: targets {0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10} x 3 arms x
seeds {101, 202, 303} = 63 volumes of 192^3 (3x3x3 patch grid), DDIM-50,
RAW weights at step 130000, coherent local-porosity field.  Every volume is
saved under runs/campaigns/03-eval-v2-buggy-decode/volumes/dose_response/<arm>/target_<t>_seed_<s>/.

Analyses per arm:
  1. GLOBAL — volume mask porosity vs requested global target (OLS, per-level
     mean±std, gate |err| < 0.005), plus void-corrected global porosity
     (union of mask and calibrated dark-voxel detector, u8 threshold from
     runs/campaigns/02-porosity-control-v1/void_mask_audit/).
  2. LOCAL — per 64^3 tile cell, delivered mask porosity vs the coherent
     field's local target (27 cells/volume -> 567 points/arm); OLS + scatter.

Outputs: $POREGEN_EVAL_ROOT/dose_response/{results.json, findings.md,
figures}; volumes under $POREGEN_EVAL_ROOT/volumes/dose_response/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import savefig, set_style, write_findings, write_json, plt  # noqa: E402
from _eval_v2 import (  # noqa: E402
    ARMS, ARM_COLORS, CKPT, DDIM_STEPS, EVAL_ROOT, GATE, GEN_BATCH, GRID,
    JOINT_WINDOW_STRIDE, SEEDS, VOLUME_VOX, REPO,
    build_generator, cell_porosities, coherent_por_map, configure_arm,
    corrected_porosity, fit_ols, load_existing, load_void_detector, run_one,
    save_volume,
)
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)

TARGETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
OUT_DIR = EVAL_ROOT / "dose_response"
VOL_ROOT = EVAL_ROOT / "volumes" / "dose_response"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-window-batch", type=int, default=16)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device("cuda")

    detector = load_void_detector()
    gen, model, schedule, step = build_generator(device)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    records = []
    all_oom = []
    jwb = args.joint_window_batch
    n_total = len(TARGETS) * len(ARMS) * len(SEEDS)
    i = 0
    for arm in ARMS:
        mode, s_por = configure_arm(gen, model, schedule, device, arm)
        for target in TARGETS:
            for seed in SEEDS:
                i += 1
                vol_dir = VOL_ROOT / arm / f"target_{target:g}_seed_{seed}"
                rec = load_existing(vol_dir)
                if rec is not None:
                    records.append(rec)
                    print(f"[{i:2d}/{n_total}] {arm:12s} target={target:.3f} "
                          f"seed={seed}  (cached)", flush=True)
                    continue
                torch.manual_seed(seed)
                por_map, field = coherent_por_map(target, seed, te_sampler,
                                                  corr_lengths)
                xct, mask, stats, wall, jwb_used, ooms = run_one(
                    gen, mode, por_map, jwb)
                all_oom.extend([f"{arm} target={target:g} seed={seed}: {e}"
                                for e in ooms])
                delivered = float(stats["actual_mask_porosity"])
                corrected, thr_used = corrected_porosity(xct, mask, detector)
                cells = cell_porosities(mask)
                rec = {
                    "arm": arm,
                    "mode": mode,
                    "conditioning_semantics": gen.conditioning_semantics,
                    "s_por": s_por,
                    "target": target,
                    "seed": seed,
                    "delivered_mask_porosity": delivered,
                    "corrected_porosity": corrected,
                    "abs_error": abs(delivered - target),
                    "abs_error_corrected": abs(corrected - target),
                    "field_mean": float(field.mean()),
                    "field_min": float(field.min()),
                    "field_max": float(field.max()),
                    "cell_targets": {f"{iz},{iy},{ix}": por_map[(iz, iy, ix)]
                                     for iz in range(GRID[0])
                                     for iy in range(GRID[1])
                                     for ix in range(GRID[2])},
                    "cell_delivered": cells,
                    "seam_xct_ratio": stats["seam_xct_ratio"],
                    "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
                    "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
                    "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
                    "seam_mask_ratio": stats["seam_mask_ratio"],
                    "void_detector_threshold_u8": thr_used,
                    "wall_s": round(wall, 1),
                    "joint_window_batch": jwb_used if mode == "joint" else None,
                }
                save_volume(vol_dir, xct, mask, rec)
                del xct, mask
                records.append(rec)
                print(f"[{i:2d}/{n_total}] {arm:12s} target={target:.3f} "
                      f"seed={seed}  delivered={delivered:.4f} "
                      f"corrected={corrected:.4f} "
                      f"|err|={rec['abs_error']:.4f}  "
                      f"seam_xct={stats['seam_xct_ratio']:.3f}  "
                      f"wall={wall:.1f}s", flush=True)
                write_json({"partial": True, "records": records}, OUT_DIR)

    # ---------------- analysis ----------------
    arms = list(ARMS)
    fits_global = {}
    fits_global_corr = {}
    fits_local = {}
    local_err = {}
    per_level = {}
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        req = np.array([r["target"] for r in rs])
        dlv = np.array([r["delivered_mask_porosity"] for r in rs])
        cor = np.array([r["corrected_porosity"] for r in rs])
        fits_global[arm] = fit_ols(req, dlv)
        fits_global_corr[arm] = fit_ols(req, cor)

        # LOCAL: every 64^3 cell of every volume
        lt = np.array([r["cell_targets"][k] for r in rs
                       for k in sorted(r["cell_targets"])])
        ld = np.array([r["cell_delivered"][k] for r in rs
                       for k in sorted(r["cell_delivered"])])
        fits_local[arm] = {**fit_ols(lt, ld), "n_points": int(len(lt))}
        ae = np.abs(ld - lt)
        local_err[arm] = {
            "abs_error_mean": float(ae.mean()),
            "abs_error_median": float(np.median(ae)),
            "abs_error_p95": float(np.percentile(ae, 95)),
            "abs_error_max": float(ae.max()),
            "frac_within_gate": float((ae < GATE).mean()),
        }

        per_level[arm] = []
        for target in TARGETS:
            ls = [r for r in rs if r["target"] == target]
            d = np.array([r["delivered_mask_porosity"] for r in ls])
            c = np.array([r["corrected_porosity"] for r in ls])
            e = np.array([r["abs_error"] for r in ls])
            sx = np.array([r["seam_xct_ratio"] for r in ls])
            sm = np.array([r["seam_mask_ratio"] for r in ls])
            per_level[arm].append({
                "target": target,
                "delivered_mean": float(d.mean()),
                "delivered_std": float(d.std(ddof=1)),
                "corrected_mean": float(c.mean()),
                "corrected_std": float(c.std(ddof=1)),
                "corrected_minus_mask_mean": float((c - d).mean()),
                "abs_error_mean": float(e.mean()),
                "passes_gate": bool(e.mean() < GATE),
                "n_seeds_within_gate": int((e < GATE).sum()),
                "seam_xct_ratio_mean": float(sx.mean()),
                "seam_xct_ratio_std": float(sx.std(ddof=1)),
                "seam_mask_ratio_mean": float(sm.mean()),
                "seam_mask_ratio_std": float(sm.std(ddof=1)),
            })

    results = {
        "checkpoint": str(CKPT),
        "step": step,
        "weights": "raw (non-EMA)",
        "ddim_steps": DDIM_STEPS,
        "volume_vox": VOLUME_VOX,
        "grid": list(GRID),
        "targets": TARGETS,
        "arms": {a: {"mode": ARMS[a][0], "conditioning_semantics": ARMS[a][1],
                     "s_por": ARMS[a][2]} for a in arms},
        "seeds": SEEDS,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "gen_batch": GEN_BATCH,
        "gate_abs_error": GATE,
        "void_detector": {**detector, "form": "material-referenced, per-volume threshold = material_mode - k*spread (raw u8 scale)"},
        "void_threshold_source": "runs/campaigns/02-porosity-control-v1/void_mask_audit/results.json "
                                 "(calibrated on real volumes, pooled dice)",
        "porosity_field": "coherent (T-E marginal + T-D smoothing), "
                          "seeded per (torch seed == field seed)",
        "volumes_root": str(VOL_ROOT),
        "oom_events": all_oom,
        "fits_global_mask": fits_global,
        "fits_global_corrected": fits_global_corr,
        "fits_local": fits_local,
        "local_abs_error": local_err,
        "per_level": per_level,
        "records": records,
    }
    p_json = write_json(results, OUT_DIR)

    # ---------------- figures ----------------
    figs = []

    # (a) GLOBAL delivered vs requested, 3 arms + identity
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ymax = max(0.115, max(r["delivered_mask_porosity"] for r in records) + 0.008)
    lim = [0.0, ymax]
    ax.plot(lim, lim, color="0.4", lw=1.0, ls="--", zorder=1,
            label="identity (delivered = requested)")
    ax.fill_between(lim, [v - GATE for v in lim], [v + GATE for v in lim],
                    color="0.85", alpha=0.5, zorder=0, label=f"gate ±{GATE}")
    for arm in arms:
        col = ARM_COLORS[arm]
        rs = [r for r in records if r["arm"] == arm]
        ax.scatter([r["target"] for r in rs],
                   [r["delivered_mask_porosity"] for r in rs],
                   s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
        means = [pl["delivered_mean"] for pl in per_level[arm]]
        stds = [pl["delivered_std"] for pl in per_level[arm]]
        f = fits_global[arm]
        ax.errorbar(TARGETS, means, yerr=stds, color=col, marker="o", ms=5,
                    lw=1.4, capsize=3, zorder=3,
                    label=f"{arm} (slope {f['slope']:.2f}, R² {f['r2']:.3f})")
    ax.set_xlim(0.0, 0.115)
    ax.set_ylim(lim)
    ax.set_xlabel("Requested global porosity")
    ax.set_ylabel("Delivered mask porosity")
    ax.set_title(f"Global dose response — ldm05 step {step} "
                 f"(RAW, DDIM-{DDIM_STEPS}, 192³, {len(SEEDS)} seeds)")
    ax.legend(loc="upper left")
    figs += savefig(fig, OUT_DIR, "dose_fig1_global_delivered_vs_requested")

    # (b) LOCAL scatter, one panel per arm
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True,
                             sharex=True, sharey=True)
    lmax = 0.0
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        lt = [r["cell_targets"][k] for r in rs for k in sorted(r["cell_targets"])]
        ld = [r["cell_delivered"][k] for r in rs for k in sorted(r["cell_delivered"])]
        lmax = max(lmax, max(lt), max(ld))
    lim = [0.0, lmax * 1.05]
    for ax, arm in zip(axes, arms):
        col = ARM_COLORS[arm]
        rs = [r for r in records if r["arm"] == arm]
        lt = np.array([r["cell_targets"][k] for r in rs
                       for k in sorted(r["cell_targets"])])
        ld = np.array([r["cell_delivered"][k] for r in rs
                       for k in sorted(r["cell_delivered"])])
        ax.plot(lim, lim, color="0.4", lw=1.0, ls="--", zorder=1)
        ax.scatter(lt, ld, s=8, color=col, alpha=0.35, edgecolors="none",
                   zorder=2)
        f = fits_local[arm]
        xx = np.array(lim)
        ax.plot(xx, f["slope"] * xx + f["intercept"], color=col, lw=1.4,
                zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Cell local target (coherent field)")
        ax.set_title(f"{arm}\nslope {f['slope']:.2f}, R² {f['r2']:.3f}, "
                     f"n={f['n_points']}")
    axes[0].set_ylabel("Cell delivered mask porosity")
    fig.suptitle("Local conditioning obedience — per 64³ tile cell")
    figs += savefig(fig, OUT_DIR, "dose_fig2_local_scatter_per_arm")

    # (c) seam ratios vs target per arm
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True,
                             sharex=True)
    for ax, key, label in zip(
            axes,
            ["seam_xct_ratio", "seam_mask_ratio"],
            ["seam_xct_ratio (seam MAD / interior MAD)", "seam_mask_ratio"]):
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="no seam excess")
        for arm in arms:
            col = ARM_COLORS[arm]
            rs = [r for r in records if r["arm"] == arm]
            ax.scatter([r["target"] for r in rs], [r[key] for r in rs],
                       s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
            mkey = key + "_mean"
            skey = key + "_std"
            ax.errorbar(TARGETS, [pl[mkey] for pl in per_level[arm]],
                        yerr=[pl[skey] for pl in per_level[arm]],
                        color=col, marker="o", ms=5, lw=1.4, capsize=3,
                        zorder=3, label=arm)
        ax.set_xlabel("Requested global porosity")
        ax.set_ylabel(label)
    axes[0].legend(loc="best")
    fig.suptitle("Seam severity vs porosity target")
    figs += savefig(fig, OUT_DIR, "dose_fig3_seam_vs_target")

    # (d) mask vs corrected porosity
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for arm in arms:
        col = ARM_COLORS[arm]
        means_m = [pl["delivered_mean"] for pl in per_level[arm]]
        means_c = [pl["corrected_mean"] for pl in per_level[arm]]
        stds_c = [pl["corrected_std"] for pl in per_level[arm]]
        ax.plot(TARGETS, means_m, color=col, marker="o", ms=4, lw=1.2,
                label=f"{arm} mask")
        ax.errorbar(TARGETS, means_c, yerr=stds_c, color=col, marker="s",
                    ms=4, lw=1.2, ls="--", capsize=3,
                    label=f"{arm} corrected (union, mat-ref)")
    lim = [0.0, 0.115]
    ax.plot(lim, lim, color="0.4", lw=1.0, ls=":")
    ax.set_xlim(lim)
    ax.set_xlabel("Requested global porosity")
    ax.set_ylabel("Delivered porosity")
    ax.set_title("Mask vs void-corrected global porosity")
    ax.legend(loc="upper left", fontsize=7.5)
    figs += savefig(fig, OUT_DIR, "dose_fig4_mask_vs_corrected")

    # ---------------- findings ----------------
    lines = [
        f"# Dose-response ({EVAL_ROOT.name}) — ldm05, step {step} (RAW weights, 3 arms)",
        "",
        f"Checkpoint: `{CKPT}`. DDIM-{DDIM_STEPS}, 192³ voxels (3×3×3 grid), "
        f"coherent local-porosity field, seeds {SEEDS}. "
        f"Gate: |delivered − requested| < {GATE}. "
        f"Void-corrected porosity: union of mask and the material-referenced "
        f"dark-voxel detector (threshold material_mode − {detector['k']:g}·spread "
        f"on the volume's own raw u8 scale; k calibrated on real volumes in "
        f"the void-mask audit). "
        f"All volumes saved under `{VOL_ROOT}`.",
        "",
        "Arms: " + "; ".join(
            f"`{a}` = mode {ARMS[a][0]}, semantics {ARMS[a][1]}, "
            f"s_por {ARMS[a][2]:g}" for a in arms),
        "",
        "## Global fits (delivered vs requested, OLS over all seeds)",
        "",
        "| arm | slope (mask) | intercept (mask) | R² (mask) "
        "| slope (corr) | intercept (corr) | R² (corr) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        f, g = fits_global[arm], fits_global_corr[arm]
        lines.append(f"| {arm} | {f['slope']:.3f} | {f['intercept']:.4f} "
                     f"| {f['r2']:.4f} | {g['slope']:.3f} "
                     f"| {g['intercept']:.4f} | {g['r2']:.4f} |")
    lines += [
        "",
        "## Local fits (cell delivered vs cell target, 27 cells/volume)",
        "",
        "| arm | slope | intercept | R² | n | mean \\|err\\| | median \\|err\\| "
        "| p95 \\|err\\| | frac within gate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        f = fits_local[arm]
        e = local_err[arm]
        lines.append(f"| {arm} | {f['slope']:.3f} | {f['intercept']:.4f} "
                     f"| {f['r2']:.4f} | {f['n_points']} "
                     f"| {e['abs_error_mean']:.4f} "
                     f"| {e['abs_error_median']:.4f} "
                     f"| {e['abs_error_p95']:.4f} "
                     f"| {e['frac_within_gate']:.2f} |")
    lines += ["", "## Per-level results", ""]
    for arm in arms:
        lines += [f"### {arm}", "",
                  "| target | mask (mean ± std) | corrected (mean ± std) "
                  "| corr − mask | mean \\|err\\| | gate | seam_xct | seam_mask |",
                  "|---|---|---|---|---|---|---|---|"]
        for pl in per_level[arm]:
            lines.append(
                f"| {pl['target']:.3f} | {pl['delivered_mean']:.4f} ± "
                f"{pl['delivered_std']:.4f} | {pl['corrected_mean']:.4f} ± "
                f"{pl['corrected_std']:.4f} "
                f"| {pl['corrected_minus_mask_mean']:+.4f} "
                f"| {pl['abs_error_mean']:.4f} | "
                f"{'PASS' if pl['passes_gate'] else 'FAIL'} "
                f"({pl['n_seeds_within_gate']}/{len(SEEDS)}) | "
                f"{pl['seam_xct_ratio_mean']:.2f} ± "
                f"{pl['seam_xct_ratio_std']:.2f} | "
                f"{pl['seam_mask_ratio_mean']:.2f} |")
        lines.append("")
    if all_oom:
        lines += ["## OOM events", ""] + [f"- {e}" for e in all_oom] + [""]
    p_md = write_findings("\n".join(lines), OUT_DIR)
    print("Wrote:", p_json, p_md, *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()

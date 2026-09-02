"""CFG sweep v2 — three arms, corrected conditioning semantics.

Arms (see scripts/analysis/_eval_v2.py): seq (sequential), joint_legacy
(joint, legacy semantics), joint_oob (joint, specimen semantics).  The sweep
varies the porosity guidance scale s_por over {0.5, 1.0, 1.5, 2.0}
(0.0 and 3.0 dropped — settled by cfg-sweep v1).

Protocol: s_por {0.5, 1.0, 1.5, 2.0} x targets {0.02, 0.05} x 3 arms x
seeds {101, 202, 303} = 72 volumes of 192^3, DDIM-50, RAW weights at step
130000, coherent local-porosity field.  Every volume is saved under
runs/eval_v2/volumes/cfg_sweep/<arm>/spor_<x>_target_<t>_seed_<s>/.

Per volume: mask porosity, void-corrected porosity (union with the
calibrated dark-voxel detector), seam ratios, degenerate-block fraction.

Outputs: runs/eval_v2/cfg_sweep/{results.json, findings.md, figures}.
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
    ARMS, ARM_COLORS, CKPT, DDIM_STEPS, DEGEN_THR, EVAL_ROOT, GATE, GEN_BATCH,
    GRID, JOINT_WINDOW_STRIDE, SEEDS, VOLUME_VOX, REPO,
    build_generator, coherent_por_map, configure_arm, corrected_porosity,
    degenerate_fraction, load_existing, load_void_detector, run_one,
    save_volume,
)
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)

S_POR_GRID = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.02, 0.05]
OUT_DIR = EVAL_ROOT / "cfg_sweep"
VOL_ROOT = EVAL_ROOT / "volumes" / "cfg_sweep"


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
    n_total = len(S_POR_GRID) * len(TARGETS) * len(ARMS) * len(SEEDS)
    i = 0
    for s_por in S_POR_GRID:
        for arm in ARMS:
            mode, _ = configure_arm(gen, model, schedule, device, arm,
                                    s_por=s_por)
            for target in TARGETS:
                for seed in SEEDS:
                    i += 1
                    vol_dir = (VOL_ROOT / arm /
                               f"spor_{s_por:g}_target_{target:g}_seed_{seed}")
                    rec = load_existing(vol_dir)
                    if rec is not None:
                        records.append(rec)
                        print(f"[{i:2d}/{n_total}] {arm:12s} s_por={s_por:.1f} "
                              f"target={target:.2f} seed={seed}  (cached)",
                              flush=True)
                        continue
                    torch.manual_seed(seed)
                    por_map, field = coherent_por_map(target, seed, te_sampler,
                                                      corr_lengths)
                    xct, mask, stats, wall, jwb_used, ooms = run_one(
                        gen, mode, por_map, jwb)
                    all_oom.extend(
                        [f"{arm} s_por={s_por:g} target={target:g} "
                         f"seed={seed}: {e}" for e in ooms])
                    delivered = float(stats["actual_mask_porosity"])
                    corrected, thr_used = corrected_porosity(xct, mask, detector)
                    degen = degenerate_fraction(mask)
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
                        "seam_xct_ratio": stats["seam_xct_ratio"],
                        "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
                        "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
                        "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
                        "seam_mask_ratio": stats["seam_mask_ratio"],
                        "degenerate_block_fraction": degen,
                        "void_detector_threshold_u8": thr_used,
                        "wall_s": round(wall, 1),
                        "joint_window_batch": jwb_used if mode == "joint" else None,
                    }
                    save_volume(vol_dir, xct, mask, rec)
                    del xct, mask
                    records.append(rec)
                    print(f"[{i:2d}/{n_total}] {arm:12s} s_por={s_por:.1f} "
                          f"target={target:.2f} seed={seed}  "
                          f"delivered={delivered:.4f} "
                          f"corrected={corrected:.4f} "
                          f"|err|={rec['abs_error']:.4f}  "
                          f"seam_xct={stats['seam_xct_ratio']:.2f}  "
                          f"degen={degen:.2f}  wall={wall:.1f}s", flush=True)
                    write_json({"partial": True, "records": records}, OUT_DIR)

    # ---------------- analysis ----------------
    arms = list(ARMS)
    per_cell = {}     # "<arm>_<target>" -> list over s_por of aggregates
    best = {}         # "<arm>_<target>" -> best s_por by mean |error|
    for arm in arms:
        for target in TARGETS:
            key = f"{arm}_{target:g}"
            rows = []
            for s_por in S_POR_GRID:
                ls = [r for r in records
                      if r["arm"] == arm and r["target"] == target
                      and r["s_por"] == s_por]
                d = np.array([r["delivered_mask_porosity"] for r in ls])
                c = np.array([r["corrected_porosity"] for r in ls])
                e = np.array([r["abs_error"] for r in ls])
                sx = np.array([r["seam_xct_ratio"] for r in ls])
                sm = np.array([r["seam_mask_ratio"] for r in ls])
                dg = np.array([r["degenerate_block_fraction"] for r in ls])
                rows.append({
                    "s_por": s_por,
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
                    "degenerate_block_fraction_mean": float(dg.mean()),
                })
            per_cell[key] = rows
            b = min(rows, key=lambda r: r["abs_error_mean"])
            best[key] = {"s_por": b["s_por"],
                         "abs_error_mean": b["abs_error_mean"],
                         "passes_gate": b["passes_gate"]}

    results = {
        "checkpoint": str(CKPT),
        "step": step,
        "weights": "raw (non-EMA)",
        "ddim_steps": DDIM_STEPS,
        "volume_vox": VOLUME_VOX,
        "grid": list(GRID),
        "s_por_grid": S_POR_GRID,
        "targets": TARGETS,
        "arms": {a: {"mode": ARMS[a][0], "conditioning_semantics": ARMS[a][1]}
                 for a in arms},
        "seeds": SEEDS,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "gen_batch": GEN_BATCH,
        "gate_abs_error": GATE,
        "degenerate_threshold": DEGEN_THR,
        "void_detector": {**detector, "form": "material-referenced, per-volume threshold = material_mode - k*spread (raw u8 scale)"},
        "porosity_field": "coherent (T-E marginal + T-D smoothing), "
                          "seeded per (torch seed == field seed)",
        "volumes_root": str(VOL_ROOT),
        "oom_events": all_oom,
        "per_cell": per_cell,
        "best_s_por": best,
        "records": records,
    }
    p_json = write_json(results, OUT_DIR)

    # ---------------- figures ----------------
    # (a) delivered vs s_por, one panel per target, all arms
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True,
                             sharex=True)
    for ax, target in zip(axes, TARGETS):
        ax.axhline(target, color="0.4", lw=1.0, ls="--",
                   label=f"target {target:g}")
        ax.axhspan(target - GATE, target + GATE, color="0.85", alpha=0.5,
                   zorder=0, label=f"gate ±{GATE}")
        for arm in arms:
            col = ARM_COLORS[arm]
            rs = [r for r in records
                  if r["arm"] == arm and r["target"] == target]
            ax.scatter([r["s_por"] for r in rs],
                       [r["delivered_mask_porosity"] for r in rs],
                       s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
            rows = per_cell[f"{arm}_{target:g}"]
            ax.errorbar(S_POR_GRID,
                        [r["delivered_mean"] for r in rows],
                        yerr=[r["delivered_std"] for r in rows],
                        color=col, marker="o", ms=5, lw=1.4, capsize=3,
                        zorder=3, label=arm)
        ax.set_xlabel("Porosity guidance scale $s_{por}$")
        ax.set_title(f"target = {target:g}")
    axes[0].set_ylabel("Delivered mask porosity")
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle(f"Delivered porosity vs $s_{{por}}$ — ldm05 step {step} "
                 f"(RAW, DDIM-{DDIM_STEPS}, 192³, {len(SEEDS)} seeds)")
    figs = savefig(fig, OUT_DIR, "cfg_fig1_delivered_vs_spor")

    # (b) seam_xct_ratio vs s_por
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--",
                   label="no seam excess (ratio 1)")
        for arm in arms:
            col = ARM_COLORS[arm]
            rs = [r for r in records
                  if r["arm"] == arm and r["target"] == target]
            ax.scatter([r["s_por"] for r in rs],
                       [r["seam_xct_ratio"] for r in rs],
                       s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
            rows = per_cell[f"{arm}_{target:g}"]
            ax.errorbar(S_POR_GRID,
                        [r["seam_xct_ratio_mean"] for r in rows],
                        yerr=[r["seam_xct_ratio_std"] for r in rows],
                        color=col, marker="o", ms=5, lw=1.4, capsize=3,
                        zorder=3, label=arm)
        ax.set_xlabel("Porosity guidance scale $s_{por}$")
        ax.set_title(f"target = {target:g}")
    axes[0].set_ylabel("seam_xct_ratio (seam MAD / interior MAD)")
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Seam severity vs guidance scale")
    figs += savefig(fig, OUT_DIR, "cfg_fig2_seam_vs_spor")

    # (c) degenerate-block fraction vs s_por
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        for arm in arms:
            col = ARM_COLORS[arm]
            rows = per_cell[f"{arm}_{target:g}"]
            ax.plot(S_POR_GRID,
                    [r["degenerate_block_fraction_mean"] for r in rows],
                    color=col, marker="o", ms=5, lw=1.4, label=arm)
        ax.set_xlabel("Porosity guidance scale $s_{por}$")
        ax.set_title(f"target = {target:g}")
    axes[0].set_ylabel(f"Degenerate 64³ block fraction (φ < {DEGEN_THR:g})")
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Degenerate blocks vs guidance scale")
    figs += savefig(fig, OUT_DIR, "cfg_fig3_degen_vs_spor")

    # ---------------- findings ----------------
    lines = [
        f"# CFG sweep v2 — ldm05, step {step} (RAW weights, 3 arms)",
        "",
        f"Checkpoint: `{CKPT}`. DDIM-{DDIM_STEPS}, 192³ voxels (3×3×3 grid), "
        f"coherent local-porosity field, seeds {SEEDS}. "
        f"s_por grid {S_POR_GRID} (0.0 and 3.0 dropped — settled in v1). "
        f"Gate: |delivered − requested| < {GATE}. Void-corrected porosity "
        f"uses the material-referenced detector (material_mode − {detector['k']:g}·spread, raw u8 scale). "
        f"All volumes saved under `{VOL_ROOT}`.",
        "",
        "Arms: " + "; ".join(
            f"`{a}` = mode {ARMS[a][0]}, semantics {ARMS[a][1]}" for a in arms),
        "",
        "## Best s_por per (arm, target) by mean |error|",
        "",
        "| arm | target | best s_por | mean \\|err\\| | gate |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        for target in TARGETS:
            b = best[f"{arm}_{target:g}"]
            lines.append(f"| {arm} | {target:g} | {b['s_por']:g} "
                         f"| {b['abs_error_mean']:.4f} "
                         f"| {'PASS' if b['passes_gate'] else 'FAIL'} |")
    lines += ["", "## Per-cell results", ""]
    for arm in arms:
        for target in TARGETS:
            lines += [f"### {arm}, target {target:g}", "",
                      "| s_por | mask (mean ± std) | corrected (mean ± std) "
                      "| corr − mask | mean \\|err\\| | gate | seam_xct "
                      "| seam_mask | degen frac |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in per_cell[f"{arm}_{target:g}"]:
                lines.append(
                    f"| {r['s_por']:g} | {r['delivered_mean']:.4f} ± "
                    f"{r['delivered_std']:.4f} | {r['corrected_mean']:.4f} ± "
                    f"{r['corrected_std']:.4f} "
                    f"| {r['corrected_minus_mask_mean']:+.4f} "
                    f"| {r['abs_error_mean']:.4f} | "
                    f"{'PASS' if r['passes_gate'] else 'FAIL'} "
                    f"({r['n_seeds_within_gate']}/{len(SEEDS)}) | "
                    f"{r['seam_xct_ratio_mean']:.2f} ± "
                    f"{r['seam_xct_ratio_std']:.2f} | "
                    f"{r['seam_mask_ratio_mean']:.2f} | "
                    f"{r['degenerate_block_fraction_mean']:.2f} |")
            lines.append("")
    if all_oom:
        lines += ["## OOM events", ""] + [f"- {e}" for e in all_oom] + [""]
    p_md = write_findings("\n".join(lines), OUT_DIR)
    print("Wrote:", p_json, p_md, *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()

"""Layup round-trip v2 — three arms, corrected conditioning semantics.

Phase 2 of the eval v2 campaign.  Same protocol as the phase-1 scripts
(arms/checkpoint/save conventions from scripts/analysis/_eval_v2.py), same
estimator machinery as scripts/analysis/layup_roundtrip.py (imported, not
forked).  The estimator-geometry validation (single 1024-px window, requested
ply edges) already PASSED on real volumes in the previous campaign
(runs/analysis/layup_roundtrip/results.json), so it is not repeated here.

Protocol: 3 layups (A_training, B_permuted, C_simple) x 3 arms
(seq, joint_legacy, joint_oob) x seeds {101, 202} = 18 volumes of
1024x1024x192 voxels (16x16x3 patch grid), DDIM-50, RAW weights at step
130000, target global porosity 0.03 as a coherent local field.  Every volume
is saved AS IT COMPLETES under runs/eval_v2/volumes/layup/<arm>/
<layup>_seed_<seed>/ (volume.tif float32, mask.tif uint8, stats.json), so a
crashed run resumes by skipping completed cells.

Per volume: T-I angle measurement (pore_axes + fft_slice + combined, direct
convention), delivered mask porosity, void-corrected porosity
(material-referenced threshold, as in phase 1), seam ratios, wall time.

Outputs: runs/eval_v2/layup/{results.json, findings.md, figures pdf+png}.

Usage
-----
    python scripts/analysis/eval_v2_layup.py                 # full run
    python scripts/analysis/eval_v2_layup.py --aggregate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style, write_findings, write_json, plt  # noqa: E402
from _eval_v2 import (  # noqa: E402
    ARMS, ARM_COLORS, CKPT, DDIM_STEPS, EVAL_ROOT,
    build_generator, configure_arm, corrected_porosity, load_existing,
    load_void_detector, save_volume,
)
import layup_roundtrip as lr  # noqa: E402  (estimator machinery, reused)

from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import theta_from_layup  # noqa: E402

import time  # noqa: E402

OUT_DIR = EVAL_ROOT / "layup"
VOL_ROOT = EVAL_ROOT / "volumes" / "layup"

LAYUPS = lr.LAYUPS                        # A_training, B_permuted, C_simple
SEEDS = [101, 202]
TARGET_POR = 0.03
VOL_SHAPE = lr.VOL_SHAPE                  # (192, 1024, 1024)
VOLUME_MM = lr.VOLUME_MM                  # (4.8, 25.6, 25.6)
PLY_VOX = lr.PLY_VOX
N_PLIES = lr.N_PLIES
GEN_BATCH = 32
DECODE_BATCH = 16
JOINT_WINDOW_STRIDE = 32
JOINT_WINDOW_BATCH = 16

ESTIMATORS = ("combined", "fft_slice", "pore_axes")
LAYUP_COLORS = lr.LAYUP_COLORS
ARM_MARKERS = {"seq": "o", "joint_legacy": "s", "joint_oob": "D"}


def generate_one(gen, mode: str, por_map: dict, joint_window_batch: int):
    """Generate one 1024x1024x192 volume; halve joint_window_batch on OOM."""
    oom_events = []
    jwb = joint_window_batch
    while True:
        try:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            with torch.no_grad():
                xct, mask, stats = gen.generate(
                    volume_size_mm=VOLUME_MM,
                    local_por_map=por_map,
                    autocast_dtype=torch.bfloat16,
                    gen_batch_size=GEN_BATCH,
                    decode_batch_size=DECODE_BATCH,
                    mode=mode,
                    joint_window_stride=JOINT_WINDOW_STRIDE,
                    joint_window_batch=jwb,
                )
            return xct, mask, stats, time.perf_counter() - t0, jwb, oom_events
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if jwb <= 1:
                raise
            oom_events.append(f"OOM at joint_window_batch={jwb}: {exc}")
            jwb //= 2
            print(f"  OOM — retrying with joint_window_batch={jwb}", flush=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_est(records: list[dict], est: str) -> dict:
    """Scores per (layup, arm), pooled per arm, and overall, one estimator."""
    out = {}
    keys = ([(ly, arm) for ly in LAYUPS for arm in ARMS]
            + [("all", arm) for arm in ARMS])
    for layup, arm in keys:
        rs = [r for r in records if r["arm"] == arm and est in r
              and (layup == "all" or r["layup"] == layup)]
        if not rs:
            continue
        de = np.concatenate([np.abs(r[est]["direct_errors_deg"]) for r in rs])
        out[f"{layup}.{arm}"] = {
            "n_volumes": len(rs),
            "n_plies": int(de.size),
            "direct_median_abs_error_deg": float(np.median(de)),
            "direct_frac_within_5": float(np.mean(de < 5.0)),
            "direct_frac_within_10": float(np.mean(de < 10.0)),
            "strict_class_accuracy": float(np.mean(
                [r[est]["strict_class_accuracy"] for r in rs])),
            "ti_class_accuracy": float(np.mean(
                [r[est]["ti_class_accuracy"] for r in rs])),
            "delivered_porosity_mean": float(np.mean(
                [r["delivered_mask_porosity"] for r in rs])),
            "corrected_porosity_mean": float(np.mean(
                [r["corrected_porosity"] for r in rs])),
        }
    recs = [r for r in records if est in r]
    if recs:
        de = np.concatenate([np.abs(r[est]["direct_errors_deg"]) for r in recs])
        meas = np.concatenate([r[est]["measured_deg"] for r in recs])
        req = np.concatenate([r[est]["requested_deg"] for r in recs])
        out["overall"] = {
            "n_volumes": len(recs),
            "n_plies": int(de.size),
            "direct_median_abs_error_deg": float(np.median(de)),
            "direct_frac_within_5": float(np.mean(de < 5.0)),
            "direct_frac_within_10": float(np.mean(de < 10.0)),
            "strict_class_accuracy": float(np.mean(
                [r[est]["strict_class_accuracy"] for r in recs])),
            "ti_class_accuracy": float(np.mean(
                [r[est]["ti_class_accuracy"] for r in recs])),
            "resultant_measured_vs_requested": float(np.abs(np.mean(
                np.exp(2j * np.deg2rad(meas - req))))),
        }
    return out


def aggregate(records: list[dict]) -> dict:
    return {est: aggregate_est(records, est) for est in ESTIMATORS}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_fig_profiles(records: list[dict], step: int) -> list[str]:
    """Requested vs measured angle per ply: rows = layups, cols = arms."""
    fig, axes = plt.subplots(len(LAYUPS), len(ARMS),
                             figsize=(4.6 * len(ARMS), 2.9 * len(LAYUPS)),
                             sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)
    for i, layup in enumerate(LAYUPS):
        req = lr.requested_sequence(LAYUPS[layup], N_PLIES)
        for j, arm in enumerate(ARMS):
            ax = axes[i][j]
            k = np.arange(N_PLIES)
            ax.step(np.append(k, N_PLIES), np.append(req, req[-1]),
                    where="post", color="0.35", lw=1.6, label="requested")
            rs = [r for r in records
                  if r["layup"] == layup and r["arm"] == arm]
            for r in rs:
                meas = np.asarray(r["combined"]["measured_deg"])
                ax.plot(k + 0.5, req + lr.wrap180(meas - req), ls="none",
                        marker="o", ms=5, color=LAYUP_COLORS[layup],
                        alpha=0.75, label=f"combined, seed {r['seed']}")
                if "pore_axes" in r:
                    mp = np.asarray(r["pore_axes"]["measured_deg"])
                    ax.plot(k + 0.5, req + lr.wrap180(mp - req), ls="none",
                            marker="^", ms=5, mfc="none",
                            color=LAYUP_COLORS[layup], alpha=0.75,
                            label=f"pore_axes, seed {r['seed']}")
            if rs:
                de = np.concatenate([np.abs(r["combined"]["direct_errors_deg"])
                                     for r in rs])
                ax.set_title(f"{layup}  |  {arm}, s_por={ARMS[arm][2]}  |  "
                             f"median |err| {np.median(de):.1f} deg")
            ax.set_yticks([-45, 0, 45, 90, 135, 180])
            ax.set_ylim(-60, 195)
            if i == len(LAYUPS) - 1:
                ax.set_xlabel("ply index (z order)")
            if j == 0:
                ax.set_ylabel("angle (deg, image frame)")
            if i == 0 and j == 0:
                ax.legend(loc="lower right", ncol=2, fontsize=6.5)
    fig.suptitle(f"Layup round-trip v2, ldm05 step {step} (RAW, "
                 f"DDIM-{DDIM_STEPS}, 1024x1024x192, combined estimator)",
                 fontsize=11)
    return savefig(fig, OUT_DIR, "layup_v2_fig1_requested_vs_measured")


def make_fig_scatter(records: list[dict], step: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    lim = [-10, 190]
    ax.plot(lim, lim, color="0.4", lw=1.0, ls="--", zorder=1)
    for gate, col in ((10.0, "0.90"), (5.0, "0.80")):
        ax.fill_between(lim, [v - gate for v in lim], [v + gate for v in lim],
                        color=col, alpha=0.6, zorder=0,
                        label=f"within {gate:.0f} deg")
    for layup in LAYUPS:
        for arm in ARMS:
            rs = [r for r in records
                  if r["layup"] == layup and r["arm"] == arm]
            if not rs:
                continue
            req = np.concatenate([r["combined"]["requested_deg"] for r in rs])
            meas = np.concatenate([r["combined"]["measured_deg"] for r in rs])
            ax.plot(req, req + lr.wrap180(meas - req), ls="none",
                    marker=ARM_MARKERS[arm], ms=5, color=LAYUP_COLORS[layup],
                    alpha=0.7, label=f"{layup} / {arm}")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Requested ply angle (deg)")
    ax.set_ylabel("Measured ply angle (deg, wrap-nearest)")
    ax.set_title(f"Per-ply angle recovery, ldm05 step {step} (combined)")
    ax.legend(loc="upper left", fontsize=7)
    return savefig(fig, OUT_DIR, "layup_v2_fig2_recovery_scatter")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def build_findings(records: list[dict], agg: dict, step: int,
                   oom: list[str], walls: dict) -> str:
    L = []
    L.append("# Layup round-trip (%s) — ldm05 step %d (RAW, DDIM-%d)\n"
             % (EVAL_ROOT.name, step, DDIM_STEPS))
    L.append("Phase 2 of the eval v2 campaign: 3 layups x 3 arms "
             "(seq s_por=1.0, joint_legacy s_por=1.5, joint_oob s_por=1.5) "
             "x seeds {101, 202} = 18 volumes of 1024x1024x192. Target "
             "porosity %.2f via the coherent field. Estimator: T-I combined "
             "(fft_slice + pore_axes), imported from layup_roundtrip.py; its "
             "single-window geometry PASSED validation on real volumes in "
             "the previous campaign (runs/analysis/layup_roundtrip/). All "
             "volumes saved under runs/eval_v2/volumes/layup/.\n" % TARGET_POR)

    L.append("## Estimator channels on generated volumes (pooled)\n")
    L.append("| estimator | direct median \\|err\\| | <=5 | <=10 | strict "
             "4-class | T-I-fit 4-class | R(meas vs req) |")
    L.append("|---|---|---|---|---|---|---|")
    for est in ESTIMATORS:
        o = agg[est].get("overall")
        if o:
            L.append("| %s | %.1f deg | %.0f%% | %.0f%% | %.1f%% | %.1f%% "
                     "| %.2f |" % (
                         est, o["direct_median_abs_error_deg"],
                         100 * o["direct_frac_within_5"],
                         100 * o["direct_frac_within_10"],
                         100 * o["strict_class_accuracy"],
                         100 * o["ti_class_accuracy"],
                         o["resultant_measured_vs_requested"]))
    L.append("")

    for est in ("combined", "pore_axes"):
        L.append("## Per (layup, arm) recovery — %s estimator\n" % est)
        L.append("| layup | arm | direct median \\|err\\| | <=5 | <=10 | "
                 "strict 4-class | delivered phi | corrected phi |")
        L.append("|---|---|---|---|---|---|---|---|")
        for layup in LAYUPS:
            for arm in ARMS:
                a = agg[est].get(f"{layup}.{arm}")
                if a:
                    L.append("| %s | %s | %.1f deg | %.0f%% | %.0f%% | %.0f%% "
                             "| %.4f | %.4f |" % (
                                 layup, arm,
                                 a["direct_median_abs_error_deg"],
                                 100 * a["direct_frac_within_5"],
                                 100 * a["direct_frac_within_10"],
                                 100 * a["strict_class_accuracy"],
                                 a["delivered_porosity_mean"],
                                 a["corrected_porosity_mean"]))
        L.append("")

    L.append("## Pooled per-arm stats (combined estimator)\n")
    L.append("| arm | n vols | direct median \\|err\\| | <=5 | <=10 | strict "
             "4-class | delivered phi | corrected phi |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        a = agg["combined"].get(f"all.{arm}")
        if a:
            L.append("| %s | %d | %.1f deg | %.0f%% | %.0f%% | %.0f%% "
                     "| %.4f | %.4f |" % (
                         arm, a["n_volumes"],
                         a["direct_median_abs_error_deg"],
                         100 * a["direct_frac_within_5"],
                         100 * a["direct_frac_within_10"],
                         100 * a["strict_class_accuracy"],
                         a["delivered_porosity_mean"],
                         a["corrected_porosity_mean"]))
    L.append("")

    L.append("## Notes and limitations\n")
    L.append("- 74/78 training volumes share layup A: recovery on B_permuted "
             "and C_simple demonstrates, not validates, generalisation.")
    L.append("- Direct scores use the frozen identity convention (no fitted "
             "offset, sign, or face reversal). The per-volume records also "
             "carry the T-I-style fitted scores.")
    L.append("- Corrected porosity = union(mask, material-referenced dark-"
             "voxel detector), the audit's recommendation for generated "
             "volumes (same method as phase 1).")
    if walls:
        L.append("- Median wall time per volume: "
                 + ", ".join(f"{a} {w:.0f}s" for a, w in walls.items()) + ".")
    L.append("- OOM events: %s"
             % (("\n  - " + "\n  - ".join(oom)) if oom else "none."))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def finalize(records: list[dict], step: int, oom: list[str]) -> None:
    agg = aggregate(records)
    walls = {arm: float(np.median([r["wall_s"] for r in records
                                   if r["arm"] == arm]))
             for arm in ARMS if any(r["arm"] == arm for r in records)}
    results = {
        "campaign": "eval_v2 phase 2 — layup round-trip, three arms",
        "checkpoint": str(CKPT),
        "step": step,
        "weights": "raw (non-EMA)",
        "ddim_steps": DDIM_STEPS,
        "volume_shape_vox": list(VOL_SHAPE),
        "grid": list(lr.GRID),
        "ply_thickness_vox": PLY_VOX,
        "n_ply_blocks": N_PLIES,
        "layups": dict(LAYUPS),
        "arms": {a: {"mode": m, "conditioning_semantics": s, "s_por": sp}
                 for a, (m, s, sp) in ARMS.items()},
        "seeds": SEEDS,
        "target_porosity": TARGET_POR,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "estimator": "T-I combined (fft_slice + pore_axes), imported",
        "geometry_validation": "PASSED in runs/analysis/layup_roundtrip/ "
                               "(agreement 0.5 deg, delta +3.7 deg)",
        "aggregate": agg,
        "oom_events": oom,
        "median_wall_s": walls,
        "records": records,
    }
    write_json(results, OUT_DIR)
    figs = (make_fig_profiles(records, step)
            + make_fig_scatter(records, step))
    write_findings(build_findings(records, agg, step, oom, walls), OUT_DIR)
    print("Wrote", OUT_DIR, "figures:", figs, flush=True)


def main() -> None:
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild results/figures/findings from saved "
                         "per-volume stats.json files")
    ap.add_argument("--joint-window-batch", type=int,
                    default=JOINT_WINDOW_BATCH)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help="seeds to generate (default %(default)s)")
    args = ap.parse_args()
    SEEDS = list(args.seeds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    if args.aggregate_only:
        records = []
        for arm in ARMS:
            for layup in LAYUPS:
                for seed in SEEDS:
                    rec = load_existing(VOL_ROOT / arm
                                        / f"{layup}_seed_{seed}")
                    if rec is not None:
                        records.append(rec)
        finalize(records, 130000, [])
        return

    device = torch.device("cuda")
    detector = load_void_detector()
    gen, model, schedule, step = build_generator(device)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    edges = lr.ply_edges(VOL_SHAPE[0], PLY_VOX)
    records: list[dict] = []
    all_oom: list[str] = []
    n_total = len(ARMS) * len(LAYUPS) * len(SEEDS)
    i = 0
    for arm in ARMS:
        mode, s_por = configure_arm(gen, model, schedule, device, arm)
        for layup_name, layup in LAYUPS.items():
            gen.theta_deg = theta_from_layup(VOL_SHAPE[0], layup, PLY_VOX)
            req = lr.requested_sequence(layup, N_PLIES)
            for seed in SEEDS:
                i += 1
                vol_dir = VOL_ROOT / arm / f"{layup_name}_seed_{seed}"
                rec = load_existing(vol_dir)
                if rec is not None:
                    records.append(rec)
                    print(f"[PHASE2 {i:2d}/{n_total}] arm={arm} "
                          f"layup={layup_name} seed={seed} (cached)",
                          flush=True)
                    continue
                torch.manual_seed(seed)
                por_map = lr.coherent_por_map(TARGET_POR, seed, te_sampler,
                                              corr_lengths)
                xct, mask, stats, wall, jwb_used, ooms = generate_one(
                    gen, mode, por_map, args.joint_window_batch)
                all_oom.extend([f"{arm} {layup_name} seed={seed}: {e}"
                                for e in ooms])
                delivered = float(stats["actual_mask_porosity"])
                corrected, thr_used = corrected_porosity(xct, mask, detector)
                est = lr.measure_volume(xct, mask, edges, PLY_VOX)
                rec = {
                    "arm": arm,
                    "mode": mode,
                    "conditioning_semantics": gen.conditioning_semantics,
                    "s_por": s_por,
                    "layup": layup_name,
                    "layup_angles": layup,
                    "seed": seed,
                    "target": TARGET_POR,
                    "delivered_mask_porosity": delivered,
                    "corrected_porosity": corrected,
                    "void_detector_threshold_u8": thr_used,
                    "seam_xct_ratio": stats["seam_xct_ratio"],
                    "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
                    "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
                    "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
                    "seam_mask_ratio": stats["seam_mask_ratio"],
                    "wall_s": round(wall, 1),
                    "joint_window_batch": jwb_used if mode == "joint" else None,
                }
                for name, e in est.items():
                    rec[name] = lr.score_recovery(e["angles"], e["weights"],
                                                  req)
                save_volume(vol_dir, xct, mask, rec)
                del xct, mask
                records.append(rec)
                c = rec["combined"]
                print(f"[PHASE2 {i:2d}/{n_total}] arm={arm} "
                      f"layup={layup_name} seed={seed} "
                      f"phi={delivered:.4f} phi_corr={corrected:.4f} "
                      f"med_err={c['direct_median_abs_error_deg']:.1f}deg "
                      f"le10={100 * c['direct_frac_within_10']:.0f}% "
                      f"strict4={100 * c['strict_class_accuracy']:.0f}% "
                      f"wall={wall:.0f}s saved={vol_dir}", flush=True)

    finalize(records, step, all_oom)


if __name__ == "__main__":
    main()

"""0b — does DDIM-200 remove interior air at 1024 scale, with the fixed decode?

Campaign 05 measured, on correctly-decoded 1024x1024x192 volumes, that the
joint_oob arm at DDIM-50 leaves an enormous amount of air the mask head does
not claim: 0.58 of the interior and 0.40 of the edge shell (layup A, seed 101).
The DDIM probe in campaign 05 tested step counts only at 192^3.  This script
tests the two together — DDIM-200 AT 1024 scale — because that is the setting
ldm06 would actually be evaluated at.

Two volumes, 192x1024x1024, ldm05 step 130000 RAW weights, joint mode with
``conditioning_semantics="specimen"``, s_por 1.5, target porosity 0.03 as a
coherent field, layup A_training, seeds 101 and 202.  Generator construction,
save layout and resume behaviour all come from ``scripts/analysis/_eval_v2.py``
and ``scripts/analysis/layup_roundtrip.py`` — imported, not forked — with the
DDIM step count raised from 50 to 200.

The audit is ``scripts/analysis/eval_v3_air_audit.py:audit_one``, imported
unchanged: absolute threshold ``T_abs = 182`` (Dice 0.842 on real volumes),
connected components below 300 voxels dropped, 32-voxel edge shell, per-64^3
cell mask capture.  The DDIM-50 row from
``runs/campaigns/05-eval-v3-fixed-decode/air_audit/per_volume.csv`` is carried
into the report next to the new numbers.

Outputs -> ``runs/campaigns/08-pre-ldm06-diagnostics/ddim200_1024/``:
volumes/<layup>_seed_<seed>/{volume.tif,mask.tif,stats.json}, per_volume.csv,
results.json, findings.md.

Usage:
    python scripts/analysis/ddim200_1024.py
    python scripts/analysis/ddim200_1024.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_findings, write_json  # noqa: E402

CAMPAIGN = REPO / "runs/campaigns/08-pre-ldm06-diagnostics"
OUT_DIR = CAMPAIGN / "ddim200_1024"
VOL_ROOT = OUT_DIR / "volumes"

# _eval_v2 resolves its campaign root at import time; point it here so nothing
# can be written into runs/campaigns/05-eval-v3-fixed-decode.
os.environ["POREGEN_EVAL_ROOT"] = str(CAMPAIGN)

from _eval_v2 import (  # noqa: E402
    CKPT, build_generator, configure_arm, corrected_porosity, load_existing,
    load_void_detector, save_volume,
)
import layup_roundtrip as lr  # noqa: E402
from eval_v3_air_audit import audit_one  # noqa: E402
from _eval_v3 import ROOT as V3_ROOT, load_calibration  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.noise_schedule import DDPMSchedule  # noqa: E402
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import DDIMSampler, theta_from_layup  # noqa: E402

ARM = "joint_oob"                   # joint mode, "specimen" semantics, s_por 1.5
DDIM_STEPS = 200
LAYUP_NAME = "A_training"
SEEDS = [101, 202]
TARGET_POR = 0.03
VOL_SHAPE = lr.VOL_SHAPE             # (192, 1024, 1024)
VOLUME_MM = lr.VOLUME_MM
PLY_VOX = lr.PLY_VOX
GEN_BATCH = 32
DECODE_BATCH = 16
JOINT_WINDOW_STRIDE = 32
JOINT_WINDOW_BATCH = 16

# The DDIM-50 comparison row, read from campaign 05 (never written).
V3_PER_VOLUME = V3_ROOT / "air_audit" / "per_volume.csv"
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate() -> list[dict]:
    device = torch.device("cuda")
    detector = load_void_detector()
    gen, model, schedule, step = build_generator(device)
    mode, s_por = configure_arm(gen, model, schedule, device, ARM)

    # configure_arm built the sampler at the module default of 50 steps.
    gen.sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS,
                              s_por=s_por, s_nb=1.0)
    layup = lr.LAYUPS[LAYUP_NAME]
    gen.theta_deg = theta_from_layup(VOL_SHAPE[0], layup, PLY_VOX)
    log(f"generator ready: step {step}, mode={mode}, "
        f"semantics={gen.conditioning_semantics}, s_por={s_por}, "
        f"DDIM-{DDIM_STEPS}, layup {LAYUP_NAME}")

    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    records = []
    for seed in SEEDS:
        vol_dir = VOL_ROOT / f"{LAYUP_NAME}_seed_{seed}"
        cached = load_existing(vol_dir)
        if cached is not None:
            log(f"seed {seed}: cached")
            records.append(cached)
            continue
        torch.manual_seed(seed)
        por_map = lr.coherent_por_map(TARGET_POR, seed, te_sampler, corr_lengths)
        xct, mask, stats, wall, jwb, ooms = lr.generate_one(
            gen, mode, por_map, JOINT_WINDOW_BATCH)
        delivered = float(stats["actual_mask_porosity"])
        corrected, thr = corrected_porosity(xct, mask, detector)
        rec = {
            "arm": ARM, "mode": mode,
            "conditioning_semantics": gen.conditioning_semantics,
            "s_por": s_por, "ddim_steps": DDIM_STEPS,
            "layup": LAYUP_NAME, "layup_angles": layup, "seed": seed,
            "target": TARGET_POR,
            "delivered_mask_porosity": delivered,
            "corrected_porosity": corrected,
            "void_detector_threshold_u8": thr,
            "seam_xct_ratio": stats["seam_xct_ratio"],
            "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
            "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
            "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
            "seam_mask_ratio": stats["seam_mask_ratio"],
            "wall_s": round(wall, 1),
            "joint_window_batch": jwb,
            "oom_events": ooms,
            "volume_shape_vox": list(VOL_SHAPE),
            "checkpoint": str(CKPT), "step": step, "weights": "raw (non-EMA)",
        }
        save_volume(vol_dir, xct, mask, rec)
        del xct, mask
        records.append(rec)
        log(f"seed {seed}: phi={delivered:.4f} phi_corr={corrected:.4f} "
            f"seam_xct={rec['seam_xct_ratio']:.3f} "
            f"seam_mask={rec['seam_mask_ratio']:.3f} wall={wall:.0f}s")
    return records


# ---------------------------------------------------------------------------
# Air audit + report
# ---------------------------------------------------------------------------

REPORT_COLS = ["unmasked_air_abs", "unmasked_interior_local_abs",
               "unmasked_edge_local_abs", "largest_comp_equiv_diam_mm",
               "mask_capture_of_detected", "detected_air_abs",
               "mask_porosity"]


def reference_row() -> dict | None:
    """DDIM-50 joint_oob layup A seed 101, from campaign 05."""
    if not V3_PER_VOLUME.exists():
        return None
    df = pd.read_csv(V3_PER_VOLUME)
    s = df[(df.experiment == "layup") & (df.arm == "joint_oob")
           & (df.name == f"{LAYUP_NAME}_seed_101")]
    if not len(s):
        return None
    r = s.iloc[0]
    return {"label": "DDIM-50 (campaign 05)", "name": r["name"],
            **{c: float(r[c]) for c in REPORT_COLS}}


def audit() -> tuple[pd.DataFrame, dict]:
    cal = load_calibration()
    t_abs = cal["t_abs"]
    rows, cells = [], []
    for seed in SEEDS:
        d = VOL_ROOT / f"{LAYUP_NAME}_seed_{seed}"
        if not (d / "volume.tif").exists():
            continue
        log(f"auditing {d.name} at T_abs={t_abs}")
        row, cell, _ = audit_one("ddim200_1024", ARM, d, t_abs)
        rows.append(row)
        cells.append(cell)
        log(f"  unmasked={row['unmasked_air_abs']:.4f} "
            f"int={row['unmasked_interior_local_abs']:.4f} "
            f"edge={row['unmasked_edge_local_abs']:.4f} "
            f"capture={row['mask_capture_of_detected']:.4f}")
    pv = pd.DataFrame(rows)
    pv.to_csv(OUT_DIR / "per_volume.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(OUT_DIR / "per_cell.csv",
                                               index=False)
    return pv, cal


def build_findings(pv: pd.DataFrame, gen_records: list[dict], cal: dict,
                   ref: dict | None) -> str:
    L = [f"# 0b — DDIM-{DDIM_STEPS} at 1024 scale, fixed decode", "",
         f"{len(pv)} volumes of {VOL_SHAPE[1]}x{VOL_SHAPE[2]}x{VOL_SHAPE[0]} "
         f"(y,x,z), ldm05 step 130000 RAW, joint mode with "
         f"`conditioning_semantics=\"specimen\"`, s_por 1.5, target phi "
         f"{TARGET_POR} as a coherent field, layup {LAYUP_NAME}, "
         f"DDIM-{DDIM_STEPS}, seeds {SEEDS}.", "",
         f"Air detector (imported from `eval_v3_air_audit.audit_one`): "
         f"`u8 < {cal['t_abs']}`, components below 300 voxels dropped, "
         f"32-voxel edge shell. Calibrated on "
         f"{cal['n_calibration_volumes']} real volumes "
         f"(Dice {cal['dice']:.3f}).", "",
         "## Air audit", "",
         "| volume | unmasked air | interior (local) | edge (local) "
         "| largest comp. diam. | mask capture | detected air | mask phi |",
         "|---|---|---|---|---|---|---|---|"]

    def fmt(label, r):
        return (f"| {label} | {r['unmasked_air_abs']:.4f} "
                f"| {r['unmasked_interior_local_abs']:.4f} "
                f"| {r['unmasked_edge_local_abs']:.4f} "
                f"| {r['largest_comp_equiv_diam_mm']:.2f} mm "
                f"| {r['mask_capture_of_detected']:.4f} "
                f"| {r['detected_air_abs']:.4f} "
                f"| {r['mask_porosity']:.4f} |")

    for _, r in pv.iterrows():
        L.append(fmt(f"DDIM-{DDIM_STEPS} {r['name']}", r))
    if ref is not None:
        L.append(fmt(f"{ref['label']} {ref['name']}", ref))
    L += ["", "## Generation", "",
          "| seed | delivered phi | corrected phi | seam_xct | seam_mask "
          "| wall time |", "|---|---|---|---|---|---|"]
    for g in gen_records:
        L.append(f"| {g['seed']} | {g['delivered_mask_porosity']:.4f} "
                 f"| {g['corrected_porosity']:.4f} "
                 f"| {g['seam_xct_ratio']:.3f} | {g['seam_mask_ratio']:.3f} "
                 f"| {g['wall_s'] / 60:.0f} min |")
    L += ["", "## Caveats", "",
          "- The comparison row is a single DDIM-50 volume (layup A, seed "
          "101) from campaign 05, not a matched pair; seed 202 was not "
          "generated at DDIM-50 for layup A at 1024 scale.",
          "- `mask capture` is the share of detected air that the mask head "
          "does claim. Near zero means the mask head is not describing the "
          "air the grey channel renders.",
          "- The detector's real-volume false-positive floor is "
          f"{cal['real_false_positive_baseline']['mean']:.4f}; every "
          "generated number must be read against it.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true",
                    help="skip generation; audit the volumes already saved")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.audit_only:
        gen_records = [load_existing(VOL_ROOT / f"{LAYUP_NAME}_seed_{s}")
                       for s in SEEDS]
        gen_records = [r for r in gen_records if r is not None]
    else:
        gen_records = generate()

    pv, cal = audit()
    ref = reference_row()
    write_json({
        "campaign": f"08 — 0b DDIM-{DDIM_STEPS} at 1024 scale",
        "question": f"Does DDIM-{DDIM_STEPS} remove interior air at 1024 "
                    "scale with the fixed decode?",
        "checkpoint": str(CKPT),
        "weights": "raw (non-EMA)",
        "arm": ARM, "ddim_steps": DDIM_STEPS, "layup": LAYUP_NAME,
        "seeds": SEEDS, "target_porosity": TARGET_POR,
        "volume_shape_vox": list(VOL_SHAPE),
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "audit_method": "scripts/analysis/eval_v3_air_audit.py:audit_one "
                        "(imported unchanged)",
        "calibration": cal,
        "reference_ddim50": ref,
        "generation": gen_records,
        "per_volume": pv.drop(columns=["top_components", "pore_stats",
                                       "material_stats"],
                              errors="ignore").to_dict("records"),
    }, OUT_DIR)
    write_findings(build_findings(pv, gen_records, cal, ref), OUT_DIR)
    log(f"wrote {OUT_DIR}/results.json, per_volume.csv and findings.md")


if __name__ == "__main__":
    main()

"""Porosity dose-response evaluation for the completed ldm05 run (130k, RAW).

Measures delivered vs requested global porosity across the training phi range
[0.002, 0.107] for BOTH generation modes (sequential, joint), 3 seeds per
cell, at the final checkpoint (step 130000, RAW non-EMA weights — matching
runs/analysis/joint_vs_sequential/compare_modes.py).  Seam metrics are
recorded per volume so the sequential-vs-joint comparison gets seed
statistics too.

Local per-patch porosity targets use the coherent field (D32 §4):
T-E marginal draw, T-D correlation smoothing, mean rescaled to the target
(src/poregen/diffusion/porosity_field.py), converted to the (iz, iy, ix)
dict that VolumeGenerator.generate takes as ``local_por_map``.

Protocol: targets {0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10} x
{sequential, joint} x seeds {101, 202, 303} = 42 volumes of 192^3 voxels
(3x3x3 patch grid), DDIM-50, joint window stride 32.  Gate (D39):
|delivered - requested| < 0.005.

Outputs: runs/analysis/dose_response/{results.json, findings.md, figures}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style, write_findings, write_json, plt  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.conditioning import resolve_group_order  # noqa: E402
from poregen.diffusion.noise_schedule import DDPMSchedule  # noqa: E402
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS, build_porosity_field,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator, theta_from_layup  # noqa: E402
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402

CKPT = REPO / ("runs/ldm/ldm05-run-0001-20260827-114902-z4-c128-bs256-lr1e-04/"
               "checkpoints/ldm_step00130000.ckpt")
LATENTS_ROOT = REPO / "data/split_v2/latents_r07z4"
OUT_DIR = REPO / "runs/analysis/dose_response"

TARGETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
MODES = ["sequential", "joint"]
SEEDS = [101, 202, 303]
VOLUME_MM = (4.8, 4.8, 4.8)              # 192 vox per axis at 0.025 mm/vox
VOLUME_VOX = 192
GRID = (3, 3, 3)                          # 3x3x3 patch grid, stride 64
LAYUP = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_VOX = 19.6
DDIM_STEPS = 50
GEN_BATCH = 27                            # GPU is free: whole grid in one batch
DECODE_BATCH = 27
JOINT_WINDOW_STRIDE = 32
GATE = 0.005                              # D39: |delivered - requested| < 0.005
PHI_RANGE = (0.002, 0.107)

MODE_COLORS = {"sequential": "#1b6ca8", "joint": "#c2571a"}


def build_generator(device: torch.device, s_por: float = 1.0):
    run_dir = CKPT.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    model = UNet3DDenoiser(UNet3DConfig.from_cfg(cfg)).to(device)
    step, _ = load_checkpoint(str(CKPT), model=model, map_location=device,
                              restore_rng=False)   # RAW weights, not EMA
    model.eval()
    print(f"Loaded RAW (non-EMA) weights at step {step}", flush=True)

    meta = json.loads((LATENTS_ROOT / "metadata.json").read_text())
    norm = meta["normalization"]
    c = len(norm["per_channel_mean"])
    latent_mean = torch.tensor(norm["per_channel_mean"], dtype=torch.float32).view(c, 1, 1, 1)
    latent_std = torch.tensor(norm["per_channel_std"], dtype=torch.float32).view(c, 1, 1, 1)
    vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), device)
    vae.requires_grad_(False)

    schedule = DDPMSchedule(T=1000, s=0.008, device=device)
    sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS,
                          s_por=s_por, s_nb=1.0)
    _st = meta["conditioning"]["por_standardisation"]
    theta = theta_from_layup(VOLUME_VOX, LAYUP, PLY_VOX)

    gen = VolumeGenerator(
        sampler=sampler, vae=vae, device=device,
        patch_size=64, generation_stride=64, neighbour_offset=64,
        latent_size=16, latent_mean=latent_mean, latent_std=latent_std,
        voxel_size_mm=0.025,
        por_log_stats=(float(_st["mean"]), float(_st["std"])),
        theta_deg=theta,
        group_order=resolve_group_order(meta),
    )
    return gen, step


def coherent_por_map(target: float, seed: int, te_sampler, corr_lengths):
    field = build_porosity_field(
        grid_shape=GRID, target=target, sampler=te_sampler,
        corr_lengths_voxels=corr_lengths, stride_voxels=64, seed=seed,
    )
    por_map = {
        (iz, iy, ix): float(field[iz, iy, ix])
        for iz in range(GRID[0]) for iy in range(GRID[1]) for ix in range(GRID[2])
    }
    return por_map, field


def run_one(gen, mode: str, por_map: dict, joint_window_batch: int):
    """Generate one volume; on CUDA OOM halve joint_window_batch and retry."""
    oom_events = []
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
                    joint_window_batch=joint_window_batch,
                )
            wall = time.perf_counter() - t0
            del xct, mask
            return stats, wall, joint_window_batch, oom_events
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if joint_window_batch <= 1:
                raise
            oom_events.append(
                f"OOM at joint_window_batch={joint_window_batch}: {exc}")
            joint_window_batch //= 2
            print(f"  OOM — retrying with joint_window_batch={joint_window_batch}",
                  flush=True)


def fit_ols(requested: np.ndarray, delivered: np.ndarray) -> dict:
    slope, intercept = np.polyfit(requested, delivered, 1)
    pred = slope * requested + intercept
    ss_res = float(np.sum((delivered - pred) ** 2))
    ss_tot = float(np.sum((delivered - delivered.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-window-batch", type=int, default=16)
    ap.add_argument("--s-por", type=float, default=1.0)
    ap.add_argument("--modes", nargs="+", default=None, choices=["sequential", "joint"])
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    global OUT_DIR, MODES
    if args.out_suffix:
        OUT_DIR = OUT_DIR.parent / (OUT_DIR.name + args.out_suffix)
    if args.modes:
        MODES = list(args.modes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device("cuda")

    gen, step = build_generator(device, s_por=args.s_por)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    records = []
    all_oom = []
    jwb = args.joint_window_batch
    n_total = len(TARGETS) * len(MODES) * len(SEEDS)
    i = 0
    for target in TARGETS:
        for mode in MODES:
            for seed in SEEDS:
                i += 1
                torch.manual_seed(seed)
                por_map, field = coherent_por_map(target, seed, te_sampler,
                                                  corr_lengths)
                stats, wall, jwb_used, ooms = run_one(gen, mode, por_map, jwb)
                all_oom.extend(ooms)
                delivered = float(stats["actual_mask_porosity"])
                rec = {
                    "target": target,
                    "mode": mode,
                    "seed": seed,
                    "delivered_mask_porosity": delivered,
                    "abs_error": abs(delivered - target),
                    "field_mean": float(field.mean()),
                    "field_min": float(field.min()),
                    "field_max": float(field.max()),
                    "seam_xct_ratio": stats["seam_xct_ratio"],
                    "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
                    "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
                    "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
                    "seam_mask_ratio": stats["seam_mask_ratio"],
                    "wall_s": round(wall, 1),
                    "joint_window_batch": jwb_used if mode == "joint" else None,
                }
                records.append(rec)
                print(f"[{i:2d}/{n_total}] target={target:.3f} {mode:10s} "
                      f"seed={seed}  delivered={delivered:.4f} "
                      f"|err|={rec['abs_error']:.4f}  "
                      f"seam_xct={stats['seam_xct_ratio']:.3f}  "
                      f"wall={wall:.1f}s", flush=True)
                # Persist incrementally so a crash keeps completed cells.
                write_json({"partial": True, "records": records}, OUT_DIR)

    # ---------------- analysis ----------------
    fits = {}
    per_level = {}
    for mode in MODES:
        rs = [r for r in records if r["mode"] == mode]
        req = np.array([r["target"] for r in rs])
        dlv = np.array([r["delivered_mask_porosity"] for r in rs])
        fits[mode] = fit_ols(req, dlv)
        per_level[mode] = []
        for target in TARGETS:
            ls = [r for r in rs if r["target"] == target]
            d = np.array([r["delivered_mask_porosity"] for r in ls])
            e = np.array([r["abs_error"] for r in ls])
            sx = np.array([r["seam_xct_ratio"] for r in ls])
            sm = np.array([r["seam_mask_ratio"] for r in ls])
            per_level[mode].append({
                "target": target,
                "delivered_mean": float(d.mean()),
                "delivered_std": float(d.std(ddof=1)),
                "abs_error_mean": float(e.mean()),
                "passes_gate": bool(e.mean() < GATE),
                "n_seeds_within_gate": int((e < GATE).sum()),
                "seam_xct_ratio_mean": float(sx.mean()),
                "seam_xct_ratio_std": float(sx.std(ddof=1)),
                "seam_mask_ratio_mean": float(sm.mean()),
            })

    results = {
        "checkpoint": str(CKPT),
        "step": step,
        "weights": "raw (non-EMA), matching compare_modes.py",
        "ddim_steps": DDIM_STEPS,
        "volume_vox": VOLUME_VOX,
        "grid": list(GRID),
        "targets": TARGETS,
        "modes": MODES,
        "seeds": SEEDS,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "gen_batch": GEN_BATCH,
        "gate_abs_error": GATE,
        "training_phi_range": list(PHI_RANGE),
        "porosity_field": "coherent (T-E marginal + T-D smoothing), "
                          "seeded per (torch seed == field seed)",
        "oom_events": all_oom,
        "fits": fits,
        "per_level": per_level,
        "records": records,
    }
    p_json = write_json(results, OUT_DIR)

    # ---------------- figures ----------------
    # (a) delivered vs requested
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ymax = max(0.115, max(r["delivered_mask_porosity"] for r in records) + 0.008)
    lim = [0.0, ymax]
    ax.plot(lim, lim, color="0.4", lw=1.0, ls="--", zorder=1,
            label="identity (delivered = requested)")
    ax.fill_between(lim, [v - GATE for v in lim], [v + GATE for v in lim],
                    color="0.85", alpha=0.5, zorder=0,
                    label=f"gate ±{GATE}")
    for mode in MODES:
        col = MODE_COLORS[mode]
        rs = [r for r in records if r["mode"] == mode]
        ax.scatter([r["target"] for r in rs],
                   [r["delivered_mask_porosity"] for r in rs],
                   s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
        means = [pl["delivered_mean"] for pl in per_level[mode]]
        stds = [pl["delivered_std"] for pl in per_level[mode]]
        ax.errorbar(TARGETS, means, yerr=stds, color=col, marker="o", ms=5,
                    lw=1.4, capsize=3, zorder=3,
                    label=f"{mode} (slope {fits[mode]['slope']:.2f}, "
                          f"R² {fits[mode]['r2']:.3f})")
    ax.set_xlim(0.0, 0.115)
    ax.set_ylim(lim)
    ax.set_xlabel("Requested global porosity")
    ax.set_ylabel("Delivered mask porosity")
    ax.set_title(f"Dose response, ldm05 step {step} (RAW, DDIM-{DDIM_STEPS}, "
                 f"192³, {len(SEEDS)} seeds)")
    ax.legend(loc="upper left")
    figs = savefig(fig, OUT_DIR, "dose_fig1_delivered_vs_requested")

    # (b) seam_xct_ratio vs target
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="no seam excess (ratio 1)")
    for mode in MODES:
        col = MODE_COLORS[mode]
        rs = [r for r in records if r["mode"] == mode]
        ax.scatter([r["target"] for r in rs],
                   [r["seam_xct_ratio"] for r in rs],
                   s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
        means = [pl["seam_xct_ratio_mean"] for pl in per_level[mode]]
        stds = [pl["seam_xct_ratio_std"] for pl in per_level[mode]]
        ax.errorbar(TARGETS, means, yerr=stds, color=col, marker="o", ms=5,
                    lw=1.4, capsize=3, zorder=3, label=mode)
    ax.set_xlabel("Requested global porosity")
    ax.set_ylabel("seam_xct_ratio (seam MAD / interior MAD)")
    ax.set_title("Seam severity vs porosity target")
    ax.legend(loc="best")
    figs += savefig(fig, OUT_DIR, "dose_fig2_seam_vs_target")

    # ---------------- findings ----------------
    lines = [
        "# Dose-response evaluation — ldm05, step {} (RAW weights)".format(step),
        "",
        f"Checkpoint: `{CKPT}`. DDIM-{DDIM_STEPS}, 192³ voxels "
        f"(3×3×3 patch grid), coherent local-porosity field, "
        f"seeds {SEEDS}. Gate: |delivered − requested| < {GATE}.",
        "",
        "## Fits (delivered vs requested, OLS over all seeds)",
        "",
        "| mode | slope | intercept | R² |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        f = fits[mode]
        lines.append(f"| {mode} | {f['slope']:.3f} | {f['intercept']:.4f} "
                     f"| {f['r2']:.4f} |")
    lines += ["", "## Per-level results", ""]
    for mode in MODES:
        lines += [f"### {mode}", "",
                  "| target | delivered (mean ± std) | mean |err| | gate "
                  "| seam_xct (mean ± std) | seam_mask |",
                  "|---|---|---|---|---|---|"]
        for pl in per_level[mode]:
            lines.append(
                f"| {pl['target']:.3f} | {pl['delivered_mean']:.4f} ± "
                f"{pl['delivered_std']:.4f} | {pl['abs_error_mean']:.4f} | "
                f"{'PASS' if pl['passes_gate'] else 'FAIL'} "
                f"({pl['n_seeds_within_gate']}/{len(SEEDS)} seeds) | "
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

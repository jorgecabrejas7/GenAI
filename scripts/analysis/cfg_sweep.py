"""CFG porosity-guidance-scale sweep for the completed ldm05 run (130k, RAW).

The dose-response evaluation (runs/analysis/dose_response/) found two
calibration errors at s_por=1.0: sequential mode overshoots proportionally
(slope 1.261) and joint mode carries a constant offset (slope 1.018,
intercept +0.0064).  This sweep varies the porosity classifier-free-guidance
scale s_por (src/poregen/diffusion/sampler.py, DDIMSampler; contract in
tests/test_cfg_guidance.py) to ask whether either error can be fixed at the
source, and at what cost in sample quality (seam metrics, degenerate blocks).

Protocol: s_por {0.0, 0.5, 1.0, 1.5, 2.0, 3.0} x targets {0.02, 0.05} x
modes {sequential, joint} x seeds {101, 202, 303} = 72 volumes of 192^3,
DDIM-50, joint window stride 32.  s_por=0.0 is the unconditional-on-porosity
baseline; s_por=1.0 is plain conditional (single-pass).  Coherent
local-porosity field as in dose_response.py.  Gate (D39):
|delivered - requested| < 0.005.

Outputs: runs/analysis/cfg_sweep/{results.json, findings.md, figures}.
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
OUT_DIR = REPO / "runs/analysis/cfg_sweep"

S_POR_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
TARGETS = [0.02, 0.05]
MODES = ["sequential", "joint"]
SEEDS = [101, 202, 303]
VOLUME_MM = (4.8, 4.8, 4.8)              # 192 vox per axis at 0.025 mm/vox
VOLUME_VOX = 192
GRID = (3, 3, 3)                          # 3x3x3 patch grid, stride 64
LAYUP = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_VOX = 19.6
DDIM_STEPS = 50
GEN_BATCH = 27
DECODE_BATCH = 27
JOINT_WINDOW_STRIDE = 32
GATE = 0.005                              # D39: |delivered - requested| < 0.005
DEGEN_THR = 1e-4                          # 64^3 subblock porosity below this

MODE_COLORS = {"sequential": "#1b6ca8", "joint": "#c2571a"}


def build_generator(device: torch.device):
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
                          s_por=1.0, s_nb=1.0)
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
    return gen, model, schedule, step


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


def degenerate_fraction(mask: np.ndarray) -> float:
    """Fraction of the 27 stride-64 64^3 subblocks with porosity < DEGEN_THR."""
    n_degen = 0
    n_tot = 0
    for iz in range(GRID[0]):
        for iy in range(GRID[1]):
            for ix in range(GRID[2]):
                blk = mask[iz * 64:(iz + 1) * 64,
                           iy * 64:(iy + 1) * 64,
                           ix * 64:(ix + 1) * 64]
                if float((blk > 0).mean()) < DEGEN_THR:
                    n_degen += 1
                n_tot += 1
    return n_degen / n_tot


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
            degen = degenerate_fraction(mask)
            del xct, mask
            return stats, degen, wall, joint_window_batch, oom_events
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if joint_window_batch <= 1:
                raise
            oom_events.append(
                f"OOM at joint_window_batch={joint_window_batch}: {exc}")
            joint_window_batch //= 2
            print(f"  OOM — retrying with joint_window_batch={joint_window_batch}",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-window-batch", type=int, default=16)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device("cuda")

    gen, model, schedule, step = build_generator(device)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    records = []
    all_oom = []
    jwb = args.joint_window_batch
    n_total = len(S_POR_GRID) * len(TARGETS) * len(MODES) * len(SEEDS)
    i = 0
    for s_por in S_POR_GRID:
        # Swap in a sampler with this guidance scale (s_nb stays 1.0; it has
        # no effect in joint mode and is not under study here).
        gen.sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS,
                                  s_por=s_por, s_nb=1.0)
        for target in TARGETS:
            for mode in MODES:
                for seed in SEEDS:
                    i += 1
                    torch.manual_seed(seed)
                    por_map, field = coherent_por_map(target, seed, te_sampler,
                                                      corr_lengths)
                    stats, degen, wall, jwb_used, ooms = run_one(
                        gen, mode, por_map, jwb)
                    all_oom.extend(ooms)
                    delivered = float(stats["actual_mask_porosity"])
                    rec = {
                        "s_por": s_por,
                        "target": target,
                        "mode": mode,
                        "seed": seed,
                        "delivered_mask_porosity": delivered,
                        "abs_error": abs(delivered - target),
                        "field_mean": float(field.mean()),
                        "seam_xct_ratio": stats["seam_xct_ratio"],
                        "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
                        "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
                        "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
                        "seam_mask_ratio": stats["seam_mask_ratio"],
                        "degenerate_block_fraction": degen,
                        "wall_s": round(wall, 1),
                        "joint_window_batch": jwb_used if mode == "joint" else None,
                    }
                    records.append(rec)
                    print(f"[{i:2d}/{n_total}] s_por={s_por:.1f} "
                          f"target={target:.2f} {mode:10s} seed={seed}  "
                          f"delivered={delivered:.4f} "
                          f"|err|={rec['abs_error']:.4f}  "
                          f"seam_xct={stats['seam_xct_ratio']:.2f}  "
                          f"degen={degen:.2f}  wall={wall:.1f}s", flush=True)
                    # Persist incrementally so a crash keeps completed cells.
                    write_json({"partial": True, "records": records}, OUT_DIR)

    # ---------------- analysis ----------------
    per_cell = {}     # (mode, target) -> list over s_por of aggregates
    best = {}         # (mode, target) -> best s_por by mean |error|
    for mode in MODES:
        for target in TARGETS:
            key = f"{mode}_{target:g}"
            rows = []
            for s_por in S_POR_GRID:
                ls = [r for r in records
                      if r["mode"] == mode and r["target"] == target
                      and r["s_por"] == s_por]
                d = np.array([r["delivered_mask_porosity"] for r in ls])
                e = np.array([r["abs_error"] for r in ls])
                sx = np.array([r["seam_xct_ratio"] for r in ls])
                sm = np.array([r["seam_mask_ratio"] for r in ls])
                dg = np.array([r["degenerate_block_fraction"] for r in ls])
                rows.append({
                    "s_por": s_por,
                    "delivered_mean": float(d.mean()),
                    "delivered_std": float(d.std(ddof=1)),
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
        "weights": "raw (non-EMA), matching dose_response.py",
        "ddim_steps": DDIM_STEPS,
        "volume_vox": VOLUME_VOX,
        "grid": list(GRID),
        "s_por_grid": S_POR_GRID,
        "targets": TARGETS,
        "modes": MODES,
        "seeds": SEEDS,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "gen_batch": GEN_BATCH,
        "gate_abs_error": GATE,
        "degenerate_threshold": DEGEN_THR,
        "porosity_field": "coherent (T-E marginal + T-D smoothing), "
                          "seeded per (torch seed == field seed)",
        "oom_events": all_oom,
        "per_cell": per_cell,
        "best_s_por": best,
        "records": records,
    }
    p_json = write_json(results, OUT_DIR)

    # ---------------- figures ----------------
    # (a) delivered vs s_por, one panel per target, both modes
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True,
                             sharex=True)
    for ax, target in zip(axes, TARGETS):
        ax.axhline(target, color="0.4", lw=1.0, ls="--",
                   label=f"target {target:g}")
        ax.axhspan(target - GATE, target + GATE, color="0.85", alpha=0.5,
                   zorder=0, label=f"gate ±{GATE}")
        for mode in MODES:
            col = MODE_COLORS[mode]
            rs = [r for r in records
                  if r["mode"] == mode and r["target"] == target]
            ax.scatter([r["s_por"] for r in rs],
                       [r["delivered_mask_porosity"] for r in rs],
                       s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
            rows = per_cell[f"{mode}_{target:g}"]
            ax.errorbar(S_POR_GRID,
                        [r["delivered_mean"] for r in rows],
                        yerr=[r["delivered_std"] for r in rows],
                        color=col, marker="o", ms=5, lw=1.4, capsize=3,
                        zorder=3, label=mode)
        ax.set_xlabel("Porosity guidance scale $s_{por}$")
        ax.set_title(f"target = {target:g}")
    axes[0].set_ylabel("Delivered mask porosity")
    axes[0].legend(loc="best")
    fig.suptitle(f"Delivered porosity vs $s_{{por}}$ — ldm05 step {step} "
                 f"(RAW, DDIM-{DDIM_STEPS}, 192³, {len(SEEDS)} seeds)")
    figs = savefig(fig, OUT_DIR, "cfg_fig1_delivered_vs_spor")

    # (b) seam_xct_ratio vs s_por
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--",
                   label="no seam excess (ratio 1)")
        for mode in MODES:
            col = MODE_COLORS[mode]
            rs = [r for r in records
                  if r["mode"] == mode and r["target"] == target]
            ax.scatter([r["s_por"] for r in rs],
                       [r["seam_xct_ratio"] for r in rs],
                       s=16, color=col, alpha=0.45, edgecolors="none", zorder=2)
            rows = per_cell[f"{mode}_{target:g}"]
            ax.errorbar(S_POR_GRID,
                        [r["seam_xct_ratio_mean"] for r in rows],
                        yerr=[r["seam_xct_ratio_std"] for r in rows],
                        color=col, marker="o", ms=5, lw=1.4, capsize=3,
                        zorder=3, label=mode)
        ax.set_xlabel("Porosity guidance scale $s_{por}$")
        ax.set_title(f"target = {target:g}")
    axes[0].set_ylabel("seam_xct_ratio (seam MAD / interior MAD)")
    axes[0].legend(loc="best")
    fig.suptitle("Seam severity vs guidance scale")
    figs += savefig(fig, OUT_DIR, "cfg_fig2_seam_vs_spor")

    # ---------------- findings ----------------
    lines = [
        f"# CFG guidance-scale sweep — ldm05, step {step} (RAW weights)",
        "",
        f"Checkpoint: `{CKPT}`. DDIM-{DDIM_STEPS}, 192³ voxels "
        f"(3×3×3 patch grid), coherent local-porosity field, seeds {SEEDS}. "
        f"s_por grid {S_POR_GRID} (0.0 = porosity-unconditional baseline, "
        f"1.0 = plain conditional). Gate: |delivered − requested| < {GATE}.",
        "",
        "## Best s_por per (mode, target) by mean |error|",
        "",
        "| mode | target | best s_por | mean \\|err\\| | gate |",
        "|---|---|---|---|---|",
    ]
    for mode in MODES:
        for target in TARGETS:
            b = best[f"{mode}_{target:g}"]
            lines.append(f"| {mode} | {target:g} | {b['s_por']:g} "
                         f"| {b['abs_error_mean']:.4f} "
                         f"| {'PASS' if b['passes_gate'] else 'FAIL'} |")
    lines += ["", "## Per-cell results", ""]
    for mode in MODES:
        for target in TARGETS:
            lines += [f"### {mode}, target {target:g}", "",
                      "| s_por | delivered (mean ± std) | mean \\|err\\| | gate "
                      "| seam_xct (mean ± std) | seam_mask | degen frac |",
                      "|---|---|---|---|---|---|---|"]
            for r in per_cell[f"{mode}_{target:g}"]:
                lines.append(
                    f"| {r['s_por']:g} | {r['delivered_mean']:.4f} ± "
                    f"{r['delivered_std']:.4f} | {r['abs_error_mean']:.4f} | "
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

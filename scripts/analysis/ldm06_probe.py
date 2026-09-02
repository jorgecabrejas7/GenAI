"""ldm06 design probe (vault note "PoreGen - ldm06 Design", D40).

Two questions, answered before any ldm06 training:

A. Are the dark (air) cells in generated volumes off-manifold latents?
   Re-encode high-air (>15% dark-air) and low-air (<1%) 64^3 cells from the
   saved dose-response volumes through the frozen r07-z4 VAE encoder and
   compare per-channel latent moments against the real-latent normalisation
   stats in data/split_v2/latents_r07z4/metadata.json.

B. Does more DDIM sampling reduce interior air?
   Generate 192^3 volumes with the ldm05 130k RAW checkpoint (joint mode,
   specimen semantics, s_por 1.5, target 0.03, coherent field) at DDIM steps
   {50, 100, 200, 300} x seeds {101, 202} and measure interior vs edge
   unmasked air with the calibrated detector from the eval-v2 audit
   (T_best 185 / T_cons 178 on the decoder-native u8 scale, min CC 300 vox).
   If the step count gives a clear monotone air reduction, one extra
   1024x1024x192 volume at 200 steps checks the effect at full scale.

Outputs to runs/analysis/ldm06_probe/: results.json, findings.md, figures
(PDF + PNG, 300 dpi), and the generated volumes under
volumes/<steps>_seed_<seed>/ (generate_volumes.py TIFF conventions).

Usage:
    python scripts/analysis/ldm06_probe.py            # both parts
    python scripts/analysis/ldm06_probe.py --part a
    python scripts/analysis/ldm06_probe.py --part b [--jwb 16]
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    REPO, plt, savefig, set_style, write_findings, write_json,
)
from eval_v2_audit import cc_filter, edge_shell, to_native_u8  # noqa: E402
from _eval_v2 import (  # noqa: E402
    CKPT, LATENTS_ROOT, build_generator, load_existing, save_volume,
)

sys.path.insert(0, str(REPO / "src"))
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS, build_porosity_field,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import DDIMSampler  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402

OUT_DIR = REPO / "runs" / "analysis" / "ldm06_probe"
VOL_OUT = OUT_DIR / "volumes"
AUDIT_DIR = REPO / "runs" / "eval_v2" / "audit"
DR_VOL_ROOT = REPO / "runs" / "eval_v2" / "volumes" / "dose_response"

PATCH = 64
GRID_192 = (3, 3, 3)

# --- Part A selection thresholds (per-64^3-cell dark-air fraction, T_best,
#     CC-filtered — the audit's cells.csv.gz definition) ---
HIGH_AIR_THR = 0.15
LOW_AIR_THR = 0.01
MAX_CELLS_PER_GROUP = 150
ENC_BATCH = 32

# --- Part B protocol ---
STEP_COUNTS = [50, 100, 200, 300]
SEEDS = [101, 202]
TARGET = 0.03
S_POR = 1.5
BIG_SHAPE_VOX = (192, 1024, 1024)          # z, y, x — real-crop convention
BIG_STEPS = 200
BIG_SEED = 101


def detector_cal() -> dict:
    det = json.loads((AUDIT_DIR / "results.json").read_text())["detector"]
    return {"t_best": int(det["t_best"]), "t_cons": int(det["t_cons"]),
            "min_cc": int(det["min_cc_voxels"])}


# ---------------------------------------------------------------------------
# Part A — latent statistics of high-air vs low-air cells
# ---------------------------------------------------------------------------

def _native_float(vol_f01: np.ndarray) -> np.ndarray:
    """Sigmoid-scale generated float [0,1] -> decoder-native float [0,1].

    Inverse of the sampler's ``expit`` (same transform as the audit's
    ``to_native_u8``, without the u8 quantisation) — this is the scale the
    VAE encoder was trained on.
    """
    frac = np.clip(vol_f01.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.clip(logit(frac), 0.0, 1.0).astype(np.float32)


def select_cells(rng: np.random.Generator) -> pd.DataFrame:
    """High- and low-air 64^3 cells from the dose-response audit table.

    ``cells.csv.gz`` rows are written per volume in C-raveled (iz, iy, ix)
    order over the 3x3x3 grid, so the within-volume row index recovers the
    cell coordinates.  Low-air cells are drawn only from volumes that also
    contain at least one high-air cell ("the same volumes").
    """
    df = pd.read_csv(AUDIT_DIR / "cells.csv.gz")
    df = df[df.experiment == "dose_response"].copy()
    df["cell_idx"] = df.groupby(["arm", "name"]).cumcount()
    df["iz"] = df.cell_idx // 9
    df["iy"] = (df.cell_idx // 3) % 3
    df["ix"] = df.cell_idx % 3

    high = df[df.cell_dark_frac > HIGH_AIR_THR].copy()
    vols_with_high = set(map(tuple, high[["arm", "name"]].drop_duplicates()
                             .itertuples(index=False)))
    low = df[(df.cell_dark_frac < LOW_AIR_THR)
             & df[["arm", "name"]].apply(tuple, axis=1).isin(vols_with_high)
             ].copy()

    def cap(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) > MAX_CELLS_PER_GROUP:
            g = g.iloc[rng.choice(len(g), MAX_CELLS_PER_GROUP, replace=False)]
        return g

    high, low = cap(high), cap(low)
    high["group"], low["group"] = "high_air", "low_air"
    return pd.concat([high, low], ignore_index=True)


def encode_cells(cells: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Encode each selected cell with the frozen VAE; per-channel mu moments."""
    meta = json.loads((LATENTS_ROOT / "metadata.json").read_text())
    vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), device)
    vae.requires_grad_(False)
    norm = meta["normalization"]
    train_mean = np.asarray(norm["per_channel_mean"], np.float64)
    train_std = np.asarray(norm["per_channel_std"], np.float64)
    n_ch = len(train_mean)

    rows: list[dict] = []
    pending_x: list[np.ndarray] = []
    pending_m: list[np.ndarray] = []
    pending_meta: list[dict] = []

    def flush():
        if not pending_x:
            return
        x = torch.from_numpy(np.stack(pending_x)[:, None]).to(device)
        m = torch.from_numpy(np.stack(pending_m)[:, None]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = vae(x, m)
        mu = out.mu.float()                                   # (B, C, 16,16,16)
        cmean = mu.mean(dim=(2, 3, 4)).cpu().numpy()
        cstd = mu.std(dim=(2, 3, 4)).cpu().numpy()
        for k, info in enumerate(pending_meta):
            row = dict(info)
            for c in range(n_ch):
                row[f"mu_mean_ch{c}"] = float(cmean[k, c])
                row[f"mu_std_ch{c}"] = float(cstd[k, c])
                row[f"shift_ch{c}"] = float(
                    (cmean[k, c] - train_mean[c]) / train_std[c])
                row[f"std_ratio_ch{c}"] = float(cstd[k, c] / train_std[c])
            row["shift_rms"] = float(np.sqrt(np.mean(
                [row[f"shift_ch{c}"] ** 2 for c in range(n_ch)])))
            row["std_ratio_mean"] = float(np.mean(
                [row[f"std_ratio_ch{c}"] for c in range(n_ch)]))
            rows.append(row)
        pending_x.clear(), pending_m.clear(), pending_meta.clear()

    for (arm, name), g in cells.groupby(["arm", "name"], sort=True):
        vol_dir = DR_VOL_ROOT / arm / name
        native = _native_float(tifffile.imread(str(vol_dir / "volume.tif")))
        mask = (tifffile.imread(str(vol_dir / "mask.tif")) > 0)
        for r in g.itertuples(index=False):
            sl = tuple(slice(i * PATCH, (i + 1) * PATCH)
                       for i in (r.iz, r.iy, r.ix))
            pending_x.append(native[sl])
            pending_m.append(mask[sl].astype(np.float32))
            pending_meta.append({
                "arm": arm, "name": name, "iz": r.iz, "iy": r.iy, "ix": r.ix,
                "group": r.group, "cell_dark_frac": r.cell_dark_frac,
                "cell_mask_porosity": r.cell_mask_porosity,
            })
            if len(pending_x) >= ENC_BATCH:
                flush()
        del native, mask
    flush()
    del vae
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def part_a(device: torch.device) -> dict:
    rng = np.random.default_rng(0)
    cells = select_cells(rng)
    n_hi = int((cells.group == "high_air").sum())
    n_lo = int((cells.group == "low_air").sum())
    print(f"[A] {n_hi} high-air + {n_lo} low-air cells from "
          f"{cells[['arm', 'name']].drop_duplicates().shape[0]} volumes",
          flush=True)
    df = encode_cells(cells, device)
    df.to_csv(OUT_DIR / "part_a_cells.csv", index=False)

    hi = df[df.group == "high_air"]
    lo = df[df.group == "low_air"]
    n_ch = sum(c.startswith("shift_ch") for c in df.columns)

    summary: dict = {
        "n_high": len(hi), "n_low": len(lo),
        "high_air_threshold": HIGH_AIR_THR, "low_air_threshold": LOW_AIR_THR,
        "per_channel": {},
    }
    for c in range(n_ch):
        summary["per_channel"][f"ch{c}"] = {
            "shift_high_mean": float(hi[f"shift_ch{c}"].mean()),
            "shift_low_mean": float(lo[f"shift_ch{c}"].mean()),
            "shift_d": cohens_d(hi[f"shift_ch{c}"].values,
                                lo[f"shift_ch{c}"].values),
            "std_ratio_high_mean": float(hi[f"std_ratio_ch{c}"].mean()),
            "std_ratio_low_mean": float(lo[f"std_ratio_ch{c}"].mean()),
            "std_ratio_d": cohens_d(hi[f"std_ratio_ch{c}"].values,
                                    lo[f"std_ratio_ch{c}"].values),
        }
    for key in ("shift_rms", "std_ratio_mean"):
        summary[key] = {
            "high_mean": float(hi[key].mean()), "high_std": float(hi[key].std()),
            "low_mean": float(lo[key].mean()), "low_std": float(lo[key].std()),
            "d": cohens_d(hi[key].values, lo[key].values),
        }
    # simple separation score: fraction of high cells above the low-group p95
    for key in ("shift_rms", "std_ratio_mean"):
        p95 = float(np.percentile(lo[key], 95))
        summary[key]["low_p95"] = p95
        summary[key]["frac_high_above_low_p95"] = float((hi[key] > p95).mean())
    # continuous relation: dark-air fraction vs latent statistics (all cells)
    from scipy.stats import spearmanr
    summary["dark_frac_correlations"] = {
        col: {"pearson": float(np.corrcoef(df.cell_dark_frac, df[col])[0, 1]),
              "spearman": float(spearmanr(df.cell_dark_frac, df[col]).statistic)}
        for col in ([f"shift_ch{c}" for c in range(n_ch)]
                    + ["shift_rms", "std_ratio_mean"])
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for grp, color, lbl in (("low_air", "#1b6ca8", f"low air (<{LOW_AIR_THR:g})"),
                            ("high_air", "#c22e2e", f"high air (>{HIGH_AIR_THR:g})")):
        g = df[df.group == grp]
        ax.scatter(g.shift_rms, g.std_ratio_mean, s=14, alpha=0.6,
                   color=color, label=lbl, edgecolors="none")
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("per-channel mean shift, RMS (train-std units)")
    ax.set_ylabel("per-channel std ratio, mean (cell / train)")
    ax.set_title("Re-encoded latent moments per 64$^3$ cell")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.arange(n_ch)
    w = 0.38
    ax.bar(xs - w / 2, [summary["per_channel"][f"ch{c}"]["std_ratio_d"]
                        for c in range(n_ch)], w, color="#2e7d32",
           label="std ratio")
    ax.bar(xs + w / 2, [summary["per_channel"][f"ch{c}"]["shift_d"]
                        for c in range(n_ch)], w, color="#c2571a",
           label="mean shift")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs), ax.set_xticklabels([f"ch{c}" for c in range(n_ch)])
    ax.set_ylabel("Cohen's d (high vs low air)")
    ax.set_title("Effect size per latent channel")
    ax.legend(fontsize=8)
    fig.suptitle("ldm06 probe A — are dark cells off-manifold latents?",
                 y=1.02)
    summary["figure"] = savefig(fig, OUT_DIR, "ldm06_fig1_latent_separation")
    return summary


# ---------------------------------------------------------------------------
# Part B — DDIM step count vs interior air
# ---------------------------------------------------------------------------

def coherent_map(target: float, seed: int, grid: tuple, te_sampler,
                 corr_lengths) -> dict:
    field = build_porosity_field(
        grid_shape=grid, target=target, sampler=te_sampler,
        corr_lengths_voxels=corr_lengths, stride_voxels=PATCH, seed=seed,
    )
    return {(iz, iy, ix): float(field[iz, iy, ix])
            for iz in range(grid[0]) for iy in range(grid[1])
            for ix in range(grid[2])}


def generate_one(gen, por_map: dict, volume_mm: tuple, jwb: int):
    """Joint-mode generation with the _eval_v2 OOM-halving retry."""
    while True:
        try:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            with torch.no_grad():
                xct, mask, stats = gen.generate(
                    volume_size_mm=volume_mm,
                    local_por_map=por_map,
                    autocast_dtype=torch.bfloat16,
                    gen_batch_size=27,
                    decode_batch_size=27,
                    mode="joint",
                    joint_window_stride=32,
                    joint_window_batch=jwb,
                )
            return xct, mask, stats, time.perf_counter() - t0, jwb
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if jwb <= 1:
                raise
            jwb //= 2
            print(f"  OOM — retrying with joint_window_batch={jwb}", flush=True)


def audit_volume(vol_dir: Path, cal: dict) -> dict:
    """Interior/edge unmasked-air with the calibrated detector (audit code)."""
    native = to_native_u8(vol_dir)
    mask = tifffile.imread(str(vol_dir / "mask.tif")) > 0
    n_vox = native.size
    edge = edge_shell(native.shape)
    n_edge = int(np.count_nonzero(edge))
    out = {"mask_porosity": float(mask.mean()),
           "shape": "x".join(str(s) for s in native.shape)}
    for tag, t in (("best", cal["t_best"]), ("cons", cal["t_cons"])):
        det = native < t
        det, _, _ = cc_filter(det, cal["min_cc"])
        unmasked = det & ~mask
        um_edge = int(np.count_nonzero(unmasked & edge))
        um_tot = int(np.count_nonzero(unmasked))
        out[f"detected_air_{tag}"] = float(np.count_nonzero(det)) / n_vox
        out[f"unmasked_air_{tag}"] = um_tot / n_vox
        out[f"unmasked_edge_local_{tag}"] = um_edge / n_edge
        out[f"unmasked_interior_local_{tag}"] = (um_tot - um_edge) / (n_vox - n_edge)
        del det, unmasked
    del native, mask
    return out


def run_cell(gen, model, schedule, device, steps: int, seed: int,
             grid: tuple, shape_vox: tuple, te_sampler, corr_lengths,
             jwb: int, cal: dict, name: str) -> dict:
    vol_dir = VOL_OUT / name
    rec = load_existing(vol_dir)
    if rec is None:
        gen.sampler = DDIMSampler(model, schedule, device, n_steps=steps,
                                  s_por=S_POR, s_nb=1.0)
        gen.conditioning_semantics = "specimen"
        torch.manual_seed(seed)
        por_map = coherent_map(TARGET, seed, grid, te_sampler, corr_lengths)
        volume_mm = tuple(s * 0.025 for s in shape_vox)
        xct, mask, stats, wall, jwb_used = generate_one(gen, por_map,
                                                        volume_mm, jwb)
        rec = {
            "name": name, "ddim_steps": steps, "seed": seed,
            "target": TARGET, "s_por": S_POR, "mode": "joint",
            "conditioning_semantics": "specimen",
            "checkpoint": str(CKPT), "weights": "raw",
            "delivered_mask_porosity": float(stats["actual_mask_porosity"]),
            "seam_xct_ratio": stats["seam_xct_ratio"],
            "seam_mask_ratio": stats["seam_mask_ratio"],
            "wall_s": round(wall, 1),
            "joint_window_batch": jwb_used,
        }
        save_volume(vol_dir, xct, mask, rec)
        del xct, mask
    rec.update(audit_volume(vol_dir, cal))
    print(f"[B] {name}: interior_best="
          f"{rec['unmasked_interior_local_best']:.4f}  "
          f"mask_por={rec['mask_porosity']:.4f}  wall={rec['wall_s']}s",
          flush=True)
    return rec


def monotone_reduction(rows: list[dict]) -> tuple[bool, str]:
    """Clear monotone air reduction: for EVERY seed the interior unmasked-air
    fraction (T_best) is non-increasing in the step count — within a noise
    tolerance of 2% of the DDIM-50 value (once the curve is at its floor,
    differences far below the starting magnitude are plateau noise, not a
    reversal) — AND the minimum sits at least 20% (relative) below DDIM-50."""
    ok, notes = True, []
    for seed in SEEDS:
        ys = [r["unmasked_interior_local_best"] for r in rows
              if r["seed"] == seed and r["shape"].startswith("192")]
        tol = max(0.02 * ys[0], 5e-4)
        mono = all(b <= a + tol for a, b in zip(ys, ys[1:]))
        drop = (ys[0] - min(ys)) / ys[0] if ys[0] > 0 else 0.0
        ok &= mono and drop >= 0.20
        notes.append(f"seed {seed}: {['%.4f' % y for y in ys]} "
                     f"mono={mono} rel_drop={drop:.2f}")
    return ok, "; ".join(notes)


def part_b(device: torch.device, jwb: int, big: str) -> dict:
    cal = detector_cal()
    gen, model, schedule, step = build_generator(device)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    rows = []
    for steps in STEP_COUNTS:
        for seed in SEEDS:
            rows.append(run_cell(
                gen, model, schedule, device, steps, seed, GRID_192,
                (192, 192, 192), te_sampler, corr_lengths, jwb, cal,
                f"{steps}_seed_{seed}"))

    mono_ok, mono_note = monotone_reduction(rows)
    run_big = {"auto": mono_ok, "force": True, "skip": False}[big]
    big_row = None
    if run_big:
        print(f"[B] monotone check: {mono_note} -> running 1024 scale check",
              flush=True)
        big_row = run_cell(
            gen, model, schedule, device, BIG_STEPS, BIG_SEED,
            tuple(s // PATCH for s in BIG_SHAPE_VOX), BIG_SHAPE_VOX,
            te_sampler, corr_lengths, jwb, cal,
            f"big1024_{BIG_STEPS}_seed_{BIG_SEED}")
    else:
        print(f"[B] monotone check failed -> skipping 1024 volume "
              f"({mono_note})", flush=True)

    # Figure: interior air vs DDIM steps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {101: "#1b6ca8", 202: "#c2571a"}
    for ax, tag, title in ((axes[0], "best", f"T_best = {cal['t_best']}"),
                           (axes[1], "cons", f"T_cons = {cal['t_cons']}")):
        for seed in SEEDS:
            sub = [r for r in rows if r["seed"] == seed]
            ax.plot(STEP_COUNTS,
                    [r[f"unmasked_interior_local_{tag}"] for r in sub],
                    "o-", color=colors[seed], label=f"interior, seed {seed}")
            ax.plot(STEP_COUNTS,
                    [r[f"unmasked_edge_local_{tag}"] for r in sub],
                    "s--", color=colors[seed], alpha=0.5,
                    label=f"edge, seed {seed}")
        if big_row is not None:
            ax.plot([BIG_STEPS], [big_row[f"unmasked_interior_local_{tag}"]],
                    "D", color="#2e7d32", ms=8,
                    label=f"1024$^2$x192, seed {BIG_SEED}")
        ax.set_xlabel("DDIM steps")
        ax.set_title(title)
        ax.set_xticks(STEP_COUNTS)
    axes[0].set_ylabel("unmasked air fraction (local)")
    axes[0].legend(fontsize=8)
    fig.suptitle("ldm06 probe B — interior unmasked air vs DDIM step count "
                 f"(joint, specimen, s_por {S_POR}, target {TARGET:g})",
                 y=1.02)
    fig_paths = savefig(fig, OUT_DIR, "ldm06_fig2_interior_air_vs_steps")

    return {
        "checkpoint": str(CKPT), "checkpoint_step": step, "weights": "raw",
        "protocol": {"mode": "joint", "conditioning_semantics": "specimen",
                     "s_por": S_POR, "target": TARGET,
                     "ddim_steps": STEP_COUNTS, "seeds": SEEDS,
                     "field": "coherent (T-E sampler + T-D corr lengths)"},
        "detector": cal,
        "volumes": rows,
        "monotone_reduction": mono_ok,
        "monotone_note": mono_note,
        "big_volume": big_row,
        "big_volume_ran": big_row is not None,
        "figure": fig_paths,
    }


# ---------------------------------------------------------------------------


def build_findings(res: dict) -> str:
    lines = ["# ldm06 probe — off-manifold latents and DDIM step count",
             "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "scripts/analysis/ldm06_probe.py. Spec: vault note "
             '"PoreGen - ldm06 Design" (D40).',
             ""]
    a = res.get("part_a")
    if a:
        sr, sh = a["std_ratio_mean"], a["shift_rms"]
        lines += [
            "## A. Latent statistics of dark cells",
            "",
            f"- {a['n_high']} high-air cells (dark-air > {a['high_air_threshold']:g})"
            f" and {a['n_low']} low-air cells (< {a['low_air_threshold']:g}),"
            " re-encoded with the frozen r07-z4 VAE.",
            f"- Mean per-channel std ratio (cell / train): high air "
            f"{sr['high_mean']:.3f} +/- {sr['high_std']:.3f}, low air "
            f"{sr['low_mean']:.3f} +/- {sr['low_std']:.3f}, Cohen's d = "
            f"{sr['d']:.2f}.",
            f"- Per-channel mean shift (RMS, train-std units): high air "
            f"{sh['high_mean']:.3f} +/- {sh['high_std']:.3f}, low air "
            f"{sh['low_mean']:.3f} +/- {sh['low_std']:.3f}, Cohen's d = "
            f"{sh['d']:.2f}.",
            f"- {sr['frac_high_above_low_p95'] * 100:.0f}% of high-air cells "
            "sit above the low-air p95 in std ratio; "
            f"{sh['frac_high_above_low_p95'] * 100:.0f}% in mean shift.",
            "- Figure: ldm06_fig1_latent_separation.",
            "",
        ]
        corr = a.get("dark_frac_correlations")
        if corr:
            top = max(corr, key=lambda k: abs(corr[k]["pearson"]))
            lines.insert(-1,
                         f"- Strongest continuous relation: dark-air fraction "
                         f"vs {top} (Pearson r = {corr[top]['pearson']:.3f}, "
                         f"Spearman {corr[top]['spearman']:.3f}).")
    b = res.get("part_b")
    if b:
        lines += ["## B. DDIM step count vs interior air", ""]
        for seed in SEEDS:
            ys = [f"{r['unmasked_interior_local_best']:.4f}"
                  for r in b["volumes"] if r["seed"] == seed]
            lines.append(f"- Seed {seed}, interior unmasked air (T_best) at "
                         f"steps {STEP_COUNTS}: {ys}.")
        lines += [
            f"- Monotone reduction criterion met: {b['monotone_reduction']}"
            f" ({b['monotone_note']}).",
            f"- 1024x1024x192 scale check ran: {b['big_volume_ran']}.",
        ]
        if b["big_volume"]:
            bb = b["big_volume"]
            lines.append(
                f"  - Big volume ({bb['shape']}, {bb['ddim_steps']} steps): "
                f"interior {bb['unmasked_interior_local_best']:.4f}, edge "
                f"{bb['unmasked_edge_local_best']:.4f} (T_best), mask "
                f"porosity {bb['mask_porosity']:.4f}, wall "
                f"{bb['wall_s']:.0f} s.")
        lines += ["- Figure: ldm06_fig2_interior_air_vs_steps.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    ap.add_argument("--jwb", type=int, default=16,
                    help="initial joint_window_batch (halved on OOM)")
    ap.add_argument("--big", choices=["auto", "force", "skip"], default="auto",
                    help="1024-scale check: auto = only on monotone reduction")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    device = torch.device("cuda")

    res_path = OUT_DIR / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    if args.part in ("a", "all"):
        res["part_a"] = part_a(device)
        write_json(res, OUT_DIR)
    if args.part in ("b", "all"):
        res["part_b"] = part_b(device, args.jwb, args.big)
        write_json(res, OUT_DIR)
    write_findings(build_findings(res), OUT_DIR)
    print("Done ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()

"""Eval v2 phase 3: void/mask audit over all saved campaign volumes.

Recalibrates the grayscale void detector on REAL volumes (ground-truth masks),
then audits every generated volume under ``runs/eval_v2/volumes/`` for
dark-air content that the generated pore mask does not label.

Detector design
---------------
1. Brightness shift: ``VolumeGenerator`` writes ``expit(xct_out)`` while the
   VAE decodes directly to [0, 1], so generated TIFFs are sigmoid-compressed.
   The audit inverts this exactly (``clip(logit(v), 0, 1) * 255``) and works on
   the decoder-native u8 scale, where real-calibrated absolute thresholds
   apply directly.  A material-mode-referenced threshold on the same native
   scale is reported per volume as a normalisation-free cross-check.
2. Absolute threshold calibrated on real val/test volumes by maximising voxel
   Dice for pores (best estimate, T_best) and, separately, the highest
   threshold that keeps pooled pore precision >= 0.95 (conservative lower
   bound, T_cons).
3. Minimum connected-component size MIN_CC (3D, 26-neighbourhood excluded --
   default scipy 6-connectivity) removes grey-texture speckle so only
   coherent air masses count.
4. Sensitivity: detected/unmasked fractions are also reported at
   T_best +/- sigma (sigma = mean real material spread).

Outputs to ``runs/eval_v2/audit/``: results.json, per_volume.csv, cells.csv,
findings.md, figures (PDF + PNG, 300 dpi).

Usage:
    python scripts/analysis/eval_v2_audit.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style, write_json  # noqa: E402
from void_mask_audit import (  # noqa: E402
    REAL_VOLUMES, dice_iou, load_real_crop, material_stats,
)

OUT_DIR = REPO / "runs" / "eval_v2" / "audit"
VOL_ROOT = REPO / "runs" / "eval_v2" / "volumes"
EXPERIMENTS = ["dose_response", "cfg_sweep", "layup"]
ARMS = ["seq", "joint_legacy", "joint_oob"]

VOXEL_UM = 25.0
VOXEL_MM = VOXEL_UM / 1000.0
VOXEL_MM3 = VOXEL_MM ** 3
PATCH = 64
MIN_CC = 300            # voxels; ~4.7e-3 mm^3, equiv. diameter ~0.21 mm
EDGE_VOX = 32           # shell within EDGE_VOX voxels of any volume face
CAL_SLICE_STEP = 8      # calibration uses every 8th z-slice of the real crops
FP_CEILING = 0.002      # conservative detector: pooled real dark-but-unmasked
                        # (false-positive) fraction of foreground <= this
ARM_COLORS = {"seq": "#1b6ca8", "joint_legacy": "#c2571a", "joint_oob": "#2e7d32"}


# ---------------------------------------------------------------------------
# Detector calibration on real volumes
# ---------------------------------------------------------------------------

def _pooled_counts(real: dict, thresholds: np.ndarray) -> tuple[dict, int]:
    """threshold -> [intersection, n_pred, n_gt] pooled over calib slices,
    plus the pooled foreground voxel count."""
    pooled = {int(t): [0, 0, 0] for t in thresholds}
    n_fg = 0
    for name, (xct, mask, fg, _) in real.items():
        sl = slice(0, xct.shape[0], CAL_SLICE_STEP)
        x, m, f = xct[sl], mask[sl], fg[sl]
        gt = m & f
        n_gt = int(np.count_nonzero(gt))
        n_fg += int(np.count_nonzero(f))
        for t in thresholds:
            pred = (x < t) & f
            pooled[int(t)][0] += int(np.count_nonzero(pred & gt))
            pooled[int(t)][1] += int(np.count_nonzero(pred))
            pooled[int(t)][2] += n_gt
    return pooled, n_fg


def _metrics(counts: list[int], n_fg: int) -> tuple[float, float, float]:
    """(dice, precision, false-positive fraction of foreground)."""
    inter, npred, ngt = counts
    dice = 2.0 * inter / (npred + ngt) if npred + ngt else 0.0
    prec = inter / npred if npred else 1.0
    fp = (npred - inter) / n_fg if n_fg else 0.0
    return dice, prec, fp


def calibrate(real: dict) -> dict:
    coarse = np.arange(60, 221, 5)
    pooled, n_fg = _pooled_counts(real, coarse)

    def pick(p):
        t_b = max(p, key=lambda t: _metrics(p[t], n_fg)[0])
        fp_ok = [t for t in p if _metrics(p[t], n_fg)[2] <= FP_CEILING]
        t_c = max(fp_ok) if fp_ok else min(p)
        return t_b, t_c

    t_best_c, t_cons_c = pick(pooled)
    fine = np.unique(np.concatenate([
        np.arange(t_best_c - 4, t_best_c + 5),
        np.arange(t_cons_c - 4, t_cons_c + 5)]))
    pooled_f, _ = _pooled_counts(real, fine)
    pooled.update(pooled_f)
    t_best, t_cons = pick(pooled)

    spreads, modes = [], []
    for name, (xct, mask, fg, _) in real.items():
        mode, spread = material_stats(xct, fg)
        modes.append(mode), spreads.append(spread)
    sigma = float(np.mean(spreads))
    k_ref = float((np.mean(modes) - t_best) / sigma)

    sweep = sorted(pooled)
    return {
        "t_best": int(t_best),
        "t_cons": int(t_cons),
        "sigma_u8": sigma,
        "k_material_referenced": k_ref,
        "real_material_modes": modes,
        "real_material_spreads": spreads,
        "dice_at_t_best": _metrics(pooled[t_best], n_fg)[0],
        "precision_at_t_best": _metrics(pooled[t_best], n_fg)[1],
        "fp_at_t_best": _metrics(pooled[t_best], n_fg)[2],
        "dice_at_t_cons": _metrics(pooled[t_cons], n_fg)[0],
        "precision_at_t_cons": _metrics(pooled[t_cons], n_fg)[1],
        "fp_at_t_cons": _metrics(pooled[t_cons], n_fg)[2],
        "fp_ceiling": FP_CEILING,
        "min_cc_voxels": MIN_CC,
        "sweep_thresholds": [int(t) for t in sweep],
        "sweep_dice": [_metrics(pooled[t], n_fg)[0] for t in sweep],
        "sweep_precision": [_metrics(pooled[t], n_fg)[1] for t in sweep],
        "sweep_fp_fraction": [_metrics(pooled[t], n_fg)[2] for t in sweep],
    }


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

def cc_filter(det: np.ndarray, min_cc: int = MIN_CC) -> tuple[np.ndarray, int, int]:
    """Remove connected components smaller than min_cc voxels, in place.
    Returns (det, n_components_kept, n_components_total)."""
    labels, n = ndimage.label(det)
    if n == 0:
        return det, 0, 0
    sizes = np.bincount(labels.ravel())
    small = sizes < min_cc
    small[0] = False
    n_kept = int(n - small[1:].sum())
    det[small[labels]] = False
    del labels
    return det, n_kept, int(n)


_EDGE_CACHE: dict[tuple, np.ndarray] = {}


def edge_shell(shape: tuple) -> np.ndarray:
    if shape not in _EDGE_CACHE:
        ax = []
        for s in shape:
            v = np.zeros(s, bool)
            v[:EDGE_VOX] = True
            v[-EDGE_VOX:] = True
            ax.append(v)
        _EDGE_CACHE[shape] = (ax[0][:, None, None] | ax[1][None, :, None]
                              | ax[2][None, None, :])
    return _EDGE_CACHE[shape]


def largest_components(unmasked: np.ndarray, top: int = 5) -> list[dict]:
    labels, n = ndimage.label(unmasked)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][:top]
    objs = ndimage.find_objects(labels)
    out = []
    shape = unmasked.shape
    for idx in order:
        size = int(sizes[idx])
        if size == 0:
            break
        sl = objs[idx - 1]
        touches = any(s.start == 0 or s.stop == shape[i] for i, s in enumerate(sl))
        out.append({
            "voxels": size,
            "volume_mm3": size * VOXEL_MM3,
            "equiv_diameter_mm": 2.0 * (3.0 * size * VOXEL_MM3 / (4.0 * np.pi)) ** (1 / 3),
            "bbox_extent_vox": [int(s.stop - s.start) for s in sl],
            "touches_face": bool(touches),
        })
    del labels
    return out


def cellwise(a: np.ndarray) -> np.ndarray:
    gz, gy, gx = (s // PATCH for s in a.shape)
    return (a[:gz * PATCH, :gy * PATCH, :gx * PATCH]
            .reshape(gz, PATCH, gy, PATCH, gx, PATCH)
            .mean(axis=(1, 3, 5)))


def to_native_u8(cell_dir: Path) -> np.ndarray:
    """Generated float [0,1] sigmoid-scale TIFF -> decoder-native u8, chunked."""
    mm = tifffile.memmap(str(cell_dir / "volume.tif"), mode="r")
    out = np.empty(mm.shape, np.uint8)
    for z0 in range(0, mm.shape[0], 32):
        chunk = np.asarray(mm[z0:z0 + 32], np.float32)
        if chunk.max() > 1.5:
            chunk = chunk / 255.0
        frac = np.clip(chunk, 1e-6, 1.0 - 1e-6)
        out[z0:z0 + 32] = (np.clip(logit(frac), 0.0, 1.0) * 255.0).astype(np.uint8)
    del mm
    return out


# ---------------------------------------------------------------------------
# Real-volume validation (full crops, CC-filtered)
# ---------------------------------------------------------------------------

def validate_real(real: dict, cal: dict) -> dict:
    out = {}
    for name, (xct, mask, fg, meta) in real.items():
        gt = mask & fg
        n_fg = int(np.count_nonzero(fg))
        row = {"n_fg_voxels": n_fg, **meta}
        for tag, t in (("best", cal["t_best"]), ("cons", cal["t_cons"])):
            det = (xct < t) & fg
            d_nocc, _ = dice_iou(det, gt)
            inter_n = float(np.count_nonzero(det & gt))
            npred_n = float(np.count_nonzero(det))
            det, n_kept, n_tot = cc_filter(det)
            d, i = dice_iou(det, gt)
            inter = float(np.count_nonzero(det & gt))
            npred = float(np.count_nonzero(det))
            unmasked = det & ~gt
            row[tag] = {
                "threshold": int(t),
                "dice_nocc": d_nocc,
                "precision_nocc": inter_n / npred_n if npred_n else 1.0,
                "recall_nocc": inter_n / float(np.count_nonzero(gt)),
                "dice": d, "iou": i,
                "precision": inter / npred if npred else 1.0,
                "recall": inter / float(np.count_nonzero(gt)),
                "detected_air_fraction": npred / n_fg,
                "false_positive_unmasked_fraction": float(np.count_nonzero(unmasked)) / n_fg,
                "components_kept": n_kept, "components_total": n_tot,
            }
        row["mask_porosity"] = float(np.count_nonzero(gt)) / n_fg
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Generated-volume audit
# ---------------------------------------------------------------------------

def audit_generated(cell_dir: Path, cal: dict, exp: str, arm: str
                    ) -> tuple[dict, pd.DataFrame]:
    stats = json.loads((cell_dir / "stats.json").read_text())
    native = to_native_u8(cell_dir)
    mask = tifffile.imread(str(cell_dir / "mask.tif")) > 0
    n_vox = native.size
    edge = edge_shell(native.shape)
    n_edge = int(np.count_nonzero(edge))

    row = {
        "experiment": exp, "arm": arm, "name": cell_dir.name,
        "target": stats.get("target"), "seed": stats.get("seed"),
        "s_por": stats.get("s_por"), "layup": stats.get("layup"),
        "shape": "x".join(str(s) for s in native.shape),
        "mask_porosity": float(mask.mean()),
    }

    # Sensitivity sweep: cons, best-sigma, best, best+sigma — all CC-filtered.
    sig = int(round(cal["sigma_u8"]))
    variants = {
        "cons": cal["t_cons"],
        "best_lo": cal["t_best"] - sig,
        "best": cal["t_best"],
        "best_hi": cal["t_best"] + sig,
    }
    cells_df = None
    for tag, t in variants.items():
        det = native < t
        det, n_kept, n_tot = cc_filter(det)
        unmasked = det & ~mask
        row[f"detected_air_{tag}"] = float(np.count_nonzero(det)) / n_vox
        row[f"unmasked_air_{tag}"] = float(np.count_nonzero(unmasked)) / n_vox
        if tag in ("best", "cons"):
            um_edge = int(np.count_nonzero(unmasked & edge))
            um_tot = int(np.count_nonzero(unmasked))
            row[f"unmasked_edge_{tag}"] = um_edge / n_vox
            row[f"unmasked_interior_{tag}"] = (um_tot - um_edge) / n_vox
            # interior fraction normalised by interior volume, edge by shell
            row[f"unmasked_edge_local_{tag}"] = um_edge / n_edge
            row[f"unmasked_interior_local_{tag}"] = (um_tot - um_edge) / (n_vox - n_edge)
        if tag == "best":
            comps = largest_components(unmasked)
            row["n_components_best"] = n_kept
            if comps:
                row["largest_comp_voxels"] = comps[0]["voxels"]
                row["largest_comp_mm3"] = comps[0]["volume_mm3"]
                row["largest_comp_equiv_diam_mm"] = comps[0]["equiv_diameter_mm"]
                row["largest_comp_touches_face"] = comps[0]["touches_face"]
            else:
                row["largest_comp_voxels"] = 0
                row["largest_comp_mm3"] = 0.0
                row["largest_comp_equiv_diam_mm"] = 0.0
                row["largest_comp_touches_face"] = False
            row["top_components"] = json.dumps(comps)
            # per-cell table for the mask-collapse analysis
            dark = cellwise(det)
            mpor = cellwise(mask)
            umf = cellwise(unmasked)
            cells_df = pd.DataFrame({
                "experiment": exp, "arm": arm, "name": cell_dir.name,
                "cell_dark_frac": dark.ravel(),
                "cell_mask_porosity": mpor.ravel(),
                "cell_unmasked_frac": umf.ravel(),
            })
        del det, unmasked

    # material-mode-referenced cross-check on the native scale (no CC filter)
    mode, spread = material_stats(native, None)
    t_rel = mode - cal["k_material_referenced"] * spread
    row["material_mode_native"] = mode
    row["material_spread_native"] = spread
    row["t_rel_native"] = float(t_rel)
    row["detected_air_rel_nocc"] = float((native < t_rel).mean())
    del native, mask
    return row, cells_df


def iter_cell_dirs():
    for exp in EXPERIMENTS:
        for arm in ARMS:
            root = VOL_ROOT / exp / arm
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if (d / "volume.tif").exists() and (d / "mask.tif").exists():
                    yield exp, arm, d


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_montage(cal: dict, real: dict, results: dict):
    """Real vs seq vs joint_legacy vs joint_oob at target 0.03 (layup arm
    volumes, A_training seed 101): grayscale / mask / unmasked-air overlay,
    worst offending slice."""
    import matplotlib.pyplot as plt

    rows = []
    rname = REAL_VOLUMES[0]
    xct_r, mask_r, fg_r, _ = real[rname]
    det = (xct_r < cal["t_best"]) & fg_r
    det, _, _ = cc_filter(det)
    unm = det & ~mask_r
    z = int(np.argmax(unm.reshape(unm.shape[0], -1).mean(axis=1)))
    rows.append(("real (Airbus panel, val)", xct_r[z], mask_r[z], unm[z], z))
    del det, unm

    for arm in ARMS:
        d = VOL_ROOT / "layup" / arm / "A_training_seed_101"
        native = to_native_u8(d)
        mask = tifffile.imread(str(d / "mask.tif")) > 0
        det = native < cal["t_best"]
        det, _, _ = cc_filter(det)
        unm = det & ~mask
        z = int(np.argmax(unm.reshape(unm.shape[0], -1).mean(axis=1)))
        rows.append((f"{arm} (layup A, seed 101)", native[z].copy(),
                     mask[z].copy(), unm[z].copy(), z))
        del native, mask, det, unm

    fig, ax = plt.subplots(len(rows), 3, figsize=(13, 3.4 * len(rows)))
    for i, (title, gray, msk, unm2d, z) in enumerate(rows):
        ax[i, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(title, fontsize=9)
        ax[i, 1].imshow(msk, cmap="gray")
        rgb = np.stack([gray] * 3, -1).astype(np.float32) / 255.0
        rgb[unm2d] = [0.85, 0.15, 0.15]
        rgb[msk & ~unm2d] = [0.2, 0.45, 0.9]
        ax[i, 2].imshow(rgb)
        ax[i, 2].set_title(f"z={z}  red = unmasked air, blue = mask", fontsize=8)
        if i == 0:
            ax[i, 0].set_title("grayscale (native u8)", fontsize=9)
            ax[i, 1].set_title("pore mask", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Worst offending slices — detector T={cal['t_best']}, "
                 f"min CC {MIN_CC} vox", y=0.995)
    return savefig(fig, OUT_DIR, "audit_fig1_slice_montage")


def fig_unmasked_by_arm(pv: pd.DataFrame, real_val: dict, cal: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    groups = EXPERIMENTS
    width = 0.25
    xs = np.arange(len(groups))
    for j, arm in enumerate(ARMS):
        best = [pv[(pv.experiment == e) & (pv.arm == arm)].unmasked_air_best.mean()
                for e in groups]
        cons = [pv[(pv.experiment == e) & (pv.arm == arm)].unmasked_air_cons.mean()
                for e in groups]
        pos = xs + (j - 1) * width
        ax.bar(pos, best, width * 0.9, color=ARM_COLORS[arm], alpha=0.45,
               label=f"{arm} (best)")
        ax.bar(pos, cons, width * 0.9, color=ARM_COLORS[arm],
               label=f"{arm} (conservative)")
    fp = np.mean([real_val[n]["best"]["false_positive_unmasked_fraction"]
                  for n in real_val])
    ax.axhline(fp, color="k", ls="--", lw=1,
               label=f"real false-positive baseline ({fp:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel("unmasked-air voxel fraction")
    ax.set_title("Unmasked air by arm and experiment "
                 f"(T_best={cal['t_best']}, T_cons={cal['t_cons']}, "
                 f"min CC {MIN_CC} vox)")
    ax.legend(ncol=2, fontsize=7.5)
    return savefig(fig, OUT_DIR, "audit_fig2_unmasked_air_by_arm")


def fig_interior_edge(pv: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xs = np.arange(len(ARMS))
    for k, (col_i, col_e, title) in enumerate((
            ("unmasked_interior_best", "unmasked_edge_best",
             "volume-normalised fractions"),
            ("unmasked_interior_local_best", "unmasked_edge_local_best",
             "region-normalised (local density)"))):
        ax = axes[k]
        ints = [pv[pv.arm == a][col_i].mean() for a in ARMS]
        edgs = [pv[pv.arm == a][col_e].mean() for a in ARMS]
        ax.bar(xs - 0.18, ints, 0.32, color="#7a3ea1", label="interior (>32 vox from face)")
        ax.bar(xs + 0.18, edgs, 0.32, color="#c9a227", label="edge shell (<=32 vox)")
        ax.set_xticks(xs)
        ax.set_xticklabels(ARMS)
        ax.set_ylabel("unmasked-air fraction")
        ax.set_title(title, fontsize=9.5)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Interior vs edge unmasked air by arm (best detector, all 153 volumes)")
    return savefig(fig, OUT_DIR, "audit_fig3_interior_vs_edge")


def fig_mask_collapse(cells: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.concatenate([[0], np.logspace(-3, 0, 16)])
    ax = axes[0]
    for arm in ARMS:
        c = cells[cells.arm == arm]
        dark = c.cell_dark_frac.to_numpy()
        capture = np.where(dark > 0,
                           1.0 - c.cell_unmasked_frac.to_numpy() / np.maximum(dark, 1e-9),
                           np.nan)
        idx = np.digitize(dark, bins)
        centers, med, lo, hi = [], [], [], []
        for b in range(1, len(bins)):
            sel = (idx == b) & np.isfinite(capture)
            if sel.sum() < 5:
                continue
            centers.append(np.sqrt(max(bins[b - 1], 1e-4) * bins[b]))
            med.append(np.median(capture[sel]))
            lo.append(np.percentile(capture[sel], 25))
            hi.append(np.percentile(capture[sel], 75))
        ax.plot(centers, med, color=ARM_COLORS[arm], label=arm)
        ax.fill_between(centers, lo, hi, color=ARM_COLORS[arm], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("cell dark-air fraction (64$^3$ cells)")
    ax.set_ylabel("mask capture of dark air (1 = mask labels it all)")
    ax.set_title("Mask capture vs cell air fraction (median, IQR)")
    ax.legend()

    ax = axes[1]
    for arm in ARMS:
        c = cells[cells.arm == arm]
        ax.scatter(c.cell_dark_frac, c.cell_mask_porosity, s=2, alpha=0.15,
                   color=ARM_COLORS[arm], label=arm, rasterized=True)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("cell dark-air fraction")
    ax.set_ylabel("cell mask porosity")
    ax.set_title("Mask porosity vs dark-air fraction per cell")
    ax.legend(markerscale=6)
    fig.suptitle("Mask-collapse behaviour on the 64$^3$ cell grid (best detector)")
    return savefig(fig, OUT_DIR, "audit_fig4_mask_collapse")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def collapse_stats(cells: pd.DataFrame) -> dict:
    out = {}
    for arm in ARMS:
        c = cells[cells.arm == arm]
        dark = c.cell_dark_frac.to_numpy()
        umf = c.cell_unmasked_frac.to_numpy()
        mpor = c.cell_mask_porosity.to_numpy()
        sel = dark > 1e-4
        capture = 1.0 - umf[sel] / np.maximum(dark[sel], 1e-9)
        r_um = float(np.corrcoef(dark, umf)[0, 1])
        r_mp = float(np.corrcoef(dark[sel], mpor[sel])[0, 1]) if sel.sum() > 2 else float("nan")
        # capture in low / mid / high air regimes
        regimes = {}
        for lo, hi, tag in ((1e-4, 0.02, "low"), (0.02, 0.15, "mid"), (0.15, 1.01, "high")):
            s = (dark >= lo) & (dark < hi)
            cap = 1.0 - umf[s] / np.maximum(dark[s], 1e-9)
            regimes[tag] = {
                "n_cells": int(s.sum()),
                "capture_median": float(np.median(cap)) if s.any() else None,
                "mask_porosity_median": float(np.median(mpor[s])) if s.any() else None,
            }
        out[arm] = {
            "n_cells": int(len(c)),
            "pearson_dark_vs_unmasked": r_um,
            "pearson_dark_vs_mask_porosity": r_mp,
            "capture_median_overall": float(np.median(capture)) if sel.any() else None,
            "regimes": regimes,
        }
    return out


def aggregate(pv: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "arm", "target"]
    agg = (pv.groupby(keys, dropna=False)
             .agg(n=("name", "count"),
                  mask_porosity=("mask_porosity", "mean"),
                  detected_air_best=("detected_air_best", "mean"),
                  unmasked_best=("unmasked_air_best", "mean"),
                  unmasked_cons=("unmasked_air_cons", "mean"),
                  unmasked_lo=("unmasked_air_best_lo", "mean"),
                  unmasked_hi=("unmasked_air_best_hi", "mean"),
                  interior_best=("unmasked_interior_best", "mean"),
                  edge_best=("unmasked_edge_best", "mean"),
                  largest_mm3=("largest_comp_mm3", "max"))
             .reset_index())
    return agg


def write_findings(res: dict, pv: pd.DataFrame, agg_arm: pd.DataFrame) -> None:
    cal = res["detector"]
    rv = res["real_validation"]
    lines = [
        "# Eval v2 — void/mask audit (phase 3)",
        "",
        f"All 153 saved campaign volumes under `runs/eval_v2/volumes/`. "
        f"Voxel size assumed {VOXEL_UM} um.",
        "",
        "## Detector (recalibrated, real ground truth)",
        "",
        f"- Scale: decoder-native u8 (exact logit inversion of the sampler's `expit`).",
        f"- Best estimate: T_best = {cal['t_best']} (dice-optimal on real val/test "
        f"volumes; pooled slice dice {cal['dice_at_t_best']:.3f}, precision "
        f"{cal['precision_at_t_best']:.3f}).",
        f"- Conservative: T_cons = {cal['t_cons']} (largest threshold whose pooled "
        f"real dark-but-unmasked false-positive fraction is <= {FP_CEILING:.3f} of "
        f"foreground; dice {cal['dice_at_t_cons']:.3f}, "
        f"FP {cal['fp_at_t_cons']:.4f}). Unmasked-air numbers at T_cons are a "
        "lower bound: on real data this detector mislabels at most "
        f"{FP_CEILING:.1%} of voxels.",
        f"- Minimum component size: {MIN_CC} voxels "
        f"({MIN_CC * VOXEL_MM3 * 1000:.1f}e-3 mm^3) — grey texture speckle removed.",
        f"- Sensitivity sigma = {cal['sigma_u8']:.1f} u8 (mean real material spread).",
        "",
        "### Real-volume validation (full crops, CC-filtered)",
        "",
        "| volume | dice (T_best, no CC) | dice (T_best, CC) | recall CC | "
        "FP-unmasked best | FP-unmasked cons |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rv.items():
        b = r['best']
        lines.append(
            f"| {name.split('__')[-1][:38]} | {b.get('dice_nocc', float('nan')):.3f} | "
            f"{b['dice']:.3f} | {b['recall']:.3f} | "
            f"{b['false_positive_unmasked_fraction']:.4f} | "
            f"{r['cons']['false_positive_unmasked_fraction']:.4f} |")
    lines += [
        "",
        "The CC filter (>= %d voxels) removes small real pores on purpose: the "
        "audited quantity is coherent air masses, not fine porosity. The "
        "unfiltered dice column is the pore-detection quality of the threshold; "
        "the CC columns describe the mass detector actually applied to the "
        "generated volumes." % MIN_CC]
    fp_best = np.mean([r["best"]["false_positive_unmasked_fraction"] for r in rv.values()])
    fp_cons = np.mean([r["cons"]["false_positive_unmasked_fraction"] for r in rv.values()])
    lines += [
        "",
        f"Real false-positive baseline (mean): best {fp_best:.4f}, "
        f"conservative {fp_cons:.4f}. Generated unmasked-air numbers above these "
        "levels are not explainable as detector noise.",
        "",
        "## Unmasked air by arm x experiment (mean over volumes)",
        "",
        "| experiment | arm | n | mask por | detected air (best) | unmasked best "
        "[-sigma, +sigma] | unmasked conservative | interior | edge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg_arm.iterrows():
        lines.append(
            f"| {r.experiment} | {r.arm} | {int(r.n)} | {r.mask_porosity:.4f} | "
            f"{r.detected_air_best:.4f} | {r.unmasked_best:.4f} "
            f"[{r.unmasked_lo:.4f}, {r.unmasked_hi:.4f}] | {r.unmasked_cons:.4f} | "
            f"{r.interior_best:.4f} | {r.edge_best:.4f} |")
    cs = res["mask_collapse"]
    lines += ["", "## Mask-collapse behaviour (64^3 cells)", ""]
    for arm, s in cs.items():
        reg = s["regimes"]
        lines.append(
            f"- {arm}: r(dark, unmasked) = {s['pearson_dark_vs_unmasked']:.3f}; "
            f"median mask capture of dark air: "
            f"low(<2%) {reg['low']['capture_median']:.2f}, "
            f"mid(2-15%) {reg['mid']['capture_median']:.2f}, "
            f"high(>15%) {reg['high']['capture_median']:.2f} "
            f"(n = {reg['low']['n_cells']}/{reg['mid']['n_cells']}/{reg['high']['n_cells']}).")
    lines += ["", "## Figures", ""] + [f"- {p}" for p in res.get("figures", [])]
    lines += ["", "## Files", "",
              f"- {OUT_DIR / 'per_volume.csv'} — full 153-volume table",
              f"- {OUT_DIR / 'aggregate_by_target.csv'}",
              f"- {OUT_DIR / 'cells.csv.gz'} — 64^3 cell table",
              f"- {OUT_DIR / 'results.json'}"]
    (OUT_DIR / "findings.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res: dict = {"voxel_size_um_assumed": VOXEL_UM, "min_cc_voxels": MIN_CC,
                 "edge_shell_vox": EDGE_VOX}

    print("Loading real calibration crops...", flush=True)
    real = {name: load_real_crop(name) for name in REAL_VOLUMES}

    print("Calibrating detector...", flush=True)
    cal = calibrate(real)
    res["detector"] = cal
    print(f"  T_best={cal['t_best']} dice={cal['dice_at_t_best']:.3f} "
          f"T_cons={cal['t_cons']} prec={cal['precision_at_t_cons']:.3f} "
          f"sigma={cal['sigma_u8']:.1f}", flush=True)

    print("Validating on real full crops (CC-filtered)...", flush=True)
    res["real_validation"] = validate_real(real, cal)
    for n, r in res["real_validation"].items():
        print(f"  {n.split('__')[-1][:40]}: dice {r['best']['dice']:.3f} "
              f"FP-unmasked best {r['best']['false_positive_unmasked_fraction']:.4f} "
              f"cons {r['cons']['false_positive_unmasked_fraction']:.4f}", flush=True)

    print("Auditing generated volumes...", flush=True)
    rows, cell_frames = [], []
    for exp, arm, d in iter_cell_dirs():
        row, cdf = audit_generated(d, cal, exp, arm)
        rows.append(row)
        cell_frames.append(cdf)
        print(f"  {exp}/{arm}/{d.name}: unmasked best "
              f"{row['unmasked_air_best']:.4f} cons {row['unmasked_air_cons']:.4f} "
              f"interior {row['unmasked_interior_best']:.4f}", flush=True)

    pv = pd.DataFrame(rows)
    cells = pd.concat(cell_frames, ignore_index=True)
    pv.drop(columns=["top_components"]).to_csv(OUT_DIR / "per_volume.csv", index=False)
    pv[["experiment", "arm", "name", "top_components"]].to_json(
        OUT_DIR / "top_components.json", orient="records", indent=1)
    cells.to_csv(OUT_DIR / "cells.csv.gz", index=False, compression="gzip")

    agg_t = aggregate(pv)
    agg_t.to_csv(OUT_DIR / "aggregate_by_target.csv", index=False)
    agg_arm = (pv.assign(target=np.nan).pipe(aggregate)
                 .drop(columns=["target"]))
    agg_arm.to_csv(OUT_DIR / "aggregate_by_arm.csv", index=False)
    res["aggregate_by_arm"] = agg_arm.to_dict(orient="records")

    print("Mask-collapse stats...", flush=True)
    res["mask_collapse"] = collapse_stats(cells)

    print("Figures...", flush=True)
    figs = []
    figs += fig_montage(cal, real, res)
    figs += fig_unmasked_by_arm(pv, res["real_validation"], cal)
    figs += fig_interior_edge(pv)
    figs += fig_mask_collapse(cells)
    res["figures"] = figs

    write_json(res, OUT_DIR)
    write_findings(res, pv, agg_arm)
    print(f"Done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

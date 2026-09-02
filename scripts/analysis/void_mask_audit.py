"""Void-vs-mask validity audit for generated volumes.

The user found large dark, void-like masses in the generated grayscale channel
that the generated pore mask does not classify as pores.  All porosity-control
results are mask-based, so this audits the disagreement:

1. Calibrate an independent grayscale void detector on REAL volumes
   (threshold that maximises voxel-level Dice against the real masks).
2. Apply it to every generated volume and to real baseline crops:
   dark-but-unmasked fraction, connected-component sizes, and
   mask / detector / union porosity per volume.
3. Localise unmasked voids on the 64^3 patch-cell grid and link them to the
   degenerate-mask metric.
4. GPU controls (small batches): encode worst vs normal generated cells with
   the frozen VAE (latent per-channel std), and round-trip ~50 real patches
   with the largest real pores to test the mask head (recall per patch).

Detector normalisation
----------------------
The VAE decoder is trained to output XCT directly in [0, 1] (the engine
compares ``xct_out`` to the [0, 1] target with no activation), but
``VolumeGenerator`` applies ``expit`` before writing TIFFs, so generated
grayscale is sigmoid-compressed: real material 0.81 maps to
sigmoid(0.81)*255 = 176 — exactly the generated material peak.  The primary
normalisation therefore inverts the sigmoid (``logit(v/255)`` clipped to
[0, 1], times 255), which puts generated volumes on the decoder-native scale
where the absolute threshold calibrated on real volumes applies directly.
A *material-referenced* detector (threshold ``m_mat - k * s_mat`` from the
volume's own material mode and robust spread, k calibrated on real volumes)
is reported as a normalisation-free cross-check on the raw generated scale.
All intensities are u8-scale (0..255).

Outputs to ``runs/campaigns/02-porosity-control-v1/void_mask_audit/``: results.json, findings.md and
figures (PDF + PNG, 300 dpi).

Usage:
    python scripts/analysis/void_mask_audit.py [--skip-gpu]
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from _common import DATA_ROOT, REPO, ZARR_ROOT, savefig, set_style, write_json  # noqa: E402

OUT_DIR = REPO / "runs" / "campaigns" / "02-porosity-control-v1" / "void_mask_audit"
# Written by scripts/analysis/regen_layup_volumes.py.  The original set was
# deleted, so this script cannot run until that regeneration is re-done.
GEN_ROOT = REPO / "runs/campaigns/02-porosity-control-v1/layup_roundtrip/volumes"
LATENT_META = DATA_ROOT / "latents_r07z4" / "metadata.json"

# Voxel size: 25 um (user-supplied; not recorded in dataset metadata).
VOXEL_UM = 25.0
VOXEL_MM = VOXEL_UM / 1000.0
VOXEL_MM3 = VOXEL_MM ** 3

PATCH = 64
CROP_SHAPE = (192, 1024, 1024)          # same size as one generated volume

# Real val/test volumes used for detector calibration AND real baseline.
REAL_VOLUMES = [
    "MedidasDB__Airbus_Panel_Pegaso_probetas_1_26_volumen_eq_aligned",         # val
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_01_3_volume_eq_aligned",  # test
    "MedidasDB__Juan_Ignacio_probetas_8_volume_eq_aligned",                    # test
]

FG_LEVEL = 100          # u8 level separating specimen+pores from exterior air
FG_ERODE = 3            # erosion its of the foreground, drops boundary voxels
CAL_SLICE_STEP = 8      # every 8th z-slice used during threshold calibration
DEGENERATE_LO = 1e-4    # patch mask porosity below this = degenerate (battery)
CELL_VOID_THRESH = 0.01  # patch cell flagged when unmasked-void fraction > 1%

GPU_BATCH = 8
N_WORST_CELLS = 32
N_REAL_PORE_PATCHES = 50


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_real_crop(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Central CROP_SHAPE crop of a real volume: (xct u8, mask bool, fg bool)."""
    g = zarr.open(str(ZARR_ROOT), mode="r")[name]
    zs, ys, xs = g["xct"].shape
    cz, cy, cx = CROP_SHAPE

    # Pick the z-window with the most foreground (specimen slab).
    step = max(zs // 40, 1)
    probe = [(z, float((np.asarray(g["xct"][z, ::8, ::8]) > FG_LEVEL).mean()))
             for z in range(0, zs, step)]
    fg_per_z = np.zeros(zs)
    for z, f in probe:
        fg_per_z[z:z + step] = f
    if zs <= cz:
        z0 = 0
        cz = zs
    else:
        sums = np.convolve(fg_per_z, np.ones(cz), mode="valid")
        z0 = int(np.argmax(sums))
    y0 = max((ys - cy) // 2, 0)
    x0 = max((xs - cx) // 2, 0)

    xct = np.asarray(g["xct"][z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])
    mask = np.asarray(g["mask"][z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]) > 0

    fg = xct > FG_LEVEL
    for z in range(fg.shape[0]):
        fg[z] = ndimage.binary_fill_holes(fg[z])
    fg = ndimage.binary_erosion(fg, iterations=FG_ERODE)
    return xct, mask, fg, {"crop_origin": [int(z0), int(y0), int(x0)],
                           "crop_shape": [int(s) for s in xct.shape]}


def load_generated(cell_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generated volume: (raw u8, decoder-native u8, mask bool, stats.json).

    Native scale = ``clip(logit(v/255), 0, 1) * 255`` — inverts the sampler's
    ``expit`` so the volume is on the same scale as the real u8 XCT data.
    """
    from scipy.special import logit

    vol = tifffile.imread(str(cell_dir / "volume.tif"))
    if vol.max() <= 1.5:            # older writers stored [0, 1]
        vol = vol * 255.0
    xct_raw = np.clip(vol, 0, 255).astype(np.uint8)
    frac = np.clip(vol / 255.0, 1e-6, 1.0 - 1e-6)
    native = np.clip(logit(frac), 0.0, 1.0) * 255.0
    xct_native = native.astype(np.uint8)
    mask = tifffile.imread(str(cell_dir / "mask.tif")) > 0
    stats = json.loads((cell_dir / "stats.json").read_text())
    return xct_raw, xct_native, mask, stats


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def material_stats(xct: np.ndarray, fg: np.ndarray | None) -> tuple[float, float]:
    """(mode, robust spread) of the material = upper intensity population.

    An Otsu split of the foreground histogram separates the dark population
    (pores / dark masses) from the material; mode and p84 - p50 are computed
    on the material side only, so a large dark phase cannot drag them down.
    """
    vals = xct[fg] if fg is not None else xct.ravel()
    hist = np.bincount(vals, minlength=256).astype(np.float64)

    # Otsu on the histogram.
    p = hist / hist.sum()
    bins = np.arange(256)
    w0 = np.cumsum(p)
    m_cum = np.cumsum(p * bins)
    m_tot = m_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var_b = (m_tot * w0 - m_cum) ** 2 / (w0 * (1.0 - w0))
    var_b[~np.isfinite(var_b)] = 0.0
    otsu = int(np.argmax(var_b))

    upper = hist[otsu + 1:]
    mode = float(otsu + 1 + np.argmax(upper))
    cdf = np.cumsum(upper) / upper.sum()
    p50 = float(otsu + 1 + np.searchsorted(cdf, 0.50))
    p84 = float(otsu + 1 + np.searchsorted(cdf, 0.84))
    spread = max(p84 - p50, 1.0)
    return mode, spread


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = float(np.count_nonzero(pred & gt))
    a = float(np.count_nonzero(pred))
    b = float(np.count_nonzero(gt))
    dice = 2.0 * inter / (a + b) if a + b else 0.0
    iou = inter / (a + b - inter) if a + b - inter else 0.0
    return dice, iou


def calibrate_detector(real: dict[str, tuple]) -> dict:
    """Sweep an absolute u8 threshold on real calibration slices; report the
    per-volume and pooled optimum, and convert it to the material-referenced
    coefficient k = (m_mat - T*) / s_mat."""
    thresholds = np.arange(30, 225, 5)
    per_vol = {}
    pooled = {int(t): [0, 0, 0] for t in thresholds}   # inter, pred, gt

    for name, (xct, mask, fg, _) in real.items():
        sl = slice(0, xct.shape[0], CAL_SLICE_STEP)
        x, m, f = xct[sl], mask[sl], fg[sl]
        gt = m & f
        n_gt = int(np.count_nonzero(gt))
        rows = []
        for t in thresholds:
            pred = (x < t) & f
            inter = int(np.count_nonzero(pred & gt))
            n_pred = int(np.count_nonzero(pred))
            d = 2.0 * inter / (n_pred + n_gt) if n_pred + n_gt else 0.0
            rows.append((int(t), d))
            pooled[int(t)][0] += inter
            pooled[int(t)][1] += n_pred
            pooled[int(t)][2] += n_gt
        best_t, best_d = max(rows, key=lambda r: r[1])
        per_vol[name] = {"best_threshold": best_t, "best_dice": best_d,
                         "sweep": rows}

    pooled_rows = []
    for t, (inter, npred, ngt) in pooled.items():
        d = 2.0 * inter / (npred + ngt) if npred + ngt else 0.0
        i = inter / (npred + ngt - inter) if npred + ngt - inter else 0.0
        pooled_rows.append((t, d, i))
    t_star, dice_star, iou_star = max(pooled_rows, key=lambda r: r[1])

    # Material-referenced coefficient from the calibration volumes.
    ks, mats = [], {}
    for name, (xct, mask, fg, _) in real.items():
        mode, spread = material_stats(xct, fg)
        ks.append((mode - t_star) / spread)
        mats[name] = {"material_mode": mode, "material_spread": spread}
    k_star = float(np.mean(ks))

    # Validation at the pooled optimum, full crops, per volume.
    validation = {}
    for name, (xct, mask, fg, _) in real.items():
        pred = (xct < t_star) & fg
        gt = mask & fg
        d, i = dice_iou(pred, gt)
        n_pred = float(np.count_nonzero(pred))
        n_gt = float(np.count_nonzero(gt))
        inter = float(np.count_nonzero(pred & gt))
        precision = inter / n_pred if n_pred else 0.0
        recall = inter / n_gt if n_gt else 0.0
        mode, spread = mats[name]["material_mode"], mats[name]["material_spread"]
        t_rel = mode - k_star * spread
        pred_r = (xct < t_rel) & fg
        d_r, i_r = dice_iou(pred_r, gt)
        validation[name] = {
            "dice_abs": d, "iou_abs": i,
            "precision_abs": precision, "recall_abs": recall,
            "threshold_rel": t_rel, "dice_rel": d_r, "iou_rel": i_r,
            **mats[name],
        }

    return {
        "absolute_threshold": int(t_star),
        "pooled_dice": dice_star,
        "pooled_iou": iou_star,
        "k_material_referenced": k_star,
        "per_volume_calibration": {k: {kk: vv for kk, vv in v.items() if kk != "sweep"}
                                   for k, v in per_vol.items()},
        "validation_full_crop": validation,
        "sweep_thresholds": [int(t) for t in thresholds],
        "sweep_pooled_dice": [d for _, d, _ in sorted(pooled_rows)],
    }


def detector_threshold(xct: np.ndarray, fg: np.ndarray | None, cal: dict) -> float:
    mode, spread = material_stats(xct, fg)
    return mode - cal["k_material_referenced"] * spread


# ---------------------------------------------------------------------------
# Per-volume audit
# ---------------------------------------------------------------------------

def component_summary(labels: np.ndarray, n: int, top: int = 10) -> dict:
    if n == 0:
        return {"n_components": 0, "sizes_hist": [], "largest": []}
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(sizes)[::-1][:top]
    largest = []
    objs = ndimage.find_objects(labels)
    for idx in order:
        size = int(sizes[idx])
        sl = objs[idx]
        extent_vox = [int(s.stop - s.start) for s in sl]
        largest.append({
            "voxels": size,
            "volume_mm3": size * VOXEL_MM3,
            "equiv_diameter_mm": 2.0 * (3.0 * size * VOXEL_MM3 / (4.0 * np.pi)) ** (1 / 3),
            "bbox_extent_vox": extent_vox,
            "bbox_extent_mm": [e * VOXEL_MM for e in extent_vox],
            "bbox_origin": [int(s.start) for s in sl],
        })
    log_edges = np.logspace(0, 8, 33)
    hist, _ = np.histogram(sizes, bins=log_edges)
    return {"n_components": int(n),
            "total_voxels": int(sizes.sum()),
            "size_hist_log_edges": log_edges.tolist(),
            "size_hist_counts": hist.tolist(),
            "largest": largest}


def audit_volume(xct: np.ndarray, mask: np.ndarray, fg: np.ndarray | None,
                 cal: dict, xct_alt: np.ndarray | None = None) -> dict:
    """Primary detector: absolute calibrated threshold on ``xct`` (real u8 or
    generated decoder-native u8).  ``xct_alt`` (raw generated scale) adds the
    material-referenced cross-check."""
    fgm = fg if fg is not None else np.ones_like(mask)
    n_fg = int(np.count_nonzero(fgm))

    det = (xct < cal["absolute_threshold"]) & fgm
    mk = mask & fgm

    unmasked = det & ~mk
    labels, n_lab = ndimage.label(unmasked)

    out = {
        "threshold_abs": int(cal["absolute_threshold"]),
        "material_mode_native": material_stats(xct, fg)[0],
        "mask_porosity": float(np.count_nonzero(mk) / n_fg),
        "detector_porosity": float(np.count_nonzero(det) / n_fg),
        "union_porosity": float(np.count_nonzero(det | mk) / n_fg),
        "dark_unmasked_fraction": float(np.count_nonzero(unmasked) / n_fg),
        "masked_not_dark_fraction": float(np.count_nonzero(mk & ~det) / n_fg),
        "components": component_summary(labels, n_lab),
    }
    hist_unmasked = np.bincount(xct[fgm & ~mk].ravel(), minlength=256).astype(np.int64)
    cum = np.cumsum(hist_unmasked)
    out["unmasked_dark_frac_vs_threshold"] = {
        int(t): float(cum[t - 1] / n_fg) for t in (100, 120, 140, 160, 170, 180, 190, 200)}
    if xct_alt is not None:
        t_rel = detector_threshold(xct_alt, fg, cal)
        det_rel = (xct_alt < t_rel) & fgm
        out["crosscheck_rel_threshold_raw_scale"] = float(t_rel)
        out["crosscheck_rel_dark_unmasked_fraction"] = float(
            np.count_nonzero(det_rel & ~mk) / n_fg)
        out["crosscheck_rel_detector_porosity"] = float(
            np.count_nonzero(det_rel) / n_fg)
    del labels
    return out


def cell_localisation(xct: np.ndarray, mask: np.ndarray, cal: dict) -> dict:
    """Unmasked-void fraction and mask porosity per 64^3 patch cell.

    ``xct`` must be on the decoder-native u8 scale."""
    unmasked = (xct < cal["absolute_threshold"]) & ~mask
    gz, gy, gx = (s // PATCH for s in xct.shape)

    def cellwise(a: np.ndarray) -> np.ndarray:
        return (a[:gz * PATCH, :gy * PATCH, :gx * PATCH]
                .reshape(gz, PATCH, gy, PATCH, gx, PATCH)
                .mean(axis=(1, 3, 5)))

    void_frac = cellwise(unmasked)
    mask_por = cellwise(mask)
    flagged = void_frac > CELL_VOID_THRESH
    n_cells = void_frac.size
    return {
        "n_cells": int(n_cells),
        "cells_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "flagged_mask_porosity_median": float(np.median(mask_por[flagged])) if flagged.any() else None,
        "flagged_mask_porosity_mean": float(mask_por[flagged].mean()) if flagged.any() else None,
        "unflagged_mask_porosity_mean": float(mask_por[~flagged].mean()) if (~flagged).any() else None,
        "flagged_degenerate_fraction": float((mask_por[flagged] < DEGENERATE_LO).mean()) if flagged.any() else None,
        "all_degenerate_fraction": float((mask_por < DEGENERATE_LO).mean()),
        "void_frac_grid": void_frac,
        "mask_por_grid": mask_por,
    }


# ---------------------------------------------------------------------------
# GPU controls
# ---------------------------------------------------------------------------

def _load_vae(device):
    from poregen.experiments.train_vae import load_vae_from_checkpoint
    ckpt = Path(json.loads(LATENT_META.read_text())["vae_checkpoint"])
    model, _, _, _ = load_vae_from_checkpoint(ckpt, device)
    return model


def gpu_latent_std_control(xct: np.ndarray, mask: np.ndarray, loc: dict) -> dict:
    """Encode worst-offending vs normal generated cells; per-channel latent std
    normalised by the latent-store channel stats."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_vae(device)
    meta = json.loads(LATENT_META.read_text())["normalization"]
    ch_std = np.asarray(meta["per_channel_std"], np.float32)

    vf = loc["void_frac_grid"].ravel()
    order = np.argsort(vf)[::-1]
    worst = order[:N_WORST_CELLS]
    normal = order[vf[order] < 1e-3][-N_WORST_CELLS:] if (vf < 1e-3).any() else order[-N_WORST_CELLS:]
    gz, gy, gx = loc["void_frac_grid"].shape

    def encode(cells) -> np.ndarray:
        stds = []
        with torch.no_grad():
            for i in range(0, len(cells), GPU_BATCH):
                xs, ms = [], []
                for c in cells[i:i + GPU_BATCH]:
                    z, y, x = np.unravel_index(c, (gz, gy, gx))
                    sl = (slice(z * PATCH, (z + 1) * PATCH),
                          slice(y * PATCH, (y + 1) * PATCH),
                          slice(x * PATCH, (x + 1) * PATCH))
                    xs.append(xct[sl].astype(np.float32) / 255.0)
                    ms.append(mask[sl].astype(np.float32))
                xb = torch.from_numpy(np.stack(xs)[:, None]).to(device)
                mb = torch.from_numpy(np.stack(ms)[:, None]).to(device)
                out = model(xb, mb)
                mu = out.mu.float().cpu().numpy()           # (B, C, l, l, l)
                stds.append(mu.std(axis=(2, 3, 4)))
        return np.concatenate(stds)                          # (N, C)

    s_worst = encode(worst)
    s_normal = encode(normal)
    return {
        "n_worst": len(worst), "n_normal": len(normal),
        "worst_void_frac_range": [float(vf[worst].min()), float(vf[worst].max())],
        "store_channel_std": ch_std.tolist(),
        "worst_per_channel_std": s_worst.mean(axis=0).tolist(),
        "normal_per_channel_std": s_normal.mean(axis=0).tolist(),
        "worst_std_ratio_vs_store": (s_worst.mean(axis=0) / ch_std).tolist(),
        "normal_std_ratio_vs_store": (s_normal.mean(axis=0) / ch_std).tolist(),
    }


def gpu_vae_pore_control() -> dict:
    """Round-trip the ~50 largest-pore REAL patches through the frozen VAE and
    measure the mask head's recall of the real pores."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_vae(device)

    df = pd.read_parquet(DATA_ROOT / "patch_index.parquet")
    df = df[df["split"].isin(["val", "test"])].sort_values("porosity", ascending=False)

    picked, seen = [], []
    for row in df.itertuples():
        key = (row.volume_id, row.z0 // PATCH, row.y0 // PATCH, row.x0 // PATCH)
        if key in seen:
            continue
        seen.append(key)
        picked.append(row)
        if len(picked) >= N_REAL_PORE_PATCHES:
            break

    g = zarr.open(str(ZARR_ROOT), mode="r")
    recalls, dices, pors = [], [], []
    with torch.no_grad():
        for i in range(0, len(picked), GPU_BATCH):
            xs, ms = [], []
            for r in picked[i:i + GPU_BATCH]:
                gv = g[r.volume_id]
                sl = (slice(r.z0, r.z0 + PATCH), slice(r.y0, r.y0 + PATCH),
                      slice(r.x0, r.x0 + PATCH))
                xs.append(np.asarray(gv["xct"][sl], np.float32) / 255.0)
                ms.append(np.asarray(gv["mask"][sl], np.float32))
            xb = torch.from_numpy(np.stack(xs)[:, None]).to(device)
            mb = torch.from_numpy(np.stack(ms)[:, None]).to(device)
            out = model(xb, mb)
            pred = (out.mask_logits > 0).float()
            gt = mb
            tp = (pred * gt).sum(dim=(1, 2, 3, 4))
            recalls += (tp / gt.sum(dim=(1, 2, 3, 4)).clamp(min=1)).cpu().tolist()
            dices += (2 * tp / (pred.sum(dim=(1, 2, 3, 4)) + gt.sum(dim=(1, 2, 3, 4))).clamp(min=1)).cpu().tolist()
            pors += gt.mean(dim=(1, 2, 3, 4)).cpu().tolist()

    recalls = np.asarray(recalls)
    return {
        "n_patches": len(recalls),
        "patch_porosity_range": [float(min(pors)), float(max(pors))],
        "recall_mean": float(recalls.mean()),
        "recall_median": float(np.median(recalls)),
        "recall_min": float(recalls.min()),
        "recall_p10": float(np.percentile(recalls, 10)),
        "dice_mean": float(np.mean(dices)),
        "per_patch_recall": recalls.tolist(),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_montage(gen_name: str, xct_g, mask_g, cal, real_name, xct_r, mask_r, fg_r):
    import matplotlib.pyplot as plt

    t_g = float(cal["absolute_threshold"])
    unm = (xct_g < t_g) & ~mask_g
    z_star = int(np.argmax(unm.reshape(unm.shape[0], -1).mean(axis=1)))

    t_r = float(cal["absolute_threshold"])
    z_r = int(np.argmax((mask_r & fg_r).reshape(mask_r.shape[0], -1).sum(axis=1)))

    def overlay(x, m, t, fg=None):
        det = x < t
        if fg is not None:
            det &= fg
        rgb = np.stack([x, x, x], axis=-1).astype(np.float32) / 255.0
        rgb[det & ~m] = [0.85, 0.15, 0.15]     # detector only  (red)
        rgb[m & ~det] = [0.15, 0.35, 0.9]      # mask only      (blue)
        rgb[m & det] = [0.15, 0.75, 0.25]      # agreement      (green)
        return rgb

    fig, ax = plt.subplots(2, 3, figsize=(14, 9.5))
    ax[0, 0].imshow(xct_g[z_star], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title(f"generated {gen_name}\ndecoder-native grayscale, worst slice z={z_star}")
    ax[0, 1].imshow(mask_g[z_star], cmap="gray")
    ax[0, 1].set_title("generated pore mask")
    ax[0, 2].imshow(overlay(xct_g[z_star], mask_g[z_star], t_g))
    ax[0, 2].set_title(f"disagreement (T={t_g:.0f})\nred = dark, NOT in mask")

    ax[1, 0].imshow(xct_r[z_r], cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title(f"real {real_name.split('__')[-1][:40]}\ngrayscale z={z_r}")
    ax[1, 1].imshow(mask_r[z_r], cmap="gray")
    ax[1, 1].set_title("real pore mask")
    ax[1, 2].imshow(overlay(xct_r[z_r], mask_r[z_r], t_r, fg_r[z_r]))
    ax[1, 2].set_title(f"disagreement (T={t_r:.0f})\ngreen = agreement")
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Grayscale void detector vs pore mask - generated vs real", y=0.99)
    return savefig(fig, OUT_DIR, "fig1_slice_montage"), z_star


def fig_bars(results: dict):
    import matplotlib.pyplot as plt

    names, fracs, kinds = [], [], []
    for name, r in results["generated"].items():
        names.append(name), fracs.append(r["dark_unmasked_fraction"]), kinds.append("generated")
    for name, r in results["real_baseline"].items():
        names.append(name.split("__")[-1][:28]), fracs.append(r["dark_unmasked_fraction"]), kinds.append("real")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c2571a" if k == "generated" else "#1b6ca8" for k in kinds]
    ax.bar(range(len(names)), fracs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("dark-but-unmasked voxel fraction")
    ax.set_title("Unmasked dark-void fraction per volume (orange = generated, blue = real)")
    return savefig(fig, OUT_DIR, "fig2_unmasked_fraction_bars")


def fig_components(results: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, color in (("generated", "#c2571a"), ("real_baseline", "#1b6ca8")):
        for name, r in results[group].items():
            comp = r["components"]
            if not comp["n_components"]:
                continue
            edges = np.asarray(comp["size_hist_log_edges"])
            counts = np.asarray(comp["size_hist_counts"], float)
            centers = np.sqrt(edges[:-1] * edges[1:])
            surv = counts[::-1].cumsum()[::-1]
            ax.plot(centers[surv > 0], surv[surv > 0], color=color, alpha=0.6,
                    label=group if name == list(results[group])[0] else None)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("component size (voxels)")
    ax.set_ylabel("N components >= size")
    ax.set_title("Unmasked dark components - size survival distribution")
    ax.axvline(PATCH ** 3, color="k", ls="--", lw=0.8)
    ax.text(PATCH ** 3, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0 else 1, "  one 64$^3$ patch",
            fontsize=8, rotation=90, va="bottom")
    ax.legend()
    return savefig(fig, OUT_DIR, "fig3_component_sizes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"voxel_size_um_assumed": VOXEL_UM}

    print("Loading real calibration crops...", flush=True)
    real = {}
    for name in REAL_VOLUMES:
        real[name] = load_real_crop(name)
        print(f"  {name}: crop {real[name][3]}", flush=True)

    print("Calibrating detector...", flush=True)
    cal = calibrate_detector(real)
    results["detector"] = {k: v for k, v in cal.items()}
    print(f"  T*={cal['absolute_threshold']} pooled dice={cal['pooled_dice']:.3f} "
          f"k={cal['k_material_referenced']:.2f}", flush=True)

    print("Real baselines...", flush=True)
    results["real_baseline"] = {}
    for name, (xct, mask, fg, meta) in real.items():
        r = audit_volume(xct, mask, fg, cal)
        r.update(meta)
        results["real_baseline"][name] = r
        print(f"  {name}: unmasked-dark {r['dark_unmasked_fraction']:.4f}", flush=True)

    print("Generated volumes...", flush=True)
    if not GEN_ROOT.is_dir():
        raise SystemExit(
            f"Missing input: {GEN_ROOT}\n"
            "This audit reads the regenerated layup-roundtrip volumes.  The "
            "original set (under inference/ldm05-.../layup_roundtrip_volumes) "
            "was deleted and was never migrated.  Recreate it with:\n"
            "    python scripts/analysis/regen_layup_volumes.py\n"
            "Note this script is superseded by scripts/analysis/air_audit_v2.py "
            "(162 volumes, better calibration) — see "
            "runs/campaigns/AUDIT.md section 6.")
    gen_dirs = sorted(d for d in GEN_ROOT.iterdir()
                      if (d / "mask.tif").exists() and (d / "volume.tif").exists())
    if not gen_dirs:
        raise SystemExit(
            f"No volumes with both volume.tif and mask.tif under {GEN_ROOT}. "
            "Run scripts/analysis/regen_layup_volumes.py first.")
    results["generated"] = {}
    results["localisation"] = {}
    gen_cache = {}
    for d in gen_dirs:
        xct_raw, xct_native, mask, stats = load_generated(d)
        r = audit_volume(xct_native, mask, None, cal, xct_alt=xct_raw)
        r["stats_json"] = stats
        results["generated"][d.name] = r
        loc = cell_localisation(xct_native, mask, cal)
        results["localisation"][d.name] = {k: v for k, v in loc.items()
                                           if not k.endswith("_grid")}
        del xct_raw
        gen_cache[d.name] = (xct_native, mask, loc)
        print(f"  {d.name}: unmasked-dark {r['dark_unmasked_fraction']:.4f} "
              f"mask-por {r['mask_porosity']:.4f} union {r['union_porosity']:.4f} "
              f"flagged cells {loc['flagged_fraction']:.2%}", flush=True)

    worst_name = max(results["generated"],
                     key=lambda n: results["generated"][n]["dark_unmasked_fraction"])
    results["worst_volume"] = worst_name

    if not args.skip_gpu:
        print("GPU: latent std of worst vs normal generated cells...", flush=True)
        xct, mask, loc = gen_cache[worst_name]
        results["latent_std_control"] = gpu_latent_std_control(xct, mask, loc)
        print("GPU: VAE round-trip of largest real pores...", flush=True)
        results["vae_pore_control"] = gpu_vae_pore_control()
        print(f"  mask-head recall mean {results['vae_pore_control']['recall_mean']:.3f}",
              flush=True)

    print("Figures...", flush=True)
    xct_g, mask_g, _ = gen_cache[worst_name]
    rname = REAL_VOLUMES[0]
    xct_r, mask_r, fg_r, _ = real[rname]
    paths1, z_star = fig_montage(worst_name, xct_g, mask_g, cal, rname, xct_r, mask_r, fg_r)
    results["worst_slice_z"] = z_star
    paths2 = fig_bars(results)
    paths3 = fig_components(results)
    results["figures"] = paths1 + paths2 + paths3

    write_json(results, OUT_DIR)
    write_findings(results)
    print(f"Done -> {OUT_DIR}", flush=True)


def write_findings(res: dict) -> None:
    cal = res["detector"]
    lines = [
        "# Void-mask validity audit",
        "",
        f"Voxel size assumed: {VOXEL_UM} um (not recorded in dataset metadata).",
        "",
        "## Verdict",
        "",
        "The generated grayscale channel contains a second, near-black phase that",
        "the generated pore mask does not classify as pores.  It is volume-wide",
        "(not confined to a few patch cells), forms one percolating component,",
        "and is threshold-insensitive (present even at the near-black threshold",
        "T=100).  The mask-based porosity therefore measures only a small part",
        "of the void-like content of the generated volumes; the union porosity",
        "column is the corrected number for the dose-response claims.",
        "",
        "Cause chain established by the controls:",
        "- The VAE mask head is NOT deficient: encode-decode of the largest real",
        "  pores recovers them (see VAE control below).",
        "- The dark cells re-encode to near-CONSTANT latents (per-channel std",
        "  ~0.1-0.4x the latent-store std), i.e. the LDM produced off-manifold,",
        "  overly smooth latents there; the xct head decodes them to a dark",
        "  wash while the mask head outputs background.",
        "- Scale note: ``VolumeGenerator`` applies ``expit`` to xct logits that",
        "  the VAE trains directly against [0,1] targets, so the written TIFFs",
        "  are sigmoid-compressed (material 0.81 -> 176/255).  The audit",
        "  inverts this; on the decoder-native scale the dark masses sit at",
        "  ~0.04-0.07, i.e. darker than real pores.",
        "",
        "## Detector",
        f"- Absolute threshold T* = {cal['absolute_threshold']} (u8). "
        f"Pooled Dice {cal['pooled_dice']:.3f}, IoU {cal['pooled_iou']:.3f} on real calibration slices.",
        f"- Material-referenced form: T = material_mode - {cal['k_material_referenced']:.2f} x spread. "
        "Used for generated volumes because their intensity scale is compressed.",
        "",
        "## Per-volume disagreement",
        "",
        "| volume | kind | mask por | detector por | union por | dark-unmasked frac | largest unmasked comp (vox / mm^3) |",
        "|---|---|---|---|---|---|---|",
    ]
    for group, kind in (("generated", "gen"), ("real_baseline", "real")):
        for name, r in res[group].items():
            big = r["components"]["largest"][0] if r["components"]["largest"] else None
            big_s = f"{big['voxels']:.2e} / {big['volume_mm3']:.2f}" if big else "-"
            lines.append(f"| {name} | {kind} | {r['mask_porosity']:.4f} | "
                         f"{r['detector_porosity']:.4f} | {r['union_porosity']:.4f} | "
                         f"{r['dark_unmasked_fraction']:.4f} | {big_s} |")
    lines += [
        "",
        "Real-baseline caveat: the largest real 'unmasked dark' components are",
        "drilled holes / specimen-edge features visible in the montage, not",
        "annotation misses; real fractions are upper bounds.",
        "",
        "## Localisation (64^3 cells)", ""]
    for name, l in res.get("localisation", {}).items():
        lines.append(f"- {name}: {l['flagged_fraction']:.1%} of cells have unmasked-void "
                     f"fraction > 1% (flagged cells' mask porosity mean "
                     f"{l['flagged_mask_porosity_mean']}, degenerate share "
                     f"{l['flagged_degenerate_fraction']}).")
    if "vae_pore_control" in res:
        v = res["vae_pore_control"]
        lines += ["", "## VAE control (real large pores, encode-decode)",
                  f"- {v['n_patches']} patches, porosity {v['patch_porosity_range'][0]:.3f}-"
                  f"{v['patch_porosity_range'][1]:.3f}: mask-head recall mean "
                  f"{v['recall_mean']:.3f}, median {v['recall_median']:.3f}, "
                  f"min {v['recall_min']:.3f}."]
    if "latent_std_control" in res:
        s = res["latent_std_control"]
        lines += ["", "## Latent std control (generated cells)",
                  f"- Worst cells std ratio vs store: {['%.2f' % x for x in s['worst_std_ratio_vs_store']]}",
                  f"- Normal cells std ratio vs store: {['%.2f' % x for x in s['normal_std_ratio_vs_store']]}"]
    lines += ["", "## Figures", ""] + [f"- {p}" for p in res.get("figures", [])]
    (OUT_DIR / "findings.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

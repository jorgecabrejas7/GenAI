#!/usr/bin/env python3
"""eval_generated_volumes.py — Stage 4 generation quality evaluation for PoreGen.

Compares generated XCT volumes (from LDM inference) against real held-out volumes
across the full evaluation suite required for the paper.

Metrics
-------
1.  Porosity fraction MAE — mean |phi_gen - phi_real| per volume; also distribution W1.
2.  Pore Size Distribution W1 — pooled equivalent diameters d=(6V/pi)^(1/3), Wasserstein-1.
3.  Two-Point Correlation S2(r) — FFT autocorrelation; W1 between mean curves; plot.
4.  Ripley's K(r) — pore centroid clustering; W1; summary value at mean pore spacing.
5.  FID on 2D slices — 64×64 native-resolution crops (~167/axis/volume); resize to 299×299;
    InceptionV3 pool3.  Full-slice resize rejected: ~10× downscale shrinks 1.79-voxel pores
    to 0.18 pixels.
6.  Boundary inconsistency — seam vs interior MAE ratio at stride-interval positions.
7.  Pore morphology — sphericity (marching cubes) and aspect ratio; mean/std/P5/P50/P95.
8.  Diversity — std(phi), W1-PSD variance, Ripley K variance across volumes.
9.  Memorisation check — min L2 distance from generated patch latents to training latents.

Volume pairs are discovered by scanning for (volume.tif|xct.tif) + mask.tif in each dir.

Usage
-----
# Full eval suite (structural metrics + FID):
python scripts/eval_generated_volumes.py \\
    --real-dir      path/to/real/volumes/ \\
    --generated-dir inference/ldm03-run-0001-.../ \\
    [--baseline-dir inference/baseline/] \\
    [--out-dir      eval_results/stage4/] \\
    [--vae-run      runs/vae/r05-run-0001-.../] \\
    [--latents-dir  data/split_v2/latents_s64_sampled/] \\
    [--stride 32] [--r-max 50] [--crops-per-volume 500] \\
    [--device cuda]

# FID-only mode (fast; skips structural metrics):
python scripts/eval_generated_volumes.py \\
    --fid \\
    --real-dir      path/to/real/volumes/ \\
    --generated-dir inference/ldm03-run-0001-.../ \\
    [--baseline-dir inference/baseline/] \\
    [--crops-per-volume 500] [--out-dir eval_results/fid/]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.stats import wasserstein_distance
from skimage.measure import label as sk_label, regionprops

logger = logging.getLogger(__name__)

# ── EDA ground truth (flag generated metrics that deviate by more than 3×) ──
EDA_PHI_MEAN      = 0.055
EDA_PHI_STD       = 0.027
EDA_MEDIAN_DIAM   = 1.79   # voxels — median equivalent pore diameter
EDA_P90_DIAM      = 2.92   # voxels
EDA_S2_R30_COEFF  = 4.73   # S2(r=30) ≈ coeff × phi²

PATCH_SIZE = 64
MEDIAN_PORE_VOL = (math.pi / 6.0) * 1.79 ** 3  # ≈ 3.0 voxels³ — for per-volume pore-count estimation


# ─────────────────────────────────────────────────────────────────────────────
# Volume discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(root: Path, mask_name: str = "only_pores.tif") -> list[tuple[Path, Path]]:
    """Recursively find (xct, mask) TIFF pairs under *root*.

    Only directories containing exactly *mask_name* are included — there is no
    fallback.  Mixing mask types (e.g. Sauvola only_pores.tif vs VAE mask.tif)
    corrupts every structural metric, so pass the correct --mask-name explicitly.

    XCT file priority per directory:
      1. volume.tif  (generate_volumes.py output)
      2. xct.tif
    """
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []

    # Only collect dirs that contain exactly mask_name — no fallback.
    # Falling back to mask.tif when only_pores.tif was requested would silently
    # mix Sauvola-segmented masks with VAE decoder masks, corrupting every metric.
    candidate_dirs: set[Path] = set()
    for p in root.rglob(mask_name):
        candidate_dirs.add(p.parent)

    for parent in sorted(candidate_dirs):
        mask_p = parent / mask_name
        if not mask_p.exists():
            continue

        # Resolve XCT
        for xct_name in ("volume.tif", "xct.tif"):
            xct_p = parent / xct_name
            if xct_p.exists():
                pairs.append((xct_p, mask_p))
                break

    return sorted(pairs)


def load_volume(xct_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an (xct, mask) TIFF pair.

    XCT: uint8 → float32 in [0, 1], or float32 pass-through (clamped to [0, 1]).
    Mask: binarised at 0.5 (float) or 128 (uint8).

    Returns
    -------
    xct  : float32 array in [0, 1], shape (D, H, W)
    mask : bool array, shape (D, H, W)
    """
    xct_raw = tifffile.imread(str(xct_path))
    msk_raw = tifffile.imread(str(mask_path))

    # Normalise XCT
    if xct_raw.dtype == np.uint8:
        xct = xct_raw.astype(np.float32) / 255.0
    else:
        xct = xct_raw.astype(np.float32)
        if xct.max() > 1.5:          # still in uint-scale
            xct = xct / 255.0
    xct = np.clip(xct, 0.0, 1.0)

    # Binarise mask — handle both {0,1} and {0,255} uint8 encodings
    if msk_raw.dtype == np.uint8:
        mask = msk_raw > 0 if msk_raw.max() <= 1 else msk_raw >= 128
    else:
        mask = msk_raw.astype(np.float32) >= 0.5

    return xct, mask


# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives (self-contained; no poregen imports needed)
# ─────────────────────────────────────────────────────────────────────────────

def _s2_radial_crop(
    binary: np.ndarray,
    r_max: int = 64,
    n_bins: int = 64,
    crop_size: int = 128,
    n_crops: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic S₂(r) via FFT autocorrelation, averaged over random crops.

    Cropping avoids OOM for large assembled volumes (128×3200×1280).
    The crop_size is clamped to the actual volume dimensions.
    """
    rng = np.random.default_rng(seed)
    D, H, W = binary.shape
    cs = min(crop_size, D, H, W)

    # Bins are capped at crop × 0.4: beyond this, FFT edge effects dominate
    safe_r_max = min(r_max, int(cs * 0.4))
    r_edges = np.linspace(0, safe_r_max, n_bins + 1)
    r_vals  = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Build 3D Hann window once per crop shape (suppresses spectral leakage from finite boundaries)
    hann_1d = np.hanning(cs)
    hann_3d = hann_1d[:, None, None] * hann_1d[None, :, None] * hann_1d[None, None, :]

    # Pre-compute window autocorrelation (constant for all crops of the same shape).
    # Dividing S2_raw by win_autocorr debiases the Hann-weighted estimate:
    #   S2_raw[r]    = Σ_x f(x)·w(x) · f(x+r)·w(x+r)  ≈  S2(r) · win_autocorr[r]
    #   win_autocorr[r] = Σ_x w(x)·w(x+r)
    # so S2_raw / win_autocorr ≈ S2(r).  Zero-lag check: S2[0,0,0] must equal φ.
    fft_win      = np.fft.fftn(hann_3d)
    win_autocorr = np.real(np.fft.ifftn(fft_win * np.conj(fft_win)))

    s2_accumulator = np.zeros(n_bins, dtype=np.float64)
    n_valid = 0

    for _ in range(n_crops):
        dz = int(rng.integers(0, max(1, D - cs + 1)))
        dy = int(rng.integers(0, max(1, H - cs + 1)))
        dx = int(rng.integers(0, max(1, W - cs + 1)))
        crop = binary[dz:dz + cs, dy:dy + cs, dx:dx + cs].astype(np.float64)

        windowed  = crop * hann_3d
        fft_crop  = np.fft.fftn(windowed)
        s2_raw    = np.real(np.fft.ifftn(fft_crop * np.conj(fft_crop)))
        autocorr  = np.where(win_autocorr > 1e-10, s2_raw / win_autocorr, 0.0)

        if __debug__:
            phi_crop = float(crop.mean())
            s2_zero  = float(autocorr[0, 0, 0])
            # Tolerance: 10% relative + 0.005 absolute (Hann weighting causes small bias)
            if phi_crop > 1e-6 and abs(s2_zero - phi_crop) > 0.1 * phi_crop + 0.005:
                logger.warning(
                    "S2 zero-lag check failed: S2[0,0,0]=%.5f but crop phi=%.5f "
                    "(normalization may be wrong)", s2_zero, phi_crop
                )

        freq_z = np.fft.fftfreq(cs) * cs
        ZZ, YY, XX = np.meshgrid(freq_z, freq_z, freq_z, indexing="ij")
        R_grid = np.sqrt(ZZ**2 + YY**2 + XX**2)

        s2 = np.zeros(n_bins)
        for i in range(n_bins):
            m = (R_grid >= r_edges[i]) & (R_grid < r_edges[i + 1])
            if m.any():
                s2[i] = autocorr[m].mean()

        s2_accumulator += s2
        n_valid += 1

    return r_vals, s2_accumulator / max(n_valid, 1)


def _s2_w1(s2_a: np.ndarray, s2_b: np.ndarray) -> float:
    """Wasserstein-1 distance between two S₂(r) curves treated as distributions.

    Returns NaN when either curve is all-zero (e.g. empty pore masks).
    """
    a = np.clip(s2_a, 0, None)
    b = np.clip(s2_b, 0, None)
    sa, sb = float(a.sum()), float(b.sum())
    if sa < 1e-30 or sb < 1e-30:
        return float("nan")
    a = a / sa
    b = b / sb
    r = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(r, r, a, b))


def _pore_diameters(binary: np.ndarray) -> np.ndarray:
    """Equivalent spherical diameter d = (6V/π)^(1/3) per connected component."""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.array([], dtype=np.float64)
    vols = np.array(ndimage.sum(binary, labeled, range(1, n + 1)), dtype=np.float64)
    return (6.0 * vols / math.pi) ** (1.0 / 3.0)


def _ripleys_k(
    binary: np.ndarray,
    r_max: int = 32,
    max_pores: int = 5000,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Ripley's K(r) estimated from pore centre-of-mass positions.

    Returns ``None`` when fewer than 3 pores are found.
    """
    from scipy.spatial.distance import pdist

    labeled = sk_label(binary)
    props   = regionprops(labeled)
    if len(props) < 3:
        return None
    if len(props) > max_pores:
        logger.warning(
            "Ripley K: subsampling %d pores to %d (O(N²) cap)", len(props), max_pores
        )
        props = random.sample(props, max_pores)

    centroids = np.array([p.centroid for p in props], dtype=np.float64)
    N = len(centroids)
    V = float(binary.shape[0] * binary.shape[1] * binary.shape[2])
    dists  = pdist(centroids)
    r_vals = np.arange(1, r_max + 1, dtype=np.float64)
    # Factor of 2: pdist gives unordered pairs (i<j); multiply by 2 to count all ordered pairs (i≠j)
    K      = np.array([(V / N**2) * 2.0 * float(np.sum(dists < r)) for r in r_vals])
    return r_vals, K


def _ripleys_k_per_patch(
    binary: np.ndarray,
    r_max: int = 32,
    max_pores: int = 5000,
    patch_size: int = PATCH_SIZE,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Ripley's K(r) averaged over non-overlapping patch_size³ sub-volumes.

    Avoids O(N²) scaling on full assembled volumes with millions of pore centres.
    Per-patch computation is consistent with the spatial scale the model operates at.
    """
    D, H, W = binary.shape
    K_list: list[np.ndarray] = []
    for z0 in range(0, D - patch_size + 1, patch_size):
        for y0 in range(0, H - patch_size + 1, patch_size):
            for x0 in range(0, W - patch_size + 1, patch_size):
                patch = binary[z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size]
                res = _ripleys_k(patch, r_max=r_max, max_pores=max_pores)
                if res is not None:
                    K_list.append(res[1])
    if not K_list:
        return None
    r_vals = np.arange(1, r_max + 1, dtype=np.float64)
    return r_vals, np.mean(K_list, axis=0)


def _boundary_inconsistency(
    xct: np.ndarray,
    mask: np.ndarray,
    stride: int = 32,
) -> dict:
    """Seam vs interior MAE ratio for volumes assembled from overlapping patches.

    Seam positions are at multiples of *stride* along each axis.  Interior
    positions are sampled (up to 200 per axis) to keep runtime bounded on
    large volumes (128×3200×1280).  Returns per-modality seam_mae,
    interior_mae, and their ratio.
    """
    rng = random.Random(0)
    result: dict = {}

    for name, vol in [("xct", xct), ("mask", mask.astype(np.float32))]:
        seam_acc, seam_n = 0.0, 0
        int_acc,  int_n  = 0.0, 0

        for axis in range(3):
            n = vol.shape[axis]
            seam_set   = set(range(stride, n, stride))
            seam_pos   = [i for i in range(1, n) if i in seam_set]
            int_pos    = [i for i in range(1, n) if i not in seam_set]

            if len(int_pos) > 200:
                int_pos = sorted(rng.sample(int_pos, 200))

            for i in seam_pos:
                a = np.take(vol, i - 1, axis=axis)
                b = np.take(vol, i,     axis=axis)
                seam_acc += float(np.abs(a - b).mean())
                seam_n += 1

            for i in int_pos:
                a = np.take(vol, i - 1, axis=axis)
                b = np.take(vol, i,     axis=axis)
                int_acc += float(np.abs(a - b).mean())
                int_n += 1

        seam_mae = seam_acc / max(seam_n, 1)
        int_mae  = int_acc  / max(int_n,  1)
        result[f"{name}_seam_mae"]   = seam_mae
        result[f"{name}_int_mae"]    = int_mae
        result[f"{name}_seam_ratio"] = seam_mae / max(int_mae, 1e-9)

    return result


def _morphology_stats(binary: np.ndarray, max_pores: int = 2000) -> dict:
    """Per-pore sphericity and aspect ratio distribution summaries.

    Sphericity = π^(1/3) · (6V)^(2/3) / A  (Wadell, via marching cubes).
    Aspect ratio = bbox longest / shortest axis.
    """
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError:
        # skimage < 0.19 uses the Lewiner variant
        from skimage.measure import marching_cubes_lewiner as marching_cubes, mesh_surface_area  # type: ignore[no-redef]

    labeled = sk_label(binary)
    props   = regionprops(labeled)

    _nan_stats: dict = {"mean": float("nan"), "std": float("nan"),
                        "p5": float("nan"), "p50": float("nan"), "p95": float("nan")}

    if not props:
        return {"sphericity": _nan_stats, "aspect_ratio": _nan_stats, "n_pores": 0}

    props = sorted(props, key=lambda p: p.area, reverse=True)[:max_pores]

    sphericities: list[float] = []
    aspect_ratios: list[float] = []

    for p in props:
        # Aspect ratio from 3D bounding-box (min/max for each spatial axis)
        bb   = p.bbox           # (min_z, min_y, min_x, max_z, max_y, max_x)
        dims = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
        if dims[0] > 0:
            aspect_ratios.append(float(dims[2] / dims[0]))

        # Sphericity via marching cubes surface area
        vol = p.area
        if vol < 4:
            continue
        try:
            sub = (labeled[p.slice] == p.label).astype(np.float32)
            verts, faces, _, _ = marching_cubes(sub, level=0.5)
            A = mesh_surface_area(verts, faces)
            if A > 0:
                sphericities.append(
                    float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / A)
                )
        except Exception:
            pass

    def _summ(vals: list[float]) -> dict:
        if not vals:
            return dict(_nan_stats)
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "p5":   float(np.percentile(arr, 5)),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
        }

    return {
        "sphericity":   _summ(sphericities),
        "aspect_ratio": _summ(aspect_ratios),
        "n_pores":      len(props),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-volume metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_metrics(
    xct: np.ndarray,
    mask: np.ndarray,
    *,
    r_max: int = 64,
    n_bins: int = 64,
    stride: int = 32,
    s2_crop_size: int = 128,
    n_s2_crops: int = 3,
    run_ripley: bool = True,
    run_morphology: bool = True,
    max_pores_ripley: int = 5000,
    max_pores_morph: int = 2000,
    ripley_per_patch: bool = False,
    volume_id: str = "",
) -> dict:
    """Compute all structural metrics for one generated or real volume.

    Returns a dict suitable for JSON serialisation.  Per-volume curves (S₂,
    Ripley K) are stored as lists for later pooling and plotting.
    """
    t0  = time.perf_counter()
    phi = float(mask.mean())

    # S₂(r) — averaged over random crops of crop_size³ for large volumes
    try:
        r_vals, s2_curve = _s2_radial_crop(
            mask, r_max=r_max, n_bins=n_bins,
            crop_size=s2_crop_size, n_crops=n_s2_crops,
        )
        s2_r30 = float(np.interp(30.0, r_vals, s2_curve))
    except Exception as exc:
        logger.warning("S2(r) failed for %s: %s", volume_id, exc)
        r_vals   = np.linspace(0, r_max, n_bins)
        s2_curve = np.full(n_bins, float("nan"))
        s2_r30   = float("nan")

    # PSD — equivalent diameters per component
    try:
        diameters = _pore_diameters(mask)
    except Exception as exc:
        logger.warning("PSD failed for %s: %s", volume_id, exc)
        diameters = np.array([])

    # Ripley's K — auto-switch to per-patch when estimated pore count exceeds 50 000
    ripley_r: Optional[np.ndarray] = None
    ripley_K: Optional[np.ndarray] = None
    if run_ripley:
        try:
            est_pore_count = phi * float(mask.size) / MEDIAN_PORE_VOL
            use_per_patch = ripley_per_patch or (est_pore_count > 50_000)
            _rk_fn = _ripleys_k_per_patch if use_per_patch else _ripleys_k
            res = _rk_fn(mask, r_max=max(r_max // 2, 16), max_pores=max_pores_ripley)
            if res is not None:
                ripley_r, ripley_K = res
        except Exception as exc:
            logger.warning("Ripley K failed for %s: %s", volume_id, exc)

    # Boundary inconsistency
    boundary: dict = {}
    try:
        boundary = _boundary_inconsistency(xct, mask, stride=stride)
    except Exception as exc:
        logger.warning("Boundary failed for %s: %s", volume_id, exc)

    # Pore morphology
    morphology: dict = {}
    if run_morphology:
        try:
            morphology = _morphology_stats(mask, max_pores=max_pores_morph)
        except Exception as exc:
            logger.warning("Morphology failed for %s: %s", volume_id, exc)

    elapsed = time.perf_counter() - t0
    logger.info(
        "  [%s] phi=%.4f  n_pores=%d  s2(r=30)=%.3e  %.1fs",
        volume_id, phi, int(len(diameters)), s2_r30, elapsed,
    )

    return {
        "volume_id":     volume_id,
        "porosity":      phi,
        "s2_r_vals":     r_vals.tolist(),
        "s2_curve":      s2_curve.tolist(),
        "psd_diameters": diameters.tolist(),
        "ripley_r":      ripley_r.tolist() if ripley_r is not None else None,
        "ripley_K":      ripley_K.tolist() if ripley_K is not None else None,
        "boundary":      boundary,
        "morphology":    morphology,
        "elapsed_s":     elapsed,
    }


def _process_all_pairs(
    pairs: list[tuple[Path, Path]],
    label: str,
    **kwargs,
) -> list[dict]:
    """Load and compute per-volume metrics for all pairs in a directory."""
    metrics: list[dict] = []
    for i, (xct_p, mask_p) in enumerate(pairs):
        vid = f"{label}_{i:03d}_{xct_p.parent.name}"
        logger.info("[%d/%d] %s", i + 1, len(pairs), vid)
        try:
            xct, mask = load_volume(xct_p, mask_p)
            m = compute_volume_metrics(xct, mask, volume_id=vid, **kwargs)
            metrics.append(m)
        except Exception as exc:
            logger.error("Failed on %s: %s", xct_p, exc)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_comparison(real_metrics: list[dict], gen_metrics: list[dict]) -> dict:
    """Pool per-volume metrics across all volumes and compute aggregate scalars."""
    result: dict = {}

    # ── 1. Porosity ──
    phi_real = [m["porosity"] for m in real_metrics]
    phi_gen  = [m["porosity"] for m in gen_metrics]

    if phi_real and phi_gen:
        result["porosity_mean_real"] = float(np.mean(phi_real))
        result["porosity_std_real"]  = float(np.std(phi_real))
        result["porosity_mean_gen"]  = float(np.mean(phi_gen))
        result["porosity_std_gen"]   = float(np.std(phi_gen))
        result["porosity_dist_mae"]  = abs(result["porosity_mean_gen"] - result["porosity_mean_real"])
        result["porosity_w1"]        = float(wasserstein_distance(phi_gen, phi_real))
        if len(phi_real) == len(phi_gen):
            result["porosity_mae_paired"] = float(np.mean([abs(g - r) for g, r in zip(phi_gen, phi_real)]))

    # ── 2. PSD W1 on equivalent diameters ──
    diam_real = (np.concatenate([m["psd_diameters"] for m in real_metrics if m["psd_diameters"]])
                 if any(m["psd_diameters"] for m in real_metrics) else np.array([]))
    diam_gen  = (np.concatenate([m["psd_diameters"] for m in gen_metrics  if m["psd_diameters"]])
                 if any(m["psd_diameters"] for m in gen_metrics)  else np.array([]))

    result["psd_n_real"] = int(len(diam_real))
    result["psd_n_gen"]  = int(len(diam_gen))

    if len(diam_real) > 0 and len(diam_gen) > 0:
        result["psd_w1"]          = float(wasserstein_distance(diam_gen, diam_real))
        result["psd_median_real"] = float(np.median(diam_real))
        result["psd_median_gen"]  = float(np.median(diam_gen))
        result["psd_p90_real"]    = float(np.percentile(diam_real, 90))
        result["psd_p90_gen"]     = float(np.percentile(diam_gen, 90))
        result["psd_mean_real"]   = float(diam_real.mean())
        result["psd_mean_gen"]    = float(diam_gen.mean())
    else:
        result["psd_w1"] = float("nan")

    # Store pooled arrays for plotting (not JSON-serialised directly)
    result["_diam_real"] = diam_real
    result["_diam_gen"]  = diam_gen

    # ── 3. S₂(r) W1 ──
    def _valid_s2(ml: list[dict]) -> list[np.ndarray]:
        curves = []
        for m in ml:
            c = np.array(m["s2_curve"])
            if not np.all(np.isnan(c)):
                curves.append(c)
        return curves

    s2_real_list = _valid_s2(real_metrics)
    s2_gen_list  = _valid_s2(gen_metrics)

    if s2_real_list and s2_gen_list:
        s2_real_mean = np.nanmean(s2_real_list, axis=0)
        s2_gen_mean  = np.nanmean(s2_gen_list,  axis=0)
        result["s2_rmse"]          = float(np.sqrt(np.mean((s2_gen_mean - s2_real_mean) ** 2)))
        result["s2_r_vals"]        = real_metrics[0]["s2_r_vals"]
        result["s2_curve_real"]    = s2_real_mean.tolist()
        result["s2_curve_gen"]     = s2_gen_mean.tolist()

        per_vol_w1 = [_s2_w1(np.array(m["s2_curve"]), s2_real_mean) for m in gen_metrics
                      if not np.all(np.isnan(m["s2_curve"]))]
        result["s2_w1_per_vol_mean"] = float(np.mean(per_vol_w1)) if per_vol_w1 else float("nan")
        result["s2_w1_per_vol_std"]  = float(np.std(per_vol_w1))  if per_vol_w1 else float("nan")

        # EDA sanity: S2(r=30) ≈ EDA_S2_R30_COEFF × phi²
        r_arr     = np.array(result["s2_r_vals"])
        s2_gen_30 = float(np.interp(30.0, r_arr, s2_gen_mean))
        phi_mu    = result.get("porosity_mean_gen", float("nan"))
        if not math.isnan(phi_mu) and phi_mu > 0:
            expected = EDA_S2_R30_COEFF * phi_mu ** 2
            result["s2_r30_check_ratio"] = s2_gen_30 / max(expected, 1e-14)
        else:
            result["s2_r30_check_ratio"] = float("nan")
    else:
        result["s2_rmse"] = float("nan")

    # ── 4. Ripley's K W1 ──
    rk_real = [m for m in real_metrics if m.get("ripley_K") is not None]
    rk_gen  = [m for m in gen_metrics  if m.get("ripley_K") is not None]

    if rk_real and rk_gen:
        K_real_mean = np.mean([m["ripley_K"] for m in rk_real], axis=0)
        K_gen_mean  = np.mean([m["ripley_K"] for m in rk_gen],  axis=0)
        result["ripley_w1"]      = float(wasserstein_distance(K_gen_mean, K_real_mean))
        result["ripley_r_vals"]  = rk_real[0]["ripley_r"]
        result["ripley_K_real"]  = K_real_mean.tolist()
        result["ripley_K_gen"]   = K_gen_mean.tolist()

        # Summary value at mean pore spacing and CSR deviation
        if len(diam_gen) > 0:
            r_ripley      = np.array(result["ripley_r_vals"])
            mean_spacing  = float(diam_gen.mean())
            K_at_spacing_gen  = float(np.interp(mean_spacing, r_ripley, K_gen_mean))
            result["ripley_K_at_mean_spacing_gen"]  = K_at_spacing_gen
            result["ripley_K_at_mean_spacing_real"] = float(np.interp(mean_spacing, r_ripley, K_real_mean))
            # K_gen / K_CSR at r* = mean pore spacing: >1 clustered, <1 regular, 1 = random
            K_csr_at_spacing = (4.0 / 3.0) * math.pi * mean_spacing ** 3
            result["ripley_K_csr_deviation_gen"] = K_at_spacing_gen / max(K_csr_at_spacing, 1e-14)
    else:
        result["ripley_w1"] = float("nan")

    # ── 6. Boundary inconsistency (generated) ──
    for key in ("xct_seam_ratio", "mask_seam_ratio", "xct_seam_mae", "mask_seam_mae"):
        vals = [m["boundary"].get(key, float("nan")) for m in gen_metrics if m.get("boundary")]
        vals = [v for v in vals if not math.isnan(v)]
        result[f"boundary_{key}_mean_gen"] = float(np.mean(vals)) if vals else float("nan")

    for key in ("xct_seam_ratio", "mask_seam_ratio"):
        vals = [m["boundary"].get(key, float("nan")) for m in real_metrics if m.get("boundary")]
        vals = [v for v in vals if not math.isnan(v)]
        result[f"boundary_{key}_mean_real"] = float(np.mean(vals)) if vals else float("nan")

    # ── 7. Morphology comparison ──
    def _morph_stat(ml: list[dict], field: str, stat: str) -> float:
        vals = [m["morphology"].get(field, {}).get(stat, float("nan"))
                for m in ml if m.get("morphology")]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    for field in ("sphericity", "aspect_ratio"):
        for stat in ("mean", "std", "p5", "p50", "p95"):
            result[f"morph_{field}_{stat}_gen"]  = _morph_stat(gen_metrics,  field, stat)
            result[f"morph_{field}_{stat}_real"] = _morph_stat(real_metrics, field, stat)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Diversity metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity(gen_metrics: list[dict], real_metrics: list[dict]) -> dict:
    """Diversity: std(phi), W1-PSD variance, Ripley K variance, vs real."""
    result: dict = {}

    # std(phi)
    phi_gen  = [m["porosity"] for m in gen_metrics]
    phi_real = [m["porosity"] for m in real_metrics]
    result["diversity_phi_std_gen"]  = float(np.std(phi_gen))  if phi_gen  else float("nan")
    result["diversity_phi_std_real"] = float(np.std(phi_real)) if phi_real else float("nan")

    # W1-PSD variance: per-volume W1(gen_vol, pooled_real)
    diam_real_pool = (
        np.concatenate([m["psd_diameters"] for m in real_metrics if m["psd_diameters"]])
        if any(m["psd_diameters"] for m in real_metrics) else np.array([])
    )

    def _per_vol_psd_w1(ml: list[dict]) -> list[float]:
        out = []
        for m in ml:
            d = m["psd_diameters"]
            if len(d) > 0 and len(diam_real_pool) > 0:
                out.append(float(wasserstein_distance(d, diam_real_pool)))
        return out

    w1_gen  = _per_vol_psd_w1(gen_metrics)
    w1_real = _per_vol_psd_w1(real_metrics)

    result["diversity_psd_w1_mean_gen"]  = float(np.mean(w1_gen))  if w1_gen  else float("nan")
    result["diversity_psd_w1_std_gen"]   = float(np.std(w1_gen))   if w1_gen  else float("nan")
    result["diversity_psd_w1_mean_real"] = float(np.mean(w1_real)) if w1_real else float("nan")
    result["diversity_psd_w1_std_real"]  = float(np.std(w1_real))  if w1_real else float("nan")

    # Ripley K variance: std of per-volume mean K over volumes
    rk_gen  = [np.mean(m["ripley_K"]) for m in gen_metrics  if m.get("ripley_K")]
    rk_real = [np.mean(m["ripley_K"]) for m in real_metrics if m.get("ripley_K")]
    result["diversity_ripley_K_std_gen"]  = float(np.std(rk_gen))  if rk_gen  else float("nan")
    result["diversity_ripley_K_std_real"] = float(np.std(rk_real)) if rk_real else float("nan")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FID on 2D slices — crop-based protocol
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid_crop_based(
    real_vols: list[np.ndarray],
    gen_vols:  list[np.ndarray],
    crops_per_volume: int = 500,
    device_str: str = "cpu",
) -> dict[str, float]:
    """FID via 64×64 native-resolution crops resized to 299×299 for InceptionV3.

    Extracts *crops_per_volume* 2D crops per 3D volume (≈ crops_per_volume//3 per
    axis), normalises to [0,1] float32, replicates to 3 channels, resizes 64→299
    with bilinear interpolation, then computes FID from empirical pool3 (2048-d)
    feature statistics.

    The 64×64 crop window matches the training-patch spatial scale, preserving pore
    structures (median diameter ~1.79 voxels) that would vanish under the ~10×
    downscale required to fit a full slice into InceptionV3's 299×299 input.

    Raises
    ------
    ValueError
        If the total crop count for either set is below 5 000.  Pass more volumes
        or increase *crops_per_volume*.

    Returns a dict with keys ``axial``, ``coronal``, ``sagittal``, ``mean``.
    """
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.models as tvm
        from scipy.linalg import sqrtm as scipy_sqrtm
    except ImportError:
        logger.warning("torchvision / scipy.linalg not available — skipping FID")
        return {"axial": float("nan"), "coronal": float("nan"),
                "sagittal": float("nan"), "mean": float("nan")}

    if device_str != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available; using CPU for FID")
        device_str = "cpu"
    device = torch.device(device_str)

    # Guard: FID is unreliable below ~5k samples per set
    MIN_CROPS = 5_000
    total_real = len(real_vols) * crops_per_volume
    total_gen  = len(gen_vols)  * crops_per_volume
    if total_real < MIN_CROPS or total_gen < MIN_CROPS:
        raise ValueError(
            f"FID requires ≥{MIN_CROPS} total crops per set to be reliable; "
            f"got {total_real} real ({len(real_vols)} volumes) and "
            f"{total_gen} generated ({len(gen_vols)} volumes) at "
            f"{crops_per_volume} crops/vol. "
            f"Increase --crops-per-volume or provide more volumes."
        )

    inception = tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)
    inception.eval().to(device)

    pool3_out: list[torch.Tensor] = []

    def _hook(m, inp, out: torch.Tensor) -> None:
        pool3_out.append(out.detach().cpu().view(out.shape[0], -1))

    inception.avgpool.register_forward_hook(_hook)

    CROP         = 64   # native-resolution spatial scale matching training patches
    INCEPTION_SZ = 299  # InceptionV3 input; bilinear upsample from CROP

    axes           = ("axial", "coronal", "sagittal")
    crops_per_axis = max(1, crops_per_volume // len(axes))

    def _extract_crops(volumes: list[np.ndarray], axis_name: str) -> list[np.ndarray]:
        axis_idx = {"axial": 0, "coronal": 1, "sagittal": 2}[axis_name]
        rng      = random.Random(42)
        crops: list[np.ndarray] = []
        for vol in volumes:
            n_slices = vol.shape[axis_idx]
            for _ in range(crops_per_axis):
                sl_idx = rng.randint(0, n_slices - 1)
                sl = np.take(vol, sl_idx, axis=axis_idx).astype(np.float32)
                h, w = sl.shape
                if h < CROP:
                    sl = np.pad(sl, ((0, CROP - h), (0, 0)), mode="reflect")
                    h  = sl.shape[0]
                if w < CROP:
                    sl = np.pad(sl, ((0, 0), (0, CROP - w)), mode="reflect")
                    w  = sl.shape[1]
                y0 = rng.randint(0, h - CROP)
                x0 = rng.randint(0, w - CROP)
                crops.append(sl[y0:y0 + CROP, x0:x0 + CROP])
        return crops

    def _crops_to_features(crops: list[np.ndarray]) -> torch.Tensor:
        all_feats: list[torch.Tensor] = []
        bs = 32
        with torch.no_grad():
            for start in range(0, len(crops), bs):
                batch_np = np.stack(crops[start:start + bs])                       # (B, 64, 64)
                t = torch.from_numpy(batch_np).unsqueeze(1)                        # (B, 1, 64, 64)
                t = F.interpolate(t, size=(INCEPTION_SZ, INCEPTION_SZ),
                                  mode="bilinear", align_corners=False)            # (B, 1, 299, 299)
                t = t.expand(-1, 3, -1, -1).to(device)                            # (B, 3, 299, 299)
                pool3_out.clear()
                inception(t)
                if pool3_out:
                    all_feats.append(pool3_out[-1])
        return torch.cat(all_feats) if all_feats else torch.zeros(0, 2048)

    def _fid(r: torch.Tensor, g: torch.Tensor) -> float:
        if r.shape[0] < 2 or g.shape[0] < 2:
            return float("nan")
        try:
            r_np  = r.numpy().astype(np.float64)
            g_np  = g.numpy().astype(np.float64)
            mu_r, mu_g = r_np.mean(0), g_np.mean(0)
            sig_r = np.cov(r_np, rowvar=False)
            sig_g = np.cov(g_np, rowvar=False)
            diff  = mu_r - mu_g
            # Add small epsilon to diagonal before sqrtm to avoid numerical instability
            eps     = 1e-6
            product = sig_r @ sig_g + eps * np.eye(sig_r.shape[0])
            cm, _   = scipy_sqrtm(product, disp=False)
            if np.iscomplexobj(cm):
                cm = cm.real
            return float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * cm))
        except Exception as exc:
            logger.warning("FID numeric error: %s", exc)
            return float("nan")

    out: dict[str, float] = {}
    vals: list[float] = []

    for axis_name in axes:
        logger.info("FID %s: extracting %d 64×64 crops per volume …",
                    axis_name, crops_per_axis)
        real_crops = _extract_crops(real_vols, axis_name)
        gen_crops  = _extract_crops(gen_vols,  axis_name)
        logger.info("  %s: %d real crops, %d gen crops",
                    axis_name, len(real_crops), len(gen_crops))
        r_feats = _crops_to_features(real_crops)
        g_feats = _crops_to_features(gen_crops)
        fid_val = _fid(r_feats, g_feats)
        out[axis_name] = fid_val
        logger.info("  FID %s = %.2f", axis_name, fid_val)
        if not math.isnan(fid_val):
            vals.append(fid_val)

    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Memorisation check
# ─────────────────────────────────────────────────────────────────────────────

def compute_memorization(
    gen_pairs:       list[tuple[Path, Path]],
    vae_run_dir:     Path,
    latents_dir:     Path,
    device_str:      str = "cpu",
    n_train_sample:  int = 10_000,
    patch_size:      int = PATCH_SIZE,
) -> dict:
    """Mean min-L2 distance from generated patch latents to training latents.

    High values indicate no memorisation; near-zero indicates overfitting.
    Requires ``--vae-run`` pointing to a directory with ``resolved_config.yaml``
    and ``checkpoints/best.ckpt``.
    """
    try:
        import torch
        import yaml
        from poregen.models.vae import build_vae
        from poregen.training.checkpoint import load_checkpoint
    except ImportError as exc:
        logger.warning("poregen or torch not importable for memorization: %s", exc)
        return {}

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    # ── Load VAE ──
    cfg_path = vae_run_dir / "resolved_config.yaml"
    # Support two checkpoint layouts: checkpoints/best.ckpt (new) and best.ckpt (legacy r05)
    ckpt_path = vae_run_dir / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = vae_run_dir / "best.ckpt"
    if not cfg_path.exists() or not ckpt_path.exists():
        logger.warning("VAE config/checkpoint not found in %s — skipping memorization", vae_run_dir)
        return {}

    with open(cfg_path) as f:
        resolved_cfg = yaml.safe_load(f)

    model_cfg = resolved_cfg.get("model", {})
    try:
        model = build_vae(
            model_cfg["name"],
            z_channels=model_cfg["z_channels"],
            base_channels=model_cfg["base_channels"],
            n_blocks=model_cfg["n_blocks"],
            patch_size=model_cfg.get("patch_size", 64),
        ).to(device)
        load_checkpoint(str(ckpt_path), model, restore_rng=False, map_location=device)
        model.eval()
    except Exception as exc:
        logger.warning("Failed to build/load VAE: %s", exc)
        return {}

    def _encode_h(mdl, xct_t: "torch.Tensor") -> "torch.Tensor":
        if hasattr(mdl, "encoder"):
            return mdl.encoder(xct_t)
        h_a = mdl.encoder_a(xct_t)
        h_b = mdl.encoder_b(xct_t)
        return mdl.fusion(torch.cat([h_a, h_b], dim=1))

    # ── Load training latents (mu only, channels 0–z_channels-1) ──
    meta_path   = latents_dir / "latents_meta.json"
    latents_bin = latents_dir / "latents.bin"
    if not meta_path.exists() or not latents_bin.exists():
        logger.warning("Latents files missing in %s — skipping memorization", latents_dir)
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    N_total   = meta["N"]
    n_ch_all  = meta["n_channels"]      # 32 (mu + logvar)
    z_ch      = meta.get("z_channels", n_ch_all // 2)   # 16 (mu channels)
    sp        = meta["spatial"]          # [16, 16, 16]

    all_latents = np.memmap(
        str(latents_bin), dtype=np.float16, mode="r",
        shape=(N_total, n_ch_all, sp[0], sp[1], sp[2]),
    )
    n_sample = min(n_train_sample, N_total)
    idxs     = np.random.choice(N_total, n_sample, replace=False)
    train_mu = all_latents[idxs, :z_ch].astype(np.float32)   # (N, z_ch, 16, 16, 16)
    del all_latents

    import torch
    train_mu_flat = torch.from_numpy(
        train_mu.reshape(n_sample, -1)                         # (N, z_ch*16*16*16)
    )

    # ── Encode generated volume patches ──
    gen_mu_list: list[torch.Tensor] = []
    with torch.no_grad():
        for xct_path, _ in gen_pairs:
            xct_raw = tifffile.imread(str(xct_path))
            xct = (xct_raw.astype(np.float32) / 255.0
                   if xct_raw.dtype == np.uint8 else xct_raw.astype(np.float32))
            xct = np.clip(xct, 0.0, 1.0)
            D, H, W = xct.shape

            patches = [
                xct[z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size]
                for z0 in range(0, D - patch_size + 1, patch_size)
                for y0 in range(0, H - patch_size + 1, patch_size)
                for x0 in range(0, W - patch_size + 1, patch_size)
            ]

            enc_bs = 4
            for b0 in range(0, len(patches), enc_bs):
                batch_np = np.stack(patches[b0:b0 + enc_bs])
                xct_t    = torch.from_numpy(batch_np).unsqueeze(1).to(device)
                h        = _encode_h(model, xct_t)
                mu       = model.to_mu(h)
                gen_mu_list.append(mu.reshape(mu.shape[0], -1).cpu())

    if not gen_mu_list:
        logger.warning("No generated patches encoded — skipping memorization")
        return {}

    gen_mu_flat = torch.cat(gen_mu_list, dim=0)   # (M, D_lat)

    # ── Min-L2 distance (chunked to avoid OOM) ──
    chunk = 256
    min_dists: list[torch.Tensor] = []
    for i in range(0, len(gen_mu_flat), chunk):
        gm = gen_mu_flat[i:i + chunk].unsqueeze(1)           # (C, 1, D)
        dm = (gm - train_mu_flat.float().unsqueeze(0)).pow(2).sum(-1).sqrt()  # (C, N)
        min_dists.append(dm.min(dim=1).values)

    all_dists = torch.cat(min_dists)
    return {
        "memorization_nn_dist_mean":     float(all_dists.mean()),
        "memorization_nn_dist_std":      float(all_dists.std()),
        "memorization_n_gen_patches":    int(gen_mu_flat.shape[0]),
        "memorization_n_train_latents":  n_sample,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks against EDA ground truth
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(result: dict, label: str = "generated") -> list[str]:
    """Check aggregate metrics against EDA ground truth.

    Returns a list of warning strings for any value that deviates > 3× from
    the EDA reference.
    """
    warns: list[str] = []

    def _check(val: float, ref: float, name: str) -> None:
        if not math.isnan(val) and not (ref / 3 < val < ref * 3):
            warns.append(
                f"WARN [{label}] {name} = {val:.4f} deviates > 3× from EDA ref {ref:.4f}"
            )

    _check(result.get("porosity_mean_gen",   float("nan")), EDA_PHI_MEAN,     "phi_mean")
    _check(result.get("psd_median_gen",      float("nan")), EDA_MEDIAN_DIAM,  "psd_median_diam_vox")
    _check(result.get("psd_p90_gen",         float("nan")), EDA_P90_DIAM,     "psd_p90_diam_vox")

    ratio = result.get("s2_r30_check_ratio", float("nan"))
    if not math.isnan(ratio) and not (1.0 / 3 < ratio < 3.0):
        warns.append(
            f"WARN [{label}] S2(r=30)/expected = {ratio:.3f} (expected near 1.0; "
            f"EDA coeff = {EDA_S2_R30_COEFF})"
        )

    return warns


def sanity_check_details(result: dict) -> list[dict]:
    """Structured pass/fail for every EDA sanity check (for report tables)."""
    checks: list[dict] = []

    def _chk(val: float, ref: float, name: str, lo: float = 3.0, hi: float = 3.0) -> None:
        if math.isnan(val):
            checks.append({"name": name, "actual": "n/a", "expected": ref, "passed": None})
        else:
            checks.append({
                "name": name, "actual": val, "expected": ref,
                "passed": (ref / lo) < val < (ref * hi),
            })

    _chk(result.get("porosity_mean_gen",   float("nan")), EDA_PHI_MEAN,    "phi_mean")
    _chk(result.get("psd_median_gen",      float("nan")), EDA_MEDIAN_DIAM, "psd_median_diam_vox")
    _chk(result.get("psd_p90_gen",         float("nan")), EDA_P90_DIAM,    "psd_p90_diam_vox")

    ratio = result.get("s2_r30_check_ratio", float("nan"))
    if math.isnan(ratio):
        checks.append({"name": "S2(r=30)/expected", "actual": "n/a", "expected": 1.0, "passed": None})
    else:
        checks.append({
            "name": "S2(r=30)/expected", "actual": ratio, "expected": 1.0,
            "passed": (1.0 / 3) < ratio < 3.0,
        })

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_s2_multi(
    r_vals: list,
    curves:  dict[str, list],   # label → curve
    out_path: Path,
) -> None:
    """Plot multiple S₂(r) curves on one axes."""
    colors = ["royalblue", "tomato", "seagreen", "darkorange"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for (label, curve), color in zip(curves.items(), colors):
        ax.plot(r_vals, curve, color=color, linewidth=2, label=label)
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("S₂(r)")
    ax.set_title("Two-Point Correlation Function S₂(r)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_psd_multi(
    diam_sets: dict[str, np.ndarray],   # label → diameter array
    out_path: Path,
) -> None:
    all_d = np.concatenate(list(diam_sets.values())) if diam_sets else np.array([1.0])
    upper = min(float(np.percentile(all_d, 99.5)), 20.0) if len(all_d) > 0 else 5.0
    bins  = np.linspace(0, upper, 60)
    colors = ["royalblue", "tomato", "seagreen", "darkorange"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for (label, d), color in zip(diam_sets.items(), colors):
        if len(d) == 0:
            continue
        for ax, scale in zip(axes, ("linear", "log")):
            ax.hist(d, bins=bins, density=True, alpha=0.6, color=color, label=label)

    for ax, scale, title in zip(axes, ("linear", "log"), ("PSD (linear)", "PSD (log scale)")):
        ax.set_xlabel("Equivalent diameter (voxels)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if scale == "log":
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ripley_multi(
    r_vals: list,
    K_curves: dict[str, list],
    out_path: Path,
) -> None:
    colors = ["royalblue", "tomato", "seagreen"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for (label, K), color in zip(K_curves.items(), colors):
        ax.plot(r_vals, K, color=color, linewidth=2, label=label)
    r_arr = np.array(r_vals, dtype=np.float64)
    ax.plot(r_arr, (4.0 / 3.0) * math.pi * r_arr ** 3,
            "k--", linewidth=1, alpha=0.5, label="CSR (4/3·π·r³)")
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("K(r)")
    ax.set_title("Ripley's K(r)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_fid_table(fid_results: dict[str, dict], out_path: Path) -> None:
    """Render FID comparison as a table image."""
    plane_names = ["axial", "coronal", "sagittal", "mean"]
    row_labels  = list(fid_results.keys())

    cell_text = []
    for label in row_labels:
        row = [
            f"{fid_results[label].get(p, float('nan')):.2f}"
            if not math.isnan(fid_results[label].get(p, float("nan"))) else "n/a"
            for p in plane_names
        ]
        cell_text.append(row)

    fig_h = max(2, len(row_labels) * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=plane_names,
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1.2, 2.0)
    ax.set_title("FID on 2D Slices — 64×64 crops (lower = better)", pad=16)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_fid_results(fid_results: dict[str, dict]) -> None:
    """Print per-axis and mean FID to stdout."""
    planes = ["axial", "coronal", "sagittal", "mean"]
    w = 72
    print("\n" + "=" * w)
    print("  FID — 64×64 native crops → bilinear 299×299 → InceptionV3 pool3")
    print("-" * w)
    print(f"  {'Set':<22}" + "".join(f"{p:>11}" for p in planes))
    print("-" * w)
    for label, fid_d in fid_results.items():
        row = f"  {label:<22}"
        for p in planes:
            v = fid_d.get(p, float("nan"))
            row += f"{'n/a':>11}" if math.isnan(v) else f"{v:>11.2f}"
        print(row)
    print("=" * w + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Results report
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_ROWS = [
    # (display label,                  gen_key,                          real_key)
    ("Porosity mean",                  "porosity_mean_gen",              "porosity_mean_real"),
    ("Porosity std",                   "porosity_std_gen",               "porosity_std_real"),
    ("Porosity W1",                    "porosity_w1",                    None),
    ("Porosity MAE (paired)",          "porosity_mae_paired",            None),
    ("PSD W1 (diameters, vox)",        "psd_w1",                         None),
    ("PSD median diam (vox)",          "psd_median_gen",                 "psd_median_real"),
    ("PSD P90 diam (vox)",             "psd_p90_gen",                    "psd_p90_real"),
    ("S2(r) RMSE",                      "s2_rmse",                        None),
    ("S2(r) per-vol W1 mean",          "s2_w1_per_vol_mean",             None),
    ("Ripley K W1",                    "ripley_w1",                      None),
    ("Ripley K at mean spacing (gen)", "ripley_K_at_mean_spacing_gen",   "ripley_K_at_mean_spacing_real"),
    ("Ripley K / CSR at mean spacing", "ripley_K_csr_deviation_gen",     None),
    ("FID mean",                       "fid_mean",                       None),
    ("FID axial",                      "fid_axial",                      None),
    ("FID coronal",                    "fid_coronal",                    None),
    ("FID sagittal",                   "fid_sagittal",                   None),
    ("Boundary XCT seam ratio",        "boundary_xct_seam_ratio_mean_gen", "boundary_xct_seam_ratio_mean_real"),
    ("Boundary mask seam ratio",       "boundary_mask_seam_ratio_mean_gen", "boundary_mask_seam_ratio_mean_real"),
    ("Morph sphericity mean",          "morph_sphericity_mean_gen",      "morph_sphericity_mean_real"),
    ("Morph sphericity P50",           "morph_sphericity_p50_gen",       "morph_sphericity_p50_real"),
    ("Morph aspect ratio mean",        "morph_aspect_ratio_mean_gen",    "morph_aspect_ratio_mean_real"),
    ("Diversity phi std",              "diversity_phi_std_gen",          "diversity_phi_std_real"),
    ("Diversity PSD W1 std",           "diversity_psd_w1_std_gen",       "diversity_psd_w1_std_real"),
    ("Diversity Ripley K std",         "diversity_ripley_K_std_gen",     "diversity_ripley_K_std_real"),
    ("Memorisation NN dist (mean)",    "memorization_nn_dist_mean",      None),
    ("Memorisation NN dist (std)",     "memorization_nn_dist_std",       None),
]


def _write_md_report(
    out_path: Path,
    *,
    fid_results: Optional[dict] = None,
    flat: Optional[dict] = None,
    warnings_list: Optional[list] = None,
    sanity_check_details_data: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Write evaluation results to *out_path* as a Markdown file.

    Sections written (each only when the relevant data is available):
      - run metadata header
      - FID table (per axis + mean, per directory)
      - Summary metrics table (generated vs real)
      - Sanity warnings
      - Figure links
    """
    def _fv2(v: float) -> str:
        return "n/a" if math.isnan(v) else f"{v:.2f}"

    def _fv4(v: float) -> str:
        return "n/a" if math.isnan(v) else f"{v:.4f}"

    lines: list[str] = ["# PoreGen Evaluation Report\n"]

    if metadata:
        lines += [
            f"- **Real dir**: `{metadata.get('real_dir', 'n/a')}`",
            f"- **Generated dir**: `{metadata.get('generated_dir', 'n/a')}`",
        ]
        if metadata.get("baseline_dir"):
            lines.append(f"- **Baseline dir**: `{metadata['baseline_dir']}`")
        lines.append(
            f"- **Volumes**: {metadata.get('n_real', '?')} real, "
            f"{metadata.get('n_gen', '?')} generated"
            + (f", {metadata['n_baseline']} baseline"
               if metadata.get("n_baseline") else "")
        )
        elapsed = metadata.get("total_elapsed_s")
        if elapsed is not None:
            lines.append(f"- **Elapsed**: {elapsed:.1f} s")
        cpv = metadata.get("crops_per_volume")
        if cpv:
            lines.append(f"- **FID crops per volume**: {cpv}")
        lines.append("")

    if fid_results:
        lines += [
            "## FID — 64×64 native crops → bilinear 299×299 → InceptionV3 pool3\n",
            "| Set | Axial | Coronal | Sagittal | Mean |",
            "|:----|------:|--------:|---------:|-----:|",
        ]
        for label, fid_d in fid_results.items():
            lines.append(
                f"| {label} "
                f"| {_fv2(fid_d.get('axial',    float('nan')))} "
                f"| {_fv2(fid_d.get('coronal',  float('nan')))} "
                f"| {_fv2(fid_d.get('sagittal', float('nan')))} "
                f"| {_fv2(fid_d.get('mean',     float('nan')))} |"
            )
        lines += [
            "",
            "> **Lower FID is better.**  "
            "Crops are 64×64 at native voxel resolution (median pore diameter ~1.79 voxels),  "
            "bilinearly upsampled to 299×299.  "
            "Full-slice resize was rejected: the ~10× downscale shrinks pores to ~0.17 px.",
            "",
            "**FID caveats (copy-paste for paper methods):** "
            "FID is computed on 64×64 crops at native voxel resolution, resized to 299×299 for "
            "InceptionV3 feature extraction. Full-slice resize to 299×299 was rejected because "
            "PoreGen volumes are ~3179×1759 voxels; the resulting ~10.6× downscale reduces the "
            "median pore diameter (1.79 voxels) to ~0.17 pixels, destroying pore-scale information "
            "before feature extraction. FID values reported here are internally consistent (all "
            "baselines evaluated with the same protocol) but are not numerically comparable to "
            "Naiff 2025, He 2024, or Pinaya 2022, which apply full-slice resize to volumes of "
            "256³ or smaller.",
            "",
        ]

    if flat:
        lines += [
            "## Summary Metrics\n",
            "| Metric | Generated | Real |",
            "|:-------|----------:|-----:|",
        ]
        for label, gk, rk in _SUMMARY_ROWS:
            gv = flat.get(gk, float("nan"))
            rv = flat.get(rk, float("nan")) if rk else float("nan")
            gs = _fv4(gv)
            rs = _fv4(rv) if rk is not None else "—"
            lines.append(f"| {label} | {gs} | {rs} |")
        lines.append("")

        lines += [
            "## Metric Comparability\n",
            "| Metric | Comparable to | Notes |",
            "|:-------|:-------------|:------|",
            "| Porosity MAE | Naiff 2025 | Identical definition |",
            "| W1-PSD | Naiff 2025 | Identical definition |",
            "| S₂(r) RMSE | Gayon-Lombardo 2020, SliceGAN 2021, Naiff 2025 | Raw curve RMSE; prior work reports curves visually |",
            "| Ripley's K | Novel for this domain | No prior porous media paper reports this |",
            "| FID | Internal only | See FID section |",
            "| Boundary inconsistency | Novel | No prior work reports this |",
            "| Diversity (φ std, PSD W1 std) | Gayon-Lombardo 2020 (mode collapse analysis) | Conceptually comparable, not numerically |",
            "",
        ]

    if sanity_check_details_data:
        lines += [
            "## Sanity Check Results\n",
            "| Check | Actual | Expected | Status |",
            "|:------|-------:|---------:|:-------|",
        ]
        for chk in sanity_check_details_data:
            act = chk["actual"]
            act_str = "n/a" if act == "n/a" else f"{act:.4f}"
            exp_str = f"{chk['expected']:.4f}"
            if chk["passed"] is None:
                status = "—"
            elif chk["passed"]:
                status = "PASS"
            else:
                status = "FAIL"
            lines.append(f"| {chk['name']} | {act_str} | {exp_str} | {status} |")
        lines.append("")

    if warnings_list:
        lines += ["## Sanity Warnings\n"]
        for w in warnings_list:
            lines.append(f"- {w}")
        lines.append("")

    # Link to figures that were written alongside this report
    fig_dir = out_path.parent / "figures"
    fig_links = [
        f"- [{title}](figures/{name})"
        for name, title in [
            ("fid_table.png",     "FID table"),
            ("s2_curves.png",     "S₂(r) curves"),
            ("psd_histogram.png", "PSD histogram"),
            ("ripley_k.png",      "Ripley K(r)"),
        ]
        if (fig_dir / name).exists()
    ]
    if fig_links:
        lines += ["## Figures\n"] + fig_links + [""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_serialisable(obj):
    """Recursively convert numpy scalars / arrays / NaN floats to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_serialisable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4 generation quality evaluation for PoreGen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real-dir",      required=True,  type=Path,
                   help="Directory with real held-out volumes (volume.tif + mask.tif pairs)")
    p.add_argument("--generated-dir", required=True,  type=Path,
                   help="Directory with generated volumes (volume.tif + mask.tif pairs)")
    p.add_argument("--baseline-dir",  default=None,   type=Path,
                   help="Optional baseline volume directory (same format)")
    p.add_argument("--out-dir",       default="eval_results", type=Path,
                   help="Output directory for eval_results.json and figures/")

    # Optional memorisation check
    p.add_argument("--vae-run",       default=None,   type=Path,
                   help="VAE run dir with resolved_config.yaml + checkpoints/best.ckpt "
                        "(enables memorisation check)")
    p.add_argument("--latents-dir",   default="data/split_v2/latents_s64_sampled", type=Path,
                   help="Training latent directory (latents.bin + latents_meta.json)")
    p.add_argument("--n-train-sample", type=int, default=10_000,
                   help="Number of training latents to sample for memorisation")

    # Metric controls
    p.add_argument("--r-max",          type=int, default=50,
                   help="Maximum radius (voxels) for S2(r) and Ripley K. "
                        "Safe limit for S2 is s2_crop × 0.4 (default 128 × 0.4 = 51 voxels; "
                        "beyond this, FFT edge effects dominate)")
    p.add_argument("--stride",         type=int, default=32,
                   help="Patch stride for boundary inconsistency (must match LDM training stride)")
    p.add_argument("--s2-crop",        type=int, default=128,
                   help="Crop size for S2(r) FFT (clamped to volume dims; avoids OOM)")
    p.add_argument("--n-s2-crops",     type=int, default=3,
                   help="Random crops to average S2(r) over per volume")
    p.add_argument("--crops-per-volume", type=int, default=500,
                   help="Random 64×64 2D crops per volume for FID (split ~evenly across 3 axes). "
                        "Total crops = n_volumes × crops_per_volume; must be ≥5000 per set.")
    p.add_argument("--fid", action="store_true",
                   help="FID-only mode: skip structural metrics, compute FID and exit. "
                        "Useful when you only need FID without the full eval suite.")
    p.add_argument("--max-pores-ripley", type=int, default=5000,
                   help="Subsample pores to this count for Ripley K (O(N²)); a warning is logged when subsampling occurs")
    p.add_argument("--ripley-per-patch", action="store_true",
                   help="Compute Ripley K on individual 64³ patches (non-overlapping) rather than the full "
                        "assembled volume; auto-enabled when estimated pore count > 50 000 "
                        "(avoids O(N²) scaling on millions of pore centres)")
    p.add_argument("--max-pores-morph", type=int, default=2000,
                   help="Largest N pores (by volume) to include in morphology stats")

    # Skip flags
    p.add_argument("--mask-name",      default="only_pores.tif",
                   help="Filename to use as pore mask inside each sample dir "
                        "(only_pores.tif = Sauvola-segmented from XCT; mask.tif = VAE decoder output)")
    p.add_argument("--skip-fid",       action="store_true", help="Skip FID (needs torchvision)")
    p.add_argument("--skip-ripley",    action="store_true", help="Skip Ripley K computation")
    p.add_argument("--skip-morphology", action="store_true", help="Skip morphology (slow for large volumes)")

    p.add_argument("--device",  default="cuda",  help="PyTorch device for FID and memorisation")
    p.add_argument("--seed",    type=int, default=42)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    t_global = time.perf_counter()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ── FID-only mode ───────────────────────────────────────────────────────
    if args.fid:
        logger.info("=== FID-only mode (--fid) ===")
        real_pairs = discover_pairs(args.real_dir, mask_name=args.mask_name)
        gen_pairs  = discover_pairs(args.generated_dir, mask_name=args.mask_name)
        logger.info("Real: %d  |  Generated: %d", len(real_pairs), len(gen_pairs))
        if not real_pairs:
            logger.error("No real volumes found in %s", args.real_dir)
            sys.exit(1)
        if not gen_pairs:
            logger.error("No generated volumes found in %s", args.generated_dir)
            sys.exit(1)

        def _load_xct(pairs: list[tuple[Path, Path]]) -> list[np.ndarray]:
            vols = []
            for xct_p, _ in pairs:
                raw = tifffile.imread(str(xct_p))
                arr = (raw.astype(np.float32) / 255.0
                       if raw.dtype == np.uint8 else raw.astype(np.float32))
                vols.append(np.clip(arr, 0.0, 1.0))
            return vols

        real_xct = _load_xct(real_pairs)
        gen_xct  = _load_xct(gen_pairs)

        try:
            fid_gen = compute_fid_crop_based(
                real_xct, gen_xct, args.crops_per_volume, args.device
            )
        except ValueError as exc:
            logger.error("FID error: %s", exc)
            sys.exit(1)

        fid_all: dict[str, dict] = {"Generated": fid_gen}

        if args.baseline_dir:
            base_pairs = discover_pairs(args.baseline_dir, mask_name=args.mask_name)
            if base_pairs:
                base_xct = _load_xct(base_pairs)
                try:
                    fid_all["Baseline"] = compute_fid_crop_based(
                        real_xct, base_xct, args.crops_per_volume, args.device
                    )
                except ValueError as exc:
                    logger.warning("Baseline FID error: %s", exc)

        if fid_all:
            _plot_fid_table(fid_all, figures_dir / "fid_table.png")

        fid_meta = {
            "real_dir":         str(args.real_dir),
            "generated_dir":    str(args.generated_dir),
            "baseline_dir":     str(args.baseline_dir) if args.baseline_dir else None,
            "n_real":           len(real_pairs),
            "n_gen":            len(gen_pairs),
            "crops_per_volume": args.crops_per_volume,
            "fid_mode":         True,
            "total_elapsed_s":  round(time.perf_counter() - t_global, 1),
        }
        output_fid = {"fid": _to_serialisable(fid_all), "metadata": fid_meta}
        json_path = out_dir / "eval_results.json"
        with open(json_path, "w") as f:
            json.dump(output_fid, f, indent=2)
        logger.info("Saved → %s", json_path)

        _write_md_report(
            out_dir / "eval_report.md",
            fid_results=fid_all,
            metadata=fid_meta,
        )
        return

    # ── Discover volumes ────────────────────────────────────────────────────
    logger.info("Discovering volumes (mask file: %s) …", args.mask_name)
    real_pairs = discover_pairs(args.real_dir,       mask_name=args.mask_name)
    gen_pairs  = discover_pairs(args.generated_dir,  mask_name=args.mask_name)
    logger.info("  Real: %d  |  Generated: %d", len(real_pairs), len(gen_pairs))

    if not real_pairs:
        logger.error(
            "No real volumes found in %s — check for volume.tif/xct.tif + %s (or mask.tif)",
            args.real_dir, args.mask_name,
        )
        sys.exit(1)
    if not gen_pairs:
        logger.error("No generated volumes found in %s", args.generated_dir)
        sys.exit(1)

    baseline_pairs: list[tuple[Path, Path]] = []
    if args.baseline_dir:
        baseline_pairs = discover_pairs(args.baseline_dir, mask_name=args.mask_name)
        logger.info("  Baseline: %d", len(baseline_pairs))

    # ── Common kwargs for per-volume metrics ───────────────────────────────
    vol_kwargs = dict(
        r_max=args.r_max,
        n_bins=args.r_max,
        stride=args.stride,
        s2_crop_size=args.s2_crop,
        n_s2_crops=args.n_s2_crops,
        run_ripley=not args.skip_ripley,
        run_morphology=not args.skip_morphology,
        max_pores_ripley=args.max_pores_ripley,
        max_pores_morph=args.max_pores_morph,
        ripley_per_patch=args.ripley_per_patch,
    )

    # ── Per-volume structural metrics ───────────────────────────────────────
    logger.info("Computing structural metrics for REAL volumes …")
    real_metrics = _process_all_pairs(real_pairs, "real", **vol_kwargs)

    logger.info("Computing structural metrics for GENERATED volumes …")
    gen_metrics  = _process_all_pairs(gen_pairs,  "gen",  **vol_kwargs)

    baseline_metrics: list[dict] = []
    if baseline_pairs:
        logger.info("Computing structural metrics for BASELINE volumes …")
        baseline_metrics = _process_all_pairs(baseline_pairs, "baseline", **vol_kwargs)

    # ── Aggregate comparison ────────────────────────────────────────────────
    logger.info("Aggregating gen vs real …")
    comparison = aggregate_comparison(real_metrics, gen_metrics)
    diversity  = compute_diversity(gen_metrics, real_metrics)

    baseline_comparison: dict = {}
    if baseline_metrics:
        logger.info("Aggregating baseline vs real …")
        baseline_comparison = aggregate_comparison(real_metrics, baseline_metrics)

    # ── FID ─────────────────────────────────────────────────────────────────
    fid_all: dict[str, dict] = {}
    fid_gen_result: dict[str, float] = {
        "axial": float("nan"), "coronal": float("nan"),
        "sagittal": float("nan"), "mean": float("nan"),
    }

    if not args.skip_fid:
        logger.info("Computing FID on 2D slices (64×64 crops → 299×299) …")

        def _load_xct_vols(pairs: list[tuple[Path, Path]]) -> list[np.ndarray]:
            vols = []
            for xct_p, _ in pairs:
                raw = tifffile.imread(str(xct_p))
                arr = raw.astype(np.float32) / 255.0 if raw.dtype == np.uint8 else raw.astype(np.float32)
                vols.append(np.clip(arr, 0.0, 1.0))
            return vols

        real_xct_vols = _load_xct_vols(real_pairs)
        gen_xct_vols  = _load_xct_vols(gen_pairs)

        try:
            fid_gen_result = compute_fid_crop_based(
                real_xct_vols, gen_xct_vols, args.crops_per_volume, args.device
            )
        except ValueError as exc:
            logger.warning("FID skipped: %s", exc)
            fid_gen_result = {"axial": float("nan"), "coronal": float("nan"),
                              "sagittal": float("nan"), "mean": float("nan")}
        fid_all["Generated"] = fid_gen_result

        if baseline_pairs:
            baseline_xct_vols = _load_xct_vols(baseline_pairs)
            try:
                fid_all["Baseline"] = compute_fid_crop_based(
                    real_xct_vols, baseline_xct_vols, args.crops_per_volume, args.device
                )
            except ValueError as exc:
                logger.warning("Baseline FID skipped: %s", exc)
    else:
        logger.info("FID skipped (--skip-fid)")

    # ── Memorisation ────────────────────────────────────────────────────────
    memorization: dict = {}
    if args.vae_run is not None:
        latents_dir = Path(args.latents_dir)
        if latents_dir.exists():
            logger.info("Computing memorisation score …")
            memorization = compute_memorization(
                gen_pairs, args.vae_run, latents_dir,
                device_str=args.device,
                n_train_sample=args.n_train_sample,
            )
        else:
            logger.warning("Latents dir %s not found — skipping memorisation", latents_dir)
    else:
        logger.info("--vae-run not set — memorisation check skipped")

    # ── Flat result dict for table / sanity ────────────────────────────────
    flat: dict = {
        **comparison,
        **diversity,
        **memorization,
        "fid_axial":    fid_gen_result.get("axial",    float("nan")),
        "fid_coronal":  fid_gen_result.get("coronal",  float("nan")),
        "fid_sagittal": fid_gen_result.get("sagittal", float("nan")),
        "fid_mean":     fid_gen_result.get("mean",     float("nan")),
    }

    # ── Sanity checks ───────────────────────────────────────────────────────
    sanity_details = sanity_check_details(flat)
    warnings_list = sanity_check(flat, label="generated")
    if baseline_comparison:
        warnings_list += sanity_check(
            {k.replace("_gen", "_gen"): baseline_comparison.get(k.replace("_gen", "_gen"), float("nan"))
             for k in flat}, label="baseline"
        )
    for w in warnings_list:
        logger.warning(w)

    # ── Save JSON ───────────────────────────────────────────────────────────
    output = {
        "comparison_gen_vs_real":      _to_serialisable(comparison),
        "diversity":                   _to_serialisable(diversity),
        "fid":                         _to_serialisable(fid_all),
        "memorization":                _to_serialisable(memorization),
        "baseline_comparison_vs_real": _to_serialisable(baseline_comparison),
        "sanity_warnings":             warnings_list,
        "per_volume_real":             _to_serialisable(real_metrics),
        "per_volume_gen":              _to_serialisable(gen_metrics),
        "per_volume_baseline":         _to_serialisable(baseline_metrics),
        "metadata": {
            "real_dir":       str(args.real_dir),
            "generated_dir":  str(args.generated_dir),
            "baseline_dir":   str(args.baseline_dir) if args.baseline_dir else None,
            "n_real":         len(real_metrics),
            "n_gen":          len(gen_metrics),
            "n_baseline":     len(baseline_metrics),
            "total_elapsed_s": round(time.perf_counter() - t_global, 1),
        },
    }

    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved → %s", json_path)

    # ── Figures ─────────────────────────────────────────────────────────────
    logger.info("Generating figures …")

    # S₂(r) curves
    if "s2_curve_real" in comparison and "s2_curve_gen" in comparison:
        s2_curves: dict[str, list] = {"Real": comparison["s2_curve_real"],
                                       "Generated": comparison["s2_curve_gen"]}
        if "s2_curve_gen" in baseline_comparison:
            s2_curves["Baseline"] = baseline_comparison["s2_curve_gen"]
        _plot_s2_multi(comparison["s2_r_vals"], s2_curves, figures_dir / "s2_curves.png")

    # PSD histograms
    diam_real = comparison.get("_diam_real", np.array([]))
    diam_gen  = comparison.get("_diam_gen",  np.array([]))
    if len(diam_real) > 0 or len(diam_gen) > 0:
        diam_sets: dict[str, np.ndarray] = {"Real": diam_real, "Generated": diam_gen}
        if baseline_metrics:
            diam_base = (np.concatenate([m["psd_diameters"] for m in baseline_metrics if m["psd_diameters"]])
                         if any(m["psd_diameters"] for m in baseline_metrics) else np.array([]))
            if len(diam_base) > 0:
                diam_sets["Baseline"] = diam_base
        _plot_psd_multi(diam_sets, figures_dir / "psd_histogram.png")

    # Ripley K
    if "ripley_r_vals" in comparison:
        K_curves: dict[str, list] = {"Real": comparison["ripley_K_real"],
                                      "Generated": comparison["ripley_K_gen"]}
        if "ripley_K_gen" in baseline_comparison:
            K_curves["Baseline"] = baseline_comparison["ripley_K_gen"]
        _plot_ripley_multi(comparison["ripley_r_vals"], K_curves, figures_dir / "ripley_k.png")

    # FID table
    if fid_all:
        _plot_fid_table(fid_all, figures_dir / "fid_table.png")

    total = time.perf_counter() - t_global

    # ── Markdown report ─────────────────────────────────────────────────────
    _write_md_report(
        out_dir / "eval_report.md",
        fid_results=fid_all if fid_all else None,
        flat=flat,
        warnings_list=warnings_list if warnings_list else None,
        sanity_check_details_data=sanity_details,
        metadata={
            "real_dir":        str(args.real_dir),
            "generated_dir":   str(args.generated_dir),
            "baseline_dir":    str(args.baseline_dir) if args.baseline_dir else None,
            "n_real":          len(real_metrics),
            "n_gen":           len(gen_metrics),
            "n_baseline":      len(baseline_metrics),
            "crops_per_volume": args.crops_per_volume,
            "total_elapsed_s": round(total, 1),
        },
    )

    logger.info(
        "EVAL COMPLETE  n_real=%d  n_gen=%d  total=%.1fs",
        len(real_metrics), len(gen_metrics), total,
    )
    if warnings_list:
        logger.warning("Sanity check warnings (%d):", len(warnings_list))
        for w in warnings_list:
            logger.warning("  %s", w)
    logger.info("Results → %s", json_path)
    logger.info("Figures → %s/", figures_dir)


if __name__ == "__main__":
    main()

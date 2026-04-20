"""Post-training evaluation script for R03 VAE (v2.conv_noattn, z_channels=16).

Sections
--------
1. Full-volume reconstruction from original TIFFs (1–3 test volumes)
   – per-volume metrics: XCT quality, mask quality, S2(r), PSD
   – slice-comparison PNGs (axial / coronal / sagittal)
   – S2(r) and PSD plots
2. 3-D pore comparison (small / medium / large) with rotating GIFs
3. Quantitative metrics over the full test set (patches from Zarr)
4. Latent space audit (KL per channel, active channels, μ/σ histograms)
5. Prior sample decode (8 samples, 2×4 grid)

Usage::

    python scripts/eval_r03.py \\
        --checkpoint runs/vae/r03-run-.../best.ckpt \\
        --data_zarr  data/split_v2 \\
        --data_tiff  raw_data \\
        [--n_volumes 3] [--out_dir results/r03_eval] [--gpu 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy import ndimage
from scipy.stats import wasserstein_distance
from skimage.measure import label as sk_label, regionprops, marching_cubes
from skimage.metrics import structural_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

import tifffile

# ── Poregen imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from poregen.models.vae import build_vae
from poregen.training.checkpoint import load_checkpoint
from poregen.dataset.loader import PatchDataset, zarr_worker_init_fn

logger = logging.getLogger(__name__)

PATCH_SIZE = 64
Z_CHANNELS = 16
SPATIAL_LATENT = PATCH_SIZE // 4  # 16


# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — model loading and TIFF utilities
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(device: torch.device) -> torch.nn.Module:
    """Build the R03 model with hardcoded architecture (in_channels=1, z=16, c=32)."""
    model = build_vae(
        "v2.conv_noattn",
        in_channels=1,
        z_channels=Z_CHANNELS,
        base_channels=32,
        n_blocks=2,
        patch_size=PATCH_SIZE,
    ).to(device)
    return model


def tiff_path_for_volume(volume_id: str, tiff_root: Path) -> Path:
    """Map a volume_id → absolute TIFF path.

    volume_id convention: ``<source_group>__<relative_stem>``
    e.g. ``MedidasDB__Airbus_Panel_Pegaso_probetas_1_8_volume_eq_aligned``
    → ``<tiff_root>/MedidasDB/Airbus_Panel_Pegaso_probetas_1_8_volume_eq_aligned.tif``
    """
    rel = volume_id.replace("__", "/")
    for ext in (".tif", ".tiff"):
        p = tiff_root / (rel + ext)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"TIFF not found for volume_id={volume_id!r} "
        f"(tried {tiff_root / rel}.tif/.tiff)"
    )


def short_name(volume_id: str) -> str:
    """Strip the MedidasDB__ prefix for use in file names."""
    return volume_id.replace("MedidasDB__", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — encode/decode with μ (no sampling)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_decode_mu(
    model: torch.nn.Module,
    xct_patch: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode a single 64³ XCT patch → decode using μ (no sampling).

    Returns
    -------
    xct_recon  : float32 [0,1], shape (64,64,64)
    mask_prob  : float32 [0,1], shape (64,64,64)  — sigmoid(mask_logits)
    mu         : float32, shape (16,16,16,16) — latent mean
    logvar     : float32, shape (16,16,16,16) — latent log-variance
    """
    xct_f = xct_patch.astype(np.float32) / 255.0
    xct_t = torch.from_numpy(xct_f).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64,64)

    h      = model.encoder(xct_t)
    mu_t   = model.to_mu(h)
    logvar_t = model.to_logvar(h)
    dec    = model.decoder(mu_t)                          # use μ directly, no sampling

    xct_out  = model.xct_head(dec).clamp(0.0, 1.0)       # [0,1] reconstruction
    mask_out = torch.sigmoid(model.mask_head(dec))        # [0,1] probability

    return (
        xct_out  .squeeze().cpu().numpy().astype(np.float32),
        mask_out .squeeze().cpu().numpy().astype(np.float32),
        mu_t     .squeeze().cpu().numpy().astype(np.float32),
        logvar_t .squeeze().cpu().numpy().astype(np.float32),
    )


def pad_to_multiple(arr: np.ndarray, multiple: int = 64) -> tuple[np.ndarray, tuple]:
    """Zero-pad arr to the next multiple of *multiple* in each dimension.

    Returns (padded_arr, pad_widths) so the caller can undo the padding.
    """
    D, H, W = arr.shape
    pd = (multiple - D % multiple) % multiple
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    pads = ((0, pd), (0, ph), (0, pw))
    if pd == 0 and ph == 0 and pw == 0:
        return arr, pads
    padded = np.pad(arr, pads, mode="reflect")
    return padded, pads


def reconstruct_volume_from_tiff(
    model: torch.nn.Module,
    volume_id: str,
    tiff_root: Path,
    zarr_root: zarr.Group,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-volume reconstruction by tiling 64³ patches from the original TIFF.

    XCT is loaded from the TIFF file (the authoritative source).
    GT mask is loaded from the Zarr (computed via Sauvola segmentation at
    dataset-build time; no mask TIFFs exist in the raw-data directory).

    Returns
    -------
    xct_gt    : float32 [0,1] — original XCT normalised, cropped to tiled region
    mask_gt   : float32 {0,1} — GT pore mask, cropped to tiled region
    xct_recon : float32 [0,1] — VAE reconstruction, same shape
    mask_pred : float32 [0,1] — predicted mask sigmoid probabilities, same shape
    """
    tiff_p = tiff_path_for_volume(volume_id, tiff_root)
    logger.info("  Loading XCT TIFF: %s", tiff_p)
    xct_raw = tifffile.imread(str(tiff_p))  # uint8, (D, H, W)
    if xct_raw.ndim != 3:
        raise ValueError(f"Expected 3-D TIFF, got shape {xct_raw.shape}")
    xct_raw = xct_raw.astype(np.uint8, copy=False)

    # GT mask from Zarr (uint8 {0,1})
    grp = zarr_root[volume_id]
    mask_raw = np.array(grp["mask"]).astype(np.uint8, copy=False)

    D, H, W = xct_raw.shape
    logger.info("  Volume shape: %s", (D, H, W))

    # Tile dimensions (drop remainder — no padding to match eval_checkpoint.py convention)
    dz = (D // PATCH_SIZE) * PATCH_SIZE
    dy = (H // PATCH_SIZE) * PATCH_SIZE
    dx = (W // PATCH_SIZE) * PATCH_SIZE

    xct_recon  = np.zeros((dz, dy, dx), dtype=np.float32)
    mask_pred  = np.zeros((dz, dy, dx), dtype=np.float32)

    model.eval()
    n_patches = (dz // PATCH_SIZE) * (dy // PATCH_SIZE) * (dx // PATCH_SIZE)
    done = 0
    for z0 in range(0, dz, PATCH_SIZE):
        for y0 in range(0, dy, PATCH_SIZE):
            for x0 in range(0, dx, PATCH_SIZE):
                patch = xct_raw[z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                xr, mp, _, _ = encode_decode_mu(model, patch, device)
                xct_recon [z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = xr
                mask_pred [z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = mp
                done += 1
        if done % max(1, n_patches // 10) == 0:
            logger.info("    %d/%d patches …", done, n_patches)

    xct_gt  = xct_raw [:dz, :dy, :dx].astype(np.float32) / 255.0
    mask_gt = mask_raw[:dz, :dy, :dx].astype(np.float32)

    return xct_gt, mask_gt, xct_recon, mask_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — per-volume metrics
# ═══════════════════════════════════════════════════════════════════════════════

def psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def boundary_mae(vol: np.ndarray, patch_size: int = PATCH_SIZE) -> float:
    """Mean absolute voxel discontinuity at patch seams."""
    D, H, W = vol.shape
    diffs: list[float] = []
    for z in range(patch_size, D, patch_size):
        diffs.append(float(np.abs(vol[z - 1] - vol[z]).mean()))
    for y in range(patch_size, H, patch_size):
        diffs.append(float(np.abs(vol[:, y - 1, :] - vol[:, y, :]).mean()))
    for x in range(patch_size, W, patch_size):
        diffs.append(float(np.abs(vol[:, :, x - 1] - vol[:, :, x]).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def s2_radial(binary: np.ndarray, r_max: int = 50, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic two-point correlation function via FFT autocorrelation."""
    vol = binary.astype(np.float64)
    fft = np.fft.fftn(vol)
    autocorr = np.real(np.fft.ifftn(fft * np.conj(fft))) / vol.size

    D, H, W = binary.shape
    dz = np.fft.fftfreq(D) * D
    dy = np.fft.fftfreq(H) * H
    dx = np.fft.fftfreq(W) * W
    ZZ, YY, XX = np.meshgrid(dz, dy, dx, indexing="ij")
    R = np.sqrt(ZZ**2 + YY**2 + XX**2)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    s2 = np.zeros(n_bins)
    for i in range(n_bins):
        m = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if m.any():
            s2[i] = autocorr[m].mean()

    r_vals = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_vals, s2


def s2_wasserstein(s2_gt: np.ndarray, s2_pred: np.ndarray) -> float:
    eps = 1e-12
    a = np.clip(s2_gt,   0, None); a = a / (a.sum() + eps)
    b = np.clip(s2_pred, 0, None); b = b / (b.sum() + eps)
    r = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(r, r, a, b))


def pore_sizes(binary: np.ndarray) -> np.ndarray:
    """Connected-component sizes (voxels) for every pore."""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.array([], dtype=np.float64)
    sizes = np.array(ndimage.sum(binary, labeled, range(1, n + 1)), dtype=np.float64)
    return sizes


def dice_precision_recall(
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
) -> dict[str, float]:
    tp = int((gt_bin & pred_bin).sum())
    fp = int((~gt_bin & pred_bin).sum())
    fn = int((gt_bin & ~pred_bin).sum())
    denom_dice   = 2 * tp + fp + fn
    denom_prec   = tp + fp
    denom_recall = tp + fn
    denom_iou    = tp + fp + fn
    return {
        "dice":      2.0 * tp / denom_dice   if denom_dice   > 0 else 0.0,
        "precision": float(tp) / denom_prec  if denom_prec   > 0 else 0.0,
        "recall":    float(tp) / denom_recall if denom_recall > 0 else 0.0,
        "iou":       float(tp) / denom_iou   if denom_iou    > 0 else 0.0,
    }


def ripleys_k_3d(
    binary: np.ndarray,
    r_max: int = 30,
    max_pores: int = 500,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ripley's K(r) on pore centroid point cloud.

    Returns None if fewer than 3 pores are found (K is not meaningful).
    """
    labeled = sk_label(binary)
    props   = regionprops(labeled)
    if len(props) < 3:
        return None

    # Subsample to avoid O(N²) blowup
    if len(props) > max_pores:
        import random
        props = random.sample(props, max_pores)

    centroids = np.array([p.centroid for p in props], dtype=np.float64)  # (N,3)
    N  = len(centroids)
    D, H, W = binary.shape
    V  = float(D * H * W)

    # pairwise distances: cheap for N ≤ 500
    from scipy.spatial.distance import pdist
    dists = pdist(centroids)

    r_vals = np.arange(1, r_max + 1, dtype=np.float64)
    K = np.array([(V / N ** 2) * float(np.sum(dists < r)) for r in r_vals])
    return r_vals, K


def compute_volume_metrics(
    volume_id: str,
    xct_gt: np.ndarray,
    mask_gt: np.ndarray,
    xct_recon: np.ndarray,
    mask_pred: np.ndarray,
    r_max: int = 50,
    run_ripley: bool = True,
) -> dict:
    """Compute all per-volume quantitative metrics."""
    bin_gt   = (mask_gt   >= 0.5).astype(bool)
    bin_pred = (mask_pred >= 0.5).astype(bool)

    # XCT quality
    mae_xct  = float(np.abs(xct_recon - xct_gt).mean())
    psnr_val = psnr(xct_gt, xct_recon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssim_val = float(structural_similarity(
            xct_gt, xct_recon, data_range=1.0, win_size=7,
        ))
    bnd_mae = boundary_mae(xct_recon)

    # Mask / porosity
    por_gt   = float(bin_gt.mean())
    por_pred = float(bin_pred.mean())
    por_mae  = abs(por_pred - por_gt)

    seg_metrics = dice_precision_recall(bin_gt, bin_pred)

    # Connected components
    cc_gt   = int(ndimage.label(bin_gt)[1])
    cc_pred = int(ndimage.label(bin_pred)[1])

    # S2(r)
    r_vals, s2_gt   = s2_radial(bin_gt,   r_max=r_max)
    _,      s2_pred = s2_radial(bin_pred, r_max=r_max)
    s2_w1 = s2_wasserstein(s2_gt, s2_pred)

    # PSD
    psd_gt   = pore_sizes(bin_gt)
    psd_pred = pore_sizes(bin_pred)
    if len(psd_gt) > 0 and len(psd_pred) > 0:
        psd_w1 = float(wasserstein_distance(psd_gt, psd_pred))
    else:
        psd_w1 = float("nan")

    # Ripley's K
    ripley_w1 = None
    if run_ripley:
        kr_gt   = ripleys_k_3d(bin_gt)
        kr_pred = ripleys_k_3d(bin_pred)
        if kr_gt is not None and kr_pred is not None:
            ripley_w1 = float(wasserstein_distance(kr_gt[1], kr_pred[1]))

    result: dict = {
        "volume_id":        volume_id,
        "xct_mae":          mae_xct,
        "xct_psnr":         psnr_val,
        "xct_ssim":         ssim_val,
        "xct_boundary_mae": bnd_mae,
        "porosity_gt":      por_gt,
        "porosity_pred":    por_pred,
        "porosity_mae":     por_mae,
        **seg_metrics,
        "pore_count_gt":    cc_gt,
        "pore_count_pred":  cc_pred,
        "s2_wasserstein":   s2_w1,
        "s2_r_vals":        r_vals.tolist(),
        "s2_gt":            s2_gt.tolist(),
        "s2_pred":          s2_pred.tolist(),
        "psd_wasserstein":  psd_w1,
        "psd_gt":           psd_gt.tolist(),
        "psd_pred":         psd_pred.tolist(),
    }
    if ripley_w1 is not None:
        result["ripley_w1"] = ripley_w1
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — volume-level visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def _axial_comparison_grid(
    xct_gt: np.ndarray,
    xct_recon: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    slice_indices: list[int],
    axis: int,
    title: str,
) -> plt.Figure:
    """3-row × 5-col figure: GT XCT | Recon XCT | GT mask | Pred mask | |diff|."""
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(n_slices, 5, figsize=(18, 4 * n_slices), squeeze=False)
    fig.suptitle(title, fontsize=12)

    col_titles = ["GT XCT", "Recon XCT", "GT mask", "Pred mask", "|GT−Recon|"]

    def _take(vol, idx):
        if axis == 0:
            return vol[idx]
        elif axis == 1:
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    for row, sl_idx in enumerate(slice_indices):
        slices = [
            _take(xct_gt,   sl_idx),
            _take(xct_recon, sl_idx),
            _take(mask_gt,   sl_idx),
            _take(mask_pred, sl_idx),
            np.abs(_take(xct_gt, sl_idx) - _take(xct_recon, sl_idx)),
        ]
        cmaps = ["gray", "gray", "binary", "binary", "hot"]
        for col, (img, cmap) in enumerate(zip(slices, cmaps)):
            ax = axes[row][col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
        axes[row][0].set_ylabel(f"slice {sl_idx}", fontsize=8)

    plt.tight_layout()
    return fig


def save_slice_comparisons(
    xct_gt: np.ndarray,
    xct_recon: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    vol_name: str,
    out_dir: Path,
) -> None:
    D, H, W = xct_gt.shape

    for axis, axis_name, dim in [
        (0, "axial",    D),
        (1, "coronal",  H),
        (2, "sagittal", W),
    ]:
        idxs = [dim // 4, dim // 2, 3 * dim // 4]
        fig = _axial_comparison_grid(
            xct_gt, xct_recon, mask_gt, mask_pred, idxs, axis,
            title=f"{vol_name} — {axis_name}",
        )
        out = out_dir / f"{vol_name}_slices_{axis_name}.png"
        fig.savefig(str(out), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", out)


def save_s2r_plot(metrics: dict, vol_name: str, out_dir: Path) -> None:
    r  = np.array(metrics["s2_r_vals"])
    gt = np.array(metrics["s2_gt"])
    pr = np.array(metrics["s2_pred"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, gt, "b-",  lw=1.5, label="GT")
    ax.plot(r, pr, "r--", lw=1.5, label="Predicted")
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("S₂(r)")
    ax.set_title(f"{vol_name}  S₂(r) — W₁={metrics['s2_wasserstein']:.4f}")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"{vol_name}_s2r.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


def save_psd_plot(metrics: dict, vol_name: str, out_dir: Path) -> None:
    psd_gt   = np.array(metrics["psd_gt"])
    psd_pred = np.array(metrics["psd_pred"])

    # Equivalent diameter from volume: d = 2 * (3V/4π)^(1/3)
    def vol_to_diam(v):
        return 2.0 * (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(0.5), np.log10(max(psd_gt.max() if len(psd_gt) else 100,
                                                    psd_pred.max() if len(psd_pred) else 100) + 1), 40)
    if len(psd_gt) > 0:
        diams_gt = vol_to_diam(psd_gt)
        ax.hist(diams_gt,   bins=bins, alpha=0.5, color="blue",   label=f"GT (n={len(psd_gt)})",    density=True)
    if len(psd_pred) > 0:
        diams_pred = vol_to_diam(psd_pred)
        ax.hist(diams_pred, bins=bins, alpha=0.5, color="red",    label=f"Pred (n={len(psd_pred)})", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Equivalent diameter (voxels)")
    ax.set_ylabel("Density")
    psd_w1_str = f"{metrics['psd_wasserstein']:.2f}" if not math.isnan(metrics.get("psd_wasserstein", float("nan"))) else "n/a"
    ax.set_title(f"{vol_name}  PSD — W₁={psd_w1_str}")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"{vol_name}_psd.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


def save_output_tiffs(
    xct_recon: np.ndarray,
    mask_pred: np.ndarray,
    vol_name: str,
    out_dir: Path,
) -> None:
    """Save reconstructed XCT (float32) and binary predicted mask (uint8 0/255)."""
    tifffile.imwrite(str(out_dir / f"{vol_name}_recon_xct.tiff"),  xct_recon)
    mask_u8 = (mask_pred >= 0.5).astype(np.uint8) * 255
    tifffile.imwrite(str(out_dir / f"{vol_name}_pred_mask.tiff"), mask_u8)
    logger.info("  Saved TIFFs for %s", vol_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — 3-D pore comparison (scatter / marching cubes + GIF)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_pore_pair(
    gt_region: np.ndarray,
    pred_region: np.ndarray,
    ax_gt: plt.Axes,
    ax_pred: plt.Axes,
    elev: float = 25,
    azim: float = 45,
) -> None:
    """Render GT and predicted pore voxels into a pair of 3-D axes."""
    for ax, region, color, label in [
        (ax_gt,   gt_region,   (0.2, 0.4, 0.9, 0.7), "GT"),
        (ax_pred, pred_region, (0.9, 0.4, 0.2, 0.7), "Predicted"),
    ]:
        ax.clear()
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7); ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        if region.sum() < 1:
            ax.text(0.5, 0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            return

        try:
            verts, faces, _, _ = marching_cubes(region.astype(float), level=0.5, allow_degenerate=False)
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh = Poly3DCollection(verts[faces], alpha=0.5, facecolor=color[:3], edgecolor="none")
            ax.add_collection3d(mesh)
            ax.set_xlim(0, region.shape[2])
            ax.set_ylim(0, region.shape[1])
            ax.set_zlim(0, region.shape[0])
        except Exception:
            # Fallback to voxel scatter if marching cubes fails
            zz, yy, xx = np.where(region)
            ax.scatter(xx, yy, zz, c=[color], s=4, depthshade=True)


def save_pore_3d(
    mask_gt_full: np.ndarray,
    mask_pred_full: np.ndarray,
    label: str,    # "small" / "medium" / "large"
    vol_name: str,
    out_dir: Path,
) -> None:
    """Select a pore of the given size category and create comparison plots + GIF."""
    bin_gt   = (mask_gt_full   >= 0.5).astype(bool)
    labeled  = sk_label(bin_gt)
    props    = regionprops(labeled)
    if not props:
        logger.warning("  No GT pores found for %s — skipping 3-D %s pore", vol_name, label)
        return

    # Sort by equivalent diameter
    diams = np.array([p.equivalent_diameter_area for p in props])
    sorted_idx = np.argsort(diams)

    target_prop = None
    if label == "small":
        candidates = [i for i in sorted_idx if 2.0 <= diams[i] <= 4.0]
        if candidates:
            target_prop = props[candidates[len(candidates) // 2]]
        else:
            target_prop = props[sorted_idx[0]]          # fallback: smallest
    elif label == "medium":
        candidates = [i for i in sorted_idx if 6.0 <= diams[i] <= 12.0]
        if candidates:
            target_prop = props[candidates[len(candidates) // 2]]
        elif len(sorted_idx) > 1:
            target_prop = props[sorted_idx[len(sorted_idx) // 2]]
    else:  # "large"
        candidates = [i for i in sorted_idx if diams[i] > 15.0]
        if candidates:
            target_prop = props[candidates[len(candidates) // 2]]
        elif sorted_idx.size > 0:
            target_prop = props[sorted_idx[-1]]

    if target_prop is None:
        logger.warning("  No suitable pore for %s / %s — skipping", vol_name, label)
        return

    # Bounding box + 4 voxel padding
    pad = 4
    d_min, h_min, w_min, d_max, h_max, w_max = target_prop.bbox
    D, H, W = mask_gt_full.shape
    d0 = max(0, d_min - pad);  d1 = min(D, d_max + pad)
    h0 = max(0, h_min - pad);  h1 = min(H, h_max + pad)
    w0 = max(0, w_min - pad);  w1 = min(W, w_max + pad)

    gt_region   = bin_gt                               [d0:d1, h0:h1, w0:w1]
    pred_region = (mask_pred_full >= 0.5).astype(bool) [d0:d1, h0:h1, w0:w1]

    # ── Static 6-angle PNG ──────────────────────────────────────────────────
    angles = [(25, 0), (25, 90), (25, 180), (25, 270), (80, 45), (-10, 45)]
    fig = plt.figure(figsize=(14, 10))
    for i, (elev, azim) in enumerate(angles):
        ax_gt   = fig.add_subplot(3, 4, 2*i + 1, projection="3d")
        ax_pred = fig.add_subplot(3, 4, 2*i + 2, projection="3d")
        _render_pore_pair(gt_region, pred_region, ax_gt, ax_pred, elev=elev, azim=azim)
    fig.suptitle(f"{vol_name}  {label} pore  d≈{target_prop.equivalent_diameter_area:.1f}vx", fontsize=10)
    fig.tight_layout()
    png_path = out_dir / f"{vol_name}_pore_{label}.png"
    fig.savefig(str(png_path), dpi=80)
    plt.close(fig)
    logger.info("  Saved %s", png_path)

    # ── Rotating GIF (36 frames, 360°) ──────────────────────────────────────
    fig_gif, (ax_gt_g, ax_pred_g) = plt.subplots(
        1, 2, figsize=(8, 4),
        subplot_kw={"projection": "3d"},
    )
    fig_gif.suptitle(f"{vol_name}  {label} pore (d≈{target_prop.equivalent_diameter_area:.1f}vx)", fontsize=9)

    n_frames = 36
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    def _update(frame_idx):
        az = float(azimuths[frame_idx])
        _render_pore_pair(gt_region, pred_region, ax_gt_g, ax_pred_g, elev=25, azim=az)
        return []

    ani = FuncAnimation(fig_gif, _update, frames=n_frames, interval=100, blit=False)
    gif_path = out_dir / f"{vol_name}_pore_{label}.gif"
    writer = PillowWriter(fps=10)
    ani.save(str(gif_path), writer=writer, dpi=80)
    plt.close(fig_gif)
    logger.info("  Saved %s", gif_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — test-set patch metrics (Zarr)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_test_patches(
    model: torch.nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    n_batches: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Compute patch-level metrics over all test patches.

    Returns a dict with mean ± std for PSNR, SSIM, MAE (XCT) and
    Dice, IoU, Precision, Recall, F1 (mask), porosity MAE, and CC counts.
    """
    from torch.utils.data import DataLoader

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_zarr_root, split="test")
    loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, worker_init_fn=zarr_worker_init_fn,
    )

    rows: list[dict] = []
    model.eval()

    for batch_idx, batch in enumerate(loader):
        if n_batches is not None and batch_idx >= n_batches:
            break

        xct_b  = batch["xct"].to(device)     # (B,1,64,64,64) float32
        mask_b = batch["mask"]                # (B,1,64,64,64) float32 cpu

        h       = model.encoder(xct_b)
        mu_b    = model.to_mu(h)
        dec_b   = model.decoder(mu_b)
        xct_out = model.xct_head(dec_b).clamp(0.0, 1.0)
        msk_out = torch.sigmoid(model.mask_head(dec_b))

        xct_np  = xct_b .squeeze(1).cpu().numpy()   # (B,64,64,64)
        recon_np = xct_out.squeeze(1).cpu().numpy()
        mask_np  = mask_b .squeeze(1).numpy()
        pred_np  = msk_out.squeeze(1).cpu().numpy()
        por_gt   = batch["porosity"]

        for i in range(xct_np.shape[0]):
            g   = xct_np [i]
            r   = recon_np[i]
            mg  = mask_np [i]
            mp  = pred_np [i]
            pg  = float(por_gt[i])
            pp  = float(mp.mean())

            mae_i  = float(np.abs(g - r).mean())
            mse_i  = float(np.mean((g - r) ** 2))
            psnr_i = 20.0 * math.log10(1.0 / math.sqrt(max(mse_i, 1e-12)))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ssim_i = float(structural_similarity(g, r, data_range=1.0, win_size=5))

            bin_g = (mg >= 0.5)
            bin_p = (mp >= 0.5)
            seg   = dice_precision_recall(bin_g, bin_p)
            f1    = seg["dice"]  # F1 == Dice

            cc_g = int(ndimage.label(bin_g)[1])
            cc_p = int(ndimage.label(bin_p)[1])

            rows.append({
                "volume_id": batch["volume_id"][i],
                "mae": mae_i, "psnr": psnr_i, "ssim": ssim_i,
                "dice": seg["dice"], "iou": seg["iou"],
                "precision": seg["precision"], "recall": seg["recall"],
                "f1": f1,
                "porosity_gt": pg, "porosity_pred": pp,
                "porosity_mae": abs(pp - pg),
                "cc_gt": cc_g, "cc_pred": cc_p,
            })

        if (batch_idx + 1) % 10 == 0:
            logger.info("  patch eval: %d batches / %d patches …",
                        batch_idx + 1, len(rows))

    if not rows:
        return {}

    df = pd.DataFrame(rows)

    def _agg(col: str) -> dict[str, float]:
        return {"mean": float(df[col].mean()), "std": float(df[col].std())}

    result: dict = {"n_patches": len(df)}
    for col in ["mae", "psnr", "ssim", "dice", "iou", "precision",
                "recall", "f1", "porosity_mae", "cc_gt", "cc_pred"]:
        result[col] = _agg(col)

    # Per-volume porosity MAE
    por_vol = (
        df.groupby("volume_id")
        .apply(lambda g: abs(g["porosity_pred"].mean() - g["porosity_gt"].mean()))
        .reset_index(name="por_vol_mae")
    )
    result["porosity_volume_mae"] = {
        "mean": float(por_vol["por_vol_mae"].mean()),
        "std":  float(por_vol["por_vol_mae"].std()),
    }

    # Porosity scatter data (for plot)
    result["_por_scatter"] = {
        "gt":   df["porosity_gt"].tolist(),
        "pred": df["porosity_pred"].tolist(),
    }

    return result


def save_porosity_scatter(patch_metrics: dict, out_dir: Path) -> None:
    sc = patch_metrics.get("_por_scatter", {})
    if not sc:
        return
    gt   = np.array(sc["gt"])
    pred = np.array(sc["pred"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt, pred, s=2, alpha=0.3, c="steelblue")
    lim = max(gt.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("φ GT")
    ax.set_ylabel("φ Pred")
    ax.set_title("Patch porosity: GT vs Predicted")
    fig.tight_layout()
    out = out_dir / "porosity_scatter.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — latent space audit
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def latent_audit(
    model: torch.nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    n_patches: int = 512,
    batch_size: int = 32,
    out_dir: Path | None = None,
) -> dict:
    """Compute per-channel KL and active-channel count over ~n_patches test patches."""
    from torch.utils.data import DataLoader, Subset
    import random as _random

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_zarr_root, split="test")
    idx = _random.sample(range(len(test_ds)), min(n_patches, len(test_ds)))
    sub = Subset(test_ds, idx)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2,
                        worker_init_fn=zarr_worker_init_fn)

    all_mu     = []   # list of (B, C, d, h, w) tensors
    all_logvar = []

    model.eval()
    for batch in loader:
        xct_b = batch["xct"].to(device)
        h     = model.encoder(xct_b)
        mu_b  = model.to_mu(h)
        lv_b  = model.to_logvar(h)
        all_mu    .append(mu_b    .cpu())
        all_logvar.append(lv_b.cpu())

    mu_all  = torch.cat(all_mu,     dim=0).float()   # (N, C, d, h, w)
    lv_all  = torch.cat(all_logvar, dim=0).float()

    # Per-channel KL: KL_c = 0.5 * mean_spatial_batch(μ² + σ² - log σ² - 1)
    sigma2 = torch.exp(lv_all)
    kl_per_ch = 0.5 * (mu_all.pow(2) + sigma2 - lv_all - 1.0)  # (N,C,d,h,w)
    kl_per_ch = kl_per_ch.mean(dim=(0, 2, 3, 4)).numpy()        # (C,)

    # Active channels: σ_avg > 0.1
    sigma_avg = sigma2.sqrt().mean(dim=(0, 2, 3, 4)).numpy()     # (C,)
    active    = int((sigma_avg > 0.1).sum())
    n_total   = int(kl_per_ch.shape[0])

    logger.info("Latent audit: %d active / %d channels", active, n_total)
    for c_idx in range(n_total):
        logger.info("  ch%02d  KL=%.4f  σ_avg=%.4f", c_idx, kl_per_ch[c_idx], sigma_avg[c_idx])

    if out_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # KL bar chart
        axes[0].bar(range(n_total), kl_per_ch, color="steelblue")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_xlabel("Channel"); axes[0].set_ylabel("KL")
        axes[0].set_title("Per-channel KL")

        # μ histogram (all channels overlaid)
        mu_flat = mu_all.reshape(-1).numpy()
        axes[1].hist(mu_flat, bins=100, color="blue", alpha=0.7, density=True)
        axes[1].set_xlabel("μ"); axes[1].set_title("μ distribution (all channels)")

        # σ bar chart
        axes[2].bar(range(n_total), sigma_avg, color="orange",
                    label=f"σ_avg  (>{0.1:.1f}: {active}/{n_total} active)")
        axes[2].axhline(0.1, color="red", lw=1, ls="--", label="threshold 0.1")
        axes[2].set_xlabel("Channel"); axes[2].set_ylabel("σ_avg")
        axes[2].set_title("Per-channel mean σ")
        axes[2].legend(fontsize=7)

        fig.suptitle(f"Latent audit — {active}/{n_total} active channels", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(out_dir / "latent_audit.png"), dpi=100)
        plt.close(fig)
        logger.info("  Saved %s", out_dir / "latent_audit.png")

    return {
        "n_active_channels": active,
        "n_total_channels":  n_total,
        "kl_per_channel":    kl_per_ch.tolist(),
        "sigma_avg":         sigma_avg.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — prior sample decode
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def prior_sample_decode(
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
    n_samples: int = 8,
    seed: int = 0,
) -> None:
    """Sample z ~ N(0,I) and decode → XCT and mask mid-slices grid."""
    torch.manual_seed(seed)
    # z shape: (n_samples, z_channels, spatial_latent, spatial_latent, spatial_latent)
    z = torch.randn(n_samples, Z_CHANNELS, SPATIAL_LATENT, SPATIAL_LATENT, SPATIAL_LATENT,
                    device=device)

    dec      = model.decoder(z)
    xct_out  = model.xct_head(dec).clamp(0.0, 1.0).cpu().numpy()   # (N,1,64,64,64)
    mask_out = torch.sigmoid(model.mask_head(dec)).cpu().numpy()   # (N,1,64,64,64)
    mask_bin = (mask_out >= 0.5).astype(np.float32)

    n_rows = 4  # XCT row 1&2, mask row 3&4
    n_cols = n_samples // 2  # 4 columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    mid = PATCH_SIZE // 2

    for j in range(n_samples):
        row_xct  = 0 if j < n_cols else 1
        row_mask = 2 if j < n_cols else 3
        col = j % n_cols

        axes[row_xct ][col].imshow(xct_out [j, 0, mid], cmap="gray", vmin=0, vmax=1)
        axes[row_mask][col].imshow(mask_bin[j, 0, mid], cmap="binary", vmin=0, vmax=1)
        axes[row_xct ][col].set_title(f"S{j} XCT",  fontsize=8)
        axes[row_mask][col].set_title(f"S{j} mask", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(f"Prior samples z~N(0,I), mid axial slice", fontsize=10)
    fig.tight_layout()
    out = out_dir / "prior_samples.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — test volume selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_test_volumes(
    data_zarr_root: Path,
    n_volumes: int,
) -> list[str]:
    """Select n_volumes test volumes spanning low/mid/high porosity."""
    index_path = data_zarr_root / "patch_index.parquet"
    df = pd.read_parquet(str(index_path))
    test = (
        df[df["split"] == "test"]
        .groupby("volume_id")["porosity"]
        .mean()
        .sort_values()
    )
    if n_volumes >= len(test):
        return list(test.index)

    # Evenly-spaced picks across the porosity range
    positions = np.linspace(0, len(test) - 1, n_volumes, dtype=int)
    return [test.index[p] for p in positions]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 — summary table
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_table(
    vol_metrics: list[dict],
    patch_metrics: dict,
    latent_info: dict,
) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("R03 EVALUATION SUMMARY")
    lines.append("=" * 90)

    if vol_metrics:
        lines.append("\nFull-volume metrics (from original TIFFs):")
        hdr = f"{'Volume':<50}  {'Por GT':>7}  {'Por MAE':>7}  {'PSNR':>7}  {'SSIM':>6}  {'Dice':>6}  {'S2-W1':>7}  {'PSD-W1':>7}  {'CC GT':>6}  {'CC Pred':>7}"
        lines.append(hdr)
        lines.append("-" * 120)
        for m in vol_metrics:
            name = short_name(m["volume_id"])[:50]
            lines.append(
                f"{name:<50}  "
                f"{m['porosity_gt']:>7.4f}  "
                f"{m['porosity_mae']:>7.4f}  "
                f"{m['xct_psnr']:>7.2f}  "
                f"{m['xct_ssim']:>6.4f}  "
                f"{m['dice']:>6.4f}  "
                f"{m['s2_wasserstein']:>7.4f}  "
                f"{m.get('psd_wasserstein', float('nan')):>7.4f}  "
                f"{m['pore_count_gt']:>6d}  "
                f"{m['pore_count_pred']:>7d}"
            )

    if patch_metrics:
        lines.append("\nPatch-level metrics (Zarr test set):")
        lines.append(f"  n_patches : {patch_metrics.get('n_patches', 'n/a')}")
        for col in ["psnr", "ssim", "mae", "dice", "iou", "precision",
                    "recall", "porosity_mae"]:
            v = patch_metrics.get(col, {})
            mu  = v.get("mean", float("nan"))
            std = v.get("std",  float("nan"))
            lines.append(f"  {col:<20} {mu:.4f} ± {std:.4f}")
        pm = patch_metrics.get("porosity_volume_mae", {})
        lines.append(f"  {'por_vol_mae':<20} {pm.get('mean', float('nan')):.4f} ± {pm.get('std', float('nan')):.4f}")

    if latent_info:
        n_act = latent_info.get("n_active_channels", "?")
        n_tot = latent_info.get("n_total_channels",  "?")
        lines.append(f"\nLatent audit: {n_act}/{n_tot} active channels (σ_avg > 0.1)")
        kl = latent_info.get("kl_per_channel", [])
        if kl:
            lines.append("  Per-channel KL: " + "  ".join(f"ch{i}={v:.3f}" for i, v in enumerate(kl)))

    lines.append("=" * 90)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comprehensive post-training evaluation for R03 VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to best.ckpt")
    p.add_argument(
        "--data_zarr",
        default="data/split_v2",
        help="Data root containing volumes.zarr and patch_index.parquet",
    )
    p.add_argument(
        "--data_tiff",
        default="raw_data",
        help="Raw TIFF root (contains MedidasDB/ subdirectory)",
    )
    p.add_argument("--out_dir",   default="results/r03_eval")
    p.add_argument("--n_volumes", type=int, default=3,
                   help="Number of test volumes for full-volume reconstruction")
    p.add_argument("--r_max",     type=int, default=50, help="Max radius for S2(r)")
    p.add_argument("--gpu",       type=int, default=0)
    p.add_argument("--no_ripley", action="store_true",
                   help="Skip Ripley's K computation (saves time)")
    p.add_argument("--patch_batches", type=int, default=None,
                   help="Limit patch-metric evaluation to this many batches")
    p.add_argument("--latent_patches", type=int, default=512,
                   help="Number of random test patches for latent audit")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = build_parser().parse_args(argv)
    t_global = time.perf_counter()

    # ── Paths ─────────────────────────────────────────────────────────────────
    out_root  = Path(args.out_dir)
    vol_dir   = out_root / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    data_zarr = Path(args.data_zarr)
    data_tiff = Path(args.data_tiff)
    zarr_path = data_zarr / "volumes.zarr"

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(device)
    step, meta = load_checkpoint(
        args.checkpoint, model, restore_rng=False, map_location=device,
    )
    model.eval()
    logger.info("Loaded checkpoint (step=%d) from %s", step, args.checkpoint)

    zarr_root = zarr.open_group(str(zarr_path), mode="r")

    # ── Select test volumes ───────────────────────────────────────────────────
    vol_ids = select_test_volumes(data_zarr, args.n_volumes)
    logger.info("Selected %d test volumes:", len(vol_ids))
    for v in vol_ids:
        logger.info("  %s", v)

    # ═════════════════════════════════════════════════════════════════════════
    # Section 1+2+3+4: full-volume reconstruction and per-volume analysis
    # ═════════════════════════════════════════════════════════════════════════
    all_vol_metrics: list[dict] = []

    for vol_id in vol_ids:
        vol_name = short_name(vol_id)
        logger.info("\n── Volume: %s ──", vol_name)
        t_vol = time.perf_counter()

        try:
            xct_gt, mask_gt, xct_recon, mask_pred = reconstruct_volume_from_tiff(
                model, vol_id, data_tiff, zarr_root, device,
            )
        except FileNotFoundError as exc:
            logger.error("  TIFF not found: %s — skipping", exc)
            continue
        except Exception as exc:
            logger.error("  Reconstruction failed: %s — skipping", exc)
            continue

        logger.info("  Reconstruction done in %.1fs", time.perf_counter() - t_vol)

        # ── Save output TIFFs ────────────────────────────────────────────────
        save_output_tiffs(xct_recon, mask_pred, vol_name, vol_dir)

        # ── Per-volume metrics ───────────────────────────────────────────────
        logger.info("  Computing per-volume metrics …")
        metrics = compute_volume_metrics(
            vol_id, xct_gt, mask_gt, xct_recon, mask_pred,
            r_max=args.r_max,
            run_ripley=not args.no_ripley,
        )
        all_vol_metrics.append(metrics)

        # Remove large array fields before saving JSON
        json_metrics = {k: v for k, v in metrics.items()
                        if k not in ("psd_gt", "psd_pred")}
        metrics_path = vol_dir / f"{vol_name}_metrics.json"
        with open(str(metrics_path), "w") as f:
            json.dump(json_metrics, f, indent=2)
        logger.info("  Saved %s", metrics_path)

        # ── Slice comparison figures ─────────────────────────────────────────
        logger.info("  Generating slice comparisons …")
        save_slice_comparisons(xct_gt, xct_recon, mask_gt, mask_pred, vol_name, vol_dir)

        # ── S2(r) and PSD plots ──────────────────────────────────────────────
        save_s2r_plot(metrics, vol_name, vol_dir)
        save_psd_plot(metrics, vol_name, vol_dir)

        # ── 3-D pore comparisons ─────────────────────────────────────────────
        logger.info("  Generating 3-D pore comparisons …")
        for size_label in ("small", "medium", "large"):
            try:
                save_pore_3d(mask_gt, mask_pred, size_label, vol_name, vol_dir)
            except Exception as exc:
                logger.warning("  3-D pore %s failed: %s", size_label, exc)

        # Per-volume summary
        m = metrics
        logger.info(
            "  por_mae=%.4f  psnr=%.2f  ssim=%.4f  dice=%.4f  "
            "s2_w1=%.4f  psd_w1=%.4f  cc_gt=%d  cc_pred=%d",
            m["porosity_mae"], m["xct_psnr"], m["xct_ssim"], m["dice"],
            m["s2_wasserstein"], m.get("psd_wasserstein", float("nan")),
            m["pore_count_gt"], m["pore_count_pred"],
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Section 5: full test-set patch metrics
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("\n── Patch metrics (full test set from Zarr) ──")
    try:
        patch_metrics = eval_test_patches(
            model, data_zarr, device,
            n_batches=args.patch_batches,
        )
        # Remove private scatter data before JSON save
        json_patch = {k: v for k, v in patch_metrics.items() if not k.startswith("_")}
        with open(str(out_root / "r03_patch_metrics.json"), "w") as f:
            json.dump({"step": step, "metrics": json_patch}, f, indent=2)
        logger.info("  Saved r03_patch_metrics.json")

        save_porosity_scatter(patch_metrics, out_root)
    except Exception as exc:
        logger.error("  Patch metrics failed: %s", exc)
        patch_metrics = {}

    # ═════════════════════════════════════════════════════════════════════════
    # Section 6: latent space audit
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("\n── Latent space audit ──")
    try:
        latent_info = latent_audit(
            model, data_zarr, device,
            n_patches=args.latent_patches,
            out_dir=out_root,
        )
        latent_json_path = out_root / "latent_audit.json"
        with open(str(latent_json_path), "w") as f:
            json.dump(latent_info, f, indent=2)
        logger.info("  Saved latent_audit.json")
    except Exception as exc:
        logger.error("  Latent audit failed: %s", exc)
        latent_info = {}

    # ═════════════════════════════════════════════════════════════════════════
    # Section 7: prior sample decode
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("\n── Prior sample decode ──")
    try:
        prior_sample_decode(model, device, out_root)
    except Exception as exc:
        logger.error("  Prior sample decode failed: %s", exc)

    # ═════════════════════════════════════════════════════════════════════════
    # Summary table
    # ═════════════════════════════════════════════════════════════════════════
    summary = build_summary_table(all_vol_metrics, patch_metrics, latent_info)
    print("\n" + summary)
    summary_path = out_root / "summary_table.txt"
    with open(str(summary_path), "w") as f:
        f.write(summary + "\n")
    logger.info("Saved summary → %s", summary_path)

    elapsed = time.perf_counter() - t_global
    logger.info("\nEVAL COMPLETE  step=%d  volumes=%d  time=%.0fs (%.1fm)",
                step, len(all_vol_metrics), elapsed, elapsed / 60)


if __name__ == "__main__":
    main()

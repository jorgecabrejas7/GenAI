"""Plot and image generation for the PoreGen VAE evaluation pipeline.

All public functions accept already-computed result objects (dataclasses from
``metrics.py`` and ``stochastic.py``) and write one or more PNG / GIF files to
a caller-supplied output directory.  No model inference happens here — that
separation keeps this module fast and independently testable.

Public API
----------
``save_slice_comparisons(vol, vol_dir)``
    Nine figures: one per (mode × axis) combination.  Each figure is a
    3-row × 5-col grid of slices.  Modes: stoch_mean, stoch_single, mu.
    Axes: axial (z), coronal (y), sagittal (x).

``save_std_images(vol, vol_dir)``
    Four figures: mid-axial and mid-sagittal slices of ``xct_stoch_std``
    and ``mask_stoch_std`` rendered with the *plasma* colormap so that
    voxels with high posterior uncertainty stand out.

``save_s2r_plot(vm, vol_dir)``
    Single figure: GT vs predicted S₂(r) two-point correlation function.

``save_psd_plot(vm, vol_dir)``
    Single figure: GT vs predicted pore-size distribution (equivalent
    diameter histogram on a log-x axis).

``save_latent_audit_plot(audit, out_dir)``
    Single 3-panel figure: per-channel KL bar chart, σ_avg bar chart with
    the LDM-readiness band shaded, and overall μ distribution histogram.

``save_prior_samples(model, device, out_dir, *, z_channels, spatial_latent)``
    Sample z ~ N(0, I) and decode to a 4-row × n-col grid of mid-axial
    XCT and mask slices.

``save_porosity_scatter(pm, out_dir)``
    GT vs predicted patch porosity coloured by volume_id.

``save_pore_gifs(vol, vol_dir)``
    For small / medium / large GT pores: a static 6-angle PNG and a
    36-frame 360° rotating GIF comparing GT and predicted pore geometry.
"""

from __future__ import annotations

import logging
import math
import random as _random
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label as sk_label, regionprops

if TYPE_CHECKING:
    from poregen.eval.metrics import LatentAudit, PatchMetrics, VolumeMetrics
    from poregen.eval.stochastic import VolumeReconstruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract one 2-D slice from a 3-D array along *axis* at position *idx*."""
    if axis == 0:
        return vol[idx]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def _vol_to_equiv_diam(volumes: np.ndarray) -> np.ndarray:
    """Convert pore volumes (voxels) to equivalent sphere diameters."""
    return 2.0 * (3.0 * volumes / (4.0 * math.pi)) ** (1.0 / 3.0)


def _short_id(volume_id: str) -> str:
    """Strip the MedidasDB__ prefix used in volume identifiers."""
    return volume_id.replace("MedidasDB__", "")


# ---------------------------------------------------------------------------
# Slice comparison grids
# ---------------------------------------------------------------------------

def _comparison_grid(
    xct_gt: np.ndarray,
    xct_recon: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    slice_indices: list[int],
    axis: int,
    title: str,
) -> plt.Figure:
    """Build a (n_slices × 5) figure: GT XCT | Recon XCT | GT mask | Pred mask | |diff|."""
    n = len(slice_indices)
    fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n), squeeze=False)
    fig.suptitle(title, fontsize=11)

    col_titles = ["GT XCT", "Recon XCT", "GT mask", "Pred mask", "|GT − Recon|"]

    for row, sl in enumerate(slice_indices):
        imgs = [
            _take_slice(xct_gt,    axis, sl),
            _take_slice(xct_recon, axis, sl),
            _take_slice(mask_gt,   axis, sl),
            _take_slice(mask_pred, axis, sl),
            np.abs(_take_slice(xct_gt, axis, sl) - _take_slice(xct_recon, axis, sl)),
        ]
        cmaps = ["gray", "gray", "binary_r", "binary_r", "hot"]
        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = axes[row][col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
        axes[row][0].set_ylabel(f"slice {sl}", fontsize=8)

    plt.tight_layout()
    return fig


def save_slice_comparisons(
    vol: "VolumeReconstruction",
    vol_dir: Path,
) -> None:
    """Save 9 slice-comparison PNGs (3 modes × 3 axes).

    Files written
    -------------
    ``<vol_dir>/<short_id>_slices_<mode>_<axis>.png``

    Each figure has 3 rows (quarter / mid / three-quarter slice positions) and
    5 columns: GT XCT, Recon XCT, GT mask, Pred mask, |GT − Recon| difference.

    Parameters
    ----------
    vol : VolumeReconstruction
        Fully stitched volume with all three reconstruction modes populated.
    vol_dir : Path
        Destination directory (must already exist).
    """
    Dz, Dy, Dx = vol.shape_tiled
    short = _short_id(vol.volume_id)

    # mode_name → (xct_recon, mask_recon)
    modes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "stoch_mean":   (vol.xct_stoch_mean,   vol.mask_stoch_mean),
        "stoch_single": (vol.xct_stoch_single, vol.mask_stoch_single),
        "mu":           (vol.xct_mu,           vol.mask_mu),
    }

    axes_cfg = [
        ("axial",    0, Dz),
        ("coronal",  1, Dy),
        ("sagittal", 2, Dx),
    ]

    for mode_name, (xct_r, mask_r) in modes.items():
        for axis_name, axis, dim in axes_cfg:
            idxs = [dim // 4, dim // 2, 3 * dim // 4]
            title = f"{short}  [{mode_name}]  {axis_name}"
            fig = _comparison_grid(
                vol.xct_gt, xct_r, vol.mask_gt, mask_r,
                idxs, axis, title,
            )
            out = vol_dir / f"{short}_slices_{mode_name}_{axis_name}.png"
            fig.savefig(str(out), dpi=100, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved %s", out)


# ---------------------------------------------------------------------------
# Uncertainty / std images
# ---------------------------------------------------------------------------

def save_std_images(
    vol: "VolumeReconstruction",
    vol_dir: Path,
) -> None:
    """Save four uncertainty-map PNGs using the *plasma* colormap.

    Files written
    -------------
    ``<vol_dir>/<short_id>_std_xct_axial.png``
    ``<vol_dir>/<short_id>_std_xct_sagittal.png``
    ``<vol_dir>/<short_id>_std_mask_axial.png``
    ``<vol_dir>/<short_id>_std_mask_sagittal.png``

    The colour scale is normalised to [0, max_std] per image so that the
    full dynamic range of the plasma ramp is used.  Brighter voxels are
    more uncertain under the posterior.

    Parameters
    ----------
    vol : VolumeReconstruction
    vol_dir : Path
    """
    Dz, _, Dx = vol.shape_tiled
    short = _short_id(vol.volume_id)

    panels = [
        ("xct",  "axial",    vol.xct_stoch_std,  0, Dz // 2),
        ("xct",  "sagittal", vol.xct_stoch_std,  2, Dx // 2),
        ("mask", "axial",    vol.mask_stoch_std, 0, Dz // 2),
        ("mask", "sagittal", vol.mask_stoch_std, 2, Dx // 2),
    ]

    for kind, axis_name, std_vol, axis, idx in panels:
        sl = _take_slice(std_vol, axis, idx)
        vmax = float(sl.max()) if sl.max() > 0 else 1e-6

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(sl, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="σ")
        ax.set_title(f"{short}  {kind} std  {axis_name}  [max={vmax:.4f}]", fontsize=9)
        ax.axis("off")
        fig.tight_layout()

        out = vol_dir / f"{short}_std_{kind}_{axis_name}.png"
        fig.savefig(str(out), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", out)


# ---------------------------------------------------------------------------
# S2(r) and PSD plots
# ---------------------------------------------------------------------------

def save_s2r_plot(vm: "VolumeMetrics", vol_dir: Path) -> None:
    """Save a GT vs predicted S₂(r) two-point correlation function plot.

    Parameters
    ----------
    vm : VolumeMetrics
        Must have ``s2_r_vals``, ``s2_gt``, ``s2_pred``, ``s2_wasserstein``
        populated (i.e. ``run_s2r=True`` was used when computing metrics).
    vol_dir : Path
    """
    if vm.s2_wasserstein is None or not vm.s2_r_vals:
        logger.debug("S2(r) data not available for %s — skipping", vm.volume_id)
        return

    short = _short_id(vm.volume_id)
    r  = np.array(vm.s2_r_vals)
    gt = np.array(vm.s2_gt)
    pr = np.array(vm.s2_pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, gt, "b-",  lw=1.5, label="GT")
    ax.plot(r, pr, "r--", lw=1.5, label="Stoch-mean")
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("S₂(r)")
    ax.set_title(f"{short}  S₂(r)  W₁ = {vm.s2_wasserstein:.4f}")
    ax.legend()
    fig.tight_layout()

    out = vol_dir / f"{short}_s2r.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


def save_psd_plot(vm: "VolumeMetrics", vol_dir: Path) -> None:
    """Save a GT vs predicted pore-size distribution histogram (log-x axis).

    Parameters
    ----------
    vm : VolumeMetrics
        Must have ``psd_gt``, ``psd_pred``, ``psd_wasserstein`` populated
        (i.e. ``run_psd=True``).
    vol_dir : Path
    """
    if not vm.psd_gt and not vm.psd_pred:
        logger.debug("PSD data not available for %s — skipping", vm.volume_id)
        return

    short = _short_id(vm.volume_id)
    psd_gt   = np.array(vm.psd_gt,   dtype=np.float64)
    psd_pred = np.array(vm.psd_pred, dtype=np.float64)

    # Bin edges in equivalent-diameter space (log scale)
    max_vol = max(
        psd_gt.max()   if len(psd_gt)   > 0 else 100.0,
        psd_pred.max() if len(psd_pred) > 0 else 100.0,
    )
    bins = np.logspace(np.log10(0.5), np.log10(max_vol + 1), 40)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(psd_gt) > 0:
        ax.hist(
            _vol_to_equiv_diam(psd_gt),
            bins=bins, alpha=0.5, color="royalblue",
            label=f"GT  (n={len(psd_gt)})", density=True,
        )
    if len(psd_pred) > 0:
        ax.hist(
            _vol_to_equiv_diam(psd_pred),
            bins=bins, alpha=0.5, color="tomato",
            label=f"Pred  (n={len(psd_pred)})", density=True,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Equivalent diameter (voxels)")
    ax.set_ylabel("Density")
    w1_str = f"{vm.psd_wasserstein:.2f}" if vm.psd_wasserstein is not None and not math.isnan(vm.psd_wasserstein) else "n/a"
    ax.set_title(f"{short}  PSD  W₁ = {w1_str}")
    ax.legend()
    fig.tight_layout()

    out = vol_dir / f"{short}_psd.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", out)


# ---------------------------------------------------------------------------
# Latent space audit plot
# ---------------------------------------------------------------------------

def save_latent_audit_plot(
    audit: "LatentAudit",
    out_dir: Path,
    *,
    ldm_sigma_low: float = 0.3,
    ldm_sigma_high: float = 0.7,
) -> None:
    """Save a 3-panel latent-audit figure.

    Panels
    ------
    Left
        Per-channel KL divergence bar chart.  Bars for channels in the LDM-
        readiness σ range are coloured green; flagged-low blue; flagged-high red.
    Centre
        Per-channel σ_avg bar chart.  A grey band marks the
        [ldm_sigma_low, ldm_sigma_high] readiness zone.
    Right
        Flat μ histogram (all channels, all spatial positions) showing the
        aggregate posterior-mean distribution.

    Files written
    -------------
    ``<out_dir>/latent_audit.png``

    Parameters
    ----------
    audit : LatentAudit
    out_dir : Path
    ldm_sigma_low, ldm_sigma_high : float
        Band boundaries shown on the σ panel.
    """
    n = audit.n_total_channels
    ch_idx = np.arange(n)
    kl_arr    = np.array(audit.kl_per_channel)
    sigma_arr = np.array(audit.sigma_avg)

    # Bar colours based on readiness classification
    bar_colors = []
    for c in range(n):
        if c in audit.channels_flagged_low:
            bar_colors.append("steelblue")
        elif c in audit.channels_flagged_high:
            bar_colors.append("tomato")
        else:
            bar_colors.append("seagreen")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Panel 0: KL per channel ---
    axes[0].bar(ch_idx, kl_arr, color=bar_colors, edgecolor="none")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Channel index")
    axes[0].set_ylabel("KL divergence")
    axes[0].set_title("Per-channel KL (blue=collapsed, red=wide)")

    # --- Panel 1: σ_avg per channel ---
    axes[1].bar(ch_idx, sigma_arr, color=bar_colors, edgecolor="none")
    axes[1].axhspan(
        ldm_sigma_low, ldm_sigma_high,
        color="lightgreen", alpha=0.25, label=f"LDM range [{ldm_sigma_low}, {ldm_sigma_high}]",
    )
    axes[1].axhline(ldm_sigma_low,  color="green", lw=1, ls="--")
    axes[1].axhline(ldm_sigma_high, color="green", lw=1, ls="--")
    axes[1].set_xlabel("Channel index")
    axes[1].set_ylabel("σ_avg")
    ready_label = "LDM ready ✓" if audit.ldm_ready else "NOT ready ✗"
    axes[1].set_title(
        f"Per-channel σ_avg  ({audit.n_active_channels}/{n} active)  {ready_label}"
    )
    axes[1].legend(fontsize=7)

    # --- Panel 2: μ histogram ---
    axes[2].set_xlabel("μ")
    axes[2].set_ylabel("Density")
    axes[2].set_title("μ distribution (all channels, all patches)")
    # We only have per-channel averages stored; draw a bar chart of mean-μ instead
    # (full μ tensor is not stored in LatentAudit — draw σ_avg vs KL scatter instead)
    axes[2].scatter(kl_arr, sigma_arr, c=range(n), cmap="tab20", s=60, zorder=3)
    for i, (kl_i, sig_i) in enumerate(zip(kl_arr, sigma_arr)):
        axes[2].annotate(str(i), (kl_i, sig_i), fontsize=6, ha="center", va="bottom")
    axes[2].axhspan(ldm_sigma_low, ldm_sigma_high, color="lightgreen", alpha=0.2)
    axes[2].set_xlabel("KL divergence")
    axes[2].set_ylabel("σ_avg")
    axes[2].set_title("KL vs σ_avg per channel")

    ldm_str = f"LDM ready: {'YES' if audit.ldm_ready else 'NO'}  "
    ldm_str += f"({len(audit.channels_in_range)}/{n} in range)"
    fig.suptitle(f"Latent audit — {ldm_str}", fontsize=11)
    fig.tight_layout()

    out = out_dir / "latent_audit.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Prior samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_prior_samples(
    model: nn.Module,
    device: torch.device,
    out_dir: Path,
    *,
    z_channels: int,
    spatial_latent: int,
    n_samples: int = 8,
    seed: int = 0,
    figure_title: str = "Prior samples  z ~ N(0, I)  —  GAN stability check",
) -> None:
    """Sample z ~ N(0, I) and decode to a grid of mid-axial XCT and mask slices.

    Layout: 4 rows × (n_samples // 2) columns.
      Row 0: XCT slices for samples 0 … n_cols-1
      Row 1: XCT slices for samples n_cols … n_samples-1
      Row 2: binary mask for samples 0 … n_cols-1
      Row 3: binary mask for samples n_cols … n_samples-1

    Files written
    -------------
    ``<out_dir>/prior_samples.png``

    Parameters
    ----------
    model : nn.Module
        VAE in eval mode.  Must expose ``decoder``, ``xct_head``, ``mask_head``.
    device : torch.device
    out_dir : Path
    z_channels : int
        Number of latent channels (C dimension).
    spatial_latent : int
        Spatial side-length of the latent grid (typically PATCH_SIZE // 4 = 16).
    n_samples : int
        Number of prior samples to decode.  Must be even.  Default 8.
    seed : int
        Seed for the random z samples.  Default 0 for reproducibility.
    """
    torch.manual_seed(seed)
    s = spatial_latent
    z = torch.randn(n_samples, z_channels, s, s, s, device=device)

    model.eval()
    dec      = model.decoder(z)
    xct_out  = model.xct_head(dec).clamp(0.0, 1.0).cpu().numpy()   # (N,1,64,64,64)
    mask_out = torch.sigmoid(model.mask_head(dec)).cpu().numpy()    # (N,1,64,64,64)
    mask_bin = (mask_out >= 0.5).astype(np.float32)

    patch_size = xct_out.shape[-1]
    mid = patch_size // 2
    n_cols = n_samples // 2  # e.g. 4

    fig, axes = plt.subplots(4, n_cols, figsize=(3 * n_cols, 12))
    for j in range(n_samples):
        row_xct  = 0 if j < n_cols else 1
        row_mask = 2 if j < n_cols else 3
        col = j % n_cols

        axes[row_xct ][col].imshow(xct_out [j, 0, mid], cmap="gray",     vmin=0, vmax=1)
        axes[row_mask][col].imshow(mask_bin[j, 0, mid], cmap="binary_r", vmin=0, vmax=1)
        axes[row_xct ][col].set_title(f"S{j} XCT",  fontsize=8)
        axes[row_mask][col].set_title(f"S{j} mask", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(figure_title, fontsize=10)
    fig.tight_layout()

    out = out_dir / "prior_samples.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Porosity scatter
# ---------------------------------------------------------------------------

def save_porosity_scatter(
    pm: "PatchMetrics",
    out_dir: Path,
) -> None:
    """Save a GT vs predicted patch-porosity scatter plot coloured by volume_id.

    Points are coloured by volume membership using the tab20 palette so that
    per-volume clustering (or lack thereof) is immediately visible.

    Files written
    -------------
    ``<out_dir>/porosity_scatter.png``

    Parameters
    ----------
    pm : PatchMetrics
        Must have ``_porosity_scatter`` populated (non-empty dict with keys
        ``"gt"``, ``"pred"``, ``"volume_id"``).
    out_dir : Path
    """
    sc = pm._porosity_scatter
    if not sc or not sc.get("gt"):
        logger.debug("No porosity scatter data in PatchMetrics — skipping")
        return

    gt   = np.array(sc["gt"],   dtype=np.float32)
    pred = np.array(sc["pred"], dtype=np.float32)
    vids = sc.get("volume_id", ["unknown"] * len(gt))

    # Assign a colour index per unique volume
    unique_vids = sorted(set(vids))
    vid_to_idx  = {v: i for i, v in enumerate(unique_vids)}
    c = [vid_to_idx[v] for v in vids]

    cmap = plt.get_cmap("tab20", max(len(unique_vids), 1))
    lim = max(float(gt.max()), float(pred.max())) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    sc_plot = ax.scatter(gt, pred, c=c, cmap=cmap, vmin=0, vmax=len(unique_vids),
                         s=3, alpha=0.4)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("φ GT")
    ax.set_ylabel("φ Pred")
    ax.set_title(
        f"Patch porosity  (n={len(gt)})\n"
        f"MAE = {pm.porosity_mae.mean:.4f} ± {pm.porosity_mae.std:.4f}"
    )
    ax.legend(fontsize=7)

    # Colour-bar showing volume labels
    if len(unique_vids) <= 20:
        cbar = plt.colorbar(sc_plot, ax=ax, ticks=np.arange(len(unique_vids)) + 0.5)
        cbar.ax.set_yticklabels(
            [_short_id(v)[-20:] for v in unique_vids], fontsize=5
        )

    fig.tight_layout()
    out = out_dir / "porosity_scatter.png"
    fig.savefig(str(out), dpi=100)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# 3-D pore GIFs
# ---------------------------------------------------------------------------

def _render_pore_pair_into(
    gt_region: np.ndarray,
    pred_region: np.ndarray,
    ax_gt: plt.Axes,
    ax_pred: plt.Axes,
    elev: float = 25.0,
    azim: float = 45.0,
) -> None:
    """Render GT and predicted pore voxels into a pair of 3-D Axes objects.

    Tries marching-cubes mesh first; falls back to voxel scatter if it fails.
    Both axes are fully cleared before rendering so the function is safe to
    call from animation update hooks.
    """
    from skimage.measure import marching_cubes

    for ax, region, facecolor, label in [
        (ax_gt,   gt_region,   (0.2, 0.4, 0.9), "GT"),
        (ax_pred, pred_region, (0.9, 0.4, 0.2), "Predicted"),
    ]:
        ax.clear()
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        if region.sum() < 1:
            ax.text2D(0.5, 0.5, "empty", ha="center", va="center",
                      transform=ax.transAxes, fontsize=8)
            continue

        try:
            verts, faces, _, _ = marching_cubes(
                region.astype(float), level=0.5, allow_degenerate=False
            )
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh = Poly3DCollection(
                verts[faces], alpha=0.5,
                facecolor=facecolor, edgecolor="none",
            )
            ax.add_collection3d(mesh)
            ax.set_xlim(0, region.shape[2])
            ax.set_ylim(0, region.shape[1])
            ax.set_zlim(0, region.shape[0])
        except Exception:
            zz, yy, xx = np.where(region)
            ax.scatter(xx, yy, zz, c=[facecolor + (0.7,)], s=4, depthshade=True)
            ax.set_xlim(0, region.shape[2])
            ax.set_ylim(0, region.shape[1])
            ax.set_zlim(0, region.shape[0])


def _select_pore(
    labeled: np.ndarray,
    props: list,
    label: str,
) -> object | None:
    """Pick one pore from *props* matching the requested size category.

    Categories and target equivalent-diameter ranges (voxels):
      * ``"small"``  — 2 … 4
      * ``"medium"`` — 6 … 12
      * ``"large"``  — > 15

    Falls back to the smallest/median/largest pore if no candidate falls
    in the ideal range.  Returns ``None`` if *props* is empty.
    """
    if not props:
        return None

    diams = np.array([p.equivalent_diameter_area for p in props])
    order = np.argsort(diams)

    if label == "small":
        candidates = [i for i in order if 2.0 <= diams[i] <= 4.0]
        if candidates:
            return props[candidates[len(candidates) // 2]]
        return props[order[0]]

    if label == "medium":
        candidates = [i for i in order if 6.0 <= diams[i] <= 12.0]
        if candidates:
            return props[candidates[len(candidates) // 2]]
        if len(order) > 1:
            return props[order[len(order) // 2]]
        return None

    # "large"
    candidates = [i for i in order if diams[i] > 15.0]
    if candidates:
        return props[candidates[len(candidates) // 2]]
    if len(order) > 0:
        return props[order[-1]]
    return None


def _save_pore_for_label(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    label: str,
    short: str,
    vol_dir: Path,
) -> None:
    """Generate a static 6-angle PNG and a rotating GIF for one pore size category.

    Parameters
    ----------
    mask_gt, mask_pred : ndarray, float32, {0, 1}
        Full-volume binary masks.
    label : str
        ``"small"``, ``"medium"``, or ``"large"``.
    short : str
        Short volume identifier used in file names.
    vol_dir : Path
    """
    bin_gt   = (mask_gt   >= 0.5)
    bin_pred = (mask_pred >= 0.5)

    labeled = sk_label(bin_gt)
    props   = regionprops(labeled)

    if not props:
        logger.warning("  No GT pores for %s — skipping 3-D pore %s", short, label)
        return

    target = _select_pore(labeled, props, label)
    if target is None:
        logger.warning("  No suitable %s pore for %s — skipping", label, short)
        return

    # Padded bounding box
    pad = 4
    d0, h0, w0, d1, h1, w1 = target.bbox
    D, H, W = mask_gt.shape
    d0 = max(0, d0 - pad); d1 = min(D, d1 + pad)
    h0 = max(0, h0 - pad); h1 = min(H, h1 + pad)
    w0 = max(0, w0 - pad); w1 = min(W, w1 + pad)

    gt_region   = bin_gt  [d0:d1, h0:h1, w0:w1]
    pred_region = bin_pred[d0:d1, h0:h1, w0:w1]

    diam = target.equivalent_diameter_area

    # ── Static 6-angle PNG ─────────────────────────────────────────────────
    angles = [(25, 0), (25, 90), (25, 180), (25, 270), (80, 45), (-10, 45)]
    fig = plt.figure(figsize=(14, 10))
    for i, (elev, azim) in enumerate(angles):
        ax_gt   = fig.add_subplot(3, 4, 2 * i + 1, projection="3d")
        ax_pred = fig.add_subplot(3, 4, 2 * i + 2, projection="3d")
        _render_pore_pair_into(gt_region, pred_region, ax_gt, ax_pred, elev=elev, azim=azim)
    fig.suptitle(
        f"{short}  {label} pore  d ≈ {diam:.1f} vx", fontsize=10
    )
    fig.tight_layout()
    png_path = vol_dir / f"{short}_pore_{label}.png"
    fig.savefig(str(png_path), dpi=80)
    plt.close(fig)
    logger.info("  Saved %s", png_path)

    # ── Rotating GIF (36 frames, 360°) ─────────────────────────────────────
    fig_gif, (ax_g, ax_p) = plt.subplots(
        1, 2, figsize=(8, 4),
        subplot_kw={"projection": "3d"},
    )
    fig_gif.suptitle(
        f"{short}  {label} pore  (d ≈ {diam:.1f} vx)", fontsize=9
    )
    n_frames = 36
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    def _update(frame_idx: int):
        _render_pore_pair_into(
            gt_region, pred_region, ax_g, ax_p,
            elev=25.0, azim=float(azimuths[frame_idx]),
        )
        return []

    ani = FuncAnimation(fig_gif, _update, frames=n_frames, interval=100, blit=False)
    gif_path = vol_dir / f"{short}_pore_{label}.gif"
    ani.save(str(gif_path), writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig_gif)
    logger.info("  Saved %s", gif_path)


def save_pore_gifs(
    vol: "VolumeReconstruction",
    vol_dir: Path,
) -> None:
    """Generate static PNG + rotating GIF for small, medium, and large GT pores.

    Uses the stochastic-mean predicted mask for comparison.  Files are written
    only for size categories where a qualifying pore exists in the GT mask.

    Files written
    -------------
    ``<vol_dir>/<short_id>_pore_{small|medium|large}.png``
    ``<vol_dir>/<short_id>_pore_{small|medium|large}.gif``

    Parameters
    ----------
    vol : VolumeReconstruction
    vol_dir : Path
    """
    short = _short_id(vol.volume_id)
    for size_label in ("small", "medium", "large"):
        try:
            _save_pore_for_label(
                vol.mask_gt, vol.mask_stoch_mean,
                size_label, short, vol_dir,
            )
        except Exception as exc:
            logger.warning("  Pore GIF (%s / %s) failed: %s", short, size_label, exc)

"""All quantitative metric computations for the PoreGen VAE eval pipeline.

This module is the authoritative source for every metric computed during
evaluation.  It covers three scopes:

Patch-level (``eval_patches``)
    Iterates the full test-set DataLoader.  All metrics except
    ``recon_std_mean`` use a single deterministic forward pass (z = μ) for
    speed.  ``recon_std_mean`` is computed on a random 512-patch subset using
    N stochastic passes to avoid O(N × n_patches) cost.

Volume-level (``eval_volume_metrics``)
    Works on a ``VolumeReconstruction`` object (arrays already in memory).
    Computes XCT quality for all three reconstruction modes, mask quality for
    the stochastic-mean mode, structural metrics (S₂(r), PSD, Ripley's K),
    and boundary consistency.

Latent-space (``latent_audit``)
    Encodes a random subset of test patches to collect per-channel KL
    divergence and posterior width (σ = exp(0.5 × logvar)).  Flags channels
    outside the LDM-readiness range [ldm_sigma_low, ldm_sigma_high] and sets
    ``ldm_ready = True`` iff all channels are in range.

FID (``compute_fid_slices``)
    Extracts 2-D mid-slices from 3-D GT and reconstructed patches and computes
    the Fréchet Inception Distance via InceptionV3 (torchvision) features.

Memorisation (``memorization_score``)
    Nearest-neighbour distance from test latents to train latents as a proxy
    for memorisation.

Historical reference: ``scripts/eval_r03.py`` and ``scripts/eval_checkpoint.py``
contain the original implementations that this module supersedes.
"""

from __future__ import annotations

import logging
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.stats import wasserstein_distance
from skimage.measure import label as sk_label, regionprops
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from poregen.eval.config import EvalConfig
    from poregen.eval.stochastic import VolumeReconstruction

logger = logging.getLogger(__name__)


def _encode(model: nn.Module, xct_t: torch.Tensor) -> torch.Tensor:
    """Architecture-agnostic encoder: handles single-branch and dual-branch VAEs."""
    if hasattr(model, "encoder"):
        return model.encoder(xct_t)
    h_a = model.encoder_a(xct_t)
    h_b = model.encoder_b(xct_t)
    return model.fusion(torch.cat([h_a, h_b], dim=1))


# ---------------------------------------------------------------------------
# Simple statistics container
# ---------------------------------------------------------------------------

@dataclass
class Stat:
    """Mean and standard deviation of a scalar metric over a sample.

    Attributes
    ----------
    mean : float
    std  : float
    """

    mean: float
    std: float

    @classmethod
    def from_list(cls, values: list[float]) -> "Stat":
        """Compute Stat from a list of scalar values.

        Returns ``Stat(nan, nan)`` for an empty list.
        """
        if not values:
            return cls(float("nan"), float("nan"))
        arr = np.array(values, dtype=np.float64)
        return cls(float(arr.mean()), float(arr.std()))


# ---------------------------------------------------------------------------
# Metric result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatchMetrics:
    """Aggregated patch-level metrics over the full test set.

    All ``Stat`` fields summarise per-patch scalar values (mean ± std).

    Attributes
    ----------
    n_patches : int
        Total number of test patches evaluated.
    psnr : Stat
        Peak signal-to-noise ratio of XCT reconstruction.  Computed as
        ``20 * log10(1 / sqrt(MSE))``.
    ssim : Stat
        Structural similarity index (XCT).
    mae : Stat
        Mean absolute error of XCT reconstruction.
    dice : Stat
        Dice coefficient of the binarised predicted mask vs GT.
    precision : Stat
        Mask precision (TP / (TP + FP)).
    recall : Stat
        Mask recall (TP / (TP + FN)).
    f1 : Stat
        F1 score (== Dice for binary segmentation).
    porosity_mae : Stat
        Absolute porosity error per patch: ``|φ_pred − φ_gt|``.
    porosity_bias : Stat
        Signed porosity error per patch: ``φ_pred − φ_gt``.  A positive mean
        indicates systematic over-prediction.
    porosity_volume_mae : Stat
        Per-volume mean absolute porosity error: first aggregate patch
        porosities per volume, then compute the absolute difference between
        mean predicted and mean GT porosity.
    sharpness_ratio : Stat
        Ratio of reconstruction sharpness to GT sharpness.  Values near 1.0
        indicate the decoder preserves high-frequency structure.
    recon_std_mean : float
        Mean over ~512 random test patches of the per-patch spatial mean of
        the voxel-wise std across N stochastic passes.  Low values (< 0.01)
        indicate a tight posterior — a prerequisite for LDM training.
    _porosity_scatter : dict
        Private dict with lists ``"gt"``, ``"pred"``, ``"volume_id"`` for
        scatter-plot generation.  Not serialised to JSON.
    """

    n_patches: int
    psnr: Stat
    ssim: Stat
    mae: Stat
    dice: Stat
    precision: Stat
    recall: Stat
    f1: Stat
    porosity_mae: Stat
    porosity_bias: Stat
    porosity_volume_mae: Stat
    sharpness_ratio: Stat
    recon_std_mean: float
    _porosity_scatter: dict = field(default_factory=dict, repr=False)


@dataclass
class VolumeMetrics:
    """All per-volume quantitative metrics.

    XCT quality metrics are computed for all three reconstruction modes
    (stochastic mean, single stochastic pass, deterministic z = μ).  Mask
    and structural metrics use the stochastic-mean mode as canonical.

    Attributes
    ----------
    volume_id : str
    xct_mae_stoch / xct_psnr_stoch / xct_ssim_stoch : float
        XCT quality for the stochastic-mean reconstruction.
    xct_mae_single / xct_psnr_single / xct_ssim_single : float
        XCT quality for the single stochastic pass.
    xct_mae_mu / xct_psnr_mu / xct_ssim_mu : float
        XCT quality for the deterministic z = μ reconstruction.
    dice / precision / recall / iou : float
        Binary segmentation metrics on the full stitched volume.
    porosity_gt / porosity_pred / porosity_mae : float
        Ground-truth porosity, predicted porosity, and their absolute difference.
    xct_recon_std_mean : float
        Mean of the voxel-wise XCT std volume (proxy for posterior tightness).
    mask_recon_std_mean : float
        Mean of the voxel-wise mask std volume.
    xct_boundary_mae : float
        Mean absolute voxel discontinuity at patch seams in the XCT volume.
    mask_boundary_mae : float
        Same for the mask volume.
    pore_count_gt / pore_count_pred : int
        Number of connected-component pores in GT and prediction.
    s2_wasserstein : float or None
        Wasserstein-1 distance between the GT and predicted S₂(r) curves.
        ``None`` when ``run_s2r=False``.
    s2_r_vals / s2_gt / s2_pred : list
        Radial distances and S₂ values for plotting.
    psd_wasserstein : float or None
        Wasserstein-1 distance between GT and predicted pore size distributions.
    psd_gt / psd_pred : list
        Pore volumes (in voxels) for GT and prediction.
    ripley_w1 : float or None
        Wasserstein-1 distance between GT and predicted Ripley's K(r) curves.
    elapsed_s : float
        Wall-clock time for this volume's computation.
    """

    volume_id: str
    xct_mae_stoch: float
    xct_psnr_stoch: float
    xct_ssim_stoch: float
    xct_mae_single: float
    xct_psnr_single: float
    xct_ssim_single: float
    xct_mae_mu: float
    xct_psnr_mu: float
    xct_ssim_mu: float
    dice: float
    precision: float
    recall: float
    iou: float
    porosity_gt: float
    porosity_pred: float
    porosity_mae: float
    xct_recon_std_mean: float
    mask_recon_std_mean: float
    xct_boundary_mae: float
    mask_boundary_mae: float
    pore_count_gt: int
    pore_count_pred: int
    s2_wasserstein: float | None
    s2_r_vals: list
    s2_gt: list
    s2_pred: list
    psd_wasserstein: float | None
    psd_gt: list
    psd_pred: list
    ripley_w1: float | None
    elapsed_s: float
    porosity_bias: float = 0.0
    sharpness_ratio_axial: float = float("nan")
    sharpness_ratio_coronal: float = float("nan")
    sharpness_ratio_sagittal: float = float("nan")
    sharpness_ratio_overall: float = float("nan")


@dataclass
class LatentAudit:
    """Per-channel latent-space statistics for the LDM readiness gate.

    Attributes
    ----------
    n_active_channels : int
        Number of channels with σ_avg > 0.1.
    n_total_channels : int
        Total number of latent channels.
    kl_per_channel : list of float
        Per-channel KL divergence ``0.5 × E[μ² + σ² − log σ² − 1]`` averaged
        over spatial dimensions and the sample of test patches.
    sigma_avg : list of float
        Per-channel mean posterior width σ = exp(0.5 × logvar), averaged over
        spatial dimensions and the patch sample.
    channels_in_range : list of int
        Channel indices where σ_avg ∈ [ldm_sigma_low, ldm_sigma_high].
    channels_flagged_low : list of int
        Channel indices where σ_avg < ldm_sigma_low.  These channels are
        over-regularised — the encoder collapses them towards zero.
    channels_flagged_high : list of int
        Channel indices where σ_avg > ldm_sigma_high.  These channels are
        under-regularised — the posterior is too wide for stable LDM training.
    ldm_ready : bool
        ``True`` iff every channel is in [ldm_sigma_low, ldm_sigma_high].
    """

    n_active_channels: int
    n_total_channels: int
    kl_per_channel: list[float]
    sigma_avg: list[float]
    channels_in_range: list[int]
    channels_flagged_low: list[int]
    channels_flagged_high: list[int]
    ldm_ready: bool
    branch_cosine_sim_mean: float | None = None


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------

def _psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio between two float arrays."""
    mse = float(np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def _ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Structural similarity index (skimage, data_range=1.0, win_size=7)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(structural_similarity(
            gt.astype(np.float32), pred.astype(np.float32),
            data_range=1.0, win_size=7,
        ))


def _dice_precision_recall(
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
) -> dict[str, float]:
    """Compute Dice, precision, recall, and IoU from two boolean arrays."""
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


def _boundary_mae(vol: np.ndarray, patch_size: int = 64) -> float:
    """Mean absolute voxel discontinuity at non-overlapping patch seams.

    For every internal face where two adjacent ``patch_size``-voxel patches
    meet, computes the mean absolute difference between the last and first
    voxel slice.  Averaged across all D, H, and W seams.

    Returns ``0.0`` for volumes smaller than two patches along every axis.
    """
    D, H, W = vol.shape
    diffs: list[float] = []
    for z in range(patch_size, D, patch_size):
        diffs.append(float(np.abs(vol[z - 1] - vol[z]).mean()))
    for y in range(patch_size, H, patch_size):
        diffs.append(float(np.abs(vol[:, y - 1, :] - vol[:, y, :]).mean()))
    for x in range(patch_size, W, patch_size):
        diffs.append(float(np.abs(vol[:, :, x - 1] - vol[:, :, x]).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def _s2_radial(
    binary: np.ndarray,
    r_max: int = 50,
    n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic two-point correlation function S₂(r) via FFT autocorrelation.

    S₂(r) = P(both randomly chosen points separated by r are in the pore phase).
    Estimated by computing the normalised autocorrelation of the binary mask and
    binning values radially.

    Parameters
    ----------
    binary : ndarray, bool or {0,1}
    r_max  : int — maximum radius in voxels
    n_bins : int — number of radial bins

    Returns
    -------
    r_vals : (n_bins,) — bin-centre radial distances
    s2     : (n_bins,) — S₂ values
    """
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
        mask = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if mask.any():
            s2[i] = autocorr[mask].mean()

    r_vals = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_vals, s2


def _s2_wasserstein(s2_gt: np.ndarray, s2_pred: np.ndarray) -> float:
    """Wasserstein-1 distance between two S₂(r) curves treated as distributions."""
    eps = 1e-12
    a = np.clip(s2_gt,   0, None); a = a / (a.sum() + eps)
    b = np.clip(s2_pred, 0, None); b = b / (b.sum() + eps)
    r = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(r, r, a, b))


def _pore_sizes(binary: np.ndarray) -> np.ndarray:
    """Connected-component pore volumes (in voxels) for every pore."""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.array([], dtype=np.float64)
    return np.array(ndimage.sum(binary, labeled, range(1, n + 1)), dtype=np.float64)


def _ripleys_k_3d(
    binary: np.ndarray,
    r_max: int = 30,
    max_pores: int = 500,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ripley's K(r) estimated on pore centroid positions.

    Returns ``None`` if fewer than 3 pores are found.

    Parameters
    ----------
    binary : ndarray, bool
    r_max  : int — maximum radius
    max_pores : int — subsample to this many centroids for O(N²) control

    Returns
    -------
    r_vals : (r_max,)
    K      : (r_max,)
    """
    from scipy.spatial.distance import pdist

    labeled = sk_label(binary)
    props   = regionprops(labeled)
    if len(props) < 3:
        return None

    if len(props) > max_pores:
        props = random.sample(props, max_pores)

    centroids = np.array([p.centroid for p in props], dtype=np.float64)
    N = len(centroids)
    D_v, H_v, W_v = binary.shape
    V = float(D_v * H_v * W_v)

    dists  = pdist(centroids)
    r_vals = np.arange(1, r_max + 1, dtype=np.float64)
    K = np.array([(V / N**2) * float(np.sum(dists < r)) for r in r_vals])
    return r_vals, K


def _sharpness_proxy(vol: np.ndarray) -> float:
    """Mean absolute gradient magnitude as a sharpness proxy."""
    if vol.ndim < 3:
        return float("nan")
    gz = float(np.abs(np.diff(vol, axis=0)).mean())
    gy = float(np.abs(np.diff(vol, axis=1)).mean())
    gx = float(np.abs(np.diff(vol, axis=2)).mean())
    return (gz + gy + gx) / 3.0


def _sharpness_per_orientation(
    gt_vol: np.ndarray,
    recon_vol: np.ndarray,
) -> dict[str, float]:
    """Sharpness ratio (Laplacian variance) for each anatomical orientation.

    Per-orientation sharpness: mean Laplacian variance over all 2-D slices
    along that axis.  Overall sharpness: 3-D Laplacian variance of the full
    volume.  Ratio = pred / gt; values near 1.0 mean the decoder preserves
    high-frequency structure.  Values < 1.0 indicate blurring; values > 1.0
    indicate over-sharpening or noise amplification (expected with adversarial
    loss).

    Note: PSNR/SSIM may be *lower* with adversarial / GAN loss even when the
    reconstruction is perceptually sharper.  Use sharpness_ratio as the
    primary sharpness diagnostic for R04.
    """
    def _lap_var_slices(vol: np.ndarray, axis: int) -> float:
        """Mean Laplacian variance of 2-D slices along *axis*."""
        vals = [
            float(ndimage.laplace(np.take(vol, i, axis=axis).astype(np.float64)).var())
            for i in range(vol.shape[axis])
        ]
        return float(np.mean(vals))

    def _ratio(gt_s: float, pred_s: float) -> float:
        return float(pred_s / gt_s) if gt_s > 1e-12 else float("nan")

    axial_gt    = _lap_var_slices(gt_vol,    axis=0)
    coronal_gt  = _lap_var_slices(gt_vol,    axis=1)
    sagittal_gt = _lap_var_slices(gt_vol,    axis=2)
    axial_pred    = _lap_var_slices(recon_vol, axis=0)
    coronal_pred  = _lap_var_slices(recon_vol, axis=1)
    sagittal_pred = _lap_var_slices(recon_vol, axis=2)

    overall_gt   = float(ndimage.laplace(gt_vol  .astype(np.float64)).var())
    overall_pred = float(ndimage.laplace(recon_vol.astype(np.float64)).var())

    return {
        "axial":    _ratio(axial_gt,    axial_pred),
        "coronal":  _ratio(coronal_gt,  coronal_pred),
        "sagittal": _ratio(sagittal_gt, sagittal_pred),
        "overall":  _ratio(overall_gt,  overall_pred),
    }


# ---------------------------------------------------------------------------
# Public: patch-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_patches(
    model: nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    eval_cfg: "EvalConfig",
) -> PatchMetrics:
    """Evaluate patch-level metrics over the test split of the Zarr dataset.

    All metrics except ``recon_std_mean`` use a single deterministic forward
    pass (z = μ, no reparameterisation) for speed.  ``recon_std_mean`` is
    computed on a random subset of ``eval_cfg.latent_patches`` patches using
    ``eval_cfg.n_stochastic_samples`` stochastic passes.

    Parameters
    ----------
    model : nn.Module
        ConvVAE3DNoAttnV2 in eval mode.  Encoder receives XCT only.
    data_zarr_root : Path
        Root of the patch dataset (contains ``volumes.zarr`` and
        ``patch_index.parquet``).
    device : torch.device
    eval_cfg : EvalConfig

    Returns
    -------
    PatchMetrics
    """
    from torch.utils.data import DataLoader, Subset

    from poregen.dataset.loader import PatchDataset, zarr_worker_init_fn

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_zarr_root, split="test")
    loader = DataLoader(
        test_ds,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=zarr_worker_init_fn,
    )

    psnr_vals:     list[float] = []
    ssim_vals:     list[float] = []
    mae_vals:      list[float] = []
    dice_vals:     list[float] = []
    prec_vals:     list[float] = []
    rec_vals:      list[float] = []
    f1_vals:       list[float] = []
    por_mae_vals:  list[float] = []
    por_bias_vals: list[float] = []
    sharp_ratio:   list[float] = []

    por_scatter_gt:  list[float] = []
    por_scatter_pred: list[float] = []
    por_scatter_vid:  list[str]  = []

    vol_por_gt:   dict[str, list[float]] = {}
    vol_por_pred: dict[str, list[float]] = {}

    model.eval()

    for batch_idx, batch in enumerate(loader):
        if eval_cfg.patch_batches is not None and batch_idx >= eval_cfg.patch_batches:
            break

        xct_b = batch["xct"].to(device, non_blocking=True)  # (B,1,64,64,64)
        mask_b = batch["mask"]                               # stays on CPU

        # Deterministic pass: z = μ
        h       = _encode(model, xct_b)
        mu_b    = model.to_mu(h)
        dec_b   = model.decoder(mu_b)
        xct_out = model.xct_head(dec_b).clamp(0.0, 1.0)
        msk_out = torch.sigmoid(model.mask_head(dec_b))

        xct_np   = xct_b .squeeze(1).cpu().numpy()   # (B,64,64,64)
        recon_np = xct_out.squeeze(1).cpu().numpy()
        mask_np  = mask_b .squeeze(1).numpy()
        pred_np  = msk_out.squeeze(1).cpu().numpy()
        por_gt_b = batch["porosity"]
        vol_ids  = batch["volume_id"]

        for i in range(xct_np.shape[0]):
            g  = xct_np [i]
            r  = recon_np[i]
            mg = (mask_np[i] >= 0.5)
            mp = (pred_np[i] >= 0.5)
            pg = float(por_gt_b[i])
            pp = float(pred_np[i].mean())
            vid = vol_ids[i]

            mse_i = float(np.mean((g - r) ** 2))
            mae_vals.append(float(np.abs(g - r).mean()))
            psnr_vals.append(20.0 * math.log10(1.0 / math.sqrt(max(mse_i, 1e-12))))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ssim_vals.append(float(structural_similarity(g, r, data_range=1.0, win_size=5)))

            seg = _dice_precision_recall(mg, mp)
            dice_vals.append(seg["dice"])
            prec_vals.append(seg["precision"])
            rec_vals.append(seg["recall"])
            f1_vals.append(seg["dice"])  # F1 == Dice for binary

            por_mae_vals.append(abs(pp - pg))
            por_bias_vals.append(pp - pg)

            sg = _sharpness_proxy(g)
            sr = _sharpness_proxy(r)
            sharp_ratio.append(sr / sg if sg > 1e-9 else float("nan"))

            por_scatter_gt.append(pg)
            por_scatter_pred.append(pp)
            por_scatter_vid.append(str(vid))
            vol_por_gt  .setdefault(str(vid), []).append(pg)
            vol_por_pred.setdefault(str(vid), []).append(pp)

        if (batch_idx + 1) % 20 == 0:
            logger.info("  Patch eval: %d batches / %d patches …",
                        batch_idx + 1, len(psnr_vals))

    # Per-volume porosity MAE
    vol_maes = [abs(np.mean(vol_por_pred[v]) - np.mean(vol_por_gt[v]))
                for v in vol_por_gt]

    # recon_std_mean on latent_patches random patches
    recon_std_mean = _compute_recon_std_mean(model, test_ds, device, eval_cfg)

    return PatchMetrics(
        n_patches=len(psnr_vals),
        psnr=Stat.from_list(psnr_vals),
        ssim=Stat.from_list(ssim_vals),
        mae=Stat.from_list(mae_vals),
        dice=Stat.from_list(dice_vals),
        precision=Stat.from_list(prec_vals),
        recall=Stat.from_list(rec_vals),
        f1=Stat.from_list(f1_vals),
        porosity_mae=Stat.from_list(por_mae_vals),
        porosity_bias=Stat.from_list(por_bias_vals),
        porosity_volume_mae=Stat.from_list(vol_maes),
        sharpness_ratio=Stat.from_list([v for v in sharp_ratio if not math.isnan(v)]),
        recon_std_mean=recon_std_mean,
        _porosity_scatter={
            "gt": por_scatter_gt,
            "pred": por_scatter_pred,
            "volume_id": por_scatter_vid,
        },
    )


@torch.no_grad()
def _compute_recon_std_mean(
    model: nn.Module,
    test_ds: Any,
    device: torch.device,
    eval_cfg: "EvalConfig",
) -> float:
    """Compute mean per-patch std over N stochastic passes on a random subset.

    For each patch in the subset: run N forward passes with z = μ + σε,
    compute voxel-wise std of XCT reconstructions, take spatial mean.
    Average this scalar over all patches in the subset.

    Parameters
    ----------
    model : nn.Module
    test_ds : PatchDataset
    device : torch.device
    eval_cfg : EvalConfig

    Returns
    -------
    float
    """
    from torch.utils.data import DataLoader, Subset

    n = min(eval_cfg.latent_patches, len(test_ds))
    torch.manual_seed(eval_cfg.stochastic_seed)
    idx = random.sample(range(len(test_ds)), n)
    sub = Subset(test_ds, idx)
    loader = DataLoader(sub, batch_size=eval_cfg.batch_size, shuffle=False, num_workers=2)

    patch_stds: list[float] = []
    model.eval()

    for batch in loader:
        xct_b = batch["xct"].to(device)

        h      = _encode(model, xct_b)
        mu_b   = model.to_mu(h)
        lv_b   = model.to_logvar(h)
        std_b  = torch.exp(0.5 * lv_b)

        xct_runs: list[torch.Tensor] = []
        for _ in range(eval_cfg.n_stochastic_samples):
            eps = torch.randn_like(mu_b)
            z   = mu_b + std_b * eps
            dec = model.decoder(z)
            xct_runs.append(model.xct_head(dec).clamp(0.0, 1.0))

        stack = torch.stack(xct_runs, dim=0)   # (N, B, 1, 64, 64, 64)
        per_patch_std = stack.std(dim=0).mean(dim=(1, 2, 3, 4))  # (B,)
        patch_stds.extend(per_patch_std.cpu().tolist())

    return float(np.mean(patch_stds)) if patch_stds else float("nan")


# ---------------------------------------------------------------------------
# Public: volume-level evaluation
# ---------------------------------------------------------------------------

def eval_volume_metrics(
    volume_id: str,
    vol: "VolumeReconstruction",
    *,
    r_max: int = 50,
    run_s2r: bool = True,
    run_psd: bool = True,
    run_ripley: bool = False,
) -> VolumeMetrics:
    """Compute all per-volume quantitative metrics from a ``VolumeReconstruction``.

    XCT quality is evaluated for all three reconstruction modes.  Mask and
    structural metrics use the stochastic-mean mode (``vol.mask_stoch_mean``)
    as the canonical prediction.

    Parameters
    ----------
    volume_id : str
    vol : VolumeReconstruction
        All reconstruction arrays (shape ``(Dz, Dy, Dx)``), already on CPU as
        float32 ndarray.
    r_max : int
        Maximum radius for S₂(r).
    run_s2r : bool
    run_psd : bool
    run_ripley : bool

    Returns
    -------
    VolumeMetrics
    """
    t0 = time.perf_counter()

    # ── XCT quality for three modes ──────────────────────────────────────────
    xct_gt = vol.xct_gt
    def _xct_metrics(pred: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.abs(pred - xct_gt).mean()),
            _psnr(xct_gt, pred),
            _ssim(xct_gt, pred),
        )

    mae_s, psnr_s, ssim_s = _xct_metrics(vol.xct_stoch_mean)
    mae_1, psnr_1, ssim_1 = _xct_metrics(vol.xct_stoch_single)
    mae_m, psnr_m, ssim_m = _xct_metrics(vol.xct_mu)

    # ── Mask quality (stoch-mean as canonical) ───────────────────────────────
    bin_gt   = (vol.mask_gt        >= 0.5).astype(bool)
    bin_pred = (vol.mask_stoch_mean >= 0.5).astype(bool)

    por_gt   = float(bin_gt.mean())
    por_pred = float(bin_pred.mean())
    seg = _dice_precision_recall(bin_gt, bin_pred)

    # ── Uncertainty ──────────────────────────────────────────────────────────
    xct_std_mean  = float(vol.xct_stoch_std .mean())
    mask_std_mean = float(vol.mask_stoch_std.mean())

    # ── Boundary consistency ─────────────────────────────────────────────────
    xct_bnd  = _boundary_mae(vol.xct_stoch_mean)
    mask_bnd = _boundary_mae(vol.mask_stoch_mean)

    # ── Connected components ─────────────────────────────────────────────────
    cc_gt   = int(ndimage.label(bin_gt  )[1])
    cc_pred = int(ndimage.label(bin_pred)[1])

    # ── S₂(r) ────────────────────────────────────────────────────────────────
    s2_w1 = None; s2_r: list = []; s2_g: list = []; s2_p: list = []
    if run_s2r:
        try:
            r_vals, s2_gt_arr = _s2_radial(bin_gt,   r_max=r_max)
            _,      s2_pred_arr = _s2_radial(bin_pred, r_max=r_max)
            s2_w1 = _s2_wasserstein(s2_gt_arr, s2_pred_arr)
            s2_r = r_vals.tolist(); s2_g = s2_gt_arr.tolist(); s2_p = s2_pred_arr.tolist()
        except Exception as exc:
            logger.warning("  S2(r) failed for %s: %s", volume_id, exc)

    # ── PSD ──────────────────────────────────────────────────────────────────
    psd_w1 = None; psd_g: list = []; psd_p: list = []
    if run_psd:
        try:
            psd_gt_arr   = _pore_sizes(bin_gt)
            psd_pred_arr = _pore_sizes(bin_pred)
            psd_g = psd_gt_arr.tolist(); psd_p = psd_pred_arr.tolist()
            if len(psd_gt_arr) > 0 and len(psd_pred_arr) > 0:
                psd_w1 = float(wasserstein_distance(psd_gt_arr, psd_pred_arr))
        except Exception as exc:
            logger.warning("  PSD failed for %s: %s", volume_id, exc)

    # ── Sharpness per orientation ─────────────────────────────────────────────
    sharpness: dict[str, float] = {}
    try:
        sharpness = _sharpness_per_orientation(vol.xct_gt, vol.xct_stoch_mean)
    except Exception as exc:
        logger.warning("  Sharpness failed for %s: %s", volume_id, exc)
        sharpness = {"axial": float("nan"), "coronal": float("nan"),
                     "sagittal": float("nan"), "overall": float("nan")}

    # ── Ripley's K ───────────────────────────────────────────────────────────
    ripley_w1 = None
    if run_ripley:
        try:
            kr_gt   = _ripleys_k_3d(bin_gt)
            kr_pred = _ripleys_k_3d(bin_pred)
            if kr_gt is not None and kr_pred is not None:
                ripley_w1 = float(wasserstein_distance(kr_gt[1], kr_pred[1]))
        except Exception as exc:
            logger.warning("  Ripley's K failed for %s: %s", volume_id, exc)

    elapsed = time.perf_counter() - t0
    logger.info(
        "  %s — psnr_stoch=%.2f ssim_stoch=%.4f dice=%.4f "
        "por_mae=%.4f s2_w1=%s psd_w1=%s  (%.1fs)",
        volume_id,
        psnr_s, ssim_s, seg["dice"],
        abs(por_pred - por_gt),
        f"{s2_w1:.4f}" if s2_w1 is not None else "n/a",
        f"{psd_w1:.4f}" if psd_w1 is not None else "n/a",
        elapsed,
    )

    return VolumeMetrics(
        volume_id=volume_id,
        xct_mae_stoch=mae_s, xct_psnr_stoch=psnr_s, xct_ssim_stoch=ssim_s,
        xct_mae_single=mae_1, xct_psnr_single=psnr_1, xct_ssim_single=ssim_1,
        xct_mae_mu=mae_m, xct_psnr_mu=psnr_m, xct_ssim_mu=ssim_m,
        dice=seg["dice"], precision=seg["precision"],
        recall=seg["recall"], iou=seg["iou"],
        porosity_gt=por_gt, porosity_pred=por_pred,
        porosity_mae=abs(por_pred - por_gt),
        porosity_bias=por_pred - por_gt,
        xct_recon_std_mean=xct_std_mean,
        mask_recon_std_mean=mask_std_mean,
        xct_boundary_mae=xct_bnd,
        mask_boundary_mae=mask_bnd,
        pore_count_gt=cc_gt, pore_count_pred=cc_pred,
        s2_wasserstein=s2_w1, s2_r_vals=s2_r, s2_gt=s2_g, s2_pred=s2_p,
        psd_wasserstein=psd_w1, psd_gt=psd_g, psd_pred=psd_p,
        ripley_w1=ripley_w1,
        elapsed_s=elapsed,
        sharpness_ratio_axial=sharpness.get("axial",    float("nan")),
        sharpness_ratio_coronal=sharpness.get("coronal",  float("nan")),
        sharpness_ratio_sagittal=sharpness.get("sagittal", float("nan")),
        sharpness_ratio_overall=sharpness.get("overall",  float("nan")),
    )


# ---------------------------------------------------------------------------
# Public: latent audit
# ---------------------------------------------------------------------------

@torch.no_grad()
def latent_audit(
    model: nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    eval_cfg: "EvalConfig",
) -> LatentAudit:
    """Compute per-channel posterior statistics for the LDM readiness gate.

    Encodes a random subset of ``eval_cfg.latent_patches`` test patches and
    collects per-channel KL divergence and σ = exp(0.5 × logvar) statistics.

    Parameters
    ----------
    model : nn.Module
    data_zarr_root : Path
    device : torch.device
    eval_cfg : EvalConfig

    Returns
    -------
    LatentAudit
    """
    from torch.utils.data import DataLoader, Subset

    from poregen.dataset.loader import PatchDataset, zarr_worker_init_fn

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_zarr_root, split="test")
    n = min(eval_cfg.latent_patches, len(test_ds))
    idx = random.sample(range(len(test_ds)), n)
    sub = Subset(test_ds, idx)
    loader = DataLoader(sub, batch_size=eval_cfg.batch_size, shuffle=False,
                        num_workers=2, worker_init_fn=zarr_worker_init_fn)

    all_mu:     list[torch.Tensor] = []
    all_logvar: list[torch.Tensor] = []
    all_cos_sim: list[float] = []

    model.eval()
    is_dual = hasattr(model, "encoder_a") and hasattr(model, "encoder_b")

    for batch in loader:
        xct_b = batch["xct"].to(device)
        h     = _encode(model, xct_b)
        all_mu    .append(model.to_mu    (h).cpu())
        all_logvar.append(model.to_logvar(h).cpu())

        if is_dual:
            import torch.nn.functional as _F
            h_a = model.encoder_a(xct_b)
            h_b = model.encoder_b(xct_b)
            ha_flat = h_a.reshape(h_a.shape[0], -1)
            hb_flat = h_b.reshape(h_b.shape[0], -1)
            cos = _F.cosine_similarity(ha_flat, hb_flat, dim=1)
            all_cos_sim.extend(cos.detach().cpu().tolist())

    mu_all  = torch.cat(all_mu,     dim=0).float()   # (N, C, d, h, w)
    lv_all  = torch.cat(all_logvar, dim=0).float()

    sigma2 = torch.exp(lv_all)

    # Per-channel KL: 0.5 * E[μ² + σ² − log σ² − 1]
    kl_per_ch = (0.5 * (mu_all.pow(2) + sigma2 - lv_all - 1.0)
                 ).mean(dim=(0, 2, 3, 4)).numpy()              # (C,)

    # Per-channel mean σ
    sigma_avg = sigma2.sqrt().mean(dim=(0, 2, 3, 4)).numpy()   # (C,)

    active   = int((sigma_avg > 0.1).sum())
    n_total  = int(kl_per_ch.shape[0])

    lo = eval_cfg.ldm_sigma_low
    hi = eval_cfg.ldm_sigma_high

    flagged_low  = [int(i) for i in range(n_total) if sigma_avg[i] < lo]
    flagged_high = [int(i) for i in range(n_total) if sigma_avg[i] > hi]
    in_range     = [int(i) for i in range(n_total) if lo <= sigma_avg[i] <= hi]
    ldm_ready    = (len(flagged_low) + len(flagged_high)) == 0

    branch_cos = float(np.mean(all_cos_sim)) if all_cos_sim else None

    logger.info("Latent audit: %d / %d active channels", active, n_total)
    _active_threshold = max(1, int(0.75 * n_total))  # warn if < 75% active (e.g. < 12/16)
    if active < _active_threshold:
        logger.warning(
            "  ⚠ Only %d / %d channels active (σ > 0.1) — "
            "expected ≥ %d. Posterior collapse likely; check KL weights.",
            active, n_total, _active_threshold,
        )
    logger.info("  LDM ready: %s  (σ target [%.2f, %.2f])", ldm_ready, lo, hi)
    if flagged_low:
        logger.warning("  Channels flagged LOW  (σ < %.2f): %s", lo, flagged_low)
    if flagged_high:
        logger.warning("  Channels flagged HIGH (σ > %.2f): %s", hi, flagged_high)
    if branch_cos is not None:
        logger.info("  Branch cosine similarity h_a vs h_b: %.4f", branch_cos)
    for c in range(n_total):
        logger.debug("  ch%02d  KL=%.4f  σ=%.4f", c, kl_per_ch[c], sigma_avg[c])

    return LatentAudit(
        n_active_channels=active,
        n_total_channels=n_total,
        kl_per_channel=kl_per_ch.tolist(),
        sigma_avg=sigma_avg.tolist(),
        channels_in_range=in_range,
        channels_flagged_low=flagged_low,
        channels_flagged_high=flagged_high,
        ldm_ready=ldm_ready,
        branch_cosine_sim_mean=branch_cos,
    )


# ---------------------------------------------------------------------------
# Public: FID on 2-D slices
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fid_slices(
    model: nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    eval_cfg: "EvalConfig",
) -> dict[str, float]:
    """Fréchet Inception Distance on 2-D mid-slices of 3-D test patches.

    For each of the three anatomical axes (axial, coronal, sagittal), extracts
    the mid-plane slice from every GT patch and its deterministic reconstruction
    (z = μ), replicates the grayscale channel to 3 channels, resizes to 299×299,
    and extracts InceptionV3 pool3 features.  FID is computed from the empirical
    means and covariances of the real and fake feature distributions.

    This provides a complementary diversity metric to the patch-level PSNR/SSIM:
    a model that blurs all patches will score low PSNR AND high FID.

    Parameters
    ----------
    model : nn.Module
    data_zarr_root : Path
    device : torch.device
    eval_cfg : EvalConfig
        Uses ``eval_cfg.patch_batches`` and ``eval_cfg.batch_size``.

    Returns
    -------
    dict with keys ``"axial"``, ``"coronal"``, ``"sagittal"`` (float).
    Returns NaN for any axis where feature extraction fails.
    """
    try:
        import torchvision.models as tvm
        import torch.nn.functional as F
        from scipy.linalg import sqrtm
    except ImportError:
        logger.warning("torchvision or scipy.linalg not available — skipping FID")
        return {"axial": float("nan"), "coronal": float("nan"), "sagittal": float("nan")}

    from poregen.dataset.loader import PatchDataset, zarr_worker_init_fn
    from torch.utils.data import DataLoader

    # Load InceptionV3, hook pool3 features
    inception = tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)
    inception.eval()

    # Replace the final FC so we get 2048-dim pool3 features
    # by hooking avgpool output
    features_hook: list[torch.Tensor] = []

    def _hook(module: Any, inp: Any, out: Any) -> None:
        features_hook.append(out.squeeze(-1).squeeze(-1).squeeze(-1))

    inception.avgpool.register_forward_hook(_hook)
    inception = inception.to(device)

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_zarr_root, split="test")
    loader = DataLoader(
        test_ds, batch_size=eval_cfg.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=zarr_worker_init_fn,
    )

    # Axes: (axis_index, name, slicer)
    axes = [
        (0, "axial",    lambda v: v[:, :, v.shape[2] // 2, :, :]),   # (B,1,H,W)
        (1, "coronal",  lambda v: v[:, :, :, v.shape[3] // 2, :]),
        (2, "sagittal", lambda v: v[:, :, :, :, v.shape[4] // 2]),
    ]

    real_feats: dict[str, list[torch.Tensor]] = {n: [] for _, n, _ in axes}
    fake_feats: dict[str, list[torch.Tensor]] = {n: [] for _, n, _ in axes}

    model.eval()

    for batch_idx, batch in enumerate(loader):
        if eval_cfg.patch_batches is not None and batch_idx >= eval_cfg.patch_batches:
            break

        xct_b = batch["xct"].to(device)
        h     = _encode(model, xct_b)
        mu_b  = model.to_mu(h)
        dec_b = model.decoder(mu_b)
        recon_b = model.xct_head(dec_b).clamp(0.0, 1.0)

        for _, ax_name, slicer in axes:
            for vol_b, store in [(xct_b, real_feats[ax_name]),
                                  (recon_b, fake_feats[ax_name])]:
                sl = slicer(vol_b)           # (B, 1, H, W) or similar
                # Ensure (B, 1, H, W)
                if sl.dim() == 3:
                    sl = sl.unsqueeze(1)
                rgb = sl.expand(-1, 3, -1, -1)    # (B,3,H,W)
                rgb = F.interpolate(rgb, size=(299, 299), mode="bilinear",
                                    align_corners=False)
                features_hook.clear()
                with torch.no_grad():
                    inception(rgb)
                if features_hook:
                    store.append(features_hook[-1].cpu())

    def _fid(real_list: list[torch.Tensor], fake_list: list[torch.Tensor]) -> float:
        if not real_list or not fake_list:
            return float("nan")
        try:
            r = torch.cat(real_list).numpy().astype(np.float64)
            f = torch.cat(fake_list).numpy().astype(np.float64)
            mu_r, mu_f = r.mean(0), f.mean(0)
            sig_r = np.cov(r, rowvar=False)
            sig_f = np.cov(f, rowvar=False)
            diff  = mu_r - mu_f
            covmean, _ = sqrtm(sig_r @ sig_f, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))
        except Exception as exc:
            logger.warning("FID computation failed: %s", exc)
            return float("nan")

    result: dict[str, float] = {}
    for _, ax_name, _ in axes:
        fid_val = _fid(real_feats[ax_name], fake_feats[ax_name])
        result[ax_name] = fid_val
        logger.info("  FID %s: %.2f", ax_name, fid_val)

    return result


# ---------------------------------------------------------------------------
# Public: memorization score
# ---------------------------------------------------------------------------

@torch.no_grad()
def memorization_score(
    model: nn.Module,
    data_zarr_root: Path,
    device: torch.device,
    eval_cfg: "EvalConfig",
    n_train: int = 5000,
) -> dict[str, float]:
    """Latent nearest-neighbour distance (test → train) as a memorisation proxy.

    For each test patch, finds the nearest training patch in the latent space
    (μ vectors, L2 distance).  A low mean nearest-neighbour distance indicates
    the encoder is memorising training samples rather than generalising.

    Parameters
    ----------
    model : nn.Module
    data_zarr_root : Path
    device : torch.device
    eval_cfg : EvalConfig
        Uses ``eval_cfg.latent_patches`` as the number of test patches to encode.
    n_train : int
        Number of training patches to encode (default 5000).

    Returns
    -------
    dict with ``"memorization_nn_dist_mean"`` and ``"memorization_nn_dist_std"``.
    """
    from torch.utils.data import DataLoader, Subset

    from poregen.dataset.loader import PatchDataset, zarr_worker_init_fn

    index_path = data_zarr_root / "patch_index.parquet"
    test_ds  = PatchDataset(index_path, data_zarr_root, split="test")
    train_ds = PatchDataset(index_path, data_zarr_root, split="train")

    def _encode(ds: Any, n: int) -> torch.Tensor:
        idx = random.sample(range(len(ds)), min(n, len(ds)))
        sub = Subset(ds, idx)
        loader = DataLoader(sub, batch_size=eval_cfg.batch_size, shuffle=False,
                            num_workers=2, worker_init_fn=zarr_worker_init_fn)
        mus: list[torch.Tensor] = []
        model.eval()
        for batch in loader:
            xct_b = batch["xct"].to(device)
            h     = _encode(model, xct_b)
            mu_b  = model.to_mu(h)
            mus.append(mu_b.reshape(mu_b.shape[0], -1).cpu())
        return torch.cat(mus, dim=0)

    logger.info("Memorisation: encoding %d test patches …", eval_cfg.latent_patches)
    z_test  = _encode(test_ds,  eval_cfg.latent_patches)
    logger.info("Memorisation: encoding %d train patches …", n_train)
    z_train = _encode(train_ds, n_train)

    chunk = 256
    min_dists: list[torch.Tensor] = []
    for i in range(0, len(z_test), chunk):
        zt = z_test[i:i + chunk].unsqueeze(1)          # (chunk, 1, D)
        d  = (zt - z_train.unsqueeze(0)).pow(2).sum(-1).sqrt()  # (chunk, n_train)
        min_dists.append(d.min(dim=1).values)

    all_dists = torch.cat(min_dists)
    result = {
        "memorization_nn_dist_mean": float(all_dists.mean().item()),
        "memorization_nn_dist_std":  float(all_dists.std().item()),
    }
    logger.info("  Memorisation NN dist: %.4f ± %.4f",
                result["memorization_nn_dist_mean"],
                result["memorization_nn_dist_std"])
    return result


# ---------------------------------------------------------------------------
# Volume selection helper
# ---------------------------------------------------------------------------

def select_test_volumes(data_zarr_root: Path, n_volumes: int) -> list[str]:
    """Select *n_volumes* test volumes spanning low / mid / high porosity.

    Reads the patch index parquet, groups by volume_id, sorts by mean porosity,
    and picks *n_volumes* evenly-spaced entries.

    Parameters
    ----------
    data_zarr_root : Path
    n_volumes : int

    Returns
    -------
    list of str — volume_ids
    """
    import pandas as pd

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

    positions = np.linspace(0, len(test) - 1, n_volumes, dtype=int)
    return [test.index[p] for p in positions]

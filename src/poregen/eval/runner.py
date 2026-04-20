"""Orchestration entry-point for a full PoreGen VAE evaluation run.

This module is the only place that knows how to wire together all the eval
sub-modules (stochastic, metrics, visualise, outputs).  It should be called
from the CLI / TUI integration point in ``poregen.cli.experiments``.

Typical call graph
------------------
::

    run_eval(run_dir, checkpoint_path, eval_cfg, repo_root)
      │
      ├─ _build_model_from_run(run_dir, device)
      ├─ load_checkpoint(checkpoint_path, model)
      ├─ make_eval_output_dir(run_dir, eval_cfg)
      │
      ├─ select_test_volumes(data_zarr_root, n_volumes)
      │
      ├─ [for each volume]
      │    ├─ _load_volume_arrays(volume_id, data_zarr_root, tiff_root)
      │    ├─ reconstruct_volume_three_modes(model, ...)
      │    ├─ save_volume_npz / save_volume_tiffs
      │    ├─ save_slice_comparisons / save_std_images
      │    ├─ eval_volume_metrics
      │    ├─ save_volume_metrics_json
      │    └─ save_s2r_plot / save_psd_plot / save_pore_gifs (conditional)
      │
      ├─ eval_patches                  (patch-level metrics)
      ├─ latent_audit
      ├─ compute_fid_slices            (conditional)
      ├─ memorization_score            (conditional)
      │
      ├─ save_porosity_scatter
      ├─ save_latent_audit_plot
      ├─ save_prior_samples
      │
      ├─ save_eval_metrics_json
      ├─ save_latent_audit_json
      ├─ save_patch_npz
      └─ write_readme
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import tifffile
import yaml
import zarr

from poregen.eval.config import EvalConfig
from poregen.eval.metrics import (
    LatentAudit,
    PatchMetrics,
    VolumeMetrics,
    compute_fid_slices,
    eval_patches,
    eval_volume_metrics,
    latent_audit,
    memorization_score,
    select_test_volumes,
)
from poregen.eval.outputs import (
    make_eval_output_dir,
    save_eval_metrics_json,
    save_latent_audit_json,
    save_patch_npz,
    save_volume_metrics_json,
    save_volume_npz,
    save_volume_tiffs,
    volume_out_dir,
    write_readme,
)
from poregen.eval.stochastic import VolumeReconstruction, reconstruct_volume_three_modes
from poregen.eval.visualise import (
    save_latent_audit_plot,
    save_pore_gifs,
    save_porosity_scatter,
    save_prior_samples,
    save_psd_plot,
    save_s2r_plot,
    save_slice_comparisons,
    save_std_images,
)
from poregen.models.vae.registry import build_vae
from poregen.training.checkpoint import load_checkpoint
from poregen.training.device import get_autocast_dtype
from poregen.training import build_patch_dataloaders
from poregen.training.engine import _run_eval
from poregen.losses import compute_total_loss

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model_from_run(run_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load ``resolved_config.yaml`` from *run_dir* and instantiate the VAE.

    Parameters
    ----------
    run_dir : Path
        Training run directory (must contain ``resolved_config.yaml``).
    device : torch.device

    Returns
    -------
    nn.Module
        The model in eval mode, on *device*.

    Raises
    ------
    FileNotFoundError
        If ``resolved_config.yaml`` is missing.
    KeyError
        If the config does not contain the required ``model`` section.
    """
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"resolved_config.yaml not found in {run_dir}. "
            "Make sure you are pointing at a valid training run directory."
        )

    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    model_cfg: dict = cfg.get("model", {})
    arch_name: str = model_cfg.get("name", "v2.conv_noattn")

    # Forward only fields that are valid VAEConfig fields — extra keys are ignored
    from poregen.models.vae.base import VAEConfig
    valid = {f.name for f in VAEConfig.__dataclass_fields__.values()}
    overrides = {k: v for k, v in model_cfg.items() if k in valid}

    logger.info("Building model: %s  overrides=%s", arch_name, overrides)
    model = build_vae(arch_name, **overrides).to(device)
    model.eval()
    return model


def _resolve_data_paths(
    run_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path]:
    """Resolve the Zarr root and TIFF root from the run's resolved config.

    Returns
    -------
    zarr_root : Path
        Directory containing ``volumes.zarr`` and ``patch_index.parquet``.
    tiff_root : Path
        ``<repo_root>/raw_data`` — directory containing the raw TIFF files
        (source-of-truth for volume reconstruction).
    """
    cfg_path = run_dir / "resolved_config.yaml"
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    dataset_root: str = cfg.get("data", {}).get("dataset_root", "split_v2")
    candidate = Path(dataset_root)
    if candidate.is_absolute():
        zarr_root = candidate.resolve()
    else:
        # Training code prepends repo_root/data/ for relative paths
        zarr_root = (repo_root / "data" / dataset_root).resolve()
    tiff_root = repo_root / "raw_data"

    if not zarr_root.exists():
        raise FileNotFoundError(
            f"Zarr data root not found: {zarr_root}. "
            f"Resolved from resolved_config.yaml data.dataset_root={dataset_root!r} "
            f"under repo_root/data/. Check that the split dataset exists."
        )

    return zarr_root, tiff_root


def _tiff_path_for_volume(volume_id: str, tiff_root: Path) -> Path:
    """Map a volume_id to its source TIFF path.

    Convention: ``<source_group>__<relative_stem>``
    e.g.  ``MedidasDB__Panel_A_volume``
    →     ``<tiff_root>/MedidasDB/Panel_A_volume.tif``
    """
    rel = volume_id.replace("__", "/")
    for ext in (".tif", ".tiff"):
        p = tiff_root / (rel + ext)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"TIFF not found for volume_id={volume_id!r} "
        f"(tried {tiff_root / rel}.tif and .tiff)"
    )


def _load_volume_arrays(
    volume_id: str,
    zarr_root: Path,
    tiff_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load XCT (from TIFF) and mask (from Zarr) for one volume.

    Returns
    -------
    xct_u8  : uint8, (D, H, W) — raw XCT intensities
    mask_u8 : uint8, (D, H, W) — binary pore mask {0, 1}
    """
    # XCT from authoritative TIFF source
    tiff_p = _tiff_path_for_volume(volume_id, tiff_root)
    logger.info("  Loading TIFF: %s", tiff_p)
    xct_u8 = tifffile.imread(str(tiff_p))
    if xct_u8.ndim != 3:
        raise ValueError(f"Expected 3-D TIFF, got shape {xct_u8.shape}")
    xct_u8 = xct_u8.astype(np.uint8, copy=False)

    # Mask from Zarr (Sauvola-segmented at dataset build time)
    zarr_path = zarr_root / "volumes.zarr"
    zarr_grp = zarr.open_group(str(zarr_path), mode="r")
    mask_u8 = np.array(zarr_grp[volume_id]["mask"]).astype(np.uint8, copy=False)

    return xct_u8, mask_u8


def _short_name(volume_id: str) -> str:
    """Strip the MedidasDB__ prefix used in volume identifiers."""
    return volume_id.replace("MedidasDB__", "")


# ---------------------------------------------------------------------------
# Split-level loss evaluation
# ---------------------------------------------------------------------------

def _eval_split_losses(
    run_dir: Path,
    model: torch.nn.Module,
    step: int,
    device: torch.device,
    repo_root: Path,
) -> dict[str, dict[str, float]]:
    """Run a full forward pass (with losses) over the complete val and test splits.

    Uses the same ``_run_eval`` path as the training engine so that every loss
    term (ELBO, KL, perceptual, adversarial, segmentation, porosity, latent
    stats) is computed identically to what is logged during training.

    Returns
    -------
    dict with keys ``"val"`` and ``"test"``, each mapping metric name → float.
    """
    cfg_path = run_dir / "resolved_config.yaml"
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)

    dataset_root = cfg.get("data", {}).get("dataset_root", "split_v2")
    candidate = Path(dataset_root)
    if candidate.is_absolute():
        data_root = candidate.resolve()
    else:
        data_root = (repo_root / "data" / dataset_root).resolve()

    _, val_loader, test_loader = build_patch_dataloaders(cfg, data_root)
    autocast_dtype = get_autocast_dtype(device)
    loss_fn = lambda output, batch, s: compute_total_loss(output, batch, s, cfg)  # noqa: E731

    results: dict[str, dict[str, float]] = {}

    logger.info("── Full val split (losses + metrics): %d batches ──", len(val_loader))
    val_agg, _ = _run_eval(
        model, iter(val_loader), loss_fn,
        len(val_loader), step, device, autocast_dtype,
        desc="Full val",
    )
    results["val"] = {k: v for k, v in val_agg.items() if not isinstance(v, list)}

    logger.info("── Full test split (losses + metrics): %d batches ──", len(test_loader))
    test_agg, _ = _run_eval(
        model, iter(test_loader), loss_fn,
        len(test_loader), step, device, autocast_dtype,
        desc="Full test",
    )
    results["test"] = {k: v for k, v in test_agg.items() if not isinstance(v, list)}

    return results


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_eval(
    run_dir: Path,
    checkpoint_path: Path,
    eval_cfg: EvalConfig,
    repo_root: Path,
    *,
    device: torch.device | None = None,
) -> Path:
    """Run a complete evaluation pipeline for one checkpoint.

    This function orchestrates every step in the evaluation pipeline —
    model loading, volume reconstruction, metric computation, and saving all
    output artefacts — and returns the path of the newly created eval output
    directory.

    The function is designed to be idempotent: run it twice and a second
    timestamped output directory is created without overwriting the first.

    Parameters
    ----------
    run_dir : Path
        Training run directory.  Must contain ``resolved_config.yaml`` and
        the nominated checkpoint.
    checkpoint_path : Path
        Explicit path to the ``.ckpt`` file to evaluate.  Usually
        ``run_dir/best.ckpt`` or ``run_dir/latest.ckpt``.
    eval_cfg : EvalConfig
        Evaluation configuration.  Controls which metrics are enabled, how
        many volumes to reconstruct, etc.
    repo_root : Path
        Repository root used to resolve relative data paths.
    device : torch.device, optional
        The device to run inference on.  If ``None``, GPU 0 is selected when
        available, otherwise CPU.

    Returns
    -------
    Path
        The evaluation output directory (``<run_dir>/eval/<timestamp>-<tier>/``).
    """
    t0 = time.perf_counter()
    run_name = run_dir.name

    # ── Device ─────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("eval device: %s", device)

    # ── Model + checkpoint ─────────────────────────────────────────────────
    logger.info("Building model from %s", run_dir)
    model = _build_model_from_run(run_dir, device)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    step, meta = load_checkpoint(str(checkpoint_path), model, restore_rng=False,
                                 map_location=device)
    model.eval()
    logger.info("Checkpoint loaded (step=%d)", step)

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = make_eval_output_dir(run_dir, eval_cfg)
    logger.info("Eval output: %s", out_dir)

    # ── Data paths ─────────────────────────────────────────────────────────
    zarr_root, tiff_root = _resolve_data_paths(run_dir, repo_root)
    logger.info("Zarr root: %s", zarr_root)
    logger.info("TIFF root: %s", tiff_root)

    # ── Select test volumes ─────────────────────────────────────────────────
    vol_ids = select_test_volumes(zarr_root, eval_cfg.n_volumes)
    logger.info("Selected %d test volumes: %s", len(vol_ids), vol_ids)

    # Infer z_channels and spatial_latent from the resolved config for prior samples
    _cfg_path = run_dir / "resolved_config.yaml"
    with _cfg_path.open() as _fh:
        _raw_cfg = yaml.safe_load(_fh)
    _model_cfg = _raw_cfg.get("model", {})
    z_channels     = int(_model_cfg.get("z_channels", 16))
    patch_size      = int(_model_cfg.get("patch_size", 64))
    n_blocks        = int(_model_cfg.get("n_blocks", 2))
    spatial_latent  = patch_size // (2 ** n_blocks)

    # ══════════════════════════════════════════════════════════════════════════
    # Volume-level reconstruction and metrics
    # ══════════════════════════════════════════════════════════════════════════
    all_vol_metrics: list[VolumeMetrics] = []
    all_vol_names:   list[str]           = []

    if eval_cfg.run_tiff_reconstruction:
        for vol_id in vol_ids:
            vol_name = _short_name(vol_id)
            logger.info("\n── Volume: %s ──", vol_name)
            t_vol = time.perf_counter()

            # Load raw arrays
            try:
                xct_u8, mask_u8 = _load_volume_arrays(vol_id, zarr_root, tiff_root)
            except FileNotFoundError as exc:
                logger.error("  Volume load failed: %s — skipping", exc)
                continue
            except Exception as exc:
                logger.error("  Unexpected error loading %s: %s — skipping", vol_id, exc)
                continue

            # Reconstruct in all three modes
            logger.info("  Reconstructing (N=%d stochastic samples) …",
                        eval_cfg.n_stochastic_samples)
            vol = reconstruct_volume_three_modes(
                model, vol_id, xct_u8, mask_u8, device,
                n_samples=eval_cfg.n_stochastic_samples,
                seed=eval_cfg.stochastic_seed,
            )
            logger.info("  Reconstruction done in %.1fs",
                        time.perf_counter() - t_vol)

            # Save arrays and TIFFs
            vdir = volume_out_dir(out_dir, vol_name)
            save_volume_npz(vol, vol_name, out_dir)
            save_volume_tiffs(vol, vol_name, out_dir)

            # Volume-level metrics
            logger.info("  Computing volume metrics …")
            vm = eval_volume_metrics(
                vol_id, vol,
                r_max=eval_cfg.r_max,
                run_s2r=eval_cfg.run_s2r,
                run_psd=eval_cfg.run_psd,
                run_ripley=eval_cfg.run_ripley,
            )
            all_vol_metrics.append(vm)
            all_vol_names.append(vol_name)
            save_volume_metrics_json(vm, vol_name, out_dir)

            # Slice comparisons and uncertainty images
            logger.info("  Saving slice comparison figures …")
            save_slice_comparisons(vol, vdir)
            save_std_images(vol, vdir)

            # Optional structural plots
            if eval_cfg.run_s2r:
                save_s2r_plot(vm, vdir)
            if eval_cfg.run_psd:
                save_psd_plot(vm, vdir)
            if eval_cfg.run_pore_gifs:
                logger.info("  Generating 3-D pore GIFs …")
                save_pore_gifs(vol, vdir)

            elapsed_vol = time.perf_counter() - t_vol
            logger.info(
                "  Volume done in %.0fs  psnr_stoch=%.2f  dice=%.4f  ldm_std=%.5f",
                elapsed_vol, vm.xct_psnr_stoch, vm.dice, vm.xct_recon_std_mean,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Full val + test split losses (all training loss terms)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Full val/test split losses ──")
    split_losses: dict[str, dict[str, float]] = {}
    try:
        split_losses = _eval_split_losses(run_dir, model, step, device, repo_root)
        (out_dir / "split_losses.json").write_text(
            json.dumps({"step": step, "splits": split_losses}, indent=2)
        )
        logger.info(
            "  val total=%.4f  test total=%.4f",
            split_losses.get("val", {}).get("total", float("nan")),
            split_losses.get("test", {}).get("total", float("nan")),
        )
    except Exception as exc:
        logger.error("  Split loss eval failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # Patch-level metrics (full test set)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Patch metrics (full test set) ──")
    try:
        pm = eval_patches(model, zarr_root, device, eval_cfg)
    except Exception as exc:
        logger.error("  Patch eval failed: %s", exc)
        # Create an empty fallback so downstream code doesn't break
        from poregen.eval.metrics import Stat
        pm = PatchMetrics(
            n_patches=0,
            psnr=Stat(float("nan"), float("nan")),
            ssim=Stat(float("nan"), float("nan")),
            mae=Stat(float("nan"), float("nan")),
            dice=Stat(float("nan"), float("nan")),
            precision=Stat(float("nan"), float("nan")),
            recall=Stat(float("nan"), float("nan")),
            f1=Stat(float("nan"), float("nan")),
            porosity_mae=Stat(float("nan"), float("nan")),
            porosity_bias=Stat(float("nan"), float("nan")),
            porosity_volume_mae=Stat(float("nan"), float("nan")),
            sharpness_ratio=Stat(float("nan"), float("nan")),
            recon_std_mean=float("nan"),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Latent audit
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Latent audit ──")
    try:
        la = latent_audit(model, zarr_root, device, eval_cfg)
    except Exception as exc:
        logger.error("  Latent audit failed: %s", exc)
        la = LatentAudit(
            n_active_channels=0,
            n_total_channels=z_channels,
            kl_per_channel=[float("nan")] * z_channels,
            sigma_avg=[float("nan")] * z_channels,
            channels_in_range=[],
            channels_flagged_low=list(range(z_channels)),
            channels_flagged_high=[],
            ldm_ready=False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Optional: FID
    # ══════════════════════════════════════════════════════════════════════════
    fid_scores: dict[str, float] = {}
    if eval_cfg.run_fid:
        logger.info("\n── FID ──")
        try:
            fid_scores = compute_fid_slices(model, zarr_root, device, eval_cfg)
            logger.info("  FID axial=%.2f  coronal=%.2f  sagittal=%.2f",
                        fid_scores.get("axial", float("nan")),
                        fid_scores.get("coronal", float("nan")),
                        fid_scores.get("sagittal", float("nan")))
        except Exception as exc:
            logger.error("  FID failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # Optional: memorization score
    # ══════════════════════════════════════════════════════════════════════════
    mem_scores: dict = {}
    if eval_cfg.run_memorization:
        logger.info("\n── Memorization ──")
        try:
            mem_scores = memorization_score(model, zarr_root, device, eval_cfg)
            logger.info("  Memorization NN dist = %.4f",
                        mem_scores.get("nn_dist_mean", float("nan")))
        except Exception as exc:
            logger.error("  Memorization failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # Root-level visualisations
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Root visualisations ──")

    try:
        save_porosity_scatter(pm, out_dir)
    except Exception as exc:
        logger.warning("  Porosity scatter failed: %s", exc)

    try:
        save_latent_audit_plot(
            la, out_dir,
            ldm_sigma_low=eval_cfg.ldm_sigma_low,
            ldm_sigma_high=eval_cfg.ldm_sigma_high,
        )
    except Exception as exc:
        logger.warning("  Latent audit plot failed: %s", exc)

    try:
        save_prior_samples(
            model, device, out_dir,
            z_channels=z_channels,
            spatial_latent=spatial_latent,
            figure_title="Prior samples  z ~ N(0, I)  —  GAN stability check",
        )
    except Exception as exc:
        logger.warning("  Prior samples failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # JSON outputs
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n── Saving JSON outputs ──")
    save_eval_metrics_json(
        all_vol_metrics, pm, la, eval_cfg,
        step=step,
        run_name=run_name,
        checkpoint_path=str(checkpoint_path),
        out_dir=out_dir,
    )
    save_latent_audit_json(la, out_dir)
    save_patch_npz(pm, out_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # README
    # ══════════════════════════════════════════════════════════════════════════
    elapsed_s = time.perf_counter() - t0
    write_readme(
        out_dir,
        run_name=run_name,
        checkpoint_path=str(checkpoint_path),
        eval_cfg=eval_cfg,
        step=step,
        elapsed_s=elapsed_s,
        vol_names=all_vol_names,
        repo_root=repo_root,
    )

    logger.info(
        "\nEVAL COMPLETE  step=%d  volumes=%d  elapsed=%.0fs (%.1fmin)  out=%s",
        step, len(all_vol_metrics), elapsed_s, elapsed_s / 60, out_dir,
    )
    logger.info(
        "NOTE: PSNR/SSIM may be lower with adversarial loss — "
        "check sharpness_ratio_* fields instead of PSNR/SSIM for R04."
    )

    return out_dir

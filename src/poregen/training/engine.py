"""Train / eval step helpers and the main training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from poregen.models.vae.base import VAEOutput
from poregen.metrics.seg import segmentation_metrics, porosity_metrics
from poregen.metrics.recon import mae, psnr, sharpness_proxy
from poregen.metrics.latent import active_units, latent_stats
from poregen.training.checkpoint import save_checkpoint


# ── helpers ──────────────────────────────────────────────────────────────

def _to_scalar(v: Any) -> Any:
    """Convert a tensor to a Python scalar or list; leave floats/ints as-is."""
    if isinstance(v, torch.Tensor):
        return v.detach().tolist() if v.numel() > 1 else v.detach().item()
    return float(v)


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _central_slice_d(vol: torch.Tensor) -> torch.Tensor:
    """Extract the central D-axis slice → (B, 1, H, W)."""
    return vol[:, :, vol.shape[2] // 2, :, :]


def _central_slice_h(vol: torch.Tensor) -> torch.Tensor:
    """Extract the central H-axis slice → (B, 1, D, W)."""
    return vol[:, :, :, vol.shape[3] // 2, :]


def _central_slice_w(vol: torch.Tensor) -> torch.Tensor:
    """Extract the central W-axis slice → (B, 1, D, H)."""
    return vol[:, :, :, :, vol.shape[4] // 2]


def _accumulate(acc: dict, new: dict) -> None:
    """Add new scalar/list values into an accumulator dict."""
    for k, v in new.items():
        if isinstance(v, list):
            acc.setdefault(k, []).extend(v)
        else:
            acc.setdefault(k, 0.0)
            acc[k] += float(v)


def _mean_acc(acc: dict, n: int) -> dict[str, float]:
    """Average an accumulator dict over n batches (scalars only)."""
    out = {}
    for k, v in acc.items():
        if isinstance(v, list):
            mean_val = sum(v) / len(v) if v else 0.0
            out[k] = mean_val
        else:
            out[k] = v / n
    return out


# ── single-step helpers ──────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: Callable[..., dict[str, Any]],
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float16,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
) -> tuple[dict[str, Any], float]:
    """Single training step with AMP, optional gradient clipping, and scheduler.

    Returns
    -------
    losses : dict
        Per-component loss values. Scalars are Python floats; ``kl_per_channel``
        is a list of length C.
    grad_norm : float
        Global gradient norm (after unscaling, before clipping). 0.0 if
        max_grad_norm is None.
    """
    model.train()
    xct = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)

    scaler.scale(losses["total"]).backward()

    grad_norm = 0.0
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).item()

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return {k: _to_scalar(v) for k, v in losses.items()}, grad_norm


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: Callable[..., dict[str, Any]],
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, Any], VAEOutput]:
    """Single eval step (no grad, AMP for speed).

    Returns the loss dict (scalars + kl_per_channel list) and the VAEOutput.
    """
    model.eval()
    xct = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)

    return {k: _to_scalar(v) for k, v in losses.items()}, output


# ── eval-over-N-batches helper ────────────────────────────────────────────

def _run_eval(
    model: nn.Module,
    data_iter: Iterator,
    loss_fn: Callable,
    n_batches: int,
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> tuple[dict[str, float], VAEOutput]:
    """Run eval over *n_batches* batches; return aggregated metrics and the
    last VAEOutput (for image logging).
    """
    loss_acc:   dict[str, Any] = {}
    seg_acc:    dict[str, Any] = {}
    recon_acc:  dict[str, Any] = {}
    latent_acc: dict[str, Any] = {}
    por_acc:    dict[str, Any] = {}
    # per-volume porosity tracking for histogram logging
    # Stores signed errors (pred - gt) per patch per volume.
    # Histogram uses |mean(signed)| = volume-level porosity MAE estimate.
    vol_por_errors: dict[str, list[float]] = {}
    last_output: VAEOutput | None = None
    last_batch:  dict | None = None

    for _ in range(n_batches):
        batch = next(data_iter)
        losses, output = eval_step(
            model, batch, loss_fn, step, device, autocast_dtype
        )
        _accumulate(loss_acc, losses)

        mask_dev = batch["mask"].to(device, non_blocking=True)
        xct_dev  = batch["xct"].to(device,  non_blocking=True)

        # Segmentation metrics
        seg = segmentation_metrics(output.mask_logits, mask_dev)
        _accumulate(seg_acc, seg)

        # Porosity metrics (primary success metric + collapse detection)
        por = porosity_metrics(output.mask_logits, mask_dev)
        _accumulate(por_acc, por)

        # Per-volume porosity tracking
        pred_por_v = torch.sigmoid(output.mask_logits).mean(dim=(1, 2, 3, 4))
        gt_por_v   = mask_dev.mean(dim=(1, 2, 3, 4))
        for i, vid in enumerate(batch["volume_id"]):
            vol_por_errors.setdefault(vid, []).append(
                (pred_por_v[i] - gt_por_v[i]).item()  # signed; histogram uses |mean|
            )

        # Reconstruction metrics — logits and GT are both in [0, 1] (uint8/255)
        recon = {
            "mae":             mae(output.xct_logits, xct_dev).item(),
            "psnr":            psnr(output.xct_logits, xct_dev).item(),
            "sharpness_recon": sharpness_proxy(output.xct_logits).item(),
            "sharpness_gt":    sharpness_proxy(xct_dev).item(),
            "recon_xct_mean":  output.xct_logits.mean().item(),
            "recon_xct_std":   output.xct_logits.std().item(),
        }
        _accumulate(recon_acc, recon)

        # Latent metrics
        lat = {**active_units(output.mu), **latent_stats(output.mu, output.logvar)}
        _accumulate(latent_acc, lat)

        last_output = output
        last_batch  = batch

    agg = {
        **_mean_acc(loss_acc, n_batches),
        **_mean_acc(seg_acc, n_batches),
        **_mean_acc(por_acc, n_batches),
        **_mean_acc(recon_acc, n_batches),
        **_mean_acc(latent_acc, n_batches),
    }
    # kl_total: sum of raw per-channel KL (collapse → 0)
    if "kl_per_channel" in agg and isinstance(agg["kl_per_channel"], list):
        agg["kl_total"] = sum(agg["kl_per_channel"])

    return agg, last_output, last_batch, vol_por_errors  # type: ignore[return-value]


# ── training loop ─────────────────────────────────────────────────────────

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: Callable,
    *,
    total_steps: int = 200,
    log_every: int = 10,
    eval_every: int = 50,
    val_batches: int = 20,
    test_loader: DataLoader | None = None,
    test_every: int = 5000,
    test_batches: int = 20,
    save_every: int = 100,
    image_log_every: int = 500,
    sample_every: int = 0,
    n_patch_samples: int = 8,
    run_dir: str | Path = "runs/vae/default",
    device: torch.device = torch.device("cpu"),
    autocast_dtype: torch.dtype = torch.float16,
    start_step: int = 0,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
    tb_writer: Any | None = None,
) -> list[dict[str, Any]]:
    """Training loop with full real-time TensorBoard monitoring.

    Parameters
    ----------
    log_every : int
        Steps between TensorBoard scalar writes (train losses, β, grad_norm,
        per-channel KL).
    eval_every : int
        Steps between validation runs.
    val_batches : int
        Number of val batches to aggregate per eval.
    test_loader : DataLoader, optional
        If provided, a test-set evaluation is run every *test_every* steps.
    test_every : int
        Steps between test-set evaluations.
    test_batches : int
        Number of test batches per test evaluation.
    image_log_every : int
        Steps between reconstruction image logs to TensorBoard.
    sample_every : int
        Steps between full 3-D patch sample saves to disk (0 = disabled).
        Saves ``n_patch_samples`` patches per split to
        ``run_dir/samples/step_XXXXXXXX/{train,val,test}.npz``.
    n_patch_samples : int
        Number of patches to save per split at each sample checkpoint.
    tb_writer : SummaryWriter, optional
        TensorBoard writer. All monitoring is skipped if None.

    Returns
    -------
    list of per-step metric dicts (train + val records, for inline plotting).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    log_file = open(log_path, "a")
    metrics_file = open(metrics_path, "a")

    train_iter = _infinite(train_loader)
    val_iter = _infinite(val_loader) if val_loader is not None else None
    test_iter = _infinite(test_loader) if test_loader is not None else None

    history: list[dict[str, Any]] = []
    t0 = time.time()

    pbar = tqdm(range(start_step, start_step + total_steps), desc="Training")

    for step in pbar:
        # ── train step ────────────────────────────────────────────────
        batch = next(train_iter)
        losses, grad_norm = train_step(
            model, batch, optimizer, scaler, loss_fn,
            step=step, device=device, autocast_dtype=autocast_dtype,
            max_grad_norm=max_grad_norm, scheduler=scheduler,
        )

        record = {"step": step, "split": "train", "elapsed": time.time() - t0, **{
            k: v for k, v in losses.items() if not isinstance(v, list)
        }}
        history.append(record)
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        pbar.set_postfix(
            loss=f"{losses['total']:.4f}",
            kl=f"{losses.get('kl', 0):.4f}",
            β=f"{losses.get('beta', 0):.4f}",
        )

        # ── TensorBoard: train scalars (every log_every steps) ────────
        if tb_writer is not None and (step + 1) % log_every == 0:
            for k, v in losses.items():
                if isinstance(v, list):
                    for i, ch_val in enumerate(v):
                        tb_writer.add_scalar(f"train/kl_ch{i:02d}", ch_val, step)
                else:
                    tb_writer.add_scalar(f"train/{k}", v, step)
            tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            if scheduler is not None:
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            # active_channels: channels with raw KL > free_bits (collapse detector)
            kl_chs = losses.get("kl_per_channel", [])
            if kl_chs:
                tb_writer.add_histogram("train/kl_per_channel", torch.tensor(kl_chs), step)
                if "freebits_used" in losses:
                    n_dead = round(losses["freebits_used"] * len(kl_chs))
                    tb_writer.add_scalar("train/active_channels", len(kl_chs) - n_dead, step)

        # ── validation ────────────────────────────────────────────────
        if val_iter is not None and (step + 1) % eval_every == 0:
            agg, last_output, last_batch, _vol_por = _run_eval(
                model, val_iter, loss_fn, val_batches, step, device, autocast_dtype
            )
            val_record = {"step": step, "split": "val", "elapsed": time.time() - t0, **{
                k: v for k, v in agg.items() if not isinstance(v, list)
            }}
            history.append(val_record)
            log_file.write(json.dumps(val_record) + "\n")
            log_file.flush()
            metrics_file.write(json.dumps(val_record) + "\n")
            metrics_file.flush()

            if tb_writer is not None:
                for k, v in agg.items():
                    if isinstance(v, list):
                        for i, ch_val in enumerate(v):
                            tb_writer.add_scalar(f"val/kl_ch{i:02d}", ch_val, step)
                    else:
                        tb_writer.add_scalar(f"val/{k}", v, step)
                # per-channel KL histogram + kl_total (collapse detector)
                if "kl_per_channel" in agg and isinstance(agg["kl_per_channel"], list):
                    kl_tensor = torch.tensor(agg["kl_per_channel"])
                    tb_writer.add_histogram("val/kl_per_channel", kl_tensor, step)

        # ── TensorBoard: reconstruction images + MC uncertainty ──────
        if (
            tb_writer is not None
            and val_iter is not None
            and (step + 1) % image_log_every == 0
        ):
            _log_recon_images(tb_writer, last_output, last_batch, step, "val", device)
            run_montecarlo_eval(
                model, last_batch, step, device, tb_writer,
                autocast_dtype=autocast_dtype,
            )

        # ── test evaluation ───────────────────────────────────────────
        if test_iter is not None and (step + 1) % test_every == 0:
            test_agg, test_output, test_batch, test_vol_por = _run_eval(
                model, test_iter, loss_fn, test_batches, step, device, autocast_dtype
            )
            test_record = {"step": step, "split": "test", "elapsed": time.time() - t0, **{
                k: v for k, v in test_agg.items() if not isinstance(v, list)
            }}
            metrics_file.write(json.dumps(test_record) + "\n")
            metrics_file.flush()

            if tb_writer is not None:
                for k, v in test_agg.items():
                    if isinstance(v, list):
                        for i, ch_val in enumerate(v):
                            tb_writer.add_scalar(f"test/kl_ch{i:02d}", ch_val, step)
                    else:
                        tb_writer.add_scalar(f"test/{k}", v, step)
                _log_recon_images(tb_writer, test_output, test_batch, step, "test", device)
                # Per-volume porosity MAE histogram.
                # |mean(signed_errors)| = |mean_pred_por - mean_gt_por| per volume.
                if test_vol_por:
                    per_vol_maes = torch.tensor(
                        [abs(sum(errs) / len(errs)) for errs in test_vol_por.values()]
                    )
                    tb_writer.add_histogram("test/porosity_mae_per_volume", per_vol_maes, step)

        # ── checkpoint ───────────────────────────────────────────────
        if (step + 1) % save_every == 0:
            ckpt_name = f"{run_dir.name}_step{step + 1:08d}.ckpt"
            save_checkpoint(
                run_dir / ckpt_name,
                model, optimizer, scaler, step=step + 1,
                metadata={"total_steps": total_steps},
                scheduler=scheduler,
            )

        # ── 3-D patch samples ─────────────────────────────────────────
        if sample_every > 0 and (step + 1) % sample_every == 0:
            _save_patch_samples(
                model,
                {"train": train_iter, "val": val_iter, "test": test_iter},
                n_patch_samples,
                step + 1,
                run_dir,
                device,
                autocast_dtype,
            )

    # Final checkpoint
    final_step = start_step + total_steps
    ckpt_name = f"{run_dir.name}_step{final_step:08d}.ckpt"
    save_checkpoint(
        run_dir / ckpt_name,
        model, optimizer, scaler, step=final_step,
        metadata={"total_steps": total_steps},
        scheduler=scheduler,
    )

    log_file.close()
    metrics_file.close()
    return history


# ── patch sample saving ──────────────────────────────────────────────────

@torch.no_grad()
def _save_patch_samples(
    model: nn.Module,
    iters: dict[str, Iterator | None],
    n_samples: int,
    step: int,
    run_dir: Path,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> None:
    """Save full 3-D patch reconstructions to disk for later visualisation.

    Writes ``run_dir/samples/step_XXXXXXXX/{split}.npz`` and a companion
    ``{split}_meta.json`` for each split.  Arrays inside the npz:

    - ``xct_gt``    (N, 1, 64, 64, 64) float32  — normalised input
    - ``mask_gt``   (N, 1, 64, 64, 64) float32  — ground-truth mask
    - ``xct_recon`` (N, 1, 64, 64, 64) float32  — sigmoid(xct_logits)
    - ``mask_recon``(N, 1, 64, 64, 64) float32  — sigmoid(mask_logits)
    """
    import numpy as np

    samples_dir = run_dir / "samples" / f"step_{step:08d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for split, data_iter in iters.items():
        if data_iter is None:
            continue

        xct_gts, mask_gts, xct_recons, mask_recons, metas = [], [], [], [], []
        collected = 0

        while collected < n_samples:
            batch = next(data_iter)
            n_take = min(n_samples - collected, batch["xct"].shape[0])

            xct  = batch["xct"] [:n_take].to(device)
            mask = batch["mask"][:n_take].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(xct, mask)

            xct_gts.append(xct.cpu().float().numpy())
            mask_gts.append(mask.cpu().float().numpy())
            xct_recons.append(torch.sigmoid(output.xct_logits) .cpu().float().numpy()[:n_take])
            mask_recons.append(torch.sigmoid(output.mask_logits).cpu().float().numpy()[:n_take])

            for i in range(n_take):
                coords = batch["coords"]
                metas.append({
                    "volume_id":   batch["volume_id"][i],
                    "z0": int(coords[0][i]), "y0": int(coords[1][i]), "x0": int(coords[2][i]),
                    "porosity":    float(batch["porosity"][i]),
                    "source_group": batch["source_group"][i],
                })

            collected += n_take

        np.savez_compressed(
            samples_dir / f"{split}.npz",
            xct_gt    = np.concatenate(xct_gts),
            mask_gt   = np.concatenate(mask_gts),
            xct_recon = np.concatenate(xct_recons),
            mask_recon= np.concatenate(mask_recons),
        )
        with open(samples_dir / f"{split}_meta.json", "w") as f:
            json.dump(metas, f, indent=2)

    model.train()


# ── image logging helper ──────────────────────────────────────────────────

def _log_recon_images(
    tb_writer: Any,
    output: VAEOutput,
    batch: dict,
    step: int,
    prefix: str,
    device: torch.device,
) -> None:
    """Log central slices along all 3 axes to TensorBoard.

    XCT logits and GT are both in [0, 1] (uint8/255 normalisation).
    We clamp the logits to [0, 1] for display — no further normalisation needed.
    Mask reconstruction is passed through sigmoid → [0, 1].
    """
    with torch.no_grad():
        xct_recon  = output.xct_logits.clamp(0.0, 1.0)
        mask_recon = torch.sigmoid(output.mask_logits)
        xct_gt  = batch["xct"].to(device, non_blocking=True)
        mask_gt = batch["mask"].to(device, non_blocking=True)

        for tag, gt_vol, recon_vol in [
            ("xct",  xct_gt,  xct_recon),
            ("mask", mask_gt, mask_recon),
        ]:
            for axis, slicer in [
                ("d", _central_slice_d),
                ("h", _central_slice_h),
                ("w", _central_slice_w),
            ]:
                tb_writer.add_images(f"{prefix}/{tag}_gt_{axis}",    slicer(gt_vol),    step)
                tb_writer.add_images(f"{prefix}/{tag}_recon_{axis}", slicer(recon_vol), step)


# ── Monte Carlo uncertainty estimation ───────────────────────────────────

def run_montecarlo_eval(
    model: nn.Module,
    batch: dict,
    step: int,
    device: torch.device,
    writer: Any,
    n_samples: int = 30,
    autocast_dtype: torch.dtype = torch.float16,
    cmap_name: str = "plasma",
) -> None:
    """Run N stochastic forward passes and log mean/uncertainty to TensorBoard.

    The model is set to eval() mode so BatchNorm/Dropout behave deterministically,
    but the VAE reparameterization samples a different z each pass (the randomness
    comes from ``torch.randn_like`` inside ``_reparameterize``, which is unaffected
    by eval() or no_grad()).

    Logs per axis (d=axial, h=coronal, w=sagittal):
      montecarlo/xct_mean_{axis}   — mean reconstruction (grayscale)
      montecarlo/xct_std_{axis}    — voxel-wise std       (colormap)
      montecarlo/mask_mean_{axis}  — mean mask sigmoid    (grayscale)
      montecarlo/mask_std_{axis}   — voxel-wise std       (colormap)

    Parameters
    ----------
    model : nn.Module
        VAE model (will be temporarily set to eval()).
    batch : dict
        A single batch dict with ``"xct"`` and ``"mask"`` keys.
    step : int
        Global training step (TensorBoard x-axis).
    device : torch.device
    writer : SummaryWriter
        TensorBoard writer.
    n_samples : int
        Number of stochastic forward passes.
    autocast_dtype : torch.dtype
        Dtype for AMP autocast (match the training loop).
    cmap_name : str
        Matplotlib colormap for std maps (e.g. 'plasma', 'inferno', 'hot').
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    xct  = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    xct_samples  = []
    mask_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(xct, mask)
            xct_samples.append(output.xct_logits.clamp(0.0, 1.0).float())
            mask_samples.append(torch.sigmoid(output.mask_logits).float())

    # Stack → (N, B, 1, D, H, W)
    xct_stack  = torch.stack(xct_samples,  dim=0)
    mask_stack = torch.stack(mask_samples, dim=0)

    xct_mean = xct_stack.mean(dim=0)   # (B, 1, D, H, W)
    xct_std  = xct_stack.std(dim=0)
    mask_mean = mask_stack.mean(dim=0)
    mask_std  = mask_stack.std(dim=0)

    # ── Diversity check: warn if predictions are not distinct ─────────────
    _DIVERSITY_EPS = 1e-5
    xct_div  = xct_std.mean().item()
    mask_div = mask_std.mean().item()
    if xct_div < _DIVERSITY_EPS or mask_div < _DIVERSITY_EPS:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "run_montecarlo_eval step=%d: predictions are nearly identical "
            "(xct_div=%.2e  mask_div=%.2e). "
            "Check reparameterization / posterior collapse.",
            step, xct_div, mask_div,
        )
    writer.add_scalar("montecarlo/xct_diversity",  xct_div,  step)
    writer.add_scalar("montecarlo/mask_diversity", mask_div, step)

    cmap = plt.get_cmap(cmap_name)

    def _std_to_rgb(std_slice: torch.Tensor) -> torch.Tensor:
        """Convert a (B, 1, H, W) std map to (B, 3, H, W) RGB via colormap."""
        arr = std_slice.squeeze(1).cpu().numpy()          # (B, H, W)
        vmax = arr.max() if arr.max() > 0 else 1.0
        arr_norm = arr / vmax                             # normalise to [0, 1]
        # apply colormap: returns (B, H, W, 4) RGBA float
        rgba = np.stack([cmap(arr_norm[b]) for b in range(arr_norm.shape[0])])
        rgb  = rgba[..., :3]                              # drop alpha → (B, H, W, 3)
        # TensorBoard wants (B, 3, H, W)
        return torch.from_numpy(rgb.transpose(0, 3, 1, 2).astype(np.float32))

    for axis, slicer in [
        ("d", _central_slice_d),
        ("h", _central_slice_h),
        ("w", _central_slice_w),
    ]:
        writer.add_images(f"montecarlo/xct_mean_{axis}",  slicer(xct_mean),  step)
        writer.add_images(f"montecarlo/mask_mean_{axis}", slicer(mask_mean), step)
        writer.add_images(f"montecarlo/xct_std_{axis}",   _std_to_rgb(slicer(xct_std)),  step)
        writer.add_images(f"montecarlo/mask_std_{axis}",  _std_to_rgb(slicer(mask_std)), step)

    # ── Single normal prediction + GT ─────────────────────────────────────
    # Use the first MC sample as the "normal" reconstruction (already computed).
    xct_recon_single  = xct_samples[0]                          # (B, 1, D, H, W)
    mask_recon_single = mask_samples[0]
    xct_gt  = xct.clamp(0.0, 1.0)
    mask_gt = mask.clamp(0.0, 1.0)

    for axis, slicer in [
        ("d", _central_slice_d),
        ("h", _central_slice_h),
        ("w", _central_slice_w),
    ]:
        writer.add_images(f"montecarlo/xct_recon_{axis}",  slicer(xct_recon_single),  step)
        writer.add_images(f"montecarlo/xct_gt_{axis}",     slicer(xct_gt),            step)
        writer.add_images(f"montecarlo/mask_recon_{axis}", slicer(mask_recon_single), step)
        writer.add_images(f"montecarlo/mask_gt_{axis}",    slicer(mask_gt),           step)

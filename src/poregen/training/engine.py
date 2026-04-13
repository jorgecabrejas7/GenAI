"""Train / eval step helpers and the main training loop."""

from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from poregen.models.vae.base import VAEOutput
from poregen.metrics.seg import segmentation_metrics, porosity_metrics
from poregen.metrics.recon import mae, psnr, sharpness_proxy
from poregen.metrics.latent import (
    active_units_from_moments,
    latent_channel_moments,
    latent_stats,
    merge_latent_channel_moments,
)
from poregen.training.checkpoint import save_checkpoint
from poregen.training.sample_export import export_patch_sample_split


# ── helpers ───────────────────────────────────────────────────────────────────

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
            out[k] = sum(v) / len(v) if v else 0.0
        else:
            out[k] = v / n
    return out


def _log_scalars_to_tb(
    tb_writer: Any,
    metrics: dict[str, Any],
    prefix: str,
    step: int,
) -> None:
    """Write all scalar metrics (and per-channel KL) to TensorBoard."""
    for k, v in metrics.items():
        if isinstance(v, list):
            for i, ch_val in enumerate(v):
                tb_writer.add_scalar(f"{prefix}/kl_ch{i:02d}", ch_val, step)
        else:
            tb_writer.add_scalar(f"{prefix}/{k}", v, step)
    if "kl_per_channel" in metrics and isinstance(metrics["kl_per_channel"], list):
        tb_writer.add_histogram(
            f"{prefix}/kl_per_channel",
            torch.tensor(metrics["kl_per_channel"]),
            step,
        )


# ── single-step helpers ───────────────────────────────────────────────────────

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
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Single training step with AMP, optional gradient clipping, and scheduler.

    Returns
    -------
    losses : dict
        Per-component loss values. Scalars are Python floats; ``kl_per_channel``
        is a list of length C.
    grad_norm : float
        Global gradient norm (after unscaling, before clipping). 0.0 if
        max_grad_norm is None.
    latent_moments : dict
        Channel-wise aggregated moments for ``output.mu``. Used to compute
        train-time ``n_active`` over a rolling window without storing all
        latent tensors.
    """
    model.train()
    xct  = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)
    latent_moments = latent_channel_moments(output.mu)

    scaler.scale(losses["total"]).backward()

    grad_norm = 0.0
    if max_grad_norm is not None:
        # Only unscale when the scaler is actually active (float16 path).
        # On bfloat16 / CPU the scaler is disabled and unscale_() is a no-op
        # that still iterates all gradient buffers — skip it.
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).item()

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return {k: _to_scalar(v) for k, v in losses.items()}, grad_norm, latent_moments


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
    xct  = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)

    return {k: _to_scalar(v) for k, v in losses.items()}, output


# ── eval-over-N-batches helper ────────────────────────────────────────────────

def _run_eval(
    model: nn.Module,
    data_iter: Iterator,
    loss_fn: Callable,
    n_batches: int,
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> tuple[dict[str, float], VAEOutput, dict, dict]:
    """Run eval over *n_batches* batches; return aggregated metrics and last outputs."""
    loss_acc:   dict[str, Any] = {}
    seg_acc:    dict[str, Any] = {}
    recon_acc:  dict[str, Any] = {}
    latent_acc: dict[str, Any] = {}
    por_acc:    dict[str, Any] = {}
    latent_moment_summaries: list[dict[str, Any]] = []
    vol_por_errors: dict[str, list[float]] = {}
    last_output: VAEOutput | None = None
    last_batch:  dict | None = None

    # Accumulators for deferred .item() calls — kept as GPU tensors until
    # after the loop to avoid per-batch CPU/GPU synchronisation stalls.
    mae_acc:             list[torch.Tensor] = []
    psnr_acc:            list[torch.Tensor] = []
    sharp_recon_acc:     list[torch.Tensor] = []
    sharp_gt_acc:        list[torch.Tensor] = []
    xct_mean_acc:        list[torch.Tensor] = []
    xct_std_acc:         list[torch.Tensor] = []
    pred_por_signed_all: list[torch.Tensor] = []
    vol_ids_all:         list[list[str]]    = []

    for _ in range(n_batches):
        batch = next(data_iter)
        losses, output = eval_step(
            model, batch, loss_fn, step, device, autocast_dtype
        )
        _accumulate(loss_acc, losses)

        mask_dev = batch["mask"].to(device, non_blocking=True)
        xct_dev  = batch["xct"].to(device,  non_blocking=True)

        # Compute sigmoid once — reused by loss, metrics, and per-volume tracking
        mask_sigmoid = torch.sigmoid(output.mask_logits)
        xct_sigmoid  = torch.sigmoid(output.xct_logits)

        # Segmentation metrics (already vectorised; pass cached sigmoid via logits path)
        seg = segmentation_metrics(output.mask_logits, mask_dev)
        _accumulate(seg_acc, seg)

        # Porosity metrics
        por = porosity_metrics(output.mask_logits, mask_dev)
        _accumulate(por_acc, por)

        # Per-volume porosity tracking — accumulate tensors, defer .item()
        pred_por_v = mask_sigmoid.mean(dim=(1, 2, 3, 4))   # (B,)
        gt_por_v   = mask_dev.mean(dim=(1, 2, 3, 4))        # (B,)
        pred_por_signed_all.append(pred_por_v - gt_por_v)
        vol_ids_all.append(list(batch["volume_id"]))

        # Reconstruction metrics — accumulate as tensors, defer .item()
        mae_acc.append(mae(output.xct_logits, xct_dev))
        psnr_acc.append(psnr(output.xct_logits, xct_dev))
        sharp_recon_acc.append(sharpness_proxy(xct_sigmoid))
        sharp_gt_acc.append(sharpness_proxy(xct_dev))
        xct_mean_acc.append(output.xct_logits.mean())
        xct_std_acc.append(output.xct_logits.std())

        # Latent metrics
        lat = latent_stats(output.mu, output.logvar)
        _accumulate(latent_acc, lat)
        latent_moment_summaries.append(latent_channel_moments(output.mu))

        last_output = output
        last_batch  = batch

    # ── deferred .item() — single sync per metric after the loop ──────────────
    sharp_recon_mean = float(torch.stack(sharp_recon_acc).mean().item())
    sharp_gt_mean    = float(torch.stack(sharp_gt_acc).mean().item())
    recon_agg = {
        "mae":             float(torch.stack(mae_acc).mean().item()),
        "psnr":            float(torch.stack(psnr_acc).mean().item()),
        "sharpness_recon": sharp_recon_mean,
        "sharpness_gt":    sharp_gt_mean,
        "sharpness_recon_over_gt": (
            sharp_recon_mean / sharp_gt_mean if sharp_gt_mean > 0.0 else float("nan")
        ),
        "recon_xct_mean": float(torch.stack(xct_mean_acc).mean().item()),
        "recon_xct_std":  float(torch.stack(xct_std_acc).mean().item()),
    }

    # Per-volume porosity errors (for histogram logging)
    for signed_batch, vids in zip(pred_por_signed_all, vol_ids_all):
        signed_cpu = signed_batch.cpu()
        for i, vid in enumerate(vids):
            vol_por_errors.setdefault(vid, []).append(float(signed_cpu[i].item()))

    agg = {
        **_mean_acc(loss_acc, n_batches),
        **_mean_acc(seg_acc, n_batches),
        **_mean_acc(por_acc, n_batches),
        **recon_agg,
        **_mean_acc(latent_acc, n_batches),
    }

    if "kl_per_channel" in agg and isinstance(agg["kl_per_channel"], list):
        agg["kl_total"] = sum(agg["kl_per_channel"])

    if latent_moment_summaries:
        merged = merge_latent_channel_moments(latent_moment_summaries)
        agg.update(active_units_from_moments(
            merged["count"], merged["sum"], merged["sum_sq"],
        ))
    else:
        agg.update({"active_fraction": 0.0, "n_active": 0, "n_total": 0})

    return agg, last_output, last_batch, vol_por_errors  # type: ignore[return-value]


def _snapshot_logging_batch(batch: dict[str, Any], n_examples: int) -> dict[str, torch.Tensor]:
    """Keep a small, fixed batch on CPU for lightweight reconstruction logging."""
    n_take = min(max(1, n_examples), batch["xct"].shape[0])
    return {
        "xct":  batch["xct"] [:n_take].cpu(),
        "mask": batch["mask"][:n_take].cpu(),
    }


# ── training loop ─────────────────────────────────────────────────────────────

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
    montecarlo_every: int | None = None,
    montecarlo_batch_size: int = 8,
    sample_every: int = 0,
    n_patch_samples: int = 8,
    run_dir: str | Path = "runs/vae/default",
    device: torch.device = torch.device("cpu"),
    autocast_dtype: torch.dtype = torch.float16,
    start_step: int = 0,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
    tb_writer: Any | None = None,
    train_active_window_batches: int = 50,
    final_full_eval: bool = False,
    compile_model: bool = False,
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
    montecarlo_every : int, optional
        Steps between Monte Carlo uncertainty logs. If None, reuse
        ``image_log_every`` for backward-compatible behavior.
    montecarlo_batch_size : int
        Number of patches kept in the fixed showcase batch for Monte Carlo
        logging.
    sample_every : int
        Steps between full 3-D patch sample saves to disk (0 = disabled).
    n_patch_samples : int
        Number of patches to save per split at each sample checkpoint.
    tb_writer : SummaryWriter, optional
        TensorBoard writer. All monitoring is skipped if None.
    train_active_window_batches : int
        Number of recent train batches used for rolling ``train/n_active``.
    final_full_eval : bool
        If True, run one full pass over validation and test loaders after the
        final training step and log the aggregated metrics.
    compile_model : bool
        If True, wrap *model* with ``torch.compile(mode="reduce-overhead")``
        before training.  Typically yields 25-30 % throughput improvement on
        Ampere/Hopper hardware with static 64³ input shapes.  The first
        forward pass will be slower (~10 s) while the kernel is compiled.

    Returns
    -------
    list of per-step metric dicts (train + val records, for inline plotting).
    """
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path     = run_dir / "log.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    montecarlo_every = image_log_every if montecarlo_every is None else montecarlo_every

    train_iter = _infinite(train_loader)
    val_iter   = _infinite(val_loader)   if val_loader   is not None else None
    test_iter  = _infinite(test_loader)  if test_loader  is not None else None

    history: list[dict[str, Any]] = []
    t0 = time.time()
    train_active_window: deque[dict[str, Any]] = deque(
        maxlen=max(1, train_active_window_batches)
    )
    # Cache the last computed active-units stats; recomputed every 10 steps.
    train_active: dict[str, Any] = {"active_fraction": 0.0, "n_active": 0, "n_total": 0}
    last_output:       VAEOutput | None = None
    last_batch:        dict[str, Any] | None = None
    montecarlo_batch:  dict[str, torch.Tensor] | None = None

    pbar = tqdm(range(start_step, start_step + total_steps), desc="Training")

    # Use ExitStack so both log files are always closed — even on exception.
    with contextlib.ExitStack() as stack:
        log_file     = stack.enter_context(open(log_path,     "a"))
        metrics_file = stack.enter_context(open(metrics_path, "a"))

        for step in pbar:
            # ── train step ────────────────────────────────────────────
            batch = next(train_iter)
            losses, grad_norm, latent_moments = train_step(
                model, batch, optimizer, scaler, loss_fn,
                step=step, device=device, autocast_dtype=autocast_dtype,
                max_grad_norm=max_grad_norm, scheduler=scheduler,
            )

            # Fixed MC batch — captured once from the first training batch
            if montecarlo_batch is None and montecarlo_batch_size > 0:
                montecarlo_batch = _snapshot_logging_batch(batch, montecarlo_batch_size)

            # Rolling active-units — recompute every 10 steps (it's a smoothed
            # display metric; recomputing every step wastes ~2-3 % of loop time).
            train_active_window.append(latent_moments)
            if (step + 1) % 10 == 0 or step == start_step:
                train_active_moments = merge_latent_channel_moments(train_active_window)
                train_active = active_units_from_moments(
                    train_active_moments["count"],
                    train_active_moments["sum"],
                    train_active_moments["sum_sq"],
                )

            record = {
                "step":    step,
                "split":   "train",
                "elapsed": time.time() - t0,
                **{k: v for k, v in losses.items() if not isinstance(v, list)},
                **train_active,
            }
            history.append(record)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            pbar.set_postfix(
                loss=f"{losses['total']:.4f}",
                kl=f"{losses.get('kl', 0):.4f}",
                β=f"{losses.get('beta', 0):.4f}",
            )

            # ── TensorBoard: train scalars ────────────────────────────
            if tb_writer is not None and (step + 1) % log_every == 0:
                _log_scalars_to_tb(tb_writer, losses, "train", step)
                for k, v in train_active.items():
                    tb_writer.add_scalar(f"train/{k}", v, step)
                tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                if scheduler is not None:
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                kl_chs = losses.get("kl_per_channel", [])
                if kl_chs and "freebits_used" in losses:
                    n_dead = round(losses["freebits_used"] * len(kl_chs))
                    tb_writer.add_scalar("train/active_channels", len(kl_chs) - n_dead, step)

            # ── validation ───────────────────────────────────────────
            if val_iter is not None and (step + 1) % eval_every == 0:
                agg, last_output, last_batch, _vol_por = _run_eval(
                    model, val_iter, loss_fn, val_batches, step, device, autocast_dtype
                )
                val_record = {
                    "step":    step,
                    "split":   "val",
                    "elapsed": time.time() - t0,
                    **{k: v for k, v in agg.items() if not isinstance(v, list)},
                }
                history.append(val_record)
                log_file.write(json.dumps(val_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(val_record) + "\n")
                metrics_file.flush()

                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, agg, "val", step)

            # ── TensorBoard: reconstruction images ───────────────────
            if (
                tb_writer is not None
                and val_iter is not None
                and last_output is not None
                and last_batch is not None
                and image_log_every > 0
                and (step + 1) % image_log_every == 0
            ):
                _log_recon_images(tb_writer, last_output, last_batch, step, "val", device)

            # ── TensorBoard: fixed showcase Monte Carlo ───────────────
            if (
                tb_writer is not None
                and montecarlo_batch is not None
                and montecarlo_every > 0
                and (step + 1) % montecarlo_every == 0
            ):
                run_montecarlo_eval(
                    model, montecarlo_batch, step, device, tb_writer,
                    autocast_dtype=autocast_dtype,
                )

            # ── test evaluation ───────────────────────────────────────
            if test_iter is not None and (step + 1) % test_every == 0:
                test_agg, test_output, test_batch, test_vol_por = _run_eval(
                    model, test_iter, loss_fn, test_batches, step, device, autocast_dtype
                )
                test_record = {
                    "step":    step,
                    "split":   "test",
                    "elapsed": time.time() - t0,
                    **{k: v for k, v in test_agg.items() if not isinstance(v, list)},
                }
                metrics_file.write(json.dumps(test_record) + "\n")
                metrics_file.flush()

                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, test_agg, "test", step)
                    _log_recon_images(tb_writer, test_output, test_batch, step, "test", device)
                    if test_vol_por:
                        per_vol_maes = torch.tensor(
                            [abs(sum(errs) / len(errs)) for errs in test_vol_por.values()]
                        )
                        tb_writer.add_histogram("test/porosity_mae_per_volume", per_vol_maes, step)

            # ── checkpoint ───────────────────────────────────────────
            if (step + 1) % save_every == 0:
                ckpt_name = f"{run_dir.name}_step{step + 1:08d}.ckpt"
                save_checkpoint(
                    run_dir / ckpt_name,
                    model, optimizer, scaler, step=step + 1,
                    metadata={"total_steps": total_steps},
                    scheduler=scheduler,
                )

            # ── 3-D patch samples ─────────────────────────────────────
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

        # ── final checkpoint ──────────────────────────────────────────
        final_step = start_step + total_steps
        ckpt_name = f"{run_dir.name}_step{final_step:08d}.ckpt"
        save_checkpoint(
            run_dir / ckpt_name,
            model, optimizer, scaler, step=final_step,
            metadata={"total_steps": total_steps},
            scheduler=scheduler,
        )

        # ── final full eval ───────────────────────────────────────────
        if final_full_eval:
            if val_loader is not None:
                fv_agg, fv_out, fv_batch, _ = _run_eval(
                    model, iter(val_loader), loss_fn,
                    len(val_loader), final_step, device, autocast_dtype,
                )
                fv_record = {
                    "step": final_step, "split": "val", "elapsed": time.time() - t0,
                    "full_eval": True, "n_batches": len(val_loader),
                    **{k: v for k, v in fv_agg.items() if not isinstance(v, list)},
                }
                log_file.write(json.dumps(fv_record) + "\n")
                metrics_file.write(json.dumps(fv_record) + "\n")
                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, fv_agg, "val_full", final_step)
                    _log_recon_images(tb_writer, fv_out, fv_batch, final_step, "val_full", device)

            if test_loader is not None:
                ft_agg, ft_out, ft_batch, ft_vol_por = _run_eval(
                    model, iter(test_loader), loss_fn,
                    len(test_loader), final_step, device, autocast_dtype,
                )
                ft_record = {
                    "step": final_step, "split": "test", "elapsed": time.time() - t0,
                    "full_eval": True, "n_batches": len(test_loader),
                    **{k: v for k, v in ft_agg.items() if not isinstance(v, list)},
                }
                metrics_file.write(json.dumps(ft_record) + "\n")
                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, ft_agg, "test_full", final_step)
                    _log_recon_images(tb_writer, ft_out, ft_batch, final_step, "test_full", device)
                    if ft_vol_por:
                        per_vol_maes = torch.tensor(
                            [abs(sum(errs) / len(errs)) for errs in ft_vol_por.values()]
                        )
                        tb_writer.add_histogram(
                            "test/porosity_mae_per_volume", per_vol_maes, final_step
                        )

    return history


# ── patch sample saving ───────────────────────────────────────────────────────

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
    """Save full 3-D patch reconstructions as per-patch TIFF stacks."""
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
            batch  = next(data_iter)
            n_take = min(n_samples - collected, batch["xct"].shape[0])

            xct  = batch["xct"] [:n_take].to(device)
            mask = batch["mask"][:n_take].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(xct, mask)

            xct_gts.append(xct.cpu().float().numpy())
            mask_gts.append(mask.cpu().float().numpy())
            xct_recons.append(output.xct_logits.clamp(0.0, 1.0).cpu().float().numpy()[:n_take])
            mask_recons.append(torch.sigmoid(output.mask_logits).cpu().float().numpy()[:n_take])

            coords = batch["coords"]
            for i in range(n_take):
                metas.append({
                    "volume_id":    batch["volume_id"][i],
                    "z0": int(coords[i][0]), "y0": int(coords[i][1]), "x0": int(coords[i][2]),
                    "porosity":     float(batch["porosity"][i]),
                    "source_group": batch["source_group"][i],
                })
            collected += n_take

        export_patch_sample_split(
            samples_dir / split,
            {
                "xct_gt":    np.concatenate(xct_gts),
                "mask_gt":   np.concatenate(mask_gts),
                "xct_recon": np.concatenate(xct_recons),
                "mask_recon": np.concatenate(mask_recons),
            },
            metas,
        )

    model.train()


# ── image logging helper ──────────────────────────────────────────────────────

def _log_recon_images(
    tb_writer: Any,
    output: VAEOutput,
    batch: dict,
    step: int,
    prefix: str,
    device: torch.device,
) -> None:
    """Log central slices along all 3 axes to TensorBoard."""
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


# ── Monte Carlo uncertainty estimation ───────────────────────────────────────

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

    The model is set to eval() so BatchNorm/Dropout behave deterministically,
    but the VAE reparameterization samples a different z each pass.

    Logs per axis (d=axial, h=coronal, w=sagittal):
      montecarlo/xct_mean_{axis}   — mean reconstruction (grayscale)
      montecarlo/xct_std_{axis}    — voxel-wise std       (colormap)
      montecarlo/mask_mean_{axis}  — mean mask sigmoid    (grayscale)
      montecarlo/mask_std_{axis}   — voxel-wise std       (colormap)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    xct  = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    xct_samples:  list[torch.Tensor] = []
    mask_samples: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(xct, mask)
            xct_samples.append(output.xct_logits.clamp(0.0, 1.0).float())
            mask_samples.append(torch.sigmoid(output.mask_logits).float())

    xct_stack  = torch.stack(xct_samples,  dim=0)   # (N, B, 1, D, H, W)
    mask_stack = torch.stack(mask_samples, dim=0)

    xct_mean  = xct_stack.mean(dim=0)
    xct_std   = xct_stack.std(dim=0)
    mask_mean = mask_stack.mean(dim=0)
    mask_std  = mask_stack.std(dim=0)

    _DIVERSITY_EPS = 1e-5
    xct_div  = xct_std.mean().item()
    mask_div = mask_std.mean().item()
    if xct_div < _DIVERSITY_EPS or mask_div < _DIVERSITY_EPS:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "run_montecarlo_eval step=%d: predictions nearly identical "
            "(xct_div=%.2e  mask_div=%.2e). "
            "Check reparameterization / posterior collapse.",
            step, xct_div, mask_div,
        )
    writer.add_scalar("montecarlo/xct_diversity",  xct_div,  step)
    writer.add_scalar("montecarlo/mask_diversity", mask_div, step)

    cmap = plt.get_cmap(cmap_name)

    def _std_to_rgb(std_slice: torch.Tensor) -> torch.Tensor:
        arr = std_slice.squeeze(1).cpu().numpy()
        vmax = arr.max() if arr.max() > 0 else 1.0
        rgba = np.stack([cmap(arr[b] / vmax) for b in range(arr.shape[0])])
        return torch.from_numpy(rgba[..., :3].transpose(0, 3, 1, 2).astype(np.float32))

    for axis, slicer in [
        ("d", _central_slice_d),
        ("h", _central_slice_h),
        ("w", _central_slice_w),
    ]:
        writer.add_images(f"montecarlo/xct_mean_{axis}",  slicer(xct_mean),  step)
        writer.add_images(f"montecarlo/mask_mean_{axis}", slicer(mask_mean), step)
        writer.add_images(f"montecarlo/xct_std_{axis}",   _std_to_rgb(slicer(xct_std)),  step)
        writer.add_images(f"montecarlo/mask_std_{axis}",  _std_to_rgb(slicer(mask_std)), step)

    xct_recon_single  = xct_samples[0]
    mask_recon_single = mask_samples[0]
    xct_gt_clamped    = xct.clamp(0.0, 1.0)
    mask_gt_clamped   = mask.clamp(0.0, 1.0)

    for axis, slicer in [
        ("d", _central_slice_d),
        ("h", _central_slice_h),
        ("w", _central_slice_w),
    ]:
        writer.add_images(f"montecarlo/xct_recon_{axis}",  slicer(xct_recon_single),  step)
        writer.add_images(f"montecarlo/xct_gt_{axis}",     slicer(xct_gt_clamped),    step)
        writer.add_images(f"montecarlo/mask_recon_{axis}", slicer(mask_recon_single), step)
        writer.add_images(f"montecarlo/mask_gt_{axis}",    slicer(mask_gt_clamped),   step)

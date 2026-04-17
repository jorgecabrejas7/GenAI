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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from poregen.models.vae.base import VAEOutput
from poregen.models.discriminator import (
    extract_multiplane_slices,
    lsgan_gen_loss,
    lsgan_disc_loss,
)
from poregen.metrics.seg import segmentation_metrics, porosity_metrics, porosity_binned_mae
from poregen.metrics.recon import sharpness_proxy
from poregen.metrics.latent import (
    active_units_from_moments,
    latent_channel_moments,
    latent_stats,
    merge_latent_channel_moments,
)
from poregen.training.checkpoint import copy_checkpoint, save_checkpoint, save_checkpoint_async
from poregen.training.sample_export import export_patch_sample_split

import logging as _logging
_logger = _logging.getLogger(__name__)


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
    log_list_scalars: bool = True,
    skip_keys: frozenset[str] = frozenset(),
) -> None:
    """Write all scalar metrics (and per-channel KL) to TensorBoard.

    Parameters
    ----------
    log_list_scalars : bool
        When False, list values are not expanded to per-channel scalars
        (the histogram is still written).  Set False for train/ to avoid
        high-cardinality per-channel scalar noise.
    skip_keys : frozenset
        Metric keys to omit entirely from TB logging.
    """
    for k, v in metrics.items():
        if k in skip_keys:
            continue
        if isinstance(v, list):
            if log_list_scalars:
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
    discriminator: nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    disc_weight: float = 0.01,
) -> tuple[dict[str, Any], float, dict[str, Any], dict[str, float]]:
    """Single training step with AMP, optional gradient clipping, and scheduler.

    Parameters
    ----------
    discriminator : optional PatchDiscriminator2D
        When provided (together with ``disc_optimizer``), runs one generator and
        one discriminator update per training step.  The discriminator operates
        in float32 (no AMP) for stability.
    disc_optimizer : optional
        Dedicated optimizer for the discriminator.  Must be provided when
        ``discriminator`` is not None.
    disc_weight : float
        Scale factor applied to the generator adversarial loss before adding it
        to the VAE total loss.  Typical range: 0.001–0.05.

    Returns
    -------
    losses : dict
        Per-component loss values plus ``mask_pred_mean``.  When a discriminator
        is active, also includes ``disc_loss``, ``gen_adv_loss``,
        ``disc_acc_real``, ``disc_acc_fake``.  Scalars are Python floats;
        ``kl_per_channel`` is a list of length C.
    grad_norm : float
        Global generator gradient norm (after unscaling, before clipping).
    latent_moments : dict
        Channel-wise aggregated moments for ``output.mu``.
    module_grad_norms : dict
        Per-module gradient norms (encoder / encoder_a / encoder_b / decoder /
        mask_head) computed after unscaling and before global clipping.
    """
    model.train()
    xct  = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)

    # ── Adversarial: generator side (float32 outside AMP) ────────────────────
    _fake_slices: torch.Tensor | None = None
    _real_slices: torch.Tensor | None = None
    _gen_adv_loss: torch.Tensor | None = None

    if discriminator is not None and disc_optimizer is not None and disc_weight > 0.0:
        # Cast from AMP dtype (fp16/bf16) to float32 — D runs in float32
        _fake_slices = extract_multiplane_slices(output.xct_logits).float()   # (3B,1,64,64)
        _real_slices = extract_multiplane_slices(xct).float()                  # (3B,1,64,64)

        # Generator wants D(fake) → 1; gradients flow through D back to the VAE
        d_fake_gen = discriminator(_fake_slices)
        _gen_adv_loss = lsgan_gen_loss(d_fake_gen)

        # Add to total BEFORE backward so generator gets adversarial signal
        losses["total"] = losses["total"] + disc_weight * _gen_adv_loss

    latent_moments = latent_channel_moments(output.mu)
    mask_pred_mean = float(torch.sigmoid(output.mask_logits).mean().item())

    scaler.scale(losses["total"]).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    # Per-module gradient norms — support both single-branch and dual-branch encoders
    module_grad_norms: dict[str, float] = {}
    for name in ("encoder", "encoder_a", "encoder_b", "decoder", "mask_head"):
        module = getattr(model, name, None)
        if module is not None:
            module_grad_norms[f"grad_norm_{name}"] = torch.nn.utils.clip_grad_norm_(
                module.parameters(), float("inf")
            ).item()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    ).item()

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    # ── Adversarial: discriminator update (float32, no scaler) ───────────────
    disc_metrics: dict[str, float] = {}
    if (
        discriminator is not None
        and disc_optimizer is not None
        and _fake_slices is not None
        and _gen_adv_loss is not None
    ):
        # Zero D grads (clears any accumulated from the generator backward)
        disc_optimizer.zero_grad(set_to_none=True)

        d_real = discriminator(_real_slices)                        # float32, with grad
        d_fake = discriminator(_fake_slices.detach())               # detached from G graph

        disc_loss = lsgan_disc_loss(d_real, d_fake)
        disc_loss.backward()
        disc_optimizer.step()

        disc_metrics = {
            "disc_loss":       float(disc_loss.item()),
            "d_loss_real":     float(0.5 * (d_real - 1.0).pow(2).mean().item()),
            "d_loss_fake":     float(0.5 * d_fake.pow(2).mean().item()),
            "gen_adv_loss":    float(_gen_adv_loss.item()),
            "disc_acc_real":   float((d_real > 0.5).float().mean().item()),
            "disc_acc_fake":   float((d_fake < 0.5).float().mean().item()),
            "disc_score_real": float(d_real.mean().item()),
            "disc_score_fake": float(d_fake.mean().item()),
            "disc_margin":     float((d_real - d_fake).mean().item()),
        }

    result = {k: _to_scalar(v) for k, v in losses.items()}
    result["mask_pred_mean"] = mask_pred_mean
    result.update(disc_metrics)
    return result, grad_norm, latent_moments, module_grad_norms


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
    desc: str = "Eval",
) -> tuple[dict[str, float], dict]:
    """Run eval over *n_batches* batches; return aggregated metrics and per-volume errors."""
    loss_acc:   dict[str, Any] = {}
    seg_acc:    dict[str, Any] = {}
    latent_acc: dict[str, Any] = {}
    por_acc:    dict[str, Any] = {}
    latent_moment_summaries: list[dict[str, Any]] = []
    vol_por_errors: dict[str, list[float]] = {}

    # Accumulators for deferred .item() calls — kept as GPU tensors until
    # after the loop to avoid per-batch CPU/GPU synchronisation stalls.
    mae_acc:             list[torch.Tensor] = []
    sharp_recon_acc:     list[torch.Tensor] = []
    sharp_gt_acc:        list[torch.Tensor] = []
    pred_por_signed_all: list[torch.Tensor] = []
    pred_por_all:        list[torch.Tensor] = []
    gt_por_all:          list[torch.Tensor] = []
    vol_ids_all:         list[list[str]]    = []

    for batch_idx in tqdm(range(n_batches), desc=desc, leave=False, unit="batch"):
        batch = next(data_iter)
        losses, output = eval_step(
            model, batch, loss_fn, step, device, autocast_dtype
        )
        _accumulate(loss_acc, losses)

        mask_dev = batch["mask"].to(device, non_blocking=True)
        xct_dev  = batch["xct"].to(device,  non_blocking=True)

        # Compute sigmoid ONCE — passed to all metrics to avoid redundant sigmoid
        # calls inside each metric function (each would otherwise create a full
        # (B,1,64,64,64) tensor = ~64 MB per call at batch_size=128, fp16).
        mask_sigmoid = torch.sigmoid(output.mask_logits)
        xct_sigmoid  = torch.sigmoid(output.xct_logits)

        # Segmentation metrics — pass pre-activated to skip internal sigmoid
        seg = segmentation_metrics(mask_sigmoid, mask_dev, apply_sigmoid=False)
        _accumulate(seg_acc, seg)

        # Porosity metrics — pass pre-activated to skip internal sigmoid
        por = porosity_metrics(mask_sigmoid, mask_dev, apply_sigmoid=False)
        _accumulate(por_acc, por)

        # Per-volume porosity tracking — accumulate tensors, defer .item()
        pred_por_v = mask_sigmoid.mean(dim=(1, 2, 3, 4))   # (B,)
        gt_por_v   = mask_dev.mean(dim=(1, 2, 3, 4))        # (B,)
        pred_por_signed_all.append((pred_por_v - gt_por_v).cpu())
        pred_por_all.append(pred_por_v.cpu())
        gt_por_all.append(gt_por_v.cpu())
        vol_ids_all.append(list(batch["volume_id"]))

        # Reconstruction metrics — use pre-activated xct_sigmoid, no double sigmoid
        mae_acc.append(F.l1_loss(xct_sigmoid, xct_dev))
        sharp_recon_acc.append(sharpness_proxy(xct_sigmoid))
        sharp_gt_acc.append(sharpness_proxy(xct_dev))

        # Latent metrics
        lat = latent_stats(output.mu, output.logvar)
        _accumulate(latent_acc, lat)
        latent_moment_summaries.append(latent_channel_moments(output.mu))

        # Explicitly release large GPU tensors — Python refcounting usually
        # handles this, but being explicit prevents accidental retention and
        # ensures the allocator can reuse the memory for the next batch.
        del output, mask_sigmoid, xct_sigmoid, mask_dev, xct_dev

    # ── deferred .item() — single sync per metric after the loop ──────────────
    sharp_recon_mean = float(torch.stack(sharp_recon_acc).mean().item())
    sharp_gt_mean    = float(torch.stack(sharp_gt_acc).mean().item())
    recon_agg = {
        "mae": float(torch.stack(mae_acc).mean().item()),
        "sharpness_recon_over_gt": (
            sharp_recon_mean / sharp_gt_mean if sharp_gt_mean > 0.0 else float("nan")
        ),
    }

    # Free GPU accumulator stacks; everything we need is already on CPU
    torch.cuda.empty_cache()

    # Per-volume porosity errors (for histogram logging) — already on CPU
    for signed_batch, vids in zip(pred_por_signed_all, vol_ids_all):
        for i, vid in enumerate(vids):
            vol_por_errors.setdefault(vid, []).append(float(signed_batch[i].item()))

    # Porosity-binned MAE across the full eval set
    all_pred_por = torch.cat(pred_por_all)
    all_gt_por   = torch.cat(gt_por_all)
    binned_mae_dict = porosity_binned_mae(all_pred_por, all_gt_por)

    agg = {
        **_mean_acc(loss_acc, n_batches),
        **_mean_acc(seg_acc, n_batches),
        **_mean_acc(por_acc, n_batches),
        **recon_agg,
        **_mean_acc(latent_acc, n_batches),
        **binned_mae_dict,
    }

    if "kl_per_channel" in agg and isinstance(agg["kl_per_channel"], list):
        agg["kl_raw"] = sum(agg["kl_per_channel"])

    if latent_moment_summaries:
        merged = merge_latent_channel_moments(latent_moment_summaries)
        agg.update(active_units_from_moments(
            merged["count"], merged["sum"], merged["sum_sq"],
        ))
    else:
        agg.update({"mu_active_fraction": 0.0, "mu_n_active": 0})

    return agg, vol_por_errors


def _snapshot_logging_batch(batch: dict[str, Any], n_examples: int) -> dict[str, torch.Tensor]:
    """Keep a small, fixed batch on CPU for lightweight reconstruction logging."""
    n_take = min(max(1, n_examples), batch["xct"].shape[0])
    return {
        "xct":  batch["xct"] [:n_take].cpu(),
        "mask": batch["mask"][:n_take].cpu(),
    }


def _parse_metric_target(metric: str | None) -> tuple[str | None, str | None]:
    """Split a metric target like ``val_full.total`` into split/key parts."""
    if not metric:
        return None, None
    split, sep, key = metric.partition(".")
    if not sep or not split or not key:
        raise ValueError(
            "Best metric targets must look like '<split>.<metric>', "
            f"got {metric!r}."
        )
    return split, key


def _is_better_metric(
    candidate: float,
    current_best: float | None,
    *,
    mode: str,
) -> bool:
    if current_best is None:
        return True
    if mode == "min":
        return candidate < current_best
    if mode == "max":
        return candidate > current_best
    raise ValueError(f"Unsupported best-metric mode: {mode!r}")


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
    full_val_every: int = 0,
    final_full_eval: bool = False,
    compile_model: bool = False,
    save_latest: bool = True,
    best_metric: str | None = None,
    best_mode: str = "min",
    discriminator: nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    disc_weight: float = 0.01,
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
        Number of recent train batches used for rolling ``train/mu_n_active``.
    full_val_every : int
        Steps between full-loader validation passes (0 = disabled).  When
        triggered, iterates *all* batches in ``val_loader`` once and logs
        aggregated metrics to the ``val_full/`` TensorBoard prefix.  Intended
        to fire once per training epoch (set to ``len(train_loader)``).  The
        regular partial eval (``eval_every`` / ``val_batches``) continues to
        run independently and logs to ``val/`` as before.
    final_full_eval : bool
        If True, run one full pass over validation and test loaders after the
        final training step and log the aggregated metrics.
    compile_model : bool
        If True, wrap *model* with ``torch.compile(mode="reduce-overhead")``
        before training.  Typically yields 25-30 % throughput improvement on
        Ampere/Hopper hardware with static 64³ input shapes.  The first
        forward pass will be slower (~10 s) while the kernel is compiled.
    save_latest : bool
        If True, copy the most recent periodic/final checkpoint to
        ``latest.ckpt`` inside ``run_dir``.
    best_metric : str, optional
        Metric target to track for best-checkpoint saving, formatted as
        ``<split>.<metric>`` (for example ``val_full.total``).
    best_mode : str
        ``"min"`` or ``"max"`` comparison mode for *best_metric*.
    discriminator : nn.Module, optional
        2D PatchGAN discriminator (e.g. ``PatchDiscriminator2D``).  When
        provided, adversarial training is enabled: one generator and one
        discriminator update are performed per training step.  Logs
        ``disc_loss``, ``gen_adv_loss``, ``disc_acc_real``, ``disc_acc_fake``
        to TensorBoard under the ``train/`` prefix.
    disc_optimizer : optional
        Dedicated optimizer for *discriminator*.  Required when discriminator
        is not None.
    disc_weight : float
        Weight applied to the generator adversarial loss.  Start small
        (0.01) and increase if the adversarial signal is too weak.

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
    best_target_split, best_target_key = _parse_metric_target(best_metric)
    best_metric_value: float | None = None

    # Train workers stay alive throughout (needed for continuous prefetch).
    # Val/test workers are only created inside their respective eval blocks and
    # released immediately afterwards — DataLoader iterators keep the worker
    # pool alive; deleting the iterator lets the OS reclaim those processes and
    # the pinned memory they hold.
    train_iter = _infinite(train_loader)

    history: list[dict[str, Any]] = []
    t0 = time.time()
    train_active_window: deque[dict[str, Any]] = deque(
        maxlen=max(1, train_active_window_batches)
    )
    # Cache the last computed active-units stats; recomputed every 10 steps.
    train_active: dict[str, Any] = {"mu_active_fraction": 0.0, "mu_n_active": 0}
    montecarlo_batch:  dict[str, torch.Tensor] | None = None

    # Background checkpoint thread — holds [thread_or_None]; at most one save
    # in flight at a time. Joined before every new save and at loop exit.
    _ckpt_thread_holder: list = [None]

    pbar = tqdm(range(start_step, start_step + total_steps), desc="Training")

    def _maybe_save_best(record: dict[str, Any]) -> None:
        nonlocal best_metric_value
        if best_target_split is None or best_target_key is None:
            return
        if record.get("split") != best_target_split:
            return
        metric_value = record.get(best_target_key)
        if metric_value is None:
            return
        metric_float = float(metric_value)
        if not _is_better_metric(metric_float, best_metric_value, mode=best_mode):
            return

        best_metric_value = metric_float
        save_checkpoint(
            run_dir / "best.ckpt",
            model,
            optimizer,
            scaler,
            step=int(record["step"]) + (0 if record.get("full_eval") else 1),
            metadata={
                "total_steps": total_steps,
                "best_metric": best_metric,
                "best_mode": best_mode,
                "best_value": metric_float,
                "best_split": record.get("split"),
            },
            scheduler=scheduler,
        )

    # Use ExitStack so both log files are always closed — even on exception.
    with contextlib.ExitStack() as stack:
        log_file     = stack.enter_context(open(log_path,     "a"))
        metrics_file = stack.enter_context(open(metrics_path, "a"))

        for step in pbar:
            # ── train step ────────────────────────────────────────────
            batch = next(train_iter)
            _step_t0 = time.perf_counter()
            losses, grad_norm, latent_moments, module_grad_norms = train_step(
                model, batch, optimizer, scaler, loss_fn,
                step=step, device=device, autocast_dtype=autocast_dtype,
                max_grad_norm=max_grad_norm, scheduler=scheduler,
                discriminator=discriminator,
                disc_optimizer=disc_optimizer,
                disc_weight=disc_weight,
            )
            _step_elapsed = time.perf_counter() - _step_t0
            step_time_ms  = _step_elapsed * 1000.0
            steps_per_sec = 1.0 / _step_elapsed if _step_elapsed > 0 else float("inf")

            # ── GPU memory: log once after the first training step ────
            if step == start_step and torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                _logger.info(
                    "Step %d — peak GPU memory allocated: %.2f GB  "
                    "(reduce batch_size if near VRAM limit)",
                    step, peak_mem_gb,
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/peak_gpu_mem_gb", peak_mem_gb, step)

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
                "step":         step,
                "split":        "train",
                "elapsed":      time.time() - t0,
                "step_time_ms": step_time_ms,
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
                _log_scalars_to_tb(
                    tb_writer, losses, "train", step,
                    log_list_scalars=False,  # suppress train/kl_ch{i} per-channel scalars
                )
                for k, v in train_active.items():
                    tb_writer.add_scalar(f"train/{k}", v, step)
                tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                for k, v in module_grad_norms.items():
                    tb_writer.add_scalar(f"train/{k}", v, step)
                tb_writer.add_scalar("train/grad_scaler_scale", scaler.get_scale(), step)
                tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, step)
                if scheduler is not None:
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                # Discriminator metrics are scalar floats in `losses` and are
                # already written by _log_scalars_to_tb above — no extra loop needed.

            # ── validation ───────────────────────────────────────────
            if val_loader is not None and (step + 1) % eval_every == 0:
                _val_iter = iter(val_loader)
                agg, _vol_por = _run_eval(
                    model, _val_iter, loss_fn, val_batches, step, device, autocast_dtype,
                    desc=f"Val step {step + 1}",
                )
                del _val_iter
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
                _maybe_save_best(val_record)

            # ── full-loader validation (once per epoch) ───────────────
            if (
                val_loader is not None
                and full_val_every > 0
                and (step + 1) % full_val_every == 0
            ):
                fv_agg, _ = _run_eval(
                    model, iter(val_loader), loss_fn,
                    len(val_loader), step, device, autocast_dtype,
                    desc=f"Val full step {step + 1}",
                )
                fv_record = {
                    "step":    step,
                    "split":   "val_full",
                    "elapsed": time.time() - t0,
                    **{k: v for k, v in fv_agg.items() if not isinstance(v, list)},
                }
                history.append(fv_record)
                log_file.write(json.dumps(fv_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(fv_record) + "\n")
                metrics_file.flush()

                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, fv_agg, "val_full", step)
                _maybe_save_best(fv_record)

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
            if test_loader is not None and (step + 1) % test_every == 0:
                _test_iter = iter(test_loader)
                test_agg, test_vol_por = _run_eval(
                    model, _test_iter, loss_fn, test_batches, step, device, autocast_dtype,
                    desc=f"Test step {step + 1}",
                )
                del _test_iter
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
                    if test_vol_por:
                        per_vol_maes = torch.tensor(
                            [abs(sum(errs) / len(errs)) for errs in test_vol_por.values()]
                        )
                        tb_writer.add_histogram("test/porosity_mae_per_volume", per_vol_maes, step)
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_p50", per_vol_maes.quantile(0.5).item(), step
                        )
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_p90", per_vol_maes.quantile(0.9).item(), step
                        )
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_max", per_vol_maes.max().item(), step
                        )
                _maybe_save_best(test_record)

            # ── checkpoint (non-blocking I/O via background thread) ──
            if (step + 1) % save_every == 0:
                ckpt_name = f"{run_dir.name}_step{step + 1:08d}.ckpt"
                save_checkpoint_async(
                    run_dir / ckpt_name,
                    model, optimizer, scaler, step=step + 1,
                    metadata={"total_steps": total_steps},
                    scheduler=scheduler,
                    latest_path=run_dir / "latest.ckpt" if save_latest else None,
                    thread_holder=_ckpt_thread_holder,
                )

            # ── 3-D patch samples ─────────────────────────────────────
            if sample_every > 0 and (step + 1) % sample_every == 0:
                _save_patch_samples(
                    model,
                    {"train": train_loader, "val": val_loader, "test": test_loader},
                    n_patch_samples,
                    step + 1,
                    run_dir,
                    device,
                    autocast_dtype,
                )

        # ── final checkpoint — wait for any in-flight background save ─
        if _ckpt_thread_holder and _ckpt_thread_holder[0] is not None:
            _ckpt_thread_holder[0].join()

        final_step = start_step + total_steps
        ckpt_name = f"{run_dir.name}_step{final_step:08d}.ckpt"
        final_ckpt_path = save_checkpoint(
            run_dir / ckpt_name,
            model, optimizer, scaler, step=final_step,
            metadata={"total_steps": total_steps},
            scheduler=scheduler,
        )
        if save_latest:
            copy_checkpoint(final_ckpt_path, run_dir / "latest.ckpt")

        # ── final full eval ───────────────────────────────────────────
        if final_full_eval:
            if val_loader is not None:
                fv_agg, _ = _run_eval(
                    model, iter(val_loader), loss_fn,
                    len(val_loader), final_step, device, autocast_dtype,
                    desc="Final val",
                )
                fv_record = {
                    "step": final_step, "split": "val_full", "elapsed": time.time() - t0,
                    "full_eval": True, "n_batches": len(val_loader),
                    **{k: v for k, v in fv_agg.items() if not isinstance(v, list)},
                }
                log_file.write(json.dumps(fv_record) + "\n")
                metrics_file.write(json.dumps(fv_record) + "\n")
                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, fv_agg, "val_full", final_step)
                _maybe_save_best(fv_record)

            if test_loader is not None:
                ft_agg, ft_vol_por = _run_eval(
                    model, iter(test_loader), loss_fn,
                    len(test_loader), final_step, device, autocast_dtype,
                    desc="Final test",
                )
                ft_record = {
                    "step": final_step, "split": "test_full", "elapsed": time.time() - t0,
                    "full_eval": True, "n_batches": len(test_loader),
                    **{k: v for k, v in ft_agg.items() if not isinstance(v, list)},
                }
                metrics_file.write(json.dumps(ft_record) + "\n")
                if tb_writer is not None:
                    _log_scalars_to_tb(tb_writer, ft_agg, "test_full", final_step)
                    if ft_vol_por:
                        per_vol_maes = torch.tensor(
                            [abs(sum(errs) / len(errs)) for errs in ft_vol_por.values()]
                        )
                        tb_writer.add_histogram(
                            "test/porosity_mae_per_volume", per_vol_maes, final_step
                        )
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_p50",
                            per_vol_maes.quantile(0.5).item(), final_step,
                        )
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_p90",
                            per_vol_maes.quantile(0.9).item(), final_step,
                        )
                        tb_writer.add_scalar(
                            "test/porosity_mae_vol_max",
                            per_vol_maes.max().item(), final_step,
                        )
                _maybe_save_best(ft_record)

    return history


# ── patch sample saving ───────────────────────────────────────────────────────

@torch.no_grad()
def _save_patch_samples(
    model: nn.Module,
    loaders: dict[str, DataLoader | None],
    n_samples: int,
    step: int,
    run_dir: Path,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> None:
    """Save full 3-D patch reconstructions as per-patch TIFF stacks.

    Creates a fresh iterator for each split and deletes it when done so that
    DataLoader workers (and their pinned memory) are released immediately.
    """
    import numpy as np

    samples_dir = run_dir / "samples" / f"step_{step:08d}"
    samples_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for split, loader in loaders.items():
        if loader is None:
            continue

        data_iter = iter(loader)
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

        del data_iter  # release workers and pinned memory for this split

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

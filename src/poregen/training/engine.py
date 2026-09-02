"""Train / eval step helpers and the main training loop."""

from __future__ import annotations

import contextlib
import faulthandler
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

from poregen.models.vae.base import CLASS_PORE, VAEOutput, decode_xct
from poregen.models.discriminator import (
    extract_multiplane_slices,
    lsgan_gen_loss,
    lsgan_disc_loss,
)
from poregen.metrics.seg import (
    multiclass_metrics, porosity_binned_mae, porosity_metrics, segmentation_metrics,
)
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
    """Cycle *loader* forever, refusing to spin on an exhausted loader.

    If an inner pass yields zero batches, the bare ``while True: yield from
    loader`` this replaces degenerates into a silent 100 %-of-one-core busy
    spin: no batch is ever produced, so the caller never logs, never errors
    and never touches the GPU — it just burns CPU indefinitely. That is the
    exact signature vrrae-run-0001 exhibited (hung at step 9852 for 10.5 h,
    16.7 h CPU vs 6.14 h of real training, GPU at 0 %). The DataLoader's own
    ``timeout`` does NOT cover this: that guards a worker that *stalls*, not
    a loader that *ends early* (e.g. workers torn down underneath a
    ``persistent_workers`` loader), which ends iteration cleanly instead.
    """
    while True:
        n = 0
        for batch in loader:
            n += 1
            yield batch
        if n == 0:
            raise RuntimeError(
                "Training DataLoader yielded zero batches on a full pass — "
                "refusing to spin. Its workers most likely died or were torn "
                "down (check for worker OOM/segfault above)."
            )


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

# ── model input contract ──────────────────────────────────────────────────────

def encoder_input_keys(model: nn.Module) -> tuple[str, ...]:
    """Batch keys this VAE's ``forward()`` consumes, in order.

    Variants declare ``encoder_inputs`` when they need something other than the
    historic ``(xct, mask)`` pair — the r08 3-class variant takes
    ``(xct, label)``.  ``getattr`` also sees through a ``torch.compile``
    wrapper.
    """
    return tuple(getattr(model, "encoder_inputs", ("xct", "mask")))


def to_device_inputs(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    """Move the model's inputs to *device*; return the patched batch and args."""
    keys = encoder_input_keys(model)
    moved = {k: batch[k].to(device, non_blocking=True) for k in keys}
    return {**batch, **moved}, tuple(moved[k] for k in keys)


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
    batch_dev, model_args = to_device_inputs(model, batch, device)
    xct = batch_dev["xct"]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(*model_args)
        losses = loss_fn(output, batch_dev, step)

    # ── Adversarial: generator side (float32 outside AMP) ────────────────────
    _fake_slices: torch.Tensor | None = None
    _real_slices: torch.Tensor | None = None
    _gen_adv_loss: torch.Tensor | None = None

    if discriminator is not None and disc_optimizer is not None and disc_weight > 0.0:
        # Cast from AMP dtype (fp16/bf16) to float32 — D runs in float32
        _fake_slices = extract_multiplane_slices(output.xct_out).float()   # (3B,1,64,64)
        _real_slices = extract_multiplane_slices(xct).float()                  # (3B,1,64,64)

        # Generator wants D(fake) → 1; gradients flow through D back to the VAE
        d_fake_gen = discriminator(_fake_slices)
        _gen_adv_loss = lsgan_gen_loss(d_fake_gen)

        # Add to total BEFORE backward so generator gets adversarial signal
        losses["total"] = losses["total"] + disc_weight * _gen_adv_loss

    latent_moments = latent_channel_moments(output.mu)
    _mask_pred_tensor = (
        torch.sigmoid(output.mask_logits).mean().detach()
        if output.mask_logits is not None else None
    )

    scaler.scale(losses["total"]).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    # Per-module AND global gradient norms — computed from ONE foreach pass
    # over every parameter's gradient (each tensor's norm() is otherwise
    # computed twice per step: once by a manual per-module loop, once again
    # inside clip_grad_norm_'s own internal reduction over all parameters).
    # torch._foreach_norm gives one norm per tensor via a single fused op;
    # per-module and global aggregates are then just an algebraic combine
    # (||concat(v1, v2, ...)||_2 == sqrt(sum ||vi||_2^2)), so no gradient
    # tensor's norm is ever computed more than once. Values stay on GPU
    # until the single .item()/clip below.
    _owner_of: dict[int, str] = {}
    for name in ("encoder", "encoder_a", "encoder_b", "decoder", "mask_head"):
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                _owner_of[id(p)] = name

    _all_grads: list[torch.Tensor] = []
    _owner_idx: dict[str, list[int]] = {}
    for p in model.parameters():
        if p.grad is None:
            continue
        idx = len(_all_grads)
        _all_grads.append(p.grad.detach())
        owner = _owner_of.get(id(p))
        if owner is not None:
            _owner_idx.setdefault(owner, []).append(idx)

    if _all_grads:
        _per_tensor_norms = torch.stack(torch._foreach_norm(_all_grads, 2))
        _global_norm_gpu = torch.linalg.vector_norm(_per_tensor_norms, 2)
    else:
        _per_tensor_norms = None
        _global_norm_gpu = torch.zeros((), device=device)

    _norm_gpu: dict[str, torch.Tensor] = {
        f"grad_norm_{name}": torch.linalg.vector_norm(_per_tensor_norms[idxs], 2)
        for name, idxs in _owner_idx.items()
    }
    module_grad_norms: dict[str, float] = {k: v.item() for k, v in _norm_gpu.items()}

    if _all_grads:
        # Equivalent to clip_grad_norm_(model.parameters(), max_norm) but
        # reuses _global_norm_gpu instead of recomputing the same reduction.
        torch.nn.utils.clip_grads_with_norm_(
            model.parameters(),
            max_grad_norm if max_grad_norm is not None else float("inf"),
            _global_norm_gpu,
        )
    grad_norm = _global_norm_gpu.item()

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

        # Real and fake go through ONE forward. Two forwards before one
        # backward break under torch.compile: spectral norm's power iteration
        # updates its u/v buffers in-place on the second forward, and AOT
        # autograd saves the buffer itself (eager clones it), so the first
        # forward's graph sees a version mismatch at backward. Safe to concat:
        # the discriminator has no batch norm, scores are per-sample.
        _n_real = _real_slices.shape[0]
        d_all = discriminator(
            torch.cat([_real_slices, _fake_slices.detach()], dim=0)
        )                                                           # float32, with grad
        d_real, d_fake = d_all[:_n_real], d_all[_n_real:]

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
    if _mask_pred_tensor is not None:
        result["mask_pred_mean"] = float(_mask_pred_tensor.item())
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
) -> tuple[dict[str, Any], VAEOutput, torch.Tensor, torch.Tensor]:
    """Single eval step (no grad, AMP for speed).

    Returns the loss dict (scalars + kl_per_channel list), the VAEOutput, the
    device-side XCT, and the device-side segmentation input — the binary mask
    for a binary-head variant, the int64 class label for a 3-class one.
    Returning them avoids redundant H→D transfers in the caller.
    """
    model.eval()
    batch_dev, model_args = to_device_inputs(model, batch, device)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(*model_args)
        losses = loss_fn(output, batch_dev, step)

    return ({k: _to_scalar(v) for k, v in losses.items()}, output,
            batch_dev["xct"], model_args[-1])


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
        losses, output, xct_dev, mask_dev = eval_step(
            model, batch, loss_fn, step, device, autocast_dtype
        )
        _accumulate(loss_acc, losses)

        # The XCT head regresses xct/255 directly — decode_xct clamps, it does
        # NOT activate.  (Until 2026-09-01 this line applied a sigmoid by
        # analogy with the mask head, which put a ~0.135 artefact floor under
        # val/mae and crushed sharpness_recon_over_gt by the sigmoid slope.)
        xct_recon = decode_xct(output.xct_out)

        if output.mask_logits is not None:
            mask_sigmoid = torch.sigmoid(output.mask_logits)

            # Segmentation metrics — pass pre-activated to skip internal sigmoid
            seg = segmentation_metrics(mask_sigmoid, mask_dev, apply_sigmoid=False)
            _accumulate(seg_acc, seg)

            # Porosity metrics — pass pre-activated to skip internal sigmoid
            por = porosity_metrics(mask_sigmoid, mask_dev, apply_sigmoid=False)
            _accumulate(por_acc, por)

            # Per-volume porosity tracking — accumulate tensors, defer .item()
            pred_por_v = mask_sigmoid.mean(dim=(1, 2, 3, 4))   # (B,)
            gt_por_v   = mask_dev.mean(dim=(1, 2, 3, 4))        # (B,)
            pred_por_signed_all.append((pred_por_v - gt_por_v).detach())
            pred_por_all.append(pred_por_v.detach())
            gt_por_all.append(gt_por_v.detach())
            vol_ids_all.append(list(batch["volume_id"]))
        elif output.class_logits is not None:
            mask_sigmoid = None
            # mask_dev is the int64 class label for a 3-class variant.  Every
            # number here comes off the ARGMAX — the label a generated volume
            # actually carries — not off the soft probabilities.
            _accumulate(seg_acc, multiclass_metrics(output.class_logits, mask_dev))

            pred_lab   = output.class_logits.argmax(dim=1)
            pred_por_v = (pred_lab == CLASS_PORE).flatten(1).float().mean(1)
            gt_por_v   = (mask_dev == CLASS_PORE).flatten(1).float().mean(1)
            pred_por_signed_all.append((pred_por_v - gt_por_v).detach())
            pred_por_all.append(pred_por_v.detach())
            gt_por_all.append(gt_por_v.detach())
            vol_ids_all.append(list(batch["volume_id"]))
            del pred_lab
        else:
            mask_sigmoid = None

        # Reconstruction metrics — on the decoded grey level, same scale as the target
        mae_acc.append(F.l1_loss(xct_recon, xct_dev))
        sharp_recon_acc.append(sharpness_proxy(xct_recon))
        sharp_gt_acc.append(sharpness_proxy(xct_dev))

        # Latent metrics
        lat = latent_stats(output.mu, output.logvar)
        _accumulate(latent_acc, lat)
        latent_moment_summaries.append(latent_channel_moments(output.mu))

        # Explicitly release large GPU tensors — Python refcounting usually
        # handles this, but being explicit prevents accidental retention and
        # ensures the allocator can reuse the memory for the next batch.
        del output, mask_sigmoid, xct_recon, mask_dev, xct_dev

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

    # Porosity-binned MAE across the full eval set (skipped when the model has
    # no mask head — e.g. VRRAE's XCT-only decoder — since nothing was accumulated).
    if pred_por_all:
        all_pred_por = torch.cat(pred_por_all)
        all_gt_por   = torch.cat(gt_por_all)
        binned_mae_dict = porosity_binned_mae(all_pred_por, all_gt_por)
    else:
        binned_mae_dict = {}

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
    """Keep a small, fixed batch on CPU for lightweight reconstruction logging.

    EVERY tensor entry is kept, not a hand-listed pair: the consumers call
    :func:`to_device_inputs`, so the snapshot has to carry whatever the model
    declares in ``encoder_inputs`` — ``label`` for a 3-class variant, ``mask``
    for a binary one — plus ``mask`` for the ground-truth image panel. Listing
    keys here means a new variant crashes hundreds of steps into a run, at the
    first Monte-Carlo eval, which is exactly what it did.
    """
    n_take = min(max(1, n_examples), batch["xct"].shape[0])
    return {k: v[:n_take].cpu() for k, v in batch.items()
            if isinstance(v, torch.Tensor)}


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
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "val.xct_loss",
    early_stopping_mode: str = "min",
    early_stopping_min_delta: float = 0.0,
    early_stopping_warmup_steps: int = 0,
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
        # no-cudagraphs: keeps max-autotune's Triton kernel tuning (where the
        # speedup for large 3D convs comes from) but disables CUDA-graph-tree
        # replay, which cannot handle this loop — the train-mode graph emits
        # the BatchNorm running-stat buffers as graph outputs, and the first
        # eval-mode invocation then trips "accessing tensor output of
        # CUDAGraphs that has been overwritten" when it takes them as inputs.
        model = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)  # type: ignore[assignment]
        # The discriminator stays eager: it is invoked twice per step (generator
        # adversarial path + its own update) and uses spectral norm, whose
        # in-place power-iteration buffers break AOT autograd's saved tensors,
        # while max-autotune's CUDA graphs overwrite the first invocation's
        # output buffers on the second. It is a tiny 2D net (~661K params), so
        # compiling it bought nothing anyway.
        loss_fn = torch.compile(loss_fn, mode="max-autotune-no-cudagraphs")  # type: ignore[assignment]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path     = run_dir / "log.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    montecarlo_every = image_log_every if montecarlo_every is None else montecarlo_every
    best_target_split, best_target_key = _parse_metric_target(best_metric)
    best_metric_value: float | None = None
    early_target_split, early_target_key = _parse_metric_target(early_stopping_metric)
    early_best_value: float | None = None
    early_no_improve = 0
    stop_requested = False
    if early_stopping_mode not in {"min", "max"}:
        raise ValueError("early_stopping_mode must be 'min' or 'max'")

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

    def _update_early_stopping(record: dict[str, Any]) -> bool:
        nonlocal early_best_value, early_no_improve
        if early_stopping_patience <= 0:
            return False
        if int(record["step"]) + 1 < early_stopping_warmup_steps:
            return False
        if record.get("split") != early_target_split or early_target_key is None:
            return False
        metric_value = record.get(early_target_key)
        if metric_value is None:
            return False

        current = float(metric_value)
        if early_best_value is None:
            improved = True
        elif early_stopping_mode == "min":
            improved = current < early_best_value - early_stopping_min_delta
        else:
            improved = current > early_best_value + early_stopping_min_delta

        if improved:
            early_best_value = current
            early_no_improve = 0
            return False

        early_no_improve += 1
        return early_no_improve >= early_stopping_patience

    # Hang watchdog. vrrae-run-0001 wedged mid-epoch at step 9852 and sat
    # there for 10.5 h; because the process stayed alive and ptrace_scope=1
    # blocks py-spy/gdb without root, there was no way to recover a stack
    # trace after the fact. faulthandler needs no privileges: the timer is
    # re-armed below on every iteration, so it only ever fires if a single
    # iteration (batch fetch + train step + any eval/checkpoint) exceeds the
    # budget, dumping every thread's Python stack to stderr and continuing.
    # Budget is deliberately far above the slowest legitimate iteration
    # observed (452 s, a full-validation epoch boundary).
    _WATCHDOG_SECONDS = 1800.0
    faulthandler.enable()

    # Use ExitStack so both log files are always closed — even on exception.
    with contextlib.ExitStack() as stack:
        log_file     = stack.enter_context(open(log_path,     "a"))
        metrics_file = stack.enter_context(open(metrics_path, "a"))
        stack.callback(faulthandler.cancel_dump_traceback_later)

        for step in pbar:
            # Re-arm: fires only if THIS iteration overruns the budget.
            faulthandler.dump_traceback_later(_WATCHDOG_SECONDS, exit=False)

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
                stop_requested = _update_early_stopping(val_record) or stop_requested

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
                stop_requested = _update_early_stopping(fv_record) or stop_requested

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


            if stop_requested:
                stop_record = {
                    "step": step,
                    "split": "event",
                    "event": "early_stopping",
                    "metric": early_stopping_metric,
                    "best_value": early_best_value,
                    "checks_without_improvement": early_no_improve,
                    "patience": early_stopping_patience,
                    "elapsed": time.time() - t0,
                }
                history.append(stop_record)
                log_file.write(json.dumps(stop_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(stop_record) + "\n")
                metrics_file.flush()
                _logger.info("Early stopping at step %d: %s", step + 1, stop_record)
                break
        # ── final checkpoint — wait for any in-flight background save ─
        if _ckpt_thread_holder and _ckpt_thread_holder[0] is not None:
            _ckpt_thread_holder[0].join()

        # VRRAE follows the reference's two-stage lifecycle: optimize with
        # per-batch SVDs, then derive U_f from a separate full training-set
        # pass before saving or running final validation/test.
        basis_finalization = None
        finalize_inference_basis = getattr(
            model,
            "finalize_inference_basis",
            None,
        )
        if callable(finalize_inference_basis):
            del train_iter
            basis_finalization = finalize_inference_basis(
                train_loader,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            _logger.info(
                "Finalized VRRAE U_f from %d samples in %d batches",
                basis_finalization["n_samples"],
                basis_finalization["n_batches"],
            )

        final_step = step + 1
        ckpt_name = f"{run_dir.name}_step{final_step:08d}.ckpt"
        final_ckpt_path = save_checkpoint(
            run_dir / ckpt_name,
            model, optimizer, scaler, step=final_step,
            metadata={
                "total_steps": total_steps,
                "basis_finalization": basis_finalization,
                "stopped_early": stop_requested,
                "early_stopping_metric": early_stopping_metric,
                "early_stopping_best_value": early_best_value,
                "early_stopping_checks_without_improvement": early_no_improve,
                "planned_final_step": start_step + total_steps,
                "actual_final_step": final_step,
            },
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

            sub = {k: v[:n_take] for k, v in batch.items()
                   if isinstance(v, torch.Tensor)}
            _, model_args = to_device_inputs(model, sub, device)
            xct = model_args[0]
            mask = batch["mask"][:n_take].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(*model_args)

            xct_gts.append(xct.cpu().float().numpy())
            mask_gts.append(mask.cpu().float().numpy())
            xct_recons.append(decode_xct(output.xct_out).cpu().float().numpy()[:n_take])
            if output.mask_logits is not None:
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

        xct_recon_arr = np.concatenate(xct_recons)
        # export_patch_sample_split requires a mask_recon array for every variant;
        # models with no mask head (e.g. VRRAE) have nothing to put there, so
        # export zeros rather than change the shared export contract.
        mask_recon_arr = np.concatenate(mask_recons) if mask_recons else np.zeros_like(xct_recon_arr)
        export_patch_sample_split(
            samples_dir / split,
            {
                "xct_gt":    np.concatenate(xct_gts),
                "mask_gt":   np.concatenate(mask_gts),
                "xct_recon": xct_recon_arr,
                "mask_recon": mask_recon_arr,
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
        xct_recon  = decode_xct(output.xct_out)
        xct_gt  = batch["xct"].to(device, non_blocking=True)

        pairs = [("xct", xct_gt, xct_recon)]
        if output.mask_logits is not None:
            mask_recon = torch.sigmoid(output.mask_logits)
            mask_gt = batch["mask"].to(device, non_blocking=True)
            pairs.append(("mask", mask_gt, mask_recon))

        for tag, gt_vol, recon_vol in pairs:
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
    _, model_args = to_device_inputs(model, batch, device)
    xct = model_args[0]
    # Ground-truth panel for the mask images below; only read when the variant
    # has a binary mask head (has_mask), but the loader always provides it.
    mask = batch["mask"].to(device, non_blocking=True)

    has_mask = None  # set from the first forward pass below
    xct_samples:  list[torch.Tensor] = []
    mask_samples: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(*model_args)
            if has_mask is None:
                has_mask = output.mask_logits is not None
            xct_samples.append(decode_xct(output.xct_out).float())
            if has_mask:
                mask_samples.append(torch.sigmoid(output.mask_logits).float())

    xct_stack  = torch.stack(xct_samples,  dim=0)   # (N, B, 1, D, H, W)
    xct_mean  = xct_stack.mean(dim=0)
    xct_std   = xct_stack.std(dim=0)

    if has_mask:
        mask_stack = torch.stack(mask_samples, dim=0)
        mask_mean = mask_stack.mean(dim=0)
        mask_std  = mask_stack.std(dim=0)

    _DIVERSITY_EPS = 1e-5
    xct_div  = xct_std.mean().item()
    mask_div = mask_std.mean().item() if has_mask else None
    if xct_div < _DIVERSITY_EPS or (mask_div is not None and mask_div < _DIVERSITY_EPS):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "run_montecarlo_eval step=%d: predictions nearly identical "
            "(xct_div=%.2e  mask_div=%s). "
            "Check reparameterization / posterior collapse.",
            step, xct_div, f"{mask_div:.2e}" if mask_div is not None else "n/a",
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
        writer.add_images(f"montecarlo/xct_std_{axis}",   _std_to_rgb(slicer(xct_std)),  step)
        if has_mask:
            writer.add_images(f"montecarlo/mask_mean_{axis}", slicer(mask_mean), step)
            writer.add_images(f"montecarlo/mask_std_{axis}",  _std_to_rgb(slicer(mask_std)), step)

    xct_recon_single  = xct_samples[0]
    xct_gt_clamped    = xct.clamp(0.0, 1.0)

    for axis, slicer in [
        ("d", _central_slice_d),
        ("h", _central_slice_h),
        ("w", _central_slice_w),
    ]:
        writer.add_images(f"montecarlo/xct_recon_{axis}",  slicer(xct_recon_single),  step)
        writer.add_images(f"montecarlo/xct_gt_{axis}",     slicer(xct_gt_clamped),    step)
        if has_mask:
            mask_recon_single = mask_samples[0]
            mask_gt_clamped   = mask.clamp(0.0, 1.0)
            writer.add_images(f"montecarlo/mask_recon_{axis}", slicer(mask_recon_single), step)
            writer.add_images(f"montecarlo/mask_gt_{axis}",    slicer(mask_gt_clamped),   step)

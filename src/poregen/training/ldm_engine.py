"""LDM train/eval step helpers and training loop."""

from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from poregen.training.checkpoint import copy_checkpoint, save_checkpoint, save_checkpoint_async

import logging as _logging
_logger = _logging.getLogger(__name__)


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMAModel:
    """Exponential moving average of model parameters (Ho et al. 2020, Rombach et al. 2022).

    Weights are stored in float32 regardless of training dtype so that the
    accumulation never loses precision. State dict keys match the base model
    exactly — trivial to save and restore alongside a regular checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().detach().float()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().float() for k, v in state.items()}

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into *model* in-place (for inference)."""
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()}
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_scalar(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().item()
    return float(v)


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _accumulate(acc: dict, new: dict) -> None:
    for k, v in new.items():
        acc.setdefault(k, 0.0)
        acc[k] += float(v)


def _mean_acc(acc: dict, n: int) -> dict[str, float]:
    return {k: v / n for k, v in acc.items()}


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ── single-step helpers ───────────────────────────────────────────────────────

def ldm_train_step(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    schedule: Any,
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.bfloat16,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
) -> dict[str, float]:
    """Single LDM training step.

    Returns
    -------
    {"loss": float, "grad_norm": float}
    """
    model.train()
    b = _batch_to_device(batch, device)
    z          = b["z"]            # (B, C, D, H, W)
    nb_latents = b["nb_latents"]   # (B, 6, C, D, H, W)
    nb_avail   = b["nb_avail"]     # (B, 6) long
    pos_frac   = b["pos_frac"]     # (B, 3)
    global_por = b["global_por"].squeeze(1)  # (B,)
    local_por  = b["local_por"].squeeze(1)   # (B,)

    B = z.shape[0]
    t      = torch.randint(0, schedule.T, (B,), device=device)
    noise  = torch.randn_like(z)
    z_t    = schedule.q_sample(z, t, noise)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        eps_pred = model(z_t, t, nb_latents, nb_avail, pos_frac, global_por, local_por)
        loss     = F.mse_loss(eps_pred, noise)

    scaler.scale(loss).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm is not None else float("inf"),
    ).item()

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return {"loss": loss.item(), "grad_norm": grad_norm}


@torch.no_grad()
def ldm_eval_step(
    model: nn.Module,
    batch: dict[str, Any],
    schedule: Any,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Single LDM eval step (no grad)."""
    model.eval()
    b = _batch_to_device(batch, device)
    z          = b["z"]
    nb_latents = b["nb_latents"]
    nb_avail   = b["nb_avail"]
    pos_frac   = b["pos_frac"]
    global_por = b["global_por"].squeeze(1)
    local_por  = b["local_por"].squeeze(1)

    B = z.shape[0]
    t      = torch.randint(0, schedule.T, (B,), device=device)
    noise  = torch.randn_like(z)
    z_t    = schedule.q_sample(z, t, noise)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        eps_pred = model(z_t, t, nb_latents, nb_avail, pos_frac, global_por, local_por)
        loss     = F.mse_loss(eps_pred, noise)

    return {"loss": loss.item()}


def _run_eval(
    model: nn.Module,
    data_iter: Iterator,
    schedule: Any,
    n_batches: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    desc: str = "Eval",
) -> dict[str, float]:
    acc: dict[str, float] = {}
    for _ in tqdm(range(n_batches), desc=desc, leave=False, unit="batch"):
        metrics = ldm_eval_step(model, next(data_iter), schedule, device, autocast_dtype)
        _accumulate(acc, metrics)
    return _mean_acc(acc, n_batches)


# ── training loop ─────────────────────────────────────────────────────────────

def ldm_train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    schedule: Any,
    cfg: dict[str, Any],
    run_dir: str | Path,
    tb_writer: Any | None = None,
    start_step: int = 0,
    device: torch.device | None = None,
    autocast_dtype: torch.dtype = torch.bfloat16,
    scheduler: Any | None = None,
    ema_decay: float = 0.9999,
    ema_state: dict | None = None,
) -> list[dict[str, Any]]:
    """Step-based LDM training loop mirroring the VAE train_loop.

    Reads from cfg["training"]:
      total_steps, log_every, eval_every, val_batches, save_every,
      max_grad_norm, compile
    """
    if device is None:
        device = next(model.parameters()).device

    training_cfg = cfg.get("training", {})
    total_steps  = int(training_cfg["total_steps"])
    log_every    = int(training_cfg.get("log_every",  10))
    eval_every   = int(training_cfg.get("eval_every", 500))
    val_batches  = int(training_cfg.get("val_batches", 50))
    save_every   = int(training_cfg.get("save_every", 5000))
    max_grad_norm = training_cfg.get("max_grad_norm")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    compile_model = bool(training_cfg.get("compile", False))
    if compile_model:
        model = torch.compile(model, mode="max-autotune", dynamic=False)  # type: ignore[assignment]

    ema = EMAModel(model, decay=ema_decay)
    if ema_state is not None:
        ema.load_state_dict(ema_state)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    log_path     = run_dir / "log.jsonl"
    metrics_path = run_dir / "metrics.jsonl"

    best_val_loss:      float | None = None
    train_iter = _infinite(train_loader)
    history:   list[dict[str, Any]] = []
    t0 = time.time()

    _ckpt_thread_holder: list = [None]

    loss_window: deque[float] = deque(maxlen=max(1, log_every))

    pbar = tqdm(range(start_step, start_step + total_steps), desc="LDM Training")

    with contextlib.ExitStack() as stack:
        log_file     = stack.enter_context(open(log_path,     "a"))
        metrics_file = stack.enter_context(open(metrics_path, "a"))

        for step in pbar:
            batch = next(train_iter)
            _step_t0 = time.perf_counter()
            metrics = ldm_train_step(
                model, batch, optimizer, scaler, schedule,
                step=step, device=device, autocast_dtype=autocast_dtype,
                max_grad_norm=max_grad_norm, scheduler=scheduler,
            )
            ema.update(model)
            _step_elapsed = time.perf_counter() - _step_t0
            steps_per_sec = 1.0 / _step_elapsed if _step_elapsed > 0 else float("inf")

            if step == start_step and torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                _logger.info("Step %d — peak GPU memory: %.2f GB", step, peak_mem_gb)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/peak_gpu_mem_gb", peak_mem_gb, step)

            loss_window.append(metrics["loss"])
            record = {
                "step":         step,
                "split":        "train",
                "elapsed":      time.time() - t0,
                "step_time_ms": _step_elapsed * 1000.0,
                **metrics,
            }
            history.append(record)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", grad=f"{metrics['grad_norm']:.3f}")

            if tb_writer is not None and (step + 1) % log_every == 0:
                tb_writer.add_scalar("train/loss",      metrics["loss"],      step)
                tb_writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
                tb_writer.add_scalar("train/steps_per_sec", steps_per_sec,    step)
                if scheduler is not None:
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

            # ── validation ──────────────────────────────────────────────────
            if val_loader is not None and (step + 1) % eval_every == 0:
                val_batches_clamped = min(val_batches, len(val_loader))
                _val_iter = iter(val_loader)
                agg = _run_eval(
                    model, _val_iter, schedule, val_batches_clamped,
                    device, autocast_dtype, desc=f"Val step {step + 1}",
                )
                del _val_iter
                val_record = {
                    "step":    step,
                    "split":   "val",
                    "elapsed": time.time() - t0,
                    **agg,
                }
                history.append(val_record)
                log_file.write(json.dumps(val_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(val_record) + "\n")
                metrics_file.flush()

                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss", agg["loss"], step)

                val_loss = agg["loss"]
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        ckpt_dir / "best.ckpt",
                        model, optimizer, scaler, step=step + 1,
                        metadata={"best_val_loss": best_val_loss},
                        scheduler=scheduler,
                        ema_state_dict=ema.state_dict(),
                    )
                    _logger.info("New best val loss=%.6f at step %d → best.ckpt", best_val_loss, step + 1)

            # ── checkpoint ──────────────────────────────────────────────────
            if (step + 1) % save_every == 0:
                ckpt_name = f"ldm_step{step + 1:08d}.ckpt"
                save_checkpoint_async(
                    ckpt_dir / ckpt_name,
                    model, optimizer, scaler, step=step + 1,
                    metadata={"total_steps": total_steps},
                    scheduler=scheduler,
                    latest_path=ckpt_dir / "latest.ckpt",
                    thread_holder=_ckpt_thread_holder,
                    ema_state_dict=ema.state_dict(),
                )

        # ── final checkpoint ─────────────────────────────────────────────────
        if _ckpt_thread_holder and _ckpt_thread_holder[0] is not None:
            _ckpt_thread_holder[0].join()

        final_step = start_step + total_steps
        final_ckpt = save_checkpoint(
            ckpt_dir / f"ldm_step{final_step:08d}.ckpt",
            model, optimizer, scaler, step=final_step,
            metadata={"total_steps": total_steps},
            scheduler=scheduler,
            ema_state_dict=ema.state_dict(),
        )
        copy_checkpoint(final_ckpt, ckpt_dir / "latest.ckpt")

    return history

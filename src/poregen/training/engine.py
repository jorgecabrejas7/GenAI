"""Train / eval step helpers and a minimal training loop for notebooks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from poregen.models.vae.base import VAEOutput
from poregen.training.checkpoint import save_checkpoint


# ── single-step helpers ──────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: Callable[..., dict[str, torch.Tensor | float]],
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float16,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
) -> dict[str, float]:
    """Single training step with AMP, optional gradient clipping, and scheduler.

    Parameters
    ----------
    loss_fn
        Callable ``(output, batch, step) -> dict`` returning at least
        ``"total"`` key.
    max_grad_norm
        If not None, clip gradient global norm to this value.
    scheduler
        If not None, ``scheduler.step()`` is called after the optimizer step.

    Returns
    -------
    dict of scalar loss components (detached, float).
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

    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return {k: (v.detach().item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in losses.items()}


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: Callable[..., dict[str, torch.Tensor | float]],
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, float], VAEOutput]:
    """Single eval step (no grad, AMP for speed).

    Returns the loss dict and the VAEOutput for downstream metrics.
    """
    model.eval()
    xct = batch["xct"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    batch_dev = {**batch, "xct": xct, "mask": mask}

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        output: VAEOutput = model(xct, mask)
        losses = loss_fn(output, batch_dev, step)

    scalars = {k: (v.detach().item() if isinstance(v, torch.Tensor) else float(v))
               for k, v in losses.items()}
    return scalars, output


# ── infinite iterator helper ─────────────────────────────────────────────

def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


# ── training loop ────────────────────────────────────────────────────────

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: Callable,
    *,
    total_steps: int = 200,
    eval_every: int = 50,
    save_every: int = 100,
    run_dir: str | Path = "runs/vae/default",
    device: torch.device = torch.device("cpu"),
    autocast_dtype: torch.dtype = torch.float16,
    start_step: int = 0,
    log_jsonl: bool = True,
    max_grad_norm: float | None = None,
    scheduler: Any | None = None,
    tb_writer: Any | None = None,
) -> list[dict[str, Any]]:
    """Minimal training loop for notebook use.

    Parameters
    ----------
    total_steps : int
        Number of optimiser updates.
    eval_every : int
        Run a single validation batch every *eval_every* steps.
    save_every : int
        Save a checkpoint every *save_every* steps.
    run_dir : Path
        Directory for checkpoints and logs.
    log_jsonl : bool
        If True, append per-step metrics to ``{run_dir}/log.jsonl``.
    max_grad_norm : float, optional
        If set, clip gradient norms to this value.
    scheduler : optional
        LR scheduler to step after each optimizer update.
    tb_writer : optional
        ``torch.utils.tensorboard.SummaryWriter`` instance.

    Returns
    -------
    list of per-step loss dicts (for quick inline plotting).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl" if log_jsonl else None
    if log_path:
        log_file = open(log_path, "a")

    train_iter = _infinite(train_loader)
    history: list[dict[str, Any]] = []
    val_iter = _infinite(val_loader) if val_loader is not None else None

    pbar = tqdm(range(start_step, start_step + total_steps), desc="Training")
    t0 = time.time()

    for step in pbar:
        batch = next(train_iter)
        losses = train_step(
            model, batch, optimizer, scaler, loss_fn,
            step=step, device=device, autocast_dtype=autocast_dtype,
            max_grad_norm=max_grad_norm, scheduler=scheduler,
        )

        record = {"step": step, "split": "train", "elapsed": time.time() - t0, **losses}
        history.append(record)
        if log_path:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        # TensorBoard scalars
        if tb_writer is not None:
            for k, v in losses.items():
                tb_writer.add_scalar(f"train/{k}", v, step)
            if scheduler is not None:
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        pbar.set_postfix(total=f"{losses['total']:.4f}", kl=f"{losses.get('kl', 0):.4f}")

        # ── eval ──
        if val_iter is not None and (step + 1) % eval_every == 0:
            val_batch = next(val_iter)
            val_losses, _ = eval_step(
                model, val_batch, loss_fn,
                step=step, device=device, autocast_dtype=autocast_dtype,
            )
            val_record = {"step": step, "split": "val", "elapsed": time.time() - t0, **val_losses}
            history.append(val_record)
            if log_path:
                log_file.write(json.dumps(val_record) + "\n")
                log_file.flush()
            if tb_writer is not None:
                for k, v in val_losses.items():
                    tb_writer.add_scalar(f"val/{k}", v, step)

        # ── checkpoint ──
        if (step + 1) % save_every == 0:
            save_checkpoint(
                run_dir / "last.ckpt",
                model, optimizer, scaler, step=step + 1,
                metadata={"total_steps": total_steps},
                scheduler=scheduler,
            )

    # Final checkpoint
    save_checkpoint(
        run_dir / "last.ckpt",
        model, optimizer, scaler, step=start_step + total_steps,
        metadata={"total_steps": total_steps},
        scheduler=scheduler,
    )

    if log_path:
        log_file.close()

    return history

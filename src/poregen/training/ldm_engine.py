"""LDM train/eval step helpers and training loop."""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np
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
        device = next(iter(self.shadow.values())).device if self.shadow else torch.device("cpu")
        self.shadow = {k: v.clone().float().to(device) for k, v in state.items()}

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


@torch.no_grad()
def _run_full_eval(
    model: nn.Module,
    val_loader: Any,
    schedule: Any,
    device: torch.device,
    autocast_dtype: torch.dtype,
    desc: str = "Full val",
) -> dict[str, float]:
    """Pass through the entire validation set."""
    acc: dict[str, float] = {}
    n = 0
    for batch in tqdm(val_loader, desc=desc, leave=False, unit="batch"):
        _accumulate(acc, ldm_eval_step(model, batch, schedule, device, autocast_dtype))
        n += 1
    return _mean_acc(acc, max(n, 1))


# ── sample visualisation ─────────────────────────────────────────────────────

def _gaussian_por_grid(
    grid: tuple[int, int, int],
    global_por: float,
    sigma: float = 1.0,
) -> dict[tuple[int, int, int], float]:
    """Return per-patch local porosity values following a 3-D Gaussian centred on
    the volume, normalised so the mean over all patches equals *global_por*."""
    nz, ny, nx = grid
    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    weights: dict[tuple[int, int, int], float] = {}
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                d2 = (iz - cz) ** 2 + (iy - cy) ** 2 + (ix - cx) ** 2
                weights[(iz, iy, ix)] = math.exp(-d2 / (2.0 * sigma ** 2))
    total_w = sum(weights.values())
    scale = (global_por * nz * ny * nx) / total_w
    return {k: v * scale for k, v in weights.items()}


@torch.no_grad()
def _log_sample_volume(
    *,
    model: nn.Module,
    ema: "EMAModel",
    schedule: Any,
    vae: nn.Module | None,
    step: int,
    run_dir: Path,
    tb_writer: Any | None,
    device: torch.device,
    autocast_dtype: torch.dtype,
    ddim_steps: int = 20,
    grid: tuple[int, int, int] = (3, 3, 3),
    global_por: float = 0.02,
    latent_std: float = 1.0,
    patch_size: int = 64,
    patch_stride: int = 64,
    latent_size: int = 16,
) -> None:
    """Generate a small volume with DDIM, decode with VAE if available.

    When vae is None the raw latents are logged as grayscale heatmaps instead
    of decoded XCT/mask — sample generation always runs regardless.
    """
    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator

    _logger.info(
        "Generating sample volume at step %d  grid=%s  DDIM steps=%d  vae=%s",
        step, grid, ddim_steps, "yes" if vae is not None else "no (raw latents)",
    )

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply_to(model)
    model.eval()

    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    step_dir = samples_dir / f"step_{step:08d}"
    step_dir.mkdir(exist_ok=True)

    try:
        sampler = DDIMSampler(model, schedule, device, n_steps=ddim_steps)
        local_por_map = _gaussian_por_grid(grid, global_por)
        vol_shape: tuple[int, int, int] = tuple(  # type: ignore[assignment]
            (g - 1) * patch_stride + patch_size for g in grid
        )

        if vae is not None:
            generator = VolumeGenerator(
                sampler=sampler,
                vae=vae,
                device=device,
                patch_size=patch_size,
                patch_stride=patch_stride,
                latent_size=latent_size,
                latent_std=latent_std,
            )
            xct, mask = generator.generate(
                volume_shape=vol_shape,
                target_porosity=global_por,
                local_por_map=local_por_map,
                autocast_dtype=autocast_dtype,
            )
            VolumeGenerator.save_tiff(xct, mask, step_dir / "xct.tif", step_dir / "mask.tif")
            _logger.info("Saved sample TIFFs → %s", step_dir)

            if tb_writer is not None:
                D, H, W = xct.shape

                def _img(arr2d: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(arr2d.astype(np.float32) / 255.0).unsqueeze(0)

                tb_writer.add_image("samples/xct_dslice",  _img(xct[D // 2]),       step)
                tb_writer.add_image("samples/xct_hslice",  _img(xct[:, H // 2]),    step)
                tb_writer.add_image("samples/xct_wslice",  _img(xct[:, :, W // 2]), step)
                tb_writer.add_image("samples/mask_dslice", _img(mask[D // 2]),       step)
                tb_writer.add_image("samples/mask_hslice", _img(mask[:, H // 2]),    step)
                tb_writer.add_image("samples/mask_wslice", _img(mask[:, :, W // 2]), step)

        else:
            # No VAE — generate latents (via the shared two-phase checkerboard
            # schedule) and log channel-mean heatmaps without decoding.
            generator = VolumeGenerator(
                sampler=sampler,
                vae=None,  # type: ignore[arg-type]
                device=device,
                patch_size=patch_size,
                patch_stride=patch_stride,
                latent_size=latent_size,
                latent_std=latent_std,
            )
            generated, grid_origins = generator._generate_latents(
                volume_shape=vol_shape,
                target_porosity=global_por,
                local_por_map=local_por_map,
                autocast_dtype=autocast_dtype,
            )

            # Assemble channel-mean latent volume and save as npy + log to TB
            C, LS = sampler.model.cfg.z_channels, latent_size
            lat_vol = np.zeros(vol_shape, dtype=np.float32)
            wgt_vol = np.zeros(vol_shape, dtype=np.float32)
            for gi, z_gen in generated.items():
                z0, y0, x0 = grid_origins[gi]
                ze, ye, xe = z0 + patch_size, y0 + patch_size, x0 + patch_size
                lat_vol[z0:ze, y0:ye, x0:xe] += z_gen.float().cpu().mean(0).numpy()
                wgt_vol[z0:ze, y0:ye, x0:xe] += 1.0
            lat_vol /= np.maximum(wgt_vol, 1e-8)
            np.save(step_dir / "latents_mean.npy", lat_vol)
            _logger.info("Saved raw latent volume (channel mean) → %s", step_dir)

            if tb_writer is not None:
                D, H, W = lat_vol.shape
                def _norm(arr2d: np.ndarray) -> torch.Tensor:
                    lo, hi = arr2d.min(), arr2d.max()
                    norm = (arr2d - lo) / max(hi - lo, 1e-8)
                    return torch.from_numpy(norm).unsqueeze(0)
                tb_writer.add_image("samples/latent_dslice", _norm(lat_vol[D // 2]),       step)
                tb_writer.add_image("samples/latent_hslice", _norm(lat_vol[:, H // 2]),    step)
                tb_writer.add_image("samples/latent_wslice", _norm(lat_vol[:, :, W // 2]), step)

    finally:
        model.load_state_dict(orig_state)


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
    vae: nn.Module | None = None,
    latent_std: float = 1.0,
) -> list[dict[str, Any]]:
    """Step-based LDM training loop mirroring the VAE train_loop.

    Reads from cfg["training"]:
      total_steps, log_every, eval_every, val_batches, save_every,
      max_grad_norm, compile,
      full_val_every,
      sample_every, sample_ddim_steps, sample_grid, sample_global_por
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

    full_val_every    = int(training_cfg.get("full_val_every",    0))
    sample_every      = int(training_cfg.get("sample_every", 0))
    sample_ddim_steps = int(training_cfg.get("sample_ddim_steps", 20))
    _raw_grid         = training_cfg.get("sample_grid", [3, 3, 3])
    sample_grid       = tuple(int(x) for x in _raw_grid)
    sample_global_por = float(training_cfg.get("sample_global_por", 0.02))
    sample_patch_stride = int(cfg.get("data", {}).get("patch_stride", 64))

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

            # ── full validation pass ─────────────────────────────────────────
            if val_loader is not None and full_val_every > 0 and (step + 1) % full_val_every == 0:
                full_agg = _run_full_eval(
                    model, val_loader, schedule, device, autocast_dtype,
                    desc=f"Full val {step + 1}",
                )
                full_record = {
                    "step":    step,
                    "split":   "val_full",
                    "elapsed": time.time() - t0,
                    **full_agg,
                }
                history.append(full_record)
                log_file.write(json.dumps(full_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(full_record) + "\n")
                metrics_file.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar("val_full/loss", full_agg["loss"], step)
                _logger.info("Full val step %d — loss=%.6f", step + 1, full_agg["loss"])

            # ── sample visualisation ─────────────────────────────────────────
            if sample_every > 0 and (step + 1) % sample_every == 0:
                try:
                    _log_sample_volume(
                        model=model,
                        ema=ema,
                        schedule=schedule,
                        vae=vae,
                        step=step,
                        run_dir=run_dir,
                        tb_writer=tb_writer,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        ddim_steps=sample_ddim_steps,
                        grid=sample_grid,
                        global_por=sample_global_por,
                        latent_std=latent_std,
                        patch_stride=sample_patch_stride,
                    )
                except Exception as _exc:
                    _logger.warning("Sample generation failed at step %d: %s", step, _exc)

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

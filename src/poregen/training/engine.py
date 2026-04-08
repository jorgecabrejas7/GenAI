"""Train / eval step helpers and the main training loop."""

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
from poregen.metrics.seg import segmentation_metrics, porosity_error
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


def _central_slice(vol: torch.Tensor) -> torch.Tensor:
    """Extract the central axial (D-axis) slice → (B, 1, H, W)."""
    d = vol.shape[2]
    return vol[:, :, d // 2, :, :]


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
    loss_acc: dict[str, Any] = {}
    seg_acc: dict[str, Any] = {}
    recon_acc: dict[str, Any] = {}
    latent_acc: dict[str, Any] = {}
    por_acc: dict[str, Any] = {}
    last_output: VAEOutput | None = None
    last_batch: dict | None = None

    for _ in range(n_batches):
        batch = next(data_iter)
        losses, output = eval_step(
            model, batch, loss_fn, step, device, autocast_dtype
        )
        _accumulate(loss_acc, losses)

        mask_dev = batch["mask"].to(device, non_blocking=True)

        # Segmentation metrics
        seg = segmentation_metrics(output.mask_logits, mask_dev)
        _accumulate(seg_acc, seg)

        # Porosity preservation error
        por = porosity_error(output.mask_logits, mask_dev)
        _accumulate(por_acc, por)

        # Reconstruction metrics
        xct_dev = batch["xct"].to(device, non_blocking=True)
        xct_recon = torch.sigmoid(output.xct_logits)
        recon = {
            "mae":             mae(output.xct_logits, xct_dev).item(),
            "psnr":            psnr(output.xct_logits, xct_dev).item(),
            "sharpness_recon": sharpness_proxy(xct_recon).item(),
            "sharpness_gt":    sharpness_proxy(xct_dev).item(),
        }
        _accumulate(recon_acc, recon)

        # Latent metrics
        lat = {**active_units(output.mu), **latent_stats(output.mu, output.logvar)}
        _accumulate(latent_acc, lat)

        last_output = output
        last_batch = batch

    agg = {
        **_mean_acc(loss_acc, n_batches),
        **_mean_acc(seg_acc, n_batches),
        **_mean_acc(por_acc, n_batches),
        **_mean_acc(recon_acc, n_batches),
        **_mean_acc(latent_acc, n_batches),
    }
    return agg, last_output, last_batch  # type: ignore[return-value]


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
                    # kl_per_channel — log each channel individually
                    for i, ch_val in enumerate(v):
                        tb_writer.add_scalar(f"train/kl_ch{i}", ch_val, step)
                else:
                    tb_writer.add_scalar(f"train/{k}", v, step)
            tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            if scheduler is not None:
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        # ── validation ────────────────────────────────────────────────
        if val_iter is not None and (step + 1) % eval_every == 0:
            agg, last_output, last_batch = _run_eval(
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
                            tb_writer.add_scalar(f"val/kl_ch{i}", ch_val, step)
                    else:
                        tb_writer.add_scalar(f"val/{k}", v, step)
                # per-channel KL histogram
                if "kl_per_channel" in agg and isinstance(agg["kl_per_channel"], list):
                    kl_tensor = torch.tensor(agg["kl_per_channel"])
                    tb_writer.add_histogram("val/kl_per_channel", kl_tensor, step)

        # ── TensorBoard: reconstruction images ───────────────────────
        if (
            tb_writer is not None
            and val_iter is not None
            and (step + 1) % image_log_every == 0
        ):
            _log_recon_images(tb_writer, last_output, last_batch, step, "val", device)

        # ── test evaluation ───────────────────────────────────────────
        if test_iter is not None and (step + 1) % test_every == 0:
            test_agg, test_output, test_batch = _run_eval(
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
                            tb_writer.add_scalar(f"test/kl_ch{i}", ch_val, step)
                    else:
                        tb_writer.add_scalar(f"test/{k}", v, step)
                _log_recon_images(tb_writer, test_output, test_batch, step, "test", device)

        # ── checkpoint ───────────────────────────────────────────────
        if (step + 1) % save_every == 0:
            save_checkpoint(
                run_dir / "last.ckpt",
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
    save_checkpoint(
        run_dir / "last.ckpt",
        model, optimizer, scaler, step=start_step + total_steps,
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
    """Log central axial slice reconstructions to TensorBoard."""
    with torch.no_grad():
        xct_recon = torch.sigmoid(output.xct_logits)   # (B,1,D,H,W) in [0,1]
        mask_recon = torch.sigmoid(output.mask_logits) # (B,1,D,H,W) in [0,1]
        xct_gt = batch["xct"].to(device, non_blocking=True)
        mask_gt = batch["mask"].to(device, non_blocking=True)

        # Central axial slice → (B, 1, H, W) for make_grid
        xct_recon_sl = _central_slice(xct_recon)
        mask_recon_sl = _central_slice(mask_recon)
        xct_gt_sl = _central_slice(xct_gt)
        mask_gt_sl = _central_slice(mask_gt)

    tb_writer.add_images(f"{prefix}/xct_recon", xct_recon_sl, step)
    tb_writer.add_images(f"{prefix}/xct_gt", xct_gt_sl, step)
    tb_writer.add_images(f"{prefix}/mask_recon", mask_recon_sl, step)
    tb_writer.add_images(f"{prefix}/mask_gt", mask_gt_sl, step)

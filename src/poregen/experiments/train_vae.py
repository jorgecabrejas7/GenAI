"""Generic config-driven VAE training runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from poregen.configuration import ResolvedExperiment, resolve_experiment
from poregen.experiments.base import find_repo_root
from poregen.losses import compute_total_loss
from poregen.models.vae import build_vae
from poregen.runtime import (
    collect_runtime_metadata,
    create_run_context,
    format_summary,
    prepare_patch_dataloaders,
    resolve_run_directory,
    save_resolved_config,
    save_run_metadata,
    update_run_metadata,
)
from poregen.training import (
    get_autocast_dtype,
    load_checkpoint,
    make_scaler,
    seed_everything,
    select_device,
    train_loop,
)

logger = logging.getLogger(__name__)


def resolve_data_root(cfg: dict[str, Any], repo_root: Path) -> Path:
    """Resolve the dataset root for training."""
    dataset_root = Path(cfg["data"]["dataset_root"])
    if dataset_root.is_absolute():
        return dataset_root.resolve()
    return (repo_root / "data" / dataset_root).resolve()


def configure_training_schedule(
    cfg: dict[str, Any],
    *,
    train_steps_per_epoch: int,
    val_steps_per_epoch: int,
) -> int:
    """Derive validation cadence from ``training.val_batches``."""
    if train_steps_per_epoch < 1 or val_steps_per_epoch < 1:
        raise ValueError("Training and validation loaders must each have at least one batch.")

    cfg["training"]["val_batches"] = max(
        1,
        min(val_steps_per_epoch, int(cfg["training"]["val_batches"])),
    )
    val_windows_per_epoch = max(
        1,
        (val_steps_per_epoch + cfg["training"]["val_batches"] - 1)
        // cfg["training"]["val_batches"],
    )
    cfg["training"]["eval_every"] = max(1, train_steps_per_epoch // val_windows_per_epoch)
    cfg["training"]["image_log_every"] = cfg["training"]["eval_every"]
    return train_steps_per_epoch


def build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Construct the configured VAE model."""
    model_cfg = cfg["model"]
    return build_vae(
        model_cfg["name"],
        in_channels=model_cfg.get("in_channels", 2),
        z_channels=model_cfg["z_channels"],
        base_channels=model_cfg["base_channels"],
        n_blocks=model_cfg["n_blocks"],
        patch_size=model_cfg["patch_size"],
    ).to(device)


def build_optimizer(
    cfg: dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """Construct the configured optimizer."""
    training_cfg = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )


def build_scheduler(
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> Any | None:
    """Build the optional cosine scheduler."""
    training_cfg = cfg["training"]
    if training_cfg.get("scheduler", "none") != "cosine":
        return None

    warmup_steps = int(training_cfg["warmup_steps"])
    total_steps = int(training_cfg["total_steps"])
    if total_steps <= warmup_steps:
        raise ValueError(
            "training.total_steps must exceed training.warmup_steps when scheduler=cosine."
        )

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=training_cfg["lr_min"],
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _prune_jsonl(path: Path, *, max_step: int) -> int:
    if not path.exists():
        return 0

    kept_lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("step", 0) <= max_step
    ]
    path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return len(kept_lines)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _prepare_resume_state(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Any | None,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint_step, _ = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        map_location=device,
    )
    remaining_steps = int(cfg["training"]["total_steps"]) - checkpoint_step
    if remaining_steps <= 0:
        raise ValueError(
            f"Checkpoint step {checkpoint_step} has already reached total_steps="
            f"{cfg['training']['total_steps']}."
        )

    if cfg["runtime"]["resume"].get("prune_jsonl", True):
        for jsonl_path in (run_dir / "log.jsonl", run_dir / "metrics.jsonl"):
            kept = _prune_jsonl(jsonl_path, max_step=checkpoint_step)
            logger.info(
                "Pruned %s to %d entries at step <= %d.",
                jsonl_path.name,
                kept,
                checkpoint_step,
            )
    if cfg["runtime"]["resume"].get("clear_tensorboard", False):
        logger.warning(
            "runtime.resume.clear_tensorboard is deprecated and ignored on resume. "
            "TensorBoard history is preserved and the resumed writer uses purge_step=%d "
            "to continue cleanly from the checkpoint.",
            checkpoint_step,
        )

    return checkpoint_step, remaining_steps


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    resolved_config = run_dir / "resolved_config.yaml"
    if not resolved_config.exists():
        raise FileNotFoundError(f"Resolved config not found in {run_dir}.")
    return _load_yaml(resolved_config)


def _make_loss_fn(cfg: dict[str, Any]):
    return lambda output, batch, step: compute_total_loss(output, batch, step, cfg)


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return path


def _initial_run_metadata(
    *,
    resolved: ResolvedExperiment,
    run_name: str,
    run_index: int,
    run_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    runtime_meta = collect_runtime_metadata(
        repo_root=resolved.repo_root,
        capture_git=bool(cfg["runtime"]["metadata"].get("capture_git", True)),
        capture_machine=bool(cfg["runtime"]["metadata"].get("capture_machine", True)),
        capture_environment=bool(cfg["runtime"]["metadata"].get("capture_environment", True)),
    )
    return {
        "status": "created",
        "experiment_id": resolved.experiment_id,
        "experiment_name": cfg["experiment"]["name"],
        "experiment_variant": cfg["experiment"]["variant"],
        "experiment_path": str(resolved.experiment_path),
        "component_paths": resolved.component_paths,
        "source_chain": resolved.source_chain,
        "run_name": run_name,
        "run_index": run_index,
        "run_dir": str(run_dir),
        **runtime_meta,
    }


def run_experiment(
    experiment_ref: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Launch a new experiment run from a YAML experiment definition."""
    resolved = resolve_experiment(experiment_ref, repo_root=repo_root)
    cfg = resolved.cfg
    run_ctx = create_run_context(resolved)

    metadata = _initial_run_metadata(
        resolved=resolved,
        run_name=run_ctx.run_name,
        run_index=run_ctx.run_index,
        run_dir=run_ctx.run_dir,
        cfg=cfg,
    )
    save_run_metadata(run_ctx.run_dir, metadata)

    seed_everything(
        int(cfg["training"]["seed"]),
        deterministic=bool(cfg["training"].get("deterministic", False)),
    )
    gpu_id = cfg["runtime"]["device"].get("gpu_id")
    device = select_device(None if gpu_id is None else int(gpu_id))
    autocast_dtype = get_autocast_dtype(device)
    scaler = make_scaler(device)
    data_root = resolve_data_root(cfg, resolved.repo_root)
    cfg, loaders = prepare_patch_dataloaders(cfg, data_root)
    train_loader, val_loader, test_loader = loaders
    full_val_every = configure_training_schedule(
        cfg,
        train_steps_per_epoch=len(train_loader),
        val_steps_per_epoch=len(val_loader),
    )
    save_resolved_config(run_ctx.run_dir, cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = _make_loss_fn(cfg)

    logger.info("Launching %s from %s", resolved.experiment_id, resolved.experiment_path)
    logger.info("Run dir: %s", run_ctx.run_dir)
    logger.info("Data root: %s", data_root)
    logger.info("Device: %s  |  AMP dtype: %s", device, autocast_dtype)
    logger.info(
        "Loader config: batch_size=%s, num_workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s, timeout=%s",
        cfg["data"]["batch_size"],
        cfg["data"].get("num_workers"),
        cfg["data"].get("pin_memory"),
        cfg["data"].get("persistent_workers"),
        cfg["data"].get("prefetch_factor"),
        cfg["data"].get("timeout", 0),
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard support is required for training. Install the 'tensorboard' package."
        ) from exc

    tb_writer = SummaryWriter(str(run_ctx.run_dir / "tb"))
    update_run_metadata(
        run_ctx.run_dir,
        {
            "status": "running",
            "device": str(device),
            "data_root": str(data_root),
        },
    )
    try:
        try:
            history = train_loop(
                model,
                train_loader,
                val_loader,
                optimizer,
                scaler,
                loss_fn,
                total_steps=int(cfg["training"]["total_steps"]),
                log_every=cfg["training"]["log_every"],
                eval_every=cfg["training"]["eval_every"],
                val_batches=cfg["training"]["val_batches"],
                test_loader=test_loader,
                test_every=cfg["training"]["test_every"],
                test_batches=cfg["training"]["test_batches"],
                save_every=cfg["training"]["save_every"],
                image_log_every=cfg["training"]["image_log_every"],
                montecarlo_every=cfg["training"]["montecarlo_every"],
                montecarlo_batch_size=cfg["training"]["montecarlo_batch_size"],
                sample_every=cfg["training"]["sample_every"],
                n_patch_samples=cfg["training"]["n_patch_samples"],
                run_dir=run_ctx.run_dir,
                device=device,
                autocast_dtype=autocast_dtype,
                max_grad_norm=cfg["training"]["max_grad_norm"],
                scheduler=scheduler,
                tb_writer=tb_writer,
                full_val_every=full_val_every,
                final_full_eval=cfg["training"]["final_full_eval"],
                compile_model=cfg["training"].get("compile", False),
                save_latest=bool(cfg["runtime"]["checkpoints"].get("save_latest", True)),
                best_metric=cfg["runtime"]["checkpoints"].get("best_metric"),
                best_mode=cfg["runtime"]["checkpoints"].get("best_mode", "min"),
            )
        except Exception as exc:
            update_run_metadata(
                run_ctx.run_dir,
                {
                    "status": "failed",
                    "failure": str(exc),
                },
            )
            raise
    finally:
        tb_writer.close()

    summary = format_summary(history)
    _write_summary(run_ctx.run_dir, summary)
    update_run_metadata(
        run_ctx.run_dir,
        {
            "status": "completed",
            "summary": summary,
        },
    )
    return run_ctx.run_dir


def resume_run(
    run_ref: str | Path,
    *,
    checkpoint_name: str = "latest.ckpt",
    repo_root: str | Path | None = None,
) -> Path:
    """Resume an interrupted run from its saved config and checkpoint."""
    repo = find_repo_root(repo_root)
    run_dir = resolve_run_directory(run_ref, repo_root=repo)
    cfg = _load_run_config(run_dir)

    seed_everything(
        int(cfg["training"]["seed"]),
        deterministic=bool(cfg["training"].get("deterministic", False)),
    )
    gpu_id = cfg["runtime"]["device"].get("gpu_id")
    device = select_device(None if gpu_id is None else int(gpu_id))
    autocast_dtype = get_autocast_dtype(device)
    scaler = make_scaler(device)
    data_root = resolve_data_root(cfg, repo)
    cfg, loaders = prepare_patch_dataloaders(cfg, data_root)
    train_loader, val_loader, test_loader = loaders
    full_val_every = configure_training_schedule(
        cfg,
        train_steps_per_epoch=len(train_loader),
        val_steps_per_epoch=len(val_loader),
    )
    save_resolved_config(run_dir, cfg)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    start_step, remaining_steps = _prepare_resume_state(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
    )
    loss_fn = _make_loss_fn(cfg)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard support is required for training. Install the 'tensorboard' package."
        ) from exc

    tb_writer = SummaryWriter(str(run_dir / "tb"), purge_step=start_step)
    logger.info(
        "Resume loader config: batch_size=%s, num_workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s, timeout=%s",
        cfg["data"]["batch_size"],
        cfg["data"].get("num_workers"),
        cfg["data"].get("pin_memory"),
        cfg["data"].get("persistent_workers"),
        cfg["data"].get("prefetch_factor"),
        cfg["data"].get("timeout", 0),
    )
    update_run_metadata(
        run_dir,
        {
            "status": "resuming",
            "resume_from": str(checkpoint_path),
            "resume_step": start_step,
        },
    )
    try:
        try:
            history = train_loop(
                model,
                train_loader,
                val_loader,
                optimizer,
                scaler,
                loss_fn,
                total_steps=remaining_steps,
                log_every=cfg["training"]["log_every"],
                eval_every=cfg["training"]["eval_every"],
                val_batches=cfg["training"]["val_batches"],
                test_loader=test_loader,
                test_every=cfg["training"]["test_every"],
                test_batches=cfg["training"]["test_batches"],
                save_every=cfg["training"]["save_every"],
                image_log_every=cfg["training"]["image_log_every"],
                montecarlo_every=cfg["training"]["montecarlo_every"],
                montecarlo_batch_size=cfg["training"]["montecarlo_batch_size"],
                sample_every=cfg["training"]["sample_every"],
                n_patch_samples=cfg["training"]["n_patch_samples"],
                run_dir=run_dir,
                device=device,
                autocast_dtype=autocast_dtype,
                start_step=start_step,
                max_grad_norm=cfg["training"]["max_grad_norm"],
                scheduler=scheduler,
                tb_writer=tb_writer,
                full_val_every=full_val_every,
                final_full_eval=cfg["training"]["final_full_eval"],
                compile_model=cfg["training"].get("compile", False),
                save_latest=bool(cfg["runtime"]["checkpoints"].get("save_latest", True)),
                best_metric=cfg["runtime"]["checkpoints"].get("best_metric"),
                best_mode=cfg["runtime"]["checkpoints"].get("best_mode", "min"),
            )
        except Exception as exc:
            update_run_metadata(
                run_dir,
                {
                    "status": "failed",
                    "failure": str(exc),
                },
            )
            raise
    finally:
        tb_writer.close()

    summary = format_summary(history)
    _write_summary(run_dir, summary)
    update_run_metadata(
        run_dir,
        {
            "status": "completed",
            "summary": summary,
        },
    )
    return run_dir

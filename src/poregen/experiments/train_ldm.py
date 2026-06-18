"""Config-driven LDM training runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from poregen.configuration import ResolvedExperiment, resolve_experiment
from poregen.diffusion.latent_dataset import build_latent_dataloaders
from poregen.diffusion.sampled_latent_dataset import build_sampled_latent_dataloaders
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.experiments.base import find_repo_root
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
from poregen.runtime import (
    collect_runtime_metadata,
    create_run_context,
    format_summary,
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
)
from poregen.training.ldm_engine import EMAModel, ldm_train_loop

logger = logging.getLogger(__name__)


def _build_dataloaders(cfg: dict[str, Any], latents_root: Path) -> tuple[Any, Any]:
    """Route to the correct dataloader factory based on cfg['data']['latent_mode'].

    ``latent_mode: mean``    (default) — loads pre-computed mu; reproduces ldm01 exactly.
    ``latent_mode: sampled`` — samples z = mu + sigma*eps per batch; used by ldm02.
    """
    mode = cfg.get("data", {}).get("latent_mode", "mean")
    if mode == "sampled":
        logger.info("latent_mode=sampled — using SampledLatentPatchDataset (ldm02 path)")
        return build_sampled_latent_dataloaders(cfg, latents_root)
    if mode != "mean":
        raise ValueError(
            f"Unknown latent_mode '{mode}'. Choose 'mean' (default, ldm01) or 'sampled' (ldm02)."
        )
    logger.info("latent_mode=mean — using LatentPatchDataset (ldm01 path)")
    return build_latent_dataloaders(cfg, latents_root)


def _resolve_latents_root(cfg: dict[str, Any], repo_root: Path) -> Path:
    root = Path(cfg["data"]["latents_root"])
    if root.is_absolute():
        return root.resolve()
    return (repo_root / root).resolve()


def _build_model(cfg: dict[str, Any], device: torch.device) -> UNet3DDenoiser:
    model_cfg = UNet3DConfig.from_cfg(cfg)
    return UNet3DDenoiser(model_cfg).to(device)


def _build_schedule(cfg: dict[str, Any], device: torch.device) -> DDPMSchedule:
    ns = cfg.get("noise_schedule", {})
    return DDPMSchedule(
        T=int(ns.get("T", 1000)),
        s=float(ns.get("s", 0.008)),
        device=device,
    )


def _build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    tc = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        weight_decay=float(tc.get("weight_decay", 0.01)),
    )


def _build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer) -> Any | None:
    tc = cfg["training"]
    if tc.get("scheduler", "none") != "cosine":
        return None
    warmup_steps = int(tc.get("warmup_steps", 0))
    total_steps  = int(tc["total_steps"])
    if total_steps <= warmup_steps:
        raise ValueError("training.total_steps must exceed warmup_steps for cosine scheduler.")
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                               eta_min=float(tc.get("lr_min", 1e-5)))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


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
        "run_name": run_name,
        "run_index": run_index,
        "run_dir": str(run_dir),
        **runtime_meta,
    }


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def _load_latent_std(latents_root: Path) -> float:
    stats_path = latents_root / "latent_scale_stats.json"
    if not stats_path.exists():
        return 1.0
    return float(json.loads(stats_path.read_text())["std"])


def _load_vae_for_sampling(
    vae_run_dir: Path,
    device: torch.device,
) -> torch.nn.Module | None:
    """Load a frozen VAE decoder from a completed VAE run directory."""
    from poregen.models.vae.registry import build_vae

    cfg_path = vae_run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        logger.warning("VAE run dir has no resolved_config.yaml: %s", vae_run_dir)
        return None

    vae_cfg = yaml.safe_load(cfg_path.read_text())
    model_cfg = dict(vae_cfg["model"])
    vae_name = model_cfg.pop("name")
    vae = build_vae(vae_name, **model_cfg)

    ckpt_path = vae_run_dir / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = vae_run_dir / "latest.ckpt"
    if not ckpt_path.exists():
        logger.warning("No VAE checkpoint found in %s, skipping sample visualisation.", vae_run_dir)
        return None

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model"]
    # strip _orig_mod. prefix produced by torch.compile
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    vae.load_state_dict(state)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info("Loaded VAE for sampling from %s", ckpt_path)
    return vae


def _resolve_vae_run_dir(cfg: dict[str, Any], repo_root: Path) -> Path | None:
    ref = cfg.get("training", {}).get("sample_vae_run")
    if not ref:
        return None
    p = Path(ref)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def run_ldm_experiment(
    experiment_ref: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Launch a new LDM experiment run from a YAML experiment definition."""
    resolved = resolve_experiment(experiment_ref, repo_root=repo_root)
    cfg      = resolved.cfg
    run_ctx  = create_run_context(resolved)

    metadata = _initial_run_metadata(
        resolved=resolved,
        run_name=run_ctx.run_name,
        run_index=run_ctx.run_index,
        run_dir=run_ctx.run_dir,
        cfg=cfg,
    )
    save_run_metadata(run_ctx.run_dir, metadata)

    seed_everything(int(cfg["training"]["seed"]))
    gpu_id = cfg.get("runtime", {}).get("device", {}).get("gpu_id")
    device = select_device(None if gpu_id is None else int(gpu_id))
    autocast_dtype = get_autocast_dtype(device)
    scaler         = make_scaler(device)

    latents_root = _resolve_latents_root(cfg, resolved.repo_root)
    train_loader, val_loader = _build_dataloaders(cfg, latents_root)
    latent_std = _load_latent_std(latents_root)

    save_resolved_config(run_ctx.run_dir, cfg)

    model     = _build_model(cfg, device)
    schedule  = _build_schedule(cfg, device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999))

    vae_run_dir = _resolve_vae_run_dir(cfg, resolved.repo_root)
    vae = _load_vae_for_sampling(vae_run_dir, device) if vae_run_dir is not None else None

    logger.info("Launching %s from %s", resolved.experiment_id, resolved.experiment_path)
    logger.info("Run dir:      %s", run_ctx.run_dir)
    logger.info("Latents root: %s", latents_root)
    logger.info("Device: %s  |  AMP dtype: %s", device, autocast_dtype)
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))
    if vae is not None:
        logger.info("Sample VAE:   loaded from %s (latent_std=%.4f)", vae_run_dir, latent_std)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard support is required. Install the 'tensorboard' package."
        ) from exc

    tb_writer = SummaryWriter(str(run_ctx.run_dir / "tb"))
    update_run_metadata(run_ctx.run_dir, {"status": "running", "device": str(device)})

    try:
        try:
            history = ldm_train_loop(
                model, train_loader, val_loader,
                optimizer, scaler, schedule,
                cfg=cfg,
                run_dir=run_ctx.run_dir,
                tb_writer=tb_writer,
                device=device,
                autocast_dtype=autocast_dtype,
                scheduler=scheduler,
                ema_decay=ema_decay,
                vae=vae,
                latent_std=latent_std,
            )
        except Exception as exc:
            update_run_metadata(run_ctx.run_dir, {"status": "failed", "failure": str(exc)})
            raise
    finally:
        tb_writer.close()

    summary = format_summary(history)
    _write_summary(run_ctx.run_dir, summary)
    update_run_metadata(run_ctx.run_dir, {"status": "completed", "summary": summary})
    return run_ctx.run_dir


def resume_ldm_run(
    run_ref: str | Path,
    *,
    checkpoint_name: str = "checkpoints/latest.ckpt",
    repo_root: str | Path | None = None,
) -> Path:
    """Resume an interrupted LDM run."""
    repo    = find_repo_root(repo_root)
    run_dir = resolve_run_directory(run_ref, repo_root=repo)
    cfg     = _load_yaml(run_dir / "resolved_config.yaml")

    seed_everything(int(cfg["training"]["seed"]))
    gpu_id = cfg.get("runtime", {}).get("device", {}).get("gpu_id")
    device = select_device(None if gpu_id is None else int(gpu_id))
    autocast_dtype = get_autocast_dtype(device)
    scaler         = make_scaler(device)

    latents_root = _resolve_latents_root(cfg, repo)
    train_loader, val_loader = _build_dataloaders(cfg, latents_root)
    latent_std = _load_latent_std(latents_root)

    model     = _build_model(cfg, device)
    schedule  = _build_schedule(cfg, device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999))

    vae_run_dir = _resolve_vae_run_dir(cfg, repo)
    vae = _load_vae_for_sampling(vae_run_dir, device) if vae_run_dir is not None else None

    ckpt_path = Path(checkpoint_name)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    start_step, _ = load_checkpoint(
        ckpt_path, model=model, optimizer=optimizer,
        scaler=scaler, scheduler=scheduler, map_location=device,
    )
    remaining = int(cfg["training"]["total_steps"]) - start_step
    if remaining <= 0:
        raise ValueError(f"Checkpoint at step {start_step} already reached total_steps.")

    # Restore EMA state if present in the checkpoint
    import torch as _torch
    _raw_ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _ema_state = _raw_ckpt.get("ema", None)
    del _raw_ckpt

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard support is required. Install the 'tensorboard' package."
        ) from exc

    tb_writer = SummaryWriter(str(run_dir / "tb"), purge_step=start_step)
    update_run_metadata(
        run_dir,
        {"status": "resuming", "resume_from": str(ckpt_path), "resume_step": start_step},
    )
    try:
        try:
            cfg_resume = dict(cfg)
            cfg_resume["training"] = dict(cfg["training"])
            cfg_resume["training"]["total_steps"] = remaining
            history = ldm_train_loop(
                model, train_loader, val_loader,
                optimizer, scaler, schedule,
                cfg=cfg_resume,
                run_dir=run_dir,
                tb_writer=tb_writer,
                start_step=start_step,
                device=device,
                autocast_dtype=autocast_dtype,
                scheduler=scheduler,
                ema_decay=ema_decay,
                ema_state=_ema_state,
                vae=vae,
                latent_std=latent_std,
            )
        except Exception as exc:
            update_run_metadata(run_dir, {"status": "failed", "failure": str(exc)})
            raise
    finally:
        tb_writer.close()

    summary = format_summary(history)
    _write_summary(run_dir, summary)
    update_run_metadata(run_dir, {"status": "completed", "summary": summary})
    return run_dir

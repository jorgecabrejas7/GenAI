"""Config-driven LDM training runner (ldm06 latent pipeline)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from poregen.configuration import ResolvedExperiment, resolve_experiment
from poregen.diffusion.latents import build_latent_dataloaders
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
from poregen.training.ldm_engine import ldm_train_loop

logger = logging.getLogger(__name__)


def _resolve_latents_root(cfg: dict[str, Any], repo_root: Path) -> Path:
    root = Path(cfg["data"]["latents_root"])
    if root.is_absolute():
        return root.resolve()
    return (repo_root / root).resolve()


def _build_model(cfg: dict[str, Any], device: torch.device) -> UNet3DDenoiser:
    model_cfg = UNet3DConfig.from_cfg(cfg)
    return UNet3DDenoiser(model_cfg).to(device)


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


def _load_vae_decoder(
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    repo_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the frozen VAE named in cfg['vae']['checkpoint'].

    The latent-store metadata records which checkpoint built the latents; the
    two must match — decoding with a different VAE than the encoder that
    produced the latents would be silently wrong.
    """
    from poregen.experiments.train_vae import load_vae_from_checkpoint

    vae_cfg = cfg.get("vae") or {}
    ref = vae_cfg.get("checkpoint")
    if not ref:
        raise ValueError("cfg['vae']['checkpoint'] is required for LDM training.")
    ckpt = Path(ref)
    ckpt = ckpt.resolve() if ckpt.is_absolute() else (repo_root / ckpt).resolve()

    meta_ckpt = Path(metadata["vae_checkpoint"]).resolve()
    if ckpt != meta_ckpt:
        raise RuntimeError(
            "VAE checkpoint mismatch:\n"
            f"  experiment config: {ckpt}\n"
            f"  latent store:      {meta_ckpt}\n"
            "The latent dataset was built with a different VAE than the one "
            "configured for decoding. Fix cfg['vae']['checkpoint'] or rebuild "
            "the latent store."
        )

    vae, _, _, _ = load_vae_from_checkpoint(ckpt, device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _prepare_data_and_vae(
    cfg: dict[str, Any],
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Build dataloaders, extract normalisation stats, and load the frozen VAE."""
    latents_root = _resolve_latents_root(cfg, repo_root)
    train_loader, val_loader = build_latent_dataloaders(cfg, latents_root)
    train_ds = train_loader.dataset

    vae = _load_vae_decoder(cfg, train_ds.metadata, repo_root, device)

    # Conditioning provenance recorded by the latent store.  The porosity
    # transform stats are what cond_por is built from at generation time, so a
    # store without them cannot be sampled reproducibly.
    cond_meta = train_ds.metadata.get("conditioning") or {}
    st = cond_meta.get("por_standardisation")
    if st is None:
        raise RuntimeError(
            "Latent store has no conditioning.por_standardisation — cond_por "
            "cannot be built at generation time.  Run scripts/build_conditioning.py."
        )
    por_log_stats = (float(st["mean"]), float(st["std"]))

    return {
        "por_log_stats": por_log_stats,
        "latents_root": latents_root,
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "latent_mean":  train_ds.channel_mean,   # (C,1,1,1) CPU float32
        "latent_std":   train_ds.channel_std,    # (C,1,1,1) CPU float32
        "val_phi":      val_loader.dataset.df["phi"].to_numpy(dtype=np.float64),
        "vae":          vae,
    }


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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _init_from_checkpoint(cfg: dict, model, device, repo_root) -> None:
    """Warm-start the WEIGHTS from another run, with a fresh optimiser.

    ``training.init_from`` is not ``resume``.  Resume continues one run: same
    directory, same step counter, same optimiser and scheduler state.  This
    starts a NEW run that happens to begin from trained weights — its own
    schedule, step budget and dropout — which is what a short corrective
    fine-tune is.

    The EMA weights are preferred when the checkpoint has them, because they
    are the weights generation uses and therefore the ones a fine-tune should
    start from.  Nothing else is carried over, and a missing file is an error
    rather than a silent train-from-scratch that would still call itself a
    fine-tune.
    """
    ref = (cfg.get("training") or {}).get("init_from")
    if not ref:
        return
    path = Path(ref)
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.exists():
        raise FileNotFoundError(
            f"training.init_from names {path}, which does not exist."
        )
    raw = torch.load(path, map_location=device, weights_only=False)
    which = "ema" if raw.get("ema") else "model"
    state = raw.get("ema") or raw.get("model")
    if state is None:
        raise KeyError(f"{path} carries neither 'ema' nor 'model' weights.")
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    logger.info(
        "Warm start from %s (step %s, %s weights); optimiser, scheduler and "
        "step counter are fresh.", path, raw.get("step", "?"), which,
    )


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

    data = _prepare_data_and_vae(cfg, resolved.repo_root, device)

    save_resolved_config(run_ctx.run_dir, cfg)

    model     = _build_model(cfg, device)
    # Fresh runs only: a resume already has its weights, its optimiser and its
    # step counter from the checkpoint it is continuing.
    _init_from_checkpoint(cfg, model, device, resolved.repo_root)
    schedule  = DDPMSchedule.from_cfg(cfg, device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999))

    logger.info("Launching %s from %s", resolved.experiment_id, resolved.experiment_path)
    logger.info("Run dir:      %s", run_ctx.run_dir)
    logger.info("Latents root: %s", data["latents_root"])
    logger.info("Device: %s  |  AMP dtype: %s", device, autocast_dtype)
    logger.info("Train batches: %d  |  Val batches: %d",
                len(data["train_loader"]), len(data["val_loader"]))

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
                model, data["train_loader"], data["val_loader"],
                optimizer, scaler, schedule,
                cfg=cfg,
                run_dir=run_ctx.run_dir,
                tb_writer=tb_writer,
                device=device,
                autocast_dtype=autocast_dtype,
                scheduler=scheduler,
                ema_decay=ema_decay,
                vae=data["vae"],
                latent_mean=data["latent_mean"],
                latent_std=data["latent_std"],
                val_phi=data["val_phi"],
                por_log_stats=data["por_log_stats"],
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

    data = _prepare_data_and_vae(cfg, repo, device)

    model     = _build_model(cfg, device)
    schedule  = DDPMSchedule.from_cfg(cfg, device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    ema_decay = float(cfg["training"].get("ema_decay", 0.9999))

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
    _raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
                model, data["train_loader"], data["val_loader"],
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
                vae=data["vae"],
                latent_mean=data["latent_mean"],
                latent_std=data["latent_std"],
                val_phi=data["val_phi"],
                por_log_stats=data["por_log_stats"],
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

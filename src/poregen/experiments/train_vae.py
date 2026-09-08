"""Generic config-driven VAE training runner."""

from __future__ import annotations

import json
import logging
import math
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


def resolve_early_stopping_patience(cfg: dict[str, Any]) -> int:
    """Early-stopping patience in EVAL CHECKS, from whichever field is set.

    ``early_stopping_patience_steps`` is the field to use. Patience counted in
    eval checks silently depends on the data: ``eval_every`` is derived from
    the split sizes, so the split_v3 re-split — which grew val 89 % by moving
    Na_01 into it — halved eval_every from 264 to 120 and with it the effective
    patience, from 3168 training steps to 1440, without a config changing. A
    rung sweep that shares a dataset stays self-consistent, but nothing is
    comparable across datasets and nobody would notice.

    ``early_stopping_patience`` (eval checks) is kept because r03-r07,
    vrrae03/04 and ldm05/06 are configured with it and their runs are the
    record; rewriting those configs would change what a historical experiment
    id resolves to. The two are mutually exclusive per config, not a fallback
    chain: setting both raises.
    """
    tcfg = cfg["training"]
    steps = int(tcfg.get("early_stopping_patience_steps", 0) or 0)
    checks = int(tcfg.get("early_stopping_patience", 0) or 0)
    if steps > 0 and checks > 0:
        raise ValueError(
            "training.early_stopping_patience_steps and "
            "training.early_stopping_patience are both set. They are the same "
            "knob in different units; pick one (set the other to 0)."
        )
    if steps <= 0:
        return checks
    eval_every = max(1, int(tcfg["eval_every"]))
    n = max(1, math.ceil(steps / eval_every))
    logger.info(
        "Early stopping: %d steps of patience = %d eval checks at "
        "eval_every=%d", steps, n, eval_every,
    )
    return n


def configure_training_schedule(
    cfg: dict[str, Any],
    *,
    train_steps_per_epoch: int,
    val_steps_per_epoch: int,
) -> int:
    """Derive validation cadence from ``training.val_batches``.

    When ``training.auto_schedule`` is ``false`` the function only clamps
    ``val_batches`` to the available number of validation batches and leaves
    ``eval_every`` / ``image_log_every`` unchanged (they are taken directly
    from config).  This lets individual experiments pin an exact cadence.
    """
    if train_steps_per_epoch < 1 or val_steps_per_epoch < 1:
        raise ValueError("Training and validation loaders must each have at least one batch.")

    cfg["training"]["val_batches"] = max(
        1,
        min(val_steps_per_epoch, int(cfg["training"]["val_batches"])),
    )

    if cfg["training"].get("auto_schedule", True):
        val_windows_per_epoch = max(
            1,
            (val_steps_per_epoch + cfg["training"]["val_batches"] - 1)
            // cfg["training"]["val_batches"],
        )
        cfg["training"]["eval_every"] = max(1, train_steps_per_epoch // val_windows_per_epoch)
        cfg["training"]["image_log_every"] = cfg["training"]["eval_every"]

    return train_steps_per_epoch


def build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Construct the configured VAE model.

    Forwards the 4 core ``VAEConfig`` keys every variant needs
    (``in_channels``, ``base_channels``, ``n_blocks``, ``patch_size``), plus
    ``z_channels`` for every variant EXCEPT ``v2.vrrae`` and its ablation
    twin ``v2.vrrae_linear`` — they have no spatial latent-channel concept at
    all (their latent is a flat ``(B, vrrae_rank)`` vector), so whatever value happens to be in its resolved config (e.g.
    inherited from a parent experiment's own overrides — see
    ``configs/experiments/vrrae/base.yaml``) is deliberately never forwarded
    to ``build_vae``, keyed on the model name rather than key
    presence/absence so this holds regardless of what the config-inheritance
    chain happens to set. Also forwards (when present in ``model_cfg``) the
    ``v2.vrrae``-specific keys (``vrrae_dim``, ``vrrae_rank``,
    ``vrrae_basis_history_size``, ``vrrae_conv_channels``). These are only
    forwarded if explicitly set in the experiment's model config — the other
    4 variants (``v2.conv_noattn``, ``v2.conv``, ``v2.unet``,
    ``v2.conv_noattn_dualbranch``) never set them, so nothing changes for
    those; ``VAEConfig``'s own defaults apply when omitted.
    """
    model_cfg = cfg["model"]
    kwargs: dict[str, Any] = dict(
        in_channels=model_cfg.get("in_channels", 2),
        base_channels=model_cfg["base_channels"],
        n_blocks=model_cfg["n_blocks"],
        patch_size=model_cfg["patch_size"],
    )
    if model_cfg["name"] not in ("v2.vrrae", "v2.vrrae_linear"):
        kwargs["z_channels"] = model_cfg.get("z_channels", 8)
    for key in ("vrrae_dim", "vrrae_rank", "vrrae_basis_history_size", "vrrae_conv_channels"):
        if key in model_cfg:
            kwargs[key] = model_cfg[key]
    return build_vae(model_cfg["name"], **kwargs).to(device)


def load_vae_from_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str, Path]:
    """Load a trained VAE and its resolved config from a checkpoint path.

    The run directory's ``resolved_config.yaml`` is the exact config the
    training run used, so the rebuilt architecture always matches the weights
    (independent of later edits to ``configs/experiments/``).  Used by
    ``scripts/build_latent_dataset.py`` and the LDM training pipeline, so the
    two always load the decoder identically.

    Returns
    -------
    (model, cfg, cfg_text, run_dir) — model is in eval mode.
    """
    checkpoint = Path(checkpoint)
    # The checkpoint lives in the run dir, or in a checkpoints/ subdir of it.
    run_dir = checkpoint.parent
    if not (run_dir / "resolved_config.yaml").exists():
        run_dir = run_dir.parent
    cfg_path = run_dir / "resolved_config.yaml"
    cfg_text = cfg_path.read_text()
    cfg = yaml.safe_load(cfg_text)

    model = build_model(cfg, device)
    # restore_rng=False: loading a frozen VAE for inference must not clobber
    # the caller's RNG state (the LDM run seeds its own generators).
    load_checkpoint(str(checkpoint), model=model, map_location=device, restore_rng=False)
    model.eval()
    logger.info("Loaded VAE %s from %s", cfg["model"]["name"], checkpoint)
    return model, cfg, cfg_text, run_dir


def apply_transfer(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    repo_root: Path,
    *,
    init: bool = True,
) -> dict[str, Any]:
    """Initialise from another run's weights and freeze part of the model.

    Two optional ``training`` keys, both no-ops when absent, so every existing
    experiment is unaffected::

        training:
          init_from_checkpoint: runs/vae/<run>/best.ckpt
          freeze_modules: [encoder_a, encoder_b, fusion, to_mu, to_logvar]

    This exists for the decoder fine-tune (``r08/decoder-ft``): start from the
    chosen rung's weights and train only the decoder, so the latent space the
    LDM was built on cannot move underneath it.

    A frozen name that does not match a real child is an ERROR, not a warning.
    The whole point is that the encoder does not move, and a typo that silently
    froze nothing would produce a run that looks right, trains everything, and
    invalidates the latent store without ever saying so.

    Returns a dict recorded in run metadata, so the run's own provenance says
    what it started from and what it held fixed.
    """
    training_cfg = cfg.get("training", {})
    info: dict[str, Any] = {}

    # The adversarial signal's source. "recon" is what train_step implements:
    # the discriminator sees the model's own reconstruction of a real patch.
    # "ldm_latents" (D43 option 2, the refiner fallback) would instead decode
    # LDM-sampled latents and score those against real patches, which
    # train_step cannot do — it never sees an LDM. Refusing here is the point:
    # a config asking for the refiner would otherwise train option 1 silently
    # and report it under the fallback's name.
    adv_source = training_cfg.get("adversarial_source", "recon")
    if adv_source not in ("recon", "ldm_latents"):
        raise ValueError(
            f"training.adversarial_source={adv_source!r} is not a known source. "
            "Use 'recon' (the discriminator sees the model's reconstruction of a "
            "real patch) or 'ldm_latents' (it sees decoder(z) for LDM-sampled z)."
        )
    if adv_source == "ldm_latents" and not training_cfg.get("latent_bank_root"):
        # Without a bank there is nothing to decode, and the run would fall
        # back to the option-1 objective while reporting itself as option 2.
        raise ValueError(
            "training.adversarial_source='ldm_latents' needs training.latent_bank_root "
            "— the campaign holding latents.npy files from "
            "`eval_v4 generate --save-latents`. Without it the discriminator would "
            "see reconstructions, i.e. option 1 under option 2's name."
        )
    if adv_source != "recon":
        info["adversarial_source"] = adv_source

    ckpt_ref = training_cfg.get("init_from_checkpoint") if init else None
    if ckpt_ref:
        ckpt = Path(ckpt_ref)
        if not ckpt.is_absolute():
            ckpt = repo_root / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"training.init_from_checkpoint does not exist: {ckpt}")
        # restore_rng=False and no optimizer: this is a fresh run that borrows
        # weights, not a resume. Restoring the RNG would make the fine-tune
        # replay the parent's data order.
        step, _ = load_checkpoint(
            str(ckpt), model=model, map_location="cpu", restore_rng=False
        )
        info["init_from_checkpoint"] = str(ckpt)
        info["init_from_step"] = int(step)
        logger.info("Initialised weights from %s (step %s)", ckpt, step)

    freeze = list(training_cfg.get("freeze_modules") or [])
    if freeze:
        children = dict(model.named_children())
        missing = [n for n in freeze if n not in children]
        if missing:
            raise ValueError(
                f"training.freeze_modules names no such submodule: {missing}. "
                f"Available: {sorted(children)}"
            )
        frozen_params = 0
        for name in freeze:
            for p in children[name].parameters():
                p.requires_grad_(False)
                frozen_params += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info["freeze_modules"] = freeze
        info["frozen_params"] = frozen_params
        info["trainable_params"] = trainable
        logger.info(
            "Froze %s: %d params frozen, %d trainable of %d (%.1f%%)",
            ", ".join(freeze), frozen_params, trainable, total, 100.0 * trainable / total,
        )
        if trainable == 0:
            raise ValueError("freeze_modules left nothing trainable.")
    return info


def build_latent_bank(cfg: dict[str, Any], repo_root: Path):
    """The LDM latent bank for the refiner fine-tune, or None.

    Only built when ``training.adversarial_source`` is ``ldm_latents``;
    ``apply_transfer`` has already refused that source without a bank root.
    """
    training_cfg = cfg.get("training", {})
    if training_cfg.get("adversarial_source", "recon") != "ldm_latents":
        return None
    from poregen.training.latent_bank import LatentBank, discover_latents

    root = Path(training_cfg["latent_bank_root"])
    if not root.is_absolute():
        root = repo_root / root
    paths = discover_latents(root)
    if not paths:
        raise FileNotFoundError(
            f"no latents.npy under {root}. Generate them first with "
            "`eval_v4 generate <assessment> --save-latents`."
        )
    bank = LatentBank(paths, stride=int(training_cfg.get("latent_bank_stride", 16)))
    z = int(cfg["model"]["z_channels"])
    if bank.channels != z:
        # A width mismatch would surface as a shape error deep in the decoder
        # thousands of steps in; the bank must come from the VAE being refined.
        raise ValueError(
            f"the latent bank is {bank.channels}-channel but the model is z={z}. "
            "The bank must come from the same VAE this fine-tune is refining."
        )
    logger.info("Latent bank: %d windows from %d canvases under %s",
                len(bank), len(paths), root)
    return bank


def build_discriminator(
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module | None, torch.optim.Optimizer | None, float]:
    """Build the optional 2D PatchGAN discriminator from config.

    Reads the optional ``discriminator`` section of the config::

        discriminator:
          enabled: true
          in_channels: 1
          base_channels: 64
          lr: 2.0e-4
          weight_decay: 0.01
          disc_weight: 0.01

    Returns (discriminator, disc_optimizer, disc_weight).  All are ``None``/0
    when the section is absent or ``enabled: false``.
    """
    from poregen.models.discriminator import PatchDiscriminator2D

    disc_cfg = cfg.get("discriminator", {})
    if not disc_cfg or not disc_cfg.get("enabled", True):
        return None, None, 0.0

    disc = PatchDiscriminator2D(
        in_channels=disc_cfg.get("in_channels", 1),
        base_channels=disc_cfg.get("base_channels", 64),
    ).to(device)

    disc_optimizer = torch.optim.AdamW(
        disc.parameters(),
        lr=float(disc_cfg.get("lr", 2e-4)),
        weight_decay=float(disc_cfg.get("weight_decay", 0.01)),
        betas=(0.5, 0.999),   # standard GAN betas — lower β1 for stability
    )
    disc_weight = float(disc_cfg.get("disc_weight", 0.01))

    logger.info(
        "Discriminator enabled: PatchDiscriminator2D(in=%d, base=%d) | "
        "disc_weight=%.4f | disc_lr=%.2e",
        disc_cfg.get("in_channels", 1),
        disc_cfg.get("base_channels", 64),
        disc_weight,
        float(disc_cfg.get("lr", 2e-4)),
    )
    return disc, disc_optimizer, disc_weight


def build_optimizer(
    cfg: dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """Construct the configured optimizer."""
    training_cfg = cfg["training"]
    # Only trainable parameters. AdamW handed a frozen parameter still carries
    # optimiser state for it, and weight decay would act on it the moment a
    # grad appeared — a silent way for a "frozen" encoder to drift.
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
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
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int]:
    checkpoint_step, _ = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        map_location=device,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
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
    transfer_info = apply_transfer(cfg, model, resolved.repo_root)
    if transfer_info:
        update_run_metadata(run_ctx.run_dir, {"transfer": transfer_info})
    latent_bank = build_latent_bank(cfg, resolved.repo_root)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = _make_loss_fn(cfg)
    discriminator, disc_optimizer, disc_weight = build_discriminator(cfg, device)

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
                early_stopping_patience=resolve_early_stopping_patience(cfg),
                early_stopping_metric=cfg["training"].get("early_stopping_metric", "val.xct_loss"),
                early_stopping_mode=cfg["training"].get("early_stopping_mode", "min"),
                early_stopping_min_delta=float(
                    cfg["training"].get("early_stopping_min_delta", 0.0)
                ),
                early_stopping_warmup_steps=int(
                    cfg["training"].get("early_stopping_warmup_steps", 0)
                ),
                discriminator=discriminator,
                disc_optimizer=disc_optimizer,
                disc_weight=disc_weight,
                latent_bank=latent_bank,
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
    # init=False: the weights come from this run's own checkpoint below.
    # The freeze must still be re-applied or the encoder unfreezes on resume.
    apply_transfer(cfg, model, repo, init=False)
    # A resumed refiner needs its bank rebuilt too, or the adversarial
    # branch silently reverts to reconstructions on resume.
    latent_bank = build_latent_bank(cfg, repo)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Built before the load: the checkpoint carries the discriminator weights
    # and its optimizer moments, and a fresh pair would restart the GAN.
    discriminator, disc_optimizer, disc_weight = build_discriminator(cfg, device)

    start_step, remaining_steps = _prepare_resume_state(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
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
                early_stopping_patience=resolve_early_stopping_patience(cfg),
                early_stopping_metric=cfg["training"].get("early_stopping_metric", "val.xct_loss"),
                early_stopping_mode=cfg["training"].get("early_stopping_mode", "min"),
                early_stopping_min_delta=float(
                    cfg["training"].get("early_stopping_min_delta", 0.0)
                ),
                early_stopping_warmup_steps=int(
                    cfg["training"].get("early_stopping_warmup_steps", 0)
                ),
                discriminator=discriminator,
                disc_optimizer=disc_optimizer,
                disc_weight=disc_weight,
                latent_bank=latent_bank,
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

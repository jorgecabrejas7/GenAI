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

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_UNKNOWN,
    neighbour_shared_voxels,
    validate_neighbour_geometry,
)
from poregen.losses.decoded import DecodedAuxLoss, DecodedLossConfig
from poregen.training.checkpoint import copy_checkpoint, save_checkpoint, save_checkpoint_async

import logging as _logging
_logger = _logging.getLogger(__name__)

# Layup "A" from data/layup_ground_truth.json — the stacking sequence shared by
# 74 of 78 specimens.  Used only for in-training sample visualisation; real
# generation runs pass the requested layup explicitly.
_DEFAULT_LAYUP: list[float] = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]

# Voxel side of one patch — fixed by the VAE the latent store was built with.
_PATCH_SIZE = 64


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

_COND_KEYS = ("cond_por", "cond_depth", "cond_dist6", "cond_orient",
              "cond_material", "nb_latents", "nb_std", "nb_avail")


def _unpack_cond(b: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Pull the ldm06 conditioning tensors out of a device batch."""
    missing = [k for k in _COND_KEYS if k not in b]
    if missing:
        raise KeyError(
            f"Batch is missing {missing} — the ldm06 batch contract requires "
            f"z, std, {', '.join(_COND_KEYS)}. Rebuild the latent store and the "
            "conditioning sidecar."
        )
    return {k: b[k] for k in _COND_KEYS}


def sample_neighbours(
    nb_latents: torch.Tensor,
    nb_std: torch.Tensor,
    nb_avail: torch.Tensor,
) -> torch.Tensor:
    """Draw each EXISTS neighbour from its posterior: ``mu + sigma*eps``.

    The store serves a neighbour as a posterior mean and std, but the training
    TARGET is a posterior DRAW (``data.latent_mode: sampled``).  Conditioning
    on mean-valued neighbours would teach the denoiser to predict a sample
    from inputs whose per-cell variance is short by ``E[sigma^2]`` — a gap
    that does not exist at generation time, where every neighbour is a real
    latent.  The eps is fresh on every call, exactly like the target's.

    OOB and UNKNOWN neighbours are left untouched: they carry no posterior,
    and ``noise_neighbours`` zeroes them anyway.

    Parameters
    ----------
    nb_latents : (B, 6, C, D, H, W) neighbour posterior means
    nb_std     : (B, 6, C, D, H, W) neighbour posterior stds, same store rows
    nb_avail   : (B, 6) long — EXISTS / OOB as served by the store
    """
    exists = (nb_avail == NB_EXISTS).view(*nb_avail.shape, 1, 1, 1, 1)
    return nb_latents + torch.randn_like(nb_latents) * nb_std * exists


def noise_neighbours(
    schedule: Any,
    nb_latents: torch.Tensor,
    nb_avail: torch.Tensor,
    t: torch.Tensor,
    *,
    nb_t_mix: float,
    drop_nb_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise each neighbour to its own timestep (ldm06 training).

    The neighbours arriving here are clean latents — posterior draws from
    :func:`sample_neighbours` — but at generation time a neighbour is never
    clean: inside a chunk it is the canvas at the current timestep, and in a
    finished chunk it is a clean latent re-noised to that same timestep.  A
    denoiser trained only on clean neighbours would meet an input
    distribution it has never seen on its very first sampling step.

    Per item and per neighbour::

        with probability nb_t_mix   t_nb = t              (the sampler's case)
        otherwise                   t_nb ~ Uniform{0..t}  (the general case)

    then ``q_sample`` with fresh noise.  Separately, with probability
    ``drop_nb_p`` an ITEM loses all six neighbours at once: availability
    becomes UNKNOWN, latents zero, ``nb_t`` zero.  That is the neighbour null
    arm of the nested CFG, so training and ``DDIMSampler`` share one definition
    of "no neighbour information".

    Parameters
    ----------
    nb_latents : (B, 6, C, D, H, W) clean neighbour latents (posterior draws)
    nb_avail   : (B, 6) long — EXISTS / OOB as served by the store
    t          : (B,) long — the target's timestep

    Returns
    -------
    (nb_noisy, nb_avail, nb_t) — all on the input device.
    """
    B, K = nb_avail.shape
    device = nb_avail.device
    t_exp = t.view(B, 1).expand(B, K)

    # Uniform{0..t} inclusive: floor(u·(t+1)) with the clamp guarding u == 1.
    u = torch.rand(B, K, device=device)
    t_lower = (u * (t_exp.float() + 1.0)).floor().long().clamp_(max=t_exp)
    same = torch.rand(B, K, device=device) < nb_t_mix
    nb_t = torch.where(same, t_exp, t_lower)

    if drop_nb_p > 0.0:
        drop = (torch.rand(B, 1, device=device) < drop_nb_p).expand(B, K)
        nb_avail = torch.where(drop, torch.full_like(nb_avail, NB_UNKNOWN), nb_avail)

    exists = nb_avail == NB_EXISTS
    nb_t = torch.where(exists, nb_t, torch.zeros_like(nb_t))

    flat = nb_latents.reshape(B * K, *nb_latents.shape[2:])
    noisy = schedule.q_sample(flat, nb_t.reshape(-1)).view_as(nb_latents)
    return noisy * exists.view(B, K, 1, 1, 1, 1).to(noisy.dtype), nb_avail, nb_t


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
    sample_posterior: bool = True,
    drop_por_p: float = 0.0,
    drop_nb_p: float = 0.0,
    nb_t_mix: float = 0.5,
    decoded_aux: "DecodedAuxLoss | None" = None,
) -> dict[str, float]:
    """Single LDM training step on the ldm06 latent dataset.

    Batch keys: ``z`` (normalised posterior mean), ``std`` (posterior std under
    the same affine map), ``cond_por`` / ``cond_depth`` scalars, ``cond_dist6``
    (6,), ``cond_orient`` (2,16,16,16), ``cond_material`` (1,16,16,16),
    ``nb_latents`` / ``nb_std`` (6,C,16,16,16) the neighbours' clean posterior,
    ``nb_avail`` (6,).  With ``sample_posterior=True`` the training target is a
    fresh posterior draw z = mu + sigma*eps (stochastic encoding), and every
    EXISTS neighbour is drawn the same way before it is noised — target and
    conditioning then live in the same distribution.

    ``drop_por_p`` is the CFG porosity dropout rate and ``drop_nb_p`` the CFG
    neighbour dropout rate; ``nb_t_mix`` is the probability that a neighbour is
    noised to the target's own timestep instead of a lower one (see
    :func:`noise_neighbours`).

    ``decoded_aux`` adds the decoded-space auxiliary loss (ldm06/aux): the
    lowest-t items' x̂₀ estimates are decoded through the frozen VAE and scored
    on air placement, pore Dice, delivered porosity and grey/label agreement.
    It sits OUTSIDE the autocast block on purpose — it runs its own autocast
    around the decode only, so the denoiser's forward and the decoder's are not
    forced into one dtype policy.

    Returns
    -------
    ``{"loss": float, "grad_norm": float}``, plus ``latent_loss`` and one
    ``aux_*`` entry per decoded term when ``decoded_aux`` is active.
    """
    model.train()
    b = _batch_to_device(batch, device)
    z = b["z"]                        # (B, C, D, H, W) — normalised mu

    if sample_posterior:
        z = z + b["std"] * torch.randn_like(z)

    B = z.shape[0]
    c = _unpack_cond(b)

    drop_por_mask: torch.Tensor | None = None
    if drop_por_p > 0.0:
        drop_por_mask = torch.rand(B, device=device) < drop_por_p

    t      = torch.randint(0, schedule.T, (B,), device=device)
    noise  = torch.randn_like(z)
    z_t    = schedule.q_sample(z, t, noise)
    nb = c["nb_latents"]
    if sample_posterior:
        nb = sample_neighbours(nb, c["nb_std"], c["nb_avail"])
    nb_latents, nb_avail, nb_t = noise_neighbours(
        schedule, nb, c["nb_avail"], t,
        nb_t_mix=nb_t_mix, drop_nb_p=drop_nb_p,
    )

    optimizer.zero_grad(set_to_none=True)

    target = schedule.training_target(z, noise, t)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        model_out = model(z_t, t, nb_latents, nb_avail, nb_t, c["cond_por"],
                          c["cond_depth"], c["cond_dist6"], c["cond_orient"],
                          c["cond_material"], drop_por_mask)
        loss      = F.mse_loss(model_out, target)

    extra: dict[str, float] = {}
    if decoded_aux is not None:
        latent_loss = float(loss.detach())
        aux_loss, aux_metrics = decoded_aux(
            model_out=model_out, z_t=z_t, t=t, schedule=schedule,
            batch=b, step=step, autocast_dtype=autocast_dtype,
        )
        extra = {"latent_loss": latent_loss, **aux_metrics}
        if aux_loss is not None:
            loss = loss + aux_loss

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

    return {"loss": loss.item(), "grad_norm": grad_norm, **extra}


@torch.no_grad()
def ldm_eval_step(
    model: nn.Module,
    batch: dict[str, Any],
    schedule: Any,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.bfloat16,
    sample_posterior: bool = True,
) -> dict[str, float]:
    """Single LDM eval step (no grad): sampled neighbours at t, no dropout."""
    model.eval()
    b = _batch_to_device(batch, device)
    z = b["z"]

    if sample_posterior:
        z = z + b["std"] * torch.randn_like(z)

    B = z.shape[0]
    c = _unpack_cond(b)

    t      = torch.randint(0, schedule.T, (B,), device=device)
    noise  = torch.randn_like(z)
    z_t    = schedule.q_sample(z, t, noise)
    # Eval pins the sampler's own case: every neighbour a posterior draw at the
    # target timestep, no dropout — so val loss measures the situation
    # generation actually runs in and is comparable across steps.
    nb = c["nb_latents"]
    if sample_posterior:
        nb = sample_neighbours(nb, c["nb_std"], c["nb_avail"])
    nb_latents, nb_avail, nb_t = noise_neighbours(
        schedule, nb, c["nb_avail"], t, nb_t_mix=1.0, drop_nb_p=0.0,
    )

    target = schedule.training_target(z, noise, t)

    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        model_out = model(z_t, t, nb_latents, nb_avail, nb_t, c["cond_por"],
                          c["cond_depth"], c["cond_dist6"], c["cond_orient"],
                          c["cond_material"])
        loss      = F.mse_loss(model_out, target)

    return {"loss": loss.item()}


def _run_eval(
    model: nn.Module,
    data_iter: Iterator,
    schedule: Any,
    n_batches: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    desc: str = "Eval",
    sample_posterior: bool = True,
) -> dict[str, float]:
    acc: dict[str, float] = {}
    for _ in tqdm(range(n_batches), desc=desc, leave=False, unit="batch"):
        metrics = ldm_eval_step(model, next(data_iter), schedule, device, autocast_dtype,
                                sample_posterior=sample_posterior)
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
    sample_posterior: bool = True,
) -> dict[str, float]:
    """Pass through the entire validation set."""
    acc: dict[str, float] = {}
    n = 0
    for batch in tqdm(val_loader, desc=desc, leave=False, unit="batch"):
        _accumulate(acc, ldm_eval_step(model, batch, schedule, device, autocast_dtype,
                                       sample_posterior=sample_posterior))
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
    vae: nn.Module,
    step: int,
    run_dir: Path,
    tb_writer: Any | None,
    device: torch.device,
    autocast_dtype: torch.dtype,
    ddim_steps: int = 50,
    grid: tuple[int, int, int] = (3, 3, 3),
    global_por: float = 0.02,
    latent_mean: torch.Tensor | float = 0.0,
    latent_std: torch.Tensor | float = 1.0,
    patch_size: int = _PATCH_SIZE,
    latent_size: int = 16,
    chunk_tiles: tuple[int, int, int] = (3, 3, 3),
    window_stride: int = 32,
    decode_stride: int = 32,
    window_batch: int = 32,
    s_por: float = 1.0,
    s_nb: float = 1.0,
    cfg_rescale: float = 0.0,
    por_log_stats: tuple[float, float] | None = None,
    layup_angles: "list[float] | None" = None,
    ply_thickness_vox: float = 19.6,
) -> None:
    """Generate a small volume with the production sampler and log it.

    ``grid`` is the sample volume's size in 64-voxel TILES; the whole thing
    runs through the same :class:`~poregen.diffusion.sampler.VolumeGenerator`
    a real generation run uses, so the seam metrics logged here are the same
    numbers, measured the same way.
    """
    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator, theta_from_layup

    _logger.info(
        "Generating sample volume at step %d  tiles=%s  chunk_tiles=%s  DDIM steps=%d",
        step, grid, chunk_tiles, ddim_steps,
    )

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply_to(model)
    model.eval()

    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    step_dir = samples_dir / f"step_{step:08d}"
    step_dir.mkdir(exist_ok=True)

    try:
        sampler = DDIMSampler(model, schedule, device, n_steps=ddim_steps,
                              s_por=s_por, s_nb=s_nb, cfg_rescale=cfg_rescale)
        vol_shape: tuple[int, int, int] = tuple(g * patch_size for g in grid)  # type: ignore[assignment]
        local_por_map = _gaussian_por_grid(grid, global_por)
        theta_deg = theta_from_layup(
            vol_shape[0], layup_angles if layup_angles else _DEFAULT_LAYUP,
            ply_thickness_vox,
        )
        generator = VolumeGenerator(
            sampler=sampler,
            vae=vae,
            device=device,
            patch_size=patch_size,
            latent_size=latent_size,
            latent_mean=latent_mean,
            latent_std=latent_std,
            por_log_stats=por_log_stats,
            theta_deg=theta_deg,
            chunk_tiles=chunk_tiles,
            window_stride=window_stride,
            decode_stride=decode_stride,
        )
        vol_size_mm = tuple(d * generator.voxel_size_mm for d in vol_shape)
        xct, label, gen_stats = generator.generate(
            volume_size_mm=vol_size_mm,
            target_porosity=global_por,
            local_por_map=local_por_map,
            autocast_dtype=autocast_dtype,
            window_batch=window_batch,
        )
        VolumeGenerator.save_tiff(xct, label, step_dir / "xct.tif", step_dir / "label.tif")
        _logger.info("Saved sample TIFFs → %s", step_dir)

        if tb_writer is not None:
            for _k, _v in gen_stats.items():
                if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                    tb_writer.add_scalar(f"samples/{_k}", _v, step)
            D, H, W = xct.shape

            def _img(arr2d: np.ndarray, scale: float) -> torch.Tensor:
                return torch.from_numpy(arr2d.astype(np.float32) / scale).unsqueeze(0)

            tb_writer.add_image("samples/xct_dslice",  _img(xct[D // 2], 255.0),       step)
            tb_writer.add_image("samples/xct_hslice",  _img(xct[:, H // 2], 255.0),    step)
            tb_writer.add_image("samples/xct_wslice",  _img(xct[:, :, W // 2], 255.0), step)
            # label is {0 material, 1 pore, 2 air} — scale by 2 to fill [0, 1].
            tb_writer.add_image("samples/label_dslice", _img(label[D // 2], 2.0),       step)
            tb_writer.add_image("samples/label_hslice", _img(label[:, H // 2], 2.0),    step)
            tb_writer.add_image("samples/label_wslice", _img(label[:, :, W // 2], 2.0), step)

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
    latent_mean: torch.Tensor | float = 0.0,
    latent_std: torch.Tensor | float = 1.0,
    val_phi: "np.ndarray | None" = None,
    por_log_stats: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Step-based LDM training loop mirroring the VAE train_loop.

    Trains in normalised latent space; *latent_mean*/*latent_std* are the
    per-channel denormalisation stats from the latent store's metadata, used
    by every decode path (generation eval, sample visualisation).  What the
    denoiser regresses — ε or v — is the schedule's business, not this loop's:
    ``schedule.training_target`` names the target and every conversion back to
    x̂₀ goes through the schedule too.

    Reads from cfg["training"]:
      total_steps, log_every, eval_every, val_batches, save_every,
      max_grad_norm, compile,
      full_val_every, early_stopping_patience,
      gen_eval_every, gen_eval_samples, gen_eval_ddim_steps,
      sample_every, sample_ddim_steps, sample_grid, sample_global_por,
      drop_por   (CFG porosity dropout — learned null porosity token),
      drop_nb    (CFG neighbour dropout — all six neighbours to UNKNOWN),
      nb_t_mix   (probability a neighbour is noised to the target's own t)

    Reads from cfg["data"]:
      latent_mode  ('sampled' — z=mu+sigma*eps per batch, default — or 'mean')

    Reads from cfg["generation"]:
      chunk_tiles, window_stride, decode_stride, window_batch
      (the sampler geometry the in-training sample volumes use)

    Reads from cfg["guidance"]:
      s_por, s_nb, cfg_rescale  (guidance scales and the Lin et al. guidance
      rescale factor, for in-training sample viz)

    Reads from cfg["loss"]["decoded"]:
      enabled, t_max_frac, ramp_steps, decoded_max_items, weights
      (the decoded-space auxiliary loss; see poregen.losses.decoded)
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

    full_val_every          = int(training_cfg.get("full_val_every",          0))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
    sample_every      = int(training_cfg.get("sample_every", 0))
    sample_ddim_steps = int(training_cfg.get("sample_ddim_steps", 20))
    _raw_grid         = training_cfg.get("sample_grid", [3, 3, 3])
    sample_grid       = tuple(int(x) for x in _raw_grid)
    sample_global_por = float(training_cfg.get("sample_global_por", 0.02))
    sample_layup      = training_cfg.get("sample_layup") or _DEFAULT_LAYUP
    sample_ply_thickness_vox = float(training_cfg.get("sample_ply_thickness_vox", 19.6))

    # Three distinct strides.  sample_stride (dataset density) is a data-side
    # parameter and is only logged here; generation_stride drives the decode
    # tiling grid and neighbour_offset the spatial relation the model is
    # trained on.  The last two must be >= patch_size so neighbours only touch.
    data_cfg          = cfg.get("data", {})
    sample_stride     = int(data_cfg.get("sample_stride", 32))
    generation_stride = int(data_cfg.get("generation_stride", 64))
    neighbour_offset  = int(data_cfg.get("neighbour_offset", 64))
    validate_neighbour_geometry(neighbour_offset, _PATCH_SIZE)
    _logger.info(
        "Strides — sample_stride=%d (data density)  generation_stride=%d (tiling grid)  "
        "neighbour_offset=%d (neighbour relation, %d shared voxels)",
        sample_stride, generation_stride, neighbour_offset,
        neighbour_shared_voxels(neighbour_offset, _PATCH_SIZE),
    )

    # Sampler geometry for the in-training sample volumes — the same knobs a
    # production generation run uses, so the sample is not a special case.
    generation_cfg = cfg.get("generation", {}) or {}
    chunk_tiles    = tuple(int(v) for v in generation_cfg.get("chunk_tiles", [3, 3, 3]))
    window_stride  = int(generation_cfg.get("window_stride", 32))
    decode_stride  = int(generation_cfg.get("decode_stride", 32))
    window_batch   = int(generation_cfg.get("window_batch", 32))

    # Generation eval (off-manifold diagnostics + decode-based porosity eval)
    gen_eval_every      = int(training_cfg.get("gen_eval_every", 0))
    gen_eval_samples    = int(training_cfg.get("gen_eval_samples", 64))
    gen_eval_ddim_steps = int(training_cfg.get("gen_eval_ddim_steps", 50))

    # Latent mode: 'sampled' draws z=μ+σε per batch (stochastic encoding)
    latent_mode = str(cfg.get("data", {}).get("latent_mode", "sampled"))
    if latent_mode not in ("sampled", "mean"):
        raise ValueError(f"Unknown data.latent_mode '{latent_mode}' (use 'sampled' or 'mean').")
    sample_posterior = latent_mode == "sampled"

    # CFG dropout rates and the neighbour-timestep mixture (see noise_neighbours)
    drop_por_p = float(training_cfg.get("drop_por", 0.0))
    drop_nb_p  = float(training_cfg.get("drop_nb", 0.0))
    nb_t_mix   = float(training_cfg.get("nb_t_mix", 0.5))

    # Guidance scales for in-training sample visualisation (default 1.0 = un-guided)
    guidance_cfg = cfg.get("guidance", {})
    s_por_scale  = float(guidance_cfg.get("s_por", 1.0))
    s_nb_scale   = float(guidance_cfg.get("s_nb",  1.0))
    cfg_rescale  = float(guidance_cfg.get("cfg_rescale", 0.0))

    _logger.info(
        "CFG dropout — drop_por=%.2f  drop_nb=%.2f  |  nb_t_mix=%.2f",
        drop_por_p, drop_nb_p, nb_t_mix,
    )
    if s_por_scale != 1.0 or s_nb_scale != 1.0:
        _logger.info(
            "Guided sample viz enabled — s_por=%.2f  s_nb=%.2f  cfg_rescale=%.2f",
            s_por_scale, s_nb_scale, cfg_rescale,
        )

    # Decoded-space auxiliary loss (ldm06/aux).  Built once: it holds the
    # frozen VAE and the store's denormalisation stats, neither of which
    # changes per step.
    decoded_cfg = DecodedLossConfig.from_cfg(cfg)
    decoded_aux: DecodedAuxLoss | None = None
    if decoded_cfg is not None:
        if vae is None:
            raise RuntimeError(
                "loss.decoded.enabled requires the frozen VAE (vae=...) — the "
                "auxiliary terms are scored on the DECODED x0 estimate."
            )
        decoded_aux = DecodedAuxLoss(decoded_cfg, vae, latent_mean, latent_std)
        _logger.info(
            "Decoded auxiliary loss ON — t < %.2f*T, <=%d items/step, ramp %d steps, "
            "weights air=%.2f dice=%.2f por=%.2f grey=%.2f",
            decoded_cfg.t_max_frac, decoded_cfg.max_items, decoded_cfg.ramp_steps,
            decoded_cfg.w_air_outside_material, decoded_cfg.w_pore_dice,
            decoded_cfg.w_porosity_consistency, decoded_cfg.w_grey_agreement,
        )

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
    best_val_full_loss: float | None = None
    val_full_no_improve: int = 0
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
                sample_posterior=sample_posterior, drop_por_p=drop_por_p,
                drop_nb_p=drop_nb_p, nb_t_mix=nb_t_mix,
                decoded_aux=decoded_aux,
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
                # Every decoded term goes out on its own scalar: a single
                # summed aux number cannot say WHICH defect moved.
                for _k in ("latent_loss",):
                    if _k in metrics:
                        tb_writer.add_scalar(f"train/{_k}", metrics[_k], step)
                for _k, _v in metrics.items():
                    if _k.startswith("aux_"):
                        tb_writer.add_scalar(f"train/{_k}", _v, step)
                if scheduler is not None:
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

            # ── validation ──────────────────────────────────────────────────
            if val_loader is not None and (step + 1) % eval_every == 0:
                val_batches_clamped = min(val_batches, len(val_loader))
                _val_iter = iter(val_loader)
                agg = _run_eval(
                    model, _val_iter, schedule, val_batches_clamped,
                    device, autocast_dtype, desc=f"Val step {step + 1}",
                    sample_posterior=sample_posterior,
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
                    sample_posterior=sample_posterior,
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

                full_loss = full_agg["loss"]
                if best_val_full_loss is None or full_loss < best_val_full_loss:
                    best_val_full_loss = full_loss
                    val_full_no_improve = 0
                    _logger.info("Full val step %d — loss=%.6f  (new best)", step + 1, full_loss)
                else:
                    val_full_no_improve += 1
                    _logger.info(
                        "Full val step %d — loss=%.6f  (no improve %d/%s)",
                        step + 1, full_loss, val_full_no_improve,
                        early_stopping_patience if early_stopping_patience > 0 else "∞",
                    )

                if early_stopping_patience > 0 and val_full_no_improve >= early_stopping_patience:
                    _logger.info(
                        "Early stopping triggered at step %d — "
                        "no val_full improvement for %d consecutive checks (best=%.6f)",
                        step + 1, val_full_no_improve, best_val_full_loss,
                    )
                    break

            # ── generation eval: off-manifold + decode diagnostics ──────────
            if gen_eval_every > 0 and (step + 1) % gen_eval_every == 0:
                from poregen.diffusion.generation_eval import generation_eval

                if vae is None or val_phi is None or val_loader is None:
                    raise RuntimeError(
                        "gen_eval_every > 0 requires a loaded VAE, the val loader "
                        "and the val phi distribution (vae=..., val_phi=...)."
                    )
                orig_state = {k: v.clone() for k, v in model.state_dict().items()}
                ema.apply_to(model)
                try:
                    gen_metrics = generation_eval(
                        model, schedule, vae,
                        val_dataset=val_loader.dataset,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        val_phi=val_phi,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        n_samples=gen_eval_samples,
                        ddim_steps=gen_eval_ddim_steps,
                        seed=step + 1,
                    )
                finally:
                    model.load_state_dict(orig_state)
                gen_record = {
                    "step":    step,
                    "split":   "gen",
                    "elapsed": time.time() - t0,
                    **gen_metrics,
                }
                history.append(gen_record)
                log_file.write(json.dumps(gen_record) + "\n")
                log_file.flush()
                metrics_file.write(json.dumps(gen_record) + "\n")
                metrics_file.flush()
                if tb_writer is not None:
                    for k, v in gen_metrics.items():
                        tb_writer.add_scalar(f"gen/{k}", v, step)

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
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        chunk_tiles=chunk_tiles,
                        window_stride=window_stride,
                        decode_stride=decode_stride,
                        window_batch=window_batch,
                        s_por=s_por_scale,
                        s_nb=s_nb_scale,
                        cfg_rescale=cfg_rescale,
                        por_log_stats=por_log_stats,
                        layup_angles=sample_layup,
                        ply_thickness_vox=sample_ply_thickness_vox,
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

        # Name the final checkpoint after the step actually reached — an
        # early-stop break exits before the budget, and naming from the
        # budget stamps a wrong step into the file and its "step" field.
        final_step = step + 1 if total_steps > 0 else start_step
        final_ckpt = save_checkpoint(
            ckpt_dir / f"ldm_step{final_step:08d}.ckpt",
            model, optimizer, scaler, step=final_step,
            metadata={"total_steps": total_steps},
            scheduler=scheduler,
            ema_state_dict=ema.state_dict(),
        )
        copy_checkpoint(final_ckpt, ckpt_dir / "latest.ckpt")

    return history

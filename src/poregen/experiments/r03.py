"""R03 experiment runtime and analysis utilities.

Import surface for R03 notebooks — a single import gives access to
everything needed for the auxiliary-decoder and latent-analysis analyses:

    from poregen.experiments.r03 import (
        R03Runtime,
        load_r03_runtime,
        build_patch_loader,
        find_repo_root,
        AuxiliaryXCTDecoder,
        train_auxiliary_decoder,
        evaluate_auxiliary_decoder,
        ...
    )
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from poregen.losses.recon import charbonnier_loss
from poregen.models.vae import VAEConfig, build_vae
from poregen.training import get_autocast_dtype, make_scaler

from poregen.experiments.base import ExperimentRuntime, build_patch_loader, find_repo_root  # noqa: F401 — re-exported for notebooks

__all__ = [
    # runtime
    "R03Runtime",
    "load_r03_runtime",
    # data
    "build_patch_loader",
    "find_repo_root",
    # model utilities
    "freeze_module",
    "encode_mu",
    "sigmoid_xct",
    "charbonnier_on_sigmoid_logits",
    # sample-wise metrics
    "samplewise_charbonnier",
    "samplewise_sharpness_proxy",
    # mask transition utilities
    "local_mask_variance_map",
    "mask_local_variance_density",
    "mask_gradient_density",
    "transition_percentile_threshold",
    "find_transition_focus",
    # latent utilities
    "mean_latent_channels",
    "flatten_mu",
    # auxiliary decoder
    "AuxiliaryXCTDecoder",
    "train_auxiliary_decoder",
    "evaluate_auxiliary_decoder",
]


# ---------------------------------------------------------------------------
# R03 runtime
# ---------------------------------------------------------------------------

@dataclass
class R03Runtime(ExperimentRuntime):
    """R03-specific runtime: ConvVAE3D (no-attention) loaded from an R03 checkpoint."""

    @classmethod
    def _build_model(cls, cfg: dict[str, Any]) -> nn.Module:
        model_cfg = cfg["model"]
        return build_vae(
            model_cfg["name"],
            z_channels=model_cfg.get("z_channels", 8),
            base_channels=model_cfg.get("base_channels", 32),
            n_blocks=model_cfg.get("n_blocks", 2),
            patch_size=model_cfg.get("patch_size", 64),
        )


def load_r03_runtime(
    checkpoint_path: str,
    *,
    config_path: str = "src/poregen/configs/r03.yaml",
    data_root: str | None = None,
    device: torch.device | None = None,
    repo_root: str | None = None,
) -> R03Runtime:
    """Load the R03 checkpoint — thin wrapper around :meth:`R03Runtime.from_checkpoint`.

    Kept as a module-level function so existing notebook cells don't need
    to change their call style.
    """
    return R03Runtime.from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        data_root=data_root,
        device=device,
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze every parameter in *module* in-place and set to eval mode."""
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


def encode_mu(model: nn.Module, xct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encode a batch into posterior means (no sampling)."""
    x = torch.cat([xct, mask], dim=1)
    h = model.encoder(x)
    return model.to_mu(h)


def sigmoid_xct(logits: torch.Tensor) -> torch.Tensor:
    """Map XCT decoder logits → [0, 1] reconstruction space."""
    return torch.sigmoid(logits)


def charbonnier_on_sigmoid_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Charbonnier loss after projecting decoder logits to [0, 1]."""
    return charbonnier_loss(sigmoid_xct(logits), target, eps=eps)


# ---------------------------------------------------------------------------
# Sample-wise metrics
# ---------------------------------------------------------------------------

def samplewise_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one Charbonnier value per patch."""
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).flatten(1).mean(dim=1)


def samplewise_sharpness_proxy(x: torch.Tensor) -> torch.Tensor:
    """Return one sharpness score per patch (mean absolute finite differences)."""
    gd = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().flatten(1).mean(dim=1)
    gh = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().flatten(1).mean(dim=1)
    gw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().flatten(1).mean(dim=1)
    return (gd + gh + gw) / 3.0


# ---------------------------------------------------------------------------
# Mask transition utilities
# ---------------------------------------------------------------------------

def local_mask_variance_map(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Local binary-mask variance map — locates transition-heavy regions."""
    pad = kernel_size // 2
    mask = mask.float()
    mean = F.avg_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool3d(mask * mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (mean_sq - mean * mean).clamp_min(0.0)


def mask_local_variance_density(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Scalar transition-density proxy: mean local mask variance per patch."""
    return local_mask_variance_map(mask, kernel_size=kernel_size).flatten(1).mean(dim=1)


def mask_gradient_density(mask: torch.Tensor) -> torch.Tensor:
    """Scalar transition-density proxy: mean absolute finite differences per patch."""
    mask = mask.float()
    gz = (mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]).abs().flatten(1).mean(dim=1)
    gy = (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().flatten(1).mean(dim=1)
    gx = (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().flatten(1).mean(dim=1)
    return (gz + gy + gx) / 3.0


def transition_percentile_threshold(
    scores: torch.Tensor | Iterable[float],
    percentile: float = 75.0,
) -> float:
    """Return the *percentile* cutoff of a transition-density score vector."""
    if isinstance(scores, torch.Tensor):
        values = scores.detach().float().cpu()
    else:
        values = torch.tensor(list(scores), dtype=torch.float32)
    return float(torch.quantile(values, percentile / 100.0).item())


def find_transition_focus(mask: torch.Tensor, kernel_size: int = 5) -> tuple[int, int, int]:
    """Return the ``(z, y, x)`` voxel with the highest local transition score."""
    if mask.ndim == 5:
        if mask.shape[0] != 1:
            raise ValueError("find_transition_focus expects a single patch for a batched tensor.")
        mask = mask[0]
    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError("find_transition_focus expects a single-channel mask tensor.")
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    score_map = local_mask_variance_map(mask.float(), kernel_size=kernel_size)[0, 0]
    flat_idx = int(score_map.reshape(-1).argmax().item())
    depth, height, width = score_map.shape
    z = flat_idx // (height * width)
    rem = flat_idx % (height * width)
    y = rem // width
    x = rem % width
    return z, y, x


# ---------------------------------------------------------------------------
# Latent utilities
# ---------------------------------------------------------------------------

def mean_latent_channels(mu: torch.Tensor) -> torch.Tensor:
    """Spatially average each latent channel: ``(B, C, d, h, w) → (B, C)``."""
    return mu.mean(dim=(2, 3, 4))


def flatten_mu(mu: torch.Tensor) -> torch.Tensor:
    """Flatten the latent tensor to one feature vector per patch."""
    return mu.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Auxiliary XCT decoder
# ---------------------------------------------------------------------------

def _norm_groups(channels: int) -> int:
    return min(32, channels)


class _ResidualConvBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=False),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _WideUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=False),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=False),
            _ResidualConvBlock3D(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AuxiliaryXCTDecoder(nn.Module):
    """Wider / deeper XCT-only decoder for probing whether R03's ``mu`` still carries texture."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        base_schedule = cfg.channel_schedule()
        decoder_schedule = [
            base_schedule[i] * 2
            for i in range(cfg.n_blocks - 1, -1, -1)
        ]

        stem_channels = decoder_schedule[0]
        self.stem = nn.Sequential(
            nn.Conv3d(cfg.z_channels, stem_channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(stem_channels), stem_channels),
            nn.SiLU(inplace=False),
            _ResidualConvBlock3D(stem_channels),
        )

        blocks: list[nn.Module] = []
        in_ch = stem_channels
        for out_ch in decoder_schedule:
            blocks.append(_WideUpBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.Sequential(*blocks)

        self.refine = nn.Sequential(
            nn.Conv3d(in_ch, cfg.base_channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(cfg.base_channels), cfg.base_channels),
            nn.SiLU(inplace=False),
            _ResidualConvBlock3D(cfg.base_channels),
        )
        self.xct_head = nn.Conv3d(cfg.base_channels, 1, 1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        h = self.stem(mu)
        h = self.decoder(h)
        h = self.refine(h)
        return self.xct_head(h)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_auxiliary_epoch(
    model: nn.Module,
    aux_decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> float:
    model.eval()
    aux_decoder.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        xct  = batch["xct"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        mu   = encode_mu(model, xct, mask)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            logits = aux_decoder(mu)
            loss   = charbonnier_on_sigmoid_logits(logits, xct)
        batch_size   = xct.shape[0]
        total_loss  += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def train_auxiliary_decoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 8,
    patience: int = 2,
    progress: bool = True,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train a higher-capacity decoder on frozen R03 posterior means.

    Returns the best aux decoder state and a loss-history DataFrame.
    """
    if not hasattr(model, "cfg") or not isinstance(model.cfg, VAEConfig):
        raise TypeError(
            "train_auxiliary_decoder expects a VAE model with a VAEConfig in model.cfg."
        )

    freeze_module(model)
    aux_decoder  = AuxiliaryXCTDecoder(model.cfg).to(device)
    optimizer    = torch.optim.AdamW(aux_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scaler       = make_scaler(device)
    autocast_dtype   = get_autocast_dtype(device)
    autocast_enabled = device.type == "cuda"

    best_val   = float("inf")
    best_state = copy.deepcopy(aux_decoder.state_dict())
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        aux_decoder.train()
        running_loss = 0.0
        n_seen = 0

        iterator = tqdm(
            train_loader,
            desc=f"Aux decoder epoch {epoch}/{max_epochs}",
            leave=False,
            disable=not progress,
        )
        for batch in iterator:
            xct  = batch["xct"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                mu = encode_mu(model, xct, mask)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                logits = aux_decoder(mu)
                loss   = charbonnier_on_sigmoid_logits(logits, xct)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size    = xct.shape[0]
            running_loss += loss.item() * batch_size
            n_seen       += batch_size
            iterator.set_postfix(train_loss=running_loss / max(n_seen, 1))

        train_loss = running_loss / max(n_seen, 1)
        val_loss   = (
            _eval_auxiliary_epoch(
                model, aux_decoder, val_loader, device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            if val_loader is not None
            else float("nan")
        )

        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        tracked_val = train_loss if val_loader is None else val_loss
        if tracked_val < best_val:
            best_val   = tracked_val
            best_state = copy.deepcopy(aux_decoder.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break

    aux_decoder.load_state_dict(best_state)
    aux_decoder.eval()
    return aux_decoder, pd.DataFrame(history_rows)


@torch.no_grad()
def evaluate_auxiliary_decoder(
    model: nn.Module,
    aux_decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
    transition_threshold: float,
    transition_kernel_size: int = 5,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate original vs auxiliary decoders patch-by-patch.

    Returns a DataFrame with one row per patch containing per-patch Charbonnier
    losses, sharpness ratios, porosity, and transition labels.
    """
    model.eval()
    aux_decoder.eval()

    rows: list[dict[str, Any]] = []
    dataset_index = 0

    iterator = tqdm(loader, desc=f"Eval {split_name}", leave=False, disable=not progress)
    for batch in iterator:
        xct  = batch["xct"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        output     = model(xct, mask)
        aux_logits = aux_decoder(output.mu)

        xct_recon_original = sigmoid_xct(output.xct_logits)
        xct_recon_aux      = sigmoid_xct(aux_logits)

        loss_original     = samplewise_charbonnier(xct_recon_original, xct).cpu()
        loss_aux          = samplewise_charbonnier(xct_recon_aux, xct).cpu()
        sharp_gt          = samplewise_sharpness_proxy(xct).cpu()
        sharp_original    = samplewise_sharpness_proxy(xct_recon_original).cpu()
        sharp_aux         = samplewise_sharpness_proxy(xct_recon_aux).cpu()
        transition_density = mask_local_variance_density(
            mask, kernel_size=transition_kernel_size,
        ).cpu()
        porosity = mask.mean(dim=(1, 2, 3, 4)).cpu()

        # coords is now a (B, 3) tensor: [z0, y0, x0] per sample
        coords    = batch["coords"]
        batch_size = xct.shape[0]
        for i in range(batch_size):
            gt_sharp = float(sharp_gt[i].item())
            rows.append({
                "split":           split_name,
                "dataset_index":   dataset_index,
                "volume_id":       batch["volume_id"][i],
                "z0":              int(coords[i][0]),
                "y0":              int(coords[i][1]),
                "x0":              int(coords[i][2]),
                "porosity":                 float(porosity[i].item()),
                "transition_density":       float(transition_density[i].item()),
                "transition_label":         (
                    "high_transition"
                    if float(transition_density[i].item()) >= transition_threshold
                    else "interior"
                ),
                "xct_loss_original":        float(loss_original[i].item()),
                "xct_loss_auxiliary":       float(loss_aux[i].item()),
                "sharpness_gt":             gt_sharp,
                "sharpness_recon_original": float(sharp_original[i].item()),
                "sharpness_recon_auxiliary":float(sharp_aux[i].item()),
                "sharpness_ratio_original": (
                    float(sharp_original[i].item()) / gt_sharp if gt_sharp > 0.0 else float("nan")
                ),
                "sharpness_ratio_auxiliary": (
                    float(sharp_aux[i].item()) / gt_sharp if gt_sharp > 0.0 else float("nan")
                ),
            })
            dataset_index += 1

    return pd.DataFrame(rows)

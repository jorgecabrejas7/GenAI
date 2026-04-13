"""Utilities for the R03 auxiliary-decoder and latent-space notebooks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from poregen.configs.config import load_config
from poregen.dataset.loader import PatchDataset
from poregen.losses.recon import charbonnier_loss
from poregen.models.vae import VAEConfig, build_vae
from poregen.training import get_autocast_dtype, load_checkpoint, make_scaler, select_device


@dataclass
class R03Runtime:
    """Loaded R03 model plus the resolved config/data context."""

    model: nn.Module
    cfg: dict[str, Any]
    device: torch.device
    checkpoint_step: int
    checkpoint_meta: dict[str, Any]
    data_root: Path
    repo_root: Path


def _find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or ".").resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "poregen").exists():
            return candidate
    raise FileNotFoundError("Could not infer repo root from the current working directory.")


def _model_overrides_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    return {
        "z_channels": model_cfg.get("z_channels", 8),
        "base_channels": model_cfg.get("base_channels", 32),
        "n_blocks": model_cfg.get("n_blocks", 2),
        "patch_size": model_cfg.get("patch_size", 64),
    }


def load_r03_runtime(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path = "src/poregen/configs/r03.yaml",
    data_root: str | Path | None = None,
    device: torch.device | None = None,
    repo_root: str | Path | None = None,
) -> R03Runtime:
    """Load the R03 checkpoint using the project's standard build/load path."""

    repo = _find_repo_root(repo_root)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (repo / ckpt_path).resolve()

    cfg = load_config(cfg_path)
    model = build_vae(cfg["model"]["name"], **_model_overrides_from_cfg(cfg))
    runtime_device = device or select_device()
    step, meta = load_checkpoint(
        ckpt_path,
        model=model,
        restore_rng=False,
        map_location=runtime_device,
    )
    model = model.to(runtime_device).eval()

    resolved_data_root = (
        Path(data_root)
        if data_root is not None
        else repo / "data" / cfg["data"].get("dataset_root", "split_v1")
    ).resolve()

    return R03Runtime(
        model=model,
        cfg=cfg,
        device=runtime_device,
        checkpoint_step=step,
        checkpoint_meta=meta,
        data_root=resolved_data_root,
        repo_root=repo,
    )


def build_patch_loader(
    cfg: dict[str, Any],
    data_root: str | Path,
    split: str,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a PatchDataset/DataLoader pair consistent with the repo config."""

    data_cfg = cfg.get("data", {})
    root = Path(data_root)
    dataset = PatchDataset(root / "patch_index.parquet", root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size or int(data_cfg.get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0) if num_workers is None else num_workers),
        pin_memory=bool(data_cfg.get("pin_memory", False) if pin_memory is None else pin_memory),
        drop_last=drop_last,
    )


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze every parameter in *module* in-place."""

    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module


def encode_mu(model: nn.Module, xct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encode a batch into posterior means without sampling."""

    x = torch.cat([xct, mask], dim=1)
    h = model.encoder(x)
    return model.to_mu(h)


def sigmoid_xct(logits: torch.Tensor) -> torch.Tensor:
    """Map XCT decoder logits to the [0, 1] reconstruction space."""

    return torch.sigmoid(logits)


def charbonnier_on_sigmoid_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Charbonnier loss after projecting decoder logits to [0, 1]."""

    return charbonnier_loss(sigmoid_xct(logits), target, eps=eps)


def samplewise_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one Charbonnier value per patch."""

    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.flatten(1).mean(dim=1)


def samplewise_sharpness_proxy(x: torch.Tensor) -> torch.Tensor:
    """Return one sharpness score per patch."""

    gd = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().flatten(1).mean(dim=1)
    gh = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().flatten(1).mean(dim=1)
    gw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().flatten(1).mean(dim=1)
    return (gd + gh + gw) / 3.0


def local_mask_variance_map(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Local binary-mask variance map used to locate transition-heavy regions."""

    pad = kernel_size // 2
    mask = mask.float()
    mean = F.avg_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool3d(mask * mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (mean_sq - mean * mean).clamp_min(0.0)


def mask_local_variance_density(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Scalar transition-density proxy based on mean local mask variance."""

    return local_mask_variance_map(mask, kernel_size=kernel_size).flatten(1).mean(dim=1)


def mask_gradient_density(mask: torch.Tensor) -> torch.Tensor:
    """Scalar transition-density proxy based on mean absolute finite differences."""

    mask = mask.float()
    gz = (mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]).abs().flatten(1).mean(dim=1)
    gy = (mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]).abs().flatten(1).mean(dim=1)
    gx = (mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]).abs().flatten(1).mean(dim=1)
    return (gz + gy + gx) / 3.0


def transition_percentile_threshold(
    scores: torch.Tensor | Iterable[float],
    percentile: float = 75.0,
) -> float:
    """Return the percentile cutoff for a transition-density score vector."""

    if isinstance(scores, torch.Tensor):
        values = scores.detach().float().cpu()
    else:
        values = torch.tensor(list(scores), dtype=torch.float32)
    return float(torch.quantile(values, percentile / 100.0).item())


def find_transition_focus(mask: torch.Tensor, kernel_size: int = 5) -> tuple[int, int, int]:
    """Return the (z, y, x) voxel with the highest local transition score."""

    if mask.ndim == 5:
        if mask.shape[0] != 1:
            raise ValueError("find_transition_focus expects a single patch when given a batched tensor.")
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


def mean_latent_channels(mu: torch.Tensor) -> torch.Tensor:
    """Spatially average each latent channel: (B, C, d, h, w) -> (B, C)."""

    return mu.mean(dim=(2, 3, 4))


def flatten_mu(mu: torch.Tensor) -> torch.Tensor:
    """Flatten the latent tensor to one feature vector per patch."""

    return mu.flatten(start_dim=1)


def _norm_groups(channels: int) -> int:
    return min(32, channels)


class _ResidualConvBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _WideUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            _ResidualConvBlock3D(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AuxiliaryXCTDecoder(nn.Module):
    """A wider/deeper XCT-only decoder for probing whether R03's mu still carries texture."""

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
            nn.SiLU(inplace=True),
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
            nn.SiLU(inplace=True),
            _ResidualConvBlock3D(cfg.base_channels),
        )
        self.xct_head = nn.Conv3d(cfg.base_channels, 1, 1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        h = self.stem(mu)
        h = self.decoder(h)
        h = self.refine(h)
        return self.xct_head(h)


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
        xct = batch["xct"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        mu = encode_mu(model, xct, mask)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            logits = aux_decoder(mu)
            loss = charbonnier_on_sigmoid_logits(logits, xct)
        batch_size = xct.shape[0]
        total_loss += loss.item() * batch_size
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
    """Train a higher-capacity decoder on frozen R03 posterior means."""

    if not hasattr(model, "cfg") or not isinstance(model.cfg, VAEConfig):
        raise TypeError("train_auxiliary_decoder expects a VAE model with a VAEConfig in model.cfg.")

    freeze_module(model)
    aux_decoder = AuxiliaryXCTDecoder(model.cfg).to(device)
    optimizer = torch.optim.AdamW(aux_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = make_scaler(device)
    autocast_dtype = get_autocast_dtype(device)
    autocast_enabled = device.type == "cuda"

    best_val = float("inf")
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
            xct = batch["xct"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                mu = encode_mu(model, xct, mask)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = aux_decoder(mu)
                loss = charbonnier_on_sigmoid_logits(logits, xct)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = xct.shape[0]
            running_loss += loss.item() * batch_size
            n_seen += batch_size
            iterator.set_postfix(train_loss=running_loss / max(n_seen, 1))

        train_loss = running_loss / max(n_seen, 1)
        val_loss = (
            _eval_auxiliary_epoch(
                model,
                aux_decoder,
                val_loader,
                device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            if val_loader is not None
            else float("nan")
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        tracked_val = train_loss if val_loader is None else val_loss
        if tracked_val < best_val:
            best_val = tracked_val
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
    """Evaluate original vs auxiliary decoders patch-by-patch."""

    model.eval()
    aux_decoder.eval()

    rows: list[dict[str, Any]] = []
    dataset_index = 0

    iterator = tqdm(loader, desc=f"Eval {split_name}", leave=False, disable=not progress)
    for batch in iterator:
        xct = batch["xct"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        output = model(xct, mask)
        aux_logits = aux_decoder(output.mu)

        xct_gt = xct
        xct_recon_original = sigmoid_xct(output.xct_logits)
        xct_recon_aux = sigmoid_xct(aux_logits)

        loss_original = samplewise_charbonnier(xct_recon_original, xct_gt).cpu()
        loss_aux = samplewise_charbonnier(xct_recon_aux, xct_gt).cpu()
        sharp_gt = samplewise_sharpness_proxy(xct_gt).cpu()
        sharp_original = samplewise_sharpness_proxy(xct_recon_original).cpu()
        sharp_aux = samplewise_sharpness_proxy(xct_recon_aux).cpu()
        transition_density = mask_local_variance_density(
            mask,
            kernel_size=transition_kernel_size,
        ).cpu()
        porosity = mask.mean(dim=(1, 2, 3, 4)).cpu()

        coords = batch["coords"]
        batch_size = xct.shape[0]
        for i in range(batch_size):
            gt_sharp = float(sharp_gt[i].item())
            rows.append({
                "split": split_name,
                "dataset_index": dataset_index,
                "volume_id": batch["volume_id"][i],
                "z0": int(coords[0][i]),
                "y0": int(coords[1][i]),
                "x0": int(coords[2][i]),
                "porosity": float(porosity[i].item()),
                "transition_density": float(transition_density[i].item()),
                "transition_label": (
                    "high_transition"
                    if float(transition_density[i].item()) >= transition_threshold
                    else "interior"
                ),
                "xct_loss_original": float(loss_original[i].item()),
                "xct_loss_auxiliary": float(loss_aux[i].item()),
                "sharpness_gt": gt_sharp,
                "sharpness_recon_original": float(sharp_original[i].item()),
                "sharpness_recon_auxiliary": float(sharp_aux[i].item()),
                "sharpness_ratio_original": (
                    float(sharp_original[i].item()) / gt_sharp if gt_sharp > 0.0 else float("nan")
                ),
                "sharpness_ratio_auxiliary": (
                    float(sharp_aux[i].item()) / gt_sharp if gt_sharp > 0.0 else float("nan")
                ),
            })
            dataset_index += 1

    return pd.DataFrame(rows)

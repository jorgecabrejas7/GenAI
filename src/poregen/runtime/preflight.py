"""Preflight checks and robust DataLoader preparation."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from poregen.training.data import build_patch_dataloaders

logger = logging.getLogger(__name__)


def _estimate_sample_bytes(cfg: dict[str, Any]) -> int:
    patch_size = int(cfg["model"]["patch_size"])
    voxels = patch_size ** 3
    # XCT float32 + mask float32
    return voxels * 4 * 2


def estimate_prefetch_ram_gb(cfg: dict[str, Any]) -> float:
    """Estimate worker-side prefetched batch memory in RAM."""
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    prefetch_factor = int(cfg["data"].get("prefetch_factor", 2) or 1)
    inflight_batches = max(1, num_workers) * max(1, prefetch_factor)
    total_bytes = _estimate_sample_bytes(cfg) * batch_size * inflight_batches
    return total_bytes / (1024 ** 3)


def _warmup_loader(loader: DataLoader, n_batches: int) -> None:
    iterator = iter(loader)
    try:
        for _ in range(max(0, n_batches)):
            next(iterator)
    finally:
        del iterator


def _apply_preflight_caps(cfg: dict[str, Any]) -> None:
    preflight_cfg = cfg["runtime"]["preflight"]
    data_cfg = cfg["data"]

    max_num_workers = preflight_cfg.get("max_num_workers")
    if max_num_workers is not None:
        data_cfg["num_workers"] = min(int(data_cfg.get("num_workers", 0)), int(max_num_workers))

    ram_cap = preflight_cfg.get("max_prefetch_ram_gb")
    if ram_cap is None:
        return

    while (
        int(data_cfg.get("num_workers", 0)) > 0
        and estimate_prefetch_ram_gb(cfg) > float(ram_cap)
    ):
        current_workers = int(data_cfg.get("num_workers", 0))
        current_prefetch = int(data_cfg.get("prefetch_factor", 2) or 1)
        if current_prefetch > 1:
            data_cfg["prefetch_factor"] = max(1, current_prefetch // 2)
        else:
            data_cfg["num_workers"] = max(
                int(preflight_cfg.get("worker_retry_min", 0)),
                current_workers // 2,
            )
        logger.warning(
            "Reduced loader settings during preflight to num_workers=%s, prefetch_factor=%s "
            "to stay under the configured RAM cap (estimated %.2f GiB).",
            data_cfg.get("num_workers"),
            data_cfg.get("prefetch_factor"),
            estimate_prefetch_ram_gb(cfg),
        )


def prepare_patch_dataloaders(
    cfg: dict[str, Any],
    data_root: str | Path,
) -> tuple[dict[str, Any], tuple[DataLoader, DataLoader, DataLoader]]:
    """Apply preflight checks and return a possibly adjusted config plus loaders."""
    adjusted = copy.deepcopy(cfg)
    preflight_cfg = adjusted["runtime"]["preflight"]
    enabled = bool(preflight_cfg.get("enabled", True))
    if not enabled:
        return adjusted, build_patch_dataloaders(adjusted, data_root)

    _apply_preflight_caps(adjusted)
    warmup_batches = int(preflight_cfg.get("loader_warmup_batches", 1))
    warmup_splits = list(preflight_cfg.get("warmup_splits", ["train"]))
    auto_reduce = bool(preflight_cfg.get("auto_reduce_workers", True))
    min_workers = int(preflight_cfg.get("worker_retry_min", 0))

    while True:
        loaders = build_patch_dataloaders(adjusted, data_root)
        try:
            split_to_loader = {
                "train": loaders[0],
                "val": loaders[1],
                "test": loaders[2],
            }
            for split in warmup_splits:
                if split not in split_to_loader:
                    continue
                _warmup_loader(split_to_loader[split], warmup_batches)
            return adjusted, loaders
        except Exception as exc:
            current_workers = int(adjusted["data"].get("num_workers", 0))
            if not auto_reduce or current_workers <= min_workers:
                raise RuntimeError(
                    "DataLoader preflight failed before training started. "
                    "Fix the loader configuration or dataset state and relaunch."
                ) from exc

            next_workers = max(min_workers, current_workers // 2)
            if next_workers == current_workers and current_workers > min_workers:
                next_workers = current_workers - 1
            adjusted["data"]["num_workers"] = next_workers
            if adjusted["data"].get("prefetch_factor") not in (None, 1):
                adjusted["data"]["prefetch_factor"] = max(
                    1,
                    int(adjusted["data"]["prefetch_factor"]) // 2,
                )
            logger.warning(
                "DataLoader preflight failed with num_workers=%d; retrying with num_workers=%d. "
                "Original error: %s",
                current_workers,
                next_workers,
                exc,
            )

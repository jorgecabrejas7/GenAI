"""Patch DataLoader helpers shared by training runners and preflight."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from poregen.dataset.loader import MemmapPatchDataset, PatchDataset, zarr_worker_init_fn

logger = logging.getLogger(__name__)


def build_dataloader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build DataLoader kwargs from the resolved config."""
    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "worker_init_fn": zarr_worker_init_fn if num_workers > 0 else None,
    }
    timeout = int(data_cfg.get("timeout", 0))
    if timeout > 0:
        kwargs["timeout"] = timeout
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def build_patch_dataloaders(
    cfg: dict[str, Any],
    data_root: str | Path,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct the train/val/test patch DataLoaders.

    Backend selection
    -----------------
    :class:`MemmapPatchDataset` is used when the files it actually reads are
    all present — ``patches_meta.json``, ``patches_xct.bin`` and
    ``patches_label.bin`` (produced by
    ``scripts/extract_patches_memmap.py``).  It reads pre-extracted patches
    from flat memmaps with sequential I/O and no chunk amplification, and does
    not require a ``worker_init_fn``.

    Otherwise, falls back to :class:`PatchDataset` (Zarr backend), which builds
    the same 3-class label from the store's ``mask`` and ``sample_mask``
    arrays.  A split root built before the 3-class label — ``split_v2``, which
    has ``patches_mask.bin`` and no ``patches_label.bin`` — therefore reads
    through Zarr rather than failing on the missing file.
    """
    root       = Path(data_root)
    index_path = root / "patch_index.parquet"

    required = ("patches_meta.json", "patches_xct.bin", "patches_label.bin")
    missing = [f for f in required if not (root / f).exists()]
    use_memmap = not missing
    if use_memmap:
        DatasetClass = MemmapPatchDataset
        logger.info("Patch backend: memmap (%s)", root / "patches_meta.json")
    else:
        DatasetClass = PatchDataset
        logger.info("Patch backend: zarr (%s missing from %s)",
                    ", ".join(missing), root)

    train_ds = DatasetClass(index_path, root, split="train")
    val_ds   = DatasetClass(index_path, root, split="val")
    test_ds  = DatasetClass(index_path, root, split="test")

    dl_kwargs = build_dataloader_kwargs(cfg)
    if use_memmap:
        # Memmap file descriptors are fork-safe; no init_fn needed.
        dl_kwargs["worker_init_fn"] = None

    val_generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]) + 1)

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=True,  generator=val_generator, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False,                  **dl_kwargs)
    return train_loader, val_loader, test_loader

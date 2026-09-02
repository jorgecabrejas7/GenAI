"""Load side of the ldm04 latent dataset.

Reads the layout produced by ``scripts/build_latent_dataset.py``::

    <root>/                      e.g. data/split_v2/latents_r07z4/
    ├── metadata.json            — latent shape, per-channel train-split
    │                              normalisation stats, VAE provenance
    └── <split>/
        ├── latents.zarr/        — arrays "mu" and "std", (N, C, d, h, w) float16
        └── index.parquet        — source_row, volume_id, z0, y0, x0, phi, …

Row i of ``index.parquet`` aligns with row i of both zarr arrays.

Latents are stored raw; :class:`LatentDataset` optionally applies per-channel
normalisation ``(mu - mean_c) / std_c`` at load time.  The posterior std is
scaled by ``1 / std_c`` under the same affine map.

Worker safety: the zarr group is opened once per instance; after ``fork()``
each DataLoader worker inherits its own copy of the handle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """Random-access dataset of pre-computed VAE posterior latents.

    Parameters
    ----------
    root : str | Path
        Latent dataset root (contains ``metadata.json`` and split dirs).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    normalize : bool
        If True, return per-channel normalised latents using the train-split
        stats stored in ``metadata.json``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        normalize: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.normalize = normalize

        with open(self.root / "metadata.json") as fh:
            self.metadata: dict[str, Any] = json.load(fh)

        norm = self.metadata["normalization"]
        c = len(norm["per_channel_mean"])
        self.channel_mean = torch.tensor(
            norm["per_channel_mean"], dtype=torch.float32
        ).view(c, 1, 1, 1)
        self.channel_std = torch.tensor(
            norm["per_channel_std"], dtype=torch.float32
        ).view(c, 1, 1, 1)

        split_dir = self.root / split
        self.df = pd.read_parquet(str(split_dir / "index.parquet"))
        store = zarr.open_group(str(split_dir / "latents.zarr"), mode="r")
        self._mu = store["mu"]
        self._std = store["std"]

        if len(self.df) != self._mu.shape[0]:
            raise RuntimeError(
                f"LatentDataset [{split}]: index.parquet has {len(self.df)} rows "
                f"but latents.zarr/mu has {self._mu.shape[0]} — rebuild the store."
            )
        self.latent_shape: tuple[int, ...] = tuple(self._mu.shape[1:])
        logger.info(
            "LatentDataset [%s]: %d latents %s  normalize=%s",
            split, len(self.df), self.latent_shape, normalize,
        )

    # -- normalisation helpers -------------------------------------------

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Per-channel normalise a raw latent (C, d, h, w) or (B, C, d, h, w)."""
        return (z - self.channel_mean) / self.channel_std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize_latent`."""
        return z * self.channel_std + self.channel_mean

    # -- Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        z = torch.from_numpy(np.asarray(self._mu[idx], dtype=np.float32))
        std = torch.from_numpy(np.asarray(self._std[idx], dtype=np.float32))

        if self.normalize:
            z = self.normalize_latent(z)
            std = std / self.channel_std

        return {
            "z": z,                                                      # (C, d, h, w)
            "std": std,                                                  # (C, d, h, w)
            "phi": torch.tensor([float(row["phi"])], dtype=torch.float32),
            "volume_id": row["volume_id"],
            "coords": torch.tensor(
                [int(row["z0"]), int(row["y0"]), int(row["x0"])], dtype=torch.int32
            ),
            "source_row": int(row["source_row"]),
        }


def build_latent_dataloaders(
    cfg: dict[str, Any],
    latents_root: str | Path,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders over :class:`LatentDataset`.

    The LDM always trains in normalised latent space (``normalize=True``);
    denormalisation happens only immediately before VAE decoding.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg    = cfg.get("data", {})
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    train_ds = LatentDataset(latents_root, "train", normalize=True)
    val_ds   = LatentDataset(latents_root, "val",   normalize=True)

    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"]    = int(data_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader

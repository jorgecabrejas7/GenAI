"""Dataset for pre-computed VAE latents used in LDM training.

Expected on-disk layout (produced by ``scripts/encode_latents.py``)::

    <latents_root>/
    ├── latents.zarr/
    │   └── latents    — (N_total, z_ch, 16, 16, 16)  float16  Blosc-zstd
    └── latents_index.parquet
        columns: volume_id, z0, y0, x0, ps, porosity, split,
                 source_group, vol_depth, vol_height, vol_width

The parquet row index aligns 1-to-1 with the Zarr array index, so
``latents.zarr/latents[i]`` is the encoded latent for row ``i``.

Neighbor lookup
---------------
For each patch at ``(volume_id, z0, y0, x0)``, the 6 axis-aligned neighbors
are patches in the same volume at coordinates offset by ±``patch_stride`` in
exactly one dimension.  Availability states:

    0 (OOB)     — position not present in the index (outside the volume)
    1 (EXISTS)  — latent found in the index
    2 (UNKNOWN) — EXISTS but randomly masked with probability *nb_dropout_prob*
                  during training to simulate the inference regime where
                  neighbors have not yet been generated.

Worker safety
-------------
The Zarr store is opened once per :class:`LatentPatchDataset` instance.
After ``fork()`` each DataLoader worker inherits its own copy of the handle
— no explicit ``worker_init_fn`` required beyond the existing
:func:`~poregen.dataset.loader.zarr_worker_init_fn` pattern.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from poregen.diffusion.conditioning import NB_EXISTS, NB_OOB, NB_UNKNOWN

logger = logging.getLogger(__name__)

_NEIGHBOR_DIRS = [
    ( 1, 0, 0), (-1, 0, 0),
    ( 0, 1, 0), ( 0,-1, 0),
    ( 0, 0, 1), ( 0, 0,-1),
]


class LatentPatchDataset(Dataset):
    """Random-access dataset of pre-computed VAE latents.

    Parameters
    ----------
    latents_root : str | Path
        Directory containing ``latents.zarr`` and ``latents_index.parquet``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    nb_dropout_prob : float
        Probability of randomly masking an existing neighbor as UNKNOWN
        during training.  Set to 0.0 for deterministic evaluation.
    patch_stride : int
        Voxel stride between adjacent patches (default 32).
    """

    def __init__(
        self,
        latents_root: str | Path,
        split: str,
        *,
        nb_dropout_prob: float = 0.2,
        patch_stride: int = 32,
    ) -> None:
        self.latents_root    = Path(latents_root)
        self.split           = split
        self.nb_dropout_prob = nb_dropout_prob
        self.patch_stride    = patch_stride

        # Load and filter index
        idx_path = self.latents_root / "latents_index.parquet"
        df_full  = pd.read_parquet(str(idx_path))
        self.df  = df_full[df_full["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"LatentPatchDataset: no patches found for split='{split}' in {idx_path}")
        logger.info("LatentPatchDataset [%s]: %d patches", split, len(self.df))

        # Open Zarr store (handle shared via fork across workers)
        zarr_path = self.latents_root / "latents.zarr"
        self._zarr: zarr.Array = zarr.open_array(str(zarr_path / "latents"), mode="r")

        # z_channels inferred from Zarr array shape (N, C, D, H, W)
        self.z_channels  = self._zarr.shape[1]
        self.latent_size = self._zarr.shape[2]   # spatial side length (16)

        # Load latent scale stats for normalisation (computed from train split by encode_latents.py)
        _stats_path = self.latents_root / "latent_scale_stats.json"
        if _stats_path.exists():
            with open(_stats_path) as _sf:
                _stats = json.load(_sf)
            self.latent_std = float(_stats["std"])
            logger.info("LatentPatchDataset [%s]: latent_std=%.4f", split, self.latent_std)
        else:
            self.latent_std = 1.0
            logger.warning("latent_scale_stats.json not found — using std=1.0")

        # Per-volume mean porosity for global_por conditioning (volume-level VVF)
        self._vol_porosity: dict[str, float] = (
            df_full.groupby("volume_id")["porosity"].mean().to_dict()
        )

        # Build coordinate → row-index lookup for neighbor resolution
        # Key: (volume_id, z0, y0, x0) → integer row index in the FULL parquet
        # We need the global index (into the Zarr array), not the split-filtered index.
        self._coord_to_global: dict[tuple[str, int, int, int], int] = {}
        for global_idx, row in df_full.iterrows():
            key = (row["volume_id"], int(row["z0"]), int(row["y0"]), int(row["x0"]))
            self._coord_to_global[key] = int(global_idx)

    # ------------------------------------------------------------------
    # Neighbour helpers
    # ------------------------------------------------------------------

    def _get_neighbor(
        self,
        vol_id: str,
        z0: int,
        y0: int,
        x0: int,
        dz: int,
        dy: int,
        dx: int,
    ) -> tuple[torch.Tensor, int]:
        """Load one neighbor latent and return its availability state.

        Returns
        -------
        (latent_tensor, state)
        latent_tensor : (z_ch, D, H, W) float32 — zeros if unavailable
        state         : NB_OOB | NB_EXISTS | NB_UNKNOWN
        """
        key = (vol_id, z0 + dz * self.patch_stride, y0 + dy * self.patch_stride, x0 + dx * self.patch_stride)
        zero = torch.zeros(self.z_channels, self.latent_size, self.latent_size, self.latent_size)

        if key not in self._coord_to_global:
            return zero, NB_OOB

        # EXISTS — apply dropout to simulate UNKNOWN during training
        if self.nb_dropout_prob > 0.0 and random.random() < self.nb_dropout_prob:
            return zero, NB_UNKNOWN

        global_idx = self._coord_to_global[key]
        raw = np.array(self._zarr[global_idx], dtype=np.float32)  # (C, D, H, W)
        return torch.from_numpy(raw) / self.latent_std, NB_EXISTS

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row    = self.df.iloc[idx]
        vid    = row["volume_id"]
        z0     = int(row["z0"])
        y0     = int(row["y0"])
        x0     = int(row["x0"])
        por    = float(row["porosity"])

        # Volume dimensions for normalised position
        vol_d  = int(row["vol_depth"])
        vol_h  = int(row["vol_height"])
        vol_w  = int(row["vol_width"])

        # Clean latent (x0 for diffusion), normalised to ~N(0,1) by dividing by latent_std
        # Global index = original parquet row; split-filtered df re-indexed from 0
        global_idx = self._coord_to_global[(vid, z0, y0, x0)]
        raw_z = np.array(self._zarr[global_idx], dtype=np.float32)
        z     = torch.from_numpy(raw_z) / self.latent_std           # (C, D, H, W)

        # Neighbors
        nb_latents = []
        nb_avail   = []
        for dz, dy, dx in _NEIGHBOR_DIRS:
            nb_z, state = self._get_neighbor(vid, z0, y0, x0, dz, dy, dx)
            nb_latents.append(nb_z)
            nb_avail.append(state)

        nb_latents_t = torch.stack(nb_latents, dim=0)               # (6, C, D, H, W)
        nb_avail_t   = torch.tensor(nb_avail, dtype=torch.long)     # (6,)

        # Normalised 3-D position (clamp to [0,1] in case of floating-point edge)
        pos_frac = torch.tensor([
            z0 / max(vol_d - 1, 1),
            y0 / max(vol_h - 1, 1),
            x0 / max(vol_w - 1, 1),
        ], dtype=torch.float32).clamp(0.0, 1.0)

        global_por = self._vol_porosity.get(vid, por)

        return {
            "z":          z,                                                        # (C, D, H, W) float32
            "nb_latents": nb_latents_t,                                             # (6, C, D, H, W) float32
            "nb_avail":   nb_avail_t,                                               # (6,) long
            "pos_frac":   pos_frac,                                                 # (3,) float32
            "global_por": torch.tensor([global_por], dtype=torch.float32),          # (1,) volume-mean VVF
            "local_por":  torch.tensor([por],         dtype=torch.float32),         # (1,) patch VVF
            "volume_id":  vid,
            "coords":     torch.tensor([z0, y0, x0], dtype=torch.int32),
        }


def build_latent_dataloaders(
    cfg: dict[str, Any],
    latents_root: str | Path,
) -> tuple[Any, Any]:
    """Build train and val :class:`LatentPatchDataset` DataLoaders.

    Parameters
    ----------
    cfg : resolved experiment config dict
    latents_root : path to the latents directory

    Returns
    -------
    (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    data_cfg    = cfg.get("data", {})
    stride      = int(data_cfg.get("patch_stride", 32))
    nb_drop     = float(data_cfg.get("nb_dropout_prob", 0.2))
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory  = bool(data_cfg.get("pin_memory", True))

    train_ds = LatentPatchDataset(latents_root, "train", nb_dropout_prob=nb_drop, patch_stride=stride)
    val_ds   = LatentPatchDataset(latents_root, "val",   nb_dropout_prob=0.0,     patch_stride=stride)

    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"]    = int(data_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader

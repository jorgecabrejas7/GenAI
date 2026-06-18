"""Dataset for pre-computed VAE latents used in LDM training.

Expected on-disk layout (produced by ``scripts/encode_latents.py``)::

    <latents_root>/
    ├── latents.zarr/
    │   └── latents    — (N_total, z_ch, 16, 16, 16)  float16  Blosc-zstd
    └── latents_index.parquet
        columns: volume_id, z0, y0, x0, ps, stride, porosity, vol_porosity, split,
                 source_group, vol_depth, vol_height, vol_width,
                 grid_iz, grid_iy, grid_ix, parity

The parquet row index aligns 1-to-1 with the Zarr array index, so
``latents.zarr/latents[i]`` is the encoded latent for row ``i``.

Neighbor conditioning (inference-faithful)
------------------------------------------
Each patch is classified by its checkerboard parity:
``parity = (grid_iz + grid_iy + grid_ix) % 2``

For each of the 6 axis-aligned neighbors the availability state is:

    0 (OOB)     — neighbor position outside the volume (same for both parities)
    1 (EXISTS)  — parity-1 patch: neighbor is an anchor that is already
                  generated; real latent loaded from Zarr
    2 (UNKNOWN) — parity-0 patch: neighbor is inside the volume but has not
                  been generated yet at inference time; zeros returned

This exactly mirrors the checkerboard BFS generation order used at inference:
anchors (parity=0) generate first with all in-bounds neighbors UNKNOWN;
non-anchors (parity=1) generate after all anchors, seeing all in-bounds
neighbors as EXISTS.  No random dropout is applied — the conditioning is
deterministic and structurally correct by construction.

Worker safety
-------------
The Zarr store is opened once per :class:`LatentPatchDataset` instance.
After ``fork()`` each DataLoader worker inherits its own copy of the handle.
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
    patch_stride : int
        Voxel stride between adjacent patches.  Must match the stride used
        when building the latent store (default 64).
    """

    def __init__(
        self,
        latents_root: str | Path,
        split: str,
        *,
        patch_stride: int = 64,
    ) -> None:
        self.latents_root = Path(latents_root)
        self.split        = split
        self.patch_stride = patch_stride

        # Load and filter index
        idx_path = self.latents_root / "latents_index.parquet"
        df_full  = pd.read_parquet(str(idx_path))

        if "parity" not in df_full.columns:
            raise ValueError(
                f"latents_index.parquet at {idx_path} is missing the 'parity' column. "
                "Re-run encode_latents.py to rebuild the latent store."
            )

        self.df = df_full[df_full["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"LatentPatchDataset: no patches found for split='{split}' in {idx_path}")
        logger.info("LatentPatchDataset [%s]: %d patches", split, len(self.df))

        # Open Zarr store (handle shared via fork across workers)
        zarr_path = self.latents_root / "latents.zarr"
        self._zarr: zarr.Array = zarr.open_array(str(zarr_path / "latents"), mode="r")

        self.z_channels  = self._zarr.shape[1]
        self.latent_size = self._zarr.shape[2]   # spatial side length (16)

        # Latent scale stats for normalisation (computed from train split)
        _stats_path = self.latents_root / "latent_scale_stats.json"
        if _stats_path.exists():
            with open(_stats_path) as _sf:
                _stats = json.load(_sf)
            self.latent_std = float(_stats["std"])
            logger.info("LatentPatchDataset [%s]: latent_std=%.4f", split, self.latent_std)
        else:
            self.latent_std = 1.0
            logger.warning("latent_scale_stats.json not found — using std=1.0")

        # Per-volume true VVF for global_por conditioning.
        # Use vol_porosity (mask.mean() over the full volume) when available;
        # fall back to mean(patch_porosity) for legacy parquet, which is only
        # correct when stride == patch_size (no overlap).
        if "vol_porosity" in df_full.columns:
            self._vol_porosity: dict[str, float] = (
                df_full.drop_duplicates("volume_id")
                       .set_index("volume_id")["vol_porosity"]
                       .to_dict()
            )
        else:
            logger.warning(
                "latents_index.parquet has no 'vol_porosity' column — "
                "falling back to mean(patch_porosity), which is biased when "
                "stride < patch_size. Re-run encode_latents.py to fix this."
            )
            self._vol_porosity = (
                df_full.groupby("volume_id")["porosity"].mean().to_dict()
            )

        # Coordinate → global Zarr index, built from ALL splits so neighbor
        # lookups across split boundaries work correctly.
        self._coord_to_global: dict[tuple[str, int, int, int], int] = {}
        for global_idx, row in df_full.iterrows():
            key = (row["volume_id"], int(row["z0"]), int(row["y0"]), int(row["x0"]))
            self._coord_to_global[key] = int(global_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row    = self.df.iloc[idx]
        vid    = row["volume_id"]
        z0     = int(row["z0"])
        y0     = int(row["y0"])
        x0     = int(row["x0"])
        por    = float(row["porosity"])
        parity = int(row["parity"])

        vol_d = int(row["vol_depth"])
        vol_h = int(row["vol_height"])
        vol_w = int(row["vol_width"])

        # Target latent
        global_idx = self._coord_to_global[(vid, z0, y0, x0)]
        raw_z = np.array(self._zarr[global_idx], dtype=np.float32)
        z     = torch.from_numpy(raw_z) / self.latent_std    # (C, 16, 16, 16)

        # Neighbor conditioning — deterministic, inference-faithful
        zero = torch.zeros(self.z_channels, self.latent_size, self.latent_size, self.latent_size)
        nb_latents_list: list[torch.Tensor] = []
        nb_avail_list:   list[int]          = []

        for dz, dy, dx in _NEIGHBOR_DIRS:
            nb_key = (
                vid,
                z0 + dz * self.patch_stride,
                y0 + dy * self.patch_stride,
                x0 + dx * self.patch_stride,
            )

            if nb_key not in self._coord_to_global:
                # Outside the volume — genuine boundary, same for both parities
                nb_latents_list.append(zero)
                nb_avail_list.append(NB_OOB)
            elif parity == 0:
                # Anchor: neighbor exists in the volume but has not been
                # generated yet at inference time → UNKNOWN
                nb_latents_list.append(zero)
                nb_avail_list.append(NB_UNKNOWN)
            else:
                # Non-anchor: neighbor is an anchor already generated → EXISTS
                nb_global = self._coord_to_global[nb_key]
                raw_nb = np.array(self._zarr[nb_global], dtype=np.float32)
                nb_latents_list.append(torch.from_numpy(raw_nb) / self.latent_std)
                nb_avail_list.append(NB_EXISTS)

        nb_latents_t = torch.stack(nb_latents_list, dim=0)          # (6, C, 16, 16, 16)
        nb_avail_t   = torch.tensor(nb_avail_list, dtype=torch.long) # (6,)

        # Normalised 3-D position within the volume
        pos_frac = torch.tensor([
            z0 / max(vol_d - 1, 1),
            y0 / max(vol_h - 1, 1),
            x0 / max(vol_w - 1, 1),
        ], dtype=torch.float32).clamp(0.0, 1.0)

        global_por = self._vol_porosity.get(vid, por)

        return {
            "z":          z,                                                       # (C, 16, 16, 16)
            "nb_latents": nb_latents_t,                                            # (6, C, 16, 16, 16)
            "nb_avail":   nb_avail_t,                                              # (6,) long
            "pos_frac":   pos_frac,                                                # (3,) float32
            "global_por": torch.tensor([global_por], dtype=torch.float32),         # (1,) volume VVF
            "local_por":  torch.tensor([por],         dtype=torch.float32),        # (1,) patch VVF
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
    stride      = int(data_cfg.get("patch_stride", 64))
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory  = bool(data_cfg.get("pin_memory", True))

    train_ds = LatentPatchDataset(latents_root, "train", patch_stride=stride)
    val_ds   = LatentPatchDataset(latents_root, "val",   patch_stride=stride)

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

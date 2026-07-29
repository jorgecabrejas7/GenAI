"""Dataset for VAE posterior parameters (mu, logvar) used in ldm02 training.

Samples z = mu + sigma*eps with fresh eps per __getitem__ call, giving proper
stochastic encoding (Rombach et al., noise augmentation) rather than the frozen
posterior mean used in ldm01.

Expected on-disk layout (produced by ``scripts/encode_latents_sampled.py``)::

    <latents_root>/
    ├── latents.zarr/
    │   └── latents   (N_total, 2*z_ch, 16, 16, 16) float16
    │                 channels 0..z_ch-1      = mu
    │                 channels z_ch..2*z_ch-1 = logvar
    ├── latents_index.parquet
    │   columns: volume_id, z0, y0, x0, ps, stride, porosity, vol_porosity, split,
    │             source_group, vol_depth, vol_height, vol_width,
    │             grid_iz, grid_iy, grid_ix, parity
    └── latent_scale_stats.json
        {"z_channels": C, "mean": ..., "std": ...}
        — std computed from z=mu+sigma*eps on train split only

Parquet row i aligns 1-to-1 with zarr index i in latents.zarr/latents.
A single zarr read per item retrieves both mu and logvar — same I/O cost as
ldm01's single-array design.

Neighbor conditioning follows the same inference-faithful checkerboard scheme as
latent_dataset.py: parity-0 anchors see UNKNOWN neighbors, parity-1 non-anchors
see EXISTS neighbors.  For EXISTS neighbors the latent is also sampled fresh:
``z_nb = mu_nb + sigma_nb * eps_nb`` with independent eps_nb.
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


class SampledLatentPatchDataset(Dataset):
    """Random-access dataset of VAE posterior parameters for ldm02.

    On each ``__getitem__`` call z = mu + exp(0.5*logvar)*eps is drawn fresh,
    providing stochastic encoding as training-time data augmentation.

    Parameters
    ----------
    latents_root : str | Path
        Directory containing ``latents.zarr``, ``latents_index.parquet``,
        and ``latent_scale_stats.json``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    patch_stride : int
        Voxel stride between adjacent patches.
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

        idx_path = self.latents_root / "latents_index.parquet"
        df_full  = pd.read_parquet(str(idx_path))

        if "parity" not in df_full.columns:
            raise ValueError(
                f"latents_index.parquet at {idx_path} is missing the 'parity' column. "
                "Re-run encode_latents_sampled.py to rebuild the latent store."
            )

        self.df = df_full[df_full["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"SampledLatentPatchDataset: no patches found for split='{split}' in {idx_path}"
            )
        logger.info("SampledLatentPatchDataset [%s]: %d patches", split, len(self.df))

        # --- backend selection ---
        _mmap_bin  = self.latents_root / "latents.bin"
        _mmap_meta = self.latents_root / "latents_meta.json"
        _zarr_arr  = self.latents_root / "latents.zarr" / "latents"

        if _mmap_bin.exists() and _mmap_meta.exists():
            with open(_mmap_meta) as _f:
                _meta = json.load(_f)
            if _meta.get("pack_scheme") != "mu_then_logvar":
                raise ValueError(
                    f"SampledLatentPatchDataset requires pack_scheme='mu_then_logvar' in "
                    f"latents_meta.json, got {_meta.get('pack_scheme')!r}. "
                    "Re-run encode_latents_sampled.py to rebuild."
                )
            assert len(df_full) == _meta["N"], (
                f"Parquet has {len(df_full)} rows but latents_meta.json reports N={_meta['N']}."
            )
            total_ch = int(_meta["n_channels"])
            self._mmap = np.memmap(
                str(_mmap_bin), dtype=np.dtype(_meta["dtype"]), mode="r",
                shape=(_meta["N"], total_ch, *_meta["spatial"]),
            )
            self._arr  = None
            self.latent_size = int(_meta["spatial"][0])
            logger.info(
                "SampledLatentPatchDataset: memmap backend shape=%s z_channels=%d",
                self._mmap.shape, _meta["z_channels"],
            )
        elif (_zarr_arr.parent.exists()):
            self._mmap = None
            self._arr  = zarr.open_array(str(_zarr_arr), mode="r")
            total_ch = self._arr.shape[1]
            self.latent_size = self._arr.shape[2]
            logger.warning(
                "SampledLatentPatchDataset: zarr fallback — latents.bin not found in %s. "
                "Run scripts/convert_zarr_to_memmap.py to upgrade.",
                self.latents_root,
            )
        else:
            raise FileNotFoundError(
                f"No latent store found in {self.latents_root}. "
                "Expected latents.bin + latents_meta.json (memmap) or "
                "latents.zarr/latents (legacy zarr)."
            )

        if total_ch % 2 != 0:
            raise ValueError(
                f"Expected packed zarr/memmap with even channel count (2*z_ch), got {total_ch}. "
                "Re-run encode_latents_sampled.py."
            )
        self.z_channels = total_ch // 2
        logger.info(
            "SampledLatentPatchDataset: z_channels=%d  latent_size=%d",
            self.z_channels, self.latent_size,
        )

        # Scale stats computed on sampled z (train split only)
        _stats_path = self.latents_root / "latent_scale_stats.json"
        if _stats_path.exists():
            with open(_stats_path) as _sf:
                _stats = json.load(_sf)
            self.latent_std = float(_stats["std"])
            logger.info(
                "SampledLatentPatchDataset [%s]: latent_std=%.4f (sampled-z scale)",
                split, self.latent_std,
            )
        else:
            self.latent_std = 1.0
            logger.warning(
                "latent_scale_stats.json not found in %s — using std=1.0 (check encode script)",
                self.latents_root,
            )

        # Per-volume VVF for global_por conditioning
        if "vol_porosity" in df_full.columns:
            self._vol_porosity: dict[str, float] = (
                df_full.drop_duplicates("volume_id")
                       .set_index("volume_id")["vol_porosity"]
                       .to_dict()
            )
        else:
            logger.warning(
                "latents_index.parquet has no 'vol_porosity' column — "
                "falling back to mean(patch_porosity)."
            )
            self._vol_porosity = (
                df_full.groupby("volume_id")["porosity"].mean().to_dict()
            )

        # Coordinate → global zarr index (all splits, for neighbor lookups)
        self._coord_to_global: dict[tuple[str, int, int, int], int] = {}
        for global_idx, row in df_full.iterrows():
            key = (row["volume_id"], int(row["z0"]), int(row["y0"]), int(row["x0"]))
            self._coord_to_global[key] = int(global_idx)

    def _read_packed(self, global_idx: int) -> np.ndarray:
        """Read one (2*z_channels, 16, 16, 16) packed [mu|logvar] item as float32."""
        if self._mmap is not None:
            return np.array(self._mmap[global_idx], dtype=np.float32)
        return np.array(self._arr[global_idx], dtype=np.float32)

    def _sample_z(self, global_idx: int) -> torch.Tensor:
        """One read for packed [mu | logvar]; sample z = mu + sigma*eps.

        Uses torch.randn so that DataLoader's per-worker torch-seed (set
        automatically per worker) gives independent draws across workers.
        np.random.randn must NOT be used here — numpy's global RNG is NOT
        re-seeded per worker after fork.
        """
        packed = torch.from_numpy(
            self._read_packed(global_idx)
        )   # (2*C, 16, 16, 16)
        mu     = packed[:self.z_channels]
        logvar = packed[self.z_channels:]
        sigma  = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(sigma)
        return mu + sigma * eps   # (C, 16, 16, 16)

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

        global_idx = self._coord_to_global[(vid, z0, y0, x0)]
        z = self._sample_z(global_idx) / self.latent_std    # (C, 16, 16, 16)

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
                nb_latents_list.append(zero)
                nb_avail_list.append(NB_OOB)
            elif parity == 0:
                nb_latents_list.append(zero)
                nb_avail_list.append(NB_UNKNOWN)
            else:
                nb_global = self._coord_to_global[nb_key]
                nb_z = self._sample_z(nb_global) / self.latent_std
                nb_latents_list.append(nb_z)
                nb_avail_list.append(NB_EXISTS)

        nb_latents_t = torch.stack(nb_latents_list, dim=0)            # (6, C, 16, 16, 16)
        nb_avail_t   = torch.tensor(nb_avail_list, dtype=torch.long)  # (6,)

        pos_frac = torch.tensor([
            z0 / max(vol_d - 1, 1),
            y0 / max(vol_h - 1, 1),
            x0 / max(vol_w - 1, 1),
        ], dtype=torch.float32).clamp(0.0, 1.0)

        global_por = self._vol_porosity.get(vid, por)

        return {
            "z":          z,
            "nb_latents": nb_latents_t,
            "nb_avail":   nb_avail_t,
            "pos_frac":   pos_frac,
            "global_por": torch.tensor([global_por], dtype=torch.float32),
            "local_por":  torch.tensor([por],         dtype=torch.float32),
            "volume_id":  vid,
            "coords":     torch.tensor([z0, y0, x0], dtype=torch.int32),
        }


def build_sampled_latent_dataloaders(
    cfg: dict[str, Any],
    latents_root: str | Path,
) -> tuple[Any, Any]:
    """Build train and val DataLoaders from the sampled latent store (ldm02).

    Drop-in replacement for ``build_latent_dataloaders`` when
    ``cfg["data"]["latent_mode"] == "sampled"``.
    """
    from torch.utils.data import DataLoader

    data_cfg    = cfg.get("data", {})
    stride      = int(data_cfg.get("patch_stride", 64))
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory  = bool(data_cfg.get("pin_memory", True))

    train_ds = SampledLatentPatchDataset(latents_root, "train", patch_stride=stride)
    val_ds   = SampledLatentPatchDataset(latents_root, "val",   patch_stride=stride)

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

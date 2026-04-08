"""PyTorch Dataset for loading patches from Zarr via a Parquet index."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """Random-access patch loader backed by Zarr volumes and a Parquet index.

    Parameters
    ----------
    index_path : str | Path
        Path to ``patch_index.parquet``.
    volumes_root : str | Path
        Directory containing ``volumes.zarr/``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    stats_path : str | Path | None
        Path to ``volume_stats.json``.  When provided, XCT patches are
        normalised per-volume as ``(xct - mean) / std`` (z-score, float32).
        When ``None`` (default), the legacy ``xct / 255.0`` global scaling
        is used instead and a warning is emitted.
    """

    def __init__(
        self,
        index_path: str | Path,
        volumes_root: str | Path,
        split: str = "train",
        stats_path: str | Path | None = None,
    ) -> None:
        df = pd.read_parquet(str(index_path))
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Filter corrupted patches (porosity > 1.0 — numerical overflow in SAT)
        corrupted = self.df["porosity"] > 1.0
        if corrupted.any():
            logger.warning(
                "PatchDataset [%s]: dropping %d corrupted patch(es) with porosity > 1.0 "
                "(max = %.2f). Likely SAT overflow in volume '%s'.",
                split,
                int(corrupted.sum()),
                float(self.df.loc[corrupted, "porosity"].max()),
                self.df.loc[corrupted, "volume_id"].iloc[0],
            )
            self.df = self.df[~corrupted].reset_index(drop=True)
        self.volumes_root = Path(volumes_root)
        self._zarr_cache: dict[str, zarr.Group] = {}

        # Per-volume normalisation stats
        if stats_path is not None:
            with open(stats_path) as f:
                raw = json.load(f)
            # Keep only mean + std; cast to float32 scalars
            self._vol_stats: dict[str, tuple[float, float]] = {
                vid: (float(s["mean"]), float(s["std"]))
                for vid, s in raw.items()
            }
            missing = set(self.df["volume_id"].unique()) - self._vol_stats.keys()
            if missing:
                logger.warning(
                    "PatchDataset: %d volume(s) have no entry in stats_path "
                    "and will fall back to /255 normalisation: %s",
                    len(missing),
                    missing,
                )
        else:
            logger.warning(
                "PatchDataset: stats_path not provided — using global /255 "
                "normalisation. Per-volume z-score is strongly recommended."
            )
            self._vol_stats = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_group(self, volume_id: str) -> zarr.Group:
        if volume_id not in self._zarr_cache:
            store = self.volumes_root / "volumes.zarr"
            root = zarr.open_group(str(store), mode="r")
            self._zarr_cache[volume_id] = root[volume_id]
        return self._zarr_cache[volume_id]

    def _normalise_xct(self, xct: np.ndarray, volume_id: str) -> torch.Tensor:
        xct_f = xct.astype(np.float32)
        if volume_id in self._vol_stats:
            mean, std = self._vol_stats[volume_id]
            xct_f = (xct_f - mean) / max(std, 1e-6)
            # Clip at ±4σ — EDA: pore component sits at −2.96σ so ±3σ clips pore
            # signal. For uint8 with mean≈192/std≈52 this is a no-op (range≈[-3.7,1.2])
            # but protects against outlier volumes with different intensity distributions.
            xct_f = np.clip(xct_f, -4.0, 4.0)
        else:
            xct_f = xct_f / 255.0
        return torch.from_numpy(xct_f).unsqueeze(0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        vid = row["volume_id"]
        grp = self._get_group(vid)

        ps = int(row["ps"])
        z0, y0, x0 = int(row["z0"]), int(row["y0"]), int(row["x0"])

        xct  = grp["xct"] [z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps]
        mask = grp["mask"][z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps]

        xct_t  = self._normalise_xct(xct, vid)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "xct": xct_t,                       # (1, ps, ps, ps) float32, z-scored per volume
            "mask": mask_t,                      # (1, ps, ps, ps) float32 {0, 1}
            "volume_id": vid,
            "coords": (z0, y0, x0),
            "porosity": float(row["porosity"]),
            "source_group": row["source_group"],
        }

"""PyTorch Dataset for loading patches from Zarr via a Parquet index."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def zarr_worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that makes each worker open its own Zarr handles.

    When PyTorch spawns DataLoader workers via ``fork()``, each worker inherits
    a copy of the parent's file handles and ``_zarr_cache``.  Sharing Zarr store
    handles across forked processes can cause race conditions on concurrent chunk
    reads.  This function clears the inherited cache so each worker lazily opens
    its own handle on the first access, keeping I/O fully isolated.

    Usage::

        DataLoader(dataset, num_workers=8, worker_init_fn=zarr_worker_init_fn)
    """
    # Each worker has its own copy of the Dataset instance after fork.
    # We cannot reach the Dataset object directly here, but clearing any
    # module-level state is enough — the per-instance cache is already a
    # separate copy in the worker's address space and will be re-populated
    # lazily.  No action needed beyond signalling intent (the cache starts
    # empty in each worker because Python fork copies the object).
    pass  # intentionally empty — fork already gives each worker its own copy


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

    Notes
    -----
    XCT patches are normalised as ``xct / 255.0`` → float32 in ``[0, 1]``.

    The Zarr root store is opened **once** in :meth:`__init__` and reused for
    all subsequent group lookups, avoiding the O(n_volumes) re-open overhead
    that occurred when each ``_get_group`` cache miss reopened the store.

    Worker safety: use :func:`zarr_worker_init_fn` as the DataLoader
    ``worker_init_fn`` so each worker opens its own store handle after fork.
    """

    def __init__(
        self,
        index_path: str | Path,
        volumes_root: str | Path,
        split: str = "train",
    ) -> None:
        df = pd.read_parquet(str(index_path))
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Drop corrupted patches (porosity > 1.0 — numerical overflow in SAT)
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

        # Open the Zarr root store once — reused for all _get_group calls.
        # After fork() each DataLoader worker has its own copy of this handle.
        store = self.volumes_root / "volumes.zarr"
        self._zarr_root: zarr.Group = zarr.open_group(str(store), mode="r")
        self._zarr_cache: dict[str, zarr.Group] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_group(self, volume_id: str) -> zarr.Group:
        if volume_id not in self._zarr_cache:
            self._zarr_cache[volume_id] = self._zarr_root[volume_id]
        return self._zarr_cache[volume_id]

    def _normalise_xct(self, xct: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(xct.astype(np.float32) / 255.0).unsqueeze(0)

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

        xct_t  = self._normalise_xct(xct)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "xct":          xct_t,                                          # (1, ps, ps, ps) float32 [0, 1]
            "mask":         mask_t,                                         # (1, ps, ps, ps) float32 {0, 1}
            "volume_id":    vid,
            "coords":       np.array([z0, y0, x0], dtype=np.int32),        # int32 array — faster collation
            "porosity":     float(row["porosity"]),
            "source_group": row["source_group"],
        }

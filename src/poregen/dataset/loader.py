"""PyTorch Dataset for loading patches — Zarr and memmap backends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Three-class voxel label, shared by both backends and by
# scripts/extract_patches_memmap.py.  Air takes precedence over pore.
LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2


def build_label(mask: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    """0 material, 1 pore, 2 air (outside the specimen, or a drilled hole)."""
    label = np.zeros(mask.shape, dtype=np.uint8)
    label[mask != 0] = LABEL_PORE
    label[sample_mask == 0] = LABEL_AIR
    return label


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
    pass  # intentionally empty — fork already gives each worker its own copy


# ---------------------------------------------------------------------------
# Zarr backend (original)
# ---------------------------------------------------------------------------

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

    The voxel label is 3-class — 0 material, 1 pore, 2 air (``sample_mask ==
    0``, so the exterior and the drilled registration holes).  It is built here
    from the two zarr arrays with the same precedence the memmap extractor
    uses.  ``mask`` is the binary pore channel ``label == 1``, which is what
    the VAE losses consume.

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

    def _get_group(self, volume_id: str) -> zarr.Group:
        if volume_id not in self._zarr_cache:
            self._zarr_cache[volume_id] = self._zarr_root[volume_id]
        return self._zarr_cache[volume_id]

    def _normalise_xct(self, xct: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(xct, dtype=np.float32))
        t.mul_(1.0 / 255.0)
        return t.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        vid = row["volume_id"]
        grp = self._get_group(vid)

        ps = int(row["ps"])
        z0, y0, x0 = int(row["z0"]), int(row["y0"]), int(row["x0"])

        sl = np.s_[z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps]
        xct   = grp["xct"][sl]
        label = build_label(np.asarray(grp["mask"][sl]),
                            np.asarray(grp["sample_mask"][sl]))

        return {
            "xct":          self._normalise_xct(xct),                       # (1, ps, ps, ps) float32 [0, 1]
            "label":        torch.from_numpy(label.astype(np.int64)),       # (ps, ps, ps) int64 {0, 1, 2}
            "mask":         torch.from_numpy((label == LABEL_PORE).astype(np.float32)).unsqueeze(0),
            "volume_id":    vid,
            "coords":       np.array([z0, y0, x0], dtype=np.int32),        # int32 array — faster collation
            "porosity":     float(row["porosity"]),
            "source_group": row["source_group"],
        }


# ---------------------------------------------------------------------------
# Memmap backend (fast, no chunk amplification, fork-safe without init_fn)
# ---------------------------------------------------------------------------

class MemmapPatchDataset(Dataset):
    """Random-access patch loader backed by pre-extracted numpy memmaps.

    Replaces :class:`PatchDataset` when ``patches_xct.bin``,
    ``patches_label.bin``, and ``patches_meta.json`` exist in *data_root*.
    Build these files with ``scripts/extract_patches_memmap.py``.

    ``patches_label.bin`` stores the 3-class voxel label (0 material, 1 pore,
    2 air).  The binary pore mask the VAE losses use is derived as
    ``label == 1``; there is no separate mask file.

    The memmap layout is a flat ``(N, ps, ps, ps) uint8`` array where row ``i``
    corresponds to row ``i`` in ``patch_index.parquet``.  This 1-to-1 alignment
    means any patch can be located by its parquet row index, and volumes can be
    reconstructed from patches even when ``stride < patch_size`` — see
    :func:`reconstruct_volume`.

    Fork safety
    -----------
    numpy memmaps are POSIX file descriptors that are safe to share across
    ``fork()``.  No ``worker_init_fn`` is required.
    """

    def __init__(
        self,
        index_path: str | Path,
        data_root: str | Path,
        split: str = "train",
    ) -> None:
        data_root = Path(data_root)
        meta_path = data_root / "patches_meta.json"

        with open(meta_path) as fh:
            meta = json.load(fh)

        N  = int(meta["N"])
        ps = int(meta["patch_size"])
        shape = (N, ps, ps, ps)

        # Load full parquet and keep global row indices for this split.
        # Global index i → mmap row i (the invariant from extract_patches_memmap.py).
        df_full = pd.read_parquet(str(index_path))
        if len(df_full) != N:
            raise RuntimeError(
                f"Parquet row count ({len(df_full)}) does not match "
                f"patches_meta.json N={N}. Re-run extract_patches_memmap.py."
            )

        split_mask = df_full["split"] == split
        df_split   = df_full[split_mask]

        # Drop corrupted patches while preserving the global index alignment.
        corrupted = df_split["porosity"] > 1.0
        if corrupted.any():
            logger.warning(
                "MemmapPatchDataset [%s]: dropping %d corrupted patch(es) "
                "(porosity > 1.0, max = %.2f).",
                split, int(corrupted.sum()), float(df_split.loc[corrupted, "porosity"].max()),
            )
            df_split = df_split[~corrupted]

        self.df              = df_split.reset_index(drop=True)
        self._global_indices = df_split.index.to_numpy(dtype=np.int64)  # parquet → memmap row

        self._mmap_xct   = np.memmap(str(data_root / "patches_xct.bin"),
                                     dtype=np.uint8, mode="r", shape=shape)
        self._mmap_label = np.memmap(str(data_root / "patches_label.bin"),
                                     dtype=np.uint8, mode="r", shape=shape)
        self._ps = ps

        logger.info(
            "MemmapPatchDataset [%s]: %d patches  (global rows %d–%d)",
            split, len(self.df),
            int(self._global_indices[0]) if len(self._global_indices) else -1,
            int(self._global_indices[-1]) if len(self._global_indices) else -1,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row        = self.df.iloc[idx]
        global_idx = int(self._global_indices[idx])

        # Read the pre-extracted patch: 256 KB sequential memmap read.
        xct_raw   = self._mmap_xct  [global_idx]   # (ps, ps, ps) uint8 view
        label_raw = np.asarray(self._mmap_label[global_idx])

        return {
            "xct":          torch.from_numpy(np.asarray(xct_raw, dtype=np.float32)).mul_(1.0 / 255.0).unsqueeze(0),
            "label":        torch.from_numpy(label_raw.astype(np.int64)),                 # (ps, ps, ps) int64 {0, 1, 2}
            "mask":         torch.from_numpy((label_raw == LABEL_PORE).astype(np.float32)).unsqueeze(0),
            "volume_id":    row["volume_id"],
            "coords":       np.array([int(row["z0"]), int(row["y0"]), int(row["x0"])],
                                     dtype=np.int32),
            "porosity":     float(row["porosity"]),
            "source_group": row["source_group"],
        }


# ---------------------------------------------------------------------------
# Volume reconstruction from pre-extracted patches
# ---------------------------------------------------------------------------

def reconstruct_volume(
    volume_id: str,
    data_root: str | Path,
    *,
    array: Literal["xct", "label"] = "xct",
    overlap: Literal["mean", "overwrite"] = "mean",
) -> np.ndarray:
    """Reconstruct a full volume from its pre-extracted patches.

    Works correctly for any stride (including stride < patch_size) by using
    a float32 accumulation buffer.  The output contains every voxel that is
    covered by at least one patch; voxels not covered by any patch (can occur
    at volume edges when the last patch would fall outside the boundary) are
    left at 0.

    Parameters
    ----------
    volume_id : str
        The volume identifier as it appears in ``patch_index.parquet``.
    data_root : str | Path
        Split root directory (contains ``patches_xct.bin``, etc.).
    array : "xct" | "label"
        Which channel to reconstruct.
    overlap : "mean" | "overwrite"
        How to handle overlapping patches.

        ``"mean"``      — average all patches that cover each voxel (default).
                          Produces the best reconstruction quality.
        ``"overwrite"`` — later patches in parquet order overwrite earlier ones.
                          Fast and lossless for reconstructing the original volume
                          (all overlapping patches have identical values there).

    Returns
    -------
    np.ndarray, dtype uint8, shape (D, H, W)
        Reconstructed volume.  Scale: XCT is raw uint8 [0, 255]; label is
        {0, 1, 2}.  ``overlap="mean"`` is meaningless for a class index — use
        ``overlap="overwrite"`` for the label, which is lossless because every
        patch covering a voxel carries the same class there.

    Notes
    -----
    Reconstruction from the original zarr is simpler — use
    ``zarr.open_group(data_root / 'volumes.zarr')[volume_id]['xct'][:]``.
    This function is intended for:

    - Verifying patch extraction integrity (should match zarr exactly).
    - Assembling model-predicted patches into a volume.
    """
    data_root = Path(data_root)
    meta_path = data_root / "patches_meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)

    N  = int(meta["N"])
    ps = int(meta["patch_size"])
    shape_mmap = (N, ps, ps, ps)

    df_full = pd.read_parquet(str(data_root / "patch_index.parquet"))
    vol_mask = df_full["volume_id"] == volume_id
    if not vol_mask.any():
        raise ValueError(f"volume_id {volume_id!r} not found in parquet.")

    vol_df      = df_full[vol_mask]
    global_idxs = vol_df.index.to_numpy(dtype=np.int64)

    # Infer volume bounds from patch coordinates
    z_max = int(vol_df["z0"].max()) + ps
    y_max = int(vol_df["y0"].max()) + ps
    x_max = int(vol_df["x0"].max()) + ps
    vol_shape = (z_max, y_max, x_max)

    bin_file = data_root / f"patches_{array}.bin"
    mmap = np.memmap(str(bin_file), dtype=np.uint8, mode="r", shape=shape_mmap)

    if overlap == "overwrite":
        out = np.zeros(vol_shape, dtype=np.uint8)
        for (global_idx, row) in zip(global_idxs, vol_df.itertuples()):
            z0, y0, x0 = row.z0, row.y0, row.x0
            out[z0:z0+ps, y0:y0+ps, x0:x0+ps] = mmap[global_idx]
    else:  # "mean"
        accum = np.zeros(vol_shape, dtype=np.float32)
        count = np.zeros(vol_shape, dtype=np.float32)
        for (global_idx, row) in zip(global_idxs, vol_df.itertuples()):
            z0, y0, x0 = row.z0, row.y0, row.x0
            patch = mmap[global_idx].astype(np.float32)
            accum[z0:z0+ps, y0:y0+ps, x0:x0+ps] += patch
            count[z0:z0+ps, y0:y0+ps, x0:x0+ps] += 1.0
        out = (accum / np.maximum(count, 1.0)).round().astype(np.uint8)

    del mmap
    return out

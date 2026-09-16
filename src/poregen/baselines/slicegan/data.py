"""Real 64x64 slices for the critics, drawn from split_v3 TRAIN patches.

A BANK IN RAM, NOT A MEMMAP STREAM. The obvious implementation memory-maps
`patches_xct.bin` (564 GB) and reads a random 64-cubed patch per slice. Over a
24-hour run that walks the whole file and fills the page cache, and on this
machine host and device share one 121 GB pool: a CUDA allocation then fails
while `free` still reports tens of GB, which is the failure documented in
`docs/DEVELOPMENT.md` and the one that made the memorisation search call
`madvise` on every chunk.

So the slices are extracted ONCE into a bounded array and training reads from
RAM. 200 000 slices of 64x64, grey and label as uint8, is 1.6 GB.

That is also more data than the paper had: SliceGAN trains on the slices of a
SINGLE training volume, and this bank is drawn across every training volume in
the split.

Slices are taken along all three axes in equal measure, because the three
critics are per-axis and a bank skewed toward one axis would starve one of them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PATCH = 64
N_CLASSES = 3
DATA_ROOT = Path("data/split_v3")


@dataclass
class SliceBank:
    """``grey`` and ``label`` as (N, 64, 64) uint8, plus which axis each came from."""

    grey: np.ndarray
    label: np.ndarray
    axis: np.ndarray

    def __len__(self) -> int:
        return int(self.grey.shape[0])

    def axis_rows(self, axis: int) -> np.ndarray:
        return np.flatnonzero(self.axis == axis)


def build_slice_bank(
    data_root: Path = DATA_ROOT,
    n_slices: int = 200_000,
    seed: int = 0,
    slices_per_patch: int = 8,
    repo_root: Path | None = None,
) -> SliceBank:
    """Extract a fixed bank of real slices from the TRAIN split.

    ``slices_per_patch`` amortises the read: a patch costs 256 KB to fetch and
    yields 64 slices per axis, so taking several is free where taking one is
    not. It is not set to 64 because slices from one patch are correlated, and a
    bank of 200 000 slices from 3 000 patches would be a bank of 3 000 patches.
    """
    import pandas as pd

    root = Path(repo_root) if repo_root else Path.cwd()
    data = root / data_root if not Path(data_root).is_absolute() else Path(data_root)
    meta = json.loads((data / "patches_meta.json").read_text())
    if int(meta["patch_size"]) != PATCH:
        raise ValueError(f"expected {PATCH}-voxel patches, store says {meta['patch_size']}")
    n_rows = int(meta["N"])

    idx = pd.read_parquet(data / "patch_index.parquet", columns=["split"])
    train_rows = np.flatnonzero((idx["split"] == "train").to_numpy())
    if train_rows.size == 0:
        raise RuntimeError("no train rows in the patch index")
    logger.info("train patches available: %d", train_rows.size)

    xct = np.memmap(data / "patches_xct.bin", dtype=np.uint8, mode="r",
                    shape=(n_rows, PATCH, PATCH, PATCH))
    lab = np.memmap(data / "patches_label.bin", dtype=np.uint8, mode="r",
                    shape=(n_rows, PATCH, PATCH, PATCH))

    rng = np.random.default_rng(seed)
    per_axis = n_slices // 3
    total = per_axis * 3
    grey = np.empty((total, PATCH, PATCH), np.uint8)
    label = np.empty((total, PATCH, PATCH), np.uint8)
    axis_of = np.empty(total, np.int8)

    w = 0
    for axis in range(3):
        need = per_axis
        n_patches = int(np.ceil(need / slices_per_patch))
        rows = rng.choice(train_rows, size=n_patches, replace=n_patches > train_rows.size)
        for r in rows:
            if need <= 0:
                break
            k = min(slices_per_patch, need)
            where = rng.choice(PATCH, size=k, replace=False)
            gp = np.asarray(xct[r])
            lp = np.asarray(lab[r])
            for j in where:
                grey[w] = np.take(gp, j, axis=axis)
                label[w] = np.take(lp, j, axis=axis)
                axis_of[w] = axis
                w += 1
            need -= k
            del gp, lp
    logger.info("slice bank: %d slices (%d per axis), %.2f GB",
                w, per_axis, (grey.nbytes + label.nbytes) / 1e9)
    return SliceBank(grey[:w], label[:w], axis_of[:w])


def to_critic_batch(bank: SliceBank, rows: np.ndarray) -> np.ndarray:
    """(n, 4, 64, 64) float32: grey in [-1, 1] then the 3-class one-hot.

    Grey is mapped to [-1, 1] to match the generator's tanh, and the label to a
    one-hot that matches its softmax — so real and fake slices occupy the same
    space and the critic cannot separate them on encoding alone.
    """
    g = bank.grey[rows].astype(np.float32) / 127.5 - 1.0
    lab = bank.label[rows]
    # Shape from the DATA, not from the PATCH constant: a bank of a different
    # slice size is a legitimate thing to hold (a test builds one), and hard
    # coding 64 here turns that into a broadcast error far from its cause.
    oh = np.zeros((rows.size, N_CLASSES, *lab.shape[1:]), np.float32)
    for c in range(N_CLASSES):
        oh[:, c] = (lab == c)
    return np.concatenate([g[:, None], oh], axis=1)


def save_bank(bank: SliceBank, path: Path) -> None:
    np.savez_compressed(path, grey=bank.grey, label=bank.label, axis=bank.axis)


def load_bank(path: Path) -> SliceBank:
    with np.load(path) as z:
        return SliceBank(z["grey"], z["label"], z["axis"])

"""64-cubed training patches for the pixel-space DDPM, from the TRAIN split.

No bank in RAM here, unlike SliceGAN: the DDPM denoises whole 64-cubed VOLUMES,
so it needs the patches themselves rather than 2-D sections of them, and a bank
of volumes would be the store again. It reads the memmap directly.

That means this dataset STREAMS THE STORE, and the rule in
`docs/DEVELOPMENT.md` applies: nothing else may read it while this trains. The
dataset is constructed lazily — opening the memmap costs nothing until a worker
actually asks for a patch — so importing this module beside another training
run is safe; only iterating it is not.

Grey is mapped to [-1, 1] and the label to a 3-channel one-hot, so the four
channels are diffused together on the same scale and a sample is a PAIRED
grey+label volume rather than two things generated apart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PATCH = 64
N_CLASSES = 3


class PatchVolumes(Dataset):
    """(4, 64, 64, 64) float32 per item: grey in [-1, 1] then the one-hot label."""

    def __init__(self, data_root: Path, split: str = "train") -> None:
        self.data_root = Path(data_root)
        meta = json.loads((self.data_root / "patches_meta.json").read_text())
        if int(meta["patch_size"]) != PATCH:
            raise ValueError(f"expected {PATCH}-voxel patches, store says {meta['patch_size']}")
        self.n_rows = int(meta["N"])
        import pandas as pd
        idx = pd.read_parquet(self.data_root / "patch_index.parquet", columns=["split"])
        self.rows = np.flatnonzero((idx["split"] == split).to_numpy()).astype(np.int64)
        if self.rows.size == 0:
            raise RuntimeError(f"no rows for split {split!r}")
        logger.info("%s patches: %d", split, self.rows.size)
        # Opened per worker, after fork: a memmap shared across forks is a
        # source of silent corruption and of page-cache duplication.
        self._xct = None
        self._lab = None

    def _open(self) -> None:
        if self._xct is None:
            shape = (self.n_rows, PATCH, PATCH, PATCH)
            self._xct = np.memmap(self.data_root / "patches_xct.bin",
                                  dtype=np.uint8, mode="r", shape=shape)
            self._lab = np.memmap(self.data_root / "patches_label.bin",
                                  dtype=np.uint8, mode="r", shape=shape)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, i: int) -> torch.Tensor:
        self._open()
        r = int(self.rows[i])
        g = np.asarray(self._xct[r], np.float32) / 127.5 - 1.0
        lab = np.asarray(self._lab[r])
        oh = np.zeros((N_CLASSES, PATCH, PATCH, PATCH), np.float32)
        for c in range(N_CLASSES):
            oh[c] = (lab == c)
        return torch.from_numpy(np.concatenate([g[None], oh], axis=0))


def decode_sample(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """A generated (4, D, H, W) tensor -> (grey uint8, 3-class label).

    Grey is clamped before scaling rather than after: a diffusion sample can
    leave [-1, 1] slightly, and scaling first would wrap those voxels to the
    opposite end of the range.
    """
    g = ((x[0].clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255)
    label = x[1:].argmax(dim=0).to(torch.uint8)
    return g.to(torch.uint8).cpu().numpy(), label.cpu().numpy()

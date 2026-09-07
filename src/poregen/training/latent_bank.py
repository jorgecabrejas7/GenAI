"""A bank of LDM-sampled latents, for the refiner decoder fine-tune (D43 option 2).

Why a bank rather than sampling in the loop: one DDIM-200 chunk per training
step would make the fine-tune slower than the LDM run that produced it. The
latents are generated once by ``eval_v4 generate --save-latents`` (or
``generate_volumes.py --save-latents``) and read back here as 16-cell windows —
the shape the decoder was trained on.

What this is FOR, and what it must never become: the discriminator's *fake*
branch. The reconstruction and class losses stay on real patches, because a
generated latent carries no ground-truth label to score against. A bank used as
anything but the adversarial fake branch would be training the decoder to
reproduce the LDM's own errors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: Latent cells per decoded window — the VAE's own patch, 16^3 cells -> 64^3 voxels.
LATENT_WIN = 16


class LatentBank(Dataset):
    """Every ``LATENT_WIN``-cubed window of every saved latent canvas.

    Windows are indexed lazily against memory-mapped files: the eval-v4 canvases
    run to hundreds of MB each and the bank is read once per step, so holding
    them all resident would cost more RAM than the training batch.
    """

    def __init__(self, paths: list[str | Path], *, stride: int = LATENT_WIN) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("LatentBank was given no files.")
        self.stride = int(stride)
        self._arrays: dict[int, np.ndarray] = {}
        self.index: list[tuple[int, int, int, int]] = []
        self.channels: int | None = None

        for fi, p in enumerate(self.paths):
            if not p.exists():
                raise FileNotFoundError(f"latent canvas does not exist: {p}")
            a = np.load(p, mmap_mode="r")
            if a.ndim != 4:
                raise ValueError(f"{p}: expected (C, Z, Y, X), got shape {a.shape}")
            c, Z, Y, X = a.shape
            if self.channels is None:
                self.channels = int(c)
            elif int(c) != self.channels:
                # A bank mixing latent widths would feed the decoder tensors it
                # cannot accept, and the failure would surface as a shape error
                # thousands of steps in rather than here.
                raise ValueError(
                    f"{p} has {c} latent channels but the bank is {self.channels}. "
                    "Every canvas in a bank must come from the same VAE."
                )
            if min(Z, Y, X) < LATENT_WIN:
                logger.warning("%s: canvas %s is smaller than one window, skipped", p, (Z, Y, X))
                continue
            for z in range(0, Z - LATENT_WIN + 1, self.stride):
                for y in range(0, Y - LATENT_WIN + 1, self.stride):
                    for x in range(0, X - LATENT_WIN + 1, self.stride):
                        self.index.append((fi, z, y, x))
        if not self.index:
            raise ValueError(
                "LatentBank is empty: no canvas held a full "
                f"{LATENT_WIN}-cubed window."
            )
        logger.info("LatentBank: %d windows over %d canvases (%d channels, stride %d)",
                    len(self.index), len(self.paths), self.channels, self.stride)

    def _array(self, fi: int) -> np.ndarray:
        a = self._arrays.get(fi)
        if a is None:
            a = np.load(self.paths[fi], mmap_mode="r")
            self._arrays[fi] = a
        return a

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        fi, z, y, x = self.index[i]
        w = self._array(fi)[:, z:z + LATENT_WIN, y:y + LATENT_WIN, x:x + LATENT_WIN]
        # .copy(): the slice is a view on a memmap, and torch.from_numpy on a
        # view keeps the whole mapping alive for as long as the tensor lives.
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))


def discover_latents(root: str | Path, pattern: str = "*/*/latents.npy") -> list[Path]:
    """Saved canvases under an eval-v4 campaign, sorted for reproducibility."""
    return sorted(Path(root).glob(pattern))

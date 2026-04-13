"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed applied to all RNG sources.
    deterministic : bool
        If ``True``, forces ``cudnn.deterministic = True`` and disables
        cuDNN auto-tuning (``benchmark = False``).  This guarantees
        bit-exact reproducibility across runs at the cost of ~5-10 %
        throughput on CUDA kernels that would otherwise auto-tune.

        If ``False`` (default), cuDNN benchmark mode is **enabled**,
        allowing the runtime to select the fastest kernels for the
        current hardware and input shapes.  Results may differ at the
        last bit between runs but training is faster.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

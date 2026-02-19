"""Checkpoint save / load with metadata, scheduler, and RNG states."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    metadata: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> Path:
    """Save model + optimizer + scaler + scheduler state atomically.

    Writes to a temporary file first, then renames, so a crash mid-write
    won't corrupt the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "metadata": metadata or {},
        # RNG states for exact resumability
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Load checkpoint into *model* (and optionally *optimizer* / *scaler* / *scheduler*).

    Parameters
    ----------
    restore_rng : bool
        If True, restore Python/NumPy/CUDA RNG states for exact resumability.

    Returns
    -------
    step : int
        Training step at which the checkpoint was saved.
    metadata : dict
        Arbitrary metadata dict stored alongside the checkpoint.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if restore_rng:
        if "rng_python" in state:
            random.setstate(state["rng_python"])
        if "rng_numpy" in state:
            np.random.set_state(state["rng_numpy"])
        if "rng_torch" in state:
            torch.set_rng_state(state["rng_torch"])
        if "rng_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])

    return state.get("step", 0), state.get("metadata", {})

"""GPU selection and AMP helpers."""

from __future__ import annotations

import torch


def select_device(gpu_id: int | None = None) -> torch.device:
    """Pick a CUDA device or fall back to CPU.

    Parameters
    ----------
    gpu_id : int, optional
        Specific GPU ordinal.  ``None`` picks GPU 0 if available.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id is None:
        gpu_id = 0
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Requested GPU {gpu_id} but only {torch.cuda.device_count()} available."
        )
    return torch.device(f"cuda:{gpu_id}")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the best AMP dtype for *device*.

    - Ampere+ (sm_80+): ``bfloat16`` (no loss scaling needed).
    - Older CUDA: ``float16``.
    - CPU: ``bfloat16`` (PyTorch >= 2.0 supports CPU bfloat16 autocast).
    """
    if device.type == "cpu":
        return torch.bfloat16
    cap = torch.cuda.get_device_capability(device)
    if cap[0] >= 8:  # Ampere+
        return torch.bfloat16
    return torch.float16


def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Create a :class:`GradScaler` appropriate for *device*.

    Scaling is only meaningful for float16; bfloat16 / CPU get a
    disabled scaler so calls to ``scaler.scale()`` are no-ops.
    """
    dtype = get_autocast_dtype(device)
    enabled = dtype == torch.float16 and device.type == "cuda"
    return torch.amp.GradScaler(enabled=enabled)

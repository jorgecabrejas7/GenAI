"""GPU selection and AMP helpers."""

from __future__ import annotations

import os

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
    device = torch.device(f"cuda:{gpu_id}")
    _cap_cuda_memory(device)
    return device


# The GB10 has 121 GB of UNIFIED memory: a CUDA allocation that will not fit
# does not raise torch.cuda.OutOfMemoryError, it drains the host until the
# kernel's OOM killer takes tmux, dbus and the desktop with it (02:29 and
# 10:17, 2026-09-23). A per-process cap makes the caching allocator raise a
# clean OutOfMemoryError at the cap instead, while the host keeps the rest.
# 0.8 of 121 GB leaves ~24 GB for dataloader workers, page cache and shells.
# Override with POREGEN_CUDA_MEM_FRACTION (e.g. 0.6 for a smoke test); "1"
# or "off" disables the cap.
CUDA_MEM_FRACTION_ENV = "POREGEN_CUDA_MEM_FRACTION"
CUDA_MEM_FRACTION_DEFAULT = 0.8


def _cap_cuda_memory(device: torch.device) -> None:
    raw = os.environ.get(CUDA_MEM_FRACTION_ENV, str(CUDA_MEM_FRACTION_DEFAULT)).strip().lower()
    if raw in ("off", "1", "1.0", ""):
        return
    frac = float(raw)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"{CUDA_MEM_FRACTION_ENV}={raw!r}: expected a fraction in (0, 1), '1' or 'off'")
    torch.cuda.set_per_process_memory_fraction(frac, device)
    total = torch.cuda.get_device_properties(device).total_memory
    print(f"CUDA memory capped at {frac:.2f} of {total / 2**30:.0f} GiB = {frac * total / 2**30:.0f} GiB "
          f"({CUDA_MEM_FRACTION_ENV})", flush=True)


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

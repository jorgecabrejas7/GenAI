"""Volume I/O: discovery, TIFF loading, Zarr storage, and per-volume stats."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import zarr

logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class VolumeInfo:
    """Metadata for a discovered raw volume."""

    volume_id: str
    path: Path
    source_group: str
    shape: tuple[int, int, int] = field(default=(0, 0, 0))


def discover_volumes(raw_root: str | Path) -> list[VolumeInfo]:
    """Recursively find TIFF volumes under *raw_root*.

    Source-group logic:
      - ``MedidasDB`` → ``"MedidasDB"``
      - any other top-level directory → ``"others/<dirname>"``

    ``volume_id`` is the relative path (without extension) with path
    separators replaced by ``__`` and spaces by ``_``, making it a
    valid Zarr group name.
    """
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    volumes: list[VolumeInfo] = []

    for tif_path in sorted(raw_root.rglob("*")):
        if tif_path.suffix.lower() not in TIFF_EXTENSIONS:
            continue
        if not tif_path.is_file():
            continue

        rel = tif_path.relative_to(raw_root)
        parts = rel.parts

        # Source group from top-level directory name
        top_dir = parts[0] if len(parts) > 1 else "ungrouped"
        if top_dir == "MedidasDB":
            source_group = "MedidasDB"
        else:
            source_group = f"others/{top_dir}"

        # Stable volume_id from relative path
        vol_id = (
            str(rel.with_suffix(""))
            .replace(os.sep, "__")
            .replace(" ", "_")
        )

        volumes.append(
            VolumeInfo(
                volume_id=vol_id,
                path=tif_path,
                source_group=source_group,
            )
        )

    logger.info("Discovered %d volumes under %s", len(volumes), raw_root)
    return volumes


def load_volume(path: str | Path) -> np.ndarray:
    """Load a single multi-page TIFF as a uint8 3-D array ``(D, H, W)``."""
    vol = tifffile.imread(str(path))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {vol.shape} from {path}")
    return vol.astype(np.uint8, copy=False)


def compute_mask(xct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute pore mask and sample (foreground) mask via ``onlypores``.

    Returns
    -------
    pore_mask : uint8 array, values in {0, 1}
    sample_mask : bool array, True where material (not background/air)
    """
    try:
        from onlypores import onlypores as _onlypores
    except ImportError:
        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from onlypores import onlypores as _onlypores

    pore_mask, sample_mask, _binary = _onlypores(xct)

    if pore_mask is None:
        logger.warning("onlypores returned None — using zeros pore mask and full foreground")
        return np.zeros_like(xct, dtype=np.uint8), np.ones(xct.shape, dtype=bool)

    return pore_mask.astype(np.uint8), sample_mask.astype(bool)


def compute_volume_stats(xct: np.ndarray, sample_mask: np.ndarray) -> dict:
    """Compute foreground intensity statistics using the sample mask from ``onlypores``.

    Parameters
    ----------
    xct : uint8 volume array
    sample_mask : bool array, True where material (output of ``onlypores``)

    Returns
    -------
    dict with keys: mean, std, n_foreground
    """
    fg_vals = xct[sample_mask.astype(bool)].astype(np.float64)
    if len(fg_vals) == 0:
        logger.warning("compute_volume_stats: no foreground voxels — returning fallback stats")
        return {"mean": 128.0, "std": 50.0, "n_foreground": 0}
    return {
        "mean": float(fg_vals.mean()),
        "std": float(fg_vals.std()),
        "n_foreground": int(len(fg_vals)),
    }


def compute_volume_stats_from_zarr(zarr_xct: zarr.Array, chunk_z: int = 64) -> dict:
    """Stream through a zarr XCT array and compute foreground intensity stats.

    Uses Otsu thresholding (computed from a full histogram pass) to define
    foreground, then accumulates mean and std in a second streaming pass.
    Memory usage is O(chunk_z × H × W) — safe for multi-GB volumes.

    Parameters
    ----------
    zarr_xct : zarr.Array, shape (D, H, W), dtype uint8
    chunk_z  : number of Z-slices to read per iteration

    Returns
    -------
    dict with keys: mean, std, n_foreground, otsu_threshold
    """
    D = zarr_xct.shape[0]

    # ── Pass 1: build 256-bin histogram for Otsu threshold ──
    hist = np.zeros(256, dtype=np.int64)
    for z in range(0, D, chunk_z):
        chunk = np.array(zarr_xct[z : min(z + chunk_z, D)])
        h, _ = np.histogram(chunk.ravel(), bins=256, range=(0, 256))
        hist += h

    thresh = _otsu_from_hist(hist)
    logger.info("compute_volume_stats_from_zarr: Otsu threshold = %d", thresh)

    # ── Pass 2: streaming mean/std over foreground voxels ──
    n: int = 0
    s: float = 0.0
    sq: float = 0.0
    for z in range(0, D, chunk_z):
        chunk = np.array(zarr_xct[z : min(z + chunk_z, D)]).astype(np.float64)
        fg = chunk > thresh
        if fg.any():
            v = chunk[fg]
            n += len(v)
            s += float(v.sum())
            sq += float((v ** 2).sum())

    if n == 0:
        logger.warning("compute_volume_stats_from_zarr: no foreground voxels found")
        return {"mean": 128.0, "std": 50.0, "n_foreground": 0, "otsu_threshold": thresh}

    mean = s / n
    variance = max(sq / n - mean ** 2, 0.0)
    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "n_foreground": int(n),
        "otsu_threshold": int(thresh),
    }


def _otsu_from_hist(hist: np.ndarray) -> int:
    """Compute Otsu threshold from a 256-bin intensity histogram."""
    total = int(hist.sum())
    if total == 0:
        return 128
    p = hist.astype(np.float64) / total
    levels = np.arange(len(hist), dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(levels * p)
    mu_T = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega)),
            0.0,
        )
    return int(np.argmax(sigma_b))


# ── Stats file I/O ────────────────────────────────────────────────────────────

def load_volume_stats(out_root: str | Path) -> dict[str, dict]:
    """Load per-volume intensity statistics from ``volume_stats.json``.

    Returns an empty dict if the file does not exist yet.
    """
    path = Path(out_root) / "volume_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_volume_stats(stats: dict[str, dict], out_root: str | Path) -> None:
    """Persist per-volume intensity statistics to ``volume_stats.json``."""
    path = Path(out_root) / "volume_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved volume stats for %d volumes → %s", len(stats), path)


# ── Zarr storage ──────────────────────────────────────────────────────────────

def save_volume_zarr(
    xct: np.ndarray,
    mask: np.ndarray,
    out_root: str | Path,
    volume_id: str,
    chunk_size: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Write *xct* and *mask* into ``<out_root>/volumes.zarr/<volume_id>/``.

    Chunks are aligned to the default patch size (64³) so each patch read
    hits exactly one chunk per array.  No compression is applied — on fast
    NVMe the decompression overhead exceeds the I/O savings.
    """
    store_path = Path(out_root) / "volumes.zarr"

    root = zarr.open_group(str(store_path), mode="a")
    grp = root.require_group(volume_id)

    for name, arr in [("xct", xct), ("mask", mask)]:
        grp.create_array(
            name,
            data=arr.astype(np.uint8, copy=False),
            chunks=chunk_size,
            compressors=None,
            overwrite=True,
        )

    logger.info(
        "Saved %s  xct=%s  mask=%s  chunks=%s  compression=none",
        volume_id,
        xct.shape,
        mask.shape,
        chunk_size,
    )

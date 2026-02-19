"""Volume I/O: discovery, TIFF loading, and Zarr storage."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec

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


def compute_mask(xct: np.ndarray) -> np.ndarray:
    """Compute pore mask via the repo-level ``onlypores`` module.

    Returns uint8 array with values in {0, 1}.
    """
    try:
        from onlypores import onlypores as _onlypores
    except ImportError:
        # Fallback: add repo root (3 levels up from this file) to sys.path
        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from onlypores import onlypores as _onlypores

    pore_mask, _sample_mask, _binary = _onlypores(xct)

    if pore_mask is None:
        logger.warning("onlypores returned None — using zeros mask")
        return np.zeros_like(xct, dtype=np.uint8)
    return pore_mask.astype(np.uint8)


def save_volume_zarr(
    xct: np.ndarray,
    mask: np.ndarray,
    out_root: str | Path,
    volume_id: str,
    chunk_size: tuple[int, int, int] = (32, 32, 32),
    clevel: int = 3,
) -> None:
    """Write *xct* and *mask* into ``<out_root>/volumes.zarr/<volume_id>/``."""
    store_path = Path(out_root) / "volumes.zarr"
    compressor = BloscCodec(cname="zstd", clevel=clevel)

    root = zarr.open_group(str(store_path), mode="a")
    grp = root.require_group(volume_id)

    for name, arr in [("xct", xct), ("mask", mask)]:
        grp.create_array(
            name,
            data=arr.astype(np.uint8, copy=False),
            chunks=chunk_size,
            compressors=compressor,
            overwrite=True,
        )

    logger.info(
        "Saved %s  xct=%s  mask=%s  chunks=%s",
        volume_id,
        xct.shape,
        mask.shape,
        chunk_size,
    )

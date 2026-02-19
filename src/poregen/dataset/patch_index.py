"""Patch coordinate generation, integral-volume porosity, and Parquet index.

The 3-D summed-area table (integral volume) allows O(1) porosity queries
per patch, avoiding re-reading voxels from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch coordinates
# ---------------------------------------------------------------------------

def generate_patch_coords(
    shape: tuple[int, int, int],
    patch_size: int,
    stride: int,
) -> list[tuple[int, int, int]]:
    """Return all ``(z0, y0, x0)`` where a ``patch_size³`` cube fits inside *shape*.

    Patches that would extend beyond any boundary are **not** generated.
    """
    D, H, W = shape
    coords: list[tuple[int, int, int]] = []
    for z0 in range(0, D - patch_size + 1, stride):
        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                coords.append((z0, y0, x0))
    return coords


# ---------------------------------------------------------------------------
# 3-D Integral Volume (Summed-Area Table)
# ---------------------------------------------------------------------------

def compute_integral_volume(mask: np.ndarray) -> np.ndarray:
    """Compute a 3-D summed-area table of *mask*.

    Returns
    -------
    sat : np.ndarray, shape ``(D+1, H+1, W+1)``, dtype int64
        ``sat[i, j, k]`` equals ``mask[0:i, 0:j, 0:k].sum()`` using
        exclusive upper bounds (Python-slice semantics).
        Row/column/depth 0 is zero-padding for boundary handling.
    """
    D, H, W = mask.shape
    sat = np.zeros((D + 1, H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:, 1:] = mask.astype(np.int64)
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)
    np.cumsum(sat, axis=2, out=sat)
    return sat


def query_integral_volume(
    sat: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    patch_size: int,
) -> int:
    """Sum of the mask inside a cubic patch using inclusion–exclusion."""
    z1 = z0 + patch_size
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    return int(
        sat[z1, y1, x1]
        - sat[z0, y1, x1]
        - sat[z1, y0, x1]
        - sat[z1, y1, x0]
        + sat[z0, y0, x1]
        + sat[z0, y1, x0]
        + sat[z1, y0, x0]
        - sat[z0, y0, x0]
    )


# ---------------------------------------------------------------------------
# Vectorised patch-index builder for one volume
# ---------------------------------------------------------------------------

def build_patch_index_for_volume(
    mask: np.ndarray,
    volume_id: str,
    source_group: str,
    split: str,
    patch_size: int,
    stride: int,
) -> pd.DataFrame:
    """Build a patch-index DataFrame for a single volume while *mask* is in RAM.

    Porosity per patch is computed in O(1) via the integral volume — no
    per-patch voxel reads.
    """
    coords = generate_patch_coords(mask.shape, patch_size, stride)
    if not coords:
        logger.warning(
            "No patches fit in %s  ps=%d stride=%d shape=%s",
            volume_id,
            patch_size,
            stride,
            mask.shape,
        )
        return pd.DataFrame()

    sat = compute_integral_volume(mask)
    vol = patch_size ** 3

    # Vectorised porosity query
    ca = np.asarray(coords, dtype=np.int64)          # (N, 3)
    z0s, y0s, x0s = ca[:, 0], ca[:, 1], ca[:, 2]
    z1s = z0s + patch_size
    y1s = y0s + patch_size
    x1s = x0s + patch_size

    pore_sums = (
        sat[z1s, y1s, x1s]
        - sat[z0s, y1s, x1s]
        - sat[z1s, y0s, x1s]
        - sat[z1s, y1s, x0s]
        + sat[z0s, y0s, x1s]
        + sat[z0s, y1s, x0s]
        + sat[z1s, y0s, x0s]
        - sat[z0s, y0s, x0s]
    )
    porosities = (pore_sums / vol).astype(np.float32)

    df = pd.DataFrame(
        {
            "volume_id": volume_id,
            "source_group": source_group,
            "split": split,
            "z0": z0s,
            "y0": y0s,
            "x0": x0s,
            "ps": patch_size,
            "stride": stride,
            "porosity": porosities,
        }
    )
    logger.info(
        "Volume %s: %d patches, mean porosity=%.4f",
        volume_id,
        len(df),
        df["porosity"].mean(),
    )
    return df


def save_patch_index(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write combined patch index to a Parquet file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False, engine="pyarrow")
    logger.info("Wrote patch index (%d rows) to %s", len(df), out_path)
    return out_path

"""Material-map construction: onlypores ``sample_mask`` → latent-resolution cells.

The ldm06 data pipeline (D40 §1) stores, per patch, a MATERIAL MAP at latent
resolution plus a scalar air fraction, both derived from the volume's
``sample_mask`` (True where the specimen's material envelope is).

Storage format (chosen in D40, implemented by ``scripts/build_material_maps.py``):

- ``material.bin`` — uint8 memmap of shape ``(N, L, L, L)`` where L is the
  latent size (16 for 64³ patches at f=4).  ``value / 255`` is the material
  fraction of the corresponding ``f³``-voxel cell.  uint8 is the compact
  choice: half the bytes of float16 (~9 GB vs ~19 GB over the live store) and
  the 1/255 quantisation is far below segmentation noise.
- ``air.bin`` — float32 memmap of shape ``(N,)``: ``1 - sample_mask.mean()``
  over the full-resolution patch (computed BEFORE quantisation, so exact).

Both are row-aligned with the split's ``index.parquet`` — sibling files beside
``latents.bin`` per the D34 binary-store convention; nothing existing changes.
"""

from __future__ import annotations

import numpy as np


def pool_material_fractions(sample_mask: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean pool a {0,1}/bool mask by *factor* along every axis.

    Trailing voxels that do not fill a complete block are cropped; patch
    origins are stride-aligned, so no stored patch ever reaches into the
    cropped remainder.  Returns float32 fractions in [0, 1] with shape
    ``(D//factor, H//factor, W//factor)``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    d, h, w = (s // factor for s in sample_mask.shape)
    m = sample_mask[: d * factor, : h * factor, : w * factor]
    m = m.reshape(d, factor, h, factor, w, factor)
    return m.mean(axis=(1, 3, 5), dtype=np.float32)


def patch_material_cells(
    pooled: np.ndarray, z0: int, y0: int, x0: int, factor: int, n_cells: int
) -> np.ndarray:
    """The ``(n_cells,)³`` cell block of one patch from the pooled volume.

    Patch origins must be aligned to the pooling grid (they are: every stride
    in use is a multiple of the spatial downsampling factor).
    """
    if z0 % factor or y0 % factor or x0 % factor:
        raise ValueError(
            f"patch origin ({z0}, {y0}, {x0}) is not aligned to the pooling "
            f"factor {factor}"
        )
    cz, cy, cx = z0 // factor, y0 // factor, x0 // factor
    cells = pooled[cz : cz + n_cells, cy : cy + n_cells, cx : cx + n_cells]
    if cells.shape != (n_cells,) * 3:
        raise ValueError(
            f"patch at ({z0}, {y0}, {x0}) exceeds the pooled volume "
            f"{pooled.shape}: got cells {cells.shape}"
        )
    return cells


def encode_material_u8(cells: np.ndarray) -> np.ndarray:
    """Fraction [0, 1] → uint8 (round(f * 255))."""
    return np.rint(np.clip(cells, 0.0, 1.0) * 255.0).astype(np.uint8)


def decode_material_u8(u8: np.ndarray) -> np.ndarray:
    """uint8 → float32 fraction in [0, 1]."""
    return u8.astype(np.float32) / 255.0


def air_fraction(cells: np.ndarray) -> float:
    """``1 - mean(material fraction)`` — exact for equal-size cells."""
    return float(1.0 - cells.mean(dtype=np.float64))

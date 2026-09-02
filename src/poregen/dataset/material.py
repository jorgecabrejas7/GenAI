"""Material-map construction: the 3-class voxel label → latent-resolution cells.

The ldm06 data pipeline stores, per patch, a MATERIAL MAP at latent resolution
plus a scalar air fraction, both derived from the split_v3 voxel label
(0 material, 1 pore, 2 air).  ``scripts/build_latent_dataset.py`` writes them
in the same pass that encodes the latents, so they can never drift out of
alignment with ``index.parquet``.

**Material means label 0 only.**  Pores are not material and neither is
exterior air, so ``material + pore + air = 1`` per cell and the air fraction is
NOT ``1 - material``.  That is deliberate: the map answers "where is there
solid to build in", which is the question the denoiser is asked, while the
porosity conditioning answers "how much of it should be void".

Storage format:

- ``material.bin`` — uint8 memmap of shape ``(N, L, L, L)`` where L is the
  latent size (16 for 64³ patches at f=4).  ``value / 255`` is the material
  fraction of the corresponding ``f³``-voxel cell.  uint8 is the compact
  choice: half the bytes of float16 (~9 GB over the live store) and the 1/255
  quantisation is far below segmentation noise.
- ``air.bin`` — float32 memmap of shape ``(N,)``: the fraction of the
  full-resolution patch labelled air, computed before quantisation.

Both are row-aligned with the split's ``index.parquet``, sibling files beside
``latents.bin``.
"""

from __future__ import annotations

import numpy as np


def pool_material_fractions(is_material: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean pool the trailing three axes of a {0,1}/bool array.

    Leading axes (e.g. a batch of patches) pass through untouched.  Trailing
    voxels that do not fill a complete block are cropped; patch origins are
    stride-aligned, so no stored patch ever reaches into the cropped remainder.

    Returns float32 fractions in [0, 1] with the trailing axes divided by
    *factor*.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    a = np.asarray(is_material)
    if a.ndim < 3:
        raise ValueError(f"expected at least 3 dimensions, got {a.shape}")
    lead = a.shape[:-3]
    d, h, w = (s // factor for s in a.shape[-3:])
    m = a[..., : d * factor, : h * factor, : w * factor]
    m = m.reshape(*lead, d, factor, h, factor, w, factor)
    return m.mean(axis=(-5, -3, -1), dtype=np.float32)


def encode_material_u8(cells: np.ndarray) -> np.ndarray:
    """Fraction [0, 1] → uint8 (round(f * 255))."""
    return np.rint(np.clip(cells, 0.0, 1.0) * 255.0).astype(np.uint8)


def decode_material_u8(u8: np.ndarray) -> np.ndarray:
    """uint8 → float32 fraction in [0, 1]."""
    return u8.astype(np.float32) / 255.0

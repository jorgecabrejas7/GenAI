"""Material-map construction: the 3-class voxel label → latent-resolution cells.

The ldm06 data pipeline stores, per patch, a MATERIAL MAP at latent resolution
plus a scalar air fraction, both derived from the split_v3 voxel label
(0 material, 1 pore, 2 air).  ``scripts/build_latent_dataset.py`` writes them
in the same pass that encodes the latents, so they can never drift out of
alignment with ``index.parquet``.

**Material means the specimen ENVELOPE — ``label != 2``, pores included.**
So ``material = 1 - air`` per cell, and the map is 1 everywhere inside the
specimen, 0 outside it, and fractional only in the cells the outer surface (or
a drilled hole) passes through.  It says WHERE THE SPECIMEN IS, not how much of
it is solid.

Pooling ``label == 0`` instead would make the map ``1 - pore fraction`` at
4³-voxel cells — a copy of the pore mask at 100 µm resolution handed to the
denoiser as an input.  The model would learn to upsample it rather than to
generate pores, and the porosity conditioning would have nothing left to do.
That is the whole reason for the definition: the envelope carries information
only at the specimen surface and outside it, which is exactly the information
the model cannot otherwise have.

Storage format:

- ``material.bin`` — uint8 memmap of shape ``(N, L, L, L)`` where L is the
  latent size (16 for 64³ patches at f=4).  ``value / 255`` is the envelope
  fraction of the corresponding ``f³``-voxel cell.  uint8 is the compact
  choice: half the bytes of float16 (~9 GB over the live store) and the 1/255
  quantisation is far below segmentation noise.
- ``air.bin`` — float32 memmap of shape ``(N,)``: the fraction of the
  full-resolution patch labelled air, computed from the pooled cells before
  quantisation, so ``air == 1 - material.mean()`` exactly.

Both are row-aligned with the split's ``index.parquet``, sibling files beside
``latents.bin``.
"""

from __future__ import annotations

import numpy as np


def pool_material_fractions(is_material: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean pool the trailing three axes of a {0,1}/bool array.

    Pass ``label != CLASS_AIR`` — the specimen envelope, pores included.  See
    the module docstring for why ``label == CLASS_MATERIAL`` would be a pore
    mask in disguise.

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


def air_fraction(cells: np.ndarray) -> np.ndarray:
    """``1 - mean(envelope fraction)`` over the trailing three axes.

    Exact, because the cells are equal-sized blocks of one patch: the mean of
    the block means is the mean over voxels.  Computing it here rather than
    from the raw label keeps ``air == 1 - material.mean()`` true by
    construction, which is the identity every consumer of the pair assumes.
    """
    return (1.0 - np.asarray(cells, dtype=np.float64).mean(axis=(-3, -2, -1))
            ).astype(np.float32)

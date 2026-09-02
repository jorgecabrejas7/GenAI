"""Find the drilled registration holes of a coupon from its ``sample_mask``.

Every scanned coupon carries three ~200-voxel through-holes drilled for
registration.  They are open at both ends, so ``fill_voids`` never closed them:
they are ``False`` in ``sample_mask`` and ``0`` in the pore ``mask``.  That
combination makes them invisible to a porosity-based filter — a patch centred
in a hole reports porosity exactly 0 while being 95 % air — and they are the
only interior source of large air in the dataset.  ``split_v3`` drops every
patch that touches one.

Method
------
1. Take the middle ``z_fraction`` of the slices, away from the top and bottom
   air.
2. In-plane bounding box of the specimen = the extent of "inside the mask on
   at least one of those slices".
3. Inside that box, project ``~sample_mask`` along z with a MINIMUM: a pixel is
   marked only when EVERY one of those slices is outside the material there.
   That is the definition of a through-hole, and it is what makes the detector
   work. A maximum projection marks a pixel that is outside on any single
   slice, which on the Airbus_Panel_Pegaso coupons connects the holes to the
   exterior (the specimen shifts slightly across z) so they are then discarded
   as border-touching, and leaves ~23 one-slice internal voids behind instead.
   Measured on three volumes: maximum gives 26 / 4 / 3 components, minimum
   gives 3 / 3 / 3 at 201-204 px diameter.
4. Label the marked pixels.  Everything outside the specimen outline touches
   the box border, so DROPPING border-touching components leaves exactly the
   interior features.  Components at or below ``min_area`` pixels are noise.
5. Dilate what remains by ``dilate_vox`` voxels (exact Euclidean), so a patch
   merely near a hole is dropped too.

The result is a 2-D mask in the volume's full ``(H, W)`` frame: the holes go
through the specimen, so their footprint does not depend on z.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

Z_FRACTION = 0.6
MIN_AREA_PX = 400
DILATE_VOX = 32
EXPECTED_HOLES = 3


def _projections(sample_mask, z_lo: int, z_hi: int, z_chunk: int = 32):
    """(inside_any, outside_every) over ``[z_lo, z_hi)``.

    ``inside_any`` bounds the specimen in-plane; ``outside_every`` marks the
    pixels that no slice covers — the through-holes and the exterior.
    """
    _, H, W = sample_mask.shape
    inside = np.zeros((H, W), dtype=bool)
    outside = np.ones((H, W), dtype=bool)
    for z0 in range(z_lo, z_hi, z_chunk):
        z1 = min(z_hi, z0 + z_chunk)
        blk = np.asarray(sample_mask[z0:z1]) > 0
        inside |= blk.any(axis=0)
        outside &= (~blk).all(axis=0)
    return inside, outside


def detect_holes(sample_mask, *, z_fraction: float = Z_FRACTION,
                 min_area_px: int = MIN_AREA_PX,
                 dilate_vox: int = DILATE_VOX,
                 volume_id: str = "") -> dict:
    """Locate the drilled holes of one volume.

    Parameters
    ----------
    sample_mask : array-like (D, H, W)
        The material envelope; zarr arrays are read in z chunks.
    z_fraction : float
        Central fraction of slices used for the projection.
    min_area_px : int
        Components at or below this many pixels are discarded as noise.
    dilate_vox : int
        Euclidean dilation applied to the kept components.
    volume_id : str
        Only used in the log line.

    Returns
    -------
    dict with keys ``mask`` (bool (H, W), dilated), ``holes`` (one record per
    kept component: ``centre_yx``, ``area_px``, ``equiv_diameter_px``),
    ``n_holes``, ``bbox`` (``[y_lo, y_hi, x_lo, x_hi]``, half-open),
    ``dilated_fraction`` and ``z_range``.
    """
    D, H, W = sample_mask.shape
    margin = int(round(D * (1.0 - z_fraction) / 2.0))
    z_lo, z_hi = margin, max(margin + 1, D - margin)

    inside, outside = _projections(sample_mask, z_lo, z_hi)
    if not inside.any():
        raise ValueError(f"{volume_id or 'volume'}: sample_mask is empty over "
                         f"z[{z_lo}, {z_hi})")
    ys = np.flatnonzero(inside.any(axis=1))
    xs = np.flatnonzero(inside.any(axis=0))
    y_lo, y_hi = int(ys[0]), int(ys[-1]) + 1
    x_lo, x_hi = int(xs[0]), int(xs[-1]) + 1

    sub = outside[y_lo:y_hi, x_lo:x_hi]
    lab, n = ndi.label(sub)

    # A component that reaches the bounding box border is the exterior of the
    # specimen, not a hole.
    border = set(np.unique(np.concatenate([
        lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    areas = np.bincount(lab.ravel(), minlength=n + 1)

    keep, holes = [], []
    for i in range(1, n + 1):
        if i in border or areas[i] <= min_area_px:
            continue
        keep.append(i)
        cy, cx = ndi.center_of_mass(lab == i)
        holes.append({
            "centre_yx": [float(cy) + y_lo, float(cx) + x_lo],
            "area_px": int(areas[i]),
            "equiv_diameter_px": float(2.0 * np.sqrt(areas[i] / np.pi)),
        })
    holes.sort(key=lambda h: -h["area_px"])

    core = np.zeros((H, W), dtype=bool)
    if keep:
        core[y_lo:y_hi, x_lo:x_hi] = np.isin(lab, keep)
        # Exact Euclidean dilation — a disk of radius dilate_vox.
        mask = ndi.distance_transform_edt(~core) <= dilate_vox
    else:
        mask = core

    logger.info("%s: %d hole(s), diameters %s px, dilated footprint %.4f of "
                "the slice", volume_id or "volume", len(holes),
                [round(h["equiv_diameter_px"]) for h in holes],
                float(mask.mean()))
    return {
        "mask": mask,
        "holes": holes,
        "n_holes": len(holes),
        "bbox": [y_lo, y_hi, x_lo, x_hi],
        "dilated_fraction": float(mask.mean()),
        "z_range": [z_lo, z_hi],
        "params": {"z_fraction": z_fraction, "min_area_px": min_area_px,
                   "dilate_vox": dilate_vox},
    }


def patches_touching_holes(hole_mask: np.ndarray, y0: np.ndarray,
                           x0: np.ndarray, patch_size: int) -> np.ndarray:
    """Bool per patch — does its (y, x) footprint intersect ``hole_mask``?

    A 2-D summed-area table gives every patch in one vectorised query.
    """
    ii = np.zeros((hole_mask.shape[0] + 1, hole_mask.shape[1] + 1),
                  dtype=np.int64)
    ii[1:, 1:] = hole_mask.astype(np.int64)
    np.cumsum(ii, axis=0, out=ii)
    np.cumsum(ii, axis=1, out=ii)
    y1, x1 = y0 + patch_size, x0 + patch_size
    counts = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
    return counts > 0

"""Pick axis-aligned regions of a real volume that are fully inside the material.

Both campaign-08 diagnostics need the same thing: a 64-aligned box of a real
scan that contains only material — no exterior air, and none of the three
~200-voxel drilled registration holes.  ``sample_mask`` already encodes both:
it is False outside the specimen AND inside the holes (they are open at both
ends, so ``fill_voids`` never closed them).  So "fully inside ``sample_mask``"
is the whole selection rule; no separate hole geometry is needed.

The search runs on a small summary array built by one pass over
``sample_mask``: ``ok_z[z, iy, ix]`` is True when every voxel of the 64x64 cell
``(iy, ix)`` of slice ``z`` is inside the mask.  For a real scan that is a few
hundred thousand booleans, so any box can then be tested for free.

z origins are NOT forced onto the 64 grid.  A laminate is only ~200 voxels
deep, so a 192-deep box has ~10 voxels of freedom and a multiple of 64 rarely
lands inside it; the y/x origins always are 64-aligned.  Seam planes are
defined relative to the extracted box, so the z phase does not affect them.
"""

from __future__ import annotations

import numpy as np

CELL = 64


def cell_ok_by_slice(sm_arr, cell: int = CELL, z_chunk: int = 32) -> np.ndarray:
    """(D, ny, nx) bool — cell (iy, ix) of slice z is entirely inside sample_mask."""
    D, H, W = sm_arr.shape
    ny, nx = H // cell, W // cell
    ok = np.zeros((D, ny, nx), dtype=bool)
    for z0 in range(0, D, z_chunk):
        z1 = min(D, z0 + z_chunk)
        blk = np.asarray(sm_arr[z0:z1, :ny * cell, :nx * cell]) > 0
        ok[z0:z1] = blk.reshape(z1 - z0, ny, cell, nx, cell).all(axis=(2, 4))
    return ok


def find_window(ok: np.ndarray, ky: int, kx: int) -> tuple[int, int] | None:
    """Top-left cell index of the all-usable (ky, kx) window nearest the centre."""
    ny, nx = ok.shape
    if ky > ny or kx > nx:
        return None
    ii = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(ok.astype(np.int64), axis=0), axis=1)
    counts = (ii[ky:, kx:] - ii[:-ky or None, kx:]
              - ii[ky:, :-kx or None] + ii[:-ky or None, :-kx or None])
    cand = np.argwhere(counts == ky * kx)
    if not len(cand):
        return None
    centre = np.array([(ny - ky) / 2.0, (nx - kx) / 2.0])
    iy, ix = cand[int(np.argmin(((cand - centre) ** 2).sum(axis=1)))]
    return int(iy), int(ix)


def find_window_best(ok: np.ndarray, ky: int, kx: int) -> tuple[int, int, float]:
    """(iy, ix, usable_fraction) of the (ky, kx) window with the most usable
    cells; ties broken by distance to the in-plane centre."""
    ny, nx = ok.shape
    if ky > ny or kx > nx:
        raise ValueError(f"window {ky}x{kx} cells does not fit in {ok.shape}")
    ii = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(ok.astype(np.int64), axis=0), axis=1)
    counts = (ii[ky:, kx:] - ii[:-ky or None, kx:]
              - ii[ky:, :-kx or None] + ii[:-ky or None, :-kx or None])
    best = int(counts.max())
    cand = np.argwhere(counts == best)
    centre = np.array([(ny - ky) / 2.0, (nx - kx) / 2.0])
    iy, ix = cand[int(np.argmin(((cand - centre) ** 2).sum(axis=1)))]
    return int(iy), int(ix), best / float(ky * kx)


def _z_order(z_lo: int, z_hi: int, depth: int, cell: int) -> list[int]:
    """z origins to try: 64-aligned first, then every other one, both nearest
    the middle of the allowed range first."""
    lo, hi = int(z_lo), int(z_hi) - depth
    if hi < lo:
        return []
    mid = (lo + hi) / 2.0
    allz = list(range(lo, hi + 1))
    aligned = sorted((z for z in allz if z % cell == 0), key=lambda z: abs(z - mid))
    rest = sorted((z for z in allz if z % cell != 0), key=lambda z: abs(z - mid))
    return aligned + rest


def find_region(sm_arr, depth: int, ny_cells: int, nx_cells: int,
                z_lo: int | None = None, z_hi: int | None = None,
                cell: int = CELL, ok_z: np.ndarray | None = None) -> dict | None:
    """Find a (depth, ny_cells*cell, nx_cells*cell) box entirely inside sample_mask.

    ``z_lo``/``z_hi`` bound the z origin search (default: the whole array).
    Pass ``ok_z`` to reuse a summary array across several searches.

    Returns ``{"z0", "y0", "x0", "z_aligned", "usable_cell_fraction"}`` or None.
    """
    if ok_z is None:
        ok_z = cell_ok_by_slice(sm_arr, cell)
    D = ok_z.shape[0]
    z_lo = 0 if z_lo is None else max(0, int(z_lo))
    z_hi = D if z_hi is None else min(D, int(z_hi))
    for z0 in _z_order(z_lo, z_hi, depth, cell):
        ok = ok_z[z0:z0 + depth].all(axis=0)
        hit = find_window(ok, ny_cells, nx_cells)
        if hit is not None:
            iy, ix = hit
            return {"z0": int(z0), "y0": int(iy * cell), "x0": int(ix * cell),
                    "z_aligned": bool(z0 % cell == 0),
                    "usable_cell_fraction": float(ok.mean())}
    return None

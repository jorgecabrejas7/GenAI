"""Every measurement the suite makes, and the manifest fields each one needs.

A metric is a plain function over numpy arrays that carries a
:func:`~poregen.eval_v4.manifest.requires` declaration.  The declaration is
enforced before the body runs, so a metric cannot be applied to a volume whose
manifest does not say what was asked of it.

The calling convention the decorator depends on:

* **positional** array arguments are volume-shaped and are checked against
  ``manifest.volume_shape``;
* auxiliary arrays that are NOT volume-shaped (a tile-grid field, a second
  volume) go in by keyword.

Definitions shared across metrics:

``material``
    the requested specimen envelope at voxel resolution
    (:meth:`poregen.eval_v4.io.Case.material_voxels`).  Every fraction is taken
    inside it, because a volume that was asked for a hole should not be scored
    for the pores it does not put in the hole.
``interior``
    more than :data:`EDGE_VOX` voxels from every face of the volume.  Air near
    a face may be the specimen surface the model was asked for; air in the
    interior cannot be.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poregen.diffusion.sampler import AXIS_NAMES, seam_discontinuity
from poregen.eval_v4.io import (
    LABEL_AIR,
    LABEL_MATERIAL,
    LABEL_PORE,
    TILE,
    repo_root,
)
from poregen.eval_v4.manifest import Manifest, ManifestError, requires

#: Shell excluded from the interior, in voxels.  The same 32 as
#: ``scripts/analysis/air_audit_v2.EDGE_VOX``, so v3 and v4 interior numbers
#: mean the same thing; :func:`grey_air_detector` asserts they still agree.
EDGE_VOX = 32

#: |delivered - requested| gate on porosity (decision D39, kept from v3).
POROSITY_GATE = 0.005

#: A volume is a generation FAILURE at these values - not merely inaccurate.
FAIL_PHI_LOW = 1e-4
FAIL_PHI_HIGH = 0.5
FAIL_AIR_IN_MATERIAL = 0.2

#: 4-class angle grid: a ply is 0, 45, 90 or 135 degrees.
CLASS_STEP_DEG = 45.0


# ---------------------------------------------------------------------------
# Small shared statistics
# ---------------------------------------------------------------------------

def fit_ols(x, y) -> dict:
    """Ordinary least squares of ``y`` on ``x`` with the coefficient of
    determination.  ``None`` everywhere when the fit is not defined."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return {"slope": None, "intercept": None, "r2": None, "n": int(len(x))}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 0 else None,
        "n": int(len(x)),
    }


def err_stats(err, gate: float = POROSITY_GATE) -> dict:
    """Absolute-error summary plus the fraction inside ``gate``."""
    a = np.abs(np.asarray(err, float).ravel())
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {
        "abs_error_mean": float(a.mean()),
        "abs_error_median": float(np.median(a)),
        "abs_error_p95": float(np.percentile(a, 95)),
        "abs_error_max": float(a.max()),
        "frac_within_gate": float((a < gate).mean()),
        "gate": float(gate),
        "n": int(a.size),
    }


def mean_sd(values) -> dict:
    """Mean and sample standard deviation over the seeds of one cell."""
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }


def wrap180(d) -> np.ndarray:
    """Signed axial angle difference in [-90, 90)."""
    return (np.asarray(d, float) + 90.0) % 180.0 - 90.0


def angle_class(deg) -> np.ndarray:
    """Nearest of 0 / 45 / 90 / 135 degrees, as an index 0..3."""
    return np.round((np.asarray(deg, float) % 180.0) / CLASS_STEP_DEG).astype(int) % 4


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def interior_mask(shape: tuple[int, int, int], margin: int = EDGE_VOX) -> np.ndarray:
    """Voxels more than ``margin`` from every face - the block where no face
    effect and no specimen surface can reach."""
    if any(s <= 2 * margin for s in shape):
        raise ValueError(
            f"a {shape} volume has no interior at margin {margin}; the assessment "
            "must use a volume more than twice the margin on every axis."
        )
    m = np.zeros(shape, dtype=bool)
    m[margin:-margin, margin:-margin, margin:-margin] = True
    return m


def tile_grid(shape: tuple[int, int, int], tile: int = TILE) -> tuple[int, int, int]:
    return tuple(s // tile for s in shape)  # type: ignore[return-value]


def block_sum(a: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Sum of ``a`` over every ``tile``-cubed cell; returns the tile grid."""
    gz, gy, gx = tile_grid(a.shape, tile)
    view = a[: gz * tile, : gy * tile, : gx * tile]
    return view.reshape(gz, tile, gy, tile, gx, tile).sum(axis=(1, 3, 5), dtype=np.float64)


def chunk_period(manifest: Manifest) -> tuple[int, int, int]:
    """Seam period of the chunk planes, in voxels, per axis."""
    manifest.require(("chunk_tiles",), "chunk_period")
    return tuple(TILE * int(c) for c in manifest.chunk_tiles)  # type: ignore[return-value]


def crop_region(arr: np.ndarray, manifest: Manifest) -> np.ndarray:
    """The sub-block of ``arr`` the case is about (the whole volume by default)."""
    off, shape = manifest.region
    sl = tuple(slice(off[a], off[a] + shape[a]) for a in range(3))
    return arr[(..., *sl)] if arr.ndim == 4 else arr[sl]


# ---------------------------------------------------------------------------
# 1 - porosity and air
# ---------------------------------------------------------------------------

@requires()
def phase_fractions(label: np.ndarray, material: np.ndarray, *, manifest: Manifest) -> dict:
    """Pore and air fraction, over the whole volume and inside the request.

    ``phi_pore`` is MATERIAL porosity: pore / material, air excluded from the
    denominator.  ``phi_pore_all`` is pore / whole volume, air included, and is
    reported beside it so the two can never be read for each other.  The LDM is
    conditioned on the second definition (the store's phi is pore / 64**3); the
    sampler converts a material-porosity request into it per window.

    Needs no request of its own, so a real volume goes through it unchanged -
    this is the row every table is floored against.
    """
    mat_n = float(material.sum())
    if mat_n == 0:
        raise ValueError("the requested material is empty; nothing to measure.")
    inside = material
    inter = interior_mask(label.shape) & inside
    return {
        "phi_pore": float((label[inside] == LABEL_PORE).mean()),
        "phi_pore_all": float((label == LABEL_PORE).mean()),
        "air_fraction": float((label[inside] == LABEL_AIR).mean()),
        "air_fraction_all": float((label == LABEL_AIR).mean()),
        "air_fraction_interior": float((label[inter] == LABEL_AIR).mean()),
        "material_voxel_fraction": mat_n / float(label.size),
        "interior_voxels": int(inter.sum()),
    }


@requires("requested_global_phi")
def porosity_error(label: np.ndarray, material: np.ndarray, *, manifest: Manifest) -> dict:
    """Delivered minus requested global porosity, BOTH as pore / material.

    The request is a material porosity and it is scored as one: air outside the
    specimen envelope is in neither numerator nor denominator.
    """
    delivered = float((label[material] == LABEL_PORE).mean())
    requested = float(manifest.requested_global_phi)
    err = delivered - requested
    return {
        "requested_phi": requested,
        "delivered_phi": delivered,
        "error": err,
        "abs_error": abs(err),
        "within_gate": bool(abs(err) < POROSITY_GATE),
        "gate": POROSITY_GATE,
    }


@requires()
def degenerate_cells(label: np.ndarray, material: np.ndarray, *, manifest: Manifest) -> dict:
    """Fraction of 64-voxel tiles whose delivered porosity is degenerate.

    Degenerate uses the same two numbers as the volume-level failure rule, so a
    volume that fails and a volume where a tenth of the tiles fail are described
    on one scale.
    """
    pore = block_sum((label == LABEL_PORE) & material)
    mat = block_sum(material)
    ok = mat > 0.5 * TILE ** 3
    if not ok.any():
        return {"degenerate_cell_fraction": None, "n_cells": 0}
    phi = np.full(pore.shape, np.nan)
    phi[ok] = pore[ok] / mat[ok]
    bad = (phi[ok] < FAIL_PHI_LOW) | (phi[ok] > FAIL_PHI_HIGH)
    return {
        "degenerate_cell_fraction": float(bad.mean()),
        "n_cells": int(ok.sum()),
        "cell_phi_min": float(np.nanmin(phi)),
        "cell_phi_max": float(np.nanmax(phi)),
    }


@requires()
def failure_flags(label: np.ndarray, material: np.ndarray, *, manifest: Manifest) -> dict:
    """Did this generation fail outright?

    Three ways, all from the task's definition: porosity below 1e-4, porosity
    above 0.5, or more than 20 % air inside the material the caller asked for.
    """
    phi = float((label[material] == LABEL_PORE).mean())
    air = float((label[material] == LABEL_AIR).mean())
    flags = {
        "phi_collapsed": bool(phi < FAIL_PHI_LOW),
        "phi_saturated": bool(phi > FAIL_PHI_HIGH),
        "air_in_material": bool(air > FAIL_AIR_IN_MATERIAL),
    }
    flags["failed"] = bool(any(flags.values()))
    flags["phi"] = phi
    flags["air"] = air
    return flags


# ---------------------------------------------------------------------------
# 2 - local (within-volume) porosity obedience
# ---------------------------------------------------------------------------

@requires()
def local_obedience(
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
    requested_tiles: np.ndarray | None = None,
    min_material_frac: float = 0.5,
) -> dict:
    """Does the volume put the pores where the field asked for them?

    The score is the WITHIN-volume regression: each volume's own mean is
    subtracted from both the delivered and the requested cell values before the
    fit.  Pooling cells from volumes at different global targets and quoting the
    resulting R-squared is what made eval v3 look obedient - most of that
    variance is the global dose response, which assessment 2 already measures.
    The pooled fit is still reported, under a name that says it is not
    obedience.

    ``requested_tiles`` is the requested phi per 64-voxel tile.  Pass ``None``
    for a real volume: there is no request, so only the cell-to-cell spread is
    returned, and that spread is the noise floor the slope must beat.

    Both sides are MATERIAL porosity, pore / material: ``delivered[c]`` divides
    a tile's pore count by its material count, not by 64**3, and the requested
    field is a material porosity too.  A tile holding less than
    ``min_material_frac`` material is dropped rather than measured against a
    denominator that is mostly air.
    """
    pore = block_sum((label == LABEL_PORE) & material)
    mat = block_sum(material)
    usable = mat > min_material_frac * TILE ** 3
    if usable.sum() < 3:
        raise ValueError(
            f"only {int(usable.sum())} tiles hold more than "
            f"{min_material_frac:.0%} material; the within-volume fit needs 3."
        )
    delivered = np.full(mat.shape, np.nan)
    delivered[usable] = pore[usable] / mat[usable]

    d = delivered[usable]
    out = {
        "n_cells": int(usable.sum()),
        "n_cells_total": int(mat.size),
        "delivered_cell_mean": float(d.mean()),
        "delivered_cell_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "delivered_cell_min": float(d.min()),
        "delivered_cell_max": float(d.max()),
    }
    if requested_tiles is None:
        out["requested"] = False
        return out

    req = np.asarray(requested_tiles, float)
    if req.shape != mat.shape:
        raise ManifestError(
            f"local_obedience: requested field has shape {req.shape}, expected the "
            f"tile grid {mat.shape} of volume_shape {tuple(manifest.volume_shape)}."
        )
    r = req[usable]
    fit = fit_ols(r - r.mean(), d - d.mean())
    out.update({
        "requested": True,
        "requested_cell_mean": float(r.mean()),
        "requested_cell_sd": float(r.std(ddof=1)) if r.size > 1 else 0.0,
        "within_volume_slope": fit["slope"],
        "within_volume_r2": fit["r2"],
        "within_volume_intercept": fit["intercept"],
        "per_cell": err_stats(d - r),
        "cells_requested": r.tolist(),
        "cells_delivered": d.tolist(),
    })
    return out


def pooled_dose_fit(cells_requested, cells_delivered) -> dict:
    """The fit across ALL cells of ALL volumes.

    Named for what it is.  It mixes the global dose response into the local
    question, so it is reported as context and never as local obedience.
    """
    fit = fit_ols(np.concatenate(cells_requested), np.concatenate(cells_delivered))
    return {f"pooled_{k}_not_obedience": v for k, v in fit.items()}


# ---------------------------------------------------------------------------
# 3 - assembly seams
# ---------------------------------------------------------------------------

@requires("chunk_tiles", "window_stride")
def seam_metrics(
    xct_u8: np.ndarray,
    *,
    manifest: Manifest,
    pore_logit: np.ndarray | None = None,
) -> dict:
    """Discontinuity at the window planes (period 64) and the chunk planes.

    Both are judged against the SAME interior baseline - the interior excludes
    the 64-voxel window planes in both cases - so the two ratios are
    comparable.  A ratio near 1 means the plane is indistinguishable from
    ordinary internal texture change.

    The grey channel is always available.  The pore log-odds is the decoder's
    own continuous segmentation output; it is measured only when the case
    stored ``probs.npz``, and its absence is reported rather than filled in
    from the argmax, which is not a continuous field.
    """
    grey = xct_u8.astype(np.float32) / 255.0
    period = chunk_period(manifest)
    out: dict = {
        "window_period": TILE,
        "chunk_period": list(period),
        **seam_discontinuity(grey, TILE, prefix="seam_xct"),
        **seam_discontinuity(grey, period, prefix="seam_chunk_xct", interior_exclude=TILE),
    }
    if pore_logit is None:
        out["pore_logit_available"] = False
        return out
    manifest.check_array(pore_logit, "seam_metrics", "pore_logit")
    pl = np.asarray(pore_logit, np.float32)
    out["pore_logit_available"] = True
    out.update(seam_discontinuity(pl, TILE, prefix="seam_pore"))
    out.update(
        seam_discontinuity(pl, period, prefix="seam_chunk_pore", interior_exclude=TILE)
    )
    return out


def pore_dice(a: np.ndarray, b: np.ndarray, where: np.ndarray | None = None) -> float:
    """Dice of the pore class between two volumes of the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"pore_dice needs matching shapes, got {a.shape} and {b.shape}.")
    pa = a == LABEL_PORE
    pb = b == LABEL_PORE
    if where is not None:
        pa = pa & where
        pb = pb & where
    n = float(pa.sum()) + float(pb.sum())
    if n == 0:
        return float("nan")
    return 2.0 * float((pa & pb).sum()) / n


def chunk_plane_slab(shape: tuple[int, int, int], period, half_width: int = TILE // 2) -> np.ndarray:
    """Voxels within ``half_width`` of any chunk plane.

    The chunk plane is where two independently denoised canvases meet, so it is
    the only place a neighbour-conditioning change can first show.  Restricting
    the comparison to a slab around it keeps the bulk of the volume - which is
    the same by construction - out of the number.
    """
    periods = (int(period),) * 3 if np.isscalar(period) else tuple(int(p) for p in period)
    m = np.zeros(shape, dtype=bool)
    for axis in range(3):
        n, p = shape[axis], periods[axis]
        idx = np.arange(n)
        near = np.zeros(n, bool)
        for plane in range(p, n, p):
            near |= (idx >= plane - half_width) & (idx < plane + half_width)
        shaped = near.reshape([-1 if a == axis else 1 for a in range(3)])
        m |= np.broadcast_to(shaped, shape)
    return m


# ---------------------------------------------------------------------------
# 3b - the same seams, resolved PER CHUNK along the generation order
# ---------------------------------------------------------------------------
#
# ``seam_metrics`` answers "is there a seam in this volume".  These answer
# "which chunk's seam, and does it get worse the further the chunk is from the
# first one" — the question a chunked sampler's failure mode is actually
# shaped like, because chunk k assembles against material chunk k-1 already
# produced, and an error that compounds shows up as a trend in k and not as a
# worse volume average.
#
# The grid is a PARAMETER, never the case's own ``chunk_tiles``.  Four arms
# with four chunk geometries have to be read at the same planes or the table
# compares four different measurements.

def chunk_blocks(shape: tuple[int, int, int], period) -> list[dict]:
    """The reference chunk grid, in RASTER (generation) order.

    ``period`` is the chunk size in voxels, per axis or one value for all three.
    The last block on an axis is short when the volume is not a whole number of
    chunks, exactly as ``VolumeGenerator._chunk_ranges`` cuts it.

    Each block carries the axes on which it has a LOWER chunk plane: the plane
    at its own origin, where it met material that already existed.  Attributing
    a plane to the block ABOVE it is what makes "seam of chunk k" well defined —
    every plane has exactly one owner, and chunk 0 owns none.
    """
    shape = tuple(int(s) for s in shape)
    per = (int(period),) * 3 if np.isscalar(period) else tuple(int(p) for p in period)
    if any(p <= 0 for p in per):
        raise ValueError(f"chunk period must be positive on every axis, got {per}.")
    starts = [list(range(0, shape[a], per[a])) for a in range(3)]
    out = []
    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                origin = (z0, y0, x0)
                out.append({
                    "chunk_index": len(out),
                    "origin": list(origin),
                    "shape": [min(per[a], shape[a] - origin[a]) for a in range(3)],
                    "lower_plane_axes": [a for a in range(3) if origin[a] > 0],
                })
    return out


def _plane_mad(vol: np.ndarray, axis: int, boundary: int, foot) -> float:
    """Mean |slice-to-slice difference| at one plane, over one block's footprint.

    ``boundary`` indexes the plane BETWEEN voxel ``boundary - 1`` and
    ``boundary`` — the same convention ``seam_discontinuity`` uses.
    """
    sl_lo = list(foot)
    sl_hi = list(foot)
    sl_lo[axis] = slice(boundary - 1, boundary)
    sl_hi[axis] = slice(boundary, boundary + 1)
    return float(np.abs(vol[tuple(sl_hi)].astype(np.float32)
                        - vol[tuple(sl_lo)].astype(np.float32)).mean())


def chunk_seam_profile(vol: np.ndarray, block: dict, tile: int = TILE) -> dict:
    """Seams of ONE chunk, split into the chunk family and the tile family.

    Three plane families inside and on the lower face of a block:

    ``chunk``    the block's own lower-face planes — where two independently
                 denoised canvases meet.  Absent for chunk 0.
    ``tile``     planes strictly inside the block at multiples of ``tile`` —
                 where two overlapping WINDOWS of the same solve meet.
    ``interior`` every other plane strictly inside the block.  This is the
                 baseline both ratios are divided by, so the two families are
                 judged against the same natural slice-to-slice variation —
                 the convention ``seam_metrics`` already applies volume-wide.

    Each axis contributes its planes weighted by the number of voxels behind
    them, so an anisotropic block does not over-weight its thin axis.
    """
    origin = tuple(int(v) for v in block["origin"])
    extent = tuple(int(v) for v in block["shape"])
    foot = tuple(slice(origin[a], origin[a] + extent[a]) for a in range(3))
    fam: dict[str, list[tuple[float, float]]] = {"chunk": [], "tile": [], "interior": []}
    per_axis: dict[str, dict] = {}

    inner = np.asarray(vol[foot], np.float32)
    for axis in range(3):
        elems = float(extent[(axis + 1) % 3] * extent[(axis + 2) % 3])
        axis_fam: dict[str, list[float]] = {"chunk": [], "tile": [], "interior": []}
        if axis in block["lower_plane_axes"]:
            # The lower plane lies BETWEEN this block and the previous one, so
            # it is the one plane that cannot come from a diff of the block.
            axis_fam["chunk"].append(_plane_mad(vol, axis, origin[axis], foot))
        others = tuple(i for i in range(3) if i != axis)
        per_plane = np.abs(np.diff(inner, axis=axis)).mean(axis=others)
        boundary = np.arange(origin[axis] + 1, origin[axis] + extent[axis])
        is_tile = (boundary % int(tile)) == 0
        axis_fam["tile"] = [float(v) for v in per_plane[is_tile]]
        axis_fam["interior"] = [float(v) for v in per_plane[~is_tile]]
        per_axis[AXIS_NAMES[axis]] = {
            f"{k}_mad": (float(np.mean(v)) if v else None) for k, v in axis_fam.items()
        }
        per_axis[AXIS_NAMES[axis]]["n_chunk_planes"] = len(axis_fam["chunk"])
        for k, v in axis_fam.items():
            fam[k] += [(m, elems) for m in v]

    def agg(vals) -> float | None:
        if not vals:
            return None
        w = sum(e for _, e in vals)
        return float(sum(m * e for m, e in vals) / w) if w else None

    mad = {k: agg(v) for k, v in fam.items()}
    base = mad["interior"]

    def ratio(key: str) -> float | None:
        if mad[key] is None or base is None or base <= 1e-12:
            return None
        return float(mad[key] / base)

    return {
        "chunk_plane_mad": mad["chunk"],
        "tile_plane_mad": mad["tile"],
        "interior_mad": base,
        "chunk_plane_ratio": ratio("chunk"),
        "tile_plane_ratio": ratio("tile"),
        "n_chunk_planes": len(fam["chunk"]),
        "n_tile_planes": len(fam["tile"]),
        "n_interior_planes": len(fam["interior"]),
        "per_axis": per_axis,
    }


def chunk_porosity(label: np.ndarray, material: np.ndarray, block: dict) -> dict:
    """Material porosity and air fraction inside one chunk block."""
    sl = tuple(slice(block["origin"][a], block["origin"][a] + block["shape"][a])
               for a in range(3))
    lab, mat = label[sl], material[sl]
    n_mat = float(mat.sum())
    return {
        "phi_pore": float(((lab == LABEL_PORE) & mat).sum() / n_mat) if n_mat else None,
        "air_fraction": float((lab == LABEL_AIR).mean()),
        "material_fraction": float(mat.mean()),
    }


def _centred_window(centre: int, side: int, limit: int) -> tuple[int, int] | None:
    """``(lo, hi)`` of a ``side``-long window centred at ``centre`` inside
    ``[0, limit)``, or ``None`` when it does not fit."""
    if side > limit:
        return None
    lo = int(np.clip(centre - side // 2, 0, limit - side))
    return lo, lo + side


def chunk_s2(
    label: np.ndarray,
    material: np.ndarray,
    block: dict,
    *,
    window: int,
    min_material: float,
) -> dict:
    """S2 inside one chunk against S2 straddling each of its chunk planes.

    The inside window is centred on the block; each across window is centred on
    one of the block's lower-face planes, so half of it is material the previous
    chunk produced and half is this chunk's.  If the join is sound the two
    curves agree; if the structure stops at the plane, the across curve loses
    correlation at exactly the lag that reaches over it.

    ``s2_relative_distance`` is ``mean|across - inside| / mean(inside)``, which
    is 0 for identical curves and is scale-free, so chunks at different porosity
    are still comparable.  ``None`` wherever a window does not fit or is not at
    least ``min_material`` requested specimen — a window half outside the
    specimen would measure the envelope and not the join.
    """
    from poregen.eval_v4.microstructure import s2_radial  # noqa: PLC0415

    origin = tuple(int(v) for v in block["origin"])
    extent = tuple(int(v) for v in block["shape"])
    centre = tuple(origin[a] + extent[a] // 2 for a in range(3))

    def curve(centres) -> tuple[list | None, float | None]:
        box = [_centred_window(centres[a], window, label.shape[a]) for a in range(3)]
        if any(b is None for b in box):
            return None, None
        sl = tuple(slice(b[0], b[1]) for b in box)
        mat = material[sl]
        if float(mat.mean()) < min_material:
            return None, None
        _, s2 = s2_radial((label[sl] == LABEL_PORE) & mat)
        return [float(v) for v in s2], float(s2[0])

    inside, inside_zero = curve(centre)
    out: dict = {
        "window": int(window),
        "s2_inside": inside,
        "s2_inside_zero_lag": inside_zero,
        "across": {},
    }
    for axis in block["lower_plane_axes"]:
        centres = list(centre)
        centres[axis] = origin[axis]
        across, across_zero = curve(centres)
        dist = None
        if inside is not None and across is not None:
            a, b = np.asarray(across, float), np.asarray(inside, float)
            ok = np.isfinite(a) & np.isfinite(b)
            denom = float(np.abs(b[ok]).mean()) if ok.any() else 0.0
            if denom > 1e-12:
                dist = float(np.abs(a[ok] - b[ok]).mean() / denom)
        out["across"][AXIS_NAMES[axis]] = {
            "s2": across,
            "s2_zero_lag": across_zero,
            "s2_relative_distance": dist,
        }
    vals = [v["s2_relative_distance"] for v in out["across"].values()
            if v["s2_relative_distance"] is not None]
    out["s2_relative_distance"] = float(np.mean(vals)) if vals else None
    return out


@requires()
def chunk_profile(
    xct_u8: np.ndarray,
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
    period,
    pore_logit: np.ndarray | None = None,
    with_s2: bool = True,
) -> list[dict]:
    """Every per-chunk number for one volume, in generation order.

    ``period`` is the REFERENCE chunk grid in voxels and is given by the
    assessment, not read from the manifest: the whole point is to read volumes
    assembled on different grids at the same planes.  A real crop goes through
    unchanged — it carries no request, and nothing here needs one — which is
    what makes the floor row possible.
    """
    from poregen.eval_v4.microstructure import S2_MIN_MATERIAL, S2_WINDOW  # noqa: PLC0415

    grey = xct_u8.astype(np.float32) / 255.0
    rows = []
    for block in chunk_blocks(label.shape, period):
        row = {
            "chunk_index": block["chunk_index"],
            "origin": block["origin"],
            "shape": block["shape"],
            "lower_plane_axes": block["lower_plane_axes"],
            "xct": chunk_seam_profile(grey, block),
            "porosity": chunk_porosity(label, material, block),
        }
        if pore_logit is not None:
            row["pore"] = chunk_seam_profile(np.asarray(pore_logit, np.float32), block)
        if with_s2:
            row["s2"] = chunk_s2(label, material, block,
                                 window=S2_WINDOW, min_material=S2_MIN_MATERIAL)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 4 - the grey air detector, and cross-head disagreement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GreyDetector:
    """Real-calibrated dark-voxel detector, imported not forked.

    ``t_abs`` is the Dice-optimal absolute u8 threshold measured inside
    ``sample_mask`` on 8 real volumes in campaign 04, and ``min_cc`` drops
    components below 300 voxels.  Both come from the v3 constants so a v3 and a
    v4 air number mean the same thing.
    """

    t_abs: int
    min_cc: int
    edge_vox: int
    source: str


def grey_air_detector(repo: str | Path | None = None) -> GreyDetector:
    """Load the v3 detector settings.

    ``scripts/analysis`` is not an installable package, so it is put on the
    path here and nowhere else.  The import is deliberately late: only the
    assembly assessment needs the detector, and the rest of the suite must not
    fail because a campaign directory is missing.
    """
    root = Path(repo) if repo else repo_root()
    analysis = root / "scripts" / "analysis"
    if not analysis.exists():
        raise FileNotFoundError(
            f"{analysis} does not exist; the grey detector constants live there."
        )
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))
    import _eval_v3  # noqa: PLC0415  (late by design - see the docstring)

    cal = _eval_v3.load_calibration()
    if int(cal["edge_shell_vox"]) != EDGE_VOX:
        raise ValueError(
            f"the v3 edge shell is {cal['edge_shell_vox']} voxels but eval_v4 uses "
            f"{EDGE_VOX}; the interior numbers would not be comparable."
        )
    return GreyDetector(
        t_abs=int(cal["t_abs"]),
        min_cc=int(cal["min_cc_voxels"]),
        edge_vox=int(cal["edge_shell_vox"]),
        source=str(cal["source"]),
    )


def dark_components(xct_u8: np.ndarray, detector: GreyDetector) -> np.ndarray:
    """Dark voxels in components of at least ``min_cc`` voxels."""
    from scipy import ndimage  # noqa: PLC0415  (scipy is only needed here)

    det = xct_u8 < detector.t_abs
    labels, n = ndimage.label(det)
    if n == 0:
        return det
    sizes = np.bincount(labels.ravel())
    small = sizes < detector.min_cc
    small[0] = False
    det[small[labels]] = False
    return det


@requires()
def cross_head_disagreement(
    xct_u8: np.ndarray,
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
    detector: GreyDetector,
) -> dict:
    """Where the two decoder heads contradict each other.

    The grey head renders a dark void; the class head calls the same voxels
    material.  One of them is wrong, and the disagreement is a defect of the
    volume no single-head metric can see.  Measured inside the requested
    material only - dark voxels outside it are the exterior air that was asked
    for.
    """
    dark = dark_components(xct_u8.copy(), detector)
    mat_n = float(material.sum())
    inter = interior_mask(label.shape) & material
    disagree = dark & (label == LABEL_MATERIAL) & material
    return {
        "t_abs": detector.t_abs,
        "min_cc": detector.min_cc,
        "detector_source": detector.source,
        "dark_fraction_in_material": float((dark & material).sum()) / mat_n,
        "disagreement_fraction": float(disagree.sum()) / mat_n,
        "disagreement_fraction_interior": (
            float((disagree & inter).sum()) / float(inter.sum()) if inter.any() else None
        ),
        "label_air_fraction_in_material": float(
            ((label == LABEL_AIR) & material).sum()
        ) / mat_n,
        "grey_air_claimed_by_label": (
            float((dark & material & (label == LABEL_AIR)).sum())
            / float((dark & material).sum())
            if (dark & material).any() else None
        ),
    }


# ---------------------------------------------------------------------------
# 5 - requested geometry
# ---------------------------------------------------------------------------

@requires("requested_material")
def geometry_agreement(
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
) -> dict:
    """Does the predicted air class land where the material map asked for air?

    Dice against the requested air, plus the air fraction inside and outside the
    requested material.  A model that obeys the map has a high Dice, a high air
    fraction outside and a low one inside.
    """
    if manifest.requested_material == "full":
        raise ManifestError(
            "geometry_agreement needs a painted material map; this case requested "
            "'full', so there is no requested air to score against."
        )
    pred_air = label == LABEL_AIR
    req_air = ~material
    inter = float(np.logical_and(pred_air, req_air).sum())
    n_pred, n_req = float(pred_air.sum()), float(req_air.sum())
    return {
        "dice_air": (2.0 * inter / (n_pred + n_req)) if (n_pred + n_req) else float("nan"),
        "precision_air": (inter / n_pred) if n_pred else float("nan"),
        "recall_air": (inter / n_req) if n_req else float("nan"),
        "requested_air_fraction": n_req / float(label.size),
        "air_fraction_inside_material": float(pred_air[material].mean()),
        "air_fraction_outside_material": float(pred_air[req_air].mean()) if n_req else None,
        "phi_pore_inside_material": float((label[material] == LABEL_PORE).mean()),
    }


# ---------------------------------------------------------------------------
# 6 - layup recovery
# ---------------------------------------------------------------------------

def _ti_readers(repo: Path):
    """The T-I angle estimators, imported from the campaign-08 reader.

    ``scripts/analysis/layup_roundtrip.py`` is imported INSIDE this function on
    purpose.  Its generation half is pinned to the ldm05 sampler API, which
    ldm06 replaced; only its estimator half (``measure_volume``,
    ``score_recovery``, ``ply_edges``, ``requested_sequence``) is reused, and a
    module-scope import would tie eval v4 to a dead generation path.
    """
    analysis = repo / "scripts" / "analysis"
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))
    import layup_roundtrip  # noqa: PLC0415  (late by design - see the docstring)

    return layup_roundtrip


@requires("requested_layup", "requested_ply_thickness_vox")
def layup_recovery(
    xct_u8: np.ndarray,
    label: np.ndarray,
    *,
    manifest: Manifest,
    repo: str | Path | None = None,
    readers: tuple[str, ...] = ("fft_slice", "pore_axes"),
) -> dict:
    """Read the ply angles back and score them DIRECTLY against the request.

    Direct means no offset, no sign flip, no face reversal: ``theta_from_layup``
    writes the requested angles in image coordinates, so measured == requested
    is the null hypothesis and any freedom granted here would flatter the
    model.  ``fft_slice`` reads the grey volume; ``pore_axes`` reads the
    PREDICTED pore class, so it is scored against the campaign-08 floor that
    used the STORED mask - a ceiling no predicted-mask reader can beat.
    """
    lr = _ti_readers(Path(repo) if repo else repo_root())
    window = lr.ti.WINDOW
    if xct_u8.shape[1] < window or xct_u8.shape[2] < window:
        raise ValueError(
            f"the T-I readers need a {window}x{window} in-plane window; this "
            f"volume is {xct_u8.shape[1]}x{xct_u8.shape[2]}."
        )

    depth = xct_u8.shape[0]
    pitch = float(manifest.requested_ply_thickness_vox)
    edges = lr.ply_edges(depth, pitch)
    requested = lr.requested_sequence(list(manifest.requested_layup), len(edges) - 1)

    pore_u8 = (label == LABEL_PORE).astype(np.uint8)
    est = lr.measure_volume(xct_u8, pore_u8, edges, pitch)

    out: dict = {
        "n_plies": int(len(requested)),
        "ply_thickness_vox": pitch,
        "requested_deg": requested.tolist(),
        "requested_class": angle_class(requested).tolist(),
        "requested_ply_count": int(_run_count(angle_class(requested))),
        "window": int(window),
        "readers": {},
    }
    for name in readers:
        if name not in est:
            out["readers"][name] = {"available": False}
            continue
        ang = np.asarray(est[name]["angles"], float)
        w = np.asarray(est[name]["weights"], float)
        score = lr.score_recovery(ang, w, requested)
        pred_cls = angle_class(ang)
        out["readers"][name] = {
            "available": True,
            "recovered_deg": ang.tolist(),
            "recovered_class": pred_cls.tolist(),
            "per_ply_hit": (pred_cls == angle_class(requested)).tolist(),
            "median_abs_error_deg": score["direct_median_abs_error_deg"],
            "max_abs_error_deg": score["direct_max_abs_error_deg"],
            "frac_within_10": score["direct_frac_within_10"],
            "strict_class_accuracy": score["strict_class_accuracy"],
            "errors_deg": score["direct_errors_deg"],
            "recovered_ply_count": int(_run_count(pred_cls)),
        }
    return out


def _run_count(classes: np.ndarray) -> int:
    """Number of plies a sequence of per-block classes resolves into.

    A ply boundary is a class change between consecutive blocks, so the count
    is one more than the number of changes.  Two identical adjacent plies read
    as one - which is the honest answer, because nothing in the volume
    distinguishes them.
    """
    c = np.asarray(classes).ravel()
    if c.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(c) != 0))


def layup_floor(repo: str | Path | None = None) -> dict:
    """The real-volume reader floor, read from campaign 08.

    Never hard-coded: the floor is a measurement, and a measurement that is
    typed into source drifts away from the file that produced it.
    """
    root = Path(repo) if repo else repo_root()
    path = (root / "runs" / "campaigns" / "08-pre-ldm06-diagnostics"
            / "angle_reader_floor" / "results.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; the layup readers have no real floor to be "
            "reported against."
        )
    import json  # noqa: PLC0415

    pooled = json.loads(path.read_text())["pooled"]
    return {
        name: {
            "median_abs_error_deg": pooled[name]["median_abs_error_deg"],
            "strict_class_accuracy": pooled[name]["strict_class_accuracy"],
            "frac_within_10": pooled[name]["frac_within_10"],
            "n_plies": pooled[name]["n_plies"],
        }
        for name in ("fft_slice", "pore_axes")
        if name in pooled
    } | {"source": str(path)}


def vae_tile_decode_control(repo: str | Path | None = None) -> dict:
    """The campaign-08 VAE tile-decode control row.

    Real data through the VAE with no LDM in the loop, assembled two ways.  It
    separates an assembly-side seam from a model-side one, so the assembly
    table is read against it rather than against 1.0.
    """
    root = Path(repo) if repo else repo_root()
    path = (root / "runs" / "campaigns" / "08-pre-ldm06-diagnostics"
            / "vae_tile_seam" / "results.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; the assembly control row is missing.")
    import json  # noqa: PLC0415

    res = json.loads(path.read_text())
    rows: dict[str, dict] = {}
    for rec in res["records"]:
        for asm in rec["assemblies"]:
            rows.setdefault(asm["assembly"], {"seam_xct_ratio": [], "seam_mask_ratio": []})
            rows[asm["assembly"]]["seam_xct_ratio"].append(asm["seam_xct_ratio"])
            rows[asm["assembly"]]["seam_mask_ratio"].append(asm["seam_mask_ratio"])
    return {
        "source": str(path),
        "n_volumes": len(res["records"]),
        "assemblies": {
            name: {
                "seam_xct_ratio": mean_sd(v["seam_xct_ratio"]),
                "seam_mask_ratio": mean_sd(v["seam_mask_ratio"]),
            }
            for name, v in rows.items()
        },
    }


# ---------------------------------------------------------------------------
# 9 - the specimen surface
# ---------------------------------------------------------------------------

def height_map(material: np.ndarray, *, face: str) -> np.ndarray:
    """Per-(y, x) column, the z of the outermost material voxel of one face.

    ``face="lower"`` returns the FIRST material z, ``"upper"`` the LAST plus one,
    so both are the position of the interface itself rather than of the last
    solid voxel. Columns with no material at all are NaN — they carry no
    surface, and averaging a sentinel into the height would bend the roughness
    toward whatever the sentinel was.
    """
    if face not in ("lower", "upper"):
        raise ValueError(f"face must be 'lower' or 'upper', got {face!r}")
    any_mat = material.any(axis=0)
    if face == "lower":
        idx = material.argmax(axis=0).astype(np.float64)
    else:
        # argmax on the reversed axis finds the last True.
        idx = (material.shape[0] - material[::-1].argmax(axis=0)).astype(np.float64)
    idx[~any_mat] = np.nan
    return idx


def surface_roughness(h: np.ndarray) -> dict:
    """Sa and Sq of a height map, in voxels, about its own mean plane.

    Sa is the mean absolute deviation and Sq the RMS deviation — the two
    standard areal roughness parameters. Both are taken about the MEAN of the
    map, not about the requested plane, so they measure how rough the surface is
    independently of whether it landed in the right place. Position error is
    reported separately; a surface can be flat and displaced, or centred and
    ragged, and one number cannot say which.
    """
    v = h[np.isfinite(h)]
    if v.size == 0:
        return {"sa": None, "sq": None, "n_columns": 0, "mean_z": None}
    dev = v - v.mean()
    return {
        "sa": float(np.abs(dev).mean()),
        "sq": float(np.sqrt((dev ** 2).mean())),
        "n_columns": int(v.size),
        "mean_z": float(v.mean()),
    }


#: A column whose interface is further than this from the request is an
#: outlier, not roughness. Same threshold as the paper's position gate.
SURFACE_OUTLIER_VOX = 4
#: Half-width of the partial-volume rim excluded from the second
#: dark-but-material reading.
SURFACE_RIM_VOX = 2


def _surface_outliers(h: np.ndarray, requested, label: np.ndarray,
                      face: str) -> dict:
    """Where the interface is badly displaced, and whether a pore explains it.

    A pore breaking the surface is legitimate geometry — the specimen really
    does end early in that column. A displaced surface with no pore at it is
    the model putting the interface in the wrong place. The mean position error
    cannot tell those apart, so they are counted separately.

    Clustering matters for the same reason: scattered single columns read as
    noise in the label, while a connected patch reads as a region the model got
    wrong.
    """
    from scipy import ndimage                          # noqa: PLC0415

    err = h - np.asarray(requested, dtype=np.float64)
    bad = np.isfinite(h) & (np.abs(err) > SURFACE_OUTLIER_VOX)
    n_cols = int(np.isfinite(h).sum())
    out: dict = {
        "threshold_vox": SURFACE_OUTLIER_VOX,
        "n_outliers": int(bad.sum()),
        "fraction": (float(bad.sum()) / n_cols) if n_cols else None,
    }
    if not bad.any():
        out.update({"n_clusters": 0, "largest_cluster": 0,
                    "with_pore_at_face": 0, "with_pore_at_face_fraction": None})
        return out

    lab, n = ndimage.label(bad)                        # 4-connectivity in 2-D
    sizes = ndimage.sum(bad, lab, index=np.arange(1, n + 1))
    out["n_clusters"] = int(n)
    out["largest_cluster"] = int(sizes.max())
    out["cluster_size_median"] = float(np.median(sizes))
    # Scattered singletons vs a coherent patch, in one number.
    out["singleton_clusters"] = int((sizes == 1).sum())

    # Does a pore sit at the interface of each outlier column? Checked in a
    # +/-SURFACE_RIM_VOX window around the PREDICTED interface, because that is
    # where a surface-breaking pore would be.
    d = label.shape[0]
    ys, xs = np.nonzero(bad)
    with_pore = 0
    for y, x in zip(ys, xs):
        z = int(round(float(h[y, x])))
        lo = max(0, z - SURFACE_RIM_VOX)
        hi = min(d, z + SURFACE_RIM_VOX + 1)
        if np.any(label[lo:hi, y, x] == LABEL_PORE):
            with_pore += 1
    out["with_pore_at_face"] = int(with_pore)
    out["with_pore_at_face_fraction"] = float(with_pore) / float(bad.sum())
    out["face"] = face
    return out


def _dark_but_material(xct: np.ndarray, material: np.ndarray,
                       pred_specimen: np.ndarray, z_lo: int, z_hi: int,
                       threshold: int) -> dict:
    """Dark-but-material over the whole specimen, and away from the faces.

    The two requested faces carry a partial-volume rim: a voxel straddling the
    real interface is genuinely part air, so it is genuinely dark, and counting
    it as an unlabelled void inflates the number. The second reading excludes a
    SURFACE_RIM_VOX band at each face so the rim's contribution is MEASURED
    rather than assumed away or assumed harmless.
    """
    inside = material & pred_specimen
    full = (float((xct[inside] < threshold).mean()) if inside.any() else None)

    core = inside.copy()
    d = core.shape[0]
    zz = np.arange(d)[:, None, None]
    for req in (z_lo, z_hi):
        r = np.asarray(req, dtype=np.float64)
        # A rough request has a per-column face, so the rim follows the surface
        # rather than sitting at a fixed z.
        rf = r[None, :, :] if r.ndim == 2 else float(r)
        core &= ~(np.abs(zz - rf) < SURFACE_RIM_VOX)
    excl = (float((xct[core] < threshold).mean()) if core.any() else None)
    return {
        "all_material": full,
        "excluding_face_rim": excl,
        "rim_vox": SURFACE_RIM_VOX,
        "n_voxels_all": int(inside.sum()),
        "n_voxels_excluding_rim": int(core.sum()),
        "threshold": int(threshold),
    }


def surface_agreement(
    label: np.ndarray,
    material: np.ndarray,
    *,
    z_lo: int,
    z_hi: int,
    xct: np.ndarray | None = None,
    dark_threshold: int = 182,
) -> dict:
    """Does the model render air where the map asks, and where is the interface?

    ``material`` is the REQUESTED map. ``z_lo`` / ``z_hi`` are the requested
    interface positions, so the position error is signed against a known truth
    rather than against the model's own output.

    The predicted specimen is ``label != LABEL_AIR`` — pores count as specimen,
    which is what the material map means (it says where the coupon is, not how
    solid it is). Scoring against ``label == LABEL_MATERIAL`` instead would read
    every pore near the surface as a hole in the surface.
    """
    pred_specimen = label != LABEL_AIR
    pred_air = label == LABEL_AIR
    req_air = ~material

    out: dict = {
        "requested_z_lo": (float(np.mean(z_lo)) if np.ndim(z_lo) else int(z_lo)),
        "requested_z_hi": (float(np.mean(z_hi)) if np.ndim(z_hi) else int(z_hi)),
        "request_is_field": bool(np.ndim(z_lo) or np.ndim(z_hi)),
        "air_fraction_outside_box": (float(pred_air[req_air].mean())
                                     if req_air.any() else None),
        "air_fraction_inside_box": (float(pred_air[material].mean())
                                    if material.any() else None),
    }

    for face, requested in (("lower", z_lo), ("upper", z_hi)):
        h = height_map(pred_specimen, face=face)
        rough = surface_roughness(h)
        req = np.asarray(requested, dtype=np.float64)
        req_field = req if req.ndim == 2 else np.full(h.shape, float(req))
        ok = np.isfinite(h)
        finite = h[ok]
        # Error against the REQUESTED FIELD, per column. Against a plane, a
        # correctly-followed rough request would read as position error equal to
        # the requested roughness — the model would be marked wrong for obeying.
        err = finite - req_field[ok]
        req_rough = surface_roughness(np.where(ok, req_field, np.nan))
        out[face] = {
            "requested_z": (float(req) if req.ndim == 0 else None),
            "requested_is_field": bool(req.ndim == 2),
            "requested_roughness_sa": req_rough["sa"],
            "requested_roughness_sq": req_rough["sq"],
            "roughness_ratio_to_requested": (
                (rough["sa"] / req_rough["sa"])
                if rough["sa"] is not None and req_rough["sa"] else None),
            "position_mean": (float(finite.mean()) if finite.size else None),
            "position_sd": (float(finite.std()) if finite.size else None),
            "error_mean": (float(err.mean()) if err.size else None),
            "error_abs_mean": (float(np.abs(err).mean()) if err.size else None),
            "error_max_abs": (float(np.abs(err).max()) if err.size else None),
            "columns_without_material": int(np.isnan(h).sum()),
            **{f"roughness_{k}": v for k, v in rough.items()},
            "outliers": _surface_outliers(h, req_field, label, face),
        }

    if xct is not None:
        # Dark-but-material: voxels the map says are specimen and the label
        # calls specimen, yet whose grey level is as dark as air. It catches a
        # decoder that renders the void correctly but does not label it.
        # Reported twice — see _dark_but_material for why the rim matters.
        dbm = _dark_but_material(xct, material, pred_specimen, z_lo, z_hi,
                                 dark_threshold)
        out["dark_but_material_detail"] = dbm
        out["dark_but_material"] = dbm["all_material"]   # unchanged key
        out["dark_but_material_excluding_rim"] = dbm["excluding_face_rim"]
        out["dark_threshold"] = int(dark_threshold)
    return out


# ---------------------------------------------------------------------------
# The spherical specimen (exploratory - no gate)
# ---------------------------------------------------------------------------

#: Shell width for the radial profile, in voxels.
SPHERE_SHELL_VOX = 2


def sphere_agreement(
    label: np.ndarray,
    material: np.ndarray,
    *,
    radius: float,
    requested_phi: float | None = None,
    by_octant: bool = False,
) -> dict:
    """What the model does with a curved specimen it has never been shown.

    EXPLORATORY. There is no real spherical coupon, so there is no floor and no
    gate — every number here is descriptive, and a threshold would be invented.

    The radial profile is the point: a model that has learned "specimen" as a
    slab between two z planes will not carve a sphere, and where it fails will
    show as air appearing before the requested radius or material persisting
    past it. A single Dice cannot say which.
    """
    pred_air = label == LABEL_AIR
    pred_specimen = ~pred_air
    req_air = ~material

    inter = float(np.logical_and(pred_air, req_air).sum())
    n_pred, n_req = float(pred_air.sum()), float(req_air.sum())
    out: dict = {
        "requested_radius_vox": float(radius),
        "dice_air": (2.0 * inter / (n_pred + n_req)) if (n_pred + n_req) else None,
        "air_fraction_inside_sphere": (float(pred_air[material].mean())
                                       if material.any() else None),
        "air_fraction_outside_sphere": (float(pred_air[req_air].mean())
                                        if req_air.any() else None),
        "requested_material_fraction": float(material.mean()),
        "predicted_specimen_fraction": float(pred_specimen.mean()),
    }

    # Pore porosity INSIDE the requested sphere, against what was asked.
    inside = material
    out["phi_pore_inside_sphere"] = (float((label[inside] == LABEL_PORE).mean())
                                     if inside.any() else None)
    out["requested_phi"] = (float(requested_phi) if requested_phi is not None else None)
    if requested_phi:
        out["phi_error"] = out["phi_pore_inside_sphere"] - float(requested_phi)

    d, h, w = label.shape
    zz = (np.arange(d) - (d - 1) / 2.0)[:, None, None]
    yy = (np.arange(h) - (h - 1) / 2.0)[None, :, None]
    xx = (np.arange(w) - (w - 1) / 2.0)[None, None, :]
    r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)

    r_max = float(min(d, h, w) / 2.0)
    edges = np.arange(0.0, r_max + SPHERE_SHELL_VOX, SPHERE_SHELL_VOX)
    idx = np.digitize(r.ravel(), edges) - 1
    air_flat = pred_air.ravel()
    pore_flat = (label == LABEL_PORE).ravel()
    n_shell = np.bincount(idx[idx >= 0], minlength=len(edges))
    n_air = np.bincount(idx[idx >= 0], weights=air_flat[idx >= 0], minlength=len(edges))
    n_pore = np.bincount(idx[idx >= 0], weights=pore_flat[idx >= 0], minlength=len(edges))
    keep = n_shell > 0
    out["radial_profile"] = {
        "shell_vox": SPHERE_SHELL_VOX,
        "r_centre": ((edges[:-1] + SPHERE_SHELL_VOX / 2.0)[keep[:-1]]).tolist(),
        "air_fraction": (n_air[:-1][keep[:-1]] / n_shell[:-1][keep[:-1]]).tolist(),
        "pore_fraction": (n_pore[:-1][keep[:-1]] / n_shell[:-1][keep[:-1]]).tolist(),
        "n_voxels": n_shell[:-1][keep[:-1]].astype(int).tolist(),
    }

    # Where the predicted surface actually sits, radially: the radius at which
    # the shell air fraction first crosses one half.
    rc = np.asarray(out["radial_profile"]["r_centre"])
    af = np.asarray(out["radial_profile"]["air_fraction"])
    cross = np.flatnonzero(af >= 0.5)
    r50 = float(rc[cross[0]]) if cross.size else None
    out["radial_surface"] = {
        "r_at_air_half": r50,
        "error_vox": (r50 - float(radius)) if r50 is not None else None,
        "definition": ("radius where the shell air fraction first reaches 0.5; "
                       "the radial analogue of the flat cases' surface position"),
    }

    if by_octant:
        # The radial profile in each octant separately. A chunk-boundary
        # artefact is not radially symmetric — it follows the chunk planes — so
        # it shows up as octants disagreeing with each other while the pooled
        # profile looks clean.
        oct_r50: dict[str, float | None] = {}
        for oz in (0, 1):
            for oy in (0, 1):
                for ox in (0, 1):
                    sl = (slice(d // 2, None) if oz else slice(0, d // 2),
                          slice(h // 2, None) if oy else slice(0, h // 2),
                          slice(w // 2, None) if ox else slice(0, w // 2))
                    ro = r[sl].ravel()
                    ao = pred_air[sl].ravel()
                    io = np.digitize(ro, edges) - 1
                    ns = np.bincount(io[io >= 0], minlength=len(edges))
                    na = np.bincount(io[io >= 0], weights=ao[io >= 0], minlength=len(edges))
                    k = ns[:-1] > 0
                    afo = na[:-1][k] / ns[:-1][k]
                    rco = (edges[:-1] + SPHERE_SHELL_VOX / 2.0)[k]
                    c = np.flatnonzero(afo >= 0.5)
                    oct_r50[f"z{oz}y{oy}x{ox}"] = float(rco[c[0]]) if c.size else None
        vals = [v for v in oct_r50.values() if v is not None]
        out["radial_surface_by_octant"] = {
            "r_at_air_half": oct_r50,
            "spread_vox": (float(max(vals) - min(vals)) if len(vals) > 1 else None),
            "n_octants_without_crossing": int(8 - len(vals)),
            "why": ("a chunk-boundary artefact follows the chunk planes rather "
                    "than the radius, so it appears as octant disagreement while "
                    "the pooled profile still looks clean"),
        }
    return out


def pore_dice_across_planes(label: np.ndarray, period) -> dict:
    """Pore agreement between the slabs either side of each chunk plane.

    The two slabs are produced by different chunk solves, so a pore structure
    that stops dead at the plane is an assembly failure rather than texture.
    Compared as the Dice between the last slice before the plane and the first
    slice after it, per axis, which is the cheapest statement of "does the
    structure continue".
    """
    per = period if isinstance(period, (tuple, list)) else (period,) * 3
    pore = label == LABEL_PORE
    vals: list[float] = []
    detail: dict[str, list[float]] = {}
    for axis, p_ in enumerate(per):
        if not p_:
            continue
        axis_vals = []
        for idx in range(int(p_), label.shape[axis], int(p_)):
            a = np.take(pore, idx - 1, axis=axis)
            b = np.take(pore, idx, axis=axis)
            n = float(a.sum() + b.sum())
            if n == 0:
                continue
            axis_vals.append(2.0 * float((a & b).sum()) / n)
        if axis_vals:
            detail[f"axis{axis}"] = axis_vals
            vals.extend(axis_vals)
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "min": float(np.min(vals)) if vals else None,
        "n_planes": len(vals),
        "per_axis": detail,
        "definition": ("Dice between the slices either side of each chunk plane; "
                       "low means pore structure stops at the plane"),
    }

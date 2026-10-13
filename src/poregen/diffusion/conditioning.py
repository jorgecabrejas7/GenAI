"""Shared conditioning conventions for the LDM (ldm06).

This module is the SINGLE source of truth for the conventions that the data
side (latent store construction, ``LatentDataset``) and the model/sampler side
(``UNet3DDenoiser``, ``VolumeGenerator``) must agree on exactly:

* the neighbour direction order (:data:`NEIGHBOUR_DIRS`),
* the availability state codes (:data:`NB_OOB` / :data:`NB_EXISTS` /
  :data:`NB_UNKNOWN`),
* the grid index of a patch (:func:`grid_index`),
* the six per-face distances to the specimen box (:data:`DIST6_DIRS`,
  :func:`dist6_from_box`),
* the porosity transform (:func:`porosity_to_cond`) and the report of a
  request against the train porosity (:func:`porosity_request_report`),
* the geometry guard that keeps neighbours from leaking target content
  (:func:`validate_neighbour_geometry`).

Anything that computes neighbour conditioning MUST import from here rather
than re-deriving it, otherwise training and generation silently disagree.

Neighbour availability states
-----------------------------
0 — OOB (out of bounds): the specimen ends at that face; there is no
    neighbour and there never will be.
1 — EXISTS: the neighbour latent is known and has been provided.
2 — UNKNOWN: a neighbour exists physically but has not been resolved yet —
    a chunk the sampler has not reached, or the CFG neighbour null.

The data side emits EXISTS for a stored neighbour, OOB for a missing one where
the specimen ends within one patch on that side, and UNKNOWN for a missing one
where the specimen continues (a patch removed for a hole or by the index; vault
D57).  UNKNOWN also enters training through the ``drop_nb`` dropout of the
training step (exactly the neighbour-null arm of the nested CFG) and generation
through the chunks the sampler has not written yet.

Neighbours TOUCH, they do not overlap
-------------------------------------
``neighbour_offset`` must be at least ``patch_size``.  At the original
``neighbour_offset = 32`` with ``patch_size = 64`` a face neighbour shared
HALF its voxels with the target, and the six neighbours between them tiled
the target completely.  Measured leak: the ``+z`` neighbour reproduced the
target's overlap region with MAE 0.067 against 0.744 for a random patch
(ratio 0.09) — a verbatim copy.  The model learned to copy rather than to
generate.  At ``neighbour_offset = 64`` the neighbour is face-adjacent: zero
shared voxels, so nothing of the target can be read off it, and the whole
unshifted neighbour latent is the correct network input.

Six per-face distances instead of one
-------------------------------------
ldm05 conditioned on a single scalar ``cond_dist`` — the distance from the
patch centre to the NEAREST outer specimen face over all three axes.  One
number cannot say *which* face is close, so a patch against the top surface
and a patch against a side wall were asked for the same thing, and the model
could not learn that exterior air lies on one particular side.
:data:`DIST6_DIRS` replaces it with one distance per face, measured from that
face of the patch to the matching face of the specimen box, capped at
:data:`DIST_CAP` voxels and normalised.  A value of 0 means "the specimen
ends right here, on this side"; 1 means "at least 64 voxels of specimen that
way".
"""

from __future__ import annotations

import numpy as np

NB_OOB     = 0
NB_EXISTS  = 1
NB_UNKNOWN = 2
N_AVAIL_STATES = 3

# Neighbour directions as (dz, dy, dx) grid-index offsets.  Index i of any
# ``nb_latents`` / ``nb_avail`` / ``nb_t`` tensor refers to NEIGHBOUR_DIRS[i].
NEIGHBOUR_DIRS: tuple[tuple[int, int, int], ...] = (
    ( 1, 0, 0), (-1, 0, 0),
    ( 0, 1, 0), ( 0,-1, 0),
    ( 0, 0, 1), ( 0, 0,-1),
)

N_NEIGHBOURS = len(NEIGHBOUR_DIRS)

# Face order of ``cond_dist6``: (z-, z+, y-, y+, x-, x+).  This is DELIBERATELY
# not the ``NEIGHBOUR_DIRS`` order — cond_dist6 is a geometry vector read by one
# MLP, not a per-neighbour tensor, and low-then-high per axis is how the
# specimen box itself is written.  Both producers (the conditioning builder and
# the sampler) import these constants, so the two can never drift.
DIST6_DIRS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0), ( 1, 0, 0),
    ( 0,-1, 0), ( 0, 1, 0),
    ( 0, 0,-1), ( 0, 0, 1),
)
DIST6_NAMES: tuple[str, ...] = ("zm", "zp", "ym", "yp", "xm", "xp")
N_DIST6 = len(DIST6_DIRS)

# Distance-to-surface cap in voxels: beyond one patch the exact distance stops
# mattering, so every face saturates at 1.0.
DIST_CAP = 64.0

# Porosity transform: cond_por = (log(phi + POR_LOG_EPS) - mean) / std, with
# (mean, std) measured on the train split and recorded in the store metadata.
POR_LOG_EPS = 1e-3

# A porosity request is NEVER clamped to the training range: a request outside
# it is a legitimate generalisation test (the author, 2026-10-13), like an
# unknown stacking sequence or shape.  The only bound is physical, 0 <= phi <= 1.
# What the training set contained is recorded by the store build
# (:func:`porosity_distribution`, ``metadata["train_porosity"]``) and every
# request is REPORTED against it (:func:`porosity_request_report`).

#: Percentiles of the train patch porosity recorded by name in the store.
POR_PERCENTILES = (0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5)
#: The full quantile grid the store records, so the percentile of any request
#: can be read back by interpolation: 0, 0.5, 1, ..., 100.
POR_QUANTILE_LEVELS = tuple(float(v) for v in np.arange(0.0, 100.0 + 1e-9, 0.5))


def check_physical_porosity(phi, what: str = "porosity request") -> None:
    """Raise unless every value is a physical pore fraction, 0 <= phi <= 1."""
    a = np.asarray(phi, dtype=np.float64)
    if not np.all(np.isfinite(a)) or a.min(initial=0.0) < 0.0 or a.max(initial=0.0) > 1.0:
        raise ValueError(
            f"{what} must lie in [0, 1] (a pore volume fraction); got "
            f"min {np.nanmin(a):.4g}, max {np.nanmax(a):.4g}.")


def porosity_distribution(phi: np.ndarray) -> dict:
    """The train patch porosity distribution, as the store records it.

    ``phi`` is the per-patch pore fraction measured on the patch LABELS
    (``label == 1`` over the whole 64-cubed patch).
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.size == 0:
        raise ValueError("no train patch: the porosity distribution is undefined")
    return {
        "n_patches": int(phi.size),
        "min": float(phi.min()),
        "max": float(phi.max()),
        "percentiles": {f"{q:g}": float(np.percentile(phi, q)) for q in POR_PERCENTILES},
        "frac_zero_pore": float(np.mean(phi == 0.0)),
        "quantile_levels": list(POR_QUANTILE_LEVELS),
        "quantiles": [float(v) for v in np.percentile(phi, POR_QUANTILE_LEVELS)],
        "definition": ("full-patch pore fraction (voxel label == 1) of every TRAIN "
                       "patch, measured on the patch labels"),
    }


def train_percentile(phi: float, dist: dict) -> float:
    """Percentile of ``phi`` in the recorded train distribution (0..100)."""
    q = np.asarray(dist["quantiles"], dtype=np.float64)
    lv = np.asarray(dist["quantile_levels"], dtype=np.float64)
    # Quantiles repeat where the distribution has an atom (phi = 0); take the
    # highest level a value reaches, so phi = 0 reads as the zero-pore fraction.
    if phi < q[0]:
        return 0.0
    if phi > q[-1]:
        return 100.0
    return float(lv[np.searchsorted(q, phi, side="right") - 1]
                 if np.any(q == phi) else np.interp(phi, q, lv))


def porosity_request_report(dist: dict, global_phi=None, tile_field=None) -> dict:
    """Where a request sits against the train distribution.  Never alters it.

    A value outside ``[train min, train max]`` is flagged "outside training
    range"; the global request also gets its train percentile.  Both the global
    request and, when one is given, the per-tile field are reported.
    """
    lo, hi = float(dist["min"]), float(dist["max"])
    out: dict = {"train_min": lo, "train_max": hi, "global": None, "field": None}
    if global_phi is not None:
        g = float(global_phi)
        out["global"] = {"phi": g, "outside_training_range": not (lo <= g <= hi),
                         "train_percentile": train_percentile(g, dist)}
    if tile_field is not None:
        f = np.asarray(tile_field, dtype=np.float64)
        outside = (f < lo) | (f > hi)
        out["field"] = {"min": float(f.min()), "max": float(f.max()),
                        "frac_tiles_outside": float(outside.mean()),
                        "outside_training_range": bool(outside.any())}
    out["outside_training_range"] = bool(
        (out["global"] or {}).get("outside_training_range")
        or (out["field"] or {}).get("outside_training_range"))
    return out


# ── geometry guard: neighbours must not share voxels with the target ──────────

def neighbour_shared_voxels(neighbour_offset: int, patch_size: int) -> int:
    """Voxels a face neighbour shares with the target patch.

    A face neighbour is displaced by ``neighbour_offset`` voxels along one
    axis, so the two cubes intersect in ``max(patch_size - neighbour_offset, 0)
    × patch_size × patch_size`` voxels.  Zero means the patches only touch.
    """
    overlap_1d = max(int(patch_size) - int(neighbour_offset), 0)
    return overlap_1d * int(patch_size) * int(patch_size)


def validate_neighbour_geometry(neighbour_offset: int, patch_size: int) -> None:
    """Raise unless a face neighbour shares NO voxel with the target patch.

    ``neighbour_offset >= patch_size`` is the invariant that makes neighbour
    conditioning honest.  Below it the neighbour carries a verbatim copy of
    part of the answer and the denoiser learns to copy — the ldm05 failure this
    guard exists to prevent (see the module docstring).
    """
    shared = neighbour_shared_voxels(neighbour_offset, patch_size)
    if shared == 0:
        return
    raise ValueError(
        f"neighbour_offset={neighbour_offset} < patch_size={patch_size}: each face "
        f"neighbour would share {shared} voxels with the target "
        f"({patch_size - neighbour_offset} of {patch_size} planes), handing the "
        f"denoiser part of the answer.  Use neighbour_offset >= {patch_size} "
        f"(touching neighbours)."
    )


def grid_index(origin, neighbour_offset: int) -> tuple[int, int, int]:
    """Assembly-grid index of a patch origin ``(z0, y0, x0)`` in voxels.

    Floor division by ``neighbour_offset`` — not by the dataset sampling
    stride — so that a ±``neighbour_offset`` displacement is exactly ±1 grid
    step on that axis whatever the origin is.  Patches sampled at a finer
    stride form several interleaved copies of the same grid; each copy is
    self-contained because neighbour relations never cross between copies.
    """
    g = int(neighbour_offset)
    return (int(origin[0]) // g, int(origin[1]) // g, int(origin[2]) // g)


# ── per-face distance to the specimen box ─────────────────────────────────────

def dist6_from_box(
    origin,
    patch_size: int,
    box_lo,
    box_hi,
    dist_cap: float = DIST_CAP,
) -> np.ndarray:
    """Six per-face distances of one patch to the specimen box, normalised.

    Parameters
    ----------
    origin     : (z0, y0, x0) patch origin in voxels
    patch_size : voxel side length of the patch
    box_lo     : (z, y, x) lower corner of the specimen box, INCLUSIVE
    box_hi     : (z, y, x) upper corner of the specimen box, EXCLUSIVE
    dist_cap   : saturation distance in voxels

    Returns
    -------
    (6,) float32 in [0, 1], ordered by :data:`DIST6_DIRS`.  Entry ``2·a`` is
    the gap between the patch's low face on axis ``a`` and the box's low face;
    entry ``2·a + 1`` the gap at the high faces.
    """
    o = np.asarray(origin, dtype=np.float64)
    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    p = float(patch_size)
    gaps = np.empty(N_DIST6, dtype=np.float64)
    gaps[0::2] = o - lo                 # low faces: z-, y-, x-
    gaps[1::2] = hi - (o + p)           # high faces: z+, y+, x+
    return (np.clip(gaps, 0.0, dist_cap) / dist_cap).astype(np.float32)


def dist6_from_box_array(
    origins: np.ndarray,
    patch_size: int,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    dist_cap: float = DIST_CAP,
) -> np.ndarray:
    """Vectorised :func:`dist6_from_box` over many patches.

    Parameters
    ----------
    origins : (N, 3) patch origins in voxels
    box_lo  : (N, 3) inclusive lower corners
    box_hi  : (N, 3) exclusive upper corners

    Returns
    -------
    (N, 6) float32, same face ordering as :func:`dist6_from_box`.
    """
    o = np.asarray(origins, dtype=np.float64)
    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    gaps = np.empty((o.shape[0], N_DIST6), dtype=np.float64)
    gaps[:, 0::2] = o - lo
    gaps[:, 1::2] = hi - (o + float(patch_size))
    return (np.clip(gaps, 0.0, dist_cap) / dist_cap).astype(np.float32)


# ── porosity transform ────────────────────────────────────────────────────────

def porosity_to_cond(phi, por_log_stats: tuple[float, float] | None):
    """Map raw pore volume fraction to the model's ``cond_por`` scalar.

    ``por_log_stats`` is the (mean, std) of ``log(phi + POR_LOG_EPS)`` over the
    train split, as recorded by the latent store.  Passing None is an error — a
    wrong standardisation silently mis-conditions every patch.
    """
    if por_log_stats is None:
        raise ValueError(
            "por_log_stats is required to build cond_por. Pass the (mean, std) of "
            "log(phi + 1e-3) recorded by the latent store's metadata."
        )
    mean, std = por_log_stats
    return (np.log(np.asarray(phi, dtype=np.float64) + POR_LOG_EPS) - mean) / std

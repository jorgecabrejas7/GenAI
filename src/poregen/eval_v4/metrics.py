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

from poregen.diffusion.sampler import seam_discontinuity
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

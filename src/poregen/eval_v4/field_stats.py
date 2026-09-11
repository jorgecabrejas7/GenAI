"""The spatial statistics of the DELIVERED local porosity field.

Naiff, Ramos and Wang ("Large-Scale Porous Media Generation Through
Field-Controlled Latent Diffusion Models", SSRN 10.2139/ssrn.7161201) claim that
a porosity field is a SUFFICIENT descriptor of large-scale heterogeneity.  That
claim is about a measurable quantity, and this module measures it: the marginal
distribution of local porosity per window, and how far that field stays
correlated along each axis.

Three things decide whether the answer means anything.

**Delivered, not requested.**  Every field here is read out of a LABEL - the
pore voxels a volume actually holds over the material voxels it actually holds -
never out of the ``requested_field.npy`` beside it.  The request is measured
too, and reported as its own row, because the interesting failure is a request
with the right statistics that the model does not deliver.

**One window size on both sides.**  The window is 64 voxels, the conditioning
tile the model is asked about, stepped every 32.  That is exactly the real patch
grid campaign 01's T-D measured the real correlation lengths on (z 79.45,
y 413.58, x 900.95 voxels), which are in turn the lengths
:func:`poregen.diffusion.porosity_field.build_porosity_field` smooths the
coherent request with.  Real crops, generated volumes and the request itself
therefore all produce the same kind of number.

**Per axis.**  The material is a laminate.  Its porosity field decorrelates in
about 80 voxels through the thickness and in several hundred in plane - an
order of magnitude of anisotropy that a single isotropic correlation length
would average away, which is precisely the structure most likely to be lost.

A correlation length longer than the crop cannot be measured in the crop.  When
the curve does not cross 1/e inside the lags a field holds, the length is
reported as ``None`` and the correlation at fixed lags is reported instead, so
an in-plane row is never read as a decorrelation that did not happen.  Every row
carries its own reach.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from poregen.eval_v4 import metrics as M
from poregen.eval_v4.io import LABEL_PORE, TILE

#: Side of the analysis window, in voxels.  The conditioning tile, and T-D's
#: patch size - the window the real correlation lengths were measured on.
WINDOW = TILE
#: Step between windows, in voxels.  T-D's patch stride.  Half the window, so
#: the 1/e crossing of an 80-voxel correlation length lands between resolvable
#: lags instead of beyond the first one.
STRIDE = 32
#: A window holding less than this share of material is dropped rather than
#: measured against a denominator that is mostly air - the same rule
#: :func:`poregen.eval_v4.metrics.local_obedience` applies per tile.
MIN_MATERIAL_FRAC = 0.5

AXES = ("z", "y", "x")

#: Lags the correlation is quoted at whatever the crop reaches, in voxels.  An
#: axis whose field never falls to 1/e still has a comparable number here, and
#: these lags are voxel-disjoint (>= the 64-voxel window), so they carry spatial
#: signal rather than shared voxels.
FIXED_LAGS_VOX = (64, 128, 192, 384)

#: Quantiles of the marginal that go in the results file.
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: ``(assessment, notes key, value)`` of the generated cases whose request is
#: the SAMPLED coherent field (:func:`poregen.eval_v4.cases.field_coherent`)
#: over a full box.  The sphere and rough-surface cases carry a coherent field
#: too and are left out on purpose: their envelope cuts most of the windows, so
#: what a field measured there describes is the shape of the request.
COHERENT_SOURCES = (
    ("porosity_local", "field", "coherent"),
    ("multichunk", "request", "box"),
)

#: The real crops the delivered fields are read against - the two shapes
#: ``real_floor`` cuts AT the generated shapes.  The ``micro`` references are a
#: matched-porosity PAIR built for assessment 8; they are selected for their
#: porosity, so their heterogeneity is not a sample of the material's.
REAL_SHAPE_TAGS = ("small", "large")


# ---------------------------------------------------------------------------
# The delivered field
# ---------------------------------------------------------------------------

def _box_sum(b: np.ndarray, k: int) -> np.ndarray:
    """Sum over every ``k``-cubed box of a 3-D grid, by running sums per axis."""
    out = b
    for axis in range(out.ndim):
        n = out.shape[axis]
        if n < k:
            raise ValueError(
                f"axis {axis} holds {n} steps, fewer than the {k} a window spans."
            )
        c = np.cumsum(out, axis=axis)
        head = np.take(c, [k - 1], axis=axis)
        tail = (np.take(c, np.arange(k, n), axis=axis)
                - np.take(c, np.arange(0, n - k), axis=axis))
        out = np.concatenate([head, tail], axis=axis)
    return out


def delivered_field(
    label: np.ndarray,
    material: np.ndarray,
    *,
    window: int = WINDOW,
    stride: int = STRIDE,
    min_material_frac: float = MIN_MATERIAL_FRAC,
) -> np.ndarray:
    """Material porosity per sliding window, ``NaN`` where there is too little
    material.

    Pore voxels over MATERIAL voxels, matching the conditioning the model was
    given and the requested field it was given them as.  ``window`` must be a
    whole number of ``stride`` steps: the window sums are then built from the
    stride-grid block sums rather than from a summed-area table over the volume,
    which for a 1024-cubed real crop is the difference between a few megabytes
    and a gigabyte.
    """
    if window % stride:
        raise ValueError(f"window {window} is not a whole number of strides {stride}.")
    if label.shape != material.shape:
        raise ValueError(f"label {label.shape} and material {material.shape} differ.")
    pore = M.block_sum((label == LABEL_PORE) & material, stride)
    mat = M.block_sum(material, stride)
    k = window // stride
    pore_w = _box_sum(pore, k)
    mat_w = _box_sum(mat, k)
    usable = mat_w > min_material_frac * window ** 3
    field = np.full(mat_w.shape, np.nan)
    np.divide(pore_w, mat_w, out=field, where=usable)
    return field


# ---------------------------------------------------------------------------
# The two statistics
# ---------------------------------------------------------------------------

def marginal_stats(values) -> dict:
    """Mean, spread and quantiles of the local porosity a field holds."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, "mean": None, "sd": None, "cv": None,
                "min": None, "max": None, "quantiles": {}}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {
        "n": int(v.size),
        "mean": mean,
        "sd": sd,
        "cv": (sd / mean) if mean > 0 else None,
        "min": float(v.min()),
        "max": float(v.max()),
        "quantiles": {f"{q:g}": float(np.quantile(v, q)) for q in QUANTILES},
    }


def axis_correlations(fields, *, stride: int = STRIDE,
                      min_pairs: int = M.MIN_LAG_PAIRS) -> dict:
    """Correlation curve and 1/e length per axis, in voxels, pooled over fields.

    ``reach_vox`` is the largest lag that had enough pairs to report.  A
    ``corr_length_vox`` of ``None`` beside a reach shorter than the real
    material's length on that axis is not a failure of the field; it is the crop
    being too small to see the answer, and the report has to say so.
    """
    fields = [np.asarray(f, float) for f in fields]
    out: dict = {}
    for ai, name in enumerate(AXES):
        limit = min(f.shape[ai] for f in fields) - 1
        if limit < 1:
            out[name] = {"corr_length_vox": None, "reach_vox": 0, "lag_vox": [],
                         "correlation": [], "n_pairs": [], "r_at_lag_vox": {}}
            continue
        curve = M.lag_correlation(fields, ai, limit, min_pairs=min_pairs)
        lag_vox = curve["lag"] * stride
        r = curve["r"]
        good = np.isfinite(r)
        out[name] = {
            "corr_length_vox": M.correlation_length_1_over_e(lag_vox, r),
            "reach_vox": int(lag_vox[good].max()) if good.any() else 0,
            "lag_vox": lag_vox.tolist(),
            "correlation": r.tolist(),
            "n_pairs": curve["n_pairs"].tolist(),
            "r_at_lag_vox": {
                str(lag): (float(r[lag // stride])
                           if lag % stride == 0 and lag // stride < len(r)
                           and np.isfinite(r[lag // stride]) else None)
                for lag in FIXED_LAGS_VOX
            },
        }
    return out


def anisotropy(per_axis: dict) -> dict:
    """In-plane over through-thickness correlation length.

    ``None`` when either axis has no measurable length - a ratio built from a
    missing crossing would be a number with no content.
    """
    z = per_axis.get("z", {}).get("corr_length_vox")
    out = {}
    for name in ("y", "x"):
        v = per_axis.get(name, {}).get("corr_length_vox")
        ok = v is not None and z is not None and z > 0
        out[f"{name}_over_z"] = float(v / z) if ok else None
    return out


def field_statistics(field: np.ndarray, *, window: int = WINDOW,
                     stride: int = STRIDE) -> dict:
    """Both statistics of ONE field, plus the geometry they were taken on."""
    per_axis = axis_correlations([field], stride=stride)
    return {
        "window_vox": int(window),
        "stride_vox": int(stride),
        "field_grid": list(field.shape),
        "n_windows": int(np.isfinite(field).sum()),
        "marginal": marginal_stats(field),
        "per_axis": per_axis,
        "anisotropy": anisotropy(per_axis),
    }


# ---------------------------------------------------------------------------
# Comparing two marginals
# ---------------------------------------------------------------------------

def marginal_distance(a, b) -> dict:
    """Distance between two samples of local porosity.

    ``w1`` is on the porosities themselves and answers the question as asked.
    ``w1_ratio`` first divides each sample by its own mean, which removes the
    global porosity the two sets happen to sit at and leaves the SHAPE of the
    heterogeneity - the ratio distribution the coherent field is drawn from.  A
    generated set at a different global level would otherwise score a distance
    that says nothing about its field.
    """
    a = np.asarray(a, float).ravel(); a = a[np.isfinite(a)]
    b = np.asarray(b, float).ravel(); b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return {"n_a": int(a.size), "n_b": int(b.size),
                "w1": None, "w1_ratio": None, "ks": None}
    out = {
        "n_a": int(a.size), "n_b": int(b.size),
        "w1": float(wasserstein_distance(a, b)),
        "ks": float(ks_2samp(a, b).statistic),
        "w1_ratio": None,
    }
    if a.mean() > 0 and b.mean() > 0:
        out["w1_ratio"] = float(wasserstein_distance(a / a.mean(), b / b.mean()))
    return out


def corr_length_gap(gen: dict, real: dict) -> dict:
    """Per-axis generated-minus-real correlation length, and their ratio.

    Only reported where BOTH sides measured a crossing and both reached at least
    as far as the larger of the two lengths.  Comparing a length found inside a
    192-voxel crop with one found inside a 1024-voxel crop is the mistake this
    guard exists to prevent.
    """
    out = {}
    for name in AXES:
        g, r = gen.get(name, {}), real.get(name, {})
        lg, lr = g.get("corr_length_vox"), r.get("corr_length_vox")
        reach = min(g.get("reach_vox") or 0, r.get("reach_vox") or 0)
        if lg is None or lr is None or max(lg, lr) > reach:
            out[name] = {"generated_vox": lg, "real_vox": lr, "common_reach_vox": reach,
                         "difference_vox": None, "ratio": None}
            continue
        out[name] = {
            "generated_vox": lg, "real_vox": lr, "common_reach_vox": reach,
            "difference_vox": float(lg - lr),
            "ratio": float(lg / lr) if lr > 0 else None,
        }
    return out

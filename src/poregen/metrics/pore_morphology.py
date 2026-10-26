"""Per-pore morphology: volume, Crofton surface area, sphericity, inertia and
covariance tensors, shape descriptors and orientation, distance to the surface.

Author's request, 2026-10-26: the property set of an external defect-analysis
repo (its sections 4-7), computed on real coupons and on generated volumes so
the two can be compared per porosity bin. The formulas below are that repo's,
re-implemented here; where the repo is ambiguous the choice is stated.

A PORE is a 26-connected component of ``label == 1`` (the segmentation as is:
no minimum size; ``n_voxels`` is kept on every row so a reader filters by size).
Coordinates are (x, y, z) = (axis 2, axis 1, axis 0) of the (Z, Y, X) array,
in micrometres; z is the build / through-thickness axis.

Every per-pore quantity is computed in ONE pass over the pore voxels with
``np.bincount`` sums, not a Python loop over pores; a coupon holds 10^4-10^5
pores.

Columns (units in the name; ``s1 >= s2 >= s3`` are the eigenvalues of the
source in question):

- ``n_voxels``, ``volume_um3`` = n * v^3, ``r_eq_um`` = (3V / 4 pi)^(1/3)
- ``area_um2``: Crofton-style. For each axis, count 0/1 transitions of the
  pore's own mask padded by one voxel; ``A_vox = (4/pi)(n_z + n_y + n_x)``.
  A face shared with ANOTHER pore cannot occur (face neighbours are the same
  component), so the transitions are counted on the label array directly.
- ``sphericity`` = pi^(1/3) (6V)^(2/3) / A, clamped to <= 1
- ``area_corr_um2`` / ``sphericity_corr``: the same with the 3-D Crofton
  factor 2/3 instead of 4/pi. The repo's 4/pi is the 2-D perimeter rule
  inverted (2-D: P ~ (pi/4) n). In 3-D an isotropic surface crosses (3/2) A
  voxel faces, so A ~ (2/3) n. With 4/pi a digital sphere of radius 20 scores
  sphericity 0.52 and A 1.9x too large; with 2/3 it scores ~0.99
  (tests/test_pore_morphology.py). Both are kept: the repo's for comparability,
  the corrected one for absolute values. The clamp at 1 is applied to both.
- ``bary_{x,y,z}_um``: barycentre
- inertia, per voxel (second moment, um^2), primed = centred:
  ``Ixx = sum(y'^2+z'^2)/N``, ``Iyy = sum(x'^2+z'^2)/N``, ``Izz = sum(x'^2+y'^2)/N``,
  ``Ixy = -sum x'y'/N``, ``Ixz = -sum x'z'/N``, ``Iyz = -sum y'z'/N``;
  ``inertia_eig{1,2,3}`` DESCENDING and ``inertia_vec{k}_{x,y,z}``. The largest
  inertia eigenvalue belongs to the SHORT axis, so eigenvector 1 is the minor
  axis.
- covariance: ``np.cov`` convention (N-1), ``cov_eig{1,2,3}`` descending (um^2),
  ``cov_vec{k}_{x,y,z}``; eigenvector 1 is the LONG axis. NaN for N = 1.
- shape descriptors, once per source (``cov_`` / ``inertia_`` prefix):
  ``anisotropy = 1 - s3/s1``, ``elongation = s2/s1``, ``flatness = s3/s2``,
  ``elongation_avg = 2 s1/(s2+s3)``, ``flatness_avg = s2/s3``,
  ``phi_deg = arccos(|v1_z|)`` (0-90, from the z axis),
  ``theta_deg = atan2(v1_y, v1_x)`` (-180..180, sign-ambiguous by 180).
  For the covariance source elongation and flatness are SQUARED axis ratios.
- ``dist_surface_um``: shortest Euclidean distance from any voxel of the pore
  to the nearest AIR voxel (``label == 2``), via one EDT of the non-air region;
  NaN when the volume holds no air.
- ``touches_air``: a face neighbour of the pore is air. ``touches_edge``: the
  pore's bounding box reaches the array boundary (truncated by the crop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

PORE, AIR = 1, 2
STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def _eig_desc(t: np.ndarray, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues descending and eigenvectors as rows, for a stack (n, 3, 3)."""
    w, v = np.linalg.eigh(t)  # ascending, columns
    order = np.argsort(-w, axis=1)
    w = np.take_along_axis(w, order, axis=1)
    v = np.take_along_axis(v, order[:, None, :], axis=2)  # columns reordered
    return w, np.transpose(v, (0, 2, 1))  # (n, k, xyz)


def _descriptors(prefix: str, w: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    s1, s2, s3 = w[:, 0], w[:, 1], w[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            f"{prefix}anisotropy": 1.0 - s3 / s1,
            f"{prefix}elongation": s2 / s1,
            f"{prefix}flatness": s3 / s2,
            f"{prefix}elongation_avg": 2.0 * s1 / (s2 + s3),
            f"{prefix}flatness_avg": s2 / s3,
        }
    v1 = v[:, 0, :]
    out[f"{prefix}phi_deg"] = np.degrees(np.arccos(np.clip(np.abs(v1[:, 2]), 0.0, 1.0)))
    out[f"{prefix}theta_deg"] = np.degrees(np.arctan2(v1[:, 1], v1[:, 0]))
    return out


def _transitions(labels: np.ndarray, n: int) -> np.ndarray:
    """Per-pore count of 0/1 transitions summed over the three axes, with a
    one-voxel zero pad (the array edge counts as a transition)."""
    counts = np.zeros(n + 1, dtype=np.int64)
    for ax in range(3):
        a = np.moveaxis(labels, ax, 0)
        # edges of the array: the pad makes every boundary voxel a transition
        for sl in (a[0], a[-1]):
            counts += np.bincount(sl.ravel(), minlength=n + 1)
        for i in range(a.shape[0] - 1):
            lo, hi = a[i], a[i + 1]
            d = lo != hi
            if d.any():
                counts += np.bincount(lo[d], minlength=n + 1)
                counts += np.bincount(hi[d], minlength=n + 1)
    counts[0] = 0
    return counts[1:]


def _dist_to_air(air: np.ndarray, voxel_um: float, zz, yy, xx,
                 block: int = 512, margin: int = 160) -> np.ndarray:
    """EDT of the non-air region, read at the given voxels, in y-blocks.

    A whole-coupon EDT needs ~30 GB (scipy keeps a float64 map and an int32
    index map of the full array). Each y-block is computed with ``margin``
    voxels of context on both sides, so the result is EXACT wherever the
    nearest air lies within ``margin`` voxels in y; beyond that it can only
    over-read, because air farther than the margin in y is not seen.
    ``margin`` 160 vox = 4 mm is more than a coupon's half-thickness
    (<= 120 vox) and every coupon has air on its z faces, so in practice the
    nearest air is always inside the block. A block window with no air at all
    gives NaN. Checked against the full EDT in the test suite.
    """
    out = np.empty(len(zz), dtype=np.float32)
    H = air.shape[1]
    order = np.argsort(yy, kind="stable")
    ys = yy[order]
    for y0 in range(0, H, block):
        y1 = min(y0 + block, H)
        lo, hi = np.searchsorted(ys, y0), np.searchsorted(ys, y1)
        if lo == hi:
            continue
        a0, a1 = max(y0 - margin, 0), min(y1 + margin, H)
        idx = order[lo:hi]
        if not air[:, a0:a1].any():
            out[idx] = np.nan
            continue
        edt = ndimage.distance_transform_edt(~air[:, a0:a1], sampling=voxel_um)
        out[idx] = edt[zz[idx], yy[idx] - a0, xx[idx]]
        del edt
    return out


def measure(label: np.ndarray, voxel_um: float) -> tuple[pd.DataFrame, dict]:
    """Per-pore table and a volume summary for one (Z, Y, X) 3-class label."""
    label = np.asarray(label)
    pore = label == PORE
    air = label == AIR
    labels, n = ndimage.label(pore, structure=STRUCT26)
    labels = labels.astype(np.int32, copy=False)
    n_material = int((~air).sum())
    summary = {
        "shape": list(label.shape),
        "voxel_um": voxel_um,
        "n_pores": int(n),
        "pore_voxels": int(pore.sum()),
        "air_voxels": int(air.sum()),
        "nonair_voxels": n_material,
        "porosity": float(pore.sum() / n_material) if n_material else float("nan"),
    }
    if n == 0:
        return pd.DataFrame(), summary

    zz, yy, xx = np.nonzero(labels)
    lab = labels[zz, yy, xx]
    del pore
    x = xx.astype(np.float64) * voxel_um
    y = yy.astype(np.float64) * voxel_um
    z = zz.astype(np.float64) * voxel_um
    N = np.bincount(lab, minlength=n + 1)[1:].astype(np.float64)

    def s(w):  # per-pore sum
        return np.bincount(lab, weights=w, minlength=n + 1)[1:]

    mx, my, mz = s(x) / N, s(y) / N, s(z) / N
    cx, cy, cz = x - mx[lab - 1], y - my[lab - 1], z - mz[lab - 1]
    sxx, syy, szz = s(cx * cx), s(cy * cy), s(cz * cz)
    sxy, sxz, syz = s(cx * cy), s(cx * cz), s(cy * cz)
    del cx, cy, cz, x, y, z

    inertia = np.empty((n, 3, 3))
    inertia[:, 0, 0] = (syy + szz) / N
    inertia[:, 1, 1] = (sxx + szz) / N
    inertia[:, 2, 2] = (sxx + syy) / N
    inertia[:, 0, 1] = inertia[:, 1, 0] = -sxy / N
    inertia[:, 0, 2] = inertia[:, 2, 0] = -sxz / N
    inertia[:, 1, 2] = inertia[:, 2, 1] = -syz / N
    with np.errstate(divide="ignore", invalid="ignore"):
        dof = np.where(N > 1, N - 1, np.nan)[:, None, None]
    cov = np.empty((n, 3, 3))
    cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = sxx, syy, szz
    cov[:, 0, 1] = cov[:, 1, 0] = sxy
    cov[:, 0, 2] = cov[:, 2, 0] = sxz
    cov[:, 1, 2] = cov[:, 2, 1] = syz
    cov = cov / dof
    single = N <= 1
    cov[single] = 0.0
    iw, iv = _eig_desc(inertia, True)
    cw, cv = _eig_desc(cov, True)
    cw[single] = np.nan
    cv[single] = np.nan

    vol = N * voxel_um ** 3
    r_eq = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    faces = _transitions(labels, n) * voxel_um ** 2
    area = (4.0 / np.pi) * faces
    area_corr = (2.0 / 3.0) * faces
    num = np.pi ** (1.0 / 3.0) * (6.0 * vol) ** (2.0 / 3.0)
    sph = np.minimum(num / area, 1.0)
    sph_corr = np.minimum(num / area_corr, 1.0)

    # distance to surface: EDT of the non-air region, min over each pore
    if air.any():
        d = _dist_to_air(air, voxel_um, zz, yy, xx)
        dmin = np.full(n + 1, np.inf)
        np.minimum.at(dmin, lab, d)
        dist = dmin[1:]
        touch_air = ndimage.binary_dilation(air, structure=ndimage.generate_binary_structure(3, 1))
        ta = np.zeros(n + 1, dtype=bool)
        ta[np.unique(labels[touch_air & (labels > 0)])] = True
        touches_air = ta[1:]
        del touch_air
    else:
        dist = np.full(n, np.nan)
        touches_air = np.zeros(n, dtype=bool)

    objs = ndimage.find_objects(labels)
    shp = labels.shape
    touches_edge = np.array([
        any(sl.start == 0 or sl.stop == shp[i] for i, sl in enumerate(o)) for o in objs
    ])

    cols = {
        "pore_id": np.arange(1, n + 1),
        "n_voxels": N.astype(np.int64),
        "volume_um3": vol,
        "r_eq_um": r_eq,
        "area_um2": area,
        "sphericity": sph,
        "area_corr_um2": area_corr,
        "sphericity_corr": sph_corr,
        "bary_x_um": mx, "bary_y_um": my, "bary_z_um": mz,
        "inertia_Ixx_um2": inertia[:, 0, 0], "inertia_Iyy_um2": inertia[:, 1, 1],
        "inertia_Izz_um2": inertia[:, 2, 2], "inertia_Ixy_um2": inertia[:, 0, 1],
        "inertia_Ixz_um2": inertia[:, 0, 2], "inertia_Iyz_um2": inertia[:, 1, 2],
        "dist_surface_um": dist,
        "touches_air": touches_air,
        "touches_edge": touches_edge,
    }
    for k in range(3):
        cols[f"inertia_eig{k + 1}_um2"] = iw[:, k]
        cols[f"cov_eig{k + 1}_um2"] = cw[:, k]
        for j, c in enumerate("xyz"):
            cols[f"inertia_vec{k + 1}_{c}"] = iv[:, k, j]
            cols[f"cov_vec{k + 1}_{c}"] = cv[:, k, j]
    cols.update(_descriptors("cov_", cw, cv))
    cols.update(_descriptors("inertia_", iw, iv))
    return pd.DataFrame(cols), summary

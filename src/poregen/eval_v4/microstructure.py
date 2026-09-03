"""The distribution statistics: does the generated material LOOK like real material?

Every other metric in this suite is *conditional* — it asks whether the volume
delivered what it was asked for.  These five ask the question the conditional
metrics cannot: whether the microstructure the model draws has the same
statistics as the material it was trained on.  A model can obey its porosity
request perfectly and still put the pores in the wrong shapes, at the wrong
spacing, with the wrong texture.

They are ported from ``scripts/eval_generated_volumes.py`` (deleted), which was
the only implementation of them, and they are the head-to-head numbers Naiff et
al. (*Computers & Geosciences* 206, 2026) report: FID on 2-D slices, a
Wasserstein-1 pore-size-distance and the two-point correlation.

**Every one of them needs a real-vs-real floor.**  A distribution distance has
no natural zero: two disjoint crops of the *same real panel* do not score 0
either, because both are finite samples.  What that pair scores is the floor,
and the generated number is only readable as a multiple of it.  That is why
:mod:`poregen.eval_v4.real_floor` cuts the reference crops in disjoint pairs.

Analysis geometry
-----------------
The generated volumes are 192 cubed; a real laminate holds no clean 192-deep
box, so the reference crops are 128 cubed.  Every statistic here is therefore
computed on geometry that both can supply:

* **S2** on 128-cubed windows — the FFT support, the Hann debias and the radial
  bin edges are then identical for both sets, which they would not be if the
  window followed the volume.  A 192-cubed volume contributes several
  overlapping windows; the overlap lowers the variance of the averaged curve
  and does not bias it.
* **PSD and Ripley's K** on the whole requested material — both are
  size-normalised (a distribution of diameters; K carries its own ``V/N**2``),
  so a bigger box is a bigger sample and not a different measurement.
* **FID** on 64x64 crops of 2-D slices, which is a per-crop measurement.
* **Memorisation** on 64-cubed patches, the size the VAE was trained on.

What changed in the port, and why
---------------------------------
* **The S2 W1 support is the radius in voxels**, not the bin index.  The two
  agree at the current 1-voxel bin width, but the old form silently changed
  units when ``n_bins`` changed.
* **Ripley's K is border-corrected.**  The old estimator counted pairs with no
  edge correction, which biases K low by a factor that grows with r, so it had
  no known value to be validated against — and this suite validates every
  measurement on data whose answer is known.  The reduced-sample (border)
  estimator used here converges to the CSR value ``(4/3)*pi*r**3``, which
  :mod:`tests.test_eval_v4_microstructure` checks on a Poisson process.
* **Connected components are 6-connected throughout.**  The old script used
  6-connectivity for the pore-size distribution (``ndimage.label``) and
  26-connectivity for Ripley and morphology (``skimage.measure.label``), so its
  two pore counts did not agree with each other.  At a median pore diameter of
  1.79 voxels, 26-connectivity fuses voids that meet at a single corner, so
  6-connectivity is the conservative reading and the one kept.
* **No pore subsampling.**  The old code capped Ripley at 5000 centroids
  because it built the full pairwise distance matrix.  A KD-tree removes the
  cap, so the estimate is over every pore rather than a random 5000.
* **The old ``ripley_w1`` is not ported.**  It was a Wasserstein distance
  between the two K-value vectors, which is a distance between two sets of
  numbers that happen to be K values — not a distance between two curves.  The
  scale-free curve distance :func:`log_ratio_distance` replaces it.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from poregen.eval_v4.io import LABEL_PORE
from poregen.eval_v4.manifest import Manifest, requires

logger = logging.getLogger(__name__)

# -- S2 ---------------------------------------------------------------------

#: Side of the cube every S2 curve is estimated on.  Fixed, so a 192-cubed
#: generated volume and a 128-cubed real crop are binned identically.
S2_WINDOW = 128
#: Stride between S2 windows.  Half a window, so a 192-cubed volume yields
#: 2x2x2 of them and a 128-cubed crop exactly one.
S2_STRIDE = S2_WINDOW // 2
#: Largest radius, in voxels.  The Hann debias is unreliable past ~0.4 of the
#: window (the window autocorrelation there is a small number in a
#: denominator), which is 51 voxels at a 128 window.
S2_R_MAX = 48
#: One bin per voxel of radius.
S2_N_BINS = S2_R_MAX
#: A window this far short of full material is not measured: S2 of a box that
#: is part exterior air measures the box, not the material.
S2_MIN_MATERIAL = 0.99

# -- Ripley -----------------------------------------------------------------

#: Largest radius for K(r), in voxels.  The border correction keeps only pores
#: further than r from every face, so r must stay well inside the 128-cubed
#: reference crop: at r = 24 the eligible core is still 80 cubed.
RIPLEY_R_MAX = 24
#: Below this many pores the estimate is noise, not a measurement.
RIPLEY_MIN_PORES = 20

# -- FID --------------------------------------------------------------------

#: Native-resolution crop side.  A full slice resized to 299 is a ~10x
#: downscale, which shrinks a 1.79-voxel pore to 0.18 pixels — the structure
#: the metric is supposed to see would be gone before Inception saw it.
FID_CROP = 64
FID_INPUT = 299
FID_AXES = ("axial", "coronal", "sagittal")
#: Crops per axis per set.  The feature is 2048-dimensional, so a covariance
#: estimated from fewer samples than that is singular; 5000 was the old
#: script's own guard and is kept.
FID_CROPS_PER_AXIS = 5000
FID_BATCH = 32
#: Named in the report, because an FID number means nothing without it.
FID_EXTRACTOR = (
    "torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT), "
    "ImageNet IMAGENET1K_V1, 2048-d pool3 (avgpool) features; 64x64 native "
    "crops in [0,1] replicated to 3 channels and bilinearly resized to 299x299"
)

# -- memorisation -----------------------------------------------------------

MEMO_PATCH = 64
#: Training latents sampled for the nearest-neighbour search.  The full store
#: is ~1.8 M patches; the distance to the nearest of a random 10 000 is an
#: upper bound on the distance to the nearest of all of them, and the same
#: bound is applied to the real crops, so the comparison is fair.
MEMO_TRAIN_SAMPLE = 10_000
MEMO_CHUNK = 256


# ---------------------------------------------------------------------------
# Two-point correlation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _hann_kernel(side: int) -> tuple[np.ndarray, np.ndarray]:
    """The 3-D Hann window and its own autocorrelation, for one window size.

    The window suppresses the spectral leakage a hard-edged box produces.
    Dividing the windowed autocorrelation by the window's own autocorrelation
    debiases it::

        S2_raw[r]     = sum_x f(x)w(x) f(x+r)w(x+r)  ~  S2(r) * win_auto[r]
        win_auto[r]   = sum_x w(x) w(x+r)

    so the ratio estimates S2(r), and S2(0) comes back as the phase fraction —
    which is what :func:`s2_radial` checks.
    """
    h = np.hanning(side)
    w = h[:, None, None] * h[None, :, None] * h[None, None, :]
    fw = np.fft.fftn(w)
    return w, np.real(np.fft.ifftn(fw * np.conj(fw)))


@functools.lru_cache(maxsize=4)
def _radial_bins(side: int, r_max: int, n_bins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Bin-centre radii and the flat index of every FFT lag in each shell."""
    f = np.fft.fftfreq(side) * side
    zz, yy, xx = np.meshgrid(f, f, f, indexing="ij")
    r_grid = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2).ravel()
    edges = np.linspace(0.0, float(r_max), n_bins + 1)
    idx = [np.flatnonzero((r_grid >= edges[i]) & (r_grid < edges[i + 1]))
           for i in range(n_bins)]
    return 0.5 * (edges[:-1] + edges[1:]), idx


def s2_radial(
    binary: np.ndarray,
    r_max: int = S2_R_MAX,
    n_bins: int = S2_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic S2(r) of one cubic window, by Hann-windowed FFT autocorrelation.

    S2(r) is the probability that two points a distance r apart are BOTH in the
    pore phase, so S2(0) is the phase fraction and S2(inf) is its square.
    """
    if binary.ndim != 3 or len({*binary.shape}) != 1:
        raise ValueError(
            f"s2_radial needs a cubic window, got shape {binary.shape}."
        )
    side = binary.shape[0]
    w, win_auto = _hann_kernel(side)
    f = np.fft.fftn(binary.astype(np.float64) * w)
    raw = np.real(np.fft.ifftn(f * np.conj(f)))
    auto = np.where(win_auto > 1e-10, raw / win_auto, 0.0).ravel()

    r_vals, shells = _radial_bins(side, r_max, n_bins)
    s2 = np.array([auto[i].mean() if i.size else np.nan for i in shells])
    return r_vals, s2


def analysis_windows(shape: tuple[int, int, int], side: int, stride: int) -> list[tuple[int, int, int]]:
    """Origins of every ``side``-cubed window on a ``stride`` grid, last one flush.

    Deterministic on purpose: the old script drew random crops, so re-running a
    measurement moved the number it produced.
    """
    if any(s < side for s in shape):
        raise ValueError(
            f"a {shape} volume is smaller than the {side}-cubed analysis window; "
            "every microstructure statistic is defined on that window."
        )

    def starts(n: int) -> list[int]:
        out = list(range(0, n - side + 1, stride))
        if out[-1] != n - side:
            out.append(n - side)
        return out

    return [(z, y, x) for z in starts(shape[0])
            for y in starts(shape[1]) for x in starts(shape[2])]


@requires()
def s2_profile(
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
) -> dict:
    """Mean S2(r) over every fully-material analysis window of one volume."""
    curves = []
    skipped = 0
    for z, y, x in analysis_windows(label.shape, S2_WINDOW, S2_STRIDE):
        sl = np.s_[z:z + S2_WINDOW, y:y + S2_WINDOW, x:x + S2_WINDOW]
        mat = material[sl]
        if mat.mean() < S2_MIN_MATERIAL:
            skipped += 1
            continue
        r_vals, s2 = s2_radial((label[sl] == LABEL_PORE) & mat)
        curves.append(s2)
    if not curves:
        raise ValueError(
            f"{manifest.assessment}/{manifest.case}: no {S2_WINDOW}-cubed window is "
            f"at least {S2_MIN_MATERIAL:.0%} requested material, so S2 has nothing "
            "to measure that is not part exterior air."
        )
    mean = np.mean(curves, axis=0)
    return {
        "r": r_vals.tolist(),
        "s2": mean.tolist(),
        "s2_zero_lag": float(mean[0]),
        "n_windows": len(curves),
        "n_windows_skipped": skipped,
        "window": S2_WINDOW,
    }


# ---------------------------------------------------------------------------
# Pore size distribution
# ---------------------------------------------------------------------------

def pore_diameters(pore: np.ndarray) -> np.ndarray:
    """Equivalent spherical diameter ``(6V/pi)**(1/3)`` per connected pore.

    6-connectivity: see the module docstring.  A single-voxel pore is 1.24
    voxels across on this definition, which is the floor of the distribution.
    """
    labels, n = ndimage.label(pore)
    if n == 0:
        return np.zeros(0, np.float64)
    vols = np.bincount(labels.ravel())[1:].astype(np.float64)
    return np.cbrt(6.0 * vols / np.pi)


@requires()
def psd_profile(
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
) -> dict:
    """The pore-size distribution of one volume, inside the requested material."""
    d = pore_diameters((label == LABEL_PORE) & material)
    if d.size == 0:
        return {"n_pores": 0, "diameters": d}
    return {
        "n_pores": int(d.size),
        "diameters": d,
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "max": float(d.max()),
    }


# ---------------------------------------------------------------------------
# Ripley's K
# ---------------------------------------------------------------------------

def pore_centroids(pore: np.ndarray) -> np.ndarray:
    """Centre of mass of every connected pore, ``(N, 3)`` in voxels."""
    labels, n = ndimage.label(pore)
    if n == 0:
        return np.zeros((0, 3), np.float64)
    return np.asarray(
        ndimage.center_of_mass(pore, labels, np.arange(1, n + 1)), np.float64
    ).reshape(n, 3)


def ripleys_k(
    points: np.ndarray,
    shape: tuple[int, int, int],
    r_max: int = RIPLEY_R_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Border-corrected Ripley's K(r) for points in a box.

    The reduced-sample (border) estimator::

        K(r) = sum_{i: b_i >= r} n_i(r) / (lambda * #{i: b_i >= r})

    where ``b_i`` is the distance from point i to the nearest face, ``n_i(r)``
    counts the OTHER points within r of i, and ``lambda = N / V``.  Only points
    that carry a complete r-ball inside the box contribute, so no pair is
    missed and the estimate is unbiased: under complete spatial randomness
    ``E[n_i(r)] = lambda * (4/3) * pi * r**3``, so K(r) converges to the volume
    of the ball.  That is the value the unit test checks.

    ``NaN`` is returned at any r where no point is far enough from every face —
    the honest answer, rather than a number computed from an empty sum.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415  (only this metric needs it)

    r_vals = np.arange(1, int(r_max) + 1, dtype=np.float64)
    n = len(points)
    volume = float(np.prod(shape))
    if n < 2:
        return r_vals, np.full(r_vals.shape, np.nan)

    lam = n / volume
    # Distance from each point to the nearest face: the smallest of its three
    # coordinates and of its three distances to the far faces.
    border = np.minimum(
        points.min(axis=1),
        (np.asarray(shape, np.float64) - points).min(axis=1),
    )
    tree = cKDTree(points)
    k = np.full(r_vals.shape, np.nan)
    for i, r in enumerate(r_vals):
        eligible = border >= r
        if not eligible.any():
            continue
        counts = tree.query_ball_point(points[eligible], r, return_length=True)
        # The tree holds every point, so each query counts the query point
        # itself; n_i(r) is the count of the OTHER points.
        k[i] = float((counts - 1).sum()) / (lam * float(eligible.sum()))
    return r_vals, k


def csr_k(r) -> np.ndarray:
    """K(r) of a 3-D Poisson process: the volume of a ball of radius r.

    In three dimensions that is ``(4/3)*pi*r**3``.  The two-dimensional value
    ``pi*r**2`` is the one usually quoted for Ripley's K and is NOT the right
    reference here — these are volumes.
    """
    r = np.asarray(r, np.float64)
    return (4.0 / 3.0) * np.pi * r ** 3


@requires()
def ripley_profile(
    label: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
) -> dict:
    """Ripley's K of the pore centroids of one volume.

    ``K(r) / csr_k(r)`` above 1 means the pores cluster more than a Poisson
    process of the same intensity; below 1 means they are more regular.
    """
    pts = pore_centroids((label == LABEL_PORE) & material)
    shape = tuple(int(s) for s in label.shape)
    if len(pts) < RIPLEY_MIN_PORES:
        return {"n_pores": int(len(pts)), "available": False,
                "reason": f"fewer than {RIPLEY_MIN_PORES} pores"}
    r, k = ripleys_k(pts, shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = k / csr_k(r)
    return {
        "available": True,
        "n_pores": int(len(pts)),
        "r": r.tolist(),
        "k": k.tolist(),
        "k_over_csr": ratio.tolist(),
        "intensity_per_voxel": float(len(pts)) / float(np.prod(shape)),
    }


# ---------------------------------------------------------------------------
# Curve distances
# ---------------------------------------------------------------------------

def curve_w1(r, a, b) -> float:
    """Wasserstein-1 between two non-negative curves over the same support.

    Each curve is clipped at zero and normalised to unit mass, then compared as
    a distribution over ``r``.  The result is in VOXELS: it is how far the mass
    of one curve has to move along the radius axis to become the other.
    Normalising discards the amplitude — S2(0) is the phase fraction, which the
    porosity metrics already report — so this is a distance between SHAPES.

    ``NaN`` when either curve carries no mass.
    """
    r = np.asarray(r, np.float64)
    a = np.clip(np.asarray(a, np.float64), 0.0, None)
    b = np.clip(np.asarray(b, np.float64), 0.0, None)
    ok = np.isfinite(a) & np.isfinite(b)
    r, a, b = r[ok], a[ok], b[ok]
    if a.sum() < 1e-30 or b.sum() < 1e-30:
        return float("nan")
    return float(wasserstein_distance(r, r, a / a.sum(), b / b.sum()))


def log_ratio_distance(a, b) -> float:
    """Mean ``|log(a/b)|`` over the support — a scale-free curve distance.

    Used for Ripley's K, whose values span three orders of magnitude across the
    r range, so an absolute difference would be a report on the largest r alone.
    0 means the two curves agree everywhere; 0.69 means they differ by a factor
    of two on average.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not ok.any():
        return float("nan")
    return float(np.abs(np.log(a[ok] / b[ok])).mean())


def psd_w1(a, b) -> float:
    """Wasserstein-1 between two pooled pore-diameter samples, in voxels."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


# ---------------------------------------------------------------------------
# FID on 2-D slices
# ---------------------------------------------------------------------------

@requires()
def fid_crops(
    xct_u8: np.ndarray,
    material: np.ndarray,
    *,
    manifest: Manifest,
    axis: str,
    n_crops: int,
    seed: int,
) -> np.ndarray:
    """``(n, 64, 64)`` float32 crops in [0, 1] from slices along one axis.

    Only crops that lie entirely inside the requested material are kept: a crop
    of exterior air is a picture of the specimen's outside, and its Inception
    features would say more about the background than the microstructure.  The
    RNG is local and seeded from the case, so the crop set is reproducible.
    """
    if axis not in FID_AXES:
        raise ValueError(f"axis must be one of {FID_AXES}, got {axis!r}")
    ax = FID_AXES.index(axis)
    rng = np.random.default_rng(seed)
    n_slices = xct_u8.shape[ax]
    h, w = [s for a, s in enumerate(xct_u8.shape) if a != ax]
    if h < FID_CROP or w < FID_CROP:
        raise ValueError(
            f"a {xct_u8.shape} volume has no {FID_CROP}x{FID_CROP} {axis} crop."
        )

    out = np.empty((n_crops, FID_CROP, FID_CROP), np.float32)
    got, attempts, budget = 0, 0, n_crops * 20
    while got < n_crops:
        attempts += 1
        if attempts > budget:
            raise ValueError(
                f"{manifest.assessment}/{manifest.case}: only {got} of {n_crops} "
                f"{axis} crops landed fully inside the requested material after "
                f"{budget} tries; the material envelope is too small for FID."
            )
        i = int(rng.integers(n_slices))
        y0 = int(rng.integers(h - FID_CROP + 1))
        x0 = int(rng.integers(w - FID_CROP + 1))
        sl = np.s_[y0:y0 + FID_CROP, x0:x0 + FID_CROP]
        if not np.take(material, i, axis=ax)[sl].all():
            continue
        out[got] = np.take(xct_u8, i, axis=ax)[sl].astype(np.float32) / 255.0
        got += 1
    return out


def inception_features(crops: np.ndarray, device=None) -> np.ndarray:
    """2048-d pool3 features for ``(n, 64, 64)`` crops.  See :data:`FID_EXTRACTOR`.

    Raises :class:`ModuleNotFoundError` when torchvision is absent; the caller
    turns that into a reported skip rather than a failed measurement, because
    torchvision is an environment choice and the other four statistics do not
    need it.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    import torchvision.models as tvm  # noqa: PLC0415

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)
    net.eval().to(device)

    pool: list = []
    handle = net.avgpool.register_forward_hook(
        lambda m, i, o: pool.append(o.detach().flatten(1).cpu())
    )
    feats: list = []
    try:
        with torch.no_grad():
            for s in range(0, len(crops), FID_BATCH):
                t = torch.from_numpy(crops[s:s + FID_BATCH]).unsqueeze(1)
                t = F.interpolate(t, size=(FID_INPUT, FID_INPUT),
                                  mode="bilinear", align_corners=False)
                pool.clear()
                net(t.expand(-1, 3, -1, -1).to(device))
                feats.append(pool[-1])
    finally:
        handle.remove()
    return torch.cat(feats).numpy().astype(np.float64)


def frechet_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    """FID between two feature sets: ``|mu_a - mu_b|^2 + tr(Sa + Sb - 2*sqrt(Sa Sb))``.

    The matrix square root of a product of two estimated covariances is
    routinely a hair off the real axis; the imaginary part is dropped only
    after checking it is numerical noise, because a genuinely complex result
    would mean one of the covariances is not positive semi-definite and the
    number would be meaningless.
    """
    from scipy import linalg  # noqa: PLC0415

    if len(fa) < 2 or len(fb) < 2:
        return float("nan")
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    sa, sb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    covmean = linalg.sqrtm(sa @ sb)
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            eps = 1e-6 * np.eye(sa.shape[0])
            covmean = linalg.sqrtm((sa + eps) @ (sb + eps))
        covmean = np.real(covmean)
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(sa + sb - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------

def train_latent_mu(
    latents_root: str | Path,
    n_sample: int = MEMO_TRAIN_SAMPLE,
    seed: int = 0,
) -> np.ndarray:
    """``(n, C*d*h*w)`` posterior means of a random sample of TRAIN latents.

    Reads the current store layout: ``metadata.json`` at the root, and per split
    a ``latents.bin`` of ``(N, 2C, d, h, w)`` float16 packed ``mu_then_std``
    beside an ``index.parquet`` that gives N.  Only the mu half is read — the
    posterior width says nothing about which patch a latent is.
    """
    import json  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    root = Path(latents_root)
    meta = json.loads((root / "metadata.json").read_text())
    storage = meta["storage"]
    if storage["pack_scheme"] != "mu_then_std":
        raise ValueError(
            f"{root}: pack_scheme is {storage['pack_scheme']!r}; this reader knows "
            "'mu_then_std' — channels 0..C-1 are the posterior mean."
        )
    c, *spatial = (int(v) for v in meta["latent_shape"])
    n = len(pd.read_parquet(str(root / "train" / "index.parquet")))
    store = np.memmap(
        str(root / "train" / "latents.bin"), dtype=np.dtype(storage["dtype"]),
        mode="r", shape=(n, 2 * c, *spatial),
    )
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(n, size=min(int(n_sample), n), replace=False))
    return np.asarray(store[take, :c], np.float32).reshape(len(take), -1)


def encode_volume_patches(
    vae,
    xct_u8: np.ndarray,
    label_u8: np.ndarray,
    material: np.ndarray,
    device=None,
    patch: int = MEMO_PATCH,
    batch: int = 4,
) -> np.ndarray:
    """``(m, C*d*h*w)`` posterior means of every whole-material patch of a volume.

    The patches are the non-overlapping ``patch``-cubed tiling, which is the
    grid the store was built on, so a generated patch and a training latent are
    the same kind of object and their distance means something.

    The encoder inputs come from the model's own ``encoder_inputs`` declaration
    through :func:`poregen.training.engine.encoder_input_keys`, and the moments
    from ``encode_moments`` — the same two entry points
    ``scripts/build_latent_dataset.py`` used to write the store.  The r08 VAE
    encodes the grey volume AND the 3-class label; feeding it the grey alone
    would produce latents from a different function to the ones in the store.
    """
    import torch  # noqa: PLC0415

    from poregen.training.engine import encoder_input_keys  # noqa: PLC0415

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = encoder_input_keys(vae)
    unknown = set(keys) - {"xct", "label"}
    if unknown:
        raise ValueError(
            f"the VAE declares encoder inputs {keys}; this check can supply only "
            f"'xct' and 'label' from a generated case, not {sorted(unknown)}."
        )

    d, h, w = xct_u8.shape
    origins = [
        (z, y, x)
        for z in range(0, d - patch + 1, patch)
        for y in range(0, h - patch + 1, patch)
        for x in range(0, w - patch + 1, patch)
        if material[z:z + patch, y:y + patch, x:x + patch].all()
    ]
    if not origins:
        return np.zeros((0, 0), np.float32)

    out = []
    with torch.no_grad():
        for s in range(0, len(origins), batch):
            sl = [np.s_[z:z + patch, y:y + patch, x:x + patch]
                  for z, y, x in origins[s:s + batch]]
            arrays = {
                "xct": torch.from_numpy(
                    np.stack([xct_u8[i] for i in sl]).astype(np.float32) / 255.0
                ).unsqueeze(1),
                "label": torch.from_numpy(
                    np.stack([label_u8[i] for i in sl]).astype(np.int64)
                ),
            }
            mu, _ = vae.encode_moments(*(arrays[k].to(device) for k in keys))
            out.append(mu.flatten(1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def nearest_neighbour_distance(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Min L2 distance from every row of ``query`` to any row of ``bank``."""
    import torch  # noqa: PLC0415

    if query.shape[1] != bank.shape[1]:
        raise ValueError(
            f"latent dimensions differ: {query.shape[1]} from the VAE against "
            f"{bank.shape[1]} in the store. The store was built with a different "
            "VAE, so a distance between them would be meaningless."
        )
    q = torch.from_numpy(query)
    b = torch.from_numpy(bank)
    out = [torch.cdist(q[s:s + MEMO_CHUNK], b).min(dim=1).values
           for s in range(0, len(q), MEMO_CHUNK)]
    return torch.cat(out).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# One volume's profile, and the two-sample comparison
# ---------------------------------------------------------------------------

@dataclass
class VolumeProfile:
    """Everything the set-level distances need from one volume."""

    case: str
    phi: float
    s2_r: np.ndarray
    s2: np.ndarray
    diameters: np.ndarray
    ripley_r: np.ndarray
    ripley_k: np.ndarray
    n_pores: int
    group: str = ""

    def summary(self) -> dict:
        """The JSON-safe part: curves and counts, never the raw diameters."""
        d = self.diameters
        return {
            "case": self.case,
            "group": self.group,
            "phi": self.phi,
            "n_pores": self.n_pores,
            "s2_r": self.s2_r.tolist(),
            "s2": self.s2.tolist(),
            "s2_zero_lag": float(self.s2[0]),
            "ripley_r": self.ripley_r.tolist(),
            "ripley_k": self.ripley_k.tolist(),
            "psd_median": float(np.median(d)) if d.size else None,
            "psd_p90": float(np.percentile(d, 90)) if d.size else None,
            "psd_mean": float(d.mean()) if d.size else None,
        }


def profile_volume(case, *, group: str = "") -> VolumeProfile:
    """Run the three array statistics over one :class:`poregen.eval_v4.io.Case`."""
    label, material, m = case.label, case.material_voxels(), case.manifest
    s2 = s2_profile(label, material, manifest=m)
    psd = psd_profile(label, material, manifest=m)
    rip = ripley_profile(label, material, manifest=m)
    return VolumeProfile(
        case=m.case,
        group=group,
        phi=float((label[material] == LABEL_PORE).mean()),
        s2_r=np.asarray(s2["r"], np.float64),
        s2=np.asarray(s2["s2"], np.float64),
        diameters=np.asarray(psd.get("diameters", np.zeros(0)), np.float64),
        ripley_r=np.asarray(rip.get("r", np.arange(1, RIPLEY_R_MAX + 1)), np.float64),
        ripley_k=np.asarray(rip.get("k", np.full(RIPLEY_R_MAX, np.nan)), np.float64),
        n_pores=int(psd["n_pores"]),
    )


def mean_curve(profiles: list[VolumeProfile], attr: str) -> np.ndarray:
    """Mean of one curve over a set of volumes, ignoring volumes that lack it."""
    stack = np.vstack([getattr(p, attr) for p in profiles])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def compare_sets(a: list[VolumeProfile], b: list[VolumeProfile]) -> dict:
    """The three array statistics between two SETS of volumes.

    ``a`` is the set under test and ``b`` the reference.  Every distance is
    symmetric, so the same function measures a generated set against real
    material and one half of the real material against the other half — which
    is the only way the first number can be read.
    """
    if not a or not b:
        raise ValueError("compare_sets needs a non-empty set on both sides.")
    s2_a, s2_b = mean_curve(a, "s2"), mean_curve(b, "s2")
    k_a, k_b = mean_curve(a, "ripley_k"), mean_curve(b, "ripley_k")
    r_s2, r_k = a[0].s2_r, a[0].ripley_r
    diam_a = np.concatenate([p.diameters for p in a]) if a else np.zeros(0)
    diam_b = np.concatenate([p.diameters for p in b]) if b else np.zeros(0)
    return {
        "n_a": len(a), "n_b": len(b),
        "s2_w1": curve_w1(r_s2, s2_a, s2_b),
        "s2_zero_lag_a": float(s2_a[0]), "s2_zero_lag_b": float(s2_b[0]),
        "psd_w1": psd_w1(diam_a, diam_b),
        "psd_median_a": float(np.median(diam_a)) if diam_a.size else None,
        "psd_median_b": float(np.median(diam_b)) if diam_b.size else None,
        "psd_n_a": int(diam_a.size), "psd_n_b": int(diam_b.size),
        "ripley_log_ratio": log_ratio_distance(k_a, k_b),
        "ripley_k_over_csr_a": float(np.nanmean(k_a / csr_k(r_k))),
        "ripley_k_over_csr_b": float(np.nanmean(k_b / csr_k(r_k))),
        "curves": {
            "s2_r": r_s2.tolist(), "s2_a": s2_a.tolist(), "s2_b": s2_b.tolist(),
            "ripley_r": r_k.tolist(), "k_a": k_a.tolist(), "k_b": k_b.tolist(),
        },
    }


def ratio(measured: float | None, floor: float | None) -> float | None:
    """``measured / floor`` — how many real-vs-real distances the model is away.

    1.0 means the generated set is as close to real material as two disjoint
    crops of the same real panel are to each other, which is as close as this
    measurement can tell.  ``None`` when either side is missing, and ``None``
    when the floor is zero: a ratio to zero is not a large number, it is an
    undefined one.
    """
    if measured is None or floor is None:
        return None
    if not np.isfinite(measured) or not np.isfinite(floor) or floor <= 0:
        return None
    return float(measured / floor)

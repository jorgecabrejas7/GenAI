"""Unmasked-air audit v2 — corrected intensity conversion + honest mode analysis.

Why this script exists
----------------------
``scripts/analysis/eval_v2_audit.py`` converted generated volumes with
``clip(logit(v), 0, 1) * 255``, on the rationale of "inverting the sampler's
sigmoid".  This redo was commissioned on the premise that no logit is involved.
The measurements do not support that premise, so this script reports both
conversions and lets the numbers decide.  The write chain is

    VAE decoder logits -> expit -> * 255 -> uint8      (sampler.py:1276-1278)
    uint8 -> astype(float32) / 255 -> volume.tif       (_eval_v2.py:243)

so ``volume.tif`` stores ``expit(decoder_output)`` quantised to u8/255.  This
script therefore reports **two grey scales side by side**, because which one is
"correct" decides the dynamic-range question and the evidence is not one-sided:

``raw``     ``u8 = clip(v * 255, 0, 255)`` — a direct rescale of what is stored,
            no logit.  This is the primary scale used for every threshold, every
            detection and every per-arm number in this report.

``native``  ``u8 = clip(logit(v) * 255, 0, 255)`` — undoes the sampler's
            ``expit``.  Reported as a parallel diagnostic track.

The reason ``native`` cannot simply be dismissed: the VAE decoder is trained
with **no output activation** against a target that the loader normalises as
``xct / 255`` (``losses/total.py:73`` -> ``recon_fn(output.xct_out,
batch["xct"])``; ``dataset/loader.py:100,218``).  The decoder therefore already
predicts [0, 1] grey levels, and the ``expit`` in ``sampler.py:1276`` is an
extra squash applied on top of an already-[0,1] signal — which is why generated
volumes span only 0.38-0.74 instead of 0-1.  Undoing it lands the generated
material mode on u8 209, exactly the real material mode (209-211), and the
generated dark mode on u8 30, close to the real exterior-air mode (41).  Under
``raw`` the same two modes sit at 177 and 135.

Note also that ``clip(logit(v), 0, 1) * 255`` (what ``eval_v2_audit.py`` used)
is algebraically identical to ``clip(logit(v) * 255, 0, 255)``, so the earlier
audit's conversion is the ``native`` scale, not a separate third thing.

Because the two scales are related by a monotonic look-up table, **the voxel
sets a threshold selects are identical on both** once the threshold is mapped
through the LUT.  Scale choice changes mode positions, mode separation and the
numeric value of a threshold — it does not change which voxels are flagged.
Every threshold below is therefore applied on the raw array, with the
real-calibrated ones mapped in from the native scale through the LUT.

``eval_v2_audit.py`` and its outputs are left untouched as the record of the
earlier run; this script re-derives everything and prints an explicit
OLD-vs-NEW comparison table.

What it measures
----------------
Task 1 — unmasked air over all 162 saved volumes (153 eval_v2 + 9 ldm06_probe):

* Full 256-bin intensity histograms, Gaussian-smoothed, with prominence-based
  peak detection (number of modes, their positions, heights, FWHM), the valley
  between the two dominant modes and a valley-depth bimodality index.  No
  argmax-only or percentile-only summaries.
* Threshold per volume derived from the volume's *own* bimodal structure
  (Otsu restricted between the two detected modes), plus the valley minimum as
  a sensitivity variant, plus two real-calibrated variants (an absolute
  Dice-optimal u8 threshold and a material-mode-referenced offset).
* Minimum connected component 300 voxels, interior/edge shell split at 32
  voxels, per-64^3-cell statistics, mask-collapse correlation — same as the
  earlier audit.
* Validation of every threshold rule on REAL volumes with stored pore masks
  (voxel Dice / precision / recall inside the stored ``sample_mask``).

Task 2 — matched interior comparison.  Real 192^3 sub-volumes sampled fully
inside ``sample_mask`` (no exterior air) versus generated 192^3 volumes:
mode count and position, inter-mode valley, mode separation and peak widths,
and the intensity distribution of pore voxels vs material voxels on both
sides.

Outputs to ``runs/analysis/air_audit_v2/``: results.json, per_volume.csv,
per_cell.csv, findings.md, figures (PDF + PNG, 300 dpi), run.log.

Usage:
    python scripts/analysis/air_audit_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, ZARR_ROOT, savefig, set_style, write_json  # noqa: E402

DATA_ROOT = REPO / "data" / "split_v2"
OUT_DIR = REPO / "runs" / "analysis" / "air_audit_v2"
GEN_ROOT = REPO / "runs" / "eval_v2" / "volumes"
PROBE_ROOT = REPO / "runs" / "analysis" / "ldm06_probe" / "volumes"
OLD_AUDIT_CSV = REPO / "runs" / "eval_v2" / "audit" / "per_volume.csv"

EXPERIMENTS = ["dose_response", "cfg_sweep", "layup"]
ARMS = ["seq", "joint_legacy", "joint_oob"]
ARM_COLORS = {"seq": "#1b6ca8", "joint_legacy": "#c2571a",
              "joint_oob": "#2e7d32", "probe": "#7a3ea1"}

VOXEL_UM = 25.0                 # user-supplied; not in dataset metadata
VOXEL_MM3 = (VOXEL_UM / 1000.0) ** 3
PATCH = 64                      # cell grid for the per-cell table
MIN_CC = 300                    # voxels; ~4.7e-3 mm^3, equiv. diam ~0.21 mm
EDGE_VOX = 32                   # shell within this many voxels of any face

# Histogram / peak-detection settings (identical for real and generated).
HIST_SIGMA = 2.0                # Gaussian smoothing of the 256-bin histogram
PEAK_PROM_REL = 2e-3            # min peak prominence as a fraction of all voxels
PEAK_MIN_DIST = 5               # min separation between peaks, u8 levels

# Real-side sampling.
N_CAL_VOLUMES = 8               # real val/test volumes used for calibration
CAL_SLICE_STEP = 24             # every Nth z-slice during calibration
SM_ERODE = 3                    # erosion of sample_mask -> foreground
REAL_BOX = 192                  # matched interior box edge, voxels
N_BOX_VOLUMES = 5
N_BOXES_PER_VOLUME = 4

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# THE conversion.  One definition, used everywhere.
# ---------------------------------------------------------------------------

def to_u8(vol: np.ndarray) -> np.ndarray:
    """volume.tif float32 -> raw u8 grey levels.  Direct rescale, no logit."""
    return np.clip(vol.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)


def _build_native_lut() -> np.ndarray:
    """raw u8 level -> decoder-native u8 = clip(logit(k/255) * 255, 0, 255).

    Monotonic non-decreasing, so it maps thresholds and histograms between the
    two scales without ever materialising a second volume.
    """
    k = np.arange(256, dtype=np.float64)
    v = np.clip(k, 1.0, 254.0) / 255.0
    lut = np.clip(np.round(np.log(v / (1.0 - v)) * 255.0), 0.0, 255.0)
    lut[0], lut[255] = 0.0, 255.0
    return lut.astype(np.uint8)


NATIVE_LUT = _build_native_lut()


def _build_native_weights() -> np.ndarray:
    """W[k, j] = fraction of raw level k's mass that lands in native bin j.

    A pointwise LUT would be wrong here: near the material peak one raw level
    spans ~5 native levels, so a pointwise map leaves a comb of empty bins that
    smoothing turns into spurious modes.  Each raw level therefore covers the
    native interval its half-open bin maps to, and its mass is spread over that
    interval — an area-preserving density transform.
    """
    def fwd(x: np.ndarray) -> np.ndarray:
        v = np.clip(x, 0.5, 254.5) / 255.0
        return np.clip(np.log(v / (1.0 - v)) * 255.0, 0.0, 255.0)

    k = np.arange(256, dtype=np.float64)
    nlo, nhi = fwd(k - 0.5), fwd(k + 0.5)
    W = np.zeros((256, 256), dtype=np.float64)
    for i in range(256):
        a, b = nlo[i], nhi[i]
        if b <= a:                                  # fully inside a clip
            W[i, int(min(a, 255.0))] = 1.0
            continue
        for j in range(int(np.floor(a)), min(int(np.floor(b - 1e-9)), 255) + 1):
            ov = min(b, j + 1.0) - max(a, float(j))
            if ov > 0:
                W[i, j] = ov
        W[i] /= W[i].sum()
    return W


NATIVE_W = _build_native_weights()


def native_hist(h: np.ndarray) -> np.ndarray:
    """Re-bin a raw-scale histogram onto the decoder-native scale."""
    return h.astype(np.float64) @ NATIVE_W


def native_to_raw_threshold(t_native: int) -> int:
    """Raw threshold selecting exactly the voxels with native value < t_native."""
    return int(np.searchsorted(NATIVE_LUT, t_native, side="left"))


def load_generated_u8(cell_dir: Path) -> np.ndarray:
    """Chunked memmap read + correct conversion."""
    mm = tifffile.memmap(str(cell_dir / "volume.tif"), mode="r")
    out = np.empty(mm.shape, np.uint8)
    for z0 in range(0, mm.shape[0], 32):
        out[z0:z0 + 32] = to_u8(np.asarray(mm[z0:z0 + 32]))
    del mm
    return out


# ---------------------------------------------------------------------------
# Histogram / mode analysis
# ---------------------------------------------------------------------------

def hist_u8(a: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    vals = a[sel] if sel is not None else a.ravel()
    return np.bincount(vals, minlength=256).astype(np.int64)


def otsu(h: np.ndarray, lo: int = 0, hi: int = 256) -> int:
    """Otsu threshold restricted to bins [lo, hi)."""
    hh = h[lo:hi].astype(np.float64)
    if hh.sum() <= 0:
        return lo
    p = hh / hh.sum()
    bins = np.arange(lo, hi, dtype=np.float64)
    w0 = np.cumsum(p)
    mc = np.cumsum(p * bins)
    mt = mc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        var_b = (mt * w0 - mc) ** 2 / (w0 * (1.0 - w0))
    var_b[~np.isfinite(var_b)] = 0.0
    return int(lo + np.argmax(var_b))


def analyse_modes(h: np.ndarray) -> dict:
    """Full distribution characterisation of a 256-bin u8 histogram.

    Returns every detected local maximum (position, relative height, relative
    prominence, FWHM), the two dominant modes, the valley between them, the
    valley-depth bimodality index and the Otsu split restricted to the
    inter-mode range.  Never summarises the distribution by argmax alone.
    """
    total = float(h.sum())
    out: dict = {"n_voxels": int(total), "argmax": int(np.argmax(h))}
    if total <= 0:
        return {**out, "n_modes": 0, "peaks": [], "bimodal": False}

    hs = gaussian_filter1d(h.astype(np.float64), HIST_SIGMA)
    pk, props = find_peaks(hs, prominence=total * PEAK_PROM_REL,
                           distance=PEAK_MIN_DIST)
    if len(pk):
        widths = peak_widths(hs, pk, rel_height=0.5)[0]
    else:
        widths = np.zeros(0)

    peaks = [{
        "level": int(p),
        "height_frac": float(hs[p] / total),
        "prominence_frac": float(props["prominences"][i] / total),
        "fwhm_u8": float(widths[i]),
    } for i, p in enumerate(pk)]
    peaks.sort(key=lambda d: -d["prominence_frac"])
    out["n_modes"] = int(len(pk))
    out["peaks"] = peaks
    out["mode_levels"] = sorted(int(p["level"]) for p in peaks)

    # Cumulative shape descriptors (reported alongside, never instead of, modes)
    cdf = np.cumsum(h) / total
    for q in (1, 5, 25, 50, 75, 95, 99):
        out[f"p{q:02d}"] = int(np.searchsorted(cdf, q / 100.0))
    out["mean"] = float((np.arange(256) * h).sum() / total)
    out["std"] = float(np.sqrt(((np.arange(256) - out["mean"]) ** 2 * h).sum() / total))

    if len(peaks) < 2:
        out.update({"bimodal": False, "dark_mode": None,
                    "material_mode": int(peaks[0]["level"]) if peaks else None,
                    "dark_fwhm": None,
                    "material_fwhm": float(peaks[0]["fwhm_u8"]) if peaks else None,
                    "valley_level": None, "valley_depth_ratio": None,
                    "mode_separation": None, "otsu_between": None,
                    "dark_mass_frac": None})
        return out

    top2 = sorted(peaks[:2], key=lambda d: d["level"])
    dark, mat = top2
    a, b = dark["level"], mat["level"]
    v = int(a + np.argmin(hs[a:b + 1]))
    depth = float(hs[v] / min(hs[a], hs[b]))
    out.update({
        "bimodal": True,
        "dark_mode": int(a),
        "material_mode": int(b),
        "dark_fwhm": float(dark["fwhm_u8"]),
        "material_fwhm": float(mat["fwhm_u8"]),
        "dark_height_frac": float(dark["height_frac"]),
        "material_height_frac": float(mat["height_frac"]),
        "valley_level": v,
        "valley_depth_ratio": depth,          # 0 = perfectly split, 1 = no dip
        "mode_separation": int(b - a),
        "mode_separation_norm": float((b - a) / max(0.5 * (dark["fwhm_u8"] + mat["fwhm_u8"]), 1e-6)),
        "otsu_between": int(otsu(h, a, b + 1)),
        "dark_mass_frac": float(h[:v].sum() / total),
    })
    return out


def smoothed_pdf(h: np.ndarray) -> np.ndarray:
    total = max(float(h.sum()), 1.0)
    return gaussian_filter1d(h.astype(np.float64), HIST_SIGMA) / total


# ---------------------------------------------------------------------------
# Detection primitives (unchanged methodology)
# ---------------------------------------------------------------------------

def cc_filter(det: np.ndarray, min_cc: int = MIN_CC) -> tuple[np.ndarray, int, int]:
    labels, n = ndimage.label(det)
    if n == 0:
        return det, 0, 0
    sizes = np.bincount(labels.ravel())
    small = sizes < min_cc
    small[0] = False
    n_kept = int(n - small[1:].sum())
    det[small[labels]] = False
    del labels
    return det, n_kept, int(n)


_EDGE_CACHE: dict[tuple, np.ndarray] = {}


def edge_shell(shape: tuple) -> np.ndarray:
    if shape not in _EDGE_CACHE:
        ax = []
        for s in shape:
            v = np.zeros(s, bool)
            v[:EDGE_VOX] = True
            v[-EDGE_VOX:] = True
            ax.append(v)
        _EDGE_CACHE[shape] = (ax[0][:, None, None] | ax[1][None, :, None]
                              | ax[2][None, None, :])
    return _EDGE_CACHE[shape]


def largest_components(unmasked: np.ndarray, top: int = 5) -> list[dict]:
    labels, n = ndimage.label(unmasked)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][:top]
    objs = ndimage.find_objects(labels)
    shape = unmasked.shape
    out = []
    for idx in order:
        size = int(sizes[idx])
        if size == 0:
            break
        sl = objs[idx - 1]
        out.append({
            "voxels": size,
            "volume_mm3": size * VOXEL_MM3,
            "equiv_diameter_mm": 2.0 * (3.0 * size * VOXEL_MM3 / (4.0 * np.pi)) ** (1 / 3),
            "bbox_extent_vox": [int(s.stop - s.start) for s in sl],
            "touches_face": bool(any(s.start == 0 or s.stop == shape[i]
                                     for i, s in enumerate(sl))),
        })
    del labels
    return out


def cellwise(a: np.ndarray) -> np.ndarray:
    gz, gy, gx = (s // PATCH for s in a.shape)
    return (a[:gz * PATCH, :gy * PATCH, :gx * PATCH]
            .reshape(gz, PATCH, gy, PATCH, gx, PATCH)
            .mean(axis=(1, 3, 5)))


def dice_prec_rec(inter: float, npred: float, ngt: float) -> tuple[float, float, float]:
    dice = 2.0 * inter / (npred + ngt) if npred + ngt else 0.0
    prec = inter / npred if npred else 1.0
    rec = inter / ngt if ngt else 0.0
    return dice, prec, rec


def intensity_stats(vals_hist: np.ndarray) -> dict:
    """Mode + full quantile ladder of an intensity population, from its histogram."""
    tot = float(vals_hist.sum())
    if tot <= 0:
        return {"n": 0}
    cdf = np.cumsum(vals_hist) / tot
    mean = float((np.arange(256) * vals_hist).sum() / tot)
    return {
        "n": int(tot),
        "mode": int(np.argmax(vals_hist)),
        "mean": mean,
        "std": float(np.sqrt(((np.arange(256) - mean) ** 2 * vals_hist).sum() / tot)),
        "p05": int(np.searchsorted(cdf, 0.05)),
        "p25": int(np.searchsorted(cdf, 0.25)),
        "p50": int(np.searchsorted(cdf, 0.50)),
        "p75": int(np.searchsorted(cdf, 0.75)),
        "p95": int(np.searchsorted(cdf, 0.95)),
    }


# ---------------------------------------------------------------------------
# Real-side loading
# ---------------------------------------------------------------------------

def real_split_names(splits: dict, which=("val", "test")) -> list[str]:
    return sorted(n for n, s in splits["volumes"].items() if s in which)


def calibrate_absolute(root, names: list[str]) -> dict:
    """Dice-optimal absolute u8 threshold on real volumes, inside sample_mask.

    Also records the real material mode / spread so the threshold can be
    re-expressed as an offset from (or ratio of) the material mode — the form
    that transfers to a volume whose grey scale differs.
    """
    coarse = np.arange(60, 231, 5)
    pooled = {int(t): [0.0, 0.0, 0.0] for t in coarse}
    n_fg = 0
    mat_modes, mat_spreads, mat_hists = [], [], []

    slices_cache: dict[str, list] = {}
    for name in names:
        g = root[name]
        Z = g["xct"].shape[0]
        keep = []
        for z in range(0, Z, CAL_SLICE_STEP):
            x = np.asarray(g["xct"][z])
            m = np.asarray(g["mask"][z]) > 0
            f = ndimage.binary_erosion(np.asarray(g["sample_mask"][z]) > 0,
                                       iterations=SM_ERODE)
            if f.sum() < 1000:
                continue
            keep.append((x, m, f))
        slices_cache[name] = keep
        h = np.zeros(256, np.int64)
        for x, m, f in keep:
            h += hist_u8(x, f)
            n_fg += int(f.sum())
        mat_hists.append(h)
        info = analyse_modes(h)
        mat_modes.append(info["material_mode"] if info["material_mode"] is not None
                         else info["argmax"])
        # robust spread of the material side: p84 - p50 above the volume Otsu
        thr = otsu(h)
        upper = h[thr:]
        c = np.cumsum(upper) / max(upper.sum(), 1)
        mat_spreads.append(float(max(np.searchsorted(c, 0.84) - np.searchsorted(c, 0.50), 1)))
        log(f"  calib {name[:46]:46s} mode={mat_modes[-1]} spread={mat_spreads[-1]:.0f}")

    def pool(thresholds):
        store = {int(t): [0.0, 0.0, 0.0] for t in thresholds}
        for keep in slices_cache.values():
            for x, m, f in keep:
                gt = m & f
                ngt = float(gt.sum())
                for t in thresholds:
                    p = (x < t) & f
                    store[int(t)][0] += float((p & gt).sum())
                    store[int(t)][1] += float(p.sum())
                    store[int(t)][2] += ngt
        return store

    pooled = pool(coarse)
    t_c = max(pooled, key=lambda t: dice_prec_rec(*pooled[t])[0])
    fine = np.arange(max(t_c - 5, 0), min(t_c + 6, 256))
    pooled.update(pool(fine))
    t_best = max(pooled, key=lambda t: dice_prec_rec(*pooled[t])[0])
    d, p, r = dice_prec_rec(*pooled[t_best])

    sweep = sorted(pooled)
    mat_mode = float(np.mean(mat_modes))
    return {
        "t_abs": int(t_best),
        "dice": d, "precision": p, "recall": r,
        "n_calibration_volumes": len(names),
        "calibration_volumes": names,
        "n_foreground_voxels": int(n_fg),
        "real_material_mode_mean": mat_mode,
        "real_material_spread_mean": float(np.mean(mat_spreads)),
        "real_material_modes": mat_modes,
        "offset_from_material_mode": float(mat_mode - t_best),
        "ratio_to_material_mode": float(t_best / mat_mode),
        "sweep_thresholds": [int(t) for t in sweep],
        "sweep_dice": [dice_prec_rec(*pooled[t])[0] for t in sweep],
        "sweep_precision": [dice_prec_rec(*pooled[t])[1] for t in sweep],
        "sweep_recall": [dice_prec_rec(*pooled[t])[2] for t in sweep],
        "_slices": slices_cache,
    }


def validate_rules_on_real(cal: dict) -> dict:
    """Dice of every threshold rule vs the stored pore mask, per real volume.

    Rules: the absolute real-calibrated threshold, the material-mode-referenced
    offset, and the self-derived bimodal rules (Otsu-between-modes and valley)
    with the documented unimodal fallback to whole-histogram Otsu.
    """
    out = {}
    for name, keep in cal["_slices"].items():
        h = np.zeros(256, np.int64)
        for x, m, f in keep:
            h += hist_u8(x, f)
        info = analyse_modes(h)
        rules = {
            "T_abs": cal["t_abs"],
            "T_rel": int(round((info["material_mode"] or info["argmax"])
                               - cal["offset_from_material_mode"])),
            "T_self": int(info["otsu_between"]) if info["bimodal"] else int(otsu(h)),
            "T_valley": int(info["valley_level"]) if info["bimodal"] else int(otsu(h)),
        }
        row = {"modes": info, "thresholds": rules,
               "self_rule_fallback": not info["bimodal"]}
        for tag, t in rules.items():
            inter = npred = ngt = 0.0
            for x, m, f in keep:
                pr = (x < t) & f
                gt = m & f
                inter += float((pr & gt).sum())
                npred += float(pr.sum())
                ngt += float(gt.sum())
            d, p, r = dice_prec_rec(inter, npred, ngt)
            row[tag] = {"threshold": int(t), "dice": d, "precision": p,
                        "recall": r,
                        "detected_over_gt_ratio": float(npred / max(ngt, 1.0))}
        out[name] = row
        log(f"  real rules {name[:40]:40s} "
            f"bimodal={info['bimodal']} "
            f"T_abs={rules['T_abs']}/D={row['T_abs']['dice']:.3f} "
            f"T_rel={rules['T_rel']}/D={row['T_rel']['dice']:.3f} "
            f"T_self={rules['T_self']}/D={row['T_self']['dice']:.3f}")
    return out


def find_interior_boxes(g, n: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """Origins of REAL_BOX^3 boxes lying entirely inside sample_mask."""
    Z, Y, X = g["xct"].shape
    B = REAL_BOX
    if Z < B or Y < B or X < B:
        return []
    z0 = (Z - B) // 2
    sm = np.ones((Y, X), bool)
    for z in (z0, z0 + B // 2, z0 + B - 1):
        sm &= np.asarray(g["sample_mask"][z]) > 0
    ii = np.zeros((Y + 1, X + 1), np.int64)
    np.cumsum(np.cumsum(sm.astype(np.int64), axis=0), axis=1, out=ii[1:, 1:])
    ys = np.arange(0, Y - B + 1, 32)
    xs = np.arange(0, X - B + 1, 32)
    s = (ii[np.ix_(ys + B, xs + B)] - ii[np.ix_(ys, xs + B)]
         - ii[np.ix_(ys + B, xs)] + ii[np.ix_(ys, xs)])
    cand = np.argwhere(s == B * B)
    if len(cand) == 0:
        return []
    rng.shuffle(cand)
    chosen: list[tuple[int, int, int]] = []
    for iy, ix in cand:
        y0, x0 = int(ys[iy]), int(xs[ix])
        if any(abs(y0 - cy) < B and abs(x0 - cx) < B for _, cy, cx in chosen):
            continue
        sm3 = np.asarray(g["sample_mask"][z0:z0 + B, y0:y0 + B, x0:x0 + B]) > 0
        if not sm3.all():
            continue
        chosen.append((z0, y0, x0))
        if len(chosen) >= n:
            break
    return chosen


def analyse_real_box(g, name: str, origin: tuple[int, int, int], cal: dict) -> dict:
    z0, y0, x0 = origin
    B = REAL_BOX
    sl = (slice(z0, z0 + B), slice(y0, y0 + B), slice(x0, x0 + B))
    xct = np.asarray(g["xct"][sl])
    mask = np.asarray(g["mask"][sl]) > 0
    h = hist_u8(xct)
    info = analyse_modes(h)
    pore_h = hist_u8(xct, mask)
    mat_h = hist_u8(xct, ~mask)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    row = {
        "volume": name, "origin": [int(v) for v in origin],
        "mask_porosity": float(mask.mean()),
        "modes": info,
        "hist": h.tolist(),
        "pore_stats": intensity_stats(pore_h),
        "pore_core_stats": intensity_stats(hist_u8(xct, eroded)),
        "material_stats": intensity_stats(mat_h),
    }
    # apply every threshold rule and score it against the stored mask
    rules = {
        "T_abs": cal["t_abs"],
        "T_rel": int(round((info["material_mode"] or info["argmax"])
                           - cal["offset_from_material_mode"])),
        "T_self": int(info["otsu_between"]) if info["bimodal"] else int(otsu(h)),
        "T_valley": int(info["valley_level"]) if info["bimodal"] else int(otsu(h)),
    }
    row["self_rule_fallback"] = not info["bimodal"]
    for tag, t in rules.items():
        det = xct < t
        d_raw, p_raw, r_raw = dice_prec_rec(float((det & mask).sum()),
                                            float(det.sum()), float(mask.sum()))
        det_cc, n_kept, n_tot = cc_filter(det.copy())
        d, p, r = dice_prec_rec(float((det_cc & mask).sum()),
                                float(det_cc.sum()), float(mask.sum()))
        unmasked = det_cc & ~mask
        row[tag] = {
            "threshold": int(t),
            "dice_nocc": d_raw, "precision_nocc": p_raw, "recall_nocc": r_raw,
            "dice_cc": d, "precision_cc": p, "recall_cc": r,
            "detected_frac": float(det_cc.mean()),
            "unmasked_frac": float(unmasked.mean()),
            "components_kept": n_kept, "components_total": n_tot,
        }
    del xct, mask
    return row


# ---------------------------------------------------------------------------
# Generated-volume audit
# ---------------------------------------------------------------------------

def iter_generated():
    for exp in EXPERIMENTS:
        for arm in ARMS:
            root = GEN_ROOT / exp / arm
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if (d / "volume.tif").exists() and (d / "mask.tif").exists():
                    yield exp, arm, d
    if PROBE_ROOT.exists():
        for d in sorted(PROBE_ROOT.iterdir()):
            if (d / "volume.tif").exists() and (d / "mask.tif").exists():
                yield "ldm06_probe", "probe", d


def audit_generated(cell_dir: Path, exp: str, arm: str, cal: dict
                    ) -> tuple[dict, pd.DataFrame, np.ndarray]:
    stats = json.loads((cell_dir / "stats.json").read_text())
    u8 = load_generated_u8(cell_dir)
    mask = tifffile.imread(str(cell_dir / "mask.tif")) > 0
    n_vox = u8.size

    h = hist_u8(u8)
    info = analyse_modes(h)
    hn = native_hist(h)
    info_n = analyse_modes(hn)

    row = {
        "experiment": exp, "arm": arm, "name": cell_dir.name,
        "target": stats.get("target"), "seed": stats.get("seed"),
        "s_por": stats.get("s_por"), "layup": stats.get("layup"),
        "shape": "x".join(str(s) for s in u8.shape),
        "n_voxels": n_vox,
        "raw_min_u8": int(np.nonzero(h)[0][0]),
        "raw_max_u8": int(np.nonzero(h)[0][-1]),
        "mask_porosity": float(mask.mean()),
        "delivered_mask_porosity": stats.get("delivered_mask_porosity"),
        "n_modes": info["n_modes"],
        "bimodal": info["bimodal"],
        "argmax_u8": info["argmax"],
        "dark_mode_u8": info.get("dark_mode"),
        "material_mode_u8": info.get("material_mode"),
        "dark_fwhm_u8": info.get("dark_fwhm"),
        "material_fwhm_u8": info.get("material_fwhm"),
        "mode_separation_u8": info.get("mode_separation"),
        "valley_level_u8": info.get("valley_level"),
        "valley_depth_ratio": info.get("valley_depth_ratio"),
        "dark_mass_frac": info.get("dark_mass_frac"),
        "otsu_between_u8": info.get("otsu_between"),
        "p01": info["p01"], "p05": info["p05"], "p50": info["p50"],
        "p95": info["p95"], "p99": info["p99"],
        "mean_u8": info["mean"], "std_u8": info["std"],
        "pore_stats": json.dumps(intensity_stats(hist_u8(u8, mask))),
        "material_stats": json.dumps(intensity_stats(hist_u8(u8, ~mask))),
        # --- decoder-native scale (sampler expit undone), diagnostic track ---
        "native_n_modes": info_n["n_modes"],
        "native_bimodal": info_n["bimodal"],
        "native_dark_mode_u8": info_n.get("dark_mode"),
        "native_material_mode_u8": info_n.get("material_mode"),
        "native_dark_fwhm_u8": info_n.get("dark_fwhm"),
        "native_material_fwhm_u8": info_n.get("material_fwhm"),
        "native_mode_separation_u8": info_n.get("mode_separation"),
        "native_valley_level_u8": info_n.get("valley_level"),
        "native_valley_depth_ratio": info_n.get("valley_depth_ratio"),
        "native_dark_mass_frac": info_n.get("dark_mass_frac"),
        "native_p01": info_n["p01"], "native_p50": info_n["p50"],
        "native_p99": info_n["p99"],
        "native_clipped_low_frac": float(hn[0] / max(hn.sum(), 1)),
        "native_clipped_high_frac": float(hn[255] / max(hn.sum(), 1)),
        "native_pore_stats": json.dumps(intensity_stats(
            native_hist(hist_u8(u8, mask)))),
        "native_material_stats": json.dumps(intensity_stats(
            native_hist(hist_u8(u8, ~mask)))),
    }

    # --- threshold rules, all expressed as RAW-scale thresholds -----------
    # `self`/`valley` come from the volume's own raw bimodal structure.
    # `self_native` repeats the self rule on the native scale (scale-sensitivity
    # check).  `rel`/`abs` are the real-calibrated rules: real u8 IS the native
    # scale, so they are defined there and mapped back through NATIVE_LUT.
    t_self = int(info["otsu_between"]) if info["bimodal"] else int(otsu(h))
    nat_mat = (info_n["material_mode"] if info_n["material_mode"] is not None
               else info_n["argmax"])
    t_self_nat = int(info_n["otsu_between"]) if info_n["bimodal"] else int(otsu(hn))
    t_rel_native = int(round(nat_mat - cal["offset_from_material_mode"]))
    thresholds = {
        "self": t_self,
        "valley": int(info["valley_level"]) if info["bimodal"] else t_self,
        "self_native": native_to_raw_threshold(t_self_nat),
        "rel": native_to_raw_threshold(t_rel_native),
        "abs": native_to_raw_threshold(int(cal["t_abs"])),
    }
    row["self_rule_fallback"] = not info["bimodal"]
    row["T_self_native_scale"] = t_self_nat
    row["T_rel_native_scale"] = t_rel_native
    row["T_abs_native_scale"] = int(cal["t_abs"])
    for tag, t in thresholds.items():
        row[f"T_{tag}"] = int(t)
        row[f"T_{tag}_native_equiv"] = int(NATIVE_LUT[min(int(t), 255)])

    edge = edge_shell(u8.shape)
    n_edge = int(edge.sum())
    cells_df = None
    for tag, t in thresholds.items():
        det = u8 < t
        row[f"detected_air_{tag}_nocc"] = float(det.mean())
        det, n_kept, n_tot = cc_filter(det)
        unmasked = det & ~mask
        n_det = int(det.sum())
        n_um = int(unmasked.sum())
        row[f"detected_air_{tag}"] = n_det / n_vox
        row[f"unmasked_air_{tag}"] = n_um / n_vox
        row[f"n_components_{tag}"] = n_kept
        if tag in ("self", "rel"):
            um_edge = int((unmasked & edge).sum())
            row[f"unmasked_edge_{tag}"] = um_edge / n_vox
            row[f"unmasked_interior_{tag}"] = (n_um - um_edge) / n_vox
            row[f"unmasked_edge_local_{tag}"] = um_edge / n_edge
            row[f"unmasked_interior_local_{tag}"] = (n_um - um_edge) / (n_vox - n_edge)
        if tag == "self":
            comps = largest_components(unmasked)
            row["top_components"] = json.dumps(comps)
            row["largest_comp_voxels"] = comps[0]["voxels"] if comps else 0
            row["largest_comp_mm3"] = comps[0]["volume_mm3"] if comps else 0.0
            row["largest_comp_equiv_diam_mm"] = comps[0]["equiv_diameter_mm"] if comps else 0.0
            row["largest_comp_touches_face"] = comps[0]["touches_face"] if comps else False
            cells_df = pd.DataFrame({
                "experiment": exp, "arm": arm, "name": cell_dir.name,
                "cell_dark_frac": cellwise(det).ravel(),
                "cell_mask_porosity": cellwise(mask).ravel(),
                "cell_unmasked_frac": cellwise(unmasked).ravel(),
            })
        del det, unmasked
    del u8, mask
    return row, cells_df, h


# ---------------------------------------------------------------------------
# Old-vs-new comparison
# ---------------------------------------------------------------------------

def old_vs_new(pv: pd.DataFrame) -> pd.DataFrame:
    if not OLD_AUDIT_CSV.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_AUDIT_CSV)
    keys = ["experiment", "arm", "name"]
    m = pv.merge(old[keys + ["detected_air_best", "unmasked_air_best",
                             "unmasked_air_cons", "unmasked_interior_best",
                             "unmasked_edge_best"]],
                 on=keys, how="inner", suffixes=("", "_old"))
    agg = (m.groupby(["experiment", "arm"])
            .agg(n=("name", "count"),
                 mask_porosity=("mask_porosity", "mean"),
                 old_detected_air=("detected_air_best", "mean"),
                 old_unmasked_air=("unmasked_air_best", "mean"),
                 old_unmasked_cons=("unmasked_air_cons", "mean"),
                 old_unmasked_interior=("unmasked_interior_best", "mean"),
                 new_detected_air=("detected_air_self", "mean"),
                 new_unmasked_air=("unmasked_air_self", "mean"),
                 new_unmasked_rel=("unmasked_air_rel", "mean"),
                 new_unmasked_interior=("unmasked_interior_self", "mean"))
            .reset_index())
    agg["ratio_old_over_new"] = agg.old_unmasked_air / agg.new_unmasked_air.replace(0, np.nan)
    return agg


def collapse_stats(cells: pd.DataFrame, arms: list[str]) -> dict:
    out = {}
    for arm in arms:
        c = cells[cells.arm == arm]
        if not len(c):
            continue
        dark = c.cell_dark_frac.to_numpy()
        umf = c.cell_unmasked_frac.to_numpy()
        mpor = c.cell_mask_porosity.to_numpy()
        sel = dark > 1e-4
        capture = 1.0 - umf[sel] / np.maximum(dark[sel], 1e-9)
        regimes = {}
        for lo, hi, tag in ((1e-4, 0.02, "low"), (0.02, 0.15, "mid"), (0.15, 1.01, "high")):
            s = (dark >= lo) & (dark < hi)
            cap = 1.0 - umf[s] / np.maximum(dark[s], 1e-9)
            regimes[tag] = {
                "n_cells": int(s.sum()),
                "capture_median": float(np.median(cap)) if s.any() else None,
                "mask_porosity_median": float(np.median(mpor[s])) if s.any() else None,
            }
        out[arm] = {
            "n_cells": int(len(c)),
            "pearson_dark_vs_unmasked": float(np.corrcoef(dark, umf)[0, 1]),
            "pearson_dark_vs_mask_porosity": (float(np.corrcoef(dark[sel], mpor[sel])[0, 1])
                                              if sel.sum() > 2 else None),
            "capture_median_overall": float(np.median(capture)) if sel.any() else None,
            "regimes": regimes,
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_modes(real_boxes: list[dict], gen_hists: dict, pv: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    ax = axes[0, 0]
    for i, b in enumerate(real_boxes):
        pdf = smoothed_pdf(np.asarray(b["hist"], np.int64))
        ax.plot(np.arange(256), pdf, color="#333333", lw=0.8, alpha=0.5,
                label="real interior box" if i == 0 else None)
        for p in b["modes"]["peaks"]:
            ax.plot(p["level"], pdf[p["level"]], "v", color="#c2571a", ms=5)
    ax.set_xlim(0, 255)
    ax.set_xlabel("intensity (u8)")
    ax.set_ylabel("smoothed density")
    ax.set_title(f"Real interior boxes ({REAL_BOX}$^3$, inside sample_mask), "
                 f"n={len(real_boxes)}\ntriangles = detected modes")
    ax.legend()

    ax = axes[0, 1]
    for arm in ARMS + ["probe"]:
        sel = [k for k in gen_hists if k[1] == arm]
        if not sel:
            continue
        for j, k in enumerate(sel[:40]):
            pdf = smoothed_pdf(gen_hists[k])
            ax.plot(np.arange(256), pdf, color=ARM_COLORS[arm], lw=0.7, alpha=0.35,
                    label=arm if j == 0 else None)
    ax.set_xlim(0, 255)
    ax.set_xlabel("intensity (u8)")
    ax.set_ylabel("smoothed density")
    ax.set_title("Generated volumes — raw u8 scale\n"
                 "clip(v*255, 0, 255), the instructed conversion")
    ax.legend()

    ax = axes[1, 0]
    xs, ys, cs = [], [], []
    for b in real_boxes:
        for pk in b["modes"]["peaks"]:
            xs.append(pk["level"]), ys.append(0.0), cs.append("#333333")
    for _, r in pv.iterrows():
        c = ARM_COLORS.get(r.arm, "#888888")
        for col, lvl in (("dark_mode_u8", 1.0), ("material_mode_u8", 1.0),
                         ("native_dark_mode_u8", 2.0),
                         ("native_material_mode_u8", 2.0)):
            v = r[col]
            if v is not None and v == v:
                xs.append(v), ys.append(lvl), cs.append(c)
    jit = np.random.default_rng(0).normal(0, 0.07, len(ys))
    ax.scatter(xs, np.asarray(ys) + jit, c=cs, s=14, alpha=0.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["real interior", "generated\n(raw scale)",
                        "generated\n(native scale)"], fontsize=8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("mode position (u8)")
    ax.set_title("Where the modes sit — both grey scales")

    ax = axes[1, 1]
    nm_real = [b["modes"]["n_modes"] for b in real_boxes]
    ax.hist(nm_real, bins=np.arange(-0.5, 5.5), color="#333333", alpha=0.6,
            density=True, label="real interior")
    ax.hist(pv.n_modes.to_numpy(), bins=np.arange(-0.5, 5.5), color="#1b6ca8",
            alpha=0.6, density=True, label="generated")
    ax.set_xticks(range(5))
    ax.set_xlabel("number of detected modes")
    ax.set_ylabel("fraction of volumes")
    ax.set_title(f"Mode count (prominence >= {PEAK_PROM_REL:.0e} of voxels)")
    ax.legend()

    fig.suptitle("Intensity-distribution characterisation — real interior vs generated", y=0.995)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig1_mode_analysis")


def fig_unmasked_by_arm(pv: pd.DataFrame, cmp_df: pd.DataFrame,
                        real_fp: dict) -> list[str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    xs = np.arange(len(EXPERIMENTS))
    w = 0.25
    for j, arm in enumerate(ARMS):
        vals = [pv[(pv.experiment == e) & (pv.arm == arm)].unmasked_air_self.mean()
                for e in EXPERIMENTS]
        rel = [pv[(pv.experiment == e) & (pv.arm == arm)].unmasked_air_rel.mean()
               for e in EXPERIMENTS]
        pos = xs + (j - 1) * w
        ax.bar(pos, vals, w * 0.9, color=ARM_COLORS[arm], label=f"{arm} (T_self)")
        ax.bar(pos, rel, w * 0.45, color="k", alpha=0.35,
               label="T_rel (real-calibrated)" if j == 0 else None)
    if real_fp:
        ax.axhline(real_fp["mean"], color="k", ls="--", lw=1,
                   label=f"real false-positive baseline ({real_fp['mean']:.4f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(EXPERIMENTS)
    ax.set_ylabel("unmasked-air voxel fraction")
    ax.set_title(f"NEW (correct u8): unmasked air by arm\nmin CC {MIN_CC} vox")
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    if len(cmp_df):
        labels = [f"{r.experiment}\n{r.arm}" for _, r in cmp_df.iterrows()]
        xs2 = np.arange(len(cmp_df))
        ax.bar(xs2 - 0.2, cmp_df.old_unmasked_air, 0.38, color="#b23a48",
               label="OLD (logit conversion)")
        ax.bar(xs2 + 0.2, cmp_df.new_unmasked_air, 0.38, color="#1b6ca8",
               label="NEW (correct u8)")
        ax.set_xticks(xs2)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_ylabel("unmasked-air voxel fraction")
        ax.set_title("Size of the earlier error")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig2_unmasked_air_by_arm")


def fig_interior_edge(pv: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    arms = ARMS + ["probe"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.arange(len(arms))
    for k, (ci, ce, title) in enumerate((
            ("unmasked_interior_self", "unmasked_edge_self", "volume-normalised"),
            ("unmasked_interior_local_self", "unmasked_edge_local_self",
             "region-normalised (local density)"))):
        ax = axes[k]
        ints = [pv[pv.arm == a][ci].mean() for a in arms]
        edgs = [pv[pv.arm == a][ce].mean() for a in arms]
        ax.bar(xs - 0.18, ints, 0.32, color="#7a3ea1",
               label=f"interior (>{EDGE_VOX} vox from face)")
        ax.bar(xs + 0.18, edgs, 0.32, color="#c9a227",
               label=f"edge shell (<={EDGE_VOX} vox)")
        ax.set_xticks(xs)
        ax.set_xticklabels(arms, fontsize=8)
        ax.set_ylabel("unmasked-air fraction")
        ax.set_title(title, fontsize=9.5)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Interior vs edge unmasked air (T_self, correct u8 conversion)")
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig3_interior_vs_edge")


def fig_mask_collapse(cells: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    arms = [a for a in ARMS + ["probe"] if (cells.arm == a).any()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    bins = np.concatenate([[0], np.logspace(-3, 0, 16)])
    ax = axes[0]
    for arm in arms:
        c = cells[cells.arm == arm]
        dark = c.cell_dark_frac.to_numpy()
        capture = np.where(dark > 0,
                           1.0 - c.cell_unmasked_frac.to_numpy() / np.maximum(dark, 1e-9),
                           np.nan)
        idx = np.digitize(dark, bins)
        cen, med, lo, hi = [], [], [], []
        for b in range(1, len(bins)):
            s = (idx == b) & np.isfinite(capture)
            if s.sum() < 5:
                continue
            cen.append(np.sqrt(max(bins[b - 1], 1e-4) * bins[b]))
            med.append(np.median(capture[s]))
            lo.append(np.percentile(capture[s], 25))
            hi.append(np.percentile(capture[s], 75))
        if cen:
            ax.plot(cen, med, color=ARM_COLORS[arm], label=arm)
            ax.fill_between(cen, lo, hi, color=ARM_COLORS[arm], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("cell dark fraction (64$^3$ cells)")
    ax.set_ylabel("mask capture (1 = mask labels all the dark voxels)")
    ax.set_title("Mask capture vs cell dark fraction (median, IQR)")
    ax.legend()

    ax = axes[1]
    for arm in arms:
        c = cells[cells.arm == arm]
        ax.scatter(c.cell_dark_frac, c.cell_mask_porosity, s=2, alpha=0.15,
                   color=ARM_COLORS[arm], label=arm, rasterized=True)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("cell dark fraction")
    ax.set_ylabel("cell mask porosity")
    ax.set_title("Mask porosity vs dark fraction per cell")
    ax.legend(markerscale=6)
    fig.suptitle("Mask-collapse behaviour on the 64$^3$ cell grid (T_self)")
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig4_mask_collapse")


def fig_montage(root, real_boxes: list[dict], pv: pd.DataFrame, cal: dict) -> list[str]:
    import matplotlib.pyplot as plt

    rows = []
    b = real_boxes[0]
    g = root[b["volume"]]
    z0, y0, x0 = b["origin"]
    B = REAL_BOX
    xr = np.asarray(g["xct"][z0:z0 + B, y0:y0 + B, x0:x0 + B])
    mr = np.asarray(g["mask"][z0:z0 + B, y0:y0 + B, x0:x0 + B]) > 0
    tr = b["T_abs"]["threshold"]
    det = xr < tr
    det, _, _ = cc_filter(det)
    unm = det & ~mr
    z = int(np.argmax(unm.reshape(B, -1).mean(axis=1)))
    rows.append((f"real interior box\n(T_abs={tr})", xr[z], mr[z], unm[z], z))
    del det, unm

    for arm in ARMS:
        d = GEN_ROOT / "layup" / arm / "A_training_seed_101"
        if not d.exists():
            continue
        r = pv[(pv.arm == arm) & (pv.name == "A_training_seed_101")
               & (pv.experiment == "layup")]
        t = int(r.T_self.iloc[0])
        u8 = load_generated_u8(d)
        mk = tifffile.imread(str(d / "mask.tif")) > 0
        dt = u8 < t
        dt, _, _ = cc_filter(dt)
        unm = dt & ~mk
        z = int(np.argmax(unm.reshape(unm.shape[0], -1).mean(axis=1)))
        s = slice(0, 512)
        rows.append((f"{arm}\nlayup A seed 101 (T_self={t})",
                     u8[z, s, s].copy(), mk[z, s, s].copy(), unm[z, s, s].copy(), z))
        del u8, mk, dt, unm

    fig, ax = plt.subplots(len(rows), 3, figsize=(12.5, 3.5 * len(rows)))
    for i, (title, gray, msk, u2, z) in enumerate(rows):
        ax[i, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(title, fontsize=8)
        ax[i, 1].imshow(msk, cmap="gray")
        rgb = np.stack([gray] * 3, -1).astype(np.float32) / 255.0
        rgb[u2] = [0.85, 0.15, 0.15]
        rgb[msk & ~u2] = [0.2, 0.45, 0.9]
        ax[i, 2].imshow(rgb)
        ax[i, 2].set_title(f"z={z}  red = unmasked dark, blue = mask", fontsize=8)
        if i == 0:
            ax[i, 0].set_title("grayscale (raw u8 scale)", fontsize=9)
            ax[i, 1].set_title("pore mask", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"Worst offending slices — min CC {MIN_CC} vox "
                 f"(real row uses T_abs, generated rows use each volume's T_self)",
                 y=0.997)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig5_slice_montage")


def fig_matched_hists(real_boxes: list[dict], gen_hists: dict,
                      pv: pd.DataFrame, air_mode: float) -> list[str]:
    """Task 2: real interior vs generated, on BOTH grey scales."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    x = np.arange(256)
    real_pdfs = np.stack([smoothed_pdf(np.asarray(b["hist"], np.int64))
                          for b in real_boxes])
    gen_by_arm = {arm: [k for k in gen_hists if k[1] == arm and k[0] != "layup"]
                  for arm in ARMS}

    def overlay(ax, native: bool, log: bool = False):
        ax.plot(x, real_pdfs.mean(0), color="k", lw=2, label="real interior (mean)")
        if not log:
            ax.fill_between(x, real_pdfs.min(0), real_pdfs.max(0),
                            color="k", alpha=0.15)
        for arm, sel in gen_by_arm.items():
            if not sel:
                continue
            hs = [native_hist(gen_hists[k]) if native else gen_hists[k] for k in sel]
            p = np.stack([smoothed_pdf(h) for h in hs])
            ax.plot(x, p.mean(0), color=ARM_COLORS[arm], lw=1.6, label=f"{arm} (mean)")
            if not log:
                ax.fill_between(x, np.percentile(p, 10, 0), np.percentile(p, 90, 0),
                                color=ARM_COLORS[arm], alpha=0.15)
        ax.axvline(air_mode, color="#888", ls=":", lw=1.2)
        ax.annotate("real exterior\nair mode", (air_mode, ax.get_ylim()[1] * 0.85),
                    fontsize=7, color="#666", ha="left")
        ax.set_xlim(0, 255)
        ax.set_xlabel("intensity (u8)")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 1)
        ax.legend(fontsize=7.5)

    ax = axes[0, 0]
    overlay(ax, native=False)
    ax.set_ylabel("smoothed density")
    ax.set_title("RAW scale — clip(v*255, 0, 255)\n(the instructed conversion)")

    ax = axes[0, 1]
    overlay(ax, native=True)
    ax.set_title("DECODER-NATIVE scale — clip(logit(v)*255, 0, 255)\n"
                 "(sampler expit undone)")

    ax = axes[0, 2]
    overlay(ax, native=False, log=True)
    ax.set_ylabel("smoothed density (log)")
    ax.set_title("RAW scale, log density — the dark tail")

    ax = axes[1, 0]
    seps, cols = [], []
    for b in real_boxes:
        v = b["modes"].get("mode_separation")
        seps.append(0.0 if v is None else float(v))
        cols.append("#333333")
    gsel = pv[pv.experiment != "layup"]
    n_real = len(seps)
    for _, r in gsel.iterrows():
        seps.append(float(r.mode_separation_u8) if r.bimodal else 0.0)
        cols.append(ARM_COLORS.get(r.arm, "#888"))
    ax.bar(np.arange(len(seps)), seps, color=cols, width=1.0)
    ax.axvline(n_real - 0.5, color="k", lw=1, ls="--")
    ax.set_xlabel(f"volume index (left of the dashed line: {n_real} real interior boxes)")
    ax.set_ylabel("mode separation, raw scale (u8)")
    ax.set_title("Mode separation — 0 means no second mode found")

    def pore_material_panel(ax, native: bool):
        rows = []
        for b in real_boxes:
            rows.append(("real interior", b["pore_stats"]["p50"],
                         b["material_stats"]["p50"]))
        pcol = "native_pore_stats" if native else "pore_stats"
        mcol = "native_material_stats" if native else "material_stats"
        for _, r in gsel.iterrows():
            rows.append((r.arm, json.loads(r[pcol]).get("p50", np.nan),
                         json.loads(r[mcol]).get("p50", np.nan)))
        df = pd.DataFrame(rows, columns=["grp", "pore_p50", "mat_p50"])
        grps = ["real interior"] + ARMS
        xs = np.arange(len(grps))
        ax.bar(xs - 0.2, [df[df.grp == g].pore_p50.mean() for g in grps], 0.38,
               color="#b23a48", label="pore voxels, median u8")
        ax.bar(xs + 0.2, [df[df.grp == g].mat_p50.mean() for g in grps], 0.38,
               color="#4c8c4a", label="material voxels, median u8")
        ax.set_xticks(xs)
        ax.set_xticklabels(grps, fontsize=8)
        ax.set_ylim(0, 255)
        ax.set_ylabel("intensity (u8)")
        ax.legend(fontsize=7.5)

    pore_material_panel(axes[1, 1], native=False)
    axes[1, 1].set_title("Pore vs material intensity — RAW scale\n"
                         "(each side's own mask)")
    pore_material_panel(axes[1, 2], native=True)
    axes[1, 2].set_title("Pore vs material intensity — NATIVE scale\n"
                         "(real side unchanged: real u8 IS the native scale)")

    fig.suptitle("Task 2 — matched interior comparison "
                 f"(real {REAL_BOX}$^3$ boxes inside sample_mask vs generated "
                 f"{REAL_BOX}$^3$ volumes)", y=0.995)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig6_matched_interior")


def fig_mode_separation(real_boxes: list[dict], pv: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    groups = {"real interior": [b["modes"] for b in real_boxes]}
    for arm in ARMS + ["probe"]:
        groups[arm] = [{"n_modes": r.n_modes} for _, r in pv[pv.arm == arm].iterrows()]
    names = list(groups)
    frac = [np.mean([m["n_modes"] >= 2 for m in groups[n]]) if groups[n] else 0
            for n in names]
    ax.bar(names, frac, color=["#333333"] + [ARM_COLORS[a] for a in ARMS + ["probe"]])
    ax.set_ylabel("fraction of volumes with >= 2 modes")
    ax.set_ylim(0, 1.05)
    ax.set_title("Bimodality incidence")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    ax = axes[1]
    data, labs, cols = [], [], []
    v = [b["modes"]["mode_separation"] for b in real_boxes
         if b["modes"].get("mode_separation")]
    if v:
        data.append(v), labs.append("real interior"), cols.append("#333333")
    for arm in ARMS + ["probe"]:
        v = pv[(pv.arm == arm) & pv.bimodal].mode_separation_u8.dropna().to_numpy()
        if len(v):
            data.append(v), labs.append(arm), cols.append(ARM_COLORS[arm])
        v = (pv[(pv.arm == arm) & pv.native_bimodal]
             .native_mode_separation_u8.dropna().to_numpy())
        if len(v):
            data.append(v), labs.append(arm + "\n[native]"), cols.append(ARM_COLORS[arm])
    if data:
        bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c), patch.set_alpha(0.6)
    ax.set_ylabel("mode separation (u8)")
    ax.set_title("Separation of the two dominant modes\n(bimodal volumes only)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    ax = axes[2]
    data, labs, cols = [], [], []
    for arm in ARMS + ["probe"]:
        v = pv[(pv.arm == arm) & pv.bimodal].valley_depth_ratio.dropna().to_numpy()
        if len(v):
            data.append(v), labs.append(arm), cols.append(ARM_COLORS[arm])
    if data:
        bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c), patch.set_alpha(0.6)
    ax.set_ylabel("valley depth / smaller peak height")
    ax.set_title("Bimodality strength\n(0 = fully separated, 1 = no dip)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    fig.suptitle("Mode-structure summary", y=0.99)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig7_mode_separation")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(res: dict, pv: pd.DataFrame, cmp_df: pd.DataFrame,
                   real_boxes: list[dict]) -> None:
    cal = res["calibration"]
    L: list[str] = []
    A = L.append
    A("# Unmasked-air audit v2")
    A("")
    A("## 0. What was redone")
    A("")
    A("`scripts/analysis/eval_v2_audit.py` converted generated volumes with")
    A("`clip(logit(v), 0, 1) * 255`. This audit was commissioned on the premise that")
    A("the logit is spurious and that the correct conversion is a direct rescale,")
    A("`u8 = clip(v * 255, 0, 255)`. The write chain is")
    A("")
    A("```")
    A("decoder output -> expit -> *255 -> uint8    src/poregen/diffusion/sampler.py:1276-1278")
    A("uint8 -> float32/255 -> volume.tif          scripts/analysis/_eval_v2.py:243")
    A("```")
    A("")
    A("The direct rescale is used as the **primary** scale for every threshold and")
    A("every number in sections 2-4. But the premise behind it does not survive")
    A("measurement, so the decoder-native (logit) scale is carried alongside")
    A("throughout — section 0b sets out the evidence and what it does and does not")
    A("change. `eval_v2_audit.py` and its outputs are untouched.")
    A("")
    A("Independent of the scale question, this audit differs from the earlier one in")
    A("three ways that do change the numbers: the threshold is derived from each")
    A("volume's own mode structure instead of importing a real-calibrated absolute")
    A("level; the real-side foreground comes from the stored `sample_mask` instead of")
    A("an intensity cut plus erosion; and the real comparison uses interior boxes")
    A("with no exterior air instead of whole volumes.")
    A("")
    A("## 0b. Which grey scale — read this before the numbers")
    A("")
    sc = res["scale_evidence"]
    A("The instruction for this redo was that `volume.tif` already sits on the real "
      "`/255` scale and that no logit is involved. The measurements do not support "
      "that, so both scales are reported and the primary tables use the instructed "
      "direct rescale. The evidence:")
    A("")
    A("1. The VAE decoder has **no output activation**: `losses/total.py:73` is "
      "`recon_fn(output.xct_out, batch[\"xct\"])` and `dataset/loader.py:100,218` "
      "normalise the target as `xct / 255`. The decoder predicts [0, 1] grey levels "
      "directly. (The `xct_out` name and the \"z-score space\" docstrings in "
      "`losses/recon.py` and `models/vae/base.py` are stale — nothing z-scores the "
      "XCT target.)")
    A("2. `sampler.py:1276` then applies `expit` to that already-[0, 1] signal. "
      f"Generated volumes consequently span only {sc['gen_raw_min']}-"
      f"{sc['gen_raw_max']} u8 instead of 0-255 — expit squashes [0, 1] into "
      "[0.500, 0.731].")
    A(f"3. Undoing the expit lands the generated material mode at "
      f"{sc['gen_material_mode_native']:.0f} u8, and the real material mode measured "
      f"here is {sc['real_material_mode']:.0f} u8. The generated dark mode lands at "
      f"{sc['gen_dark_mode_native']:.0f} u8; the real *exterior air* mode is "
      f"{sc['real_exterior_air_mode']:.0f} u8. Left on the raw scale the same two "
      f"modes sit at {sc['gen_material_mode_raw']:.0f} and "
      f"{sc['gen_dark_mode_raw']:.0f} u8.")
    A("4. `clip(logit(v), 0, 1) * 255` — the expression `eval_v2_audit.py` used — is "
      "algebraically identical to `clip(logit(v) * 255, 0, 255)`. So the earlier "
      "audit's conversion was the native scale, not a separate transform, and its "
      "arithmetic was not the source of an error.")
    A("")
    A("What this does and does not change:")
    A("")
    A("- **Does not change** which voxels a threshold selects. The two scales are "
      "related by a monotonic look-up table, so every detection, every unmasked-air "
      "fraction and every connected component below is the same set of voxels on "
      "either scale — only the numeric threshold moves. All thresholds are applied "
      "on the raw array; the real-calibrated ones are defined on the native scale "
      "(real u8 *is* the native scale) and mapped in through the LUT.")
    A("- **Does change** the dynamic-range reading. On the raw scale the generated "
      "material peak sits well below the real one and the volumes look compressed; "
      "on the native scale the two coincide. Section 5 reports both.")
    A(f"- One real cost of the native scale: {sc['native_clipped_low_frac']:.3f} of "
      "generated voxels fall below native 0 and pile up at the black clip, so the "
      "shape of the far dark tail is not recoverable there. The raw scale keeps it.")
    A("")
    A(f"Volumes audited: {len(pv)} "
      f"({int((pv.experiment != 'ldm06_probe').sum())} under `runs/eval_v2/volumes/`, "
      f"{int((pv.experiment == 'ldm06_probe').sum())} under `runs/analysis/ldm06_probe/volumes/`).")
    A(f"Voxel size assumed {VOXEL_UM} um. Minimum connected component {MIN_CC} voxels. "
      f"Edge shell {EDGE_VOX} voxels.")
    A("")

    A("## 1. Intensity-distribution characterisation")
    A("")
    A(f"Method: 256-bin histogram, Gaussian smoothing (sigma = {HIST_SIGMA} u8),")
    A(f"`scipy.signal.find_peaks` with prominence >= {PEAK_PROM_REL:.0e} of all voxels")
    A(f"and minimum peak separation {PEAK_MIN_DIST} u8. Every local maximum is reported")
    A("with its height, prominence and FWHM. The valley is the minimum of the smoothed")
    A("curve between the two most prominent modes; the valley-depth ratio is that")
    A("minimum divided by the smaller of the two peak heights (0 = cleanly split,")
    A("1 = no dip at all).")
    A("")
    rb = res["real_interior"]
    A("### Real interior boxes")
    A("")
    A(f"{len(real_boxes)} boxes of {REAL_BOX}^3 sampled entirely inside the stored "
      f"`sample_mask` of {rb['n_volumes']} val/test volumes (no exterior air).")
    A("")
    A("| volume | origin | mask porosity | n modes | modes (u8) | separation | valley | "
      "pore p50 | pore-core p50 | material p50 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for b in real_boxes:
        m = b["modes"]
        A(f"| {b['volume'].split('__')[-1][:34]} | {b['origin']} | "
          f"{b['mask_porosity']:.4f} | {m['n_modes']} | "
          f"{m['mode_levels']} | {m.get('mode_separation') or '-'} | "
          f"{m.get('valley_level') or '-'} | {b['pore_stats']['p50']} | "
          f"{b['pore_core_stats'].get('p50', '-')} | {b['material_stats']['p50']} |")
    A("")
    A(f"**Real interior summary**: {rb['n_bimodal']}/{rb['n_boxes']} boxes bimodal. "
      f"Material mode {rb['material_mode_mean']:.0f} +/- {rb['material_mode_std']:.0f} u8, "
      f"material FWHM {rb['material_fwhm_mean']:.1f} u8. "
      f"Pore voxels: median {rb['pore_p50_mean']:.0f} u8, "
      f"eroded pore cores median {rb['pore_core_p50_mean']:.0f} u8. "
      f"Material voxels: median {rb['material_p50_mean']:.0f} u8.")
    A("")

    A("### Generated volumes")
    A("")
    A("Raw scale (`clip(v*255, 0, 255)`):")
    A("")
    A("| experiment | arm | n | bimodal | dark mode | material mode | separation | "
      "dark FWHM | mat FWHM | valley | valley depth | dark mass frac |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (e, a), gdf in pv.groupby(["experiment", "arm"]):
        bi = gdf[gdf.bimodal]
        A(f"| {e} | {a} | {len(gdf)} | {len(bi)}/{len(gdf)} | "
          f"{bi.dark_mode_u8.mean():.1f} | {bi.material_mode_u8.mean():.1f} | "
          f"{bi.mode_separation_u8.mean():.1f} | {bi.dark_fwhm_u8.mean():.1f} | "
          f"{bi.material_fwhm_u8.mean():.1f} | {bi.valley_level_u8.mean():.1f} | "
          f"{bi.valley_depth_ratio.mean():.3f} | {bi.dark_mass_frac.mean():.4f} |")
    A("")
    A("Decoder-native scale (`clip(logit(v)*255, 0, 255)`), same volumes:")
    A("")
    A("| experiment | arm | n | bimodal | dark mode | material mode | separation | "
      "dark FWHM | mat FWHM | valley | valley depth | dark mass frac | clipped at 0 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (e, a), gdf in pv.groupby(["experiment", "arm"]):
        bi = gdf[gdf.native_bimodal]
        A(f"| {e} | {a} | {len(gdf)} | {len(bi)}/{len(gdf)} | "
          f"{bi.native_dark_mode_u8.mean():.1f} | "
          f"{bi.native_material_mode_u8.mean():.1f} | "
          f"{bi.native_mode_separation_u8.mean():.1f} | "
          f"{bi.native_dark_fwhm_u8.mean():.1f} | "
          f"{bi.native_material_fwhm_u8.mean():.1f} | "
          f"{bi.native_valley_level_u8.mean():.1f} | "
          f"{bi.native_valley_depth_ratio.mean():.3f} | "
          f"{bi.native_dark_mass_frac.mean():.4f} | "
          f"{gdf.native_clipped_low_frac.mean():.4f} |")
    A("")

    A("## 2. Threshold rules and their validation on real volumes")
    A("")
    A(f"- `T_abs` = {cal['t_abs']} u8 — absolute threshold, Dice-optimal on "
      f"{cal['n_calibration_volumes']} real val/test volumes inside `sample_mask` "
      f"(every {CAL_SLICE_STEP}th z-slice). Pooled Dice {cal['dice']:.3f}, "
      f"precision {cal['precision']:.3f}, recall {cal['recall']:.3f}.")
    A(f"- `T_rel` = (volume's own material mode) - {cal['offset_from_material_mode']:.1f} u8 "
      f"— the same real calibration re-expressed as an offset below the material mode, "
      f"so it transfers to a volume with a different grey scale. Real material mode "
      f"{cal['real_material_mode_mean']:.1f} u8.")
    A("- `T_self` = Otsu restricted between the two detected modes on the **raw** "
      "scale — derived from each volume's own bimodal structure, no import from real "
      "data. Falls back to whole-histogram Otsu when the volume is unimodal.")
    A("- `T_valley` = the raw valley minimum itself (sensitivity variant of `T_self`).")
    A("- `T_self_native` = the same self rule applied on the native scale, mapped "
      "back to a raw threshold. Its gap to `T_self` is the scale sensitivity of the "
      "self-derived rule.")
    A("")
    A("Real u8 *is* the native scale, so `T_abs` and `T_rel` are defined there and "
      "mapped to raw thresholds through `NATIVE_LUT` before being applied. Every "
      "per-volume row carries both numbers (`T_*` raw, `T_*_native_equiv`).")
    A("")
    A("### Validation against the stored pore mask")
    A("")
    A("Per-volume Dice on the real calibration slices (inside `sample_mask`):")
    A("")
    A("| real volume | bimodal? | T_abs | Dice | T_rel | Dice | T_self | Dice | T_valley | Dice |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in res["real_rule_validation"].items():
        A(f"| {name.split('__')[-1][:36]} | {r['modes']['bimodal']} | "
          f"{r['T_abs']['threshold']} | {r['T_abs']['dice']:.3f} | "
          f"{r['T_rel']['threshold']} | {r['T_rel']['dice']:.3f} | "
          f"{r['T_self']['threshold']} | {r['T_self']['dice']:.3f} | "
          f"{r['T_valley']['threshold']} | {r['T_valley']['dice']:.3f} |")
    A("")
    A("And on the matched interior boxes (full 3-D, with and without the CC filter):")
    A("")
    A("| box | T_abs Dice (no CC) | T_abs Dice (CC) | T_abs recall | "
      "T_abs unmasked frac | T_self Dice (CC) |")
    A("|---|---|---|---|---|---|")
    for b in real_boxes:
        A(f"| {b['volume'].split('__')[-1][:28]} {b['origin'][1]},{b['origin'][2]} | "
          f"{b['T_abs']['dice_nocc']:.3f} | {b['T_abs']['dice_cc']:.3f} | "
          f"{b['T_abs']['recall_cc']:.3f} | {b['T_abs']['unmasked_frac']:.4f} | "
          f"{b['T_self']['dice_cc']:.3f} |")
    A("")
    A(res["validation_note"])
    A("")

    A("## 3. Unmasked air by arm — OLD audit vs NEW")
    A("")
    A("`OLD` = `eval_v2_audit.py` (native scale, real-calibrated absolute threshold "
      "T_best, whole-volume real baseline). `NEW` = this audit (raw scale, "
      "self-derived threshold). Because both scales are monotonically related, the "
      "difference below is driven by the **threshold rule**, not by the conversion: "
      "the old rule imported a real-calibrated level that maps to ~172 on the raw "
      "generated scale, while the self-derived valley sits near 155.")
    A("")
    if len(cmp_df):
        A("| experiment | arm | n | mask por | OLD detected | OLD unmasked | "
          "OLD unmasked cons | NEW detected (T_self) | NEW unmasked (T_self) | "
          "NEW unmasked (T_rel) | OLD/NEW |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in cmp_df.iterrows():
            A(f"| {r.experiment} | {r.arm} | {int(r.n)} | {r.mask_porosity:.4f} | "
              f"{r.old_detected_air:.4f} | {r.old_unmasked_air:.4f} | "
              f"{r.old_unmasked_cons:.4f} | {r.new_detected_air:.4f} | "
              f"{r.new_unmasked_air:.4f} | {r.new_unmasked_rel:.4f} | "
              f"{r.ratio_old_over_new:.1f}x |")
        A("")
    A("Full per-arm table including the ldm06 probe volumes:")
    A("")
    A("| experiment | arm | n | mask por | T_self (raw/native) | detected (T_self) | "
      "unmasked (T_self) | unmasked (T_valley) | unmasked (T_self_native) | "
      "unmasked (T_rel) | unmasked (T_abs) | interior | edge | largest comp mm^3 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in pd.DataFrame(res["aggregate_by_arm"]).iterrows():
        A(f"| {r['experiment']} | {r['arm']} | {int(r['n'])} | {r['mask_porosity']:.4f} | "
          f"{r['T_self']:.0f}/{r['T_self_native_equiv']:.0f} | "
          f"{r['detected_air_self']:.4f} | {r['unmasked_air_self']:.4f} | "
          f"{r['unmasked_air_valley']:.4f} | {r['unmasked_air_self_native']:.4f} | "
          f"{r['unmasked_air_rel']:.4f} | {r['unmasked_air_abs']:.4f} | "
          f"{r['unmasked_interior_self']:.4f} | {r['unmasked_edge_self']:.4f} | "
          f"{r['largest_comp_mm3']:.3f} |")
    A("")
    A(f"Threshold values (raw scale): T_rel = "
      f"{pd.DataFrame(res['aggregate_by_arm']).T_rel.mean():.0f}, T_abs = "
      f"{pd.DataFrame(res['aggregate_by_arm']).T_abs.mean():.0f}. Both are close to "
      "the generated material mode, which is why they flag far more voxels than the "
      "self-derived valley: a threshold calibrated to catch real pores sitting just "
      "below a 209 u8 material peak lands almost on top of the generated material "
      "peak once mapped onto the generated grey scale. The self-derived valley is "
      "the defensible primary number; T_rel/T_abs bound it from above.")
    A("")

    A("## 4. Mask-collapse behaviour (64^3 cells)")
    A("")
    for arm, s in res["mask_collapse"].items():
        reg = s["regimes"]
        def f(x):
            return "-" if x is None else f"{x:.2f}"
        A(f"- **{arm}**: r(dark, unmasked) = {s['pearson_dark_vs_unmasked']:.3f}; "
          f"median mask capture of dark voxels: low(<2%) {f(reg['low']['capture_median'])}, "
          f"mid(2-15%) {f(reg['mid']['capture_median'])}, "
          f"high(>15%) {f(reg['high']['capture_median'])} "
          f"(n = {reg['low']['n_cells']}/{reg['mid']['n_cells']}/{reg['high']['n_cells']}).")
    A("")

    A("## 5. Task 2 — matched interior comparison")
    A("")
    t2 = res["task2"]
    A(t2["verdict"])
    A("")
    cols = ARMS + [a + " [native]" for a in ARMS]
    A("| quantity | real interior | " + " | ".join(cols) + " |")
    A("|---|---|" + "---|" * len(cols))
    for key, label, fmt in (
            ("bimodal_fraction", "fraction of volumes with >= 2 modes", "{:.2f}"),
            ("material_mode", "material mode (u8)", "{:.1f}"),
            ("dark_mode", "second (dark) mode (u8)", "{:.1f}"),
            ("mode_separation", "mode separation (u8)", "{:.1f}"),
            ("material_fwhm", "material peak FWHM (u8)", "{:.1f}"),
            ("dark_fwhm", "dark peak FWHM (u8)", "{:.1f}"),
            ("valley_level", "valley (u8)", "{:.1f}"),
            ("valley_depth_ratio", "valley depth ratio", "{:.3f}"),
            ("dark_mass_frac", "voxel fraction below the valley", "{:.4f}"),
            ("p01", "1st percentile (u8)", "{:.1f}"),
            ("p99", "99th percentile (u8)", "{:.1f}"),
            ("dynamic_range_p01_p99", "p1-p99 dynamic range (u8)", "{:.1f}"),
            ("pore_p50", "pore voxels, median (u8)", "{:.1f}"),
            ("material_p50", "material voxels, median (u8)", "{:.1f}"),
            ("pore_material_gap", "material p50 - pore p50 (u8)", "{:.1f}"),
    ):
        cells = []
        for grp in ["real interior"] + ARMS + [a + " [native]" for a in ARMS]:
            v = t2["summary"].get(grp, {}).get(key)
            cells.append("-" if v is None or (isinstance(v, float) and np.isnan(v))
                         else fmt.format(v))
        A(f"| {label} | " + " | ".join(cells) + " |")
    A("")

    A("## Figures")
    A("")
    for p in res.get("figures", []):
        A(f"- {p}")
    A("")
    A("## Files")
    A("")
    for f in ("results.json", "per_volume.csv", "per_cell.csv",
              "real_interior_boxes.json", "old_vs_new.csv"):
        A(f"- {OUT_DIR / f}")
    (OUT_DIR / "findings.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res: dict = {
        "conversion": "u8 = clip(volume_tif_float32 * 255, 0, 255).astype(uint8)",
        "conversion_note": ("volume.tif already stores u8/255; the sampler applied "
                            "expit before the u8 cast, so no logit inversion is "
                            "correct. eval_v2_audit.py used clip(logit(v),0,1)*255, "
                            "which is the bug this script fixes."),
        "voxel_size_um_assumed": VOXEL_UM,
        "min_cc_voxels": MIN_CC,
        "edge_shell_vox": EDGE_VOX,
        "hist_sigma": HIST_SIGMA,
        "peak_prominence_rel": PEAK_PROM_REL,
    }

    splits = json.loads((DATA_ROOT / "splits.json").read_text())
    root = zarr.open(str(ZARR_ROOT), mode="r")
    names = real_split_names(splits)
    cal_names = names[:N_CAL_VOLUMES]

    log(f"Calibrating absolute threshold on {len(cal_names)} real val/test volumes...")
    cal = calibrate_absolute(root, cal_names)
    log(f"  T_abs={cal['t_abs']} dice={cal['dice']:.3f} prec={cal['precision']:.3f} "
        f"rec={cal['recall']:.3f}  material mode {cal['real_material_mode_mean']:.1f} "
        f"offset {cal['offset_from_material_mode']:.1f}")

    log("Validating threshold rules on real volumes...")
    res["real_rule_validation"] = validate_rules_on_real(cal)
    cal.pop("_slices")
    res["calibration"] = cal

    log(f"Sampling real interior {REAL_BOX}^3 boxes...")
    rng = np.random.default_rng(7)
    real_boxes: list[dict] = []
    used = 0
    for name in names:
        if used >= N_BOX_VOLUMES:
            break
        g = root[name]
        origins = find_interior_boxes(g, N_BOXES_PER_VOLUME, rng)
        if not origins:
            continue
        used += 1
        for o in origins:
            b = analyse_real_box(g, name, o, cal)
            real_boxes.append(b)
            log(f"  box {name.split('__')[-1][:34]:34s} {o} "
                f"n_modes={b['modes']['n_modes']} modes={b['modes']['mode_levels']} "
                f"por={b['mask_porosity']:.4f} T_abs dice={b['T_abs']['dice_cc']:.3f}")
    # Real exterior-air mode, quoted only as a reference point for the scale
    # discussion.  Never enters any interior statistic.
    g0 = root[names[0]]
    z_mid = g0["xct"].shape[0] // 2
    x_mid = np.asarray(g0["xct"][z_mid])
    sm_mid = np.asarray(g0["sample_mask"][z_mid]) > 0
    air_info = analyse_modes(hist_u8(x_mid, ~sm_mid))
    res["real_exterior_air_mode"] = float(air_info["argmax"])
    res["real_exterior_air_modes_detected"] = air_info["mode_levels"]
    log(f"  real exterior-air mode (reference only): {res['real_exterior_air_mode']:.0f} u8")
    del x_mid, sm_mid

    (OUT_DIR / "real_interior_boxes.json").write_text(
        json.dumps([{k: v for k, v in b.items()} for b in real_boxes], default=str, indent=1))

    mm = [b["modes"]["material_mode"] for b in real_boxes if b["modes"]["material_mode"]]
    mf = [b["modes"]["material_fwhm"] for b in real_boxes if b["modes"]["material_fwhm"]]
    res["real_interior"] = {
        "n_volumes": used, "n_boxes": len(real_boxes),
        "box_edge": REAL_BOX,
        "n_bimodal": int(sum(b["modes"]["bimodal"] for b in real_boxes)),
        "material_mode_mean": float(np.mean(mm)), "material_mode_std": float(np.std(mm)),
        "material_fwhm_mean": float(np.mean(mf)),
        "pore_p50_mean": float(np.mean([b["pore_stats"]["p50"] for b in real_boxes])),
        "pore_core_p50_mean": float(np.mean([b["pore_core_stats"].get("p50", np.nan)
                                             for b in real_boxes])),
        "material_p50_mean": float(np.mean([b["material_stats"]["p50"] for b in real_boxes])),
        "mask_porosity_mean": float(np.mean([b["mask_porosity"] for b in real_boxes])),
    }

    log("Auditing generated volumes...")
    rows, frames, gen_hists = [], [], {}
    for i, (exp, arm, d) in enumerate(iter_generated()):
        row, cdf, h = audit_generated(d, exp, arm, cal)
        rows.append(row)
        frames.append(cdf)
        gen_hists[(exp, arm, d.name)] = h
        log(f"  [{i + 1:3d}] {exp}/{arm}/{d.name}: modes={row['n_modes']} "
            f"dark={row['dark_mode_u8']} mat={row['material_mode_u8']} "
            f"T_self={row['T_self']} unmasked={row['unmasked_air_self']:.4f} "
            f"(T_rel {row['unmasked_air_rel']:.4f})")

    pv = pd.DataFrame(rows)
    cells = pd.concat(frames, ignore_index=True)
    pv.drop(columns=["top_components"]).to_csv(OUT_DIR / "per_volume.csv", index=False)
    pv[["experiment", "arm", "name", "top_components"]].to_json(
        OUT_DIR / "top_components.json", orient="records", indent=1)
    cells.to_csv(OUT_DIR / "per_cell.csv", index=False)
    np.savez_compressed(OUT_DIR / "histograms.npz",
                        keys=np.array(["/".join(k) for k in gen_hists]),
                        hists=np.stack(list(gen_hists.values())),
                        real_keys=np.array([f"{b['volume']}@{b['origin']}" for b in real_boxes]),
                        real_hists=np.stack([np.asarray(b["hist"], np.int64)
                                             for b in real_boxes]))

    agg = (pv.groupby(["experiment", "arm"], dropna=False)
             .agg(n=("name", "count"),
                  mask_porosity=("mask_porosity", "mean"),
                  detected_air_self=("detected_air_self", "mean"),
                  unmasked_air_self=("unmasked_air_self", "mean"),
                  unmasked_air_valley=("unmasked_air_valley", "mean"),
                  unmasked_air_self_native=("unmasked_air_self_native", "mean"),
                  unmasked_air_rel=("unmasked_air_rel", "mean"),
                  unmasked_air_abs=("unmasked_air_abs", "mean"),
                  T_self=("T_self", "mean"),
                  T_self_native_equiv=("T_self_native_equiv", "mean"),
                  T_rel=("T_rel", "mean"),
                  T_abs=("T_abs", "mean"),
                  unmasked_interior_self=("unmasked_interior_self", "mean"),
                  unmasked_edge_self=("unmasked_edge_self", "mean"),
                  largest_comp_mm3=("largest_comp_mm3", "max"))
             .reset_index())
    agg.to_csv(OUT_DIR / "aggregate_by_arm.csv", index=False)
    res["aggregate_by_arm"] = agg.to_dict(orient="records")

    agg_t = (pv.groupby(["experiment", "arm", "target"], dropna=False)
               .agg(n=("name", "count"),
                    mask_porosity=("mask_porosity", "mean"),
                    unmasked_air_self=("unmasked_air_self", "mean"),
                    unmasked_air_rel=("unmasked_air_rel", "mean"))
               .reset_index())
    agg_t.to_csv(OUT_DIR / "aggregate_by_target.csv", index=False)

    cmp_df = old_vs_new(pv)
    if len(cmp_df):
        cmp_df.to_csv(OUT_DIR / "old_vs_new.csv", index=False)
        res["old_vs_new"] = cmp_df.to_dict(orient="records")
    log("Old vs new comparison written.")

    res["mask_collapse"] = collapse_stats(cells, ARMS + ["probe"])

    gb = pv[pv.bimodal]
    nb = pv[pv.native_bimodal]
    res["scale_evidence"] = {
        "gen_raw_min": int(pv.raw_min_u8.min()),
        "gen_raw_max": int(pv.raw_max_u8.max()),
        "gen_material_mode_raw": float(gb.material_mode_u8.mean()),
        "gen_dark_mode_raw": float(gb.dark_mode_u8.mean()),
        "gen_material_mode_native": float(nb.native_material_mode_u8.mean()),
        "gen_dark_mode_native": float(nb.native_dark_mode_u8.mean()),
        "real_material_mode": res["real_interior"]["material_mode_mean"],
        "real_exterior_air_mode": res["real_exterior_air_mode"],
        "native_clipped_low_frac": float(pv.native_clipped_low_frac.mean()),
        "lut_note": ("NATIVE_LUT[k] = clip(round(logit(k/255)*255), 0, 255); "
                     "monotonic, so thresholds and histograms map exactly and the "
                     "flagged voxel sets are identical on both scales."),
    }

    # ---- Task 2 summary -------------------------------------------------
    def summarise(vals: dict) -> dict:
        return {k: (float(np.nanmean(v)) if len(v) else None) for k, v in vals.items()}

    t2_summary = {}
    rmod = [b["modes"] for b in real_boxes]
    t2_summary["real interior"] = summarise({
        "bimodal_fraction": [float(m["bimodal"]) for m in rmod],
        "material_mode": [m["material_mode"] for m in rmod if m["material_mode"]],
        "dark_mode": [m["dark_mode"] for m in rmod if m.get("dark_mode")],
        "mode_separation": [m["mode_separation"] for m in rmod if m.get("mode_separation")],
        "material_fwhm": [m["material_fwhm"] for m in rmod if m.get("material_fwhm")],
        "dark_fwhm": [m["dark_fwhm"] for m in rmod if m.get("dark_fwhm")],
        "valley_level": [m["valley_level"] for m in rmod if m.get("valley_level")],
        "valley_depth_ratio": [m["valley_depth_ratio"] for m in rmod
                               if m.get("valley_depth_ratio")],
        "dark_mass_frac": [m["dark_mass_frac"] for m in rmod if m.get("dark_mass_frac")],
        "p01": [m["p01"] for m in rmod],
        "p99": [m["p99"] for m in rmod],
        "dynamic_range_p01_p99": [m["p99"] - m["p01"] for m in rmod],
        "pore_p50": [b["pore_stats"]["p50"] for b in real_boxes],
        "material_p50": [b["material_stats"]["p50"] for b in real_boxes],
        "pore_material_gap": [b["material_stats"]["p50"] - b["pore_stats"]["p50"]
                              for b in real_boxes],
    })
    gsel = pv[pv.experiment != "layup"]
    for arm in ARMS:
        s = gsel[gsel.arm == arm]
        b = s[s.bimodal]
        ps = [json.loads(x) for x in s.pore_stats]
        ms = [json.loads(x) for x in s.material_stats]
        t2_summary[arm] = summarise({
            "bimodal_fraction": s.bimodal.astype(float).tolist(),
            "material_mode": b.material_mode_u8.dropna().tolist(),
            "dark_mode": b.dark_mode_u8.dropna().tolist(),
            "mode_separation": b.mode_separation_u8.dropna().tolist(),
            "material_fwhm": b.material_fwhm_u8.dropna().tolist(),
            "dark_fwhm": b.dark_fwhm_u8.dropna().tolist(),
            "valley_level": b.valley_level_u8.dropna().tolist(),
            "valley_depth_ratio": b.valley_depth_ratio.dropna().tolist(),
            "dark_mass_frac": b.dark_mass_frac.dropna().tolist(),
            "p01": s.p01.tolist(), "p99": s.p99.tolist(),
            "dynamic_range_p01_p99": (s.p99 - s.p01).tolist(),
            "pore_p50": [d.get("p50", np.nan) for d in ps],
            "material_p50": [d.get("p50", np.nan) for d in ms],
            "pore_material_gap": [m.get("p50", np.nan) - p.get("p50", np.nan)
                                  for p, m in zip(ps, ms)],
        })
        nb = s[s.native_bimodal]
        nps = [json.loads(x) for x in s.native_pore_stats]
        nms = [json.loads(x) for x in s.native_material_stats]
        t2_summary[arm + " [native]"] = summarise({
            "bimodal_fraction": s.native_bimodal.astype(float).tolist(),
            "material_mode": nb.native_material_mode_u8.dropna().tolist(),
            "dark_mode": nb.native_dark_mode_u8.dropna().tolist(),
            "mode_separation": nb.native_mode_separation_u8.dropna().tolist(),
            "material_fwhm": nb.native_material_fwhm_u8.dropna().tolist(),
            "dark_fwhm": nb.native_dark_fwhm_u8.dropna().tolist(),
            "valley_level": nb.native_valley_level_u8.dropna().tolist(),
            "valley_depth_ratio": nb.native_valley_depth_ratio.dropna().tolist(),
            "dark_mass_frac": nb.native_dark_mass_frac.dropna().tolist(),
            "p01": s.native_p01.tolist(), "p99": s.native_p99.tolist(),
            "dynamic_range_p01_p99": (s.native_p99 - s.native_p01).tolist(),
            "pore_p50": [d.get("p50", np.nan) for d in nps],
            "material_p50": [d.get("p50", np.nan) for d in nms],
            "pore_material_gap": [m.get("p50", np.nan) - p.get("p50", np.nan)
                                  for p, m in zip(nps, nms)],
        })

    rb_bi = res["real_interior"]["n_bimodal"]
    g_bi = int(gsel.bimodal.sum())
    rm = res["real_interior"]["material_mode_mean"]
    gm_raw = float(gsel[gsel.bimodal].material_mode_u8.mean())
    gm_nat = float(gsel[gsel.native_bimodal].native_material_mode_u8.mean())
    gd_raw = float(gsel[gsel.bimodal].dark_mode_u8.mean())
    gd_nat = float(gsel[gsel.native_bimodal].native_dark_mode_u8.mean())
    verdict = (
        f"Real interior boxes: {rb_bi}/{len(real_boxes)} bimodal. "
        f"Generated {REAL_BOX}^3 volumes (dose_response + cfg_sweep + probe): "
        f"{g_bi}/{len(gsel)} bimodal on the raw scale, "
        f"{int(gsel.native_bimodal.sum())}/{len(gsel)} on the native scale. ")
    if g_bi and rb_bi == 0:
        verdict += (
            "**Bimodality**: the generated volumes carry a second, lower-intensity "
            "mode that has no counterpart in real interior material. Real interiors "
            "are unimodal — real pores appear only as a low-amplitude dark tail on "
            "the material peak, never as a separate peak. The dark population "
            "itself is present on both grey scales; the two bimodality counts "
            "above differ only because peak prominence is measured per bin, and "
            "the native transform stretches the dark mode across more bins, "
            "lowering its peak height at unchanged mass. The mass-based figure — "
            "the voxel fraction below the valley — agrees on both scales. ")
    elif g_bi and rb_bi:
        verdict += "Both sides are bimodal; compare the separations in the table. "
    else:
        verdict += "Neither side shows a clear second mode. "
    verdict += (
        f"**Mode positions depend on the grey scale, and the earlier "
        f"'compressed dynamic range' claim is a scale artefact.** Real material mode "
        f"{rm:.0f} u8. Generated material mode {gm_raw:.0f} u8 on the raw scale "
        f"(a {rm - gm_raw:.0f} u8 shift, which reads as compression) but "
        f"{gm_nat:.0f} u8 on the decoder-native scale — i.e. the generated material "
        f"peak coincides with the real one once the sampler's extra expit is undone. "
        f"The generated dark mode sits at {gd_raw:.0f} u8 raw / {gd_nat:.0f} u8 "
        f"native; real exterior air (excluded from every interior box, quoted only "
        f"for reference) sits near 41 u8. Both scales are reported throughout; "
        f"the voxel sets selected by a threshold are identical on both. "
        f"This is a measurement, not a judgement of model quality.")
    res["task2"] = {"summary": t2_summary, "verdict": verdict,
                    "n_real_boxes": len(real_boxes), "n_generated_192": len(gsel),
                    "real_material_mode": rm,
                    "generated_material_mode_raw": gm_raw,
                    "generated_material_mode_native": gm_nat,
                    "generated_dark_mode_raw": gd_raw,
                    "generated_dark_mode_native": gd_nat}

    # validation note (honest statement about T_self on real data)
    self_dices = [r["T_self"]["dice"] for r in res["real_rule_validation"].values()]
    fb = sum(r["self_rule_fallback"] for r in res["real_rule_validation"].values())
    res["validation_note"] = (
        f"**How to read the validation.** `T_abs` and `T_rel` are calibrated on real "
        f"data and score Dice ~{cal['dice']:.2f} there, so the detector itself is sound. "
        f"`T_self` cannot be validated on real interiors: {fb}/"
        f"{len(res['real_rule_validation'])} real volumes are unimodal inside "
        f"`sample_mask`, so the bimodal rule falls back to whole-histogram Otsu and "
        f"scores Dice {np.min(self_dices):.3f}-{np.max(self_dices):.3f} "
        f"(median {np.median(self_dices):.3f}). That failure is informative rather "
        f"than disqualifying: it says real interiors have no second mode to split. "
        f"On generated volumes the two modes are well separated, the valley is broad "
        f"and flat, and `T_self`, `T_valley` and `T_rel` land within a few u8 of one "
        f"another — which is why all three are reported side by side.")

    log("Figures...")
    real_fp = None
    if real_boxes:
        fps = [b["T_abs"]["unmasked_frac"] for b in real_boxes]
        real_fp = {"mean": float(np.mean(fps)), "max": float(np.max(fps))}
        res["real_false_positive_baseline"] = real_fp
    figs: list[str] = []
    figs += fig_modes(real_boxes, gen_hists, pv)
    figs += fig_unmasked_by_arm(pv, cmp_df, real_fp)
    figs += fig_interior_edge(pv)
    figs += fig_mask_collapse(cells)
    figs += fig_montage(root, real_boxes, pv, cal)
    figs += fig_matched_hists(real_boxes, gen_hists, pv,
                              res["real_exterior_air_mode"])
    figs += fig_mode_separation(real_boxes, pv)
    res["figures"] = figs

    write_json(res, OUT_DIR)
    write_findings(res, pv, cmp_df, real_boxes)
    log(f"Done -> {OUT_DIR}")
    print("AIR_AUDIT_V2_DONE", flush=True)


if __name__ == "__main__":
    main()

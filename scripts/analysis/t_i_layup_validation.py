"""T-I — Validation of measured ply orientation against the expert ground truth.

`data/layup_ground_truth.json` gives the true stacking sequence of 78 of the 80
volumes.  74 of them (every Pegaso and every Nacho volume) share ONE sequence,
so the two families that T-G and T-H reported as *different materials* are in
fact identical.  This test asks three questions:

1.  **Consistency.**  Does the measured laminate thickness divided by the true
    ply count agree with the measured porosity pitch, per family?  A mismatch
    between families means the voxel size differs between acquisition
    campaigns, which would make any *fixed* spatial-frequency band probe a
    different physical scale in each family.
2.  **Extraction.**  Build the best per-ply-block orientation estimate we can,
    fit one global in-plane rotation offset per volume (plus the unknown
    handedness and the unknown face ordering), and score the per-ply angular
    error against the ground truth, with a permutation null.
3.  **Diagnosis.**  Why did T-G and T-H separate Pegaso from Nacho?

Method summary
--------------
Per volume, two 1024x1024 in-plane windows (at 30 % and 70 % of the y extent,
centred in x) are read for every z-slice.  Each window is mean-removed, Hann
windowed and Fourier transformed.  The power spectrum is binned into a
**12 (log-radial) x 180 (angular, 1 degree)** map, each bin divided by the
number of spectrum pixels in it, so the map is a mean power *density* with no
square-grid bias.  The angular axis is rolled by 90 degrees into the real-space
texture frame.  The same is done for a **pore-suppressed** copy of the slice, in
which every voxel of the binary pore mask is replaced by the mean of the
non-pore voxels.  Storing the radial axis (rather than one fixed annulus) lets
any wavelength band -- fixed, or scaled by the volume's own ply pitch -- be
evaluated afterwards without touching the zarr again.

The orientation estimator that this test recommends is deliberately different
from T-G's.  T-G reduced the whole angular density of a slice to its second
moment, which is dominated by the *volume-constant* part of the density (a
mounting-angle plus artifact term).  Here the volume-mean angular density is
subtracted first, and the estimate is the peak of the residual:

    R_k(theta) = <h(theta)>_(ply k) - <h(theta)>_(volume)
    theta_k    = argmax R_k

A quasi-isotropic layup visits every ply direction about equally often, so its
volume mean is nearly isotropic and carries almost no layup information; what
it does carry is the constant bias.  Subtracting it is what makes the ply
sequence readable.

Usage
-----
    python scripts/analysis/t_i_layup_validation.py --smoke 3
    python scripts/analysis/t_i_layup_validation.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT, REPO, ZARR_ROOT, detrend_profile, periodogram,
    savefig, set_style, write_findings, write_json, plt,
)

TEST_ID = "T-I"
OUT_DIR = OUT_ROOT / TEST_ID
CACHE = OUT_DIR / "angular_maps.npz"
SLAB_CACHE = OUT_DIR / "slab_features.npz"
GT_PATH = REPO / "data" / "layup_ground_truth.json"

WINDOW = 1024
WINDOW_FRACS = (0.3, 0.7)      # window centres along y
CHUNK_Z = 24
N_ANG = 180                    # 1-degree angular bins
LAM_EDGES = np.geomspace(4.0, 320.0, 13)   # 12 log-spaced wavelength bins, voxels
REF_PITCH = 19.6               # voxels, the T-A ply pitch of the large families
# Voxel size, confirmed by the domain expert for every acquisition campaign.
# It makes the nominal ply pitch 0.508 mm / 25 um = 20.3 voxels (sequence A) and
# 0.25 mm / 25 um = 10.0 voxels (sequence B).
VOXEL_SIZE_UM = 25.0
# Primary band, expressed as a multiple of the volume's OWN ply pitch.
# 0.82-3.27 x 19.6 = the 16-64-voxel "tow band" of T-G/T-H, so the fixed-band
# and scaled-band results are directly comparable on Pegaso/Nacho.
BAND_PITCH = (16.0 / REF_PITCH, 64.0 / REF_PITCH)
BAND_FIXED = (16.0, 64.0)
SMOOTH_DEG = 4.0               # circular smoothing of the angular residual
AXIS_EXCISE = 1                # +-1 degree around 0 and 90 removed


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth() -> dict:
    with open(GT_PATH) as fh:
        gt = json.load(fh)
    return gt


def family(volume_id: str) -> str:
    if "Pegaso" in volume_id:
        return "Airbus_Panel_Pegaso"
    if "Nacho" in volume_id:
        return "Fabricacion_Nacho_05"
    return "Juan_Ignacio"


FAMILIES = ("Airbus_Panel_Pegaso", "Fabricacion_Nacho_05", "Juan_Ignacio")


# ---------------------------------------------------------------------------
# Per-slice angular / radial power map
# ---------------------------------------------------------------------------

def _geometry(n: int):
    """Hann window, flat spectrum-bin index and per-bin pixel counts."""
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    with np.errstate(divide="ignore"):
        lam = np.where(rad > 0, 1.0 / np.maximum(rad, 1e-12), np.inf)
    sel = (lam >= LAM_EDGES[0]) & (lam <= LAM_EDGES[-1])
    rbin = np.digitize(lam[sel], LAM_EDGES) - 1
    rbin = np.clip(rbin, 0, len(LAM_EDGES) - 2)
    # wave-vector angle, then +90 degrees into the real-space texture frame
    wang = np.rad2deg(np.arctan2(
        np.broadcast_to(fy, (n, n))[sel], np.broadcast_to(fx, (n, n))[sel]))
    abin = np.floor((wang + 90.0) % 180.0).astype(np.int64)
    abin = np.clip(abin, 0, N_ANG - 1)
    flat = rbin * N_ANG + abin
    counts = np.bincount(flat, minlength=(len(LAM_EDGES) - 1) * N_ANG)
    return win, sel, flat, counts.astype(np.float64)


def angular_map(img: np.ndarray, win, sel, flat, counts) -> np.ndarray:
    a = (img - img.mean()) * win
    p = (np.abs(np.fft.fft2(a)) ** 2)[sel]
    s = np.bincount(flat, weights=p, minlength=counts.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = s / counts
    return np.nan_to_num(d).reshape(len(LAM_EDGES) - 1, N_ANG).astype(np.float32)


def volume_maps(volume_id: str) -> dict:
    t0 = time.time()
    g = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]
    xct_a, mask_a = g["xct"], g["mask"]
    D, H, W = xct_a.shape
    n = WINDOW
    x0 = max(0, W // 2 - n // 2)
    y0s = [int(np.clip(round(f * H) - n // 2, 0, H - n)) for f in WINDOW_FRACS]
    win, sel, flat, counts = _geometry(n)
    nr = len(LAM_EDGES) - 1

    raw = np.zeros((D, nr, N_ANG), dtype=np.float32)
    nop = np.zeros((D, nr, N_ANG), dtype=np.float32)
    fg_frac = np.zeros(D)
    pore_frac = np.zeros(D)

    # Otsu threshold on a coarse histogram of the first window.
    hist = np.zeros(256, dtype=np.int64)
    for z in range(0, D, CHUNK_Z):
        blk = np.asarray(xct_a[z:min(z + CHUNK_Z, D), y0s[0]:y0s[0] + n,
                               x0:x0 + n])[::2, ::4, ::4]
        hist += np.bincount(blk.ravel(), minlength=256)
    pr = hist / max(hist.sum(), 1)
    lv = np.arange(256, dtype=np.float64)
    om, mu = np.cumsum(pr), np.cumsum(lv * pr)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = np.where((om > 0) & (om < 1), (mu[-1] * om - mu) ** 2 / (om * (1 - om)), 0.0)
    thr = int(np.nanargmax(sb))

    for z in range(0, D, CHUNK_Z):
        z1 = min(z + CHUNK_Z, D)
        for wi, y0 in enumerate(y0s):
            blk = np.asarray(xct_a[z:z1, y0:y0 + n, x0:x0 + n])
            mblk = np.asarray(mask_a[z:z1, y0:y0 + n, x0:x0 + n]).astype(bool)
            for i in range(z1 - z):
                k = z + i
                im = blk[i].astype(np.float32)
                pm = mblk[i]
                if wi == 0:
                    fg_frac[k] = float((blk[i] > thr).mean())
                    pore_frac[k] = float(pm.mean())
                raw[k] += angular_map(im, win, sel, flat, counts)
                if pm.any():
                    im2 = im.copy()
                    im2[pm] = float(im[~pm].mean()) if (~pm).any() else float(im.mean())
                else:
                    im2 = im
                nop[k] += angular_map(im2, win, sel, flat, counts)
    raw /= len(y0s)
    nop /= len(y0s)
    return {
        "volume_id": volume_id,
        "shape": np.array(xct_a.shape),
        "y_offsets": np.array(y0s),
        "otsu_threshold": thr,
        "fg_frac": fg_frac,
        "pore_frac": pore_frac,
        "raw": raw,
        "nopore": nop,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Geometry of the laminate: interior extent and ply pitch
# ---------------------------------------------------------------------------

def interior_extent(fg: np.ndarray) -> tuple[int, int]:
    """First and last slice fully inside the specimen."""
    thr = 0.9 * float(np.nanmax(fg))
    idx = np.where(fg >= thr)[0]
    return int(idx[0]), int(idx[-1])


def measured_pitch(pore_frac: np.ndarray, a: int, b: int,
                   expected: float | None = None) -> float:
    """Dominant period of the porosity profile inside the laminate.

    The search is restricted to +-30 % of the pitch the material implies
    (ply thickness / voxel size).  Without that guard the periodogram of a
    ~190-sample profile locks onto the second harmonic in a minority of
    volumes, which then mis-scales every downstream wavelength band.
    """
    lo, hi = (7.0, 45.0) if expected is None else (0.70 * expected, 1.30 * expected)
    y = detrend_profile(pore_frac[a:b + 1], 3)
    per, pw = periodogram(y)
    band = (per >= lo) & (per <= hi) & np.isfinite(per)
    if band.sum() < 3:
        return float("nan")
    return float(per[band][np.argmax(pw[band])])


# ---------------------------------------------------------------------------
# Band extraction and the orientation estimators
# ---------------------------------------------------------------------------

LAM_CENTRES = np.sqrt(LAM_EDGES[:-1] * LAM_EDGES[1:])


def band_hist(maps: np.ndarray, lam_lo: float, lam_hi: float) -> np.ndarray:
    """Average the radial bins inside a wavelength band -> (D, 180) density."""
    sel = (LAM_CENTRES >= lam_lo) & (LAM_CENTRES <= lam_hi)
    if not sel.any():
        sel = np.argmin(np.abs(LAM_CENTRES - 0.5 * (lam_lo + lam_hi)))
        sel = np.arange(len(LAM_CENTRES)) == sel
    h = maps[:, sel, :].mean(axis=1)
    return h


THETA = np.arange(N_ANG) + 0.5
_SM = np.exp(-0.5 * (((np.arange(N_ANG) + N_ANG // 2) % N_ANG - N_ANG // 2) / SMOOTH_DEG) ** 2)
_SM /= _SM.sum()
_SM_F = np.fft.fft(_SM)


def _smooth(x: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(np.fft.fft(x, axis=-1) * _SM_F, axis=-1))


def prepare_hist(h: np.ndarray) -> np.ndarray:
    """Normalise each slice to unit mean and excise the FFT-axis bins."""
    h = h / np.maximum(np.nanmean(h, axis=1, keepdims=True), 1e-30)
    h = h.copy()
    for c in (0, 90):
        for d in range(-AXIS_EXCISE, AXIS_EXCISE + 1):
            h[:, (c + d) % N_ANG] = np.nan
    # interpolate the excised bins circularly so the smoothing stays local
    good = np.isfinite(h[0])
    xi = np.arange(N_ANG)
    for i in range(h.shape[0]):
        h[i, ~good] = np.interp(xi[~good], xi[good], h[i, good], period=N_ANG)
    return h


def block_edges(a: int, b: int, n_blocks: int) -> np.ndarray:
    return np.round(np.linspace(a, b + 1, n_blocks + 1)).astype(int)


def block_means(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.array([h[edges[i]:edges[i + 1]].mean(axis=0)
                     for i in range(len(edges) - 1)])


def estimate_angles(B: np.ndarray) -> dict:
    """Per-ply-block orientation, four estimators, from the block densities."""
    R = _smooth(B - B.mean(axis=0, keepdims=True))
    peak = THETA[np.argmax(R, axis=1)]
    strength = R.max(axis=1) - R.min(axis=1)

    e2 = np.exp(-2j * np.deg2rad(THETA))
    e4 = np.exp(-4j * np.deg2rad(THETA))
    c2_res = R @ e2 / N_ANG
    c4_res = R @ e4 / N_ANG
    c2_abs = (B @ e2) / (B.sum(axis=1) + 1e-30)      # T-G's estimator

    return {
        "peak_residual": (peak, strength),
        "c2_residual": (np.rad2deg(np.angle(c2_res)) / 2 % 180, np.abs(c2_res)),
        "c4_residual": (np.rad2deg(np.angle(c4_res)) / 4 % 45, np.abs(c4_res)),
        "c2_absolute": (np.rad2deg(np.angle(c2_abs)) / 2 % 180, np.abs(c2_abs)),
        "residual": R,
        "c2_absolute_phasor": c2_abs,
    }


def refine_edges(h: np.ndarray, a: int, b: int, n_blocks: int) -> np.ndarray:
    """Shift the block grid to maximise ply-to-ply angular contrast.

    Ground-truth independent: the objective is the mean residual amplitude,
    which is large when each block holds one ply and small when blocks straddle
    two plies.
    """
    ply = (b + 1 - a) / n_blocks
    best, best_edges = -np.inf, block_edges(a, b, n_blocks)
    for da in np.arange(-0.45, 0.46, 0.10) * ply:
        for db in np.arange(-0.45, 0.46, 0.10) * ply:
            aa = int(round(a + da))
            bb = int(round(b + db))
            if aa < 0 or bb >= h.shape[0] or bb - aa < 4 * n_blocks:
                continue
            e = block_edges(aa, bb, n_blocks)
            if np.any(np.diff(e) < 2):
                continue
            B = block_means(h, e)
            R = _smooth(B - B.mean(axis=0, keepdims=True))
            obj = float(np.mean(R.max(axis=1) - R.min(axis=1)))
            if obj > best:
                best, best_edges = obj, e
    return best_edges


# ---------------------------------------------------------------------------
# Scoring against the ground truth
# ---------------------------------------------------------------------------

def _fit_one(pred: np.ndarray, truth: np.ndarray, w: np.ndarray,
             sgn: int, rev: bool) -> dict:
    p = pred * sgn
    t = truth[::-1] if rev else truth
    ph = np.exp(2j * np.deg2rad(t - p))
    res = np.sum(w * ph)
    delta = float(np.rad2deg(np.angle(res) / 2))
    err = np.rad2deg(np.angle(np.exp(2j * np.deg2rad(p + delta - t)))) / 2
    return {"median_abs_error": float(np.median(np.abs(err))), "errors": err,
            "offset_deg": delta, "sign": sgn, "reversed": rev,
            "fitted": (p + delta) % 180, "truth_used": t,
            "resultant_length": float(np.abs(res) / (np.sum(w) + 1e-30)),
            "_ph": ph, "_w": w}


def fit_offset(pred: np.ndarray, truth: np.ndarray, w: np.ndarray,
               n_boot: int = 500, seed: int = 1) -> dict:
    """Best single rotation offset over the sign / face-order ambiguity.

    The expert lists the sequence from one face to the other, and we do not know
    which face is z=0; and the in-plane angle sign convention is not tied to the
    image frame.  So four hypotheses are tried and the best is reported, with
    the margin to the runner-up, which is how confidently the two discrete
    ambiguities are resolved.  The permutation null uses exactly the same
    four-way fit, so this freedom is priced in.

    The offset uncertainty is a bootstrap over plies: the offset is ONE scalar
    fitted to n_plies observations, so it is far better determined than any
    single per-ply angle.
    """
    cands = [_fit_one(pred, truth, w, s, r) for s in (1, -1) for r in (False, True)]
    cands.sort(key=lambda c: c["median_abs_error"])
    best = cands[0]
    best["hypotheses"] = [
        {"sign": c["sign"], "reversed": c["reversed"], "offset_deg": c["offset_deg"],
         "median_abs_error": c["median_abs_error"]} for c in cands]
    best["hypothesis_margin_deg"] = float(
        cands[1]["median_abs_error"] - cands[0]["median_abs_error"])

    n = len(pred)
    ph, ww = best["_ph"], best["_w"]
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        d = np.rad2deg(np.angle(np.sum(ww[idx] * ph[idx], axis=1)) / 2)
        dev = np.rad2deg(np.angle(np.exp(2j * np.deg2rad(d - best["offset_deg"])))) / 2
        best["offset_bootstrap_sd_deg"] = float(np.std(dev))
        best["offset_ci95_deg"] = [float(x) for x in np.percentile(dev, [2.5, 97.5])]
    else:
        best["offset_bootstrap_sd_deg"] = float("nan")
        best["offset_ci95_deg"] = [float("nan"), float("nan")]
    # analytic circular standard error of the mean doubled angle
    R = best["resultant_length"]
    best["offset_circ_se_deg"] = float(
        np.rad2deg(np.sqrt(max(1.0 - R * R, 1e-9) / max(n * R * R, 1e-9))) / 2)
    del best["_ph"], best["_w"]
    return best


def fit_offset_loo(pred: np.ndarray, truth: np.ndarray, w: np.ndarray,
                   sgn: int, rev: bool) -> np.ndarray:
    """Leave-one-ply-out errors at a fixed sign/order hypothesis."""
    p = pred * sgn
    t = truth[::-1] if rev else truth
    ph = np.exp(2j * np.deg2rad(t - p))
    out = np.empty(len(p))
    for i in range(len(p)):
        m = np.ones(len(p), bool)
        m[i] = False
        delta = np.rad2deg(np.angle(np.sum(w[m] * ph[m])) / 2)
        out[i] = np.rad2deg(np.angle(np.exp(2j * np.deg2rad(p[i] + delta - t[i])))) / 2
    return out


CLASSES = np.array([0.0, 45.0, 90.0, 135.0])


def classify(angles: np.ndarray) -> np.ndarray:
    d = np.abs(np.rad2deg(np.angle(np.exp(2j * np.deg2rad(angles[:, None] - CLASSES[None, :])))) / 2)
    return CLASSES[np.argmin(d, axis=1)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compute_maps(volume_ids: list[str], workers: int) -> dict:
    recs = {}
    if workers <= 1:
        for v in volume_ids:
            recs[v] = volume_maps(v)
            print(f"  {v}  {recs[v]['seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(volume_maps, volume_ids):
                recs[r["volume_id"]] = r
                print(f"  {r['volume_id']}  {r['seconds']:.1f}s", flush=True)
    return recs


def analyse(recs: dict, gt: dict, n_perm: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    truth = gt["volumes"]
    per_volume: dict[str, dict] = {}

    for vid, r in recs.items():
        fg = np.asarray(r["fg_frac"])
        pore = np.asarray(r["pore_frac"])
        a, b = interior_extent(fg)
        gtv = truth.get(vid, {})
        plies = gtv.get("plies") or None
        n_gt = len(plies) if plies else None
        extent = b - a + 1
        t_mm = gtv.get("ply_thickness_mm")
        expected = (1000.0 * t_mm / VOXEL_SIZE_UM) if t_mm else None
        pitch = measured_pitch(pore, a, b, expected)

        rec = {
            "volume_id": vid,
            "family": family(vid),
            "shape": [int(x) for x in r["shape"]],
            "z_interior": [a, b],
            "z_extent": extent,
            "measured_pitch_voxels": pitch,
            "sequence_id": gtv.get("sequence_id"),
            "material": gtv.get("material"),
            "ply_thickness_mm": t_mm,
            "n_plies_gt": n_gt,
            "thickness_per_ply_voxels": (extent / n_gt) if n_gt else None,
            "n_blocks_from_pitch": int(round(extent / pitch)) if pitch == pitch else None,
            "mean_pore_frac": float(np.mean(pore[a:b + 1])),
            "expected_pitch_voxels": expected,
            # --- physical scale, from the newly supplied ply thickness --------
            "voxel_size_um_from_pitch": (1000.0 * t_mm / pitch) if (t_mm and pitch == pitch) else None,
            "voxel_size_um_from_extent": (1000.0 * t_mm * n_gt / extent) if (t_mm and n_gt) else None,
            "predicted_extent_from_pitch": (n_gt * pitch) if (n_gt and pitch == pitch) else None,
            "extent_excess_voxels": (extent - n_gt * pitch) if (n_gt and pitch == pitch) else None,
            "extent_excess_plies": ((extent - n_gt * pitch) / pitch) if (n_gt and pitch == pitch) else None,
        }

        for tag, maps in (("raw", r["raw"]), ("nopore", r["nopore"])):
            lo, hi = (BAND_PITCH[0] * pitch, BAND_PITCH[1] * pitch) if pitch == pitch else BAND_FIXED
            for band_name, (blo, bhi) in (("scaled", (lo, hi)), ("fixed", BAND_FIXED)):
                h = prepare_hist(band_hist(maps, blo, bhi))
                key = f"{tag}_{band_name}"
                rec[key] = {"band_voxels": [blo, bhi]}
                if n_gt is None:
                    continue
                edges = refine_edges(h, a, b, n_gt)
                B = block_means(h, edges)
                est = estimate_angles(B)
                rec[key]["block_edges"] = edges.tolist()
                rec[key]["constant_to_residual"] = float(
                    np.abs(np.mean(est["c2_absolute_phasor"]))
                    / (np.mean(np.abs(est["c2_absolute_phasor"]
                                      - np.mean(est["c2_absolute_phasor"]))) + 1e-30))
                rec[key]["constant_angle_deg"] = float(
                    np.rad2deg(np.angle(np.mean(est["c2_absolute_phasor"]))) / 2 % 180)
                rec[key]["mean_residual_amplitude"] = float(
                    np.mean(est["residual"].max(axis=1) - est["residual"].min(axis=1)))
                g = np.asarray(plies, float) % 180
                for ename in ("peak_residual", "c2_residual", "c2_absolute"):
                    ang, w = est[ename]
                    f = fit_offset(ang, g, w)
                    loo = fit_offset_loo(ang, g, w, f["sign"], f["reversed"])
                    nulls = [fit_offset(ang, rng.permutation(g), w)["median_abs_error"]
                             for _ in range(n_perm)]
                    rec[key][ename] = {
                        "angles_deg": ang.tolist(),
                        "weights": w.tolist(),
                        "offset_deg": f["offset_deg"],
                        "offset_bootstrap_sd_deg": f["offset_bootstrap_sd_deg"],
                        "offset_ci95_deg": f["offset_ci95_deg"],
                        "offset_circ_se_deg": f["offset_circ_se_deg"],
                        "resultant_length": f["resultant_length"],
                        "hypotheses": f["hypotheses"],
                        "hypothesis_margin_deg": f["hypothesis_margin_deg"],
                        "sign": f["sign"],
                        "reversed": bool(f["reversed"]),
                        "fitted_deg": f["fitted"].tolist(),
                        "truth_deg": f["truth_used"].tolist(),
                        "errors_deg": f["errors"].tolist(),
                        "median_abs_error": f["median_abs_error"],
                        "loo_errors_deg": loo.tolist(),
                        "loo_median_abs_error": float(np.median(np.abs(loo))),
                        "null_median": float(np.median(nulls)),
                        "null_p05": float(np.percentile(nulls, 5)),
                        "p_value": float((np.sum(np.asarray(nulls) <= f["median_abs_error"]) + 1)
                                         / (n_perm + 1)),
                    }
        per_volume[vid] = rec
    return per_volume


def summarise(per_volume: dict) -> dict:
    out: dict = {"geometry": {}, "extraction": {}, "diagnosis": {}}

    # --- geometry -----------------------------------------------------------
    for fam in FAMILIES:
        vs = [r for r in per_volume.values() if r["family"] == fam]
        tpp = np.array([r["thickness_per_ply_voxels"] for r in vs
                        if r["thickness_per_ply_voxels"]], float)
        pit = np.array([r["measured_pitch_voxels"] for r in vs], float)
        ext = np.array([r["z_extent"] for r in vs], float)
        nb = np.array([r["n_blocks_from_pitch"] for r in vs if r["n_blocks_from_pitch"]], float)
        ngt = np.array([r["n_plies_gt"] for r in vs if r["n_plies_gt"]], float)
        out["geometry"][fam] = {
            "n_volumes": len(vs),
            "z_extent_median": float(np.median(ext)),
            "thickness_per_ply_median": float(np.median(tpp)) if tpp.size else None,
            "thickness_per_ply_iqr": [float(x) for x in np.percentile(tpp, [25, 75])] if tpp.size else None,
            "measured_pitch_median": float(np.median(pit)),
            "measured_pitch_iqr": [float(x) for x in np.percentile(pit, [25, 75])],
            "pitch_over_thickness_per_ply": float(np.median(pit[:len(tpp)] / tpp)) if tpp.size else None,
            "n_blocks_from_pitch_median": float(np.median(nb)) if nb.size else None,
            "n_plies_gt": sorted({int(x) for x in ngt}) if ngt.size else [],
            "blocks_match_gt_fraction": float(np.mean(nb[:len(ngt)] == ngt)) if ngt.size else None,
        }
    ref = out["geometry"]["Airbus_Panel_Pegaso"]["measured_pitch_median"]
    out["geometry"]["relative_pitch_vs_Pegaso"] = {
        fam: out["geometry"][fam]["measured_pitch_median"] / ref for fam in FAMILIES}

    # --- physical scale, from the expert ply thickness ----------------------
    phys: dict = {}
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and r["voxel_size_um_from_pitch"]]
        if not vs:
            continue
        vox = np.array([r["voxel_size_um_from_pitch"] for r in vs])
        voxe = np.array([r["voxel_size_um_from_extent"] for r in vs])
        med = float(np.median(vox))
        mad = float(np.median(np.abs(vox - med))) or 1e-9
        outl = [{"volume_id": r["volume_id"], "voxel_size_um": r["voxel_size_um_from_pitch"],
                 "robust_z": float((r["voxel_size_um_from_pitch"] - med) / (1.4826 * mad))}
                for r in vs if abs(r["voxel_size_um_from_pitch"] - med) > 3 * 1.4826 * mad]
        phys[fam] = {
            "n": len(vs),
            "ply_thickness_mm": vs[0]["ply_thickness_mm"],
            "voxel_size_um_from_pitch_median": med,
            "voxel_size_um_from_pitch_iqr": [float(x) for x in np.percentile(vox, [25, 75])],
            "voxel_size_um_from_pitch_range": [float(vox.min()), float(vox.max())],
            "voxel_size_um_from_extent_median": float(np.median(voxe)),
            "outliers_3mad": outl,
        }
    if phys:
        allv = np.array([r["voxel_size_um_from_pitch"] for r in per_volume.values()
                         if r["voxel_size_um_from_pitch"]])
        phys["all_volumes"] = {
            "n": int(allv.size),
            "median_um": float(np.median(allv)),
            "iqr_um": [float(x) for x in np.percentile(allv, [25, 75])],
            "range_um": [float(allv.min()), float(allv.max())],
        }
        pm = [phys[f]["voxel_size_um_from_pitch_median"] for f in FAMILIES if f in phys]
        phys["max_family_ratio"] = float(max(pm) / min(pm))
        phys["verdict"] = (
            "SHARED SCALE: every campaign lands within a few per cent of one voxel size, "
            "so Juan_Ignacio's shorter pitch is its thinner 0.25 mm ply, not a finer scan"
            if max(pm) / min(pm) < 1.35 else
            "DIFFERENT SCALES: the campaigns were scanned at materially different voxel sizes")
    out["geometry"]["physical_scale"] = phys

    # --- thickness cross-check, per volume, flagged individually ------------
    flags = []
    for r in per_volume.values():
        if r["extent_excess_plies"] is None:
            continue
        if abs(r["extent_excess_plies"]) > 0.75:
            flags.append({
                "volume_id": r["volume_id"], "family": r["family"],
                "z_extent": r["z_extent"],
                "predicted_extent": r["predicted_extent_from_pitch"],
                "excess_voxels": r["extent_excess_voxels"],
                "excess_plies": r["extent_excess_plies"],
            })
    ex = np.array([r["extent_excess_plies"] for r in per_volume.values()
                   if r["extent_excess_plies"] is not None])
    out["geometry"]["thickness_cross_check"] = {
        "excess_plies_median": float(np.median(ex)),
        "excess_plies_iqr": [float(x) for x in np.percentile(ex, [25, 75])],
        "n_flagged_gt_0_75_ply": len(flags),
        "flagged": sorted(flags, key=lambda d: -abs(d["excess_plies"]))[:15],
        "note": ("a positive excess means the fg>=0.9*max z extent is longer than "
                 "n_plies x measured pitch, i.e. surface resin / partial edge slices "
                 "are included in the extent"),
    }

    # --- extraction ---------------------------------------------------------
    for key in ("raw_scaled", "raw_fixed", "nopore_scaled"):
        for ename in ("peak_residual", "c2_residual", "c2_absolute"):
            block: dict = {}
            for fam in FAMILIES:
                errs, loos, meds, nulls, pvals, offs, signs, revs = [], [], [], [], [], [], [], []
                for r in per_volume.values():
                    if r["family"] != fam or ename not in r.get(key, {}):
                        continue
                    e = r[key][ename]
                    errs.append(np.abs(e["errors_deg"]))
                    loos.append(np.abs(e["loo_errors_deg"]))
                    meds.append(e["median_abs_error"])
                    nulls.append(e["null_median"])
                    pvals.append(e["p_value"])
                    offs.append(e["offset_deg"])
                    signs.append(e["sign"])
                    revs.append(e["reversed"])
                if not meds:
                    continue
                E = np.concatenate(errs)
                L = np.concatenate(loos)
                block[fam] = {
                    "n_volumes": len(meds),
                    "n_plies": int(E.size),
                    "median_abs_error_pooled": float(np.median(E)),
                    "median_of_volume_medians": float(np.median(meds)),
                    "loo_median_abs_error_pooled": float(np.median(L)),
                    "frac_within_10": float(np.mean(E < 10)),
                    "frac_within_15": float(np.mean(E < 15)),
                    "frac_within_22_5": float(np.mean(E < 22.5)),
                    "loo_frac_within_22_5": float(np.mean(L < 22.5)),
                    "null_median_of_volume_medians": float(np.median(nulls)),
                    "frac_volumes_p_below_0_05": float(np.mean(np.asarray(pvals) < 0.05)),
                    "fitted_offset_median": float(np.median(offs)),
                    "fitted_offset_circ_std_deg": float(
                        np.rad2deg(np.sqrt(-2 * np.log(np.abs(np.mean(
                            np.exp(2j * np.deg2rad(np.asarray(offs)))))))) / 2),
                    "sign_plus_fraction": float(np.mean(np.asarray(signs) > 0)),
                    "reversed_fraction": float(np.mean(revs)),
                }
            out["extraction"][f"{key}.{ename}"] = block

    # --- the rotation offset: one scalar per volume, with uncertainty -------
    off: dict = {}
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and "peak_residual" in r.get("raw_scaled", {})]
        if not vs:
            continue
        e = [r["raw_scaled"]["peak_residual"] for r in vs]
        d = np.array([x["offset_deg"] for x in e])
        sd = np.array([x["offset_bootstrap_sd_deg"] for x in e])
        mg = np.array([x["hypothesis_margin_deg"] for x in e])
        Rl = np.array([x["resultant_length"] for x in e])
        ph = np.exp(2j * np.deg2rad(d))
        off[fam] = {
            "n": len(vs),
            "offset_median_deg": float(np.rad2deg(np.angle(np.mean(ph))) / 2 % 180),
            "offset_spread_circ_sd_deg": float(
                np.rad2deg(np.sqrt(-2 * np.log(max(np.abs(np.mean(ph)), 1e-9)))) / 2),
            "bootstrap_sd_median_deg": float(np.median(sd)),
            "bootstrap_sd_p90_deg": float(np.percentile(sd, 90)),
            "frac_bootstrap_sd_below_10deg": float(np.mean(sd < 10)),
            "resultant_length_median": float(np.median(Rl)),
            "hypothesis_margin_median_deg": float(np.median(mg)),
            "frac_margin_below_2deg": float(np.mean(mg < 2.0)),
            "sign_plus_fraction": float(np.mean([x["sign"] > 0 for x in e])),
            "reversed_fraction": float(np.mean([x["reversed"] for x in e])),
        }
    out["rotation_offset"] = off

    # --- +45 vs -45 discrimination -----------------------------------------
    disc: dict = {}
    for fam in FAMILIES:
        tot = corr = 0
        conf = {c: {str(int(d)): 0 for d in CLASSES} for c in [0, 45, 90, 135]}
        for r in per_volume.values():
            if r["family"] != fam or "peak_residual" not in r.get("raw_scaled", {}):
                continue
            e = r["raw_scaled"]["peak_residual"]
            pred = classify(np.asarray(e["fitted_deg"]))
            tru = classify(np.asarray(e["truth_deg"], float) % 180)
            for p, t in zip(pred, tru):
                conf[int(t)][str(int(p))] += 1
                if t in (45.0, 135.0):
                    tot += 1
                    corr += int(p == t)
        disc[fam] = {
            "confusion_true_to_pred": conf,
            "n_offaxis_plies": tot,
            "plus45_vs_minus45_accuracy": (corr / tot) if tot else None,
            "chance": 0.5,
        }
    out["extraction"]["pm45_discrimination"] = disc

    # --- diagnosis ----------------------------------------------------------
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and "constant_to_residual" in r.get("raw_fixed", {})]
        if not vs:
            continue
        ctr = np.array([r["raw_fixed"]["constant_to_residual"] for r in vs])
        cang = np.array([r["raw_fixed"]["constant_angle_deg"] for r in vs])
        amp = np.array([r["raw_fixed"]["mean_residual_amplitude"] for r in vs])
        axis_dist = np.minimum(np.abs(cang), np.minimum(np.abs(cang - 90), np.abs(cang - 180)))
        err_raw = np.array([r["raw_scaled"]["peak_residual"]["median_abs_error"] for r in vs])
        err_nop = np.array([r["nopore_scaled"]["peak_residual"]["median_abs_error"] for r in vs])
        por = np.array([r["mean_pore_frac"] for r in vs])
        out["diagnosis"][fam] = {
            "constant_to_residual_median": float(np.median(ctr)),
            "constant_angle_median_deg": float(np.median(cang)),
            "constant_angle_dist_to_image_axis_median_deg": float(np.median(axis_dist)),
            "mean_residual_amplitude_median": float(np.median(amp)),
            "mean_pore_frac_median": float(np.median(por)),
            "err_raw_median": float(np.median(err_raw)),
            "err_pore_suppressed_median": float(np.median(err_nop)),
            "err_delta_pore_suppression": float(np.median(err_nop - err_raw)),
            "corr_error_vs_porosity": float(np.corrcoef(por, err_raw)[0, 1]) if len(vs) > 2 else None,
        }

    # processing variants
    var: dict = {}
    for tag, test in (("pegaso_cleaned", lambda v: "Pegaso" in v and "cleaned" in v),
                      ("pegaso_plain", lambda v: "Pegaso" in v and "cleaned" not in v and "volumen" not in v),
                      ("pegaso_volumen", lambda v: "Pegaso" in v and "volumen" in v),
                      ("ji_rotated", lambda v: "Juan_Ignacio" in v and "rotated" in v),
                      ("ji_plain", lambda v: "Juan_Ignacio" in v and "rotated" not in v)):
        vs = [r for r in per_volume.values()
              if test(r["volume_id"]) and "peak_residual" in r.get("raw_scaled", {})]
        if not vs:
            continue
        var[tag] = {
            "n": len(vs),
            "err_median": float(np.median([r["raw_scaled"]["peak_residual"]["median_abs_error"] for r in vs])),
            "offset_median_deg": float(np.median([r["raw_scaled"]["peak_residual"]["offset_deg"] for r in vs])),
            "pitch_median": float(np.median([r["measured_pitch_voxels"] for r in vs])),
        }
    out["diagnosis"]["processing_variants"] = var

    # Was T-A's "9.6-voxel second harmonic" really the Juan_Ignacio ply pitch
    # pooled into a stacked spectrum?  Test per family on T-A's cached grey and
    # porosity z-profiles: does the 8-13-voxel band beat the 15-25-voxel band?
    ta = OUT_ROOT / "T-A" / "fine_profiles.npz"
    if ta.exists():
        prof = np.load(ta, allow_pickle=True)["profiles"].item()
        chk: dict = {}
        for fam in FAMILIES:
            for sig in ("gray", "phi"):
                vals = []
                for vid, p in prof.items():
                    if family(vid) != fam or vid not in per_volume:
                        continue
                    a, b = per_volume[vid]["z_interior"]
                    y = detrend_profile(np.asarray(p["z"][sig])[a:b + 1], 3)
                    per, pw = periodogram(y)
                    lo = (per >= 8) & (per <= 13)
                    hi = (per >= 15) & (per <= 25)
                    if lo.sum() and hi.sum():
                        vals.append(float(pw[lo].max() / (pw[hi].max() + 1e-30)))
                if vals:
                    chk[f"{fam}.{sig}"] = {
                        "n": len(vals),
                        "median_power_ratio_8_13_over_15_25": float(np.median(vals)),
                        "frac_volumes_short_period_wins": float(np.mean(np.asarray(vals) > 1)),
                    }
        out["diagnosis"]["ta_9_6_voxel_peak_check"] = chk
    return out


# ---------------------------------------------------------------------------
# Stage 2 — slab-domain orientation estimators
#
# T-G and T-H both estimated orientation from a SINGLE z-slice and then pooled.
# A ply is ~20 voxels thick and the tows run straight through it, so averaging
# the images inside one ply (never crossing a ply boundary) adds the tow signal
# coherently while the noise adds incoherently.  That is the obvious SNR win
# neither earlier test took, and it is the first thing this stage does.
#
# Four genuinely different estimators run on the same slab-mean image, so they
# can be compared honestly:
#   fft_slab   — angular power density of the 2-D spectrum (band limited)
#   st_slab    — real-space gradient structure tensor, orientation histogram
#   radon_slab — variance of the Radon projection against projection angle
#   pore_axes  — principal axes of the connected pore components in the slab
#                (morphological, no Fourier transform, no voxel-grid spike)
# fft_slice is the incoherent per-slice average of stage 1, kept as the control
# that isolates what slab averaging alone buys.
#
# Angle convention for every estimator: degrees mod 180, 0 = image +x,
# 90 = image +y, and the angle is the direction the TEXTURE runs (the tow
# direction), not the direction of the gradient or of the wave vector.
# The Radon axis is calibrated in `calibrate()`: skimage returns the projection
# angle, which is (90 - texture angle) mod 180.
# ---------------------------------------------------------------------------

ALL_ESTIMATORS = ("fft_slice", "fft_slab", "fft_slab_nowin", "st_slab", "radon_slab", "pore_axes", "combined")
SLAB_ESTIMATORS = ("fft_slice", "fft_slab", "fft_slab_nowin",
                   "st_slab", "radon_slab", "pore_axes")
# The window x excision factorial.  T-G and T-H both Hann-windowed AND excised
# the exact-0/90 bins.  Windowing suppresses the leakage cross that the excision
# was aimed at; the excision additionally deletes the bins where a 0-degree or a
# 90-degree ply of an `_aligned` specimen puts its real power.  These four
# combinations separate the two effects.
FACTORIAL = {
    "window_no_excise": ("fft_slab", False),
    "window_excise": ("fft_slab", True),
    "nowindow_no_excise": ("fft_slab_nowin", False),
    "nowindow_excise": ("fft_slab_nowin", True),      # the old T-H behaviour
}
SPEC_CROP = 64            # stored 2-D power-spectrum thumbnail per ply slab
RADON_N = 256
PORE_MIN_VOXELS = 20


def _hist180(angles_deg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    b = np.floor(angles_deg % 180.0).astype(np.int64) % N_ANG
    return np.bincount(b, weights=weights, minlength=N_ANG).astype(np.float64)


def st_density(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gradient structure tensor orientation histogram of one slab image."""
    a = ndimage.gaussian_filter(img.astype(np.float32), sigma)
    gy = ndimage.sobel(a, axis=0)
    gx = ndimage.sobel(a, axis=1)
    ang = np.rad2deg(np.arctan2(gy, gx)) + 90.0        # texture is perp to grad
    return _hist180(ang.ravel(), (gx * gx + gy * gy).ravel())


def radon_density(img: np.ndarray) -> np.ndarray:
    """Variance of the Radon projection against angle, on a 256-pixel copy."""
    from skimage.transform import radon
    f = max(1, img.shape[0] // RADON_N)
    small = img[::f, ::f].astype(np.float32)
    small = small - small.mean()
    n = min(small.shape)
    small = small[:n, :n] * np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    proj = radon(small, theta=np.arange(180.0), circle=True, preserve_range=True)
    v = proj.var(axis=0)
    # skimage projection angle p corresponds to texture angle (90 - p) mod 180
    return v[(90 - np.arange(180)) % 180]


def pore_axes_density(mask_slab: np.ndarray) -> np.ndarray:
    """In-plane principal axis of every connected pore, weighted by size.

    Purely morphological: no Fourier transform, so the square-grid 0/90 spike
    that T-H had to excise cannot arise here.  Round pores contribute nothing
    because the weight is the elongation.
    """
    lab, n = ndimage.label(mask_slab, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return np.zeros(N_ANG)
    zz, yy, xx = np.nonzero(lab)
    lv = lab[zz, yy, xx]
    cnt = np.bincount(lv, minlength=n + 1).astype(np.float64)
    sy = np.bincount(lv, weights=yy, minlength=n + 1)
    sx = np.bincount(lv, weights=xx, minlength=n + 1)
    syy = np.bincount(lv, weights=yy.astype(np.float64) ** 2, minlength=n + 1)
    sxx = np.bincount(lv, weights=xx.astype(np.float64) ** 2, minlength=n + 1)
    sxy = np.bincount(lv, weights=yy.astype(np.float64) * xx, minlength=n + 1)
    keep = cnt >= PORE_MIN_VOXELS
    keep[0] = False
    if not keep.any():
        return np.zeros(N_ANG)
    c = cnt[keep]
    my, mx = sy[keep] / c, sx[keep] / c
    cyy = syy[keep] / c - my * my
    cxx = sxx[keep] / c - mx * mx
    cxy = sxy[keep] / c - my * mx
    ang = np.rad2deg(0.5 * np.arctan2(2.0 * cxy, cxx - cyy))   # major axis from +x
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
    l1, l2 = 0.5 * tr + disc, 0.5 * tr - disc
    elong = (l1 - l2) / np.maximum(l1 + l2, 1e-12)
    return _hist180(ang, c * elong)


def volume_slabs(job: tuple) -> dict:
    """Per-ply-slab angular densities for one volume, all estimators."""
    volume_id, edges, band = job
    t0 = time.time()
    g = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]
    xct_a, mask_a = g["xct"], g["mask"]
    _, H, W = xct_a.shape
    n = WINDOW
    x0 = max(0, W // 2 - n // 2)
    y0s = [int(np.clip(round(f * H) - n // 2, 0, H - n)) for f in WINDOW_FRACS]
    win, sel, flat, counts = _geometry(n)
    nb = len(edges) - 1
    out = {e: np.zeros((nb, N_ANG)) for e in SLAB_ESTIMATORS if e != "fft_slice"}
    slab_pore_frac = np.zeros(nb)
    flat_win = np.ones_like(win)
    spec = np.zeros((nb, SPEC_CROP, SPEC_CROP), dtype=np.float32)

    # The two in-plane windows are ALSO kept separately.  They are two
    # independent measurements of the same ply, so their disagreement is a
    # direct estimate of measurement noise, and the part of the deviation from
    # nominal that both windows share is real local structure.
    per_win = {"fft_slice": np.zeros((len(y0s), nb, N_ANG)),
               "pore_axes": np.zeros((len(y0s), nb, N_ANG))}

    for i in range(nb):
        z0, z1 = int(edges[i]), int(edges[i + 1])
        for wi, y0 in enumerate(y0s):
            blk = np.asarray(xct_a[z0:z1, y0:y0 + n, x0:x0 + n]).astype(np.float32)
            # incoherent per-slice average inside the ply (the fft_slice channel)
            acc = np.zeros(N_ANG)
            for s in range(blk.shape[0]):
                acc += band_hist(angular_map(blk[s], win, sel, flat, counts)[None], *band)[0]
            per_win["fft_slice"][wi, i] = acc / max(blk.shape[0], 1)
            img = blk.mean(axis=0)                      # <- coherent slab average
            m = angular_map(img, win, sel, flat, counts)
            out["fft_slab"][i] += band_hist(m[None], *band)[0]
            mn = angular_map(img, flat_win, sel, flat, counts)
            out["fft_slab_nowin"][i] += band_hist(mn[None], *band)[0]
            # 2-D power thumbnail, Hann windowed, for visual confirmation
            p = np.fft.fftshift(np.abs(np.fft.fft2((img - img.mean()) * win)) ** 2)
            c = n // 2
            h = n // 8
            crop = p[c - h:c + h, c - h:c + h]
            f = crop.shape[0] // SPEC_CROP
            spec[i] += np.log1p(crop.reshape(SPEC_CROP, f, SPEC_CROP, f).mean(axis=(1, 3))
                                ).astype(np.float32) / len(y0s)
            out["st_slab"][i] += st_density(img)
            out["radon_slab"][i] += radon_density(img)
            msl = np.asarray(mask_a[z0:z1, y0:y0 + n, x0:x0 + n]).astype(bool)
            slab_pore_frac[i] += float(msl.mean()) / len(y0s)
            pa = pore_axes_density(msl)
            per_win["pore_axes"][wi, i] = pa
            out["pore_axes"][i] += pa
    for k in out:
        out[k] /= len(y0s)
    out["volume_id"] = volume_id
    out["per_window_fft_slice"] = per_win["fft_slice"]
    out["per_window_pore_axes"] = per_win["pore_axes"]
    out["edges"] = np.asarray(edges)
    out["slab_pore_frac"] = slab_pore_frac
    out["spectra"] = spec
    out["seconds"] = time.time() - t0
    return out


def calibrate() -> dict:
    """Synthetic check that every estimator returns the same texture angle."""
    n = 512
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    rep = {}
    for true_ang in (0.0, 22.5, 45.0, 70.0, 90.0, 135.0, 160.0):
        a = np.deg2rad(true_ang + 90.0)
        img = np.sin(2 * np.pi / 12 * (np.cos(a) * xx + np.sin(a) * yy)).astype(np.float32)
        win, sel, flat, counts = _geometry(n)
        m = angular_map(img, win, sel, flat, counts)
        fft_ang = float(THETA[np.argmax(band_hist(m[None], 8.0, 32.0)[0])])
        st_ang = float(THETA[np.argmax(st_density(img))])
        rd_ang = float(THETA[np.argmax(radon_density(img))])
        blob = np.zeros((8, n, n), bool)
        rr = ((yy - n / 2) * np.cos(np.deg2rad(true_ang))
              - (xx - n / 2) * np.sin(np.deg2rad(true_ang)))
        ss = ((yy - n / 2) * np.sin(np.deg2rad(true_ang))
              + (xx - n / 2) * np.cos(np.deg2rad(true_ang)))
        blob[:, (np.abs(rr) < 3) & (np.abs(ss) < 60)] = True
        po_ang = float(THETA[np.argmax(pore_axes_density(blob))])
        rep[str(true_ang)] = {
            "fft_slab": fft_ang, "st_slab": st_ang,
            "radon_slab": rd_ang, "pore_axes": po_ang,
            "max_abs_error_deg": float(max(
                abs(np.rad2deg(np.angle(np.exp(2j * np.deg2rad(x - true_ang)))) / 2)
                for x in (fft_ang, st_ang, rd_ang, po_ang))),
        }
    rep["max_error_over_all_angles_deg"] = float(
        max(v["max_abs_error_deg"] for v in rep.values() if isinstance(v, dict)))
    return rep


# ---------------------------------------------------------------------------
# Stage 3 — four-class ply classification
#
# The layups use only 0, +45, -45 and 90 degrees.  Mod 180 those are four
# classes on a 45-degree lattice.  A lattice is invariant under a 45-degree
# rotation, so the *phase of the lattice* -- and nothing else -- can be read
# out of the data with no labels at all, from the m = 8 angular mode:
#
#     z8 = sum_k w_k exp(8i theta_k),   phi45 = arg(z8) / 8   (mod 45)
#
# |z8| / sum w is a label-free measure of how tightly the measured per-ply
# angles fall on ONE 45-degree lattice.  For a genuine [0/+-45/90] laminate it
# must be large; for noise it is ~0.  Only 16 discrete unknowns are then left
# (which lattice site is 0 degrees: 4; angle sign: 2; which face is z=0: 2), and
# those are resolved against the known sequence -- with a shuffled-sequence null
# that pays for exactly that freedom.
# ---------------------------------------------------------------------------

CLASS_ANGLES = np.array([0.0, 45.0, 90.0, 135.0])


def lattice_phase(angles: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Phase (mod 45 deg) and concentration of a 45-degree angular lattice."""
    z = np.sum(w * np.exp(8j * np.deg2rad(angles)))
    return float(np.rad2deg(np.angle(z)) / 8.0 % 45.0), float(np.abs(z) / (np.sum(w) + 1e-30))


def classify_sequence(angles: np.ndarray, w: np.ndarray, truth: np.ndarray) -> dict:
    """Snap to the data-derived 45-degree lattice, then match the known sequence."""
    phi, conc = lattice_phase(angles, w)
    site = np.round(((angles - phi) % 180.0) / 45.0).astype(int) % 4
    best = None
    for rot in range(4):
        for sgn in (1, -1):
            for rev in (False, True):
                t = truth[::-1] if rev else truth
                tcls = (np.round((sgn * t % 180.0) / 45.0).astype(int)) % 4
                acc = float(np.mean(((site + rot) % 4) == tcls))
                if best is None or acc > best["accuracy"]:
                    best = {"accuracy": acc, "rot": rot, "sign": sgn, "reversed": rev,
                            "pred_class": ((site + rot) % 4).tolist(),
                            "true_class": tcls.tolist()}
    best["lattice_phase_deg"] = phi
    best["lattice_concentration"] = conc
    return best


def matched_filter_angles(R: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle of each ply block by circular cross-correlation with a template."""
    C = np.real(np.fft.ifft(np.fft.fft(R, axis=1) * np.conj(np.fft.fft(template))[None, :], axis=1))
    return THETA[np.argmax(C, axis=1)], C.max(axis=1) - C.min(axis=1)


def build_template(blocks: list[np.ndarray], angles: list[float]) -> np.ndarray:
    """Mean residual density of a ply, rotated so its true angle sits at 0."""
    acc = np.zeros(N_ANG)
    for R, a in zip(blocks, angles):
        acc += np.roll(R, -int(round(a)) % N_ANG)
    return acc / max(len(blocks), 1)


def prepare_density(D: np.ndarray, excise: bool) -> np.ndarray:
    """Unit-mean normalise, optionally excise the 0/90 grid bins, then residual."""
    D = D / np.maximum(np.mean(D, axis=1, keepdims=True), 1e-30)
    if excise:
        D = D.copy()
        xi = np.arange(N_ANG)
        bad = np.zeros(N_ANG, bool)
        for c in (0, 90):
            for d in range(-AXIS_EXCISE, AXIS_EXCISE + 1):
                bad[(c + d) % N_ANG] = True
        for i in range(D.shape[0]):
            D[i, bad] = np.interp(xi[bad], xi[~bad], D[i, ~bad], period=N_ANG)
    return _smooth(D - D.mean(axis=0, keepdims=True))


def analyse_slabs(slab_recs: dict, per_volume: dict, gt: dict, recs: dict,
                  n_perm: int = 200, seed: int = 3) -> dict:
    """Stage 2/3: per-estimator angles, four-class accuracy, held-out check."""
    rng = np.random.default_rng(seed)
    truth_all = gt["volumes"]
    store: dict = {}

    for vid, sr in slab_recs.items():
        plies = truth_all.get(vid, {}).get("plies")
        if not plies:
            continue
        g = np.asarray(plies, float) % 180
        rec = {"family": family(vid), "n_blocks": int(sr["edges"].size - 1)}
        # the stage-1 control: per-slice spectra averaged INCOHERENTLY over the
        # same ply block, i.e. what T-G and T-H did, on the same block grid
        p = per_volume[vid]["measured_pitch_voxels"]
        sr = dict(sr)
        sr["fft_slice"] = block_means(
            band_hist(recs[vid]["raw"], BAND_PITCH[0] * p, BAND_PITCH[1] * p),
            np.asarray(sr["edges"]))
        res: dict[str, np.ndarray] = {}
        for est in SLAB_ESTIMATORS:
            D = sr.get(est)
            if D is None or D.shape[0] != len(g) or not np.isfinite(D).all() or D.sum() == 0:
                continue
            res[est] = prepare_density(D, excise=(est != "pore_axes"))
        # the two independent physical channels, combined: grey-level tow texture
        # and pore elongation.  Each residual is scaled to unit spread first.
        if "fft_slice" in res and "pore_axes" in res:
            res["combined"] = (res["fft_slice"] / (res["fft_slice"].std() + 1e-12)
                               + res["pore_axes"] / (res["pore_axes"].std() + 1e-12))

        for est, R in res.items():
            ang = THETA[np.argmax(R, axis=1)]
            w = R.max(axis=1) - R.min(axis=1)
            f = fit_offset(ang, g, w)
            cls = classify_sequence(ang, w, g)
            nulls_c = [classify_sequence(ang, w, rng.permutation(g))["accuracy"]
                       for _ in range(n_perm)]
            nulls_e = [fit_offset(ang, rng.permutation(g), w, n_boot=0)["median_abs_error"]
                       for _ in range(n_perm)]
            rec[est] = {
                "angles_deg": ang.tolist(),
                "weights": w.tolist(),
                "residual": R,
                "median_abs_error": f["median_abs_error"],
                "errors_deg": f["errors"].tolist(),
                "offset_deg": f["offset_deg"],
                "offset_bootstrap_sd_deg": f["offset_bootstrap_sd_deg"],
                "offset_ci95_deg": f["offset_ci95_deg"],
                "resultant_length": f["resultant_length"],
                "hypotheses": f["hypotheses"],
                "hypothesis_margin_deg": f["hypothesis_margin_deg"],
                "sign": f["sign"], "reversed": bool(f["reversed"]),
                "truth_deg": f["truth_used"].tolist(),
                "class_accuracy": cls["accuracy"],
                "pred_class": cls["pred_class"], "true_class": cls["true_class"],
                "lattice_phase_deg": cls["lattice_phase_deg"],
                "lattice_concentration": cls["lattice_concentration"],
                "null_class_accuracy_median": float(np.median(nulls_c)),
                "null_class_accuracy_p95": float(np.percentile(nulls_c, 95)),
                "class_p_value": float((np.sum(np.asarray(nulls_c) >= cls["accuracy"]) + 1)
                                       / (n_perm + 1)),
                "null_median_abs_error": float(np.median(nulls_e)),
                "error_p_value": float((np.sum(np.asarray(nulls_e) <= f["median_abs_error"]) + 1)
                                       / (n_perm + 1)),
            }
        # --- window x excision factorial, on the same slab images ----------
        rec["factorial"] = {}
        for name, (src, exc) in FACTORIAL.items():
            D = sr.get(src)
            if D is None or D.shape[0] != len(g):
                continue
            R = prepare_density(D, excise=exc)
            ang = THETA[np.argmax(R, axis=1)]
            w = R.max(axis=1) - R.min(axis=1)
            cls = classify_sequence(ang, w, g)
            f = fit_offset(ang, g, w, n_boot=0)
            rec["factorial"][name] = {
                "class_accuracy": cls["accuracy"],
                "median_abs_error": f["median_abs_error"],
                "lattice_concentration": cls["lattice_concentration"],
            }
        store[vid] = rec

    # --- leave-one-VOLUME-out matched filter: is the information there? -----
    lovo: dict = {}
    for est in ALL_ESTIMATORS:
        have = [v for v, r in store.items() if est in r]
        if len(have) < 5:
            continue
        # image-frame truth angles for every volume, from its own continuous fit
        img_ang = {}
        for v in have:
            e = store[v][est]
            img_ang[v] = (e["sign"] * (np.asarray(e["truth_deg"]) - e["offset_deg"])) % 180
        per_vol = []
        for v in have:
            blocks, angs = [], []
            for u in have:
                if u == v:
                    continue
                R = store[u][est]["residual"]
                for i in range(R.shape[0]):
                    blocks.append(R[i])
                    angs.append(float(img_ang[u][i]))
            T = build_template(blocks, angs)
            R = store[v][est]["residual"]
            ang, w = matched_filter_angles(R, T)
            g = np.asarray(store[v][est]["truth_deg"], float)
            cls = classify_sequence(ang, w, g)
            f = fit_offset(ang, g, w, n_boot=0)
            nulls = [classify_sequence(ang, w, rng.permutation(g))["accuracy"]
                     for _ in range(60)]
            per_vol.append({"volume_id": v, "family": family(v),
                            "accuracy": cls["accuracy"],
                            "median_abs_error": f["median_abs_error"],
                            "null_accuracy": float(np.median(nulls)),
                            "pred_class": cls["pred_class"],
                            "true_class": cls["true_class"]})
            store[v][est]["lovo"] = per_vol[-1]
        lovo[est] = per_vol

    for r in store.values():
        for est in ALL_ESTIMATORS:
            if est in r:
                r[est].pop("residual", None)
    return {"per_volume": store, "lovo": lovo}


def summarise_slabs(sl: dict) -> dict:
    store, lovo = sl["per_volume"], sl["lovo"]
    out: dict = {"per_estimator": {}, "lovo": {}, "confusion": {}}
    for est in ALL_ESTIMATORS:
        blk: dict = {}
        for fam in list(FAMILIES) + ["ALL"]:
            rs = [r[est] for r in store.values()
                  if est in r and (fam == "ALL" or r["family"] == fam)]
            if not rs:
                continue
            acc = np.array([r["class_accuracy"] for r in rs])
            nul = np.array([r["null_class_accuracy_median"] for r in rs])
            err = np.concatenate([np.abs(r["errors_deg"]) for r in rs])
            con = np.array([r["lattice_concentration"] for r in rs])
            blk[fam] = {
                "n_volumes": len(rs),
                "n_ply_blocks": int(err.size),
                "class_accuracy_mean": float(np.mean(acc)),
                "class_accuracy_median": float(np.median(acc)),
                "null_class_accuracy_mean": float(np.mean(nul)),
                "frac_volumes_p_below_0_05": float(np.mean(
                    [r["class_p_value"] < 0.05 for r in rs])),
                "median_abs_error_pooled": float(np.median(err)),
                "null_median_abs_error_mean": float(np.mean(
                    [r["null_median_abs_error"] for r in rs])),
                "frac_within_22_5": float(np.mean(err < 22.5)),
                "lattice_concentration_median": float(np.median(con)),
            }
        out["per_estimator"][est] = blk

        if est in lovo:
            lb: dict = {}
            for fam in list(FAMILIES) + ["ALL"]:
                rs = [r for r in lovo[est] if fam == "ALL" or r["family"] == fam]
                if not rs:
                    continue
                a = np.array([r["accuracy"] for r in rs])
                lb[fam] = {
                    "n_volumes": len(rs),
                    "heldout_accuracy_mean": float(np.mean(a)),
                    "heldout_accuracy_median": float(np.median(a)),
                    "null_accuracy_mean": float(np.mean([r["null_accuracy"] for r in rs])),
                    "median_abs_error_median": float(np.median(
                        [r["median_abs_error"] for r in rs])),
                    "chance": 0.25,
                }
            out["lovo"][est] = lb

        # confusion over the four classes, pooled and per family
        conf: dict = {}
        for fam in list(FAMILIES) + ["ALL"]:
            M = np.zeros((4, 4), int)
            for r in store.values():
                if est not in r or (fam != "ALL" and r["family"] != fam):
                    continue
                for p, t in zip(r[est]["pred_class"], r[est]["true_class"]):
                    M[t, p] += 1
            if M.sum() == 0:
                continue
            conf[fam] = {
                "matrix_true_by_pred": M.tolist(),
                "per_class_recall": (np.diag(M) / np.maximum(M.sum(axis=1), 1)).tolist(),
                "overall": float(np.trace(M) / M.sum()),
            }
        out["confusion"][est] = conf

    fac: dict = {}
    for name in FACTORIAL:
        for fam in list(FAMILIES) + ["ALL"]:
            rs = [r["factorial"][name] for r in store.values()
                  if name in r.get("factorial", {})
                  and (fam == "ALL" or r["family"] == fam)]
            if not rs:
                continue
            fac.setdefault(name, {})[fam] = {
                "n_volumes": len(rs),
                "class_accuracy_mean": float(np.mean([r["class_accuracy"] for r in rs])),
                "median_abs_error_median": float(np.median([r["median_abs_error"] for r in rs])),
                "lattice_concentration_median": float(np.median(
                    [r["lattice_concentration"] for r in rs])),
            }
    out["window_excision_factorial"] = fac
    return out


def _wrap90(d: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-90, 90]."""
    return np.rad2deg(np.angle(np.exp(2j * np.deg2rad(d)))) / 2


def _combined_residual(fft_d: np.ndarray, pore_d: np.ndarray) -> np.ndarray:
    a = prepare_density(fft_d, excise=True)
    b = prepare_density(pore_d, excise=False)
    return a / (a.std() + 1e-12) + b / (b.std() + 1e-12)


def window_agreement(slab_recs: dict, per_volume: dict, gt: dict) -> dict:
    """Split the deviation from nominal into real structure and measurement noise.

    The two in-plane windows (y = 30 % and 70 % of the specimen) see the same
    plies through different material.  Writing each window's estimate as
    A = t + n_A and B = t + n_B with independent noise of variance s^2:

        var(A - B) = 2 s^2                      -> the noise
        var(mean - nominal) = var(t - nominal) + s^2 / 2

    so the REAL local deviation from the nominal design angle is

        var_real = var(mean - nominal) - var(A - B) / 4

    and corr(A - nominal, B - nominal) across plies is the shared fraction
    directly, with no variance algebra at all.  Both are reported.
    """
    out: dict = {"per_volume": {}, "per_family": {}}
    for vid, sr in slab_recs.items():
        r = per_volume.get(vid)
        if not r or not r.get("n_plies_gt") or "per_window_fft_slice" not in sr:
            continue
        g = np.asarray(gt["volumes"][vid]["plies"], float) % 180
        F, P = sr["per_window_fft_slice"], sr["per_window_pore_axes"]
        if F.shape[1] != len(g):
            continue
        angs, ws = [], []
        for wi in range(F.shape[0]):
            R = _combined_residual(F[wi], P[wi])
            angs.append(THETA[np.argmax(R, axis=1)])
            ws.append(R.max(axis=1) - R.min(axis=1))
        Rm = _combined_residual(F.mean(axis=0), P.mean(axis=0))
        ang_m = THETA[np.argmax(Rm, axis=1)]
        w_m = Rm.max(axis=1) - Rm.min(axis=1)
        f = fit_offset(ang_m, g, w_m, n_boot=0)
        t = f["truth_used"]
        # apply the SAME volume fit to each window, so a per-window constant
        # (a bulk in-plane rotation difference between the two locations)
        # stays visible instead of being absorbed
        rA = _wrap90(f["sign"] * angs[0] + f["offset_deg"] - t)
        rB = _wrap90(f["sign"] * angs[1] + f["offset_deg"] - t)
        rM = _wrap90(f["sign"] * ang_m + f["offset_deg"] - t)
        d = _wrap90(angs[0] - angs[1])
        # A ply whose CLASS is wrong contributes a 45-90-degree jump, which is
        # a classification failure, not an angular deviation.  The variance
        # decomposition is therefore done on the correctly-classified plies
        # (|deviation| < 22.5 degrees in both windows and in their mean); the
        # class-error rate is reported separately as the accuracy.
        keep = (np.abs(rA) < 22.5) & (np.abs(rB) < 22.5) & (np.abs(rM) < 22.5)
        if keep.sum() < 4:
            continue
        var_noise = float(np.var(d[keep])) / 4.0
        var_tot = float(np.var(rM[keep]))
        rA_k, rB_k, rM_k, d_k = rA[keep], rB[keep], rM[keep], d[keep]
        rec = {
            "n_plies": int(len(g)),
            "n_correctly_classified": int(keep.sum()),
            "window_difference_rms_deg": float(np.sqrt(np.mean(d_k ** 2))),
            "window_difference_median_abs_deg": float(np.median(np.abs(d_k))),
            "noise_sd_per_window_deg": float(np.sqrt(np.mean(d_k ** 2) / 2)),
            "residual_rms_deg": float(np.sqrt(np.mean(rM_k ** 2))),
            "residual_median_abs_deg": float(np.median(np.abs(rM_k))),
            "residual_rms_all_plies_deg": float(np.sqrt(np.mean(rM ** 2))),
            "var_total": var_tot,
            "var_noise_in_mean": var_noise,
            "var_real": float(var_tot - var_noise),
            "real_sd_deg": float(np.sqrt(max(var_tot - var_noise, 0.0))),
            "real_fraction_of_variance": float(np.clip((var_tot - var_noise)
                                                       / max(var_tot, 1e-12), -1, 1)),
            "window_bulk_rotation_diff_deg": float(np.mean(d_k)),
            "corr_rA_rB": (float(np.corrcoef(rA_k, rB_k)[0, 1])
                           if np.std(rA_k) > 0 and np.std(rB_k) > 0 else float("nan")),
        }
        out["per_volume"][vid] = rec

    for fam in list(FAMILIES) + ["ALL"]:
        rs = [v for k, v in out["per_volume"].items()
              if fam == "ALL" or family(k) == fam]
        if not rs:
            continue
        cc = np.array([v["corr_rA_rB"] for v in rs])
        cc = cc[np.isfinite(cc)]
        out["per_family"][fam] = {
            "n_volumes": len(rs),
            "noise_sd_per_window_deg_median": float(np.median(
                [v["noise_sd_per_window_deg"] for v in rs])),
            "residual_rms_deg_median": float(np.median([v["residual_rms_deg"] for v in rs])),
            "real_sd_deg_median": float(np.median([v["real_sd_deg"] for v in rs])),
            "real_fraction_of_variance_median": float(np.median(
                [v["real_fraction_of_variance"] for v in rs])),
            "corr_rA_rB_median": float(np.median(cc)) if cc.size else None,
            "corr_rA_rB_frac_positive": float(np.mean(cc > 0)) if cc.size else None,
        }
    return out


def depth_coherence(recs: dict, per_volume: dict, gt: dict) -> dict:
    """Is the deviation from nominal smooth in z, and worse at block edges?

    Uses the per-SLICE angular densities of stage 1 (both windows averaged).
    Real fibre waviness is spatially coherent along z; estimator noise is not.
    """
    out: dict = {"per_family": {}}
    acc: dict = {}
    for vid, r in per_volume.items():
        if not r.get("n_plies_gt") or "block_edges" not in r.get("raw_scaled", {}):
            continue
        e = np.asarray(r["raw_scaled"]["block_edges"])
        p = r["measured_pitch_voxels"]
        h = band_hist(recs[vid]["raw"], BAND_PITCH[0] * p, BAND_PITCH[1] * p)
        H = prepare_density(h[e[0]:e[-1]], excise=True)
        ang = THETA[np.argmax(H, axis=1)]
        # per-slice deviation from that ply's own block-mean angle
        blk = np.repeat(np.arange(len(e) - 1), np.diff(e))
        dev = np.empty_like(ang)
        pos = np.empty_like(ang)
        for i in range(len(e) - 1):
            m = blk == i
            if m.sum() < 3:
                dev[m] = 0.0
                pos[m] = 0.5
                continue
            bm = np.rad2deg(np.angle(np.mean(np.exp(2j * np.deg2rad(ang[m]))))) / 2
            dev[m] = _wrap90(ang[m] - bm)
            pos[m] = (np.arange(m.sum()) + 0.5) / m.sum()
        good = np.abs(dev) < 45
        if good.sum() < 20:
            continue
        d = dev[good] - dev[good].mean()
        lag1 = float(np.sum(d[:-1] * d[1:]) / max(np.sum(d * d), 1e-12))
        edge = np.minimum(pos, 1 - pos) < 0.2
        acc[vid] = {
            "lag1_autocorr_of_slice_deviation": lag1,
            "slice_deviation_sd_deg": float(np.std(dev[good])),
            "abs_dev_block_edge_deg": float(np.mean(np.abs(dev[good & edge]))),
            "abs_dev_block_centre_deg": float(np.mean(np.abs(dev[good & ~edge]))),
        }
    out["per_volume"] = acc
    for fam in list(FAMILIES) + ["ALL"]:
        rs = [v for k, v in acc.items() if fam == "ALL" or family(k) == fam]
        if not rs:
            continue
        out["per_family"][fam] = {
            "n_volumes": len(rs),
            "lag1_autocorr_median": float(np.median(
                [v["lag1_autocorr_of_slice_deviation"] for v in rs])),
            "frac_lag1_positive": float(np.mean(
                [v["lag1_autocorr_of_slice_deviation"] > 0 for v in rs])),
            "slice_deviation_sd_deg_median": float(np.median(
                [v["slice_deviation_sd_deg"] for v in rs])),
            "abs_dev_edge_over_centre": float(np.median(
                [v["abs_dev_block_edge_deg"] / max(v["abs_dev_block_centre_deg"], 1e-9)
                 for v in rs])),
        }
    return out


def single_window_control(per_volume: dict, gt: dict, n_perm: int = 60,
                          seed: int = 7) -> dict:
    """The decisive diagnosis, run on T-H's OWN cached data.

    T-H stored the per-slice angular density of a SINGLE centred 1024x1024
    window.  Re-reading exactly those numbers with the only T-I change that
    matters -- subtracting the volume-mean angular density before reading each
    ply's direction -- separates two candidate causes of the Pegaso/Nacho split:
    the in-plane window, and the missing mean subtraction.  With mean
    subtraction ON and everything else identical to T-H, a residual family gap
    would indicate the window; no gap indicates the summary statistic.
    """
    src = OUT_ROOT / "T-H" / "angular_hists.npz"
    if not src.exists():
        return {}
    H = np.load(src, allow_pickle=True)["records"].item()
    rng = np.random.default_rng(seed)
    acc: dict = {}
    for mode in ("mean_subtracted", "absolute_like_TG_TH"):
        per_fam: dict = {}
        for vid, h in H.items():
            r = per_volume.get(vid)
            if not r or not r.get("n_plies_gt") or "block_edges" not in r.get("raw_scaled", {}):
                continue
            e = np.asarray(r["raw_scaled"]["block_edges"])
            B = block_means(np.asarray(h["hists"]["tow_16_64"]), e)
            g = np.asarray(gt["volumes"][vid]["plies"], float) % 180
            if mode == "mean_subtracted":
                R = prepare_density(B, excise=True)
            else:
                # T-G / T-H behaviour: read the direction off the ABSOLUTE
                # density, with no volume-mean removal
                R = _smooth(B / np.maximum(B.mean(axis=1, keepdims=True), 1e-30))
            ang = THETA[np.argmax(R, axis=1)]
            w = R.max(axis=1) - R.min(axis=1)
            c = classify_sequence(ang, w, g)
            nl = float(np.median([classify_sequence(ang, w, rng.permutation(g))["accuracy"]
                                  for _ in range(n_perm)]))
            for key in (family(vid), "ALL"):
                per_fam.setdefault(key, {"acc": [], "null": []})
                per_fam[key]["acc"].append(c["accuracy"])
                per_fam[key]["null"].append(nl)
        acc[mode] = {k: {"n_volumes": len(v["acc"]),
                         "class_accuracy_mean": float(np.mean(v["acc"])),
                         "null_class_accuracy_mean": float(np.mean(v["null"]))}
                     for k, v in per_fam.items()}
    p, na = acc["mean_subtracted"], acc["absolute_like_TG_TH"]
    def gap(d):
        if "Airbus_Panel_Pegaso" in d and "Fabricacion_Nacho_05" in d:
            return float(d["Airbus_Panel_Pegaso"]["class_accuracy_mean"]
                         - d["Fabricacion_Nacho_05"]["class_accuracy_mean"])
        return float("nan")
    acc["pegaso_minus_nacho_gap"] = {"mean_subtracted": gap(p),
                                     "absolute_like_TG_TH": gap(na)}
    acc["note"] = ("same window, same slices, same band as T-H; the only change "
                   "is whether the volume-mean angular density is removed first")
    return acc


# ---------------------------------------------------------------------------
# The deliverable: a ground-truth-derived orientation field, in image coords
# ---------------------------------------------------------------------------

def confidence_flag(median_err: float, boot_sd: float, margin: float) -> str:
    if median_err < 15.0 and boot_sd < 10.0 and margin >= 2.0:
        return "high"
    if median_err < 22.5 and boot_sd < 20.0:
        return "medium"
    return "low"


def build_layup_field(per_volume: dict, gt: dict, sl: dict,
                      estimator: str = "combined") -> dict:
    """Per-volume, per-slice ply index and ply angle in IMAGE coordinates.

    The angles come from the expert sequence, NOT from the measurement.  The
    only measured quantity used is the single in-plane rotation offset (and the
    two discrete ambiguities), because the expert angles are relative to the
    part while the model generates in image coordinates.

        image_angle_k = sign * (gt_angle_k - offset)   (mod 180)

    which is the exact inverse of the fit `sign * measured + offset ~= gt`.
    """
    out = {
        "description": "ground-truth-derived ply orientation field in image coordinates",
        "angle_convention": "degrees mod 180; 0 = image x axis, 90 = image y axis",
        "ply_index_convention": "0 at z = z_start, increasing with z",
        "volumes": {},
    }
    for vid, r in per_volume.items():
        gtv = gt["volumes"].get(vid, {})
        se = sl["per_volume"].get(vid, {}).get(estimator)
        if not r.get("n_plies_gt") or se is None:
            out["volumes"][vid] = {
                "usable": False,
                "reason": "no expert stacking sequence for this volume",
                "family": r["family"], "shape": r["shape"],
            }
            continue
        e = se
        edges = np.asarray(r["raw_scaled"]["block_edges"])
        gt_ord = np.asarray(e["truth_deg"], float)      # gt in fitted z order
        img = (e["sign"] * (gt_ord - e["offset_deg"])) % 180.0
        n = r["shape"][0]
        ply_idx = np.full(n, -1, dtype=int)
        for i in range(len(edges) - 1):
            ply_idx[edges[i]:edges[i + 1]] = i
        meas = np.asarray(e["angles_deg"], float) % 180.0
        dev = np.rad2deg(np.angle(np.exp(2j * np.deg2rad(meas - img)))) / 2
        agree = np.abs(dev) < 22.5
        rec_ang = np.where(agree, meas, img) % 180.0
        flag = confidence_flag(e["median_abs_error"], e["offset_bootstrap_sd_deg"],
                               e["hypothesis_margin_deg"])
        pitch = r["measured_pitch_voxels"]
        gt_edges = edges.astype(float)
        pitch_edges = edges[0] + np.arange(len(edges)) * pitch
        out["volumes"][vid] = {
            "usable": True,
            "family": r["family"],
            "shape": r["shape"],
            "sequence_id": r["sequence_id"],
            "material": r["material"],
            "ply_thickness_mm": r["ply_thickness_mm"],
            "voxel_size_um": r["voxel_size_um_from_pitch"],
            "z_start": int(edges[0]),
            "z_end": int(edges[-1] - 1),
            "n_plies": int(len(edges) - 1),
            "ply_boundaries_z": edges.tolist(),
            "ply_boundaries_z_from_measured_pitch": pitch_edges.tolist(),
            "ply_boundary_max_disagreement_voxels": float(
                np.max(np.abs(gt_edges - pitch_edges))),
            "ply_boundary_rms_disagreement_voxels": float(
                np.sqrt(np.mean((gt_edges - pitch_edges) ** 2))),
            "z_order_reversed_vs_expert_list": bool(e["reversed"]),
            "angle_sign": int(e["sign"]),
            "rotation_offset_deg": e["offset_deg"],
            "rotation_offset_sd_deg": e["offset_bootstrap_sd_deg"],
            "rotation_offset_ci95_deg": e["offset_ci95_deg"],
            "hypothesis_margin_deg": e["hypothesis_margin_deg"],
            "estimator": estimator,
            "fit_median_abs_error_deg": e["median_abs_error"],
            "fit_null_median_abs_error_deg": e["null_median_abs_error"],
            "fit_p_value": e["error_p_value"],
            "four_class_accuracy": e["class_accuracy"],
            "four_class_null_accuracy": e["null_class_accuracy_median"],
            "four_class_p_value": e["class_p_value"],
            "measured_ply_class": e["pred_class"],
            "true_ply_class": e["true_class"],
            "confidence": flag,
            "ply_angle_gt_deg": gt_ord.tolist(),
            "ply_angle_image_deg": img.tolist(),
            # what the estimator actually measured for this ply, in image
            # coordinates, and its deviation from the nominal design angle.
            # T-I shows that deviation is majority REAL local structure
            # (fibre waviness / ply misalignment), not estimator noise, so the
            # recommended field keeps it wherever the ply class is right.
            "ply_angle_measured_image_deg": meas.tolist(),
            "ply_deviation_from_nominal_deg": dev.tolist(),
            "ply_class_agrees": agree.tolist(),
            "ply_angle_recommended_deg": rec_ang.tolist(),
            "slice_ply_index": ply_idx.tolist(),
            "slice_angle_image_deg": [
                (float(img[i]) if i >= 0 else None) for i in ply_idx],
            "slice_angle_recommended_deg": [
                (float(rec_ang[i]) if i >= 0 else None) for i in ply_idx],
        }
    us = [v for v in out["volumes"].values() if v["usable"]]
    out["summary"] = {
        "n_usable": len(us),
        "n_unusable": len(out["volumes"]) - len(us),
        "confidence_counts": {
            c: sum(1 for v in us if v["confidence"] == c) for c in ("high", "medium", "low")},
        "ply_boundary_rms_disagreement_median_voxels": float(np.median(
            [v["ply_boundary_rms_disagreement_voxels"] for v in us])) if us else None,
    }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(per_volume: dict, summary: dict) -> dict:
    set_style()
    figs = {}
    colors = {"Airbus_Panel_Pegaso": "#1b6ca8", "Fabricacion_Nacho_05": "#c2571a",
              "Juan_Ignacio": "#2e7d32"}

    # fig 1 — measured vs ground-truth ply angles, one volume per family
    reps = []
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and "peak_residual" in r.get("raw_scaled", {})]
        if not vs:
            continue
        vs.sort(key=lambda r: r["raw_scaled"]["peak_residual"]["median_abs_error"])
        reps.append(vs[len(vs) // 2])
    fig, axes = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 3.4), squeeze=False)
    for ax, r in zip(axes[0], reps):
        e = r["raw_scaled"]["peak_residual"]
        k = np.arange(len(e["truth_deg"])) + 1
        ax.plot(k, e["truth_deg"], "o-", color="0.25", label="ground truth", ms=5)
        ax.plot(k, e["fitted_deg"], "s--", color=colors[r["family"]],
                label="measured + offset", ms=5)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel("ply index (fitted order)")
        ax.set_ylabel("in-plane angle (deg, mod 180)")
        ax.set_title(f"{r['family']}\nmedian |err| = {e['median_abs_error']:.1f} deg", fontsize=9)
        ax.legend(loc="upper right")
    fig.suptitle("T-I fig 1 — measured ply orientation against the expert layup", y=1.02)
    figs["TI_fig1_measured_vs_truth"] = savefig(fig, OUT_DIR, "TI_fig1_measured_vs_truth")

    # fig 2 — error distribution per family, with the permutation null
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 90, 31)
    for fam in FAMILIES:
        E = np.concatenate([np.abs(r["raw_scaled"]["peak_residual"]["errors_deg"])
                            for r in per_volume.values()
                            if r["family"] == fam and "peak_residual" in r.get("raw_scaled", {})]
                           or [np.array([])])
        if E.size:
            axes[0].hist(E, bins=bins, density=True, histtype="step", lw=1.6,
                         color=colors[fam], label=f"{fam} (n={E.size})")
    axes[0].axhline(1 / 90, color="0.5", ls=":", label="uniform (chance)")
    axes[0].set_xlabel("|angular error| after offset fit (deg)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    axes[0].set_title("per-ply error, all volumes")

    labels, meas, null = [], [], []
    for fam in FAMILIES:
        b = summary["extraction"]["raw_scaled.peak_residual"].get(fam)
        if not b:
            continue
        labels.append(fam.split("_")[0] if fam != "Juan_Ignacio" else "Juan_Ig.")
        meas.append(b["median_of_volume_medians"])
        null.append(b["null_median_of_volume_medians"])
    xx = np.arange(len(labels))
    axes[1].bar(xx - 0.18, meas, 0.36, label="measured", color="#1b6ca8")
    axes[1].bar(xx + 0.18, null, 0.36, label="shuffled-ply null", color="0.65")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("median |error| (deg)")
    axes[1].set_title("measured against the permutation null")
    axes[1].legend(fontsize=8)
    fig.suptitle("T-I fig 2 — error distribution and null baseline", y=1.03)
    figs["TI_fig2_error_distribution"] = savefig(fig, OUT_DIR, "TI_fig2_error_distribution")

    # fig 3 — per-ply thickness and measured pitch per family
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (fld, ttl) in enumerate((("thickness_per_ply_voxels", "z extent / ground-truth ply count"),
                                    ("measured_pitch_voxels", "measured porosity pitch"))):
        data, labs, cs = [], [], []
        for fam in FAMILIES:
            v = [r[fld] for r in per_volume.values()
                 if r["family"] == fam and r[fld] and r[fld] == r[fld]]
            if v:
                data.append(v)
                labs.append(fam.split("_")[0] if fam != "Juan_Ignacio" else "Juan_Ig.")
                cs.append(colors[fam])
        bp = axes[i].boxplot(data, tick_labels=labs, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], cs):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        axes[i].set_ylabel("voxels per ply")
        axes[i].set_title(ttl)
        axes[i].axhline(REF_PITCH, color="0.4", ls=":", lw=1)
    fig.suptitle("T-I fig 3 — ply thickness in voxels: the campaigns are not on one scale", y=1.03)
    figs["TI_fig3_ply_thickness"] = savefig(fig, OUT_DIR, "TI_fig3_ply_thickness")

    # fig 4 — the diagnosis: constant vs residual angular density
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and "constant_to_residual" in r.get("raw_fixed", {})]
        if not vs:
            continue
        ctr = [r["raw_fixed"]["constant_to_residual"] for r in vs]
        cang = [r["raw_fixed"]["constant_angle_deg"] for r in vs]
        axes[0].scatter(cang, ctr, s=18, color=colors[fam], alpha=0.8,
                        label=fam.split("_")[0] if fam != "Juan_Ignacio" else "Juan_Ig.")
        por = [r["mean_pore_frac"] for r in vs]
        er = [r["raw_scaled"]["peak_residual"]["median_abs_error"] for r in vs]
        en = [r["nopore_scaled"]["peak_residual"]["median_abs_error"] for r in vs]
        axes[1].scatter(por, er, s=18, color=colors[fam], alpha=0.8)
        axes[2].scatter(er, en, s=18, color=colors[fam], alpha=0.8)
    for x in (0, 90, 180):
        axes[0].axvline(x, color="0.6", ls=":", lw=1)
    axes[0].set_xlabel("angle of the constant (volume-mean) 2-theta term (deg)")
    axes[0].set_ylabel("|constant| / mean |residual|")
    axes[0].set_title("the constant term sits on the image axes")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("mean pore fraction")
    axes[1].set_ylabel("median |error| (deg)")
    axes[1].set_title("error against porosity")
    lim = axes[2].get_xlim()
    axes[2].plot(lim, lim, color="0.5", ls=":")
    axes[2].set_xlabel("median |error|, raw (deg)")
    axes[2].set_ylabel("median |error|, pores suppressed (deg)")
    axes[2].set_title("pore suppression changes nothing")
    fig.suptitle("T-I fig 4 — why T-G and T-H split Pegaso from Nacho", y=1.03)
    figs["TI_fig4_diagnosis"] = savefig(fig, OUT_DIR, "TI_fig4_diagnosis")

    # fig 5 — residual angular density against depth for one volume per family
    fig, axes = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 3.4), squeeze=False)
    for ax, r in zip(axes[0], reps):
        e = r["raw_scaled"]["peak_residual"]
        ax.plot(np.asarray(e["truth_deg"]), np.asarray(e["fitted_deg"]), "o",
                color=colors[r["family"]], ms=6)
        ax.plot([0, 180], [0, 180], color="0.5", ls=":")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel("ground-truth angle (deg)")
        ax.set_ylabel("measured angle (deg)")
        ax.set_title(r["family"], fontsize=9)
    fig.suptitle("T-I fig 5 — measured against true angle, per ply", y=1.02)
    figs["TI_fig5_scatter"] = savefig(fig, OUT_DIR, "TI_fig5_scatter")

    # fig 6 — derived voxel size and the fitted rotation offset
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for fam in FAMILIES:
        vs = [r for r in per_volume.values()
              if r["family"] == fam and r["voxel_size_um_from_pitch"]]
        if not vs:
            continue
        lab = fam.split("_")[0] if fam != "Juan_Ignacio" else "Juan_Ig."
        axes[0].scatter([r["measured_pitch_voxels"] for r in vs],
                        [r["voxel_size_um_from_pitch"] for r in vs],
                        s=20, color=colors[fam], alpha=0.85, label=lab)
        vs2 = [r for r in vs if "peak_residual" in r.get("raw_scaled", {})]
        if vs2:
            axes[1].errorbar([r["raw_scaled"]["peak_residual"]["offset_deg"] % 180 for r in vs2],
                             np.arange(len(vs2)) + (0 if fam == FAMILIES[0] else 30),
                             xerr=[r["raw_scaled"]["peak_residual"]["offset_bootstrap_sd_deg"]
                                   for r in vs2],
                             fmt="o", ms=3.5, lw=0.9, color=colors[fam], alpha=0.85, label=lab)
            axes[2].scatter([r["raw_scaled"]["peak_residual"]["offset_bootstrap_sd_deg"] for r in vs2],
                            [r["raw_scaled"]["peak_residual"]["median_abs_error"] for r in vs2],
                            s=20, color=colors[fam], alpha=0.85)
    axes[0].set_xlabel("measured ply pitch (voxels)")
    axes[0].set_ylabel("derived voxel size (um)")
    axes[0].set_title("voxel size = ply thickness / measured pitch")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("fitted rotation offset (deg, mod 180)")
    axes[1].set_ylabel("volume (arbitrary order)")
    axes[1].set_title("per-volume offset, +-1 bootstrap sd")
    axes[1].legend(fontsize=7)
    axes[2].axvline(10, color="0.5", ls=":")
    axes[2].set_xlabel("offset bootstrap sd (deg)")
    axes[2].set_ylabel("per-ply median |error| (deg)")
    axes[2].set_title("offset precision against per-ply residual")
    fig.suptitle("T-I fig 6 — physical scale and the per-volume rotation offset", y=1.03)
    figs["TI_fig6_scale_and_offset"] = savefig(fig, OUT_DIR, "TI_fig6_scale_and_offset")

    return figs


def make_spectrum_figure(slab_recs: dict, sl: dict, gt: dict) -> dict:
    """Windowed 2-D power spectra of a 0, +45, -45 and 90 ply, same volume."""
    set_style()
    figs = {}
    store = sl["per_volume"]
    picks = []
    for fam in FAMILIES:
        cands = [(v, r) for v, r in store.items()
                 if r["family"] == fam and "fft_slab" in r and v in slab_recs]
        if not cands:
            continue
        cands.sort(key=lambda t: -t[1]["fft_slab"]["class_accuracy"])
        picks.append(cands[0])
    if not picks:
        return figs
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.9 * len(picks)), squeeze=False)
    names = {0.0: "0 deg", 45.0: "+45 deg", 90.0: "90 deg", 135.0: "-45 deg"}
    for row, (vid, r) in enumerate(picks):
        spec = slab_recs[vid]["spectra"]
        g = np.asarray(gt["volumes"][vid]["plies"], float) % 180
        rev = r["fft_slab"]["reversed"]
        gg = g[::-1] if rev else g
        for col, want in enumerate((0.0, 45.0, 90.0, 135.0)):
            k = int(np.argmin(np.abs(gg - want)))
            ax = axes[row][col]
            s = spec[k]
            ax.imshow(s, cmap="magma", origin="lower",
                      vmin=np.percentile(s, 2), vmax=np.percentile(s, 99.8))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ttl = names.get(float(gg[k]), f"{gg[k]:.0f} deg")
            ax.set_title(f"ply {k} — true {ttl}", fontsize=8)
            if col == 0:
                ax.set_ylabel(r["family"].split("_")[0] if r["family"] != "Juan_Ignacio"
                              else "Juan_Ig.", fontsize=8)
    fig.suptitle("T-I fig 10 — Hann-windowed 2-D power spectrum of one ply slab, "
                 "by true ply angle (log power, wavelengths >= 8 voxels)", y=1.01)
    figs["TI_fig10_ply_spectra"] = savefig(fig, OUT_DIR, "TI_fig10_ply_spectra")
    return figs


def make_slab_figures(sl: dict, ss: dict) -> dict:
    set_style()
    figs = {}
    colors = {"Airbus_Panel_Pegaso": "#1b6ca8", "Fabricacion_Nacho_05": "#c2571a",
              "Juan_Ignacio": "#2e7d32"}
    ests = [e for e in ALL_ESTIMATORS if e in ss["per_estimator"]
            and "ALL" in ss["per_estimator"][e]]

    # fig 7 — four-class accuracy per estimator, against the null and chance
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xx = np.arange(len(ests))
    acc = [ss["per_estimator"][e]["ALL"]["class_accuracy_mean"] for e in ests]
    nul = [ss["per_estimator"][e]["ALL"]["null_class_accuracy_mean"] for e in ests]
    axes[0].bar(xx - 0.19, acc, 0.38, color="#1b6ca8", label="measured")
    axes[0].bar(xx + 0.19, nul, 0.38, color="0.65", label="shuffled-sequence null")
    axes[0].axhline(0.25, color="#b00020", ls="--", lw=1, label="chance (4 classes)")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(ests, rotation=20, ha="right")
    axes[0].set_ylabel("four-class ply accuracy")
    axes[0].set_title("all volumes, offset fitted per volume")
    axes[0].legend(fontsize=7)

    for e in ests:
        if e not in ss["lovo"]:
            continue
        for fam in FAMILIES:
            b = ss["lovo"][e].get(fam)
            if b:
                axes[1].scatter([ests.index(e)], [b["heldout_accuracy_mean"]],
                                s=45, color=colors[fam],
                                label=fam if e == ests[0] else None)
    axes[1].axhline(0.25, color="#b00020", ls="--", lw=1)
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(ests, rotation=20, ha="right")
    axes[1].set_ylabel("held-out-volume accuracy")
    axes[1].set_title("leave-one-volume-out matched filter")
    axes[1].legend(fontsize=7)
    fig.suptitle("T-I fig 7 — does any estimator recover the four ply classes?", y=1.04)
    figs["TI_fig7_class_accuracy"] = savefig(fig, OUT_DIR, "TI_fig7_class_accuracy")

    # fig 8 — confusion matrices of the best estimator
    best = max(ests, key=lambda e: ss["per_estimator"][e]["ALL"]["class_accuracy_mean"])
    fams = [f for f in FAMILIES if f in ss["confusion"][best]]
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(3.6 * (len(fams) + 1), 3.4))
    names = ["0", "+45", "90", "-45"]
    for ax, fam in zip(axes, fams + ["ALL"]):
        M = np.array(ss["confusion"][best][fam]["matrix_true_by_pred"], float)
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(Mn, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if Mn[i, j] > 0.5 else "0.2")
        ax.set_xticks(range(4)); ax.set_xticklabels(names)
        ax.set_yticks(range(4)); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{fam}\noverall {ss['confusion'][best][fam]['overall']:.2f}", fontsize=9)
        ax.grid(False)
    fig.colorbar(im, ax=axes[-1], fraction=0.046)
    fig.suptitle(f"T-I fig 8 — four-class confusion, estimator '{best}'", y=1.04)
    figs["TI_fig8_confusion"] = savefig(fig, OUT_DIR, "TI_fig8_confusion")

    # fig 9 — the 45-degree lattice: label-free evidence of four directions
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for fam in FAMILIES:
        c = [r[best]["lattice_concentration"] for r in sl["per_volume"].values()
             if best in r and r["family"] == fam]
        a = [r[best]["class_accuracy"] for r in sl["per_volume"].values()
             if best in r and r["family"] == fam]
        if not c:
            continue
        lab = fam.split("_")[0] if fam != "Juan_Ignacio" else "Juan_Ig."
        axes[0].hist(c, bins=np.linspace(0, 1, 21), histtype="step", lw=1.6,
                     color=colors[fam], label=lab)
        axes[1].scatter(c, a, s=20, color=colors[fam], alpha=0.85, label=lab)
    axes[0].set_xlabel("|z8| / sum w  (concentration on one 45-deg lattice)")
    axes[0].set_ylabel("volumes")
    axes[0].set_title("label-free four-direction evidence")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.25, color="#b00020", ls="--", lw=1)
    axes[1].set_xlabel("lattice concentration")
    axes[1].set_ylabel("four-class accuracy")
    axes[1].set_title("does lattice sharpness predict accuracy?")
    axes[1].legend(fontsize=7)
    fig.suptitle("T-I fig 9 — the 45-degree ply-angle lattice", y=1.03)
    figs["TI_fig9_lattice"] = savefig(fig, OUT_DIR, "TI_fig9_lattice")

    return figs


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run on this many volumes only and print a projection")
    ap.add_argument("--recompute-maps", action="store_true")
    ap.add_argument("--recompute-slabs", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()
    g = zarr.open_group(str(ZARR_ROOT), mode="r")
    volume_ids = sorted(k for k in g.group_keys() if k in gt["volumes"])
    print(f"{len(volume_ids)} volumes with ground truth")

    if args.smoke:
        picks = []
        for fam in FAMILIES:
            c = [v for v in volume_ids if family(v) == fam]
            if c:
                picks.append(c[0])
        picks = picks[:args.smoke] if args.smoke <= len(picks) else picks
        t0 = time.time()
        recs = compute_maps(picks, workers=min(args.workers, len(picks)))
        wall = time.time() - t0
        cpu = np.mean([r["seconds"] for r in recs.values()])
        proj = cpu * len(volume_ids) / args.workers
        print(f"\nSMOKE: {len(picks)} volumes, {wall:.1f}s wall, {cpu:.1f}s CPU/volume")
        print(f"PROJECTED full run: {proj/60:.1f} min at {args.workers} workers "
              f"({cpu*len(volume_ids)/60:.1f} min CPU total)")
        return

    if CACHE.exists() and not args.recompute_maps:
        recs = np.load(CACHE, allow_pickle=True)["records"].item()
        print(f"reused {CACHE}")
    else:
        t0 = time.time()
        recs = compute_maps(volume_ids, workers=args.workers)
        print(f"maps computed in {(time.time()-t0)/60:.1f} min")
        np.savez_compressed(CACHE, records=recs)

    per_volume = analyse(recs, gt)

    # --- stage 2: slab-domain estimators -----------------------------------
    cal = calibrate()
    print(f"estimator calibration: max angle error "
          f"{cal['max_error_over_all_angles_deg']:.1f} deg on synthetic gratings")
    if SLAB_CACHE.exists() and not args.recompute_slabs:
        slab_recs = np.load(SLAB_CACHE, allow_pickle=True)["records"].item()
        print(f"reused {SLAB_CACHE}")
    else:
        jobs = []
        for vid, r in per_volume.items():
            if not r["n_plies_gt"] or "block_edges" not in r.get("raw_scaled", {}):
                continue
            p = r["measured_pitch_voxels"]
            band = (BAND_PITCH[0] * p, BAND_PITCH[1] * p)
            jobs.append((vid, np.asarray(r["raw_scaled"]["block_edges"]), band))
        t0 = time.time()
        slab_recs = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for s in ex.map(volume_slabs, jobs):
                slab_recs[s["volume_id"]] = s
                print(f"  slabs {s['volume_id']}  {s['seconds']:.1f}s", flush=True)
        print(f"slabs computed in {(time.time()-t0)/60:.1f} min")
        np.savez_compressed(SLAB_CACHE, records=slab_recs)

    sl = analyse_slabs(slab_recs, per_volume, gt, recs)
    slab_summary = summarise_slabs(sl)
    slab_summary["single_window_control"] = single_window_control(per_volume, gt)
    slab_summary["window_agreement"] = window_agreement(slab_recs, per_volume, gt)
    slab_summary["depth_coherence"] = depth_coherence(recs, per_volume, gt)
    # does the residual track the per-volume ply-thickness deviation?
    wa = slab_summary["window_agreement"]["per_volume"]
    if len(wa) > 5:
        pit = np.array([per_volume[v]["measured_pitch_voxels"] for v in wa])
        exp = np.array([per_volume[v]["expected_pitch_voxels"] for v in wa])
        res = np.array([wa[v]["residual_rms_deg"] for v in wa])
        real = np.array([wa[v]["real_sd_deg"] for v in wa])
        slab_summary["window_agreement"]["thickness_link"] = {
            "corr_residual_vs_pitch_ratio": float(np.corrcoef(pit / exp, res)[0, 1]),
            "corr_real_sd_vs_pitch_ratio": float(np.corrcoef(pit / exp, real)[0, 1]),
            "pitch_over_expected_median": float(np.median(pit / exp)),
        }

    summary = summarise(per_volume)
    summary["slab_estimators"] = slab_summary
    summary["estimator_calibration"] = cal
    figs = make_figures(per_volume, summary)
    figs.update(make_slab_figures(sl, slab_summary))
    figs.update(make_spectrum_figure(slab_recs, sl, gt))

    field = build_layup_field(per_volume, gt, sl)
    fp = write_json(field, OUT_DIR, "layup_field.json")
    print(f"wrote {fp}  ({field['summary']})")

    results = {
        "test_id": TEST_ID,
        "description": "validation of measured ply orientation against the expert layup",
        "ground_truth_source": gt["source"],
        "n_volumes": len(recs),
        "n_volumes_scored": sum(1 for r in per_volume.values() if r["n_plies_gt"]),
        "excluded_no_ground_truth": [v for v, r in per_volume.items() if not r["n_plies_gt"]],
        "method": {
            "window": WINDOW,
            "window_fracs_y": list(WINDOW_FRACS),
            "n_angular_bins": N_ANG,
            "wavelength_bin_edges_voxels": LAM_EDGES.tolist(),
            "primary_band": "0.82-3.27 x the volume's own measured ply pitch",
            "fixed_band_voxels": list(BAND_FIXED),
            "smooth_deg": SMOOTH_DEG,
            "axis_bins_excised_deg": AXIS_EXCISE,
        },
        "materials": gt.get("materials"),
        "summary": summary,
        "slab_per_volume": sl["per_volume"],
        "layup_field_summary": field["summary"],
        "layup_field_path": fp,
        "per_volume": per_volume,
        "figures": figs,
    }
    p = write_json(results, OUT_DIR)
    print(f"wrote {p}\n")
    print("four-class ply accuracy (chance 0.25), all volumes:")
    for est, blk in slab_summary["per_estimator"].items():
        if "ALL" not in blk:
            continue
        b = blk["ALL"]
        lo = slab_summary["lovo"].get(est, {}).get("ALL", {})
        print(f"  {est:11s} acc {b['class_accuracy_mean']:.3f} "
              f"(null {b['null_class_accuracy_mean']:.3f}), "
              f"median |err| {b['median_abs_error_pooled']:.1f} deg, "
              f"held-out {lo.get('heldout_accuracy_mean', float('nan')):.3f}")
    swc = slab_summary.get("single_window_control", {})
    if swc:
        print("\nsingle centred window (T-H's own cached data), four-class accuracy:")
        for mode in ("absolute_like_TG_TH", "mean_subtracted"):
            d = swc[mode]
            print(f"  {mode:22s} Pegaso {d['Airbus_Panel_Pegaso']['class_accuracy_mean']:.3f}  "
                  f"Nacho {d['Fabricacion_Nacho_05']['class_accuracy_mean']:.3f}  "
                  f"ALL {d['ALL']['class_accuracy_mean']:.3f}")
        print(f"  Pegaso-minus-Nacho gap: {swc['pegaso_minus_nacho_gap']}")
    wa = slab_summary["window_agreement"]["per_family"]
    print("\ntwo-window decomposition of the deviation from the nominal layup:")
    for f, b in wa.items():
        print(f"  {f:22s} residual rms {b['residual_rms_deg_median']:5.1f} deg = "
              f"real {b['real_sd_deg_median']:5.1f} + noise {b['noise_sd_per_window_deg_median']:5.1f} "
              f"| real frac {b['real_fraction_of_variance_median']:.2f} "
              f"| corr(A,B) {b['corr_rA_rB_median']}")
    dc = slab_summary["depth_coherence"]["per_family"]
    print("\ndepth coherence of the within-ply slice-to-slice deviation:")
    for f, b in dc.items():
        print(f"  {f:22s} lag-1 autocorr {b['lag1_autocorr_median']:+.2f} "
              f"(positive in {b['frac_lag1_positive']:.2f} of volumes), "
              f"slice sd {b['slice_deviation_sd_deg_median']:.1f} deg, "
              f"edge/centre {b['abs_dev_edge_over_centre']:.2f}")
    print("\nwindow x excision factorial (all volumes):")
    for name, blk in slab_summary["window_excision_factorial"].items():
        b = blk.get("ALL")
        if b:
            print(f"  {name:20s} acc {b['class_accuracy_mean']:.3f}  "
                  f"median |err| {b['median_abs_error_median']:.1f} deg  "
                  f"lattice {b['lattice_concentration_median']:.3f}")


if __name__ == "__main__":
    main()

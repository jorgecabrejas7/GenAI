"""T-G — In-plane fibre orientation vs. depth: what is the laminate layup?

T-A established a porosity period of 19.4-19.8 voxels along z (the ply pitch).
This test asks whether the *in-plane texture orientation* also changes with
depth, and with what period.  If the plies alternate in orientation, the
orientation pattern repeats over 2x or 4x the ply pitch, and a depth feature
whose shortest wavelength is the ply pitch cannot say which ply orientation the
current slice belongs to.  The answer decides whether the ldm05 depth encoding
needs subharmonic frequencies.

Method, per volume
------------------
* A fixed 1024x1024 in-plane window, centred in y and x, is read for every
  z-slice (the narrowest volume is 1579 voxels in x, so the window is always
  fully inside the specimen).  Every z-slice is used -- no subsampling; a
  volume is only 185-332 slices thick and the whole pass costs ~15 s.
* **Primary estimator — spectral second moment.**  The slice is mean-removed,
  multiplied by a 2-D Hann window, and Fourier transformed.  The power spectrum
  is restricted to an annulus of wavelengths, and the second-moment (structure)
  tensor of the power over unit wave-vectors is formed.  Its doubled-angle
  representation is
      c2 = <p (ux^2 - uy^2)> / <p>,   s2 = <p (2 ux uy)> / <p>
  which is free of the modulo-180-degree wraparound.  The dominant *spectral*
  direction is 0.5*atan2(s2, c2); the real-space texture (fibre/tow) direction
  is perpendicular to it.  The anisotropy strength is A = hypot(c2, s2), in
  [0, 1]: 0 for an isotropic slice, 1 for a perfectly 1-D texture.
* Three annuli are evaluated, because the physical scale of the signal is not
  known in advance: 8-16 voxels (fine), **16-64 voxels (tow scale, primary)**
  and 64-256 voxels (coarse).  Individual carbon fibres are ~7 um and are not
  resolved at this voxel size, so the orientation signal can only come from
  tow-level structure; the band sweep makes that testable rather than assumed.
* **Cross-check — real-space gradient structure tensor.**  On the same slice
  downsampled 2x, J = G_sigma * (grad I grad I^T).  The real-space texture
  direction is the *minimum*-gradient eigenvector.  This is an independent
  estimator that shares no machinery with the FFT one.
* **Cross-check — pore-mask orientation.**  The same spectral estimator applied
  to the binary pore mask.  Pores in CFRP elongate along the fibres, so this is
  a second, physically independent, view of the same quantity.

Depth analysis
--------------
The angle profile is turned into the complex signal u(z) = A(z) exp(2 i theta(z)).
Its periodogram (sum of the periodograms of Re u and Im u) shows the period of
the *orientation* pattern.  A [0/90]n stack gives a period of 2x the ply pitch;
a [0/+-45/90]n stack gives 4x the ply pitch.  Autocorrelation of the same
signal is reported as an independent check.

Cross-check against T-A: the orientation change rate |du/dz| is correlated
against the 1-voxel porosity profile cached by T-A, to test whether ply
interfaces coincide with porosity maxima.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT, ZARR_ROOT, autocorrelation, periodogram,
    savefig, set_style, write_findings, write_json, plt,
)

TEST_ID = "T-G"
OUT_DIR = OUT_ROOT / TEST_ID
TA_PROFILES = OUT_ROOT / "T-A" / "fine_profiles.npz"

WINDOW = 1024                      # in-plane analysis window (power of two)
CHUNK_Z = 32                       # z slab size for streaming reads
BANDS = {                          # wavelength annuli, in voxels
    "fine_8_16": (8.0, 16.0),
    "tow_16_64": (16.0, 64.0),
    "coarse_64_256": (64.0, 256.0),
}
PRIMARY_BAND = "tow_16_64"
PLY_PITCH = 19.6                   # voxels, from T-A
ST_SIGMA = 3.0                     # real-space structure-tensor smoothing (on 2x downsample)
MIN_ANISO = 0.02                   # below this the slice angle is not trusted


# ---------------------------------------------------------------------------
# Per-slice orientation estimators
# ---------------------------------------------------------------------------

def _band_geometry(n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Hann window and, per band, the flattened (ux^2-uy^2, 2 ux uy) weights."""
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    geo = {}
    for name, (lam_lo, lam_hi) in BANDS.items():
        sel = (rad >= 1.0 / lam_hi) & (rad <= 1.0 / lam_lo)
        kx = np.broadcast_to(fx, (n, n))[sel]
        ky = np.broadcast_to(fy, (n, n))[sel]
        nrm = np.sqrt(kx * kx + ky * ky)
        ux, uy = kx / nrm, ky / nrm
        geo[name] = (sel, (ux * ux - uy * uy).astype(np.float32),
                     (2.0 * ux * uy).astype(np.float32))
    return win, geo


def spectral_orientation(img: np.ndarray, win: np.ndarray, geo: dict) -> dict:
    """Doubled-angle spectral second moment of one slice, per band.

    Returns {band: (c2, s2)} where the dominant SPECTRAL direction is
    0.5*atan2(s2, c2) and the anisotropy is hypot(c2, s2).
    """
    a = (img - img.mean()) * win
    p = np.abs(np.fft.fft2(a)) ** 2
    out = {}
    for name, (sel, wc, ws) in geo.items():
        pb = p[sel]
        s = float(pb.sum())
        if s <= 0:
            out[name] = (np.nan, np.nan)
            continue
        out[name] = (float(pb @ wc) / s, float(pb @ ws) / s)
    return out


def realspace_orientation(img: np.ndarray, sigma: float = ST_SIGMA) -> tuple[float, float]:
    """Gradient structure tensor, doubled-angle, on a 2x-downsampled slice.

    Returns (c2, s2) of the GRADIENT direction; the texture direction is
    perpendicular to it, so the same +90-degree convention as the spectral
    estimator applies.
    """
    small = img[::2, ::2].astype(np.float32)
    gy = ndimage.sobel(small, axis=0)
    gx = ndimage.sobel(small, axis=1)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    num_c = float((jxx - jyy).sum())
    num_s = float((2.0 * jxy).sum())
    den = float((jxx + jyy).sum())
    if den <= 0:
        return np.nan, np.nan
    return num_c / den, num_s / den


# ---------------------------------------------------------------------------
# Per-volume worker
# ---------------------------------------------------------------------------

def volume_orientation(volume_id: str, realspace: bool = True) -> dict:
    g = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]
    xct_a, mask_a = g["xct"], g["mask"]
    D, H, W = xct_a.shape
    n = WINDOW
    y0 = max(0, H // 2 - n // 2)
    x0 = max(0, W // 2 - n // 2)
    win, geo = _band_geometry(n)

    c2 = {b: np.full(D, np.nan) for b in BANDS}
    s2 = {b: np.full(D, np.nan) for b in BANDS}
    mc2 = np.full(D, np.nan)
    ms2 = np.full(D, np.nan)
    rc2 = np.full(D, np.nan)
    rs2 = np.full(D, np.nan)
    fg_frac = np.zeros(D)
    pore_frac = np.zeros(D)

    # Otsu threshold from a coarse histogram of the analysis window.
    hist = np.zeros(256, dtype=np.int64)
    for z in range(0, D, CHUNK_Z):
        blk = np.asarray(xct_a[z: min(z + CHUNK_Z, D), y0:y0 + n, x0:x0 + n])[::2, ::4, ::4]
        hist += np.bincount(blk.ravel(), minlength=256)
    total = hist.sum()
    pr = hist / max(total, 1)
    lv = np.arange(256, dtype=np.float64)
    om = np.cumsum(pr)
    mu = np.cumsum(lv * pr)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = np.where((om > 0) & (om < 1), (mu[-1] * om - mu) ** 2 / (om * (1 - om)), 0.0)
    thr = int(np.nanargmax(sb))

    for z in range(0, D, CHUNK_Z):
        z1 = min(z + CHUNK_Z, D)
        blk = np.asarray(xct_a[z:z1, y0:y0 + n, x0:x0 + n])
        mblk = np.asarray(mask_a[z:z1, y0:y0 + n, x0:x0 + n]).astype(bool)
        for i in range(z1 - z):
            k = z + i
            im = blk[i]
            fg = im > thr
            fg_frac[k] = float(fg.mean())
            pore_frac[k] = float(mblk[i].mean())
            imf = im.astype(np.float32)
            for b, (a, c) in spectral_orientation(imf, win, geo).items():
                c2[b][k], s2[b][k] = a, c
            mm = mblk[i].astype(np.float32)
            if mm.any():
                r = spectral_orientation(mm, win, geo)[PRIMARY_BAND]
                mc2[k], ms2[k] = r
            if realspace:
                rc2[k], rs2[k] = realspace_orientation(imf)

    return {
        "volume_id": volume_id,
        "shape": [int(D), int(H), int(W)],
        "window": n,
        "otsu_threshold": thr,
        "fg_frac": fg_frac,
        "pore_frac": pore_frac,
        "c2": {b: c2[b] for b in BANDS},
        "s2": {b: s2[b] for b in BANDS},
        "mask_c2": mc2, "mask_s2": ms2,
        "rs_c2": rc2, "rs_s2": rs2,
    }


def _worker(vid: str) -> dict:
    t0 = time.time()
    out = volume_orientation(vid)
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Depth analysis of one orientation profile
# ---------------------------------------------------------------------------

def interior_mask(fg_frac: np.ndarray, min_fg: float = 0.9) -> np.ndarray:
    """Slices fully inside the specimen (drops the air-contaminated z ends)."""
    ok = fg_frac >= min_fg
    if ok.sum() < 32:
        ok = fg_frac >= max(0.5, np.nanpercentile(fg_frac, 60))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ok
    keep = np.zeros_like(ok)
    keep[idx.min(): idx.max() + 1] = True
    return keep & ok


def orientation_signal(c2: np.ndarray, s2: np.ndarray, keep: np.ndarray) -> dict:
    """Turn (c2, s2) into the analysis quantities on the interior slices."""
    c = np.asarray(c2, float)[keep]
    s = np.asarray(s2, float)[keep]
    good = np.isfinite(c) & np.isfinite(s)
    if good.sum() < 32:
        return {"n": int(good.sum()), "ok": False}
    c = np.where(good, c, 0.0)
    s = np.where(good, s, 0.0)
    aniso = np.hypot(c, s)
    # SPECTRAL angle; real-space texture direction = spectral + 90 deg.
    spec_deg = (np.degrees(0.5 * np.arctan2(s, c))) % 180.0
    tex_deg = (spec_deg + 90.0) % 180.0
    return {
        "n": int(good.sum()), "ok": True,
        "c2": c, "s2": s, "aniso": aniso,
        "texture_angle_deg": tex_deg, "good": good,
    }


def profile_period(c2: np.ndarray, s2: np.ndarray,
                   period_min: float = 5.0, period_max: float = 150.0) -> dict:
    """Periodogram of the complex doubled-angle signal u = c2 + i s2.

    Power of Re u and Im u is summed, so the result is invariant to how the
    scan happens to be rotated in plane.
    """
    n = len(c2)
    if n < 40:
        return {"ok": False, "reason": "profile too short"}
    per, pc = periodogram(c2)
    _, ps = periodogram(s2)
    pw = pc + ps
    band = (per >= period_min) & (per <= period_max) & np.isfinite(per)
    if band.sum() < 3:
        return {"ok": False, "reason": "no bins in band"}
    # background = running median of the spectrum, to remove the red-noise slope
    win = max(5, (band.sum() // 8) * 2 + 1)
    bg = ndimage.median_filter(pw, size=win, mode="nearest")
    excess = pw / np.maximum(bg, 1e-30)
    k = int(np.argmax(np.where(band, excess, -np.inf)))

    def _at(target: float) -> float:
        j = int(np.argmin(np.abs(np.where(band, per, np.inf) - target)))
        return float(excess[j])

    return {
        "ok": True,
        "peak_period_voxels": float(per[k]),
        "peak_excess_over_background": float(excess[k]),
        "excess_at_1x_pitch": _at(PLY_PITCH),
        "excess_at_2x_pitch": _at(2 * PLY_PITCH),
        "excess_at_4x_pitch": _at(4 * PLY_PITCH),
        "period_grid": per[band], "excess_grid": excess[band],
    }


def transition_profile(c2: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """|du/dz| — large where the in-plane orientation turns."""
    u = np.asarray(c2, float) + 1j * np.asarray(s2, float)
    d = np.gradient(u)
    return np.abs(d)


def angle_modes(tex_deg: np.ndarray, aniso: np.ndarray,
                bin_width: float = 5.0) -> dict:
    """Anisotropy-weighted histogram of the texture angle, and its modes."""
    nb = int(round(180.0 / bin_width))
    edges = np.linspace(0, 180, nb + 1)
    h, _ = np.histogram(tex_deg, bins=edges, weights=aniso)
    h = h / max(h.sum(), 1e-30)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # circular smoothing over +-1 bin, then local maxima above 1.5x the mean
    hs = (np.roll(h, 1) + h + np.roll(h, -1)) / 3.0
    peaks = []
    for i in range(nb):
        if hs[i] >= hs[(i - 1) % nb] and hs[i] >= hs[(i + 1) % nb] and hs[i] > 1.5 / nb:
            peaks.append((float(centres[i]), float(hs[i])))
    peaks.sort(key=lambda t: -t[1])
    return {"bin_centres_deg": centres, "weighted_hist": h,
            "modes_deg": [p[0] for p in peaks[:6]],
            "mode_weights": [p[1] for p in peaks[:6]]}


def ply_step_analysis(c2: np.ndarray, s2: np.ndarray,
                      pitch: float = PLY_PITCH) -> dict:
    """Average the orientation over one ply, then step from ply to ply.

    The ply grid offset is chosen to maximise the total within-ply coherence
    |mean u|, i.e. to put the window boundaries on the ply interfaces.  The
    angle between consecutive plies is then read off directly, which is what
    names the layup: steps of ~90 deg mean [0/90], steps of ~45 deg mean a
    [0/+-45/90] family, steps near 0 mean unidirectional.
    """
    u = np.asarray(c2, float) + 1j * np.asarray(s2, float)
    n = len(u)
    if n < 3 * pitch:
        return {"ok": False}
    best = None
    for off in range(int(round(pitch))):
        edges = np.arange(off, n - pitch + 1e-9, pitch)
        if len(edges) < 3:
            continue
        means, coh = [], 0.0
        for e in edges:
            a, b = int(round(e)), int(round(e + pitch))
            seg = u[a:b]
            if len(seg) < 3:
                continue
            m = seg.mean()
            means.append(m)
            coh += abs(m)
        if len(means) < 3:
            continue
        score = coh / len(means)
        if best is None or score > best[0]:
            best = (score, off, np.array(means))
    if best is None:
        return {"ok": False}
    score, off, means = best
    steps = means[1:] * np.conj(means[:-1])
    # real-space angle step, in (-90, 90]
    step_deg = np.degrees(np.angle(steps)) / 2.0
    w = np.minimum(np.abs(means[1:]), np.abs(means[:-1]))
    keep = w >= max(MIN_ANISO, np.percentile(w, 25))
    return {
        "ok": True,
        "ply_grid_offset": int(off),
        "within_ply_coherence": float(score),
        "n_plies": int(len(means)),
        "step_deg": step_deg,
        "step_weight": w,
        "abs_step_deg_median": float(np.median(np.abs(step_deg[keep]))) if keep.any() else None,
        "frac_step_near_0": float(np.mean(np.abs(step_deg[keep]) < 15)) if keep.any() else None,
        "frac_step_near_45": float(np.mean(np.abs(np.abs(step_deg[keep]) - 45) < 15)) if keep.any() else None,
        "frac_step_near_90": float(np.mean(np.abs(step_deg[keep]) > 75)) if keep.any() else None,
    }


def circular_dispersion(c2: np.ndarray, s2: np.ndarray) -> float:
    """1 - |mean(u)|/mean(|u|).  0 = one fixed orientation, 1 = fully spread."""
    u = np.asarray(c2, float) + 1j * np.asarray(s2, float)
    m = np.abs(u).mean()
    if m <= 0:
        return np.nan
    return float(1.0 - np.abs(u.mean()) / m)


def analyse_volume(rec: dict, phi_z: np.ndarray | None) -> dict:
    keep = interior_mask(rec["fg_frac"])
    out = {"volume_id": rec["volume_id"], "shape": rec["shape"],
           "otsu_threshold": rec["otsu_threshold"],
           "n_slices_total": int(len(rec["fg_frac"])),
           "n_slices_interior": int(keep.sum()),
           "seconds": rec.get("seconds")}

    per_band = {}
    for b in BANDS:
        sig = orientation_signal(rec["c2"][b], rec["s2"][b], keep)
        if not sig["ok"]:
            per_band[b] = {"ok": False}
            continue
        pk = profile_period(sig["c2"], sig["s2"])
        modes = angle_modes(sig["texture_angle_deg"], sig["aniso"])
        entry = {
            "ok": True,
            "mean_anisotropy": float(np.mean(sig["aniso"])),
            "median_anisotropy": float(np.median(sig["aniso"])),
            "frac_slices_reliable": float(np.mean(sig["aniso"] >= MIN_ANISO)),
            "circular_dispersion": circular_dispersion(sig["c2"], sig["s2"]),
            "angle_modes_deg": modes["modes_deg"],
            "angle_mode_weights": modes["mode_weights"],
        }
        if pk["ok"]:
            entry.update({k: v for k, v in pk.items()
                          if k not in ("period_grid", "excess_grid", "ok")})
        per_band[b] = entry
    out["bands"] = per_band

    # primary band, kept in full for the figures and the cross-checks
    sig = orientation_signal(rec["c2"][PRIMARY_BAND], rec["s2"][PRIMARY_BAND], keep)
    out["primary_ok"] = sig["ok"]
    if not sig["ok"]:
        return out

    ac = autocorrelation(sig["c2"]) + autocorrelation(sig["s2"])
    ac = ac / max(ac[0], 1e-30)
    lo, hi = 6, min(len(ac) - 1, 120)
    lag = int(lo + np.argmax(ac[lo:hi + 1])) if hi > lo else 0
    out["autocorr_first_peak_lag"] = lag
    out["autocorr_first_peak_value"] = float(ac[lag]) if hi > lo else None
    for k in (1, 2, 3, 4):
        L = int(round(k * PLY_PITCH))
        out[f"autocorr_at_{k}x_pitch"] = float(ac[L]) if len(ac) > L else None

    step = ply_step_analysis(sig["c2"], sig["s2"])
    out["ply_steps"] = {k: v for k, v in step.items()
                        if k not in ("step_deg", "step_weight")}

    # cross-check estimators, same interior slices
    rs = orientation_signal(rec["rs_c2"], rec["rs_s2"], keep)
    if rs["ok"]:
        du = sig["texture_angle_deg"] - rs["texture_angle_deg"]
        du = (du + 90.0) % 180.0 - 90.0
        w = sig["aniso"] * rs["aniso"]
        out["realspace_cross_check"] = {
            "mean_anisotropy": float(np.mean(rs["aniso"])),
            "weighted_median_abs_angle_diff_deg":
                float(np.median(np.abs(du)[w >= np.percentile(w, 50)])),
            "doubled_angle_correlation": float(np.corrcoef(
                np.r_[sig["c2"], sig["s2"]], np.r_[rs["c2"], rs["s2"]])[0, 1]),
        }
        pk = profile_period(rs["c2"], rs["s2"])
        if pk["ok"]:
            out["realspace_cross_check"]["peak_period_voxels"] = pk["peak_period_voxels"]
            out["realspace_cross_check"]["peak_excess"] = pk["peak_excess_over_background"]

    mk = orientation_signal(rec["mask_c2"], rec["mask_s2"], keep)
    if mk["ok"]:
        du = sig["texture_angle_deg"] - mk["texture_angle_deg"]
        du = (du + 90.0) % 180.0 - 90.0
        w = mk["aniso"]
        thr_w = np.percentile(w, 75)
        out["mask_cross_check"] = {
            "mean_anisotropy": float(np.mean(mk["aniso"])),
            "median_abs_angle_diff_deg_top25pct_aniso":
                float(np.median(np.abs(du)[w >= thr_w])),
            "doubled_angle_correlation": float(np.corrcoef(
                np.r_[sig["c2"], sig["s2"]], np.r_[mk["c2"], mk["s2"]])[0, 1]),
        }

    # cross-check against T-A porosity
    if phi_z is not None and len(phi_z) == len(rec["fg_frac"]):
        phi = np.asarray(phi_z, float)[keep]
        g = transition_profile(sig["c2"], sig["s2"])
        good = np.isfinite(phi)
        if good.sum() > 40:
            p = phi[good] - np.nanmean(phi[good])
            q = g[good] - g[good].mean()
            denom = np.sqrt((p ** 2).sum() * (q ** 2).sum())
            r0 = float((p * q).sum() / denom) if denom > 0 else np.nan
            # lag scan over +-1 ply pitch
            lags = np.arange(-20, 21)
            rr = []
            for L in lags:
                pp = np.roll(p, L)
                d2 = np.sqrt((pp ** 2).sum() * (q ** 2).sum())
                rr.append((pp * q).sum() / d2 if d2 > 0 else np.nan)
            rr = np.array(rr)
            best = int(np.nanargmax(np.abs(rr)))
            out["porosity_cross_check"] = {
                "corr_phi_vs_orientation_change_lag0": r0,
                "best_lag_voxels": int(lags[best]),
                "best_corr": float(rr[best]),
                "lags": lags, "corr_vs_lag": rr,
            }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_examples(records: dict, analyses: dict, phis: dict, vids: list[str]) -> list[str]:
    n = len(vids)
    fig, axs = plt.subplots(n, 1, figsize=(7.2, 2.35 * n), sharex=False)
    axs = np.atleast_1d(axs)
    for ax, vid in zip(axs, vids):
        rec = records[vid]
        keep = interior_mask(rec["fg_frac"])
        sig = orientation_signal(rec["c2"][PRIMARY_BAND], rec["s2"][PRIMARY_BAND], keep)
        z = np.flatnonzero(keep)
        a = sig["aniso"]
        rel = a >= MIN_ANISO
        ax.scatter(z[rel], sig["texture_angle_deg"][rel], s=6,
                   c=a[rel], cmap="viridis", vmin=0, vmax=max(0.05, np.percentile(a, 95)))
        ax.scatter(z[~rel], sig["texture_angle_deg"][~rel], s=4, c="0.75", marker="x",
                   linewidths=0.5)
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_ylabel("texture angle (deg)")
        ax.set_title(vid.split("__")[1][:58], fontsize=9)
        phi = phis.get(vid)
        if phi is not None and len(phi) == len(rec["fg_frac"]):
            ax2 = ax.twinx()
            ax2.plot(z, np.asarray(phi, float)[keep], color="#c2571a", lw=0.9, alpha=0.8)
            ax2.set_ylabel("porosity $\\varphi$", color="#c2571a")
            ax2.tick_params(axis="y", colors="#c2571a")
            ax2.grid(False)
        ax.set_xlabel("z (voxels)")
    fig.suptitle("T-G — in-plane texture orientation vs. depth (colour = anisotropy;\n"
                 "grey crosses = unreliable slice; orange = T-A porosity profile)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return savefig(fig, OUT_DIR, "TG_fig1_orientation_vs_depth")


def fig_spectrum(records: dict, stacked: dict) -> list[str]:
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axs[0]
    for b, col in zip(BANDS, ("#888888", "#1b6ca8", "#2e7d32")):
        s = stacked[b]
        ax.plot(s["period_grid"], s["median_excess"], color=col, label=b.replace("_", " "))
    for m, lab in ((1, "1x pitch"), (2, "2x"), (4, "4x")):
        ax.axvline(m * PLY_PITCH, color="#c2571a", ls="--", lw=0.9)
        ax.text(m * PLY_PITCH, ax.get_ylim()[1], f" {lab}", color="#c2571a",
                fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("period of the orientation signal (voxels)")
    ax.set_ylabel("median spectral excess over background")
    ax.set_title("Orientation power spectrum, 80 volumes")
    ax.legend()

    ax = axs[1]
    s = stacked["autocorr"]
    ax.plot(s["lag"], s["median"], color="#1b6ca8")
    ax.fill_between(s["lag"], s["p25"], s["p75"], color="#1b6ca8", alpha=0.2)
    for m in (1, 2, 4):
        ax.axvline(m * PLY_PITCH, color="#c2571a", ls="--", lw=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (voxels)")
    ax.set_ylabel("autocorrelation of $u=e^{2i\\theta}$")
    ax.set_title("Orientation autocorrelation (median, IQR)")
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "TG_fig2_orientation_spectrum")


FAM_COLORS = {"Airbus_Panel_Pegaso": "#1b6ca8",
              "Fabricacion_Nacho_05": "#c2571a",
              "Juan_Ignacio": "#2e7d32"}


def fig_layup(ply_steps: dict, stacked: dict) -> list[str]:
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axs[0]
    for fam, s in ply_steps.items():
        e = np.asarray(s["abs_step_hist_edges"])
        h = np.asarray(s["abs_step_hist_counts"], float)
        h = h / max(h.sum(), 1)
        ax.step(0.5 * (e[:-1] + e[1:]), h, where="mid",
                color=FAM_COLORS.get(fam, "0.4"),
                label=f"{fam} ({s['n_ply_steps']} steps)")
    for a in (0, 45, 90):
        ax.axvline(a, color="k", ls=":", lw=0.8)
    ax.set_xlabel("|angle change| between consecutive plies (deg)")
    ax.set_ylabel("fraction of ply-to-ply steps")
    ax.set_title("Ply-to-ply orientation step\n(ply grid = 19.6 voxels from T-A)")
    ax.legend(fontsize=7.5)

    ax = axs[1]
    for fam, s in stacked["autocorr"]["by_family"].items():
        ax.plot(stacked["autocorr"]["lag"], s["median"],
                color=FAM_COLORS.get(fam, "0.4"),
                label=f"{fam} (n={s['n_volumes']})")
    for k in (1, 2, 3, 4):
        ax.axvline(k * PLY_PITCH, color="#777777", ls="--", lw=0.8)
        ax.text(k * PLY_PITCH, 1.0, f"{k}x", fontsize=8, color="#777777", va="top")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (voxels)")
    ax.set_ylabel("autocorrelation of $u=e^{2i\\theta}$")
    ax.set_title("Orientation autocorrelation by specimen family")
    ax.legend(fontsize=7.5)
    fig.suptitle("T-G — inferred layup", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return savefig(fig, OUT_DIR, "TG_fig4_layup_steps")


def fig_consistency(analyses: dict, families: dict, stacked: dict) -> list[str]:
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.0))

    ax = axs[0, 0]
    vals = [a["bands"][PRIMARY_BAND]["mean_anisotropy"]
            for a in analyses.values() if a["bands"][PRIMARY_BAND].get("ok")]
    fams = [families[v] for v, a in analyses.items() if a["bands"][PRIMARY_BAND].get("ok")]
    for fam, col in zip(sorted(set(fams)), ("#1b6ca8", "#c2571a", "#2e7d32")):
        d = [v for v, f in zip(vals, fams) if f == fam]
        ax.hist(d, bins=np.linspace(0, max(vals) * 1.05, 26), alpha=0.65,
                label=f"{fam} (n={len(d)})", color=col)
    ax.axvline(MIN_ANISO, color="k", ls=":", lw=1.0)
    ax.set_xlabel("mean slice anisotropy (tow band)")
    ax.set_ylabel("volumes")
    ax.set_title("How strongly oriented the slices are")
    ax.legend(fontsize=7.5)

    ax = axs[0, 1]
    vals = [a["circular_dispersion"] for a in
            (x["bands"][PRIMARY_BAND] for x in analyses.values()) if a.get("ok")]
    for fam, col in zip(sorted(set(fams)), ("#1b6ca8", "#c2571a", "#2e7d32")):
        d = [v for v, f in zip(vals, fams) if f == fam]
        ax.hist(d, bins=np.linspace(0, 1, 26), alpha=0.65, color=col,
                label=f"{fam} (n={len(d)})")
    ax.set_xlabel("circular dispersion of orientation over depth")
    ax.set_ylabel("volumes")
    ax.set_title("0 = one fixed angle (UD), 1 = fully alternating")
    ax.legend(fontsize=7.5)

    ax = axs[1, 0]
    pks = [a["bands"][PRIMARY_BAND].get("peak_period_voxels")
           for a in analyses.values() if a["bands"][PRIMARY_BAND].get("ok")]
    pks = [p for p in pks if p is not None]
    ax.hist(pks, bins=np.logspace(np.log10(5), np.log10(150), 40), color="#1b6ca8")
    for m, lab in ((1, "1x"), (2, "2x"), (4, "4x")):
        ax.axvline(m * PLY_PITCH, color="#c2571a", ls="--", lw=0.9)
        ax.text(m * PLY_PITCH, ax.get_ylim()[1] * 0.95, f" {lab} pitch",
                color="#c2571a", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("dominant orientation period (voxels)")
    ax.set_ylabel("volumes")
    ax.set_title("Per-volume dominant orientation period")

    ax = axs[1, 1]
    cc = [a["porosity_cross_check"]["best_corr"] for a in analyses.values()
          if "porosity_cross_check" in a]
    lg = [a["porosity_cross_check"]["best_lag_voxels"] for a in analyses.values()
          if "porosity_cross_check" in a]
    ax.scatter(lg, cc, s=14, c="#1b6ca8", alpha=0.75)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="#c2571a", ls="--", lw=0.9)
    ax.set_xlabel("lag of best correlation (voxels)")
    ax.set_ylabel("corr($\\varphi$, |d$u$/dz|)")
    ax.set_title("Do orientation turns sit on porosity peaks?")
    fig.suptitle("T-G — cross-volume consistency (80 volumes)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return savefig(fig, OUT_DIR, "TG_fig3_cross_volume_consistency")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def family_of(vid: str) -> str:
    tail = vid.split("__", 1)[1]
    return tail.split("_probetas")[0].split("_Probetas")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N volumes and print the projected full runtime")
    args = ap.parse_args()

    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ta = np.load(TA_PROFILES, allow_pickle=True)["profiles"].item()
    vids = sorted(ta.keys())
    phis = {v: np.asarray(ta[v]["z"]["phi"], float) for v in vids}
    families = {v: family_of(v) for v in vids}
    run_vids = vids[: args.smoke] if args.smoke else vids

    print(f"[T-G] {len(run_vids)} volumes, window {WINDOW}x{WINDOW}, "
          f"{args.workers} workers", flush=True)

    cache = OUT_DIR / "orientation_profiles.npz"
    if cache.exists() and not args.smoke:
        print(f"[T-G] loading cached orientation profiles from {cache}", flush=True)
        records = np.load(cache, allow_pickle=True)["records"].item()
        wall = 0.0
        per_vol = float(np.mean([r["seconds"] for r in records.values()]))
    else:
        records = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, v): v for v in run_vids}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                records[r["volume_id"]] = r
                print(f"  [{i:3d}/{len(run_vids)}] {r['volume_id'][-38:]:38s} "
                      f"D={r['shape'][0]:3d} otsu={r['otsu_threshold']:3d} "
                      f"{r['seconds']:5.1f}s elapsed={time.time() - t0:6.1f}s", flush=True)
        wall = time.time() - t0
        per_vol = float(np.mean([r["seconds"] for r in records.values()]))
        print(f"[T-G] {len(records)} volumes in {wall:.1f}s "
              f"({per_vol:.1f}s cpu/volume)", flush=True)
        if args.smoke:
            proj = per_vol * len(vids) / args.workers
            print(f"[T-G] SMOKE: projected full run over {len(vids)} volumes with "
                  f"{args.workers} workers = {proj / 60:.1f} min", flush=True)
            return
        np.savez_compressed(cache, records=np.array(records, dtype=object))

    print("[T-G] depth analysis ...", flush=True)
    analyses = {v: analyse_volume(records[v], phis.get(v)) for v in records}

    # stacked spectra / autocorrelation
    grid = np.logspace(np.log10(5), np.log10(150), 300)
    stacked = {}
    for b in BANDS:
        rows = []
        for v, rec in records.items():
            keep = interior_mask(rec["fg_frac"])
            sig = orientation_signal(rec["c2"][b], rec["s2"][b], keep)
            if not sig["ok"]:
                continue
            pk = profile_period(sig["c2"], sig["s2"])
            if not pk["ok"]:
                continue
            rows.append(np.interp(grid, pk["period_grid"][::-1],
                                  pk["excess_grid"][::-1], left=np.nan, right=np.nan))
        arr = np.array(rows)
        stacked[b] = {"period_grid": grid,
                      "median_excess": np.nanmedian(arr, axis=0),
                      "n_volumes": int(arr.shape[0])}
    max_lag = 120
    rows, row_fams = [], []
    steps_by_family: dict[str, list[np.ndarray]] = {}
    step_w_by_family: dict[str, list[np.ndarray]] = {}
    for v, rec in records.items():
        keep = interior_mask(rec["fg_frac"])
        sig = orientation_signal(rec["c2"][PRIMARY_BAND], rec["s2"][PRIMARY_BAND], keep)
        if not sig["ok"]:
            continue
        st = ply_step_analysis(sig["c2"], sig["s2"])
        if st["ok"]:
            steps_by_family.setdefault(families[v], []).append(st["step_deg"])
            step_w_by_family.setdefault(families[v], []).append(st["step_weight"])
        if sig["n"] < max_lag + 10:
            continue
        ac = autocorrelation(sig["c2"], max_lag) + autocorrelation(sig["s2"], max_lag)
        rows.append(ac / max(ac[0], 1e-30))
        row_fams.append(families[v])
    arr = np.array(rows)
    row_fams = np.array(row_fams)
    stacked["autocorr"] = {"lag": np.arange(max_lag + 1),
                           "median": np.nanmedian(arr, axis=0),
                           "p25": np.nanpercentile(arr, 25, axis=0),
                           "p75": np.nanpercentile(arr, 75, axis=0),
                           "n_volumes": int(arr.shape[0]),
                           "by_family": {
                               f: {"median": np.nanmedian(arr[row_fams == f], axis=0),
                                   "n_volumes": int((row_fams == f).sum())}
                               for f in sorted(set(row_fams.tolist()))}}

    ply_steps = {}
    for fam, lst in steps_by_family.items():
        s = np.concatenate(lst)
        w = np.concatenate(step_w_by_family[fam])
        keep_s = w >= MIN_ANISO
        s = s[keep_s]
        if s.size == 0:
            continue
        h, edges = np.histogram(np.abs(s), bins=np.linspace(0, 90, 19))
        ply_steps[fam] = {
            "n_ply_steps": int(s.size),
            "abs_step_hist_counts": h, "abs_step_hist_edges": edges,
            "median_abs_step_deg": float(np.median(np.abs(s))),
            "frac_near_0_deg": float(np.mean(np.abs(s) < 15)),
            "frac_near_45_deg": float(np.mean(np.abs(np.abs(s) - 45) < 15)),
            "frac_near_90_deg": float(np.mean(np.abs(s) > 75)),
        }

    # ------------------------------------------------------------------
    # summary numbers
    # ------------------------------------------------------------------
    def _band_summary(b: str) -> dict:
        ok = [a["bands"][b] for a in analyses.values() if a["bands"][b].get("ok")]
        pk = np.array([e["peak_period_voxels"] for e in ok if "peak_period_voxels" in e])
        return {
            "n_volumes": len(ok),
            "mean_anisotropy_median": float(np.median([e["mean_anisotropy"] for e in ok])),
            "circular_dispersion_median": float(np.median([e["circular_dispersion"] for e in ok])),
            "frac_slices_reliable_median": float(np.median([e["frac_slices_reliable"] for e in ok])),
            "peak_period_median_voxels": float(np.median(pk)) if pk.size else None,
            "frac_peak_near_1x": float(np.mean(np.abs(pk / PLY_PITCH - 1) < 0.2)) if pk.size else None,
            "frac_peak_near_2x": float(np.mean(np.abs(pk / (2 * PLY_PITCH) - 1) < 0.2)) if pk.size else None,
            "frac_peak_near_4x": float(np.mean(np.abs(pk / (4 * PLY_PITCH) - 1) < 0.2)) if pk.size else None,
            "median_excess_at_1x": float(np.median([e["excess_at_1x_pitch"] for e in ok if "excess_at_1x_pitch" in e])),
            "median_excess_at_2x": float(np.median([e["excess_at_2x_pitch"] for e in ok if "excess_at_2x_pitch" in e])),
            "median_excess_at_4x": float(np.median([e["excess_at_4x_pitch"] for e in ok if "excess_at_4x_pitch" in e])),
        }

    per_family = {}
    for fam in sorted(set(families.values())):
        sub = [a for v, a in analyses.items() if families[v] == fam
               and a["bands"][PRIMARY_BAND].get("ok")]
        if not sub:
            continue
        disp = np.array([a["bands"][PRIMARY_BAND]["circular_dispersion"] for a in sub])
        ani = np.array([a["bands"][PRIMARY_BAND]["mean_anisotropy"] for a in sub])
        pk = np.array([a["bands"][PRIMARY_BAND].get("peak_period_voxels", np.nan) for a in sub])
        nmodes = np.array([len(a["bands"][PRIMARY_BAND]["angle_modes_deg"]) for a in sub])
        acs = {f"autocorr_at_{k}x_pitch_median": float(np.nanmedian(
            [a.get(f"autocorr_at_{k}x_pitch") or np.nan for a in sub])) for k in (1, 2, 3, 4)}
        per_family[fam] = {
            "n_volumes": len(sub),
            **acs,
            "mean_anisotropy_median": float(np.median(ani)),
            "circular_dispersion_median": float(np.median(disp)),
            "circular_dispersion_p10_p90": [float(np.percentile(disp, 10)),
                                            float(np.percentile(disp, 90))],
            "frac_volumes_alternating_disp_gt_0p3": float(np.mean(disp > 0.3)),
            "peak_period_median_voxels": float(np.nanmedian(pk)),
            "n_angle_modes_median": float(np.median(nmodes)),
            "excess_at_2x_median": float(np.median(
                [a["bands"][PRIMARY_BAND].get("excess_at_2x_pitch", np.nan) for a in sub])),
            "excess_at_4x_median": float(np.median(
                [a["bands"][PRIMARY_BAND].get("excess_at_4x_pitch", np.nan) for a in sub])),
        }

    xc = [a["porosity_cross_check"] for a in analyses.values() if "porosity_cross_check" in a]
    porosity_summary = {
        "n_volumes": len(xc),
        "median_corr_lag0": float(np.median([e["corr_phi_vs_orientation_change_lag0"] for e in xc])),
        "median_best_corr": float(np.median([e["best_corr"] for e in xc])),
        "median_best_lag_voxels": float(np.median([e["best_lag_voxels"] for e in xc])),
        "frac_positive_corr_lag0": float(np.mean(
            [e["corr_phi_vs_orientation_change_lag0"] > 0 for e in xc])),
        "frac_best_lag_within_3_voxels": float(np.mean(
            [abs(e["best_lag_voxels"]) <= 3 for e in xc])),
    }

    rsc = [a["realspace_cross_check"] for a in analyses.values() if "realspace_cross_check" in a]
    mkc = [a["mask_cross_check"] for a in analyses.values() if "mask_cross_check" in a]
    cross = {
        "realspace_structure_tensor": {
            "n_volumes": len(rsc),
            "median_abs_angle_diff_deg": float(np.median(
                [e["weighted_median_abs_angle_diff_deg"] for e in rsc])),
            "median_doubled_angle_correlation": float(np.median(
                [e["doubled_angle_correlation"] for e in rsc])),
            "median_peak_period_voxels": float(np.nanmedian(
                [e.get("peak_period_voxels", np.nan) for e in rsc])),
        },
        "pore_mask_spectral": {
            "n_volumes": len(mkc),
            "median_abs_angle_diff_deg_top25pct": float(np.median(
                [e["median_abs_angle_diff_deg_top25pct_aniso"] for e in mkc])),
            "median_doubled_angle_correlation": float(np.median(
                [e["doubled_angle_correlation"] for e in mkc])),
            "median_mean_anisotropy": float(np.median([e["mean_anisotropy"] for e in mkc])),
        },
    }

    # example volumes for figure 1: most alternating, most constant, median
    disp_all = {v: a["bands"][PRIMARY_BAND]["circular_dispersion"]
                for v, a in analyses.items() if a["bands"][PRIMARY_BAND].get("ok")}
    order = sorted(disp_all, key=lambda v: disp_all[v])
    ex_vids = [order[-1], order[len(order) // 2], order[0]]

    print("[T-G] figures ...", flush=True)
    figs = {}
    figs["fig1"] = fig_examples(records, analyses, phis, ex_vids)
    figs["fig2"] = fig_spectrum(records, stacked)
    figs["fig3"] = fig_consistency(analyses, families, stacked)
    figs["fig4"] = fig_layup(ply_steps, stacked)

    results = {
        "test_id": TEST_ID,
        "method": {
            "window_voxels": WINDOW,
            "slices": "every z-slice, no subsampling",
            "bands_wavelength_voxels": {k: list(v) for k, v in BANDS.items()},
            "primary_band": PRIMARY_BAND,
            "ply_pitch_voxels_from_T_A": PLY_PITCH,
            "anisotropy_reliability_threshold": MIN_ANISO,
            "angle_convention": ("reported texture angle = real-space direction of the "
                                 "elongated structure = spectral second-moment direction "
                                 "+ 90 deg; angles are mod 180 deg and all depth "
                                 "statistics use the doubled-angle vector (c2, s2)"),
        },
        "runtime": {"volumes": len(records), "wall_seconds": wall,
                    "cpu_seconds_per_volume": float(per_vol), "workers": args.workers},
        "band_summary": {b: _band_summary(b) for b in BANDS},
        "per_family": per_family,
        "ply_step_analysis": ply_steps,
        "porosity_cross_check_summary": porosity_summary,
        "estimator_cross_checks": cross,
        "stacked": {b: {"period_grid": stacked[b]["period_grid"],
                        "median_excess": stacked[b]["median_excess"],
                        "n_volumes": stacked[b]["n_volumes"]} for b in BANDS},
        "stacked_autocorr": stacked["autocorr"],
        "example_volumes_in_fig1": ex_vids,
        "figures": figs,
        "per_volume": {v: {k: a[k] for k in a if k not in ("porosity_cross_check",)}
                       for v, a in analyses.items()},
        "per_volume_porosity_cross_check": {
            v: {k: val for k, val in a["porosity_cross_check"].items()
                if k not in ("lags", "corr_vs_lag")}
            for v, a in analyses.items() if "porosity_cross_check" in a},
        "families": families,
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-G] wrote {p}", flush=True)

    # console summary
    print("\n--- T-G summary ---")
    for b in BANDS:
        s = results["band_summary"][b]
        print(f"{b:14s} aniso={s['mean_anisotropy_median']:.4f} "
              f"disp={s['circular_dispersion_median']:.3f} "
              f"peak={s['peak_period_median_voxels']} "
              f"1x/2x/4x excess = {s['median_excess_at_1x']:.2f}/"
              f"{s['median_excess_at_2x']:.2f}/{s['median_excess_at_4x']:.2f} "
              f"near1x={s['frac_peak_near_1x']:.2f} near2x={s['frac_peak_near_2x']:.2f} "
              f"near4x={s['frac_peak_near_4x']:.2f}")
    for fam, s in per_family.items():
        print(f"{fam:22s} n={s['n_volumes']:3d} aniso={s['mean_anisotropy_median']:.4f} "
              f"disp={s['circular_dispersion_median']:.3f} "
              f"alt_frac={s['frac_volumes_alternating_disp_gt_0p3']:.2f} "
              f"peak={s['peak_period_median_voxels']:.1f} "
              f"modes={s['n_angle_modes_median']:.1f}")
    for fam, s in ply_steps.items():
        print(f"{fam:22s} ply steps n={s['n_ply_steps']:5d} "
              f"median|step|={s['median_abs_step_deg']:5.1f} deg  "
              f"0deg={s['frac_near_0_deg']:.2f} 45deg={s['frac_near_45_deg']:.2f} "
              f"90deg={s['frac_near_90_deg']:.2f}")
    for fam, s in per_family.items():
        print(f"{fam:22s} autocorr 1x/2x/3x/4x = "
              + "/".join(f"{s[f'autocorr_at_{k}x_pitch_median']:+.2f}" for k in (1, 2, 3, 4)))
    print("porosity cross-check:", porosity_summary)
    print("estimator cross-checks:", cross)


if __name__ == "__main__":
    main()

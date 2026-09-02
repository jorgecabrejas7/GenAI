"""T-H — Shape of the per-slice angular power distribution: one lobe or two?

T-G estimated ONE dominant in-plane angle per z-slice, from the doubled-angle
second moment of the 2-D power spectrum, and reported that the
`Fabricacion_Nacho_05` family (50 of 80 volumes) is predominantly
UNIDIRECTIONAL (circular dispersion 0.145, one angle mode).  The domain expert
says that is wrong.

The suspected failure is structural, not numerical.  In the doubled-angle
representation u = exp(2 i theta), two equal lobes 90 degrees apart cancel
exactly:  exp(2i*0) + exp(2i*pi/2) = 1 + (-1) = 0.  A cross-ply or woven slice
therefore returns anisotropy ~0 and an angle set by whatever residual is left.
A genuinely unidirectional slice with weak texture and a failed estimate on a
cross-ply slice are indistinguishable in every number T-G reported.

T-H measures the SHAPE of the angular power distribution instead of its first
doubled-angle moment.

Method, per z-slice
-------------------
* Same volume streaming, same 1024x1024 centred in-plane window, same slices
  and the same wavelength annuli as T-G, so every number here is directly
  comparable.  The primary band is the tow band, 16-64 voxels.
* The slice is mean-removed, Hann windowed, Fourier transformed.  The power in
  the annulus is binned by wave-vector angle into 180 bins of 1 degree over
  0-180 degrees (orientation is mod 180).  Each bin is divided by the NUMBER of
  spectrum pixels that fall in it, so the histogram is a mean power *density*
  and is not biased by the anisotropic pixel density of a square frequency grid
  (which favours 0/45/90 degrees).
* The histogram is rolled by 90 degrees, so every angle reported by this script
  is a REAL-SPACE TEXTURE angle, in the same convention as T-G.

Decomposition
-------------
With h(theta) the normalised angular density and t = theta in radians,

    C_m = sum_j h_j exp(-i m t_j) / sum_j h_j        (m = 2, 4, 6)
    a_m = 2 |C_m|            (amplitude, relative to the mean power)
    phi_m = arg(C_m) / m     (phase, i.e. the lobe direction, mod 360/m degrees)

so that  h(theta) ~ mean * (1 + a_2 cos 2(theta - phi_2) + a_4 cos 4(theta - phi_4) + ...).

|C_2| is exactly the T-G anisotropy A, computed from the same power in the same
annulus, which makes the two tests numerically cross-checkable.

**Interpretation caveat that drives the whole analysis.**  A strong a_4 alone
does NOT prove two lobes: a single NARROW lobe has harmonics at every m.  For a
single lobe the harmonics decay (a_4 < a_2) and the phases align (phi_4 = phi_2
modulo 90 degrees, the period of phi_4).  Two equal lobes 90 degrees apart give a_2 ~ 0 with a_4
large.  The discriminants used here are therefore the RATIO a_4/a_2 and the
phase alignment, plus a direct peak count as an independent check.

Lobe counting
-------------
The normalised histogram is smoothed circularly (period 180 degrees, Gaussian
sigma 4 degrees) and peaks are found with a prominence criterion, on the
circularly tiled array so that a lobe straddling 0/180 is not split.  The
angular separations between detected lobes are reported; a 0/90 cross-ply or a
plain-weave fabric must give two lobes about 90 degrees apart.

Cross-checks
------------
* The same decomposition on the binary PORE MASK: pores in CFRP elongate along
  the fibres, so the mask is a physically independent view of the same texture.
* Artifact control: ring/stripe artifacts and the `_aligned` resampling put
  power on the kx = 0 and ky = 0 lines of the spectrum, which appear as lobes at
  exactly 0 and 90 degrees in image coordinates.  The script reports how much of
  the detected lobe population sits within a few degrees of 0/90, per family and
  per volume, so this can be judged rather than assumed.
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
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT, ZARR_ROOT, autocorrelation, periodogram,
    savefig, set_style, write_findings, write_json, plt,
)

TEST_ID = "T-H"
OUT_DIR = OUT_ROOT / TEST_ID
TG_DIR = OUT_ROOT / "T-G"
TG_CACHE = TG_DIR / "orientation_profiles.npz"

WINDOW = 1024                 # identical to T-G
CHUNK_Z = 32
BANDS = {                     # identical to T-G
    "fine_8_16": (8.0, 16.0),
    "tow_16_64": (16.0, 64.0),
    "coarse_64_256": (64.0, 256.0),
}
PRIMARY_BAND = "tow_16_64"
NBINS = 180                   # 1 degree per bin, 0-180 degrees
PLY_PITCH = 19.6              # voxels, from T-A
SMOOTH_SIGMA_DEG = 4.0        # circular smoothing before peak finding
MIN_MODULATION = 0.02         # below this a slice has no resolvable texture
PEAK_PROM_ABS = 0.03          # prominence, in units of the mean power
PEAK_PROM_REL = 0.25          # prominence, as a fraction of the peak-to-trough range

FAM_COLORS = {"Airbus_Panel_Pegaso": "#1b6ca8",
              "Fabricacion_Nacho_05": "#c2571a",
              "Juan_Ignacio": "#2e7d32"}
FAM_ORDER = ["Airbus_Panel_Pegaso", "Fabricacion_Nacho_05", "Juan_Ignacio"]

BIN_CENTRES = np.arange(NBINS) + 0.5          # degrees, texture angle


# ---------------------------------------------------------------------------
# Spectrum geometry
# ---------------------------------------------------------------------------

def _geometry(n: int):
    """Hann window, and per band the selected spectrum pixels with their angle bin.

    Returns (win, {band: (sel, bin_index, bin_counts)}).  ``bin_index`` is the
    0-179 angular bin of every selected pixel, in SPECTRAL angle; the roll to
    texture angle happens later, once, on the histogram.
    """
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    ang = np.degrees(np.arctan2(np.broadcast_to(fy, (n, n)),
                                np.broadcast_to(fx, (n, n)))) % 180.0
    geo = {}
    for name, (lam_lo, lam_hi) in BANDS.items():
        sel = (rad >= 1.0 / lam_hi) & (rad <= 1.0 / lam_lo)
        idx = np.clip((ang[sel] / 180.0 * NBINS).astype(np.int64), 0, NBINS - 1)
        counts = np.bincount(idx, minlength=NBINS).astype(np.float64)
        geo[name] = (sel, idx, counts)
    return win, geo


def slice_angular_hist(img: np.ndarray, win: np.ndarray, geo: dict,
                       bands: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Summed power per angular bin, per band, for one slice (spectral angle)."""
    a = (img - img.mean()) * win
    p = np.abs(np.fft.fft2(a)) ** 2
    out = {}
    for name in bands:
        sel, idx, _ = geo[name]
        out[name] = np.bincount(idx, weights=p[sel], minlength=NBINS)
    return out


# ---------------------------------------------------------------------------
# Per-volume worker
# ---------------------------------------------------------------------------

def volume_hists(volume_id: str) -> dict:
    g = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]
    xct_a, mask_a = g["xct"], g["mask"]
    D, H, W = xct_a.shape
    n = WINDOW
    y0 = max(0, H // 2 - n // 2)
    x0 = max(0, W // 2 - n // 2)
    win, geo = _geometry(n)

    hists = {b: np.zeros((D, NBINS), dtype=np.float64) for b in BANDS}
    mask_hist = np.zeros((D, NBINS), dtype=np.float64)
    pore_frac = np.zeros(D)

    for z in range(0, D, CHUNK_Z):
        z1 = min(z + CHUNK_Z, D)
        blk = np.asarray(xct_a[z:z1, y0:y0 + n, x0:x0 + n])
        mblk = np.asarray(mask_a[z:z1, y0:y0 + n, x0:x0 + n]).astype(bool)
        for i in range(z1 - z):
            k = z + i
            imf = blk[i].astype(np.float32)
            for b, h in slice_angular_hist(imf, win, geo, tuple(BANDS)).items():
                hists[b][k] = h
            pore_frac[k] = float(mblk[i].mean())
            if mblk[i].any():
                mask_hist[k] = slice_angular_hist(
                    mblk[i].astype(np.float32), win, geo, (PRIMARY_BAND,))[PRIMARY_BAND]

    return {
        "volume_id": volume_id,
        "shape": [int(D), int(H), int(W)],
        "hists": {b: hists[b].astype(np.float32) for b in BANDS},
        "mask_hist": mask_hist.astype(np.float32),
        "pore_frac": pore_frac,
        "bin_counts": {b: geo[b][2] for b in BANDS},
    }


def _worker(vid: str) -> dict:
    t0 = time.time()
    out = volume_hists(vid)
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Angular-shape analysis of one histogram row
# ---------------------------------------------------------------------------

_T = np.deg2rad(BIN_CENTRES)
_EXP = {m: np.exp(-1j * m * _T) for m in (2, 4, 6)}


def to_density(hist_sum: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean power per angular bin, rolled from spectral to texture angle.

    Dividing by the pixel count per bin removes the square-grid bias of a
    discrete frequency plane.  The 90-degree roll converts the spectral
    direction into the real-space texture direction (T-G convention).
    """
    c = np.maximum(counts, 1.0)
    d = np.asarray(hist_sum, float) / c
    return np.roll(d, NBINS // 2, axis=-1)


def fourier_modes(dens: np.ndarray) -> dict:
    """Amplitudes and phases of the 2-, 4- and 6-theta modes of h(theta).

    ``dens`` may be (NBINS,) or (D, NBINS).  Amplitudes are 2|C_m| so that
    h ~ mean * (1 + a_m cos m(theta - phi_m)).  Phases are in degrees, in
    [0, 180/m).
    """
    d = np.atleast_2d(np.asarray(dens, float))
    tot = d.sum(axis=1)
    out = {}
    for m, e in _EXP.items():
        c = (d @ e) / np.maximum(tot, 1e-30)
        out[f"a{m}"] = 2.0 * np.abs(c)
        out[f"phi{m}"] = (np.degrees(np.angle(c)) / m) % (360.0 / m)
        out[f"C{m}"] = c
    out["mean_power"] = tot / d.shape[1]
    return out


def _smooth_circ(g: np.ndarray, sigma_deg: float = SMOOTH_SIGMA_DEG) -> np.ndarray:
    return ndimage.gaussian_filter1d(g, sigma_deg, axis=-1, mode="wrap")


def count_lobes(dens: np.ndarray) -> dict:
    """Peak count and lobe angles of one smoothed, normalised angular density."""
    g = _smooth_circ(np.asarray(dens, float) / max(np.mean(dens), 1e-30))
    rng = float(g.max() - g.min())
    prom = max(PEAK_PROM_ABS, PEAK_PROM_REL * rng)
    tiled = np.concatenate([g, g, g])
    pk, props = find_peaks(tiled, prominence=prom)
    keep = (pk >= NBINS) & (pk < 2 * NBINS)
    pk = pk[keep] - NBINS
    hgt = g[pk] if pk.size else np.array([])
    order = np.argsort(-hgt)
    return {
        "n_lobes": int(pk.size),
        "lobe_deg": BIN_CENTRES[pk[order]] if pk.size else np.array([]),
        "lobe_height": hgt[order] if pk.size else np.array([]),
        "modulation_range": rng,
    }


def lobe_separations(lobes_deg: np.ndarray) -> np.ndarray:
    """All pairwise separations, folded into [0, 90] degrees (angles are mod 180)."""
    a = np.asarray(lobes_deg, float)
    if a.size < 2:
        return np.array([])
    d = np.abs(a[:, None] - a[None, :])
    d = np.minimum(d % 180.0, 180.0 - (d % 180.0))
    iu = np.triu_indices(a.size, 1)
    return d[iu]




# ---------------------------------------------------------------------------
# Axis-artifact excision
# ---------------------------------------------------------------------------

AXIS_HALF = 1        # bins excised either side of 0 and 90 deg, primary setting
AXIS_HALF_CONTROL = 3


def excise_axes(dens: np.ndarray, half: int = AXIS_HALF) -> np.ndarray:
    """Remove the exact-0/90-degree bins and interpolate across them.

    The discrete frequency plane puts the whole kx = 0 column and ky = 0 row
    into ONE angular bin each, and their immediate neighbours get fewer pixels.
    The result is a one-bin spike at exactly 0 and 90 degrees in every volume of
    every family, with a dip either side -- a binning artifact, not texture.
    Ring/stripe artifacts and the `_aligned` resampling land in the same two
    bins.  Excising 2*half+1 bins there and interpolating removes both.  A real
    tow lobe is 8-15 degrees wide (measured below), so it survives intact.
    """
    d = np.atleast_2d(np.asarray(dens, float)).copy()
    bad = np.zeros(NBINS, bool)
    for c in (0, NBINS // 2):
        for k in range(-half, half + 1):
            bad[(c + k) % NBINS] = True
    idx = np.arange(NBINS)
    good = ~bad
    xg = np.concatenate([idx[good] - NBINS, idx[good], idx[good] + NBINS])
    for i in range(d.shape[0]):
        yg = np.concatenate([d[i, good]] * 3)
        d[i, bad] = np.interp(idx[bad], xg, yg)
    return d


def normalise(dens: np.ndarray) -> np.ndarray:
    d = np.atleast_2d(np.asarray(dens, float))
    return d / np.maximum(d.mean(axis=1, keepdims=True), 1e-30)


def ply_blocks(dens: np.ndarray, pitch: float = PLY_PITCH) -> np.ndarray:
    """Average the normalised angular density over non-overlapping ply blocks.

    One z-slice is a noisy estimate; a whole volume averages the plies together
    and hides any alternation.  A 19.6-voxel block is one ply, which is the
    scale at which the layup is defined, and gives about 4.4x the SNR of a
    single slice.
    """
    d = normalise(dens)
    nb = int(len(d) // pitch)
    if nb < 2:
        return np.empty((0, NBINS))
    return np.array([d[int(round(i * pitch)): int(round((i + 1) * pitch))].mean(0)
                     for i in range(nb)])


def dominant_direction(dens: np.ndarray) -> float:
    """Volume-level dominant texture direction, in [0, 180) degrees."""
    fm = fourier_modes(excise_axes(dens, AXIS_HALF_CONTROL))
    return float(np.degrees(np.angle(
        np.mean(np.exp(2j * np.deg2rad(fm["phi2"]))))) / 2.0 % 180.0)


def axis_distance(theta_deg: float) -> float:
    """Distance of the lobe pair (theta, theta+90) to the image axes 0 / 90."""
    t = theta_deg % 90.0
    return float(min(t, 90.0 - t))


# ---------------------------------------------------------------------------
# Per-volume analysis
# ---------------------------------------------------------------------------

def interior_mask(fg_frac: np.ndarray, min_fg: float = 0.9) -> np.ndarray:
    """Identical to T-G, so the two tests use the same slices."""
    ok = fg_frac >= min_fg
    if ok.sum() < 32:
        ok = fg_frac >= max(0.5, np.nanpercentile(fg_frac, 60))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ok
    keep = np.zeros_like(ok)
    keep[idx.min(): idx.max() + 1] = True
    return keep & ok


def lobe_stats(rows: np.ndarray) -> dict:
    """Peak count, lobe angles and pairwise separations over a set of rows."""
    counts, angles, seps = [], [], []
    for r in rows:
        lb = count_lobes(r)
        counts.append(lb["n_lobes"])
        angles.append(lb["lobe_deg"])
        s = lobe_separations(lb["lobe_deg"])
        if s.size:
            seps.append(s)
    counts = np.asarray(counts)
    angles = np.concatenate(angles) if angles else np.array([])
    seps = np.concatenate(seps) if seps else np.array([])
    return {"counts": counts, "angles": angles, "seps": seps}


def summarise_lobes(ls: dict) -> dict:
    c, s, a = ls["counts"], ls["seps"], ls["angles"]
    d0 = np.minimum(a % 90.0, 90.0 - (a % 90.0)) if a.size else np.array([])
    return {
        "n_rows": int(c.size),
        "lobe_count_hist": np.bincount(c, minlength=6)[:6],
        "frac_1_lobe": float(np.mean(c == 1)) if c.size else None,
        "frac_2_lobes": float(np.mean(c == 2)) if c.size else None,
        "frac_ge3_lobes": float(np.mean(c >= 3)) if c.size else None,
        "median_n_lobes": float(np.median(c)) if c.size else None,
        "n_separations": int(s.size),
        "sep_median_deg": float(np.median(s)) if s.size else None,
        "frac_sep_near_90": float(np.mean(np.abs(s - 90) < 15)) if s.size else None,
        "frac_sep_near_45": float(np.mean(np.abs(s - 45) < 15)) if s.size else None,
        "n_lobes_total": int(a.size),
        "frac_lobes_within_3deg_of_0_or_90": float(np.mean(d0 <= 3.0)) if d0.size else None,
    }


def peak_aligned_profile(rows: np.ndarray) -> np.ndarray:
    """Mean angular density with every row rotated so its own peak sits at 0.

    This is the artifact-immune way to ask "is there a second lobe, and where?":
    it does not care what the absolute lobe direction is, so it pools volumes
    whose layup is rotated differently.
    """
    if len(rows) == 0:
        return np.full(NBINS, np.nan)
    out = []
    for r in rows:
        rs = _smooth_circ(r)
        out.append(np.roll(rs, -int(np.argmax(rs))))
    return np.mean(out, axis=0)


def mode_power_spectrum(dens: np.ndarray, m_max: int = 88) -> tuple[np.ndarray, np.ndarray]:
    """a_m^2 for every even mode m, per row -- the angular power spectrum.

    Used to decide how many modes a conditioning descriptor needs.  The high-m
    tail is per-bin measurement noise and is subtracted as a floor.
    """
    g = normalise(dens) - 1.0
    ms = np.arange(2, m_max + 1, 2)
    p = np.array([2.0 * np.abs((g @ np.exp(-1j * m * _T)) / NBINS) ** 2 for m in ms])
    return ms, p.mean(axis=1)


def analyse_volume(rec: dict, fg_frac: np.ndarray, tg_c2: np.ndarray,
                   tg_s2: np.ndarray) -> dict:
    keep = interior_mask(fg_frac)
    out = {"volume_id": rec["volume_id"], "shape": rec["shape"],
           "n_slices_total": int(len(fg_frac)),
           "n_slices_interior": int(keep.sum()),
           "seconds": rec.get("seconds")}

    raw = {b: to_density(rec["hists"][b][keep], rec["bin_counts"][b]) for b in BANDS}
    dens = {b: normalise(excise_axes(raw[b])) for b in BANDS}

    per_band = {}
    for b in BANDS:
        fm = fourier_modes(dens[b])
        ls = lobe_stats(dens[b])
        per_band[b] = {
            "n_slices": int(dens[b].shape[0]),
            "a2_median": float(np.median(fm["a2"])),
            "a4_median": float(np.median(fm["a4"])),
            "a6_median": float(np.median(fm["a6"])),
            "ratio_a4_over_a2_median": float(np.median(fm["a4"] / np.maximum(fm["a2"], 1e-9))),
            "frac_slices_a4_gt_a2": float(np.mean(fm["a4"] > fm["a2"])),
            "frac_slices_isotropic": float(np.mean((fm["a2"] < MIN_MODULATION)
                                                   & (fm["a4"] < MIN_MODULATION))),
            **summarise_lobes(ls),
        }
    out["bands"] = per_band

    # ---- primary band, full per-slice detail ---------------------------
    d = dens[PRIMARY_BAND]
    fm = fourier_modes(d)
    ls = lobe_stats(d)
    theta_d = dominant_direction(raw[PRIMARY_BAND])
    out["dominant_direction_deg"] = theta_d
    out["axis_distance_deg"] = axis_distance(theta_d)
    out["primary"] = {"a2": fm["a2"], "a4": fm["a4"], "a6": fm["a6"],
                      "phi2": fm["phi2"], "phi4": fm["phi4"],
                      "n_lobes": ls["counts"], "z_interior": np.flatnonzero(keep)}

    # phase alignment: for a single lobe phi4 == phi2 (mod 90 deg, the period of
    # phi4).  This rules a single lobe IN, not out: two lobes exactly 90 deg
    # apart satisfy it too, because C2 ~ (w1-w2) e^{2i.theta1} and
    # C4 ~ (w1+w2) e^{4i.theta1} share the same theta1.  It separates a single
    # lobe from a 45-degree lobe pair, which is the Pegaso / Juan_Ignacio case.
    dphi = np.abs(((fm["phi4"] - fm["phi2"]) + 45.0) % 90.0 - 45.0)
    strong = (fm["a2"] >= MIN_MODULATION) & (fm["a4"] >= MIN_MODULATION)
    out["phase_alignment"] = {
        "median_abs_dphi_deg": float(np.median(dphi[strong])) if strong.any() else None,
        "frac_aligned_within_10deg": float(np.mean(dphi[strong] < 10.0)) if strong.any() else None,
        "n_slices": int(strong.sum()),
    }

    # ---- ply-block level (the layup scale) ------------------------------
    blocks = ply_blocks(raw[PRIMARY_BAND])
    blocks = normalise(excise_axes(blocks)) if len(blocks) else blocks
    if len(blocks) >= 3:
        bfm = fourier_modes(blocks)
        bls = lobe_stats(blocks)
        sm = _smooth_circ(blocks)
        th = int(round(theta_d)) % 180
        h0 = sm[:, th]
        h45 = sm[:, (th + 45) % 180]
        h90 = sm[:, (th + 90) % 180]
        out["ply_level"] = {
            "n_ply_blocks": int(len(blocks)),
            "a2_median": float(np.median(bfm["a2"])),
            "a4_median": float(np.median(bfm["a4"])),
            "ratio_a4_over_a2_median": float(np.median(bfm["a4"] / np.maximum(bfm["a2"], 1e-9))),
            **summarise_lobes(bls),
            # Does the +90 direction trade power with the dominant one, ply by
            # ply?  A geometric side lobe of the SAME tows would co-vary
            # positively.  A second ply direction trades power: negative.
            # h(+45) is the control for the correlation the normalisation
            # itself induces.
            "corr_h0_vs_h90": float(np.corrcoef(h0, h90)[0, 1]) if len(h0) > 4 else None,
            "corr_h0_vs_h45": float(np.corrcoef(h0, h45)[0, 1]) if len(h0) > 4 else None,
            "level_at_dominant": float(np.mean(h0)),
            "level_at_plus45": float(np.mean(h45)),
            "level_at_plus90": float(np.mean(h90)),
        }
    else:
        out["ply_level"] = {"n_ply_blocks": int(len(blocks))}

    # ---- T-G cross-check: |C2| must reproduce the T-G anisotropy --------
    tg_a = np.hypot(np.asarray(tg_c2, float)[keep], np.asarray(tg_s2, float)[keep])
    fm_raw = fourier_modes(normalise(raw[PRIMARY_BAND]))
    good = np.isfinite(tg_a)
    if good.sum() > 32:
        mine = 0.5 * fm_raw["a2"]
        out["tg_cross_check"] = {
            "corr_abs_C2_vs_TG_anisotropy": float(np.corrcoef(mine[good], tg_a[good])[0, 1]),
            "median_ratio": float(np.median(mine[good] / np.maximum(tg_a[good], 1e-9))),
        }

    # ---- pore-mask cross-check -----------------------------------------
    mh = rec["mask_hist"][keep]
    ok = mh.sum(axis=1) > 0
    if ok.sum() > 32:
        mdens = normalise(excise_axes(
            to_density(mh[ok], rec["bin_counts"][PRIMARY_BAND])))
        mfm = fourier_modes(mdens)
        mls = lobe_stats(mdens)
        out["mask_cross_check"] = {
            "n_slices": int(ok.sum()),
            "a2_median": float(np.median(mfm["a2"])),
            "a4_median": float(np.median(mfm["a4"])),
            "ratio_a4_over_a2_median": float(np.median(mfm["a4"] / np.maximum(mfm["a2"], 1e-9))),
            "frac_2_lobes": float(np.mean(mls["counts"] == 2)),
            "median_abs_dphi4_vs_xct_deg": float(np.median(np.abs(
                ((mfm["phi4"] - fm["phi4"][ok]) + 45.0) % 90.0 - 45.0))),
        }

    # ---- depth structure ------------------------------------------------
    out["depth"] = depth_structure(fm)

    # ---- axis / artifact control ----------------------------------------
    ctrl = fourier_modes(normalise(excise_axes(raw[PRIMARY_BAND], AXIS_HALF_CONTROL)))
    out["axis_control"] = {
        "a2_median_no_excision": float(np.median(fm_raw["a2"])),
        "a2_median_excise_1deg": float(np.median(fm["a2"])),
        "a2_median_excise_3deg": float(np.median(ctrl["a2"])),
        "a4_median_no_excision": float(np.median(fm_raw["a4"])),
        "a4_median_excise_1deg": float(np.median(fm["a4"])),
        "a4_median_excise_3deg": float(np.median(ctrl["a4"])),
        "axis_spike_ratio": float(np.median(normalise(raw[PRIMARY_BAND])[:, 0])),
        "frac_lobes_within_3deg_of_0_or_90": per_band[PRIMARY_BAND][
            "frac_lobes_within_3deg_of_0_or_90"],
    }
    return out


def depth_structure(fm: dict) -> dict:
    """How the 2-theta and 4-theta phasors vary with z.

    The key test for the conditioning design: the 4-theta phasor
    w4 = a4 exp(4 i phi4) may carry ply structure where the 2-theta phasor
    w2 = a2 exp(2 i phi2) cancels.  A 45-degree turn between plies rotates the
    4-theta phase by 180 degrees; a 90-degree turn leaves it unchanged and
    flips w2 instead.  The two phasors are therefore sensitive to different
    halves of a 45-degree-family layup.
    """
    out = {}
    for m in (2, 4):
        w = fm[f"C{m}"]
        n = len(w)
        if n < 60:
            out[f"m{m}"] = {"ok": False}
            continue
        ac = autocorrelation(np.real(w)) + autocorrelation(np.imag(w))
        ac = ac / max(ac[0], 1e-30)
        per, pc = periodogram(np.real(w))
        _, ps = periodogram(np.imag(w))
        pw = pc + ps
        band = (per >= 5.0) & (per <= 150.0) & np.isfinite(per)
        win = max(5, (int(band.sum()) // 8) * 2 + 1)
        bg = ndimage.median_filter(pw, size=win, mode="nearest")
        exc = pw / np.maximum(bg, 1e-30)
        k = int(np.argmax(np.where(band, exc, -np.inf)))
        entry = {"ok": True,
                 "peak_period_voxels": float(per[k]),
                 "peak_excess": float(exc[k]),
                 "amp_median": float(np.median(np.abs(w)) * 2.0),
                 "amp_cv_over_z": float(np.std(np.abs(w)) / max(np.mean(np.abs(w)), 1e-30)),
                 "constant_over_fluctuating":
                     float(np.abs(w.mean()) / max(np.abs(w - w.mean()).mean(), 1e-30)),
                 "circ_dispersion":
                     float(1.0 - np.abs(w.mean()) / max(np.abs(w).mean(), 1e-30))}
        for k2 in (1, 2, 3, 4):
            L = int(round(k2 * PLY_PITCH))
            entry[f"autocorr_at_{k2}x_pitch"] = float(ac[L]) if len(ac) > L else None
        lo, hi = 6, min(len(ac) - 1, 120)
        entry["autocorr_min_lag"] = int(lo + np.argmin(ac[lo:hi + 1]))
        entry["autocorr_min_value"] = float(np.min(ac[lo:hi + 1]))
        entry["autocorr"] = ac[:min(len(ac), 121)]
        out[f"m{m}"] = entry
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_example_hists(blocks_by_family, examples, analyses) -> list[str]:
    fig, axs = plt.subplots(3, 3, figsize=(11.0, 8.4))
    for r, fam in enumerate(FAM_ORDER):
        vid = examples[fam]
        bl = blocks_by_family[fam]["by_volume"][vid]
        col = FAM_COLORS[fam]
        fm = fourier_modes(bl)
        picks = [("strongest $a_4$ ply", int(np.argmax(fm["a4"]))),
                 ("strongest $a_2$ ply", int(np.argmax(fm["a2"]))),
                 ("median ply", int(np.argsort(fm["a2"])[len(fm["a2"]) // 2]))]
        for c, (lab, i) in enumerate(picks):
            ax = axs[r, c]
            ax.plot(BIN_CENTRES, bl[i], color="0.78", lw=0.8)
            ax.plot(BIN_CENTRES, _smooth_circ(bl[i]), color=col, lw=1.7)
            lb = count_lobes(bl[i])
            for a in lb["lobe_deg"]:
                ax.axvline(a, color="k", ls=":", lw=0.9)
            ax.axhline(1.0, color="k", lw=0.5, alpha=0.5)
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title(f"{lab}: $a_2$={fm['a2'][i]:.3f}  $a_4$={fm['a4'][i]:.3f}  "
                         f"lobes={lb['n_lobes']}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{fam}\npower / mean", fontsize=9, color=col)
            if r == 2:
                ax.set_xlabel("texture angle (deg)")
    fig.suptitle("T-H — angular power density of one ply block (19.6 voxels), tow band 16-64 voxels\n"
                 "grey = raw 1-deg bins, colour = smoothed, dotted = detected lobes; "
                 "the 0/90-deg FFT-axis bins are excised", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return savefig(fig, OUT_DIR, "TH_fig1_example_angular_histograms")


def fig_scatter(analyses, families) -> list[str]:
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axs[0]
    for fam in FAM_ORDER:
        xs = [a["ply_level"]["a2_median"] for v, a in analyses.items()
              if families[v] == fam and "a2_median" in a["ply_level"]]
        ys = [a["ply_level"]["a4_median"] for v, a in analyses.items()
              if families[v] == fam and "a4_median" in a["ply_level"]]
        ax.scatter(xs, ys, s=28, color=FAM_COLORS[fam], alpha=0.8,
                   label=f"{fam} (n={len(xs)})")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls="--")
    ax.text(lim * 0.6, lim * 0.93, "$a_4=a_2$", fontsize=8.5)
    for k, lab in ((0.5, "$a_4=a_2/2$"),):
        ax.plot([0, lim], [0, k * lim], color="0.5", lw=0.8, ls=":")
        ax.text(lim * 0.72, k * lim * 0.86, lab, fontsize=8, color="0.4")
    ax.set_xlabel("median $a_2$ per volume (2$\\theta$ amplitude)")
    ax.set_ylabel("median $a_4$ per volume (4$\\theta$ amplitude)")
    ax.set_title("Angular-mode amplitudes, ply blocks")
    ax.legend(fontsize=7.5)

    ax = axs[1]
    for fam in FAM_ORDER:
        d = [a["ply_level"]["ratio_a4_over_a2_median"] for v, a in analyses.items()
             if families[v] == fam and "ratio_a4_over_a2_median" in a["ply_level"]]
        ax.hist(d, bins=np.linspace(0, 1.6, 25), alpha=0.7, color=FAM_COLORS[fam],
                label=f"{fam} (n={len(d)})")
    ax.axvline(1.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("$a_4/a_2$ per volume")
    ax.set_ylabel("volumes")
    ax.set_title("Two-fold vs one-fold balance")
    ax.legend(fontsize=7.5)
    fig.suptitle("T-H — 2$\\theta$ vs 4$\\theta$ by specimen family", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return savefig(fig, OUT_DIR, "TH_fig2_a2_vs_a4_scatter")


def fig_lobes(per_family) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    w = 0.26
    xs = np.arange(5)
    for i, fam in enumerate(FAM_ORDER):
        s = per_family[fam]["ply_level"]
        h = np.asarray(s["lobe_count_hist"], float)[:5]
        h = h / max(h.sum(), 1)
        axs[0].bar(xs + (i - 1) * w, h, width=w, color=FAM_COLORS[fam],
                   label=f"{fam} ({s['n_rows']} plies)")
        e = np.asarray(s["sep_hist_edges"], float)
        hh = np.asarray(s["sep_hist_counts"], float)
        axs[1].step(0.5 * (e[:-1] + e[1:]), hh / max(hh.sum(), 1), where="mid",
                    color=FAM_COLORS[fam], label=f"{fam} ({s['n_separations']} pairs)")
        p = np.asarray(per_family[fam]["peak_aligned_ply_profile"], float)
        axs[2].plot(np.arange(NBINS), p, color=FAM_COLORS[fam], label=fam)
    axs[0].set_xticks(xs)
    axs[0].set_xlabel("lobes detected in one ply block")
    axs[0].set_ylabel("fraction of ply blocks")
    axs[0].set_title("Lobe count per ply")
    axs[0].legend(fontsize=7)
    for a in (45, 90):
        axs[1].axvline(a, color="k", ls=":", lw=0.9)
    axs[1].set_xlabel("angular separation between lobes (deg)")
    axs[1].set_ylabel("fraction of lobe pairs")
    axs[1].set_title("Lobe separation")
    axs[1].legend(fontsize=7)
    for a in (45, 90, 135):
        axs[2].axvline(a, color="k", ls=":", lw=0.8)
    axs[2].axhline(1.0, color="k", lw=0.5)
    axs[2].set_ylim(0.4, 1.8)
    axs[2].set_xticks([0, 45, 90, 135, 180])
    axs[2].set_xlabel("angle from the ply's own dominant direction (deg)")
    axs[2].set_ylabel("power / mean")
    axs[2].set_title("Peak-aligned mean ply density\n(y clipped; the peak reaches 2.1-3.8)")
    axs[2].legend(fontsize=7)
    fig.suptitle("T-H — lobe structure by specimen family, ply blocks of 19.6 voxels", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, OUT_DIR, "TH_fig3_lobe_counts")


def fig_depth(analyses, examples) -> list[str]:
    fig, axs = plt.subplots(3, 2, figsize=(11.0, 8.8))
    for r, fam in enumerate(FAM_ORDER):
        vid = examples[fam]
        p = analyses[vid]["primary"]
        z = p["z_interior"]
        col = FAM_COLORS[fam]

        ax = axs[r, 0]
        ax.plot(z, p["a2"], color="#1b6ca8", lw=1.0, label="$a_2$")
        ax.plot(z, p["a4"], color="#c2571a", lw=1.0, label="$a_4$")
        ax.set_ylabel("amplitude")
        ax.set_title(f"{fam} — {vid.split('__')[1][:50]}", fontsize=8.5, color=col)
        ax.legend(fontsize=7.5)
        if r == 2:
            ax.set_xlabel("z (voxels)")

        ax = axs[r, 1]
        ax.scatter(z, p["phi2"], s=7, c="#1b6ca8", label="$\\varphi_2$ (mod 180)")
        ax.scatter(z, p["phi4"], s=7, c="#c2571a", label="$\\varphi_4$ (mod 90)")
        for k in range(0, int((z[-1] - z[0]) / PLY_PITCH) + 1):
            ax.axvline(z[0] + k * PLY_PITCH, color="0.75", lw=0.6, ls="--")
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_ylabel("phase (deg)")
        ax.set_title("phases vs depth (grey = 19.6-voxel ply grid)", fontsize=8.5)
        ax.legend(fontsize=7)
        if r == 2:
            ax.set_xlabel("z (voxels)")
    fig.suptitle("T-H — 2$\\theta$ and 4$\\theta$ amplitude and phase vs depth, "
                 "representative volume per family", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return savefig(fig, OUT_DIR, "TH_fig4_modes_vs_depth")


def fig_artifact(per_family, nacho_subsets, mode_spec) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ax = axs[0]
    for fam in FAM_ORDER:
        p = np.asarray(per_family[fam]["raw_mean_density"], float)
        ax.plot(np.arange(-8, 9), np.roll(p, 8)[:17], marker="o", ms=3,
                color=FAM_COLORS[fam], label=fam)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("angle from the 0-deg image axis (deg)")
    ax.set_ylabel("power / mean, volume-pooled")
    ax.set_title("The FFT-axis spike: one bin wide,\nwith a dip either side (all families)")
    ax.legend(fontsize=7)

    ax = axs[1]
    for lab, col in (("on_axis", "#999999"), ("off_axis", "#c2571a")):
        s = nacho_subsets[lab]
        ax.plot(np.arange(NBINS), np.asarray(s["peak_aligned_ply_profile"], float),
                color=col, label=f"Nacho {lab.replace('_', '-')} "
                                 f"({s['n_volumes']} vols, {s['n_ply_blocks']} plies)")
    for a in (45, 90, 135):
        ax.axvline(a, color="k", ls=":", lw=0.8)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_ylim(0.4, 1.6)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("angle from the ply's own dominant direction (deg)")
    ax.set_ylabel("power / mean")
    ax.set_title("Nacho second lobe survives when BOTH\nlobes are off the image axes")
    ax.legend(fontsize=7)

    ax = axs[2]
    for fam in FAM_ORDER:
        ms = np.asarray(mode_spec[fam]["m"], float)
        p = np.asarray(mode_spec[fam]["power"], float)
        ax.semilogy(ms, p, color=FAM_COLORS[fam], label=fam)
        ax.axhline(mode_spec[fam]["noise_floor"], color=FAM_COLORS[fam], ls=":", lw=0.8)
    ax.set_xlabel("angular mode $m$")
    ax.set_ylabel("$a_m^2$ (mean over slices)")
    ax.set_title("Angular mode spectrum\n(dotted = per-bin noise floor)")
    ax.legend(fontsize=7)
    fig.suptitle("T-H — artifact control and how many angular modes carry signal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, OUT_DIR, "TH_fig5_artifact_control")


def fig_slices(examples, analyses) -> list[str]:
    """Raw in-plane crops, so the layup can be judged by eye."""
    fig, axs = plt.subplots(2, 3, figsize=(11.4, 8.0))
    for c, fam in enumerate(FAM_ORDER):
        vid = examples[fam]
        g = zarr.open_group(str(ZARR_ROOT), mode="r")[vid]["xct"]
        D, H, W = g.shape
        z = D // 2
        n = 512
        img = np.asarray(g[z, H // 2 - n // 2: H // 2 + n // 2,
                            W // 2 - n // 2: W // 2 + n // 2]).astype(np.float32)
        lo, hi = np.percentile(img, (1, 99))
        for r, (a, b) in enumerate(((0, n), (n // 4, n // 4 + 192))):
            ax = axs[r, c]
            ax.imshow(img[a:b, a:b], cmap="gray", vmin=lo, vmax=hi, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"{fam}\nz={z}, {b - a} x {b - a} voxels\n"
                             f"dominant dir {analyses[vid]['dominant_direction_deg']:.0f} deg",
                             fontsize=8.5, color=FAM_COLORS[fam])
            else:
                ax.set_xlabel(f"zoom, {b - a} x {b - a} voxels", fontsize=8.5)
    fig.suptitle("T-H — raw XCT in-plane slices, one representative volume per family\n"
                 "(x to the right = 0 deg, y up = 90 deg, the same frame as every angle "
                 "in this report)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.0)
    return savefig(fig, OUT_DIR, "TH_fig6_raw_slices")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def family_of(vid: str) -> str:
    tail = vid.split("__", 1)[1]
    return tail.split("_probetas")[0].split("_Probetas")[0]


def pool_family(vids, records, fg) -> dict:
    """Pool the ply blocks and slices of a set of volumes."""
    blocks, slices, by_volume = [], [], {}
    raw_rows = []
    for v in vids:
        keep = interior_mask(fg[v])
        raw = to_density(records[v]["hists"][PRIMARY_BAND][keep],
                         records[v]["bin_counts"][PRIMARY_BAND])
        raw_rows.append(normalise(raw).mean(axis=0))
        d = normalise(excise_axes(raw))
        slices.append(d)
        b = ply_blocks(raw)
        if len(b):
            b = normalise(excise_axes(b))
            blocks.append(b)
            by_volume[v] = b
    return {"blocks": np.concatenate(blocks) if blocks else np.empty((0, NBINS)),
            "slices": np.concatenate(slices) if slices else np.empty((0, NBINS)),
            "raw_mean_density": np.mean(raw_rows, axis=0) if raw_rows else None,
            "by_volume": by_volume, "n_volumes": len(vids)}


def summarise_pool(pool: dict) -> dict:
    out = {"n_volumes": pool["n_volumes"]}
    for lab, rows in (("slice_level", pool["slices"]), ("ply_level", pool["blocks"])):
        if len(rows) == 0:
            out[lab] = {}
            continue
        fm = fourier_modes(rows)
        ls = lobe_stats(rows)
        sh, se = np.histogram(ls["seps"], bins=np.linspace(0, 90, 31))
        out[lab] = {
            "a2_p10_p50_p90": [float(np.percentile(fm["a2"], q)) for q in (10, 50, 90)],
            "a4_p10_p50_p90": [float(np.percentile(fm["a4"], q)) for q in (10, 50, 90)],
            "a6_median": float(np.median(fm["a6"])),
            "ratio_a4_over_a2_median": float(np.median(fm["a4"] / np.maximum(fm["a2"], 1e-9))),
            "frac_a4_gt_a2": float(np.mean(fm["a4"] > fm["a2"])),
            "sep_hist_counts": sh, "sep_hist_edges": se,
            **summarise_lobes(ls),
        }
    out["peak_aligned_ply_profile"] = peak_aligned_profile(pool["blocks"])
    out["n_ply_blocks"] = int(len(pool["blocks"]))
    out["raw_mean_density"] = pool["raw_mean_density"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N volumes and print the projected full runtime")
    args = ap.parse_args()

    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tg = np.load(TG_CACHE, allow_pickle=True)["records"].item()
    vids = sorted(tg.keys())
    fg = {v: np.asarray(tg[v]["fg_frac"], float) for v in vids}
    families = {v: family_of(v) for v in vids}
    run_vids = vids[: args.smoke] if args.smoke else vids

    print(f"[T-H] {len(run_vids)} volumes, window {WINDOW}, {NBINS} angular bins, "
          f"{args.workers} workers", flush=True)

    cache = OUT_DIR / "angular_hists.npz"
    if cache.exists() and not args.smoke:
        print(f"[T-H] loading cached angular histograms from {cache}", flush=True)
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
                      f"D={r['shape'][0]:3d} {r['seconds']:6.1f}s "
                      f"elapsed={time.time() - t0:7.1f}s", flush=True)
        wall = time.time() - t0
        per_vol = float(np.mean([r["seconds"] for r in records.values()]))
        print(f"[T-H] {len(records)} volumes in {wall:.1f}s ({per_vol:.1f}s cpu/volume)",
              flush=True)
        if args.smoke:
            proj = per_vol * len(vids) / args.workers
            print(f"[T-H] SMOKE: projected full run over {len(vids)} volumes with "
                  f"{args.workers} workers = {proj / 60:.1f} min", flush=True)
            return
        np.savez_compressed(cache, records=np.array(records, dtype=object))

    print("[T-H] angular-shape analysis ...", flush=True)
    analyses = {v: analyse_volume(records[v], fg[v],
                                  tg[v]["c2"][PRIMARY_BAND], tg[v]["s2"][PRIMARY_BAND])
                for v in records}

    pools = {fam: pool_family([v for v in records if families[v] == fam], records, fg)
             for fam in FAM_ORDER}
    per_family = {fam: summarise_pool(p) for fam, p in pools.items()}

    # per-volume ply-level correlation between the dominant and the +90 direction
    for fam in FAM_ORDER:
        sub = [v for v in records if families[v] == fam and "corr_h0_vs_h90" in analyses[v]["ply_level"]]
        c90 = [analyses[v]["ply_level"]["corr_h0_vs_h90"] for v in sub]
        c45 = [analyses[v]["ply_level"]["corr_h0_vs_h45"] for v in sub]
        per_family[fam]["orthogonal_lobe_test"] = {
            "n_volumes": len(sub),
            "median_corr_dominant_vs_plus90": float(np.median(c90)),
            "frac_negative_corr_plus90": float(np.mean(np.array(c90) < 0)),
            "median_corr_dominant_vs_plus45_control": float(np.median(c45)),
            "median_level_at_dominant": float(np.median(
                [analyses[v]["ply_level"]["level_at_dominant"] for v in sub])),
            "median_level_at_plus45": float(np.median(
                [analyses[v]["ply_level"]["level_at_plus45"] for v in sub])),
            "median_level_at_plus90": float(np.median(
                [analyses[v]["ply_level"]["level_at_plus90"] for v in sub])),
        }
        for m in ("m2", "m4"):
            e = [analyses[v]["depth"][m] for v in records
                 if families[v] == fam and analyses[v]["depth"][m].get("ok")]
            per_family[fam][f"depth_{m}"] = {
                "n_volumes": len(e),
                **{k: float(np.nanmedian([x.get(k) if x.get(k) is not None else np.nan
                                          for x in e]))
                   for k in ("autocorr_at_1x_pitch", "autocorr_at_2x_pitch",
                             "autocorr_at_3x_pitch", "autocorr_at_4x_pitch",
                             "peak_period_voxels", "amp_median", "amp_cv_over_z",
                             "constant_over_fluctuating", "circ_dispersion")},
            }
        per_family[fam]["mask_cross_check"] = {
            k: float(np.nanmedian([analyses[v].get("mask_cross_check", {}).get(k, np.nan)
                                   for v in records if families[v] == fam]))
            for k in ("a2_median", "a4_median", "ratio_a4_over_a2_median",
                      "frac_2_lobes", "median_abs_dphi4_vs_xct_deg")}
        per_family[fam]["axis_control"] = {
            k: float(np.nanmedian([analyses[v]["axis_control"][k] for v in records
                                   if families[v] == fam]))
            for k in ("a2_median_no_excision", "a2_median_excise_1deg",
                      "a2_median_excise_3deg", "a4_median_no_excision",
                      "a4_median_excise_1deg", "a4_median_excise_3deg",
                      "axis_spike_ratio", "frac_lobes_within_3deg_of_0_or_90")}
        per_family[fam]["dominant_directions_deg"] = sorted(
            analyses[v]["dominant_direction_deg"] for v in records if families[v] == fam)

    # ---- Nacho on-axis / off-axis artifact control ----------------------
    nacho = [v for v in records if families[v] == "Fabricacion_Nacho_05"]
    subsets = {"on_axis": [v for v in nacho if analyses[v]["axis_distance_deg"] < 10.0],
               "off_axis": [v for v in nacho if analyses[v]["axis_distance_deg"] >= 10.0]}
    nacho_subsets = {}
    for lab, vs in subsets.items():
        p = pool_family(vs, records, fg)
        s = summarise_pool(p)
        prof = np.asarray(s["peak_aligned_ply_profile"], float)
        s["second_lobe_at_90"] = {
            "level_at_90": float(prof[90]),
            "local_background_75_105": float(np.mean(prof[[70, 75, 105, 110]])),
            "relative_excess": float(prof[90] / np.mean(prof[[70, 75, 105, 110]]) - 1.0),
        }
        c90 = [analyses[v]["ply_level"]["corr_h0_vs_h90"] for v in vs
               if "corr_h0_vs_h90" in analyses[v]["ply_level"]]
        c45 = [analyses[v]["ply_level"]["corr_h0_vs_h45"] for v in vs
               if "corr_h0_vs_h45" in analyses[v]["ply_level"]]
        s["orthogonal_lobe_test"] = {
            "median_corr_dominant_vs_plus90": float(np.median(c90)) if c90 else None,
            "frac_negative_corr_plus90": float(np.mean(np.array(c90) < 0)) if c90 else None,
            "median_corr_dominant_vs_plus45_control": float(np.median(c45)) if c45 else None,
        }
        s["volumes"] = vs
        nacho_subsets[lab] = s

    # ---- how many angular modes does a descriptor need? -----------------
    mode_spec = {}
    for fam in FAM_ORDER:
        ms, p = mode_power_spectrum(pools[fam]["slices"])
        floor = float(np.median(p[ms >= 50]))
        sig = np.maximum(p - floor, 0.0)
        tot = float(sig.sum())
        mode_spec[fam] = {
            "m": ms, "power": p, "noise_floor": floor,
            "cumulative_fraction": {f"m<={int(m)}": float(sig[:i + 1].sum() / max(tot, 1e-30))
                                    for i, m in enumerate(ms) if m in (2, 4, 6, 8, 12, 16, 20, 30)},
        }

    tgx = [a["tg_cross_check"] for a in analyses.values() if "tg_cross_check" in a]
    tg_summary = {
        "n_volumes": len(tgx),
        "median_corr_abs_C2_vs_TG_anisotropy": float(np.median(
            [e["corr_abs_C2_vs_TG_anisotropy"] for e in tgx])),
        "median_ratio": float(np.median([e["median_ratio"] for e in tgx])),
    }

    # representative volume per family: the one with the median ply-level a4
    examples = {}
    for fam in FAM_ORDER:
        sub = [v for v in records if families[v] == fam and v in pools[fam]["by_volume"]]
        vals = np.array([analyses[v]["ply_level"]["a4_median"] for v in sub])
        examples[fam] = sub[int(np.argsort(vals)[len(vals) // 2])]

    print("[T-H] figures ...", flush=True)
    figs = {
        "fig1": fig_example_hists(pools, examples, analyses),
        "fig2": fig_scatter(analyses, families),
        "fig3": fig_lobes(per_family),
        "fig4": fig_depth(analyses, examples),
        "fig5": fig_artifact(per_family, nacho_subsets, mode_spec),
        "fig6": fig_slices(examples, analyses),
    }

    results = {
        "test_id": TEST_ID,
        "method": {
            "window_voxels": WINDOW,
            "slices": "every z-slice of all 80 volumes, no subsampling",
            "bands_wavelength_voxels": {k: list(v) for k, v in BANDS.items()},
            "primary_band": PRIMARY_BAND,
            "angular_bins": NBINS,
            "density_correction": ("each angular bin is divided by the number of "
                                   "spectrum pixels in it, so the histogram is a mean "
                                   "power density with no square-grid bias"),
            "axis_excision_halfwidth_deg": AXIS_HALF,
            "axis_excision_control_halfwidth_deg": AXIS_HALF_CONTROL,
            "ply_block_voxels": PLY_PITCH,
            "angle_convention": ("all angles are REAL-SPACE texture angles, mod 180 deg; "
                                 "the spectral histogram is rolled by 90 deg, the same "
                                 "convention as T-G; x to the right = 0 deg, y up = 90 deg"),
            "amplitude_definition": "a_m = 2|C_m|, C_m = <h exp(-i m theta)> / <h>",
            "phase_definition": "phi_m = arg(C_m)/m, in [0, 360/m) degrees",
            "lobe_criterion": {"smoothing_sigma_deg": SMOOTH_SIGMA_DEG,
                               "prominence_abs": PEAK_PROM_ABS,
                               "prominence_rel_to_range": PEAK_PROM_REL},
            "isotropic_threshold": MIN_MODULATION,
            "ply_pitch_voxels_from_T_A": PLY_PITCH,
        },
        "runtime": {"volumes": len(records), "wall_seconds": wall,
                    "cpu_seconds_per_volume": per_vol, "workers": args.workers},
        "per_family": per_family,
        "nacho_axis_subsets": nacho_subsets,
        "angular_mode_spectrum": mode_spec,
        "tg_consistency": tg_summary,
        "example_volumes": examples,
        "figures": figs,
        "per_volume": {v: {
            "family": families[v],
            "shape": a["shape"],
            "n_slices_interior": a["n_slices_interior"],
            "dominant_direction_deg": a["dominant_direction_deg"],
            "axis_distance_deg": a["axis_distance_deg"],
            "bands": a["bands"],
            "ply_level": a["ply_level"],
            "phase_alignment": a["phase_alignment"],
            "tg_cross_check": a.get("tg_cross_check"),
            "mask_cross_check": a.get("mask_cross_check"),
            "axis_control": a["axis_control"],
            "depth": {m: {k: val for k, val in a["depth"][m].items() if k != "autocorr"}
                      for m in a["depth"]},
        } for v, a in analyses.items()},
        "families": families,
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-H] wrote {p}", flush=True)

    # ------------------------------------------------------------------
    print("\n--- T-H summary (tow band 16-64 voxels, ply blocks of 19.6 voxels) ---")
    for fam in FAM_ORDER:
        s = per_family[fam]["ply_level"]
        print(f"{fam:22s} n={per_family[fam]['n_volumes']:3d} plies={s['n_rows']:5d} "
              f"a2={s['a2_p10_p50_p90'][1]:.3f} a4={s['a4_p10_p50_p90'][1]:.3f} "
              f"a4/a2={s['ratio_a4_over_a2_median']:.2f}")
    for fam in FAM_ORDER:
        s = per_family[fam]["ply_level"]
        print(f"{fam:22s} lobes 1/2/3+ = {s['frac_1_lobe']:.2f}/{s['frac_2_lobes']:.2f}/"
              f"{s['frac_ge3_lobes']:.2f}  sep_med={s['sep_median_deg']:.0f} "
              f"near90={s['frac_sep_near_90']:.2f} near45={s['frac_sep_near_45']:.2f}")
    for fam in FAM_ORDER:
        s = per_family[fam]["orthogonal_lobe_test"]
        print(f"{fam:22s} orthogonal-lobe test: corr(dominant, +90)={s['median_corr_dominant_vs_plus90']:+.3f} "
              f"({s['frac_negative_corr_plus90']:.2f} negative), control corr(+45)="
              f"{s['median_corr_dominant_vs_plus45_control']:+.3f}")
    for fam in FAM_ORDER:
        for m in ("m2", "m4"):
            s = per_family[fam][f"depth_{m}"]
            print(f"{fam:22s} depth {m}: ac 1x/2x/4x = {s['autocorr_at_1x_pitch']:+.3f}/"
                  f"{s['autocorr_at_2x_pitch']:+.3f}/{s['autocorr_at_4x_pitch']:+.3f} "
                  f"peak={s['peak_period_voxels']:.1f} const/fluct={s['constant_over_fluctuating']:.2f}")
    for lab, s in nacho_subsets.items():
        print(f"Nacho {lab:9s} n={s['n_volumes']:3d} plies={s['n_ply_blocks']:4d} "
              f"2-lobe={s['ply_level']['frac_2_lobes']:.2f} sep={s['ply_level']['sep_median_deg']:.0f} "
              f"| +90 lobe level {s['second_lobe_at_90']['level_at_90']:.3f} vs local bg "
              f"{s['second_lobe_at_90']['local_background_75_105']:.3f} "
              f"(+{100 * s['second_lobe_at_90']['relative_excess']:.0f}%) "
              f"| corr(dom,+90)={s['orthogonal_lobe_test']['median_corr_dominant_vs_plus90']:+.3f} "
              f"control(+45)={s['orthogonal_lobe_test']['median_corr_dominant_vs_plus45_control']:+.3f}")
    for fam in FAM_ORDER:
        print(f"{fam:22s} modes: " + "  ".join(
            f"{k}:{v:.2f}" for k, v in mode_spec[fam]["cumulative_fraction"].items()))
    print("T-G consistency:", tg_summary)
    print("examples:", examples)


if __name__ == "__main__":
    main()

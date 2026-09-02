"""T-A — Periodicity of porosity and grayscale along z, y and x.

Two independent passes:

1. **Coarse pass** — mean patch porosity vs. absolute patch coordinate, taken
   straight from ``patch_index.parquet``.  Resolution is the patch stride
   (32 voxels), so the Nyquist period is 64 voxels.  This is the pass the
   analysis brief asked for; it is reported, but it cannot resolve a ply
   period shorter than 64 voxels.

2. **Fine pass** — exact per-slice profiles read from ``volumes.zarr``:
   for every volume and every axis, the porosity fraction (labelled pore
   voxels / foreground voxels) and the mean foreground grayscale, at
   1-voxel resolution.  A full-volume mask+xct stream costs ~10 s per volume,
   so this is cheap and gives the resolution the hypothesis needs.

Periodicity is tested with a Hann-windowed periodogram on the polynomial-
detrended profile, against an AR(1) red-noise null (300 surrogates).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AXIS_COLORS, OUT_ROOT, PATCH_INDEX, STRIDE, VOXEL_SIZE_UM, ZARR_ROOT,
    detrend_profile, dominant_period, periodogram, autocorrelation,
    savefig, set_style, write_findings, write_json, plt,
)

TEST_ID = "T-A"
OUT_DIR = OUT_ROOT / TEST_ID
AXES = ("z", "y", "x")
CHUNK_Z = 64


# ---------------------------------------------------------------------------
# Fine pass — per-volume worker
# ---------------------------------------------------------------------------

def _otsu_from_hist(hist: np.ndarray) -> int:
    total = int(hist.sum())
    if total == 0:
        return 128
    p = hist.astype(np.float64) / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(levels * p)
    mu_T = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega)),
            0.0,
        )
    return int(np.nanargmax(sigma_b))


def volume_profiles(volume_id: str) -> dict:
    """Stream one volume and return exact per-slice profiles for all 3 axes."""
    g = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]
    xct_a, mask_a = g["xct"], g["mask"]
    D, H, W = xct_a.shape

    # Pass 1 — histogram for the Otsu foreground threshold.  Chunk-aligned slab
    # reads, sub-sampled in y and x (the threshold does not need every voxel).
    hist = np.zeros(256, dtype=np.int64)
    for z in range(0, D, CHUNK_Z):
        chunk = np.asarray(xct_a[z: min(z + CHUNK_Z, D)])[::2, ::4, ::4]
        hist += np.bincount(chunk.ravel(), minlength=256)
    thr = _otsu_from_hist(hist)

    fg_z = np.zeros(D); pore_z = np.zeros(D); gray_z = np.zeros(D)
    fg_y = np.zeros(H); pore_y = np.zeros(H); gray_y = np.zeros(H)
    fg_x = np.zeros(W); pore_x = np.zeros(W); gray_x = np.zeros(W)

    # Pass 2 — accumulate per-axis sums.  Everything stays in uint8/bool with
    # int64 accumulators, so no large float temporaries are ever allocated.
    for z in range(0, D, CHUNK_Z):
        z1 = min(z + CHUNK_Z, D)
        xct = np.asarray(xct_a[z:z1])
        msk = np.asarray(mask_a[z:z1]).astype(bool)
        fg = xct > thr
        gray = xct * fg                 # uint8, zero outside the specimen
        por = msk & fg                  # pore voxels are labelled inside only

        fg_z[z:z1] += fg.sum(axis=(1, 2), dtype=np.int64)
        pore_z[z:z1] += por.sum(axis=(1, 2), dtype=np.int64)
        gray_z[z:z1] += gray.sum(axis=(1, 2), dtype=np.int64)

        fg_y += fg.sum(axis=(0, 2), dtype=np.int64)
        pore_y += por.sum(axis=(0, 2), dtype=np.int64)
        gray_y += gray.sum(axis=(0, 2), dtype=np.int64)

        fg_x += fg.sum(axis=(0, 1), dtype=np.int64)
        pore_x += por.sum(axis=(0, 1), dtype=np.int64)
        gray_x += gray.sum(axis=(0, 1), dtype=np.int64)

    def _norm(pore, gray, fg):
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.where(fg > 0, pore / np.maximum(fg, 1), np.nan)
            gr = np.where(fg > 0, gray / np.maximum(fg, 1), np.nan)
        return phi, gr, fg

    pz, gz, fz = _norm(pore_z, gray_z, fg_z)
    py, gy, fy = _norm(pore_y, gray_y, fg_y)
    px, gx, fx = _norm(pore_x, gray_x, fg_x)
    return {
        "volume_id": volume_id,
        "shape": [int(D), int(H), int(W)],
        "otsu_threshold": int(thr),
        "z": {"phi": pz, "gray": gz, "fg": fz},
        "y": {"phi": py, "gray": gy, "fg": fy},
        "x": {"phi": px, "gray": gx, "fg": fx},
    }


def _worker(vid: str):
    t0 = time.time()
    out = volume_profiles(vid)
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Periodicity analysis of a set of profiles
# ---------------------------------------------------------------------------

def analyse_profiles(profiles: dict[str, dict], axis: str, key: str,
                     period_min: float, period_max: float,
                     min_fg_frac: float = 0.2) -> dict:
    """Run detrend + periodogram + AR(1) test on one (axis, signal) pair."""
    rng = np.random.default_rng(12345)
    per_volume = {}
    for vid, prof in profiles.items():
        y = np.asarray(prof[axis][key], dtype=np.float64)
        fg = np.asarray(prof[axis]["fg"], dtype=np.float64)
        # keep only the interior where the specimen actually fills the slice
        if fg.max() > 0:
            keep = fg > min_fg_frac * np.nanmax(fg)
        else:
            keep = np.ones_like(fg, dtype=bool)
        idx = np.where(keep)[0]
        if len(idx) < 32:
            per_volume[vid] = {"detected": False, "reason": "too few valid slices"}
            continue
        y = y[idx[0]: idx[-1] + 1]
        y = np.nan_to_num(y, nan=float(np.nanmean(y)))
        d = detrend_profile(y, poly_order=3)
        pmax = min(period_max, len(d) / 3.0)
        if pmax <= period_min:
            per_volume[vid] = {"detected": False, "reason": "axis too short"}
            continue
        res = dominant_period(d, period_min, pmax, n_surrogate=300, rng=rng)
        res["valid_extent_voxels"] = int(len(d))
        per_volume[vid] = res

    detected = [v for v in per_volume.values() if v.get("detected")]
    periods = np.array([v["peak_period_voxels"] for v in detected]) if detected else np.array([])
    all_peaks = np.array([v["peak_period_voxels"] for v in per_volume.values()
                          if v.get("peak_period_voxels") is not None])
    summary = {
        "all_volume_peak_periods": all_peaks.tolist(),
        "all_peak_median_voxels": float(np.median(all_peaks)) if len(all_peaks) else None,
        "axis": axis,
        "signal": key,
        "period_band_voxels": [period_min, period_max],
        "n_volumes": len(per_volume),
        "n_detected": len(detected),
        "detection_rate": len(detected) / max(len(per_volume), 1),
        "detected_period_median_voxels": float(np.median(periods)) if len(periods) else None,
        "detected_period_iqr_voxels": (
            [float(np.percentile(periods, 25)), float(np.percentile(periods, 75))]
            if len(periods) else None
        ),
        "detected_period_mad_voxels": (
            float(np.median(np.abs(periods - np.median(periods)))) if len(periods) else None
        ),
        "detected_periods": periods.tolist(),
        "per_volume": per_volume,
    }
    return summary


def stacked_spectrum(profiles: dict[str, dict], axis: str, key: str,
                     period_grid: np.ndarray, min_fg_frac: float = 0.2) -> np.ndarray:
    """Median of the background-normalised periodograms, on a common period grid."""
    curves = []
    for prof in profiles.values():
        y = np.asarray(prof[axis][key], dtype=np.float64)
        fg = np.asarray(prof[axis]["fg"], dtype=np.float64)
        keep = fg > min_fg_frac * np.nanmax(fg) if fg.max() > 0 else np.ones_like(fg, bool)
        idx = np.where(keep)[0]
        if len(idx) < 32:
            continue
        y = np.nan_to_num(y[idx[0]: idx[-1] + 1], nan=float(np.nanmean(y)))
        d = detrend_profile(y, 3)
        per, pw = periodogram(d)
        ok = np.isfinite(per) & (per > 2)
        per, pw = per[ok], pw[ok]
        if pw.max() <= 0:
            continue
        # Divide by a running median of the spectrum (in frequency order).  This
        # removes the red-noise slope, so a peak means "excess over the local
        # background", not simply "long wavelength".
        from scipy.ndimage import median_filter
        bg = median_filter(pw, size=max(11, len(pw) // 40), mode="nearest")
        pw = pw / np.maximum(bg, 1e-30)
        order = np.argsort(per)
        curves.append(np.interp(period_grid, per[order], pw[order],
                                left=np.nan, right=np.nan))
    if not curves:
        return np.full_like(period_grid, np.nan)
    # median across volumes: a line present in only a few scans cannot create a peak
    return np.nanmedian(np.vstack(curves), axis=0)


# ---------------------------------------------------------------------------
# Coarse pass (patch index)
# ---------------------------------------------------------------------------

def coarse_pass(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(7)
    out = {"resolution_voxels": STRIDE, "nyquist_period_voxels": 2 * STRIDE, "axes": {}}
    for axis in AXES:
        col = f"{axis}0"
        per_vol = {}
        for vid, sub in df.groupby("volume_id", observed=True):
            prof = sub.groupby(col, observed=True)["porosity"].mean().sort_index()
            y = prof.to_numpy(dtype=np.float64)
            if len(y) < 16:
                per_vol[vid] = {"n_positions": int(len(y)), "detected": False,
                                "reason": "fewer than 16 patch positions on this axis"}
                continue
            d = detrend_profile(y, 3)
            res = dominant_period(d, 3.0, len(d) / 3.0, n_surrogate=200, rng=rng)
            # periods are in units of stride -> convert to voxels
            if res.get("peak_period_voxels") is not None:
                res["peak_period_voxels"] = res["peak_period_voxels"] * STRIDE
            res["n_positions"] = int(len(y))
            per_vol[vid] = res
        det = [v for v in per_vol.values() if v.get("detected")]
        out["axes"][axis] = {
            "n_volumes": len(per_vol),
            "n_detected": len(det),
            "detection_rate": len(det) / max(len(per_vol), 1),
            "median_detected_period_voxels": (
                float(np.median([v["peak_period_voxels"] for v in det])) if det else None
            ),
            "per_volume": per_vol,
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_profiles(profiles: dict, out_dir: Path, example_ids: list[str]) -> list[str]:
    """Example profiles. Only the interior is drawn: slices where the specimen
    fills less than 20 % of its maximum cross-section are dropped, because their
    porosity ratio is meaningless (a handful of foreground voxels)."""
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), constrained_layout=True)
    # 50 % threshold here (stricter than the 20 % used for the statistics) so a
    # single part-filled edge slice does not stretch the vertical scale
    ylabs = {"phi": "porosity $\\varphi$  (pore / foreground voxels)",
             "gray": "mean foreground grey level (0-255)"}
    for j, ax_name in enumerate(AXES):
        for i, key in ((0, "phi"), (1, "gray")):
            ax = axes[i][j]
            for vid in example_ids:
                p = profiles[vid]
                y = np.asarray(p[ax_name][key], float)
                fg = np.asarray(p[ax_name]["fg"], float)
                keep = fg > 0.5 * np.nanmax(fg)
                idx = np.where(keep)[0]
                if len(idx) == 0:
                    continue
                sl = slice(idx[0], idx[-1] + 1)
                ax.plot(np.arange(len(y))[sl], y[sl], lw=0.8, alpha=0.85,
                        label=vid.split("probetas_")[-1].replace("_volume_eq_aligned", ""))
            ax.set_xlabel(f"{ax_name} coordinate (voxels)")
            if j == 0:
                ax.set_ylabel(ylabs[key])
            ax.set_title(f"{ax_name}-axis, {'porosity' if key == 'phi' else 'grey level'}")
            if key == "phi":
                ax.set_ylim(bottom=0)
    axes[0][0].set_xlim(0, 120)   # zoom: makes the ply ripple visible
    axes[1][0].set_xlim(0, 120)
    axes[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("T-A  Exact 1-voxel-resolution profiles, four example volumes "
                 "(z panels zoomed to the first 120 voxels)", fontsize=12)
    return savefig(fig, out_dir, "TA_fig1_example_profiles")


def fig_spectra(profiles: dict, out_dir: Path, results: dict) -> list[str]:
    set_style()
    grid = np.linspace(4, 200, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for key, ax, title in (("phi", axes[0], "porosity $\\varphi$"),
                           ("gray", axes[1], "foreground grayscale")):
        for ax_name in AXES:
            s = stacked_spectrum(profiles, ax_name, key, grid)
            ax.plot(grid, s, color=AXIS_COLORS[ax_name], label=f"{ax_name}-axis")
        ax.axhline(1.0, color="k", ls=":", lw=0.9, label="local background level")
        ax.set_xlabel("period (voxels)")
        ax.set_ylabel("median power / local background (dimensionless)")
        ax.set_title(f"Stacked periodogram — {title}")
        ax.set_xscale("log")
        ax.legend(frameon=False)
    fig.suptitle("T-A  Cross-volume median power spectra, background-normalised "
                 "(80 volumes)", fontsize=12)
    return savefig(fig, out_dir, "TA_fig2_stacked_spectra")


def fig_period_hist(results: dict, out_dir: Path) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, key, title in ((axes[0], "phi", "porosity $\\varphi$"),
                           (axes[1], "gray", "foreground grayscale")):
        for ax_name in AXES:
            per = results["fine"][key][ax_name]["detected_periods"]
            if per:
                ax.hist(per, bins=np.linspace(4, 120, 30), alpha=0.6,
                        color=AXIS_COLORS[ax_name],
                        label=f"{ax_name} (n={len(per)}/{results['fine'][key][ax_name]['n_volumes']})")
        ax.set_xlabel("dominant period (voxels)")
        ax.set_ylabel("number of volumes")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("T-A  Dominant period per volume, significant detections only "
                 "(AR(1) null, p < 0.05 and SNR > 3)", fontsize=11)
    return savefig(fig, out_dir, "TA_fig3_period_histogram")


def fig_autocorr(profiles: dict, out_dir: Path) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    for j, ax_name in enumerate(AXES):
        ax = axes[j]
        curves = []
        maxlag = 150
        for prof in profiles.values():
            y = np.asarray(prof[ax_name]["phi"], dtype=np.float64)
            fg = np.asarray(prof[ax_name]["fg"], dtype=np.float64)
            keep = fg > 0.2 * np.nanmax(fg)
            idx = np.where(keep)[0]
            if len(idx) < 2 * maxlag:
                continue
            y = np.nan_to_num(y[idx[0]:idx[-1] + 1], nan=float(np.nanmean(y)))
            d = detrend_profile(y, 3)
            ac = autocorrelation(d, maxlag)
            curves.append(ac)
        if curves:
            C = np.vstack(curves)
            lags = np.arange(C.shape[1])
            ax.plot(lags, C.mean(0), color=AXIS_COLORS[ax_name], lw=1.6)
            ax.fill_between(lags, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                            color=AXIS_COLORS[ax_name], alpha=0.22, lw=0)
            ax.text(0.97, 0.93, f"n={C.shape[0]} volumes", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"lag along {ax_name} (voxels)")
        ax.set_ylabel("autocorrelation of detrended $\\varphi$" if j == 0 else "")
        ax.set_title(f"{ax_name}-axis")
    fig.suptitle("T-A  Autocorrelation of the detrended 1-voxel porosity profile "
                 "(mean and IQR over volumes)", fontsize=11)
    return savefig(fig, out_dir, "TA_fig4_profile_autocorrelation")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-volumes", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PATCH_INDEX)
    df = df[df["porosity"] <= 1.0]
    vids = sorted(df["volume_id"].unique().tolist())
    if args.limit_volumes:
        vids = vids[: args.limit_volumes]
    print(f"[T-A] {len(vids)} volumes, {len(df)} patches", flush=True)

    print("[T-A] coarse pass (patch index, 32-voxel resolution) ...", flush=True)
    coarse = coarse_pass(df)

    cache = OUT_DIR / "fine_profiles.npz"
    if cache.exists():
        print(f"[T-A] loading cached fine profiles from {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        profiles = z["profiles"].item()
    else:
        print(f"[T-A] fine pass: streaming {len(vids)} volumes with {args.workers} workers ...",
              flush=True)
        profiles = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, v): v for v in vids}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                profiles[r["volume_id"]] = r
                print(f"  [{i:3d}/{len(vids)}] {r['volume_id'][-38:]:38s} "
                      f"shape={r['shape']} otsu={r['otsu_threshold']:3d} "
                      f"{r['seconds']:5.1f}s  elapsed={time.time() - t0:6.1f}s", flush=True)
        np.savez_compressed(cache, profiles=np.array(profiles, dtype=object))

    print("[T-A] periodicity tests ...", flush=True)
    fine = {"phi": {}, "gray": {}}
    for key in ("phi", "gray"):
        for ax_name in AXES:
            fine[key][ax_name] = analyse_profiles(profiles, ax_name, key, 6.0, 150.0)
            s = fine[key][ax_name]
            print(f"  {key:5s} {ax_name}: detected {s['n_detected']}/{s['n_volumes']} "
                  f"median period = {s['detected_period_median_voxels']}", flush=True)

    # cross-volume stacked spectrum: one peak per (signal, axis), all 80 volumes
    grid = np.linspace(5, 200, 800)
    for key in ("phi", "gray"):
        for ax_name in AXES:
            s = stacked_spectrum(profiles, ax_name, key, grid)
            ok = np.isfinite(s)
            if ok.sum() < 10:
                continue
            k = int(np.nanargmax(np.where(ok, s, -np.inf)))
            band = s[ok]
            fine[key][ax_name]["stacked_spectrum"] = {
                "period_grid_voxels": grid.tolist(),
                "power": np.where(ok, s, np.nan).tolist(),
                "peak_period_voxels": float(grid[k]),
                "peak_power_over_background": float(s[k]),
                "median_power": float(np.nanmedian(band)),
                "peak_over_median": float(s[k] / max(np.nanmedian(band), 1e-12)),
            }
            print(f"  stacked {key:5s} {ax_name}: peak at {grid[k]:6.1f} vox, "
                  f"power/red-noise = {s[k]:.2f}", flush=True)

    # per-volume excess in a fixed band around each stacked peak, plus the same
    # ply band applied to all three axes so the axes can be compared directly
    from scipy.ndimage import median_filter
    ply_band_ref = fine["phi"]["z"]["stacked_spectrum"]["peak_period_voxels"]
    jobs = [(k, a, None) for k in ("phi", "gray") for a in AXES]
    jobs += [("phi", a, [ply_band_ref * 0.9, ply_band_ref * 1.1]) for a in AXES]
    for key, ax_name, fixed_band in jobs:
            st = fine[key][ax_name].get("stacked_spectrum")
            if not st:
                continue
            p0 = st["peak_period_voxels"]
            band = fixed_band if fixed_band else [p0 * 0.9, p0 * 1.1]
            exc = []
            for prof in profiles.values():
                y = np.asarray(prof[ax_name][key], float)
                fg = np.asarray(prof[ax_name]["fg"], float)
                keep = fg > 0.2 * np.nanmax(fg)
                idx = np.where(keep)[0]
                if len(idx) < 64:
                    continue
                y = np.nan_to_num(y[idx[0]:idx[-1] + 1], nan=float(np.nanmean(y)))
                per, pw = periodogram(detrend_profile(y, 3))
                ok = np.isfinite(per) & (per > 2)
                per, pw = per[ok], pw[ok]
                bg = median_filter(pw, size=max(11, len(pw) // 40), mode="nearest")
                ratio = pw / np.maximum(bg, 1e-30)
                m = (per >= band[0]) & (per <= band[1])
                if m.any():
                    exc.append(float(ratio[m].max()))
            exc = np.array(exc)
            slot = "ply_band_test" if fixed_band else "band_test"
            fine[key][ax_name][slot] = {
                "band_voxels": band,
                "n_volumes": int(len(exc)),
                "median_excess_over_background": float(np.median(exc)) if len(exc) else None,
                "frac_volumes_excess_gt_3": float(np.mean(exc > 3)) if len(exc) else None,
                "frac_volumes_excess_gt_5": float(np.mean(exc > 5)) if len(exc) else None,
                "frac_volumes_excess_gt_10": float(np.mean(exc > 10)) if len(exc) else None,
            }
            bt = fine[key][ax_name][slot]
            print(f"  {slot[:3]} {key:5s} {ax_name}: {band[0]:5.1f}-{band[1]:5.1f} vox  "
                  f"median excess {bt['median_excess_over_background']:6.2f}  "
                  f">3 in {bt['frac_volumes_excess_gt_3']*100:5.1f}% of volumes", flush=True)

    # artifact screen: how many volumes carry a strong narrow spectral line
    # anywhere in 6-100 voxels, per axis (a comb of narrow lines in x or y would
    # indicate a scan or resampling artifact rather than material structure)
    for key in ("phi", "gray"):
        for ax_name in AXES:
            hits, best = [], []
            for prof in profiles.values():
                y = np.asarray(prof[ax_name][key], float)
                fg = np.asarray(prof[ax_name]["fg"], float)
                keep = fg > 0.2 * np.nanmax(fg)
                idx = np.where(keep)[0]
                if len(idx) < 64:
                    continue
                y = np.nan_to_num(y[idx[0]:idx[-1] + 1], nan=float(np.nanmean(y)))
                per, pw = periodogram(detrend_profile(y, 3))
                ok = np.isfinite(per) & (per > 2)
                per, pw = per[ok], pw[ok]
                bg = median_filter(pw, size=max(11, len(pw) // 40), mode="nearest")
                ratio = pw / np.maximum(bg, 1e-30)
                m = (per >= 6) & (per <= 100)
                if m.any():
                    hits.append(float(ratio[m].max()))
                    best.append(float(per[m][int(np.argmax(ratio[m]))]))
            hits = np.array(hits); best = np.array(best)
            fine[key][ax_name]["narrow_line_screen"] = {
                "search_band_voxels": [6, 100],
                "n_volumes": int(len(hits)),
                "frac_volumes_line_gt_10x_background": float(np.mean(hits > 10)),
                "frac_volumes_line_gt_20x_background": float(np.mean(hits > 20)),
                "median_strongest_line_excess": float(np.median(hits)),
                "strongest_line_period_median_voxels": float(np.median(best)),
                "strongest_line_period_iqr_voxels": [float(np.percentile(best, 25)),
                                                     float(np.percentile(best, 75))],
            }
            nl = fine[key][ax_name]["narrow_line_screen"]
            print(f"  lines {key:5s} {ax_name}: >10x in "
                  f"{nl['frac_volumes_line_gt_10x_background']*100:5.1f}% of volumes, "
                  f"median line period {nl['strongest_line_period_median_voxels']:6.1f} vox "
                  f"(IQR {nl['strongest_line_period_iqr_voxels'][0]:.0f}-"
                  f"{nl['strongest_line_period_iqr_voxels'][1]:.0f})", flush=True)

    # global profile statistics used by the summary
    prof_stats = {}
    for ax_name in AXES:
        lens = [len(profiles[v][ax_name]["phi"]) for v in profiles]
        rng_ratio = []
        for v in profiles:
            y = np.asarray(profiles[v][ax_name]["phi"], float)
            fg = np.asarray(profiles[v][ax_name]["fg"], float)
            keep = fg > 0.2 * np.nanmax(fg)
            y = y[keep]
            if len(y) > 10 and np.nanmean(y) > 0:
                rng_ratio.append(float(np.nanstd(y) / np.nanmean(y)))
        prof_stats[ax_name] = {
            "extent_voxels_median": float(np.median(lens)),
            "extent_voxels_min": int(np.min(lens)),
            "extent_voxels_max": int(np.max(lens)),
            "profile_cv_median": float(np.median(rng_ratio)),
            "profile_cv_iqr": [float(np.percentile(rng_ratio, 25)),
                               float(np.percentile(rng_ratio, 75))],
        }

    results = {
        "test_id": TEST_ID,
        "description": "Periodicity of porosity and grayscale along z, y, x.",
        "voxel_size_um": VOXEL_SIZE_UM,
        "voxel_size_note": (
            "No voxel size is recorded in patches_meta.json, volume_stats.json, "
            "splits.json, the zarr attributes or the build_dataset code. "
            "All lengths are therefore reported in voxels only."
        ),
        "n_volumes": len(vids),
        "volume_shapes": {v: profiles[v]["shape"] for v in profiles},
        "axis_semantics": (
            "z is the short axis (96-320 voxels) = laminate through-thickness; "
            "y (2944-3840) and x (1568-1952) are the in-plane axes."
        ),
        "profile_stats": prof_stats,
        "coarse": coarse,
        "fine": fine,
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-A] wrote {p}", flush=True)

    ex_ids = sorted(profiles.keys())[:4]
    figs = []
    figs += fig_profiles(profiles, OUT_DIR, ex_ids)
    figs += fig_spectra(profiles, OUT_DIR, results)
    figs += fig_period_hist(results, OUT_DIR)
    figs += fig_autocorr(profiles, OUT_DIR)
    print("[T-A] figures:", *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()

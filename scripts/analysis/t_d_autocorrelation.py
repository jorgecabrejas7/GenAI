"""T-D — Spatial autocorrelation of porosity along z, y and x.

Two resolutions:

1. **Patch level** — every volume's patches form a regular grid with spacing
   ``stride`` = 32 voxels.  For each axis the Pearson correlation between patch
   pairs separated by ``k`` grid steps is pooled over all volumes, which gives a
   correlation curve at 32-voxel resolution out to thousands of voxels.
   This is the scale that decides whether neighbour conditioning is useful.

2. **Voxel level** — the exact 1-voxel porosity profiles cached by T-A are
   re-used to look for a secondary autocorrelation peak at the ply period.
   The patch grid cannot resolve anything below 64 voxels, so this second pass
   is what actually tests the ply hypothesis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AXIS_COLORS, OUT_ROOT, PATCH_INDEX, PATCH_SIZE, STRIDE,
    autocorrelation, correlation_length, detrend_profile,
    savefig, set_style, write_json, plt,
)

TEST_ID = "T-D"
OUT_DIR = OUT_ROOT / TEST_ID
AXES = ("z", "y", "x")


def build_grid(sub: pd.DataFrame) -> np.ndarray:
    """(nz, ny, nx) array of patch porosity, NaN where a patch is missing."""
    iz = (sub["z0"].to_numpy() // STRIDE).astype(int)
    iy = (sub["y0"].to_numpy() // STRIDE).astype(int)
    ix = (sub["x0"].to_numpy() // STRIDE).astype(int)
    g = np.full((iz.max() + 1, iy.max() + 1, ix.max() + 1), np.nan, dtype=np.float64)
    g[iz, iy, ix] = sub["porosity"].to_numpy(dtype=np.float64)
    return g


def pooled_lag_correlation(grids: list[np.ndarray], axis: int, max_lag: int) -> dict:
    """Pooled Pearson correlation between patch pairs at each lag along one axis."""
    n = np.zeros(max_lag + 1)
    sx = np.zeros(max_lag + 1); sy = np.zeros(max_lag + 1)
    sxx = np.zeros(max_lag + 1); syy = np.zeros(max_lag + 1); sxy = np.zeros(max_lag + 1)
    for g in grids:
        L = g.shape[axis]
        for k in range(0, min(max_lag, L - 1) + 1):
            a = np.take(g, np.arange(0, L - k), axis=axis)
            b = np.take(g, np.arange(k, L), axis=axis)
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                continue
            av, bv = a[m], b[m]
            n[k] += av.size
            sx[k] += av.sum(); sy[k] += bv.sum()
            sxx[k] += (av * av).sum(); syy[k] += (bv * bv).sum()
            sxy[k] += (av * bv).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = sx / n; my = sy / n
        cov = sxy / n - mx * my
        vx = sxx / n - mx * mx
        vy = syy / n - my * my
        r = cov / np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    lags_vox = np.arange(max_lag + 1) * STRIDE
    return {"lag_voxels": lags_vox, "r": r, "n_pairs": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lag-patches", type=int, default=100)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PATCH_INDEX)
    df = df[df["porosity"] <= 1.0]
    vids = sorted(df["volume_id"].unique().tolist())
    print(f"[T-D] {len(vids)} volumes, {len(df)} patches", flush=True)

    grids, grids_dm = [], []
    for vid, sub in df.groupby("volume_id", observed=True):
        g = build_grid(sub)
        grids.append(g)
        # per-volume mean removed: measures within-specimen structure only,
        # not the between-specimen spread in global porosity
        grids_dm.append(g - np.nanmean(g))

    results = {
        "test_id": TEST_ID,
        "patch_size_voxels": PATCH_SIZE,
        "stride_voxels": STRIDE,
        "n_volumes": len(vids),
        "n_patches": int(len(df)),
        "patch_level": {},
        "voxel_level": {},
        "note_overlap": (
            f"Patches overlap: stride {STRIDE} < patch size {PATCH_SIZE}. "
            f"Lags below {PATCH_SIZE} voxels therefore share voxels and are "
            "correlated by construction. Lags >= 64 voxels are voxel-disjoint "
            "and carry the real spatial signal."
        ),
    }

    for ai, axis in enumerate(AXES):
        for tag, gl in (("raw", grids), ("volume_mean_removed", grids_dm)):
            out = pooled_lag_correlation(gl, ai, args.max_lag_patches)
            lags, r = out["lag_voxels"], out["r"]
            valid = np.isfinite(r) & (out["n_pairs"] > 1000)
            key = f"{axis}_{tag}"
            results["patch_level"][key] = {
                "lag_voxels": lags[valid].tolist(),
                "correlation": r[valid].tolist(),
                "n_pairs": out["n_pairs"][valid].tolist(),
                "corr_length_1_over_e_voxels": correlation_length(lags[valid], r[valid]),
                **{f"r_at_{k * STRIDE}_voxels":
                   (float(r[k]) if (len(r) > k and valid[k]) else None)
                   for k in (2, 4, 8, 16, 32)},
                "max_lag_with_pairs_voxels": int(lags[valid].max()) if valid.any() else 0,
            }
            print(f"  patch-level {key:26s} L_1/e = "
                  f"{results['patch_level'][key]['corr_length_1_over_e_voxels']:.1f} vox, "
                  f"r(64)={results['patch_level'][key]['r_at_64_voxels']}", flush=True)

    # ---- voxel-level, from the T-A cache ----
    cache = OUT_ROOT / "T-A" / "fine_profiles.npz"
    voxel_curves = {}
    if cache.exists():
        profiles = np.load(cache, allow_pickle=True)["profiles"].item()
        for axis in AXES:
            maxlag = 80 if axis == "z" else 200
            curves = []
            for prof in profiles.values():
                y = np.asarray(prof[axis]["phi"], float)
                fg = np.asarray(prof[axis]["fg"], float)
                keep = fg > 0.2 * np.nanmax(fg)
                idx = np.where(keep)[0]
                if len(idx) < 2 * maxlag:
                    continue
                y = np.nan_to_num(y[idx[0]:idx[-1] + 1], nan=float(np.nanmean(y)))
                curves.append(autocorrelation(detrend_profile(y, 3), maxlag))
            if not curves:
                continue
            C = np.vstack(curves)
            mean_c = C.mean(0)
            lags = np.arange(len(mean_c), dtype=float)
            voxel_curves[axis] = (lags, C)
            # secondary peak = highest local maximum beyond lag 4
            d = mean_c
            loc = [i for i in range(5, len(d) - 1) if d[i] > d[i - 1] and d[i] > d[i + 1]]
            sec = max(loc, key=lambda i: d[i]) if loc else None
            results["voxel_level"][axis] = {
                "n_volumes": int(C.shape[0]),
                "mean_autocorrelation": mean_c.tolist(),
                "corr_length_1_over_e_voxels": correlation_length(lags, mean_c),
                "secondary_peak_lag_voxels": int(sec) if sec is not None else None,
                "secondary_peak_value": float(d[sec]) if sec is not None else None,
            }
            print(f"  voxel-level {axis}: L_1/e = "
                  f"{results['voxel_level'][axis]['corr_length_1_over_e_voxels']:.1f} vox, "
                  f"secondary peak at lag {results['voxel_level'][axis]['secondary_peak_lag_voxels']}",
                  flush=True)
    else:
        print("[T-D] T-A profile cache missing — skipping the voxel-level pass", flush=True)

    p = write_json(results, OUT_DIR)
    print(f"[T-D] wrote {p}", flush=True)

    # ---- figures ----
    set_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for tag, ax, title in (("raw", axs[0], "raw patch $\\varphi$"),
                           ("volume_mean_removed", axs[1], "volume mean removed")):
        for axis in AXES:
            d = results["patch_level"][f"{axis}_{tag}"]
            L = d["corr_length_1_over_e_voxels"]
            lab = (f"{axis}   $L_{{1/e}}$ = {L:.0f} vox" if np.isfinite(L)
                   else f"{axis}   $L_{{1/e}}$ > {max(d['lag_voxels']):.0f} vox")
            ax.plot(d["lag_voxels"], d["correlation"], color=AXIS_COLORS[axis], label=lab)
        ax.axhline(np.exp(-1), color="k", ls="--", lw=0.9, label="$1/e$")
        ax.axvline(PATCH_SIZE, color="grey", ls=":", lw=1.0)
        ax.text(PATCH_SIZE * 1.06, 0.92, "patch size\n64 vox", fontsize=8, color="grey")
        ax.set_xscale("log")
        ax.set_xlabel("separation between patch centres (voxels)")
        ax.set_ylabel("Pearson correlation of $\\varphi$" if tag == "raw" else "")
        ax.set_title(title)
        ax.set_ylim(-0.2, 1.02)
        ax.legend(frameon=False)
    fig.suptitle("T-D  Spatial autocorrelation of patch porosity, per axis "
                 f"({results['n_patches']:,} patches, {results['n_volumes']} volumes)", fontsize=11)
    f1 = savefig(fig, OUT_DIR, "TD_fig1_patch_autocorrelation")

    figs = list(f1)
    if voxel_curves:
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
        for j, axis in enumerate(AXES):
            ax = axs[j]
            if axis not in voxel_curves:
                ax.set_axis_off()
                continue
            lags, C = voxel_curves[axis]
            ax.plot(lags, C.mean(0), color=AXIS_COLORS[axis], lw=1.6)
            ax.fill_between(lags, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                            color=AXIS_COLORS[axis], alpha=0.22, lw=0)
            sec = results["voxel_level"][axis]["secondary_peak_lag_voxels"]
            if sec:
                ax.axvline(sec, color="k", ls=":", lw=1.0)
                ax.text(sec, 0.85, f" lag {sec}", fontsize=8)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel(f"lag along {axis} (voxels)")
            ax.set_ylabel("autocorrelation" if j == 0 else "")
            ax.set_title(f"{axis}-axis")
        fig.suptitle("T-D  Voxel-resolution autocorrelation of the detrended $\\varphi$ profile "
                     "(mean and IQR over volumes)", fontsize=11)
        figs += savefig(fig, OUT_DIR, "TD_fig2_voxel_autocorrelation")
    print("[T-D] figures:", *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()

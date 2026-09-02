"""T-C — Air / specimen boundary: which coordinate predicts exterior air?

Reads the grayscale voxels of every patch from ``patches_xct.bin`` (a flat
(N, 64, 64, 64) uint8 memmap whose row *i* is row *i* of ``patch_index.parquet``)
and computes, per patch:

* a 256-bin grayscale histogram, from which we take Otsu's threshold and
  Otsu's separability eta = between-class variance / total variance
  (eta near 1 = a clean two-mode histogram, i.e. air plus material);
* the air-voxel fraction, defined with the **volume-level** Otsu threshold that
  T-A already computed, so the measure is comparable across patches;
* mean and standard deviation of the grayscale.

These are then regressed against two competing position descriptors:

* fractional position inside the volume, per axis, and its distance from the
  mid-plane;
* absolute distance to the specimen surface, taken from the per-volume
  foreground extent measured in T-A.

Run ``--smoke N`` first: it measures throughput on N random patches and prints
the projected full-dataset runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AXIS_COLORS, DATA_ROOT, OUT_ROOT, PATCH_INDEX, PATCH_SIZE,
    savefig, set_style, write_json, plt,
)

TEST_ID = "T-C"
OUT_DIR = OUT_ROOT / TEST_ID
PART_DIR = OUT_DIR / "parts"
MEMMAP = DATA_ROOT / "patches_xct.bin"
META = DATA_ROOT / "patches_meta.json"
NVOX = PATCH_SIZE ** 3


def _otsu_stats(hist: np.ndarray) -> tuple[int, float, float, float]:
    """Otsu threshold, separability eta, global mean and variance from a histogram."""
    total = hist.sum()
    p = hist / total
    lev = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(lev * p)
    mu_T = mu[-1]
    var_T = float(np.sum(p * (lev - mu_T) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = np.where((omega > 1e-12) & (omega < 1 - 1e-12),
                      (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega)), 0.0)
    k = int(np.nanargmax(sb))
    eta = float(sb[k] / var_T) if var_T > 0 else 0.0
    return k, eta, float(mu_T), var_T


def process_block(args) -> str:
    """Process a contiguous block of memmap rows; write a .npz part file."""
    start, stop, vol_thr, part_path = args
    part = Path(part_path)
    if part.exists():
        return str(part)
    with open(META) as fh:
        N = int(json.load(fh)["N"])
    mm = np.memmap(MEMMAP, dtype=np.uint8, mode="r", shape=(N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
    n = stop - start
    otsu = np.zeros(n, dtype=np.int16)
    eta = np.zeros(n, dtype=np.float32)
    mean = np.zeros(n, dtype=np.float32)
    std = np.zeros(n, dtype=np.float32)
    air = np.zeros(n, dtype=np.float32)
    dark_p1 = np.zeros(n, dtype=np.float32)
    step = 256
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        block = np.asarray(mm[start + i0: start + i1])
        for j in range(i1 - i0):
            h = np.bincount(block[j].ravel(), minlength=256).astype(np.float64)
            k, e, m, v = _otsu_stats(h)
            r = i0 + j
            otsu[r], eta[r], mean[r], std[r] = k, e, m, np.sqrt(max(v, 0.0))
            t = vol_thr[r]
            air[r] = h[:t].sum() / NVOX
            dark_p1[r] = h[:64].sum() / NVOX
    part.parent.mkdir(parents=True, exist_ok=True)
    np.savez(part, start=start, stop=stop, otsu=otsu, eta=eta,
             mean=mean, std=std, air=air, dark_p1=dark_p1)
    return str(part)


def load_volume_geometry() -> tuple[dict, dict]:
    """Per-volume Otsu threshold and foreground extent, from the T-A voxel pass."""
    cache = OUT_ROOT / "T-A" / "fine_profiles.npz"
    if not cache.exists():
        raise SystemExit("T-C needs the T-A profile cache — run t_a_periodicity.py first.")
    profiles = np.load(cache, allow_pickle=True)["profiles"].item()
    thr, extent = {}, {}
    for vid, p in profiles.items():
        thr[vid] = int(p["otsu_threshold"])
        e = {}
        for a in ("z", "y", "x"):
            fg = np.asarray(p[a]["fg"], float)
            idx = np.where(fg > 0.5 * fg.max())[0]
            e[a] = [int(idx[0]), int(idx[-1])] if len(idx) else [0, len(fg) - 1]
        e["shape"] = p["shape"]
        extent[vid] = e
    return thr, extent


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def binned_r2(x: np.ndarray, y: np.ndarray, n_bins: int = 50) -> float:
    """Variance of y explained by a non-parametric bin-mean fit on x."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    pred = np.zeros_like(y)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            pred[m] = y[m].mean()
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run a throughput smoke test on N random patches and exit")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=20000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PATCH_INDEX)
    N = len(df)
    vol_thr_map, extent = load_volume_geometry()
    vol_thr_all = df["volume_id"].map(vol_thr_map).to_numpy()
    if np.isnan(vol_thr_all.astype(float)).any():
        raise SystemExit("some volumes have no T-A threshold")
    vol_thr_all = vol_thr_all.astype(np.int16)

    # ---------------- smoke test ----------------
    if args.smoke:
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(N, size=args.smoke, replace=False))
        mm = np.memmap(MEMMAP, dtype=np.uint8, mode="r",
                       shape=(N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
        t0 = time.time()
        for r in rows:
            h = np.bincount(np.asarray(mm[r]).ravel(), minlength=256).astype(np.float64)
            _otsu_stats(h)
        dt = time.time() - t0
        rate = args.smoke / dt
        gbs = args.smoke * NVOX / dt / 1e9
        print("\n" + "=" * 70)
        print(f"[T-C SMOKE] {args.smoke} random patches in {dt:.1f} s")
        print(f"[T-C SMOKE] throughput        : {rate:8.1f} patches/s/worker "
              f"({gbs:.2f} GB/s)")
        print(f"[T-C SMOKE] full dataset      : {N:,} patches "
              f"= {N * NVOX / 1e12:.2f} TB of voxel reads")
        print(f"[T-C SMOKE] projected 1 worker: {N / rate / 60:8.1f} min")
        for w in (4, 8, 12):
            print(f"[T-C SMOKE] projected {w:2d} workers (I/O-bound, ~linear): "
                  f"{N / rate / 60 / w:8.1f} min")
        print("=" * 70 + "\n", flush=True)
        return

    # ---------------- full run ----------------
    PART_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for s in range(0, N, args.block):
        e = min(s + args.block, N)
        blocks.append((s, e, vol_thr_all[s:e].copy(),
                       str(PART_DIR / f"part_{s:09d}.npz")))
    todo = [b for b in blocks if not Path(b[3]).exists()]
    print(f"[T-C] {N:,} patches in {len(blocks)} blocks; {len(todo)} still to do; "
          f"{args.workers} workers", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_block, b) for b in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                done_patches = i * args.block
                eta_s = el / i * (len(todo) - i)
                print(f"  [{i:4d}/{len(todo)}] {done_patches/1e6:5.2f}M patches  "
                      f"elapsed {el/60:6.1f} min  rate {done_patches/el:7.0f} p/s  "
                      f"ETA {eta_s/60:6.1f} min", flush=True)

    # ---------------- assemble ----------------
    otsu = np.zeros(N, np.int16); eta = np.zeros(N, np.float32)
    mean = np.zeros(N, np.float32); std = np.zeros(N, np.float32)
    air = np.zeros(N, np.float32); dark = np.zeros(N, np.float32)
    for s, e, _, p in blocks:
        z = np.load(p)
        otsu[s:e] = z["otsu"]; eta[s:e] = z["eta"]; mean[s:e] = z["mean"]
        std[s:e] = z["std"]; air[s:e] = z["air"]; dark[s:e] = z["dark_p1"]
    np.savez_compressed(OUT_DIR / "patch_air_stats.npz", otsu=otsu, eta=eta,
                        mean=mean, std=std, air=air, dark_p1=dark)

    # ---------------- position descriptors ----------------
    zc = df["z0"].to_numpy() + PATCH_SIZE / 2
    yc = df["y0"].to_numpy() + PATCH_SIZE / 2
    xc = df["x0"].to_numpy() + PATCH_SIZE / 2
    ext = df["volume_id"].map(extent)
    D = np.array([e["shape"][0] for e in ext]); Hh = np.array([e["shape"][1] for e in ext])
    Ww = np.array([e["shape"][2] for e in ext])
    zlo = np.array([e["z"][0] for e in ext]); zhi = np.array([e["z"][1] for e in ext])
    ylo = np.array([e["y"][0] for e in ext]); yhi = np.array([e["y"][1] for e in ext])
    xlo = np.array([e["x"][0] for e in ext]); xhi = np.array([e["x"][1] for e in ext])

    fz, fy, fx = zc / D, yc / Hh, xc / Ww
    dz = np.minimum(zc - zlo, zhi - zc)
    dy = np.minimum(yc - ylo, yhi - yc)
    dx = np.minimum(xc - xlo, xhi - xc)
    dmin = np.minimum(np.minimum(dz, dy), dx)

    predictors = {
        "frac_z": fz, "frac_y": fy, "frac_x": fx,
        "abs_frac_z_from_midplane": np.abs(fz - 0.5),
        "abs_frac_y_from_mid": np.abs(fy - 0.5),
        "abs_frac_x_from_mid": np.abs(fx - 0.5),
        "dist_to_z_surface_voxels": dz,
        "dist_to_y_surface_voxels": dy,
        "dist_to_x_surface_voxels": dx,
        "dist_to_nearest_surface_voxels": dmin,
    }

    air64 = air.astype(np.float64)
    results = {
        "test_id": TEST_ID,
        "n_patches": int(N),
        "air_threshold_definition": ("voxels darker than the volume-level Otsu "
                                     "threshold computed in T-A"),
        "air_fraction": {
            "mean": float(air64.mean()), "median": float(np.median(air64)),
            "p90": float(np.percentile(air64, 90)), "p99": float(np.percentile(air64, 99)),
            "max": float(air64.max()),
            "frac_patches_air_gt_0.01": float(np.mean(air64 > 0.01)),
            "frac_patches_air_gt_0.05": float(np.mean(air64 > 0.05)),
            "frac_patches_air_gt_0.20": float(np.mean(air64 > 0.20)),
            "frac_patches_air_gt_0.50": float(np.mean(air64 > 0.50)),
            "n_patches_air_gt_0.01": int(np.sum(air64 > 0.01)),
            "n_patches_air_gt_0.05": int(np.sum(air64 > 0.05)),
        },
        "otsu_separability_eta": {
            "mean": float(eta.mean()), "median": float(np.median(eta)),
            "p90": float(np.percentile(eta, 90)),
            "frac_eta_gt_0.5": float(np.mean(eta > 0.5)),
            "frac_eta_gt_0.7": float(np.mean(eta > 0.7)),
        },
        "predictors": {},
    }
    for name, v in predictors.items():
        v = np.asarray(v, dtype=np.float64)
        results["predictors"][name] = {
            "spearman_vs_air": spearman(v, air64),
            "binned_r2_vs_air": binned_r2(v, air64),
            "binned_r2_vs_eta": binned_r2(v, eta.astype(np.float64)),
        }
        print(f"  {name:34s} rho={results['predictors'][name]['spearman_vs_air']:+.3f} "
              f"R2(air)={results['predictors'][name]['binned_r2_vs_air']:.3f} "
              f"R2(eta)={results['predictors'][name]['binned_r2_vs_eta']:.3f}", flush=True)

    best_air = max(results["predictors"], key=lambda k: results["predictors"][k]["binned_r2_vs_air"])
    results["best_predictor_of_air"] = best_air

    # affected patches broken down by z position
    aff = air64 > 0.01
    results["affected_patches_by_z"] = {
        "frac_of_affected_in_outer_20pct_of_z": float(
            np.mean((np.abs(fz - 0.5) > 0.3)[aff])) if aff.any() else None,
        "frac_of_all_patches_in_outer_20pct_of_z": float(np.mean(np.abs(fz - 0.5) > 0.3)),
        "median_dist_to_z_surface_affected": float(np.median(dz[aff])) if aff.any() else None,
        "median_dist_to_z_surface_all": float(np.median(dz)),
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-C] wrote {p}", flush=True)

    # ---------------- figures ----------------
    set_style()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    ax = axs[0]
    ax.hist(air64, bins=np.linspace(0, 1, 101), color="#1b6ca8", log=True)
    ax.set_xlabel("air-voxel fraction of the patch")
    ax.set_ylabel("number of patches")
    ax.set_title("Air content per patch")
    ax = axs[1]
    ax.hist(eta, bins=np.linspace(0, 1, 101), color="#c2571a", log=True)
    ax.set_xlabel("Otsu separability $\\eta$ (dimensionless)")
    ax.set_ylabel("number of patches")
    ax.set_title("Grayscale bimodality per patch")
    ax = axs[2]
    for name, lab, col in (("frac_z", "fractional z", AXIS_COLORS["z"]),
                           ("frac_y", "fractional y", AXIS_COLORS["y"]),
                           ("frac_x", "fractional x", AXIS_COLORS["x"])):
        v = predictors[name]
        edges = np.linspace(0, 1, 41)
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, 39)
        m = np.array([air64[idx == b].mean() if (idx == b).any() else np.nan for b in range(40)])
        ax.plot(0.5 * (edges[1:] + edges[:-1]), m, color=col, label=lab)
    ax.set_xlabel("fractional position in the volume")
    ax.set_ylabel("mean air fraction")
    ax.set_title("Air vs fractional position")
    ax.legend(frameon=False)
    fig.suptitle("T-C  Exterior air in patches", fontsize=12)
    figs = savefig(fig, OUT_DIR, "TC_fig1_air_overview")

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    ax = axs[0]
    for name, lab, col in (("dist_to_z_surface_voxels", "distance to z surface", AXIS_COLORS["z"]),
                           ("dist_to_nearest_surface_voxels", "distance to nearest surface", "#6a3d9a")):
        v = np.asarray(predictors[name], float)
        edges = np.linspace(0, min(400, np.percentile(v, 99)), 41)
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, 39)
        m = np.array([air64[idx == b].mean() if (idx == b).any() else np.nan for b in range(40)])
        ax.plot(0.5 * (edges[1:] + edges[:-1]), m, color=col, label=lab)
    ax.axvline(PATCH_SIZE / 2, color="grey", ls=":", lw=1.0)
    ax.text(PATCH_SIZE / 2 * 1.05, 0.5 * ax.get_ylim()[1], "half a patch", fontsize=8, color="grey")
    ax.set_xlabel("distance to the specimen surface (voxels)")
    ax.set_ylabel("mean air fraction")
    ax.set_title("Air vs absolute distance to surface")
    ax.legend(frameon=False)
    ax = axs[1]
    names = list(results["predictors"].keys())
    r2 = [results["predictors"][n]["binned_r2_vs_air"] for n in names]
    order = np.argsort(r2)[::-1]
    ax.barh([names[i] for i in order][::-1], [r2[i] for i in order][::-1], color="#2e7d32")
    ax.set_xlabel("$R^2$ of a non-parametric bin fit, air fraction")
    ax.set_title("Which coordinate predicts air?")
    ax.grid(axis="y", alpha=0)
    fig.suptitle("T-C  Fractional position vs absolute distance to surface", fontsize=12)
    figs += savefig(fig, OUT_DIR, "TC_fig2_predictors")
    print("[T-C] figures:", *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()

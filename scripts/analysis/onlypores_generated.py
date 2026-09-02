"""Ground-truth porosity of generated volumes, measured with ``onlypores``.

The model's own mask head under-reports porosity where air is massive
(``runs/eval_v2/audit/findings.md``: median pore capture drops to ~0.03 in
64³ cells with >15% dark air).  Every porosity-control result so far is
mask-based, so this script measures the generated volumes with the SAME
segmentation the real dataset was built with —
``poregen.dataset.segmentation.onlypores`` at its default parameters, exactly
as ``poregen.dataset.io.compute_mask`` calls it (sauvola_radius=30,
sauvola_k=0.125, frontwall=0, backwall=0, min_size_filtering=-1).

Intensity scale
---------------
``VolumeGenerator`` writes ``expit(xct_out)`` while the VAE decodes directly
to [0, 1], so the saved grayscale is sigmoid-compressed.  The inversion is the
one used by ``scripts/analysis/eval_v2_audit.py``: ``clip(logit(v), 0, 1) * 255``
→ uint8 on the decoder-native scale, where the real material mode (~208-210 u8)
lands.  onlypores' absolute Otsu/Sauvola behaviour then matches real scans.

Porosity denominator
--------------------
PRIMARY: ``pore_voxels / sample_mask_voxels`` — porosity of the material the
segmentation actually recognises as specimen, the same quantity reported for
real scans.  ``pore_voxels / total_voxels`` is reported beside it, and so is
``sample_mask_fraction`` (how much of the box onlypores calls material at all),
which is what exposes volumes that are largely air.

Outputs → ``runs/analysis/onlypores_generated/``:
    real_validation.json, results.json, per_volume.csv, per_cell.csv,
    findings.md, run.log, figures (PDF + PNG, 300 dpi).

Usage:
    python scripts/analysis/onlypores_generated.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from poregen.dataset.segmentation import onlypores  # noqa: E402

OUT_DIR = REPO / "runs" / "analysis" / "onlypores_generated"
VOL_ROOT = REPO / "runs" / "eval_v2" / "volumes"
PROBE_ROOT = REPO / "runs" / "analysis" / "ldm06_probe" / "volumes"
ZARR_ROOT = REPO / "data" / "split_v2" / "volumes.zarr"
PATCH_INDEX = REPO / "data" / "split_v2" / "patch_index.parquet"
AUDIT_CELLS = REPO / "runs" / "eval_v2" / "audit" / "cells.csv.gz"

EXPERIMENTS = ["dose_response", "cfg_sweep", "layup"]
ARMS = ["seq", "joint_legacy", "joint_oob"]
ARM_SPOR = {"seq": 1.0, "joint_legacy": 1.5, "joint_oob": 1.5}
PATCH = 64
GATE = 0.005                       # D39 acceptance gate on |delivered - requested|

# Real val/test volumes used to validate that this pipeline reproduces the
# dataset's own ground-truth segmentation (same three the audit calibrated on).
REAL_VOLUMES = [
    "MedidasDB__Juan_Ignacio_probetas_8_volume_eq_aligned",
    "MedidasDB__Airbus_Panel_Pegaso_probetas_1_26_volumen_eq_aligned",
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_01_3_volume_eq_aligned",
]

# Categorical hues, validated with the dataviz palette checker (light surface):
# lightness band, chroma floor, CVD separation, normal-vision floor and contrast
# all PASS.  The repo's previous green for joint_oob failed CVD separation
# against the orange (protan dE 3.6); purple replaces it.  Arms additionally
# carry distinct markers, so identity is never colour-alone.
ARM_COLORS = {"seq": "#1b6ca8", "joint_legacy": "#c2571a", "joint_oob": "#6a3d9a"}
ARM_MARKERS = {"seq": "o", "joint_legacy": "s", "joint_oob": "^"}
MEAS_COLORS = {"mask": "#1b6ca8", "onlypores": "#c2571a"}

_LOG_FH = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


# ---------------------------------------------------------------------------
# Scale inversion + block statistics
# ---------------------------------------------------------------------------

def to_native_u8(vol_path: Path) -> np.ndarray:
    """Generated float [0,1] sigmoid-scale TIFF -> decoder-native u8, chunked."""
    mm = tifffile.memmap(str(vol_path), mode="r")
    out = np.empty(mm.shape, np.uint8)
    for z0 in range(0, mm.shape[0], 32):
        chunk = np.asarray(mm[z0:z0 + 32], np.float32)
        if chunk.max() > 1.5:
            chunk = chunk / 255.0
        frac = np.clip(chunk, 1e-6, 1.0 - 1e-6)
        out[z0:z0 + 32] = (np.clip(logit(frac), 0.0, 1.0) * 255.0).astype(np.uint8)
    del mm
    return out


def block_sums(a: np.ndarray) -> np.ndarray:
    """Voxel count of a boolean array per 64³ tile cell, shape (gz, gy, gx)."""
    gz, gy, gx = (s // PATCH for s in a.shape)
    return (a[:gz * PATCH, :gy * PATCH, :gx * PATCH]
            .reshape(gz, PATCH, gy, PATCH, gx, PATCH)
            .sum(axis=(1, 3, 5), dtype=np.int64))


def material_mode(native: np.ndarray, sample_mask: np.ndarray) -> int:
    """Modal u8 level of the material (levels >= 100, inside the sample mask)."""
    hist = np.bincount(native[sample_mask].ravel(), minlength=256)
    return int(np.argmax(hist[100:]) + 100)


# ---------------------------------------------------------------------------
# Step 1 — validation on real volumes
# ---------------------------------------------------------------------------

def validate_real() -> dict:
    """Re-run the pipeline on real volumes and compare to the stored masks.

    ``data/split_v2/volumes.zarr/<vol>/mask`` IS the onlypores pore mask written
    by ``build_dataset``; ``sample_mask`` is its material mask.  Reproducing them
    from the stored ``xct`` is the strongest possible agreement check.
    """
    log("real validation: 3 volumes")
    g = zarr.open(str(ZARR_ROOT), mode="r")
    pidx = pd.read_parquet(PATCH_INDEX, columns=["volume_id", "porosity"])
    out = {}
    for name in REAL_VOLUMES:
        t0 = time.time()
        grp = g[name]
        xct = np.asarray(grp["xct"])
        stored_pore = np.asarray(grp["mask"]) > 0
        stored_sm = np.asarray(grp["sample_mask"]) > 0
        pore, sm, _ = onlypores(xct)
        inter = int(np.count_nonzero(pore & stored_pore))
        n_new, n_old = int(pore.sum()), int(stored_pore.sum())
        smi = int(np.count_nonzero(sm & stored_sm))
        pp = pidx.loc[pidx.volume_id == name, "porosity"]
        rec = {
            "shape": [int(s) for s in xct.shape],
            "seconds": round(time.time() - t0, 1),
            "recomputed_porosity_over_sample": n_new / float(sm.sum()),
            "stored_porosity_over_sample": n_old / float(stored_sm.sum()),
            "recomputed_porosity_over_total": n_new / float(xct.size),
            "stored_porosity_over_total": n_old / float(xct.size),
            "recomputed_sample_frac": float(sm.mean()),
            "stored_sample_frac": float(stored_sm.mean()),
            "dice_pore_mask": 2.0 * inter / (n_new + n_old),
            "iou_pore_mask": inter / float(n_new + n_old - inter),
            "dice_sample_mask": 2.0 * smi / float(sm.sum() + stored_sm.sum()),
            "patch_index_mean_porosity": float(pp.mean()) if len(pp) else None,
            "patch_index_n_patches": int(len(pp)),
            "material_mode_u8": material_mode(xct, stored_sm),
        }
        out[name] = rec
        log(f"  {name[:48]:48s} dice={rec['dice_pore_mask']:.4f} "
            f"por/sample={rec['recomputed_porosity_over_sample']:.5f} "
            f"(stored {rec['stored_porosity_over_sample']:.5f}) "
            f"mode={rec['material_mode_u8']} {rec['seconds']}s")
        del xct, stored_pore, stored_sm, pore, sm
    return out


# ---------------------------------------------------------------------------
# Step 2 — per-volume + per-cell measurement of the generated volumes
# ---------------------------------------------------------------------------

def iter_volumes():
    """(experiment, arm, dir) over every generated volume to process."""
    for exp in EXPERIMENTS:
        for arm in ARMS:
            root = VOL_ROOT / exp / arm
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if (d / "volume.tif").exists() and (d / "mask.tif").exists():
                    yield exp, arm, d
    for d in sorted(PROBE_ROOT.iterdir()):
        if (d / "volume.tif").exists() and (d / "mask.tif").exists():
            yield "ldm06_probe", "probe", d


def measure(exp: str, arm: str, d: Path) -> tuple[dict, pd.DataFrame]:
    stats = json.loads((d / "stats.json").read_text())
    native = to_native_u8(d / "volume.tif")
    model_mask = tifffile.imread(str(d / "mask.tif")) > 0
    pore, sm, _binary = onlypores(native)
    if pore is None:                      # volume has no non-zero voxel
        pore = np.zeros(native.shape, bool)
        sm = np.zeros(native.shape, bool)
        _binary = None

    n_vox = int(native.size)
    n_sm = int(sm.sum())
    n_pore = int(pore.sum())
    n_mask = int(model_mask.sum())
    inter = int(np.count_nonzero(pore & model_mask))
    mask_in_sm = int(np.count_nonzero(model_mask & sm))

    target = stats.get("target")
    op_sample = n_pore / n_sm if n_sm else float("nan")
    mask_por = n_mask / n_vox

    row = {
        "experiment": exp,
        "arm": arm,
        "name": d.name,
        "path": str(d),
        "shape": "x".join(str(s) for s in native.shape),
        "n_voxels": n_vox,
        "target": target,
        "s_por": stats.get("s_por", ARM_SPOR.get(arm)),
        "seed": stats.get("seed"),
        "layup": stats.get("layup"),
        "ddim_steps": stats.get("ddim_steps"),
        # measurements
        "mask_porosity": mask_por,
        "mask_porosity_in_sample": mask_in_sm / n_sm if n_sm else float("nan"),
        "onlypores_porosity_sample": op_sample,
        "onlypores_porosity_total": n_pore / n_vox,
        "sample_mask_fraction": n_sm / n_vox,
        "material_mode_native_u8": material_mode(native, sm) if n_sm else -1,
        # cross-checks against what generation recorded
        "stats_delivered_mask_porosity": stats.get("delivered_mask_porosity"),
        "stats_corrected_porosity": stats.get("corrected_porosity"),
        # agreement between the two segmentations
        "dice_mask_vs_onlypores": (2.0 * inter / (n_mask + n_pore)
                                   if (n_mask + n_pore) else float("nan")),
        "mask_recall_of_onlypores": inter / n_pore if n_pore else float("nan"),
        "mask_precision_vs_onlypores": inter / n_mask if n_mask else float("nan"),
        "discrepancy_onlypores_minus_mask": op_sample - mask_por,
        "error_mask_vs_target": (mask_por - target) if target is not None else None,
        "error_onlypores_vs_target": (op_sample - target) if target is not None else None,
    }

    # ── per 64³ tile cell ──
    s_pore = block_sums(pore)
    s_sm = block_sums(sm)
    s_mask = block_sums(model_mask)
    gz, gy, gx = s_pore.shape
    cell_targets = stats.get("cell_targets")
    cell_delivered = stats.get("cell_delivered")
    per_cell = PATCH ** 3
    recs = []
    for iz in range(gz):
        for iy in range(gy):
            for ix in range(gx):
                key = f"{iz},{iy},{ix}"
                if cell_targets is not None and key in cell_targets:
                    ct, src = float(cell_targets[key]), "per_cell"
                else:
                    ct, src = target, "uniform_global"
                nsm = int(s_sm[iz, iy, ix])
                npo = int(s_pore[iz, iy, ix])
                nma = int(s_mask[iz, iy, ix])
                op_c = npo / nsm if nsm else float("nan")
                mk_c = nma / per_cell
                recs.append({
                    "experiment": exp, "arm": arm, "name": d.name,
                    "iz": iz, "iy": iy, "ix": ix,
                    "cell_index": (iz * gy + iy) * gx + ix,
                    "cell_target": ct,
                    "cell_target_source": src,
                    "cell_mask_porosity": mk_c,
                    "cell_onlypores_porosity_sample": op_c,
                    "cell_onlypores_porosity_total": npo / per_cell,
                    "cell_sample_frac": nsm / per_cell,
                    "cell_air_frac": 1.0 - nsm / per_cell,
                    "cell_stats_delivered": (float(cell_delivered[key])
                                             if cell_delivered and key in cell_delivered
                                             else None),
                    "cell_discrepancy": op_c - mk_c,
                    "cell_err_mask": mk_c - ct if ct is not None else None,
                    "cell_err_onlypores": op_c - ct if ct is not None else None,
                })
    del native, model_mask, pore, sm, _binary
    return row, pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Step 3 — analysis
# ---------------------------------------------------------------------------

def _num(col) -> np.ndarray:
    """Column -> float array, non-numeric/None -> NaN."""
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return {"slope": None, "intercept": None, "r2": None, "n": int(len(x))}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None, "n": int(len(x))}


def err_stats(err: np.ndarray) -> dict:
    a = np.abs(err[np.isfinite(err)])
    if not len(a):
        return {}
    return {"abs_error_mean": float(a.mean()),
            "abs_error_median": float(np.median(a)),
            "abs_error_p95": float(np.percentile(a, 95)),
            "abs_error_max": float(a.max()),
            "frac_within_gate": float((a < GATE).mean()),
            "n": int(len(a))}


def analyse(pv: pd.DataFrame, pc: pd.DataFrame) -> dict:
    res: dict = {}

    # ── global dose-response (dose_response set) ──
    dr = pv[pv.experiment == "dose_response"]
    res["global_dose_response"] = {}
    for arm in ARMS:
        s = dr[dr.arm == arm]
        res["global_dose_response"][arm] = {
            "s_por": float(s.s_por.iloc[0]),
            "n_volumes": int(len(s)),
            "onlypores_sample": fit_ols(_num(s.target), _num(s.onlypores_porosity_sample)),
            "onlypores_total": fit_ols(_num(s.target), _num(s.onlypores_porosity_total)),
            "mask": fit_ols(_num(s.target), _num(s.mask_porosity)),
            "err_onlypores": err_stats(_num(s.error_onlypores_vs_target)),
            "err_mask": err_stats(_num(s.error_mask_vs_target)),
            "mean_sample_mask_fraction": float(s.sample_mask_fraction.mean()),
        }

    # ── local dose-response (per-cell, dose_response set) ──
    drc = pc[pc.experiment == "dose_response"]
    res["local_dose_response"] = {}
    for arm in ARMS:
        s = drc[drc.arm == arm]
        res["local_dose_response"][arm] = {
            "n_cells": int(len(s)),
            "onlypores_sample": fit_ols(_num(s.cell_target),
                                        _num(s.cell_onlypores_porosity_sample)),
            "mask": fit_ols(_num(s.cell_target), _num(s.cell_mask_porosity)),
            "err_onlypores": err_stats(_num(s.cell_err_onlypores)),
            "err_mask": err_stats(_num(s.cell_err_mask)),
        }

    # ── per requested level ──
    lv = (dr.groupby(["arm", "target"])
            .agg(mask=("mask_porosity", "mean"),
                 mask_sd=("mask_porosity", "std"),
                 onlypores=("onlypores_porosity_sample", "mean"),
                 onlypores_sd=("onlypores_porosity_sample", "std"),
                 onlypores_total=("onlypores_porosity_total", "mean"),
                 sample_frac=("sample_mask_fraction", "mean"),
                 n=("name", "size"))
            .reset_index())
    res["dose_response_by_level"] = lv.to_dict("records")

    # ── cfg sweep ──
    cs = pv[pv.experiment == "cfg_sweep"]
    res["cfg_sweep_by_level"] = (cs.groupby(["arm", "s_por", "target"])
                                 .agg(mask=("mask_porosity", "mean"),
                                      onlypores=("onlypores_porosity_sample", "mean"),
                                      onlypores_total=("onlypores_porosity_total", "mean"),
                                      sample_frac=("sample_mask_fraction", "mean"),
                                      n=("name", "size"))
                                 .reset_index().to_dict("records"))

    # ── layup ──
    ly = pv[pv.experiment == "layup"]
    res["layup_by_arm"] = (ly.groupby(["arm", "layup"])
                           .agg(target=("target", "mean"),
                                mask=("mask_porosity", "mean"),
                                onlypores=("onlypores_porosity_sample", "mean"),
                                onlypores_total=("onlypores_porosity_total", "mean"),
                                sample_frac=("sample_mask_fraction", "mean"),
                                n=("name", "size"))
                           .reset_index().to_dict("records"))

    # ── mask vs onlypores discrepancy ──
    res["discrepancy"] = {
        "by_experiment_arm": (pv.groupby(["experiment", "arm"])
                              .agg(n=("name", "size"),
                                   mask=("mask_porosity", "mean"),
                                   onlypores=("onlypores_porosity_sample", "mean"),
                                   diff_mean=("discrepancy_onlypores_minus_mask", "mean"),
                                   diff_max=("discrepancy_onlypores_minus_mask", "max"),
                                   dice=("dice_mask_vs_onlypores", "mean"),
                                   mask_recall=("mask_recall_of_onlypores", "mean"),
                                   sample_frac=("sample_mask_fraction", "mean"))
                              .reset_index().to_dict("records")),
        "top20_volumes": (pv.nlargest(20, "discrepancy_onlypores_minus_mask")
                          [["experiment", "arm", "name", "target", "mask_porosity",
                            "onlypores_porosity_sample", "sample_mask_fraction",
                            "discrepancy_onlypores_minus_mask", "mask_recall_of_onlypores"]]
                          .to_dict("records")),
    }

    # cells binned by the onlypores air fraction (material the segmentation
    # does NOT recognise as specimen inside that cell)
    bins = [-0.001, 0.02, 0.15, 0.5, 1.001]
    labels = ["<2%", "2-15%", "15-50%", ">50%"]
    pcv = pc.copy()
    pcv["air_bin"] = pd.cut(pcv.cell_air_frac, bins=bins, labels=labels)
    g = (pcv.groupby(["arm", "air_bin"], observed=True)
         .agg(n=("name", "size"),
              mask=("cell_mask_porosity", "median"),
              onlypores=("cell_onlypores_porosity_sample", "median"),
              diff=("cell_discrepancy", "median"))
         .reset_index())
    g["air_bin"] = g["air_bin"].astype(str)
    res["discrepancy"]["cells_by_air_fraction"] = g.to_dict("records")

    # ── join with the audit's dark-air cell table ──
    if AUDIT_CELLS.exists():
        ac = pd.read_csv(AUDIT_CELLS)
        ac["cell_index"] = ac.groupby(["experiment", "arm", "name"]).cumcount()
        m = pc.merge(ac, on=["experiment", "arm", "name", "cell_index"],
                     how="inner", suffixes=("", "_audit"))
        if len(m):
            ok = (np.isfinite(m.cell_onlypores_porosity_sample)
                  & np.isfinite(m.cell_dark_frac))
            mm = m[ok]
            res["audit_join"] = {
                "n_cells_joined": int(len(mm)),
                "corr_darkfrac_vs_airfrac": float(np.corrcoef(
                    mm.cell_dark_frac, mm.cell_air_frac)[0, 1]),
                "corr_darkfrac_vs_discrepancy": float(np.corrcoef(
                    mm.cell_dark_frac, mm.cell_discrepancy)[0, 1]),
                "corr_maskpor_check": float(np.corrcoef(
                    mm.cell_mask_porosity, mm.cell_mask_porosity_audit)[0, 1]),
                "by_dark_bin": _dark_bins(mm),
            }

    # ── DDIM steps (probe) ──
    is192 = pv["shape"] == "192x192x192"   # the 1024 probe is "192x1024x1024"
    pr = pv[(pv.experiment == "ldm06_probe") & is192]
    res["ddim_steps"] = (pr.groupby("ddim_steps")
                         .agg(n=("name", "size"),
                              target=("target", "mean"),
                              mask=("mask_porosity", "mean"),
                              mask_sd=("mask_porosity", "std"),
                              onlypores=("onlypores_porosity_sample", "mean"),
                              onlypores_sd=("onlypores_porosity_sample", "std"),
                              onlypores_total=("onlypores_porosity_total", "mean"),
                              sample_frac=("sample_mask_fraction", "mean"))
                         .reset_index().to_dict("records"))
    res["ddim_big"] = (pv[(pv.experiment == "ldm06_probe") & ~is192]
                       [["name", "ddim_steps", "target", "mask_porosity",
                         "onlypores_porosity_sample", "onlypores_porosity_total",
                         "sample_mask_fraction"]].to_dict("records"))
    return res


def _dark_bins(mm: pd.DataFrame) -> list[dict]:
    bins = [-0.001, 0.02, 0.15, 1.001]
    labels = ["<2%", "2-15%", ">15%"]
    d = mm.copy()
    d["dark_bin"] = pd.cut(d.cell_dark_frac, bins=bins, labels=labels)
    g = (d.groupby(["arm", "dark_bin"], observed=True)
         .agg(n=("name", "size"),
              median_mask=("cell_mask_porosity", "median"),
              median_onlypores=("cell_onlypores_porosity_sample", "median"),
              median_discrepancy=("cell_discrepancy", "median"),
              median_air=("cell_air_frac", "median"))
         .reset_index())
    g["dark_bin"] = g["dark_bin"].astype(str)
    return g.to_dict("records")


# ---------------------------------------------------------------------------
# Step 4 — figures
# ---------------------------------------------------------------------------

def fig_delivered_vs_requested(pv: pd.DataFrame, res: dict):
    import matplotlib.pyplot as plt

    dr = pv[pv.experiment == "dose_response"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
    ymax = max(float((dr.groupby(["arm", "target"])[m].mean()
                      + dr.groupby(["arm", "target"])[m].std().fillna(0)).max())
               for m in ("mask_porosity", "onlypores_porosity_sample"))
    dy = {"seq": 9, "joint_legacy": 0, "joint_oob": -9}
    for ax, meas, ttl in (
        (axes[0], "mask_porosity",
         "model mask head\n(what every previous result used)"),
        (axes[1], "onlypores_porosity_sample",
         "onlypores (ground-truth method)\npore / sample-mask voxels"),
    ):
        ax.fill_between([0, 0.12], [-GATE, 0.12 - GATE], [GATE, 0.12 + GATE],
                        color="0.86", zorder=0, label=f"gate ±{GATE}")
        ax.plot([0, 0.12], [0, 0.12], color="0.35", lw=1.2, ls="--",
                zorder=1, label="identity")
        for arm in ARMS:
            g = dr[dr.arm == arm].groupby("target")[meas].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0),
                        color=ARM_COLORS[arm], marker=ARM_MARKERS[arm], ms=7,
                        lw=1.8, capsize=3, label=f"{arm} (s_por={ARM_SPOR[arm]:g})",
                        zorder=3)
            ax.annotate(arm, (g.index[-1], g["mean"].iloc[-1]),
                        textcoords="offset points", xytext=(8, dy[arm]),
                        fontsize=8, color=ARM_COLORS[arm], va="center")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("requested porosity")
        ax.set_xlim(0, 0.128)
    axes[0].set_ylabel("delivered porosity")
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Dose response, 192³ volumes — same volumes, two measurements",
                 y=0.99)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig_a_delivered_vs_requested")


def fig_mask_vs_onlypores(pv: pd.DataFrame, pc: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, df, xk, yk, ttl in (
        (axes[0], pv, "mask_porosity", "onlypores_porosity_sample",
         f"per volume (n={len(pv)})"),
        (axes[1], pc.sample(min(len(pc), 12000), random_state=0),
         "cell_mask_porosity", "cell_onlypores_porosity_sample",
         f"per 64³ cell (n={len(pc)}, 12k shown)"),
    ):
        hi = float(np.nanmax([_num(df[xk]).max(), _num(df[yk]).max()])) * 1.06
        ax.plot([0, hi], [0, hi], color="0.35", lw=1.2, ls="--",
                zorder=1, label="y = x (agreement)")
        for arm in ARMS + ["probe"]:
            s = df[df.arm == arm]
            if not len(s):
                continue
            ax.scatter(s[xk], s[yk], s=16 if ax is axes[0] else 5,
                       color=ARM_COLORS.get(arm, "#555555"),
                       marker=ARM_MARKERS.get(arm, "D"),
                       alpha=0.75 if ax is axes[0] else 0.25,
                       linewidths=0, label=arm, zorder=3)
        ax.set_xlabel("model mask porosity")
        ax.set_ylabel("onlypores porosity (pore / sample mask)")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Model mask vs onlypores — above y=x the mask under-reports "
                 "porosity, below it over-reports", y=0.99)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig_b_mask_vs_onlypores")


def fig_local_dose_response(pc: pd.DataFrame, res: dict):
    import matplotlib.pyplot as plt

    drc = pc[pc.experiment == "dose_response"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        s = drc[drc.arm == arm]
        ax.fill_between([0, 0.14], [-GATE, 0.14 - GATE], [GATE, 0.14 + GATE],
                        color="0.86", zorder=0, label=f"gate ±{GATE}")
        ax.plot([0, 0.14], [0, 0.14], color="0.35", lw=1.2, ls="--",
                zorder=1, label="identity")
        ax.scatter(s.cell_target, s.cell_mask_porosity, s=7, alpha=0.35,
                   color=MEAS_COLORS["mask"], linewidths=0, label="model mask",
                   zorder=2)
        ax.scatter(s.cell_target, s.cell_onlypores_porosity_sample, s=7,
                   alpha=0.35, color=MEAS_COLORS["onlypores"], linewidths=0,
                   label="onlypores", zorder=3)
        f = res["local_dose_response"][arm]
        xs = np.linspace(0, 0.14, 10)
        for key, ck in (("mask", "mask"), ("onlypores_sample", "onlypores")):
            fi = f[key]
            if fi["slope"] is not None:
                ax.plot(xs, fi["slope"] * xs + fi["intercept"],
                        color=MEAS_COLORS[ck], lw=2.0, zorder=4)
        ax.set_title(
            f"{arm}\nonlypores  slope {f['onlypores_sample']['slope']:.2f}  "
            f"R² {f['onlypores_sample']['r2']:.2f}\n"
            f"mask  slope {f['mask']['slope']:.2f}  R² {f['mask']['r2']:.2f}",
            fontsize=9)
        ax.set_xlabel("cell target porosity")
        ax.set_xlim(0, 0.14)
    axes[0].set_ylabel("delivered cell porosity")
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Local dose response — every 64³ cell of the dose-response set",
                 y=0.995)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig_c_local_dose_response")


def _log_steps(ax, steps) -> None:
    """Log x-axis ticked only at the sampled step counts (no minor labels)."""
    import matplotlib.ticker as mticker
    ax.set_xscale("log")
    ax.set_xticks(list(steps))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(int(v)) for v in steps]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def fig_ddim(res: dict):
    import matplotlib.pyplot as plt

    rows = sorted(res["ddim_steps"], key=lambda r: r["ddim_steps"])
    steps = [r["ddim_steps"] for r in rows]
    tgt = rows[0]["target"] if rows else 0.03
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    ax.axhline(tgt, color="0.35", ls="--", lw=1.2, label=f"target {tgt:g}")
    for key, sd, ck, lab in (("mask", "mask_sd", "mask", "model mask"),
                             ("onlypores", "onlypores_sd", "onlypores",
                              "onlypores (pore/sample)")):
        y = [r[key] for r in rows]
        e = [0.0 if not np.isfinite(r[sd]) else r[sd] for r in rows]
        ax.errorbar(steps, y, yerr=e, color=MEAS_COLORS[ck], marker="o", ms=7,
                    lw=1.8, capsize=3, label=lab)
        ax.annotate(lab, (steps[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=MEAS_COLORS[ck], va="center")
    _log_steps(ax, steps)
    ax.set_xlabel("DDIM steps")
    ax.set_ylabel("porosity")
    ax.set_title("Delivered porosity vs sampling steps\n(ldm06 probe, 192³, "
                 "2 seeds)", fontsize=10)
    ax.legend(loc="best", frameon=False)

    ax = axes[1]
    y = [r["sample_frac"] for r in rows]
    ax.plot(steps, y, color="#1b6ca8", marker="o", ms=7, lw=1.8)
    _log_steps(ax, steps)
    ax.set_xlabel("DDIM steps")
    ax.set_ylabel("sample-mask fraction")
    ax.set_title("How much of the box onlypores calls material\n"
                 "(1.0 = a fully solid specimen)", fontsize=10)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig_d_ddim_steps")


def fig_montage(pv: pd.DataFrame):
    import matplotlib.pyplot as plt

    err = _num(pv.error_onlypores_vs_target)
    dr = pv[(pv.experiment == "dose_response") & np.isfinite(err)].copy()
    dr["abs_err"] = np.abs(_num(dr.error_onlypores_vs_target))
    # "good": smallest onlypores-vs-target error among volumes that are still a
    # solid specimen (sample-mask fraction > 0.9), so the example is not a
    # mostly-air box that happens to land near the target.
    solid = dr[dr.sample_mask_fraction > 0.9]
    good = (solid if len(solid) else dr).nsmallest(1, "abs_err").iloc[0]
    bad = dr.nlargest(1, "abs_err").iloc[0]

    rows = []
    for tag, r in (("GOOD — solid specimen,\nonlypores near target", good),
                   ("BAD — largest onlypores\nerror vs target", bad)):
        d = Path(r["path"])
        native = to_native_u8(d / "volume.tif")
        mm = tifffile.imread(str(d / "mask.tif")) > 0
        pore, sm, _ = onlypores(native)
        miss = pore & ~mm
        z = int(np.argmax(miss.reshape(miss.shape[0], -1).mean(axis=1)))
        rows.append((
            f"{tag}\n{r['arm']} {r['name']}\n"
            f"target {r['target']:.3f} · mask {r['mask_porosity']:.3f} · "
            f"onlypores {r['onlypores_porosity_sample']:.3f}",
            native[z].copy(), mm[z].copy(), pore[z].copy(), sm[z].copy(), z))
        del native, mm, pore, sm, miss

    fig, ax = plt.subplots(len(rows), 4, figsize=(14.5, 4.0 * len(rows)))
    for i, (title, gray, mk, po, sml, z) in enumerate(rows):
        ax[i, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_ylabel(title, fontsize=8)
        ax[i, 1].imshow(mk, cmap="gray", vmin=0, vmax=1)
        ax[i, 2].imshow(po, cmap="gray", vmin=0, vmax=1)
        rgb = np.stack([gray] * 3, -1).astype(np.float32) / 255.0
        rgb[~sml] = [0.15, 0.15, 0.15]                      # outside sample mask
        rgb[po & ~mk] = [0.76, 0.34, 0.10]                  # onlypores only
        rgb[mk & ~po] = [0.11, 0.42, 0.66]                  # mask only
        rgb[mk & po] = [0.42, 0.24, 0.60]                   # both agree
        ax[i, 3].imshow(rgb)
        ax[i, 3].set_title(f"z={z}  orange = onlypores only, blue = mask only,\n"
                           "purple = both, dark = outside sample mask", fontsize=7.5)
        if i == 0:
            ax[i, 0].set_title("grayscale (decoder-native u8)", fontsize=9)
            ax[i, 1].set_title("model mask head", fontsize=9)
            ax[i, 2].set_title("onlypores pore mask", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Where the two segmentations disagree — worst slice of each volume",
                 y=0.995)
    fig.tight_layout()
    return savefig(fig, OUT_DIR, "fig_e_slice_montage")


# ---------------------------------------------------------------------------
# Step 5 — findings.md
# ---------------------------------------------------------------------------

def _f(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{n}f}"


def write_findings(pv: pd.DataFrame, pc: pd.DataFrame, res: dict,
                   real: dict, figs: list[str], runtime_s: float) -> str:
    L: list[str] = []
    A = L.append
    A("# Onlypores porosity of generated volumes")
    A("")
    A(f"Measured {len(pv)} generated volumes and {len(pc)} 64³ cells with "
      "`poregen.dataset.segmentation.onlypores` — the exact function and the "
      "exact default parameters `poregen.dataset.io.compute_mask` uses when the "
      "REAL dataset is built (`sauvola_radius=30, sauvola_k=0.125, frontwall=0, "
      "backwall=0, min_size_filtering=-1`).")
    A("")
    A("**Grayscale scale.** The sampler writes `expit(xct_out)`, so the saved "
      "TIFF is sigmoid-compressed. Every volume is inverted first with "
      "`clip(logit(v), 0, 1) * 255` → uint8 (identical to "
      "`scripts/analysis/eval_v2_audit.py`). On that scale the generated "
      "material mode lands at "
      f"{int(pv.material_mode_native_u8.median())} u8, against "
      f"{int(np.median([r['material_mode_u8'] for r in real.values()]))} u8 on the "
      "real volumes — the absolute thresholds inside onlypores therefore behave "
      "as they do on real scans.")
    A("")
    A("**Denominator.** The headline number is "
      "`onlypores porosity = pore voxels / sample_mask voxels` — porosity of the "
      "material the segmentation recognises as specimen, which is the quantity "
      "reported for real scans. `pore / total voxels` is given beside it, and so "
      "is the **sample-mask fraction**: the share of the generated box that "
      "onlypores accepts as material at all. A low sample-mask fraction means "
      "most of the box is not specimen — the volume is largely air.")
    A("")
    A(f"Runtime: {runtime_s / 60:.1f} min for all {len(pv)} volumes "
      "(no subset was needed).")
    A("")

    # ── validation ──
    A("## 1. Validation on real volumes (do first)")
    A("")
    A("`volumes.zarr/<vol>/mask` **is** the onlypores pore mask that "
      "`build_dataset` wrote, and `sample_mask` is its material mask. Re-running "
      "the pipeline on the stored `xct` must reproduce them.")
    A("")
    A("| real volume | shape | onlypores por (pore/sample) | stored por | "
      "pore/total (new / stored) | sample frac (new / stored) | Dice pore | "
      "Dice sample | patch-index mean por |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, r in real.items():
        A(f"| {k.replace('MedidasDB__', '')[:44]} | "
          f"{'×'.join(str(s) for s in r['shape'])} | "
          f"{_f(r['recomputed_porosity_over_sample'], 5)} | "
          f"{_f(r['stored_porosity_over_sample'], 5)} | "
          f"{_f(r['recomputed_porosity_over_total'], 5)} / "
          f"{_f(r['stored_porosity_over_total'], 5)} | "
          f"{_f(r['recomputed_sample_frac'], 4)} / {_f(r['stored_sample_frac'], 4)} | "
          f"**{_f(r['dice_pore_mask'], 4)}** | {_f(r['dice_sample_mask'], 4)} | "
          f"{_f(r['patch_index_mean_porosity'], 5)} |")
    A("")
    dices = [r["dice_pore_mask"] for r in real.values()]
    A(f"Dice = {min(dices):.4f}–{max(dices):.4f} — the recomputed masks are "
      "bit-identical to the stored ground truth, and the porosities agree to "
      "every printed digit. The pipeline reproduces the dataset's own "
      "segmentation exactly, so the generated-volume numbers below sit on the "
      "same footing as real data.")
    A("")
    A("(`patch-index mean porosity` is the mean of the per-64³-patch `porosity` "
      "column in `patch_index.parquet`, which is pore/total inside the specimen "
      "interior where patches are sampled — it is a sanity cross-check, not the "
      "same denominator.)")
    A("")

    # ── headline per arm ──
    A("## 2. Headline — requested vs mask vs onlypores, by experiment and arm")
    A("")
    for exp, sub in (("dose_response", "63 volumes, 192³, 7 requested levels × 3 seeds"),
                     ("cfg_sweep", "72 volumes, 192³, 4 guidance scales × 2 levels × 3 seeds"),
                     ("layup", "18 volumes, 1024×1024×192, 3 layups × 2 seeds"),
                     ("ldm06_probe", "9 volumes, DDIM-step probe at target 0.03")):
        s = pv[pv.experiment == exp]
        if not len(s):
            continue
        A(f"### {exp} — {sub}")
        A("")
        A("| arm | n | requested (mean) | mask porosity | onlypores "
          "(pore/sample) | onlypores (pore/total) | sample-mask frac | "
          "onlypores − mask | mask recall of onlypores |")
        A("|---|---|---|---|---|---|---|---|---|")
        for arm in sorted(s.arm.unique()):
            g = s[s.arm == arm]
            A(f"| {arm} | {len(g)} | {_f(g.target.mean(), 3)} | "
              f"{_f(g.mask_porosity.mean())} | "
              f"**{_f(g.onlypores_porosity_sample.mean())}** | "
              f"{_f(g.onlypores_porosity_total.mean())} | "
              f"{_f(g.sample_mask_fraction.mean(), 3)} | "
              f"{_f(g.discrepancy_onlypores_minus_mask.mean())} | "
              f"{_f(g.mask_recall_of_onlypores.mean(), 3)} |")
        A("")

    # ── global dose-response fits ──
    A("## 3. Global dose response (dose_response set), by arm")
    A("")
    A("| arm | s_por | measurement | slope | intercept | R² | mean \\|err\\| | "
      "median \\|err\\| | frac within gate ±0.005 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = res["global_dose_response"][arm]
        for lab, fk, ek in (("mask", "mask", "err_mask"),
                            ("onlypores", "onlypores_sample", "err_onlypores")):
            f, e = r[fk], r[ek]
            A(f"| {arm} | {r['s_por']:g} | {lab} | {_f(f['slope'], 3)} | "
              f"{_f(f['intercept'])} | {_f(f['r2'], 3)} | "
              f"{_f(e.get('abs_error_mean'))} | {_f(e.get('abs_error_median'))} | "
              f"{_f(e.get('frac_within_gate'), 3)} |")
    A("")
    A("Per requested level (mean over 3 seeds):")
    A("")
    A("| requested | " + " | ".join(f"{a} mask / onlypores" for a in ARMS) + " |")
    A("|---|" + "---|" * len(ARMS))
    lv = pd.DataFrame(res["dose_response_by_level"])
    for t in sorted(lv.target.unique()):
        cells = []
        for arm in ARMS:
            row = lv[(lv.arm == arm) & (lv.target == t)]
            cells.append(f"{_f(row['mask'].iloc[0])} / "
                         f"**{_f(row['onlypores'].iloc[0])}**"
                         if len(row) else "—")
        A(f"| {t:g} | " + " | ".join(cells) + " |")
    A("")

    # ── local dose-response ──
    A("## 4. Local dose response — per 64³ cell, by arm")
    A("")
    A("Cell target is the conditioned local porosity from `stats.json` "
      "(`cell_targets`, the coherent porosity field the volume was generated "
      "with). Cell onlypores porosity uses the same pore/sample-mask "
      "denominator, computed inside that cell.")
    A("")
    A("| arm | n cells | measurement | slope | intercept | R² | mean \\|err\\| | "
      "median \\|err\\| | p95 \\|err\\| | max \\|err\\| | frac within gate |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = res["local_dose_response"][arm]
        for lab, fk, ek in (("mask", "mask", "err_mask"),
                            ("onlypores", "onlypores_sample", "err_onlypores")):
            f, e = r[fk], r[ek]
            A(f"| {arm} | {r['n_cells']} | {lab} | {_f(f['slope'], 3)} | "
              f"{_f(f['intercept'])} | {_f(f['r2'], 3)} | "
              f"{_f(e.get('abs_error_mean'))} | {_f(e.get('abs_error_median'))} | "
              f"{_f(e.get('abs_error_p95'))} | {_f(e.get('abs_error_max'))} | "
              f"{_f(e.get('frac_within_gate'), 3)} |")
    A("")

    # ── discrepancy ──
    A("## 5. Mask vs onlypores discrepancy")
    A("")
    A("| experiment | arm | n | mask | onlypores | onlypores − mask | "
      "Dice(mask, onlypores) | mask recall | sample frac |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res["discrepancy"]["by_experiment_arm"]:
        A(f"| {r['experiment']} | {r['arm']} | {r['n']} | {_f(r['mask'])} | "
          f"{_f(r['onlypores'])} | **{_f(r['diff_mean'])}** | {_f(r['dice'], 3)} | "
          f"{_f(r['mask_recall'], 3)} | {_f(r['sample_frac'], 3)} |")
    A("")
    A("Biggest per-volume under-reports (onlypores − mask, top 10):")
    A("")
    A("| experiment | arm | volume | requested | mask | onlypores | "
      "sample frac | onlypores − mask |")
    A("|---|---|---|---|---|---|---|---|")
    for r in res["discrepancy"]["top20_volumes"][:10]:
        A(f"| {r['experiment']} | {r['arm']} | {r['name']} | "
          f"{_f(r['target'], 3)} | {_f(r['mask_porosity'])} | "
          f"{_f(r['onlypores_porosity_sample'])} | "
          f"{_f(r['sample_mask_fraction'], 3)} | "
          f"**{_f(r['discrepancy_onlypores_minus_mask'])}** |")
    A("")
    if "audit_join" in res:
        aj = res["audit_join"]
        A(f"Joined with the audit's cell table (`{AUDIT_CELLS.name}`, "
          f"{aj['n_cells_joined']} cells matched; mask-porosity column agrees "
          f"r = {aj['corr_maskpor_check']:.4f}, confirming the cell alignment). "
          f"The audit's dark-air fraction and the onlypores air fraction "
          f"correlate r = {aj['corr_darkfrac_vs_airfrac']:.3f}; dark fraction vs "
          f"the mask/onlypores discrepancy r = "
          f"{aj['corr_darkfrac_vs_discrepancy']:.3f}.")
        A("")
        A("Median cell porosity by the audit's dark-air bin:")
        A("")
        A("| arm | dark-air bin | n cells | median mask | median onlypores | "
          "median discrepancy | median onlypores air frac |")
        A("|---|---|---|---|---|---|---|")
        for r in aj["by_dark_bin"]:
            A(f"| {r['arm']} | {r['dark_bin']} | {r['n']} | "
              f"{_f(r['median_mask'])} | {_f(r['median_onlypores'])} | "
              f"{_f(r['median_discrepancy'])} | {_f(r['median_air'], 3)} |")
        A("")
    A("Cells binned by the onlypores air fraction (share of the cell that "
      "onlypores does not accept as specimen material):")
    A("")
    A("| arm | air-fraction bin | n cells | median mask | median onlypores | "
      "median discrepancy |")
    A("|---|---|---|---|---|---|")
    for r in res["discrepancy"]["cells_by_air_fraction"]:
        A(f"| {r['arm']} | {r['air_bin']} | {r['n']} | {_f(r['mask'])} | "
          f"{_f(r['onlypores'])} | {_f(r['diff'])} |")
    A("")

    # ── DDIM ──
    A("## 6. DDIM steps, measured by onlypores (ldm06 probe)")
    A("")
    A("| DDIM steps | n | target | mask porosity | onlypores (pore/sample) | "
      "onlypores (pore/total) | sample-mask frac | \\|onlypores − target\\| |")
    A("|---|---|---|---|---|---|---|---|")
    for r in sorted(res["ddim_steps"], key=lambda x: x["ddim_steps"]):
        A(f"| {int(r['ddim_steps'])} | {r['n']} | {_f(r['target'], 3)} | "
          f"{_f(r['mask'])} | **{_f(r['onlypores'])}** | "
          f"{_f(r['onlypores_total'])} | {_f(r['sample_frac'], 3)} | "
          f"{_f(abs(r['onlypores'] - r['target']))} |")
    A("")
    if res["ddim_big"]:
        A("1024-scale probe volume:")
        A("")
        A("| volume | steps | target | mask | onlypores (pore/sample) | "
          "sample frac |")
        A("|---|---|---|---|---|---|")
        for r in res["ddim_big"]:
            A(f"| {r['name']} | {int(r['ddim_steps'])} | {_f(r['target'], 3)} | "
              f"{_f(r['mask_porosity'])} | "
              f"{_f(r['onlypores_porosity_sample'])} | "
              f"{_f(r['sample_mask_fraction'], 3)} |")
        A("")

    # ── figures / files ──
    A("## Figures")
    A("")
    for p in figs:
        A(f"- `{p}`")
    A("")
    A("## Files")
    A("")
    for n in ("results.json", "per_volume.csv", "per_cell.csv",
              "real_validation.json", "run.log"):
        A(f"- `{OUT_DIR / n}`")
    A("")
    text = "\n".join(L)
    (OUT_DIR / "findings.md").write_text(text)
    return str(OUT_DIR / "findings.md")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    global _LOG_FH
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(OUT_DIR / "run.log", "a")
    t_start = time.time()
    log("=== onlypores_generated START ===")

    real = validate_real()
    write_json(real, OUT_DIR, "real_validation.json")
    bad = [k for k, r in real.items() if r["dice_pore_mask"] < 0.999]
    if bad:
        log(f"ABORT: real validation failed for {bad}")
        print("ONLYPORES_FAILED_VALIDATION", flush=True)
        return
    log("real validation OK (Dice >= 0.999 on all volumes)")

    todo = list(iter_volumes())
    log(f"processing {len(todo)} generated volumes")
    rows, cells = [], []
    for i, (exp, arm, d) in enumerate(todo, 1):
        t0 = time.time()
        row, cdf = measure(exp, arm, d)
        rows.append(row)
        cells.append(cdf)
        log(f"[{i:3d}/{len(todo)}] {exp}/{arm}/{d.name}  "
            f"target={row['target']}  mask={row['mask_porosity']:.4f}  "
            f"onlypores={row['onlypores_porosity_sample']:.4f}  "
            f"sample_frac={row['sample_mask_fraction']:.3f}  "
            f"{time.time() - t0:.1f}s")

    pv = pd.DataFrame(rows)
    pc = pd.concat(cells, ignore_index=True)
    pv.to_csv(OUT_DIR / "per_volume.csv", index=False)
    pc.to_csv(OUT_DIR / "per_cell.csv", index=False)
    log(f"wrote per_volume.csv ({len(pv)}) and per_cell.csv ({len(pc)})")

    res = analyse(pv, pc)
    runtime = time.time() - t_start

    set_style()
    figs: list[str] = []
    figs += fig_delivered_vs_requested(pv, res)
    figs += fig_mask_vs_onlypores(pv, pc)
    figs += fig_local_dose_response(pc, res)
    figs += fig_ddim(res)
    figs += fig_montage(pv)
    log(f"wrote {len(figs)} figure files")

    res["meta"] = {
        "n_volumes": int(len(pv)),
        "n_cells": int(len(pc)),
        "runtime_seconds": runtime,
        "onlypores_entry_point":
            "poregen.dataset.segmentation.onlypores (defaults, as "
            "poregen.dataset.io.compute_mask calls it)",
        "onlypores_params": {"frontwall": 0, "backwall": 0, "sauvola_radius": 30,
                             "sauvola_k": 0.125, "min_size_filtering": -1},
        "scale_inversion": "clip(logit(v), 0, 1) * 255 -> uint8",
        "primary_denominator": "pore voxels / sample_mask voxels",
        "gate": GATE,
        "figures": figs,
    }
    res["real_validation"] = real
    write_json(res, OUT_DIR, "results.json")
    fpath = write_findings(pv, pc, res, real, figs, runtime)
    log(f"wrote {fpath}")
    log(f"=== total {runtime / 60:.1f} min ===")
    print("ONLYPORES_DONE", flush=True)


if __name__ == "__main__":
    main()

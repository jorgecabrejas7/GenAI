"""Stage-by-stage inspection pack for ``onlypores`` on generated vs real volumes.

`runs/analysis/onlypores_generated/` reports that onlypores measures a porosity
on the generated volumes that does not track the requested target.  Before that
number can be read as a statement about the MODEL, the segmentation itself has
to be shown to work on synthetic grayscale.  This script produces the evidence
for that check — it draws no conclusion.

What it produces (all under ``runs/analysis/onlypores_inspection/``):

1. ``stages/`` — one PNG per (volume, z-slice) with every intermediate of the
   onlypores pipeline side by side:
       a raw decoder-native grayscale        f final onlypores pore mask
       b Sauvola binarisation (material)     g the model's own mask.tif
       c global-Otsu binarisation            h overlay of f and g on grayscale
       d material_mask BEFORE fill_voids     i audit dark-voxel detector (T_best)
       e material_mask AFTER fill_voids
   ``stacks/`` — the same stages as ImageJ hyperstacks (ZCYX) plus an RGB
   overlay stack (ZYXS), for Fiji.

2. ``fig_hist_real_vs_generated.*`` and ``fig_hist_small_multiples.*`` —
   intensity histograms with the Otsu threshold, the Sauvola local-threshold
   distribution, the material mode and the audit's T_best/T_cons marked.

3. ``sensitivity.csv`` + ``fig_sensitivity_*.*`` — onlypores knobs swept on
   three volumes: sauvola_radius x sauvola_k, and the material-mask
   thresholding approach.

The production functions in ``poregen.dataset.segmentation`` are NOT modified.
This module re-implements the same steps with the intermediates exposed, and
asserts on the first volume that its Sauvola output is bit-identical to
``segmentation.sauvola_thresholding`` and that its pore mask is bit-identical to
``segmentation.onlypores``.

Usage:
    python scripts/analysis/onlypores_inspection.py            # everything
    python scripts/analysis/onlypores_inspection.py --stages   # part 1 only
    python scripts/analysis/onlypores_inspection.py --hist
    python scripts/analysis/onlypores_inspection.py --sens
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import fill_voids
import numpy as np
import pandas as pd
import tifffile
import zarr
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.special import logit
from skimage import filters, measure
from skimage.filters import threshold_sauvola
from skimage.measure import regionprops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from poregen.dataset import segmentation as seg  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT_DIR = REPO / "runs" / "analysis" / "onlypores_inspection"
VOL_ROOT = REPO / "runs" / "eval_v2" / "volumes"
PROBE_ROOT = REPO / "runs" / "analysis" / "ldm06_probe" / "volumes"
ZARR_ROOT = REPO / "data" / "split_v2" / "volumes.zarr"

# onlypores production defaults, exactly as poregen.dataset.io.compute_mask calls it
SAUVOLA_RADIUS = 30
SAUVOLA_K = 0.125
MIN_SIZE_FILTERING = -1

# audit-calibrated dark-voxel detector (runs/eval_v2/audit/results.json)
T_BEST = 185
T_CONS = 178
MIN_CC = 300
REAL_MATERIAL_MODE = 209

N_JOBS = -1
DISP_MAX = 1024          # largest in-plane display window (voxels)

STAGE_KEYS = ["raw", "sauvola", "otsu", "prefill", "sample", "pore",
              "model_mask", "overlay", "detector"]
STAGE_TITLES = {
    "raw": "(a) raw grayscale (decoder-native u8)",
    "sauvola": "(b) Sauvola binary  (white = material)",
    "otsu": "(c) global Otsu binary  (white = above T)",
    "prefill": "(d) material_mask BEFORE fill_voids",
    "sample": "(e) material_mask AFTER fill_voids  = sample_mask",
    "pore": "(f) onlypores pore mask  = NOT(b) AND (e)",
    "model_mask": "(g) model mask.tif / stored dataset mask",
    "overlay": "(h) overlay: red = onlypores only, cyan = mask only, yellow = both",
    "detector": f"(i) audit dark detector  (u8 < {T_BEST}, CC >= {MIN_CC})",
}

_LOG_FH = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


# ---------------------------------------------------------------------------
# Volume selection
# ---------------------------------------------------------------------------

# Real volumes chosen from the stored zarr masks so that pore/sample_mask
# (the same denominator used for the generated volumes) lands near 0.02 / 0.05.
REAL_LOW = "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_07_1_volume_eq_aligned"
REAL_HIGH = "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_10_1_volume_eq_aligned"


def volume_specs() -> list[dict]:
    """The inspection set: 8 generated + 2 real + 2 real 192-cubed controls."""
    dr = VOL_ROOT / "dose_response"
    specs: list[dict] = []
    for t in ("0.005", "0.02", "0.05", "0.1"):
        specs.append({
            "vid": f"gen_dose_joint_oob_t{t}",
            "kind": "generated",
            "path": dr / "joint_oob" / f"target_{t}_seed_101",
            "label": f"generated - dose_response joint_oob, target {t}, seed 101",
            "target": float(t),
        })
    specs.append({
        "vid": "gen_dose_seq_t0.05",
        "kind": "generated",
        "path": dr / "seq" / "target_0.05_seed_101",
        "label": "generated - dose_response seq, target 0.05, seed 101",
        "target": 0.05,
    })
    for steps in ("50", "200"):
        specs.append({
            "vid": f"gen_probe_ddim{steps}",
            "kind": "generated",
            "path": PROBE_ROOT / f"{steps}_seed_101",
            "label": f"generated - ldm06_probe DDIM {steps} steps, target 0.03, seed 101",
            "target": 0.03,
        })
    specs.append({
        "vid": "gen_layup_joint_oob_A_1024",
        "kind": "generated",
        "path": VOL_ROOT / "layup" / "joint_oob" / "A_training_seed_101",
        "label": "generated - layup joint_oob A_training, 1024x1024x192, target 0.03, seed 101",
        "target": 0.03,
    })
    specs.append({
        "vid": "real_phi0.02_full",
        "kind": "real",
        "path": REAL_LOW,
        "label": "REAL - Nacho 07_1 (stored pore/sample ~0.020), full volume",
        "target": None,
    })
    specs.append({
        "vid": "real_phi0.05_full",
        "kind": "real",
        "path": REAL_HIGH,
        "label": "REAL - Nacho 10_1 (stored pore/sample ~0.053), full volume",
        "target": None,
    })
    # Control: the same real data cut to the generated box size, so that "192-cubed
    # box with a global Otsu" is separated from "synthetic grayscale".
    specs.append({
        "vid": "real_phi0.02_crop192",
        "kind": "real_crop",
        "path": REAL_LOW,
        "label": "REAL control - Nacho 07_1, interior 192-cubed crop (generated box size)",
        "target": None,
    })
    specs.append({
        "vid": "real_phi0.05_crop192",
        "kind": "real_crop",
        "path": REAL_HIGH,
        "label": "REAL control - Nacho 10_1, interior 192-cubed crop (generated box size)",
        "target": None,
    })
    return specs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def to_native_u8(vol_path: Path) -> np.ndarray:
    """Generated float [0,1] sigmoid-scale TIFF -> decoder-native u8, chunked.

    Identical to ``scripts/analysis/eval_v2_audit.py`` and
    ``scripts/analysis/onlypores_generated.py``.
    """
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


def raw_level_stats(vol_path: Path) -> dict:
    """Occupied grey levels of a generated TIFF BEFORE the logit inversion.

    ``_eval_v2.save_volume`` writes ``xct_u8.astype(float32) / 255``, so the
    saved float32 volume is already quantised to u8 steps on the sigmoid scale.
    """
    mm = tifffile.memmap(str(vol_path), mode="r")
    hist = np.zeros(256, np.int64)
    for z0 in range(0, mm.shape[0], 32):
        chunk = np.asarray(mm[z0:z0 + 32], np.float32)
        if chunk.max() > 1.5:
            chunk = chunk / 255.0
        hist += np.bincount(np.clip(np.rint(chunk * 255.0), 0, 255).astype(np.uint8).ravel(),
                            minlength=256)
    del mm
    occ = np.nonzero(hist)[0]
    return {"raw_u8_min": int(occ.min()), "raw_u8_max": int(occ.max()),
            "raw_levels_occupied": int(len(occ))}


def _interior_crop_192(grp) -> tuple[slice, slice, slice]:
    """A 192-cubed box centred in the specimen of a real zarr volume."""
    sm = grp["sample_mask"]
    nz, ny, nx = sm.shape
    zc, yc, xc = nz // 2, ny // 2, nx // 2
    half = 96
    z0 = max(0, min(nz - 192, zc - half))
    y0 = max(0, min(ny - 192, yc - half))
    x0 = max(0, min(nx - 192, xc - half))
    return slice(z0, z0 + 192), slice(y0, y0 + 192), slice(x0, x0 + 192)


def load_entry(spec: dict) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """-> (native u8 volume, reference mask (model / dataset) or None, meta)."""
    meta: dict = {}
    if spec["kind"] == "generated":
        d = Path(spec["path"])
        native = to_native_u8(d / "volume.tif")
        ref = tifffile.imread(str(d / "mask.tif")) > 0
        sp = d / "stats.json"
        if sp.exists():
            st = json.loads(sp.read_text())
            meta["stats_target"] = st.get("target")
            meta["stats_delivered_mask_porosity"] = st.get("delivered_mask_porosity")
            meta["ddim_steps"] = st.get("ddim_steps")
        meta["ref_mask_source"] = "model mask.tif"
        meta.update(raw_level_stats(d / "volume.tif"))
        return native, ref, meta

    g = zarr.open(str(ZARR_ROOT), mode="r")
    grp = g[spec["path"]]
    if spec["kind"] == "real":
        native = np.asarray(grp["xct"])
        ref = np.asarray(grp["mask"]) > 0
    else:                                    # real_crop
        sz, sy, sx = _interior_crop_192(grp)
        native = np.asarray(grp["xct"][sz, sy, sx])
        ref = np.asarray(grp["mask"][sz, sy, sx]) > 0
        meta["crop_origin_zyx"] = [int(sz.start), int(sy.start), int(sx.start)]
    meta["ref_mask_source"] = "dataset mask (volumes.zarr/<vol>/mask)"
    return native, ref, meta


# ---------------------------------------------------------------------------
# Debug re-implementation of the onlypores pipeline (intermediates exposed)
# ---------------------------------------------------------------------------

def sauvola_debug(volume: np.ndarray, window_size: int,
                  k: float) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``segmentation.sauvola_thresholding_concurrent`` that also
    returns a 0-255 histogram of every local threshold in the volume.

    Returns (binary [material=True], thr_hist (256,) int64).
    """
    if window_size % 2 == 0:
        window_size += 1

    def _plane(i):
        s = volume[:, i, :]
        t = threshold_sauvola(s, window_size=window_size, k=k, r=128)
        h = np.bincount(np.clip(np.rint(t), 0, 255).astype(np.uint8).ravel(),
                        minlength=256)
        return (s > t), h

    res = Parallel(n_jobs=N_JOBS)(delayed(_plane)(i) for i in range(volume.shape[1]))
    binary = np.transpose(np.array([r[0] for r in res]), (1, 0, 2))
    thr_hist = np.sum(np.stack([r[1] for r in res]), axis=0)
    return binary, thr_hist


def material_threshold(cropped: np.ndarray, approach: str) -> float:
    """Global material threshold for the requested approach."""
    if approach == "otsu":
        return float(filters.threshold_otsu(cropped))
    if approach == "li":
        return float(filters.threshold_li(cropped))
    if approach == "yen":
        return float(filters.threshold_yen(cropped))
    if approach == "triangle":
        return float(filters.threshold_triangle(cropped))
    if approach == "fixed_t_best":
        return float(T_BEST)
    if approach == "fixed_t_cons":
        return float(T_CONS)
    if approach == "mode_minus_24":
        # material mode of this volume, offset by the real-calibrated
        # (real mode 209 - T_best 185) = 24 u8 gap.
        h = np.bincount(cropped.ravel(), minlength=256)
        mode = int(np.argmax(h[100:]) + 100)
        return float(mode - 24)
    raise ValueError(approach)


def material_mask_debug(cropped: np.ndarray, approach: str = "otsu",
                        component: str = "first",
                        bbox_crop: bool = True) -> dict:
    """Mirror of ``segmentation.material_mask`` with intermediates exposed.

    ``approach='otsu', component='first', bbox_crop=True`` reproduces production
    exactly, including its use of ``regionprops(...)[0]`` — the FIRST label of
    the max-projection, not the largest component.
    """
    t = material_threshold(cropped, approach)
    binary = cropped > t

    max_proj = np.max(binary, axis=0)
    labels = measure.label(max_proj)
    props = regionprops(labels)

    info = {
        "threshold": t,
        "approach": approach,
        "component": component,
        "n_maxproj_components": len(props),
    }

    if props:
        areas = np.array([p.area for p in props], dtype=np.int64)
        i_first, i_largest = 0, int(np.argmax(areas))
        info.update({
            "maxproj_area_first": int(areas[0]),
            "maxproj_area_largest": int(areas[i_largest]),
            "maxproj_area_total": int(areas.sum()),
            "maxproj_first_is_largest": bool(i_first == i_largest),
            "maxproj_bbox_first": [int(v) for v in props[0].bbox],
            "maxproj_bbox_largest": [int(v) for v in props[i_largest].bbox],
        })
        p = props[i_first if component == "first" else i_largest]
        minr, minc, maxr, maxc = p.bbox
    else:
        info.update({"maxproj_area_first": 0, "maxproj_area_largest": 0,
                     "maxproj_area_total": 0, "maxproj_first_is_largest": True,
                     "maxproj_bbox_first": None, "maxproj_bbox_largest": None})
        minr = minc = 0
        maxr, maxc = binary.shape[1], binary.shape[2]

    if not props:
        # production falls back to the raw threshold mask, with no void filling
        info["bbox_used"] = None
        return {"info": info, "otsu_binary": binary,
                "prefill": binary.copy(), "sample": binary.copy()}
    if not bbox_crop:
        minr, minc = 0, 0
        maxr, maxc = binary.shape[1], binary.shape[2]

    prefill = np.zeros_like(binary)
    prefill[:, minr:maxr, minc:maxc] = binary[:, minr:maxr, minc:maxc]
    filled = fill_voids.fill(binary[:, minr:maxr, minc:maxc], in_place=False)
    sample = np.zeros_like(binary)
    sample[:, minr:maxr, minc:maxc] = filled

    info["bbox_used"] = [int(minr), int(minc), int(maxr), int(maxc)]
    return {"info": info, "otsu_binary": binary, "prefill": prefill, "sample": sample}


def detector_mask(native: np.ndarray, t: int = T_BEST,
                  min_cc: int = MIN_CC) -> np.ndarray:
    """Audit dark-voxel detector: u8 < t, connected components >= min_cc."""
    det = native < t
    labels, n = ndimage.label(det)
    if n:
        sizes = np.bincount(labels.ravel())
        small = sizes < min_cc
        small[0] = False
        det[small[labels]] = False
    del labels
    return det


# ---------------------------------------------------------------------------
# Part 1 — stage-by-stage visuals
# ---------------------------------------------------------------------------

def _disp_window(shape_yx: tuple[int, int], centre_yx: tuple[int, int]) -> tuple[slice, slice]:
    ny, nx = shape_yx
    h, w = min(DISP_MAX, ny), min(DISP_MAX, nx)
    y0 = int(np.clip(centre_yx[0] - h // 2, 0, ny - h))
    x0 = int(np.clip(centre_yx[1] - w // 2, 0, nx - w))
    return slice(y0, y0 + h), slice(x0, x0 + w)


def _frac(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else float("nan")


def process_volume(spec: dict, verify: bool = False) -> dict:
    """Run the debug pipeline, write PNGs + TIFF stacks, return the record."""
    t0 = time.time()
    native, ref_mask, meta = load_entry(spec)
    vid = spec["vid"]
    log(f"{vid}: shape {native.shape} ({native.size/1e6:.1f} Mvox)")

    bbox = seg.content_bbox(native)
    if bbox is None:
        raise RuntimeError(f"{vid}: empty volume")
    z0, z1, y0, y1, x0, x1 = bbox
    cropped = native[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    dz, dy, dx = cropped.shape

    # display z rows, in CROPPED coordinates: mid-volume + two off-centre
    z_rows = [int(dz * 0.25), int(dz * 0.5), int(dz * 0.75)]
    z_rows = sorted({max(0, min(dz - 1, z)) for z in z_rows})

    # ── (b) Sauvola ──
    binary, thr_hist = sauvola_debug(cropped, SAUVOLA_RADIUS, SAUVOLA_K)
    if verify:
        ref_bin = seg.sauvola_thresholding(cropped, window_size=SAUVOLA_RADIUS, k=SAUVOLA_K)
        meta["sauvola_matches_production"] = bool(np.array_equal(binary, ref_bin))
        del ref_bin

    # ── (c)(d)(e) material mask ──
    mm = material_mask_debug(cropped, "otsu", "first", True)
    otsu_binary, prefill, sample = mm["otsu_binary"], mm["prefill"], mm["sample"]

    # ── (f) pores ──
    pore = np.logical_and(~binary, sample)

    if verify:
        p_ref, s_ref, b_ref = seg.onlypores(native)
        meta["pore_matches_production"] = bool(np.array_equal(
            p_ref[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], pore))
        meta["sample_matches_production"] = bool(np.array_equal(
            s_ref[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], sample))
        del p_ref, s_ref, b_ref

    # ── (g) reference mask, cropped to the same box ──
    ref_c = (ref_mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
             if ref_mask is not None else np.zeros_like(pore))

    # ── (i) audit detector ──
    det = detector_mask(cropped, T_BEST, MIN_CC)

    # ── volume-level numbers ──
    n_sm = int(sample.sum())
    n_pore = int(pore.sum())
    n_ref = int(ref_c.sum())
    inter = int(np.count_nonzero(pore & ref_c))
    hist_all = np.bincount(cropped.ravel(), minlength=256).astype(np.int64)
    hist_sample = np.bincount(cropped[sample].ravel(), minlength=256).astype(np.int64)
    mat_mode = int(np.argmax(hist_sample[100:]) + 100) if n_sm else -1
    occ = np.nonzero(hist_all)[0]
    n_levels_occupied = int(len(occ))
    level_span = int(occ.max() - occ.min() + 1)

    rec = {
        "vid": vid,
        "kind": spec["kind"],
        "label": spec["label"],
        "path": str(spec["path"]),
        "target": spec["target"],
        "shape": "x".join(str(s) for s in native.shape),
        "n_voxels": int(native.size),
        "content_bbox": [int(v) for v in bbox],
        "cropped_shape": "x".join(str(s) for s in cropped.shape),
        "z_rows_cropped": z_rows,
        "z_rows_original": [int(z + z0) for z in z_rows],
        "otsu_threshold": mm["info"]["threshold"],
        "material_mode_u8": mat_mode,
        "native_levels_occupied": n_levels_occupied,
        "native_level_span": level_span,
        "native_level_fill": n_levels_occupied / max(level_span, 1),
        "sauvola_thr_mean": float((np.arange(256) * thr_hist).sum() / max(thr_hist.sum(), 1)),
        "sauvola_thr_p05": float(np.searchsorted(np.cumsum(thr_hist), 0.05 * thr_hist.sum())),
        "sauvola_thr_p50": float(np.searchsorted(np.cumsum(thr_hist), 0.50 * thr_hist.sum())),
        "sauvola_thr_p95": float(np.searchsorted(np.cumsum(thr_hist), 0.95 * thr_hist.sum())),
        "sample_mask_fraction": n_sm / cropped.size,
        "prefill_fraction": _frac(prefill),
        "otsu_above_fraction": _frac(otsu_binary),
        "sauvola_material_fraction": _frac(binary),
        "onlypores_porosity_sample": n_pore / n_sm if n_sm else float("nan"),
        "onlypores_porosity_total": n_pore / cropped.size,
        "ref_mask_fraction": n_ref / cropped.size,
        "ref_mask_porosity_in_sample": (int(np.count_nonzero(ref_c & sample)) / n_sm
                                        if n_sm else float("nan")),
        "detector_fraction_total": _frac(det),
        "detector_porosity_sample": (int(np.count_nonzero(det & sample)) / n_sm
                                     if n_sm else float("nan")),
        "dice_onlypores_vs_ref": (2.0 * inter / (n_pore + n_ref)
                                  if (n_pore + n_ref) else float("nan")),
        "ref_recall_of_onlypores": inter / n_pore if n_pore else float("nan"),
        "ref_precision_vs_onlypores": inter / n_ref if n_ref else float("nan"),
        "seconds": round(time.time() - t0, 1),
    }
    for key in ("n_maxproj_components", "maxproj_area_first", "maxproj_area_largest",
                "maxproj_area_total", "maxproj_first_is_largest",
                "maxproj_bbox_first", "maxproj_bbox_largest", "bbox_used"):
        rec[key] = mm["info"].get(key)
    rec.update(meta)

    # ── display window: centred on the material bbox of the max-projection ──
    bb = mm["info"]["maxproj_bbox_largest"] or [0, 0, dy, dx]
    centre = ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
    sy, sx = _disp_window((dy, dx), centre)
    rec["display_window_yx_in_crop"] = [int(sy.start), int(sx.start),
                                        int(sy.stop - sy.start), int(sx.stop - sx.start)]

    # ── collect the display slices ──
    stages_u8 = np.zeros((len(z_rows), len(STAGE_KEYS) - 1,
                          sy.stop - sy.start, sx.stop - sx.start), np.uint8)
    overlay_rgb = np.zeros((len(z_rows), sy.stop - sy.start,
                            sx.stop - sx.start, 3), np.uint8)
    panels: list[dict] = []
    for i, z in enumerate(z_rows):
        g_ = cropped[z, sy, sx]
        b_ = binary[z, sy, sx]
        o_ = otsu_binary[z, sy, sx]
        pf = prefill[z, sy, sx]
        sm_ = sample[z, sy, sx]
        po = pore[z, sy, sx]
        rf = ref_c[z, sy, sx]
        dt = det[z, sy, sx]
        n_sm_s = int(sm_.sum())
        inter_s = int(np.count_nonzero(po & rf))
        p = {
            "z_cropped": int(z),
            "z_original": int(z + z0),
            "raw": g_, "sauvola": b_, "otsu": o_, "prefill": pf,
            "sample": sm_, "pore": po, "model_mask": rf, "detector": dt,
            "stats": {
                "raw_mean_u8": float(g_.mean()),
                "raw_median_u8": float(np.median(g_)),
                "sauvola_nonmaterial_frac": 1.0 - _frac(b_),
                "otsu_below_frac": 1.0 - _frac(o_),
                "prefill_frac": _frac(pf),
                "sample_frac": _frac(sm_),
                "pore_over_sample": (int(po.sum()) / n_sm_s) if n_sm_s else float("nan"),
                "pore_over_total": _frac(po),
                "ref_frac": _frac(rf),
                "ref_over_sample": (int(np.count_nonzero(rf & sm_)) / n_sm_s)
                                   if n_sm_s else float("nan"),
                "detector_over_sample": (int(np.count_nonzero(dt & sm_)) / n_sm_s)
                                        if n_sm_s else float("nan"),
                "dice_slice": (2.0 * inter_s / (int(po.sum()) + int(rf.sum()))
                               if (po.sum() + rf.sum()) else float("nan")),
            },
        }
        panels.append(p)
        for j, key in enumerate(["raw", "sauvola", "otsu", "prefill", "sample",
                                 "pore", "model_mask", "detector"]):
            arr = p[key]
            stages_u8[i, j] = arr if key == "raw" else (arr.astype(np.uint8) * 255)
        overlay_rgb[i] = _overlay_rgb(g_, po, rf)

    del binary, otsu_binary, prefill, sample, pore, det, ref_c, cropped, native
    if ref_mask is not None:
        del ref_mask

    # ── write PNGs + stacks ──
    (OUT_DIR / "stages").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stacks").mkdir(parents=True, exist_ok=True)
    for p in panels:
        _panel_png(spec, rec, p)
    tifffile.imwrite(
        OUT_DIR / "stacks" / f"{vid}__stages.tif", stages_u8, imagej=True,
        metadata={"axes": "ZCYX",
                  "Labels": [STAGE_TITLES[k] for k in
                             ["raw", "sauvola", "otsu", "prefill", "sample",
                              "pore", "model_mask", "detector"]] * len(z_rows)},
    )
    tifffile.imwrite(OUT_DIR / "stacks" / f"{vid}__overlay_rgb.tif", overlay_rgb,
                     imagej=True, metadata={"axes": "ZYXS"})

    rec["panel_stats"] = [{"z_original": p["z_original"], **p["stats"]} for p in panels]
    hist = {"hist_all": hist_all, "hist_sample": hist_sample, "thr_hist": thr_hist}
    log(f"  {vid}: sample_frac={rec['sample_mask_fraction']:.3f} "
        f"otsu_T={rec['otsu_threshold']:.1f} mode={mat_mode} "
        f"por/sample={rec['onlypores_porosity_sample']:.4f} "
        f"maxproj_comps={rec['n_maxproj_components']} "
        f"first_is_largest={rec['maxproj_first_is_largest']} {rec['seconds']}s")
    return {"record": rec, "hist": hist}


def _overlay_rgb(gray: np.ndarray, pore: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Grayscale with onlypores (red), reference mask (cyan), both (yellow)."""
    base = (gray.astype(np.float32) / 255.0) * 0.75
    rgb = np.stack([base] * 3, -1)
    only_p = pore & ~ref
    only_r = ref & ~pore
    both = pore & ref
    rgb[only_p] = [1.00, 0.12, 0.20]
    rgb[only_r] = [0.00, 0.78, 0.95]
    rgb[both] = [1.00, 0.90, 0.10]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _panel_png(spec: dict, rec: dict, p: dict) -> None:
    s = p["stats"]
    imgs = {
        "raw": (p["raw"], f"mean {s['raw_mean_u8']:.1f} u8, median {s['raw_median_u8']:.0f} u8"),
        "sauvola": (p["sauvola"], f"below local T: {s['sauvola_nonmaterial_frac']:.4f} of slice"),
        "otsu": (p["otsu"], f"below Otsu T={rec['otsu_threshold']:.0f}: {s['otsu_below_frac']:.4f} of slice"),
        "prefill": (p["prefill"], f"material frac {s['prefill_frac']:.4f}"),
        "sample": (p["sample"], f"sample frac {s['sample_frac']:.4f}"),
        "pore": (p["pore"], f"pore/sample {s['pore_over_sample']:.4f}  |  pore/total {s['pore_over_total']:.4f}"),
        "model_mask": (p["model_mask"], f"mask/total {s['ref_frac']:.4f}  |  mask/sample {s['ref_over_sample']:.4f}"),
        "overlay": (None, f"slice Dice {s['dice_slice']:.3f}"),
        "detector": (p["detector"], f"detector/sample {s['detector_over_sample']:.4f}"),
    }
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.6))
    for ax, key in zip(axes.ravel(), STAGE_KEYS):
        arr, sub = imgs[key]
        if key == "overlay":
            ax.imshow(_overlay_rgb(p["raw"], p["pore"], p["model_mask"]),
                      interpolation="nearest")
            ax.legend(handles=[
                Patch(facecolor="#ff1f33", label="onlypores only"),
                Patch(facecolor="#00c7f2", label="reference mask only"),
                Patch(facecolor="#ffe61a", label="both"),
            ], loc="upper right", fontsize=7, framealpha=0.85)
        elif key == "raw":
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(arr.astype(np.uint8), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
        ax.set_title(f"{STAGE_TITLES[key]}\n{sub}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    yw, xw, hh, ww = rec["display_window_yx_in_crop"]
    fig.suptitle(
        f"{rec['vid']}   —   {rec['label']}\n"
        f"z = {p['z_original']} (original) / {p['z_cropped']} (content crop) ; "
        f"display window {hh}x{ww} at (y={yw}, x={xw}) of the {rec['cropped_shape']} content crop\n"
        f"VOLUME: sample-mask frac {rec['sample_mask_fraction']:.3f} | "
        f"onlypores pore/sample {rec['onlypores_porosity_sample']:.4f} | "
        f"reference mask/total {rec['ref_mask_fraction']:.4f} | "
        f"Otsu T {rec['otsu_threshold']:.0f} u8 | material mode {rec['material_mode_u8']} u8"
        + (f" | requested target {rec['target']}" if rec["target"] is not None else ""),
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = OUT_DIR / "stages" / f"{rec['vid']}__z{p['z_original']:04d}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2 — histogram diagnostics
# ---------------------------------------------------------------------------

def fig_hist_overlay(records: list[dict], hists: dict) -> None:
    set_style()
    lv = np.arange(256)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    groups = {
        "REAL (full volume)": ([r for r in records if r["kind"] == "real"], "#1b6ca8", "-"),
        "REAL (192-cubed control)": ([r for r in records if r["kind"] == "real_crop"], "#2e7d32", "--"),
        "GENERATED": ([r for r in records if r["kind"] == "generated"], "#c2571a", "-"),
    }
    for ax, which, ttl in ((axes[0], "hist_all", "all voxels of the content crop"),
                           (axes[1], "hist_sample", "voxels inside the onlypores sample_mask")):
        for gname, (recs, col, ls) in groups.items():
            for j, r in enumerate(recs):
                h = hists[r["vid"]][which].astype(np.float64)
                h = h / max(h.sum(), 1)
                ax.plot(lv, np.where(h > 0, h, np.nan), color=col, ls=ls,
                        alpha=0.8, lw=1.1, marker="." if r["kind"] == "generated" else None,
                        ms=2.5, label=gname if j == 0 else None)
        for t, c, lab in ((T_BEST, "#6a3d9a", f"audit T_best = {T_BEST}"),
                          (T_CONS, "#6a3d9a", f"audit T_cons = {T_CONS}"),
                          (REAL_MATERIAL_MODE, "#444444", f"real material mode = {REAL_MATERIAL_MODE}")):
            ax.axvline(t, color=c, lw=1.1,
                       ls=":" if t == T_CONS else "-.", alpha=0.9, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.0)
        # Otsu thresholds as markers on the top edge, not as full-height lines
        for gname, (recs, col, _ls) in groups.items():
            ts = [r["otsu_threshold"] for r in recs]
            if ts:
                ax.plot(ts, [0.55] * len(ts), marker="v", ls="none", ms=8,
                        color=col, mec="black", mew=0.5,
                        label=f"global Otsu T — {gname}")
        ax.set_ylabel("fraction of voxels (log)")
        ax.set_title(f"Intensity histogram — {ttl}   "
                     "(triangles on the top edge = each volume's global Otsu threshold; "
                     "gaps in the orange curves are empty u8 levels)", fontsize=9.5)
        ax.legend(fontsize=7.6, ncol=3, loc="lower right")
    axes[1].set_xlabel("decoder-native intensity (u8)")
    axes[1].set_xlim(0, 255)
    gen = [r for r in records if r["kind"] == "generated"]
    rl = [r for r in records if r["kind"] != "generated"]
    fig.suptitle(
        "Real vs generated intensity statistics on the scale onlypores sees\n"
        f"occupied u8 levels: generated {min(r['native_levels_occupied'] for r in gen)}"
        f"-{max(r['native_levels_occupied'] for r in gen)} of 256   |   "
        f"real {min(r['native_levels_occupied'] for r in rl)}"
        f"-{max(r['native_levels_occupied'] for r in rl)} of 256   "
        f"(saved generated TIFFs hold u8 levels "
        f"{min(r.get('raw_u8_min', 0) for r in gen)}-"
        f"{max(r.get('raw_u8_max', 255) for r in gen)} on the sigmoid scale "
        "before the logit inversion)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, OUT_DIR, "fig_hist_real_vs_generated")


def fig_hist_small_multiples(records: list[dict], hists: dict) -> None:
    set_style()
    lv = np.arange(256)
    n = len(records)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.3 * nrow),
                             sharex=True)
    for ax, r in zip(axes.ravel(), records):
        h = hists[r["vid"]]
        a = h["hist_all"].astype(np.float64); a /= max(a.sum(), 1)
        s = h["hist_sample"].astype(np.float64); s /= max(s.sum(), 1)
        t = h["thr_hist"].astype(np.float64); t /= max(t.sum(), 1)
        ax.fill_between(lv, a, color="#9ecae1", alpha=0.65, label="all voxels")
        ax.plot(lv, s, color="#1b6ca8", lw=1.2, label="inside sample_mask")
        ax.plot(lv, t, color="#c2571a", lw=1.2, ls="--", label="Sauvola local T")
        ax.axvline(r["otsu_threshold"], color="#000000", lw=1.1,
                   label=f"Otsu T = {r['otsu_threshold']:.0f}")
        ax.axvline(r["material_mode_u8"], color="#2e7d32", lw=1.0, ls="-.",
                   label=f"material mode = {r['material_mode_u8']}")
        ax.axvline(T_BEST, color="#6a3d9a", lw=0.9, ls=":", label=f"T_best = {T_BEST}")
        ax.axvline(T_CONS, color="#6a3d9a", lw=0.9, ls=":", alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1)
        ax.set_title(f"{r['vid']}\nsample frac {r['sample_mask_fraction']:.3f} | "
                     f"pore/sample {r['onlypores_porosity_sample']:.4f} | "
                     f"{r['native_levels_occupied']} occupied u8 levels", fontsize=8.5)
        ax.legend(fontsize=6.2, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("decoder-native u8")
    fig.suptitle("Per-volume histograms with the thresholds onlypores actually uses", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    savefig(fig, OUT_DIR, "fig_hist_small_multiples")


# ---------------------------------------------------------------------------
# Part 3 — parameter sensitivity
# ---------------------------------------------------------------------------

SENS_RADII = [15, 30, 60]
SENS_KS = [0.05, 0.125, 0.25]
SENS_MATERIAL = [
    ("otsu / first component / bbox   [PRODUCTION]", "otsu", "first", True),
    ("otsu / largest component / bbox", "otsu", "largest", True),
    ("otsu / no bbox crop", "otsu", "first", False),
    ("li / first / bbox", "li", "first", True),
    ("yen / first / bbox", "yen", "first", True),
    ("triangle / first / bbox", "triangle", "first", True),
    (f"fixed T_best={T_BEST} / first / bbox", "fixed_t_best", "first", True),
    (f"fixed T_cons={T_CONS} / first / bbox", "fixed_t_cons", "first", True),
    ("material mode - 24 u8 / first / bbox", "mode_minus_24", "first", True),
]


def sensitivity(specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        vid = spec["vid"]
        native, _ref, _m = load_entry(spec)
        bbox = seg.content_bbox(native)
        z0, z1, y0, y1, x0, x1 = bbox
        cropped = np.ascontiguousarray(native[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])
        del native

        # material-mask approach sweep (Sauvola at production defaults)
        base_bin, _ = sauvola_debug(cropped, SAUVOLA_RADIUS, SAUVOLA_K)
        for label, approach, comp, bbc in SENS_MATERIAL:
            t1 = time.time()
            mm = material_mask_debug(cropped, approach, comp, bbc)
            sm = mm["sample"]
            pore = np.logical_and(~base_bin, sm)
            n_sm = int(sm.sum())
            rows.append({
                "vid": vid, "sweep": "material", "label": label,
                "approach": approach, "component": comp, "bbox_crop": bbc,
                "sauvola_radius": SAUVOLA_RADIUS, "sauvola_k": SAUVOLA_K,
                "threshold": mm["info"]["threshold"],
                "sample_mask_fraction": n_sm / cropped.size,
                "porosity_sample": int(pore.sum()) / n_sm if n_sm else float("nan"),
                "porosity_total": int(pore.sum()) / cropped.size,
                "n_maxproj_components": mm["info"]["n_maxproj_components"],
                "maxproj_first_is_largest": mm["info"]["maxproj_first_is_largest"],
                "seconds": round(time.time() - t1, 1),
            })
            log(f"  {vid} material[{label}] sample_frac="
                f"{rows[-1]['sample_mask_fraction']:.3f} "
                f"por/sample={rows[-1]['porosity_sample']:.4f}")
            del mm, sm, pore
        del base_bin

        # Sauvola knob sweep (material mask = production)
        mm = material_mask_debug(cropped, "otsu", "first", True)
        sm = mm["sample"]
        n_sm = int(sm.sum())
        del mm
        for radius in SENS_RADII:
            for k in SENS_KS:
                t1 = time.time()
                b, thr_hist = sauvola_debug(cropped, radius, k)
                pore = np.logical_and(~b, sm)
                rows.append({
                    "vid": vid, "sweep": "sauvola",
                    "label": f"radius {radius}, k {k}",
                    "approach": "otsu", "component": "first", "bbox_crop": True,
                    "sauvola_radius": radius, "sauvola_k": k,
                    "threshold": float((np.arange(256) * thr_hist).sum()
                                       / max(thr_hist.sum(), 1)),
                    "sample_mask_fraction": n_sm / cropped.size,
                    "porosity_sample": int(pore.sum()) / n_sm if n_sm else float("nan"),
                    "porosity_total": int(pore.sum()) / cropped.size,
                    "n_maxproj_components": None,
                    "maxproj_first_is_largest": None,
                    "seconds": round(time.time() - t1, 1),
                })
                log(f"  {vid} sauvola[r={radius}, k={k}] "
                    f"por/sample={rows[-1]['porosity_sample']:.4f} "
                    f"({rows[-1]['seconds']}s)")
                del b, pore
        del sm, cropped
    return pd.DataFrame(rows)


def fig_sensitivity(df: pd.DataFrame, records_by_vid: dict) -> None:
    set_style()
    vids = list(dict.fromkeys(df.vid))
    sv = df[df.sweep == "sauvola"]
    fig, axes = plt.subplots(1, len(vids), figsize=(4.9 * len(vids), 4.4))
    axes = np.atleast_1d(axes)
    for ax, vid in zip(axes, vids):
        s = sv[sv.vid == vid]
        grid = np.full((len(SENS_RADII), len(SENS_KS)), np.nan)
        for i, r in enumerate(SENS_RADII):
            for j, k in enumerate(SENS_KS):
                m = s[(s.sauvola_radius == r) & (s.sauvola_k == k)]
                if len(m):
                    grid[i, j] = m.porosity_sample.iloc[0]
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(len(SENS_RADII)):
            for j in range(len(SENS_KS)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v < np.nanmean(grid) else "black")
        ax.set_xticks(range(len(SENS_KS)), [str(k) for k in SENS_KS])
        ax.set_yticks(range(len(SENS_RADII)), [str(r) for r in SENS_RADII])
        ax.set_xlabel("sauvola_k"); ax.set_ylabel("sauvola_radius")
        tgt = records_by_vid.get(vid, {}).get("target")
        ax.set_title(f"{vid}\npore/sample_mask"
                     + (f"  (requested {tgt})" if tgt is not None else ""), fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Sauvola sensitivity — production default is radius 30, k 0.125 "
                 "(sample_mask does not depend on these knobs)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, OUT_DIR, "fig_sensitivity_sauvola")

    mv = df[df.sweep == "material"]
    labels = [l for l, *_ in SENS_MATERIAL]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.42 * len(labels) * len(vids) + 2.6))
    y = np.arange(len(labels))
    width = 0.8 / len(vids)
    cols = ["#1b6ca8", "#c2571a", "#6a3d9a", "#2e7d32"]
    for ax, col, ttl in ((axes[0], "sample_mask_fraction", "sample_mask fraction"),
                         (axes[1], "porosity_sample", "pore / sample_mask")):
        for i, vid in enumerate(vids):
            s = mv[mv.vid == vid].set_index("label").reindex(labels)
            ax.barh(y + i * width, s[col].to_numpy(), height=width,
                    color=cols[i % len(cols)], label=vid)
        ax.set_yticks(y + width * (len(vids) - 1) / 2, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(ttl)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7.5)
    fig.suptitle("Material-mask thresholding approach — Sauvola held at the production default",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, OUT_DIR, "fig_sensitivity_material")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings_md(records: list[dict], sens: pd.DataFrame | None) -> None:
    L: list[str] = []
    L.append("# onlypores inspection pack — measurements\n")
    L.append("Evidence only. No claim is made here about whether the generator or the "
             "segmentation is responsible for the porosity-control result in "
             "`runs/analysis/onlypores_generated/`.\n")
    L.append("## 1. Per-volume stage numbers\n")
    L.append("| volume | kind | requested | Otsu T | material mode | max-proj comps | "
             "first==largest | Otsu-above frac | prefill frac | sample frac | "
             "Sauvola material frac | pore/sample | pore/total | ref mask/total | "
             "detector/sample | Dice(onlypores, ref) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        L.append(
            f"| {r['vid']} | {r['kind']} | "
            f"{'-' if r['target'] is None else r['target']} | "
            f"{r['otsu_threshold']:.0f} | {r['material_mode_u8']} | "
            f"{r['n_maxproj_components']} | {r['maxproj_first_is_largest']} | "
            f"{r['otsu_above_fraction']:.4f} | {r['prefill_fraction']:.4f} | "
            f"{r['sample_mask_fraction']:.4f} | {r['sauvola_material_fraction']:.4f} | "
            f"{r['onlypores_porosity_sample']:.4f} | {r['onlypores_porosity_total']:.4f} | "
            f"{r['ref_mask_fraction']:.4f} | {r['detector_porosity_sample']:.4f} | "
            f"{r['dice_onlypores_vs_ref']:.3f} |")
    L.append("")
    L.append("`Otsu-above frac` is the fraction of the content crop above the global "
             "threshold; `prefill frac` is that after the max-projection bounding-box "
             "crop and `sample frac` is it again after `fill_voids`. When the three are "
             "equal the bounding box and the void filling changed nothing, and the "
             "material mask IS the global threshold. `Sauvola material frac` is the "
             "fraction above the LOCAL threshold. The final pore mask is "
             "`NOT(Sauvola) AND sample_mask`, so any voxel the global threshold puts "
             "outside the material can never be counted as a pore.\n")

    L.append("## 2. Histogram summary\n")
    L.append("| volume | kind | Otsu T | material mode | mode - Otsu T | "
             "Sauvola local T: p05 / p50 / p95 | mean u8 in sample | "
             "occupied u8 levels | saved u8 range (sigmoid scale) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in records:
        rng = ("-" if r.get("raw_u8_min") is None
               else f"{r['raw_u8_min']}-{r['raw_u8_max']} ({r['raw_levels_occupied']} levels)")
        L.append(f"| {r['vid']} | {r['kind']} | {r['otsu_threshold']:.0f} | "
                 f"{r['material_mode_u8']} | "
                 f"{r['material_mode_u8'] - r['otsu_threshold']:.0f} | "
                 f"{r['sauvola_thr_p05']:.0f} / {r['sauvola_thr_p50']:.0f} / "
                 f"{r['sauvola_thr_p95']:.0f} | {r.get('mean_u8_in_sample', float('nan')):.1f} | "
                 f"{r['native_levels_occupied']} / 256 | {rng} |")
    L.append("")
    L.append("`occupied u8 levels` counts the distinct intensities that actually occur "
             "in the content crop. The generated TIFFs are written by "
             "`scripts/analysis/_eval_v2.py:save_volume` as `xct_u8.astype(float32) / 255`, "
             "so they are already quantised to u8 steps on the sigmoid scale; the logit "
             "inversion then stretches that sub-range over 0-255 and leaves empty levels "
             "between the occupied ones. Otsu and Sauvola both read this histogram.\n")

    if sens is not None and len(sens):
        L.append("## 3. Parameter sensitivity\n")
        L.append("### 3a. Sauvola radius x k  (material mask fixed at production)\n")
        L.append("| volume | sample frac | " + " | ".join(
            f"r{r} k{k}" for r in SENS_RADII for k in SENS_KS) + " |")
        L.append("|---|---|" + "---|" * (len(SENS_RADII) * len(SENS_KS)))
        for vid in dict.fromkeys(sens.vid):
            s = sens[(sens.vid == vid) & (sens.sweep == "sauvola")]
            if not len(s):
                continue
            cells = []
            for r in SENS_RADII:
                for k in SENS_KS:
                    m = s[(s.sauvola_radius == r) & (s.sauvola_k == k)]
                    cells.append(f"{m.porosity_sample.iloc[0]:.4f}" if len(m) else "-")
            L.append(f"| {vid} | {s.sample_mask_fraction.iloc[0]:.4f} | "
                     + " | ".join(cells) + " |")
        L.append("")
        L.append("The sample_mask does not depend on the Sauvola knobs — only the "
                 "material-mask step sets it.\n")
        L.append("### 3b. Material-mask approach  (Sauvola fixed at radius 30, k 0.125)\n")
        vids = list(dict.fromkeys(sens.vid))
        L.append("| approach | " + " | ".join(
            f"{v}: T / sample frac / pore-per-sample" for v in vids) + " |")
        L.append("|---|" + "---|" * len(vids))
        for label, *_ in SENS_MATERIAL:
            cells = []
            for v in vids:
                m = sens[(sens.vid == v) & (sens.sweep == "material")
                         & (sens.label == label)]
                cells.append(f"{m.threshold.iloc[0]:.0f} / "
                             f"{m.sample_mask_fraction.iloc[0]:.3f} / "
                             f"{m.porosity_sample.iloc[0]:.4f}" if len(m) else "-")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    (OUT_DIR / "findings.md").write_text("\n".join(L).strip() + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--sens", action="store_true")
    args = ap.parse_args()
    do_all = not (args.stages or args.hist or args.sens)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(OUT_DIR / "run.log", "a")
    t_start = time.time()
    log("=== onlypores inspection pack ===")
    set_style()

    specs = volume_specs()
    records: list[dict] = []
    hists: dict = {}
    results: dict = {
        "onlypores_defaults": {"sauvola_radius": SAUVOLA_RADIUS,
                               "sauvola_k": SAUVOLA_K,
                               "min_size_filtering": MIN_SIZE_FILTERING},
        "audit_detector": {"t_best": T_BEST, "t_cons": T_CONS, "min_cc": MIN_CC,
                           "real_material_mode": REAL_MATERIAL_MODE},
        "real_volumes": {"low": REAL_LOW, "high": REAL_HIGH},
    }

    if do_all or args.stages or args.hist:
        npz = OUT_DIR / "histograms.npz"
        cache = OUT_DIR / "per_volume.csv"
        for i, spec in enumerate(specs):
            out = process_volume(spec, verify=(i == 0))
            records.append(out["record"])
            hists[out["record"]["vid"]] = out["hist"]
            h = out["hist"]["hist_sample"].astype(np.float64)
            out["record"]["mean_u8_in_sample"] = float(
                (np.arange(256) * h).sum() / max(h.sum(), 1))
        np.savez_compressed(
            npz, **{f"{vid}__{k}": v for vid, hh in hists.items()
                    for k, v in hh.items()})
        pd.DataFrame([{k: v for k, v in r.items() if k != "panel_stats"}
                      for r in records]).to_csv(cache, index=False)
        results["per_volume"] = records
        log(f"wrote {cache} and {npz}")

    if (do_all or args.hist) and records:
        fig_hist_overlay(records, hists)
        fig_hist_small_multiples(records, hists)
        log("wrote histogram figures")

    sens = None
    if do_all or args.sens:
        sens_ids = ["gen_dose_joint_oob_t0.005", "gen_dose_joint_oob_t0.05",
                    "gen_layup_joint_oob_A_1024"]
        sens_specs = [s for s in specs if s["vid"] in sens_ids]
        log(f"sensitivity sweep on {[s['vid'] for s in sens_specs]}")
        sens = sensitivity(sens_specs)
        sens.to_csv(OUT_DIR / "sensitivity.csv", index=False)
        fig_sensitivity(sens, {r["vid"]: r for r in records})
        results["sensitivity"] = sens.to_dict("records")
        log("wrote sensitivity.csv and figures")

    if records:
        write_findings_md(records, sens)
    write_json(results, OUT_DIR)
    log(f"=== total {(time.time() - t_start) / 60:.1f} min ===")
    _LOG_FH.close()


if __name__ == "__main__":
    main()

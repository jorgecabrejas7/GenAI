"""Build ``data/split_v4_<segmentation>`` — split_v3 on labels from the reference pipeline.

The split_v3 labels are the reference ``onlypores`` at its function defaults.
The reference notebooks add two steps to that call: wall exclusion and the
small-component filter (vault E21, D53; ``runs/campaigns/30-reference-onlypores/audit/``).
split_v4 takes its labels from the reference notebook outputs themselves, which
``scripts/build_reference_onlypores.py`` writes beside each raw volume in
``raw_data/MedidasDB/onlypores files/`` with the commit-pinned
``preprocess_tools``.  Those files are the source of truth; this builder does
not segment.  It uses one of the two settings the notebooks use:

    --segmentation ipynb   data/split_v4         radius 30, k 0.125, min size 8  (onlypores.ipynb cell 6)
    --segmentation batch   data/split_v5         radius 15, k 0.2,   min size 8  (onlypores_batch.ipynb cell 5)
    --segmentation v3      data/split_v3_rebuild radius 30, k 0.125, no filter, no walls
                                                 (the old compute_mask; rebuilds split_v3, see
                                                 docs/REPRODUCE_SPLIT_V3.md)

All other steps are those of split_v3 (``docs/dataset_provenance.md``, section
``data/split_v3``): drilled holes removed, split by panel, 64³ patches at stride
32, a 3-class voxel label, class weights and memmaps.

Stages (idempotent, run with ``--stage``)::

    labels   volumes.zarr/<id>/{mask, sample_mask} from the reference
             <stem>_onlypores_* and <stem>_samplemask_* TIFFs (255 -> 1), and
             labels.json (walls and settings from <stem>_report_*, voxel
             counts beside split_v3).  <id>/xct is a symlink to the split_v3
             xct array; the raw TIFF is checked equal to it voxel for voxel
             first.  Resumable per volume.
    audit    Na_04_2 and Na_02_2 against the campaign-30 audit: walls and exact
             pore and sample voxel counts.  Exit code 1 on any mismatch.
    holes    holes/<volume_id>.npy + holes.json, from the new sample_mask
    splits   splits.json (split_v3's panel rule)
    index    patch_index.parquet + index_report.json
    resplit  re-assign splits in place on a built root (--dry-run)
    weights  class_weights.json
    report   build_report.md

``--stage all`` runs labels, audit, holes, splits, index, weights, report.

Usage
-----
    python scripts/build_split_v4.py --segmentation ipynb --stage labels --volumes Na_04_2 Na_02_2
    python scripts/build_split_v4.py --segmentation ipynb --stage audit
    python scripts/build_split_v4.py --segmentation ipynb --stage all

The labels stage needs the reference outputs of every volume first.  The
memmaps come afterwards, from ``scripts/extract_patches_memmap.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from poregen.dataset.holes import (  # noqa: E402
    DILATE_VOX, EXPECTED_HOLES, MIN_AREA_PX, Z_FRACTION, detect_holes,
    patches_touching_holes,
)
from poregen.dataset.io import (  # noqa: E402
    NO_WALLS, SEGMENTATION, read_reference_report, reference_outputs, save_labels_zarr,
)
from poregen.dataset.patch_index import (  # noqa: E402
    build_patch_index_for_volume, patch_fractions, save_patch_index,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_split_v4")

V3_ROOT = REPO / "data" / "split_v3"
V3_ZARR = V3_ROOT / "volumes.zarr"
RAW_ROOT = REPO / "raw_data"
RAW_SOURCE = "MedidasDB"     # volume_id = f"{RAW_SOURCE}__{stem}"
AUDIT_DIR = REPO / "runs" / "campaigns" / "30-reference-onlypores" / "audit"

#: Set by :func:`configure` from ``--segmentation``.
SEGMENTATION_NAME: str = ""
DST_ROOT: Path = Path()
ZARR_DST: Path = Path()

PATCH_SIZE = 64
STRIDE = 32
VOXEL_SIZE_UM = 25.0
INTERIOR_MARGIN = 64            # distance to the specimen bbox faces
AIR_HEAVY = 0.5                 # "mostly air" threshold for the report
SLAB = 64                       # z slices per read when comparing with split_v3

# Not in split_v2 or split_v3 either.
EXCLUDED = ("MedidasDB__Juan_Ignacio_probetas_11_volume_eq_aligned",)

# Whole panels, so no panel is split across train/val/test.  split_v3's panels.
TEST_PANELS = ("Na_05", "Na_09", "JI_8")
VAL_PANELS = ("Na_08", "Na_01", "JI_12")
SPLIT_RULE = (
    "Split by PANEL, never by coupon. test = every coupon of panels Na_05 and "
    "Na_09 plus Juan_Ignacio_probetas_8; val = every coupon of panels Na_08 "
    "and Na_01 plus Juan_Ignacio_probetas_12; train = everything else. The 24 "
    "Airbus_Panel_Pegaso coupons are one single panel, so they can only ever "
    "be train — holding them out would remove a whole material family. Each "
    "Juan_Ignacio coupon is its own panel. These are split_v3's panels. In "
    "split_v3, Na_09 and Na_01 were added so the phi >= 6 % porosity bin is "
    "judgeable on both val and test, while train keeps nearly all of that bin: "
    "the high-porosity regime is concentrated in Na_02 and Na_10, so holding "
    "those out would test a regime the model had barely seen."
)

#: Audit run whose parameters each segmentation reproduces.
AUDIT_RUN = {"ipynb": "single_notebook", "batch": "notebook", "v3": "defaults"}
#: Root directory per segmentation name.
#: The author names the builds by their ORDER, not by the notebook that
#: produced them: split_v4 is the r30/k0.125/min8 reference set and split_v5
#: the r15/k0.2/min8 one. The --segmentation flag still names the notebook,
#: because that is what the caller is choosing between.
ROOT_NAME = {"ipynb": "split_v4", "batch": "split_v5", "v3": "split_v3_rebuild"}
AUDIT_SPECIMENS = ("Na_04_2", "Na_02_2")


def configure(segmentation: str) -> None:
    global SEGMENTATION_NAME, DST_ROOT, ZARR_DST
    SEGMENTATION_NAME = segmentation
    DST_ROOT = REPO / "data" / ROOT_NAME[segmentation]
    ZARR_DST = DST_ROOT / "volumes.zarr"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_id(volume_id: str) -> str:
    """Panel a coupon was cut from.

    ``..._Na_PP_C_...`` is coupon C of Nacho panel PP (five coupons each).
    Every ``Airbus_Panel_Pegaso_probetas_1_N`` coupon comes from one panel.
    Each ``Juan_Ignacio_probetas_N`` is a panel of its own.
    """
    m = re.search(r"_Na_(\d{2})_\d_", volume_id)
    if m:
        return f"Na_{m.group(1)}"
    if "Airbus_Panel_Pegaso" in volume_id:
        return "Pegaso_1"
    m = re.search(r"Juan_Ignacio_probetas_(\d+)_", volume_id)
    if m:
        return f"JI_{m.group(1)}"
    raise ValueError(f"cannot derive a panel from {volume_id!r}")


def split_of(pid: str) -> str:
    if pid in TEST_PANELS:
        return "test"
    if pid in VAL_PANELS:
        return "val"
    return "train"


def volume_ids() -> list[str]:
    """The split_v3 volumes: the same 80 coupons."""
    g = zarr.open_group(str(V3_ZARR), mode="r")
    return sorted(v for v in g.group_keys() if v not in EXCLUDED)


def source_group(volume_id: str) -> str:
    return volume_id.split("__")[0]


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)



# ---------------------------------------------------------------------------
# Stage: labels
# ---------------------------------------------------------------------------

def stage_labels(only: list[str] | None = None) -> dict:
    """Store the reference labels of every volume; resumable per volume."""
    from preprocess_tools.io import load_tif

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = DST_ROOT / "labels.json"
    payload = (json.loads(rec_path.read_text()) if rec_path.exists() else {
        "segmentation": SEGMENTATION_NAME,
        "params": SEGMENTATION[SEGMENTATION_NAME],
        "procedure": ("preprocess_tools.io.load_tif; reslicer.rotate_90(volume, "
                      "clockwise=False); reslicer.reslice(., 'Right'); "
                      "aligner.crop_walls(resliced) -> frontwall, backwall; "
                      "onlypores.onlypores(volume, frontwall, backwall, **params)"),
        "source": ("reference notebook outputs, raw_data/MedidasDB/onlypores files/"
                   "<stem>_{onlypores,samplemask,report}_<params>, written by "
                   "scripts/build_reference_onlypores.py"),
        "label_rule": "0 material, 1 pore (mask), 2 air (sample_mask == 0); air wins",
        "volumes": {},
    })
    records = payload["volumes"]
    params = SEGMENTATION[SEGMENTATION_NAME]

    v3 = zarr.open_group(str(V3_ZARR), mode="r")
    vols = volume_ids()
    if only:
        vols = [v for v in vols if any(f"_{o}_" in v for o in only)]
        missing = [o for o in only if not any(f"_{o}_" in v for v in vols)]
        if missing:
            raise SystemExit(f"no volume matches {missing}")

    for i, vid in enumerate(vols, 1):
        if vid in records:
            log.info("[%2d/%d] %s: done already", i, len(vols), vid)
            continue
        t0 = time.perf_counter()
        source, stem = vid.split("__", 1)
        if source != RAW_SOURCE:
            raise SystemExit(f"{vid}: not a {RAW_SOURCE} volume")
        raw_path = RAW_ROOT / RAW_SOURCE / f"{stem}.tif"
        ref = reference_outputs(raw_path, SEGMENTATION_NAME)
        missing = [str(p) for p in ref.values() if not p.exists()]
        if missing:
            raise SystemExit(f"{vid}: reference outputs missing: {missing}")
        report = read_reference_report(ref["report"])
        if {k: report[k] for k in params} != params:
            raise SystemExit(f"{vid}: report parameters {report} are not {params}")
        volume = load_tif(raw_path)
        g3 = v3[vid]
        if volume.dtype != np.uint8 or volume.shape != g3["xct"].shape:
            raise SystemExit(f"{vid}: raw {volume.dtype} {volume.shape} vs "
                             f"split_v3 xct {g3['xct'].shape}")
        depth = volume.shape[0]
        # xct is symlinked, not written: it must BE the raw volume.
        for z0 in range(0, depth, SLAB):
            if not np.array_equal(volume[z0:z0 + SLAB], np.asarray(g3["xct"][z0:z0 + SLAB])):
                raise SystemExit(f"{vid}: raw TIFF differs from split_v3 xct at z {z0}")

        del volume
        pore_tif, sample_tif = load_tif(ref["onlypores"]), load_tif(ref["samplemask"])
        for name, arr in (("onlypores", pore_tif), ("samplemask", sample_tif)):
            if arr.shape != g3["xct"].shape or not np.isin(np.unique(arr), (0, 255)).all():
                raise SystemExit(f"{vid}: {name} TIFF is {arr.dtype} {arr.shape}, values "
                                 f"{np.unique(arr)[:5]}; expected 0/255 of the volume's shape")
        pores = (pore_tif == 255).astype(np.uint8)
        sample = sample_tif == 255
        del pore_tif, sample_tif
        gc.collect()

        c = dict.fromkeys(("pore", "sample", "v3_pore", "v3_sample", "shared_pore",
                           "pore_new_only", "pore_v3_only", "sample_diff"), 0)
        for z0 in range(0, depth, SLAB):
            p = pores[z0:z0 + SLAB] > 0
            s = sample[z0:z0 + SLAB]
            p3 = np.asarray(g3["mask"][z0:z0 + SLAB]) > 0
            s3 = np.asarray(g3["sample_mask"][z0:z0 + SLAB]) > 0
            c["pore"] += int(np.count_nonzero(p))
            c["sample"] += int(np.count_nonzero(s))
            c["v3_pore"] += int(np.count_nonzero(p3))
            c["v3_sample"] += int(np.count_nonzero(s3))
            c["shared_pore"] += int(np.count_nonzero(p & p3))
            c["pore_new_only"] += int(np.count_nonzero(p & ~p3))
            c["pore_v3_only"] += int(np.count_nonzero(p3 & ~p))
            c["sample_diff"] += int(np.count_nonzero(s != s3))

        save_labels_zarr(pores, sample, DST_ROOT, vid, V3_ZARR.resolve() / vid / "xct")
        del pores, sample
        gc.collect()

        records[vid] = {
            "panel_id": panel_id(vid),
            "raw_tif": str(raw_path.relative_to(REPO)),
            "reference_files": {k: str(p.relative_to(REPO)) for k, p in ref.items()},
            "reference_commit": report["reference_commit"],
            "shape_zyx": list(g3["xct"].shape),
            "raw_equals_split_v3_xct": True,
            "frontwall": report["frontwall"], "backwall": report["backwall"],
            "counts": c,
            "vvf": c["pore"] / c["sample"],
            "vvf_split_v3": c["v3_pore"] / c["v3_sample"],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        _write_json(rec_path, payload)
        r = records[vid]
        log.info("[%2d/%d] %s: walls %d/%d  VVF %.4f %% (split_v3 %.4f %%)  "
                 "sample diff %d  %.0fs", i, len(vols), vid, r["frontwall"],
                 r["backwall"], 100 * r["vvf"], 100 * r["vvf_split_v3"],
                 c["sample_diff"], r["wall_s"])
    return payload


# ---------------------------------------------------------------------------
# Stage: audit — the campaign-30 numbers, before the cohort
# ---------------------------------------------------------------------------

def stage_audit() -> dict:
    """Check the labels of Na_04_2 and Na_02_2 against the campaign-30 audit.

    The audit counted pores with the pre-0.26 scikit-image rule (keep
    components of >= 8 voxels) and recorded how many voxels sit in components
    of exactly 8 (``pores_size_eq_min_voxels``).  The pinned reference runs
    scikit-image 0.26, which removes those too, so the expected pore count is
    ``ref_pores - pores_size_eq_min_voxels``.  Walls and sample voxels must
    match exactly as well.
    """
    labels = json.loads((DST_ROOT / "labels.json").read_text())["volumes"]
    run = AUDIT_RUN[SEGMENTATION_NAME]
    rows, ok = [], True
    for sp in AUDIT_SPECIMENS:
        vid = next(v for v in labels if f"_{sp}_volume" in v)
        d = AUDIT_DIR / sp
        summ_path = next(p for p in (d / f"summary_{run}.json", d / f"summary_{run}_composed.json")
                         if p.exists())
        summ = json.loads(summ_path.read_text())
        walls = json.loads((d / "walls.json").read_text())["walls"]
        exp_sample = summ["sample_both"] + summ["sample_ref_only"]
        # The defaults run has no size filter, so no exactly-min_size voxels to take off.
        exp_pore = summ["ref_pores"] - summ.get("pores_size_eq_min_voxels", 0)
        r = labels[vid]
        got = {"frontwall": r["frontwall"], "backwall": r["backwall"],
               "pore": r["counts"]["pore"], "sample": r["counts"]["sample"]}
        if SEGMENTATION_NAME in NO_WALLS:
            walls = {"frontwall": 0, "backwall": 0}
        exp = {"frontwall": walls["frontwall"], "backwall": walls["backwall"],
               "pore": exp_pore, "sample": exp_sample}
        match = got == exp
        ok &= match
        rows.append({"specimen": sp, "audit_file": str(summ_path.relative_to(REPO)),
                     "expected": exp, "got": got, "match": match,
                     "vvf_pct": 100 * r["vvf"],
                     "audit_vvf_pct_headline": summ["vvf_ref_pct"],
                     "audit_vvf_pct_skimage_0_26": 100 * exp_pore / exp_sample})
        log.info("%s: %s  walls %d/%d  pores %d (audit %d)  sample %d (audit %d)  "
                 "VVF %.4f %%  [audit headline %.4f %%]",
                 sp, "MATCH" if match else "MISMATCH", got["frontwall"], got["backwall"],
                 got["pore"], exp_pore, got["sample"], exp_sample, 100 * r["vvf"],
                 summ["vvf_ref_pct"])
    out = {"segmentation": SEGMENTATION_NAME, "audit_run": run, "all_match": ok, "rows": rows}
    _write_json(DST_ROOT / "audit_check.json", out)
    if not ok:
        raise SystemExit("audit mismatch — stop (see audit_check.json)")
    return out


# ---------------------------------------------------------------------------
# Stage: holes
# ---------------------------------------------------------------------------

def stage_holes() -> dict:
    out_dir = DST_ROOT / "holes"
    out_dir.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(ZARR_DST), mode="r")
    v3_holes = json.loads((V3_ROOT / "holes.json").read_text())["volumes"]
    vols = volume_ids()
    records, t0 = {}, time.perf_counter()
    same_as_v3 = 0
    for i, vid in enumerate(vols, 1):
        res = detect_holes(g[vid]["sample_mask"], volume_id=vid)
        np.save(out_dir / f"{vid}.npy", res["mask"])
        records[vid] = {k: v for k, v in res.items() if k != "mask"}
        records[vid]["mask_file"] = f"holes/{vid}.npy"
        records[vid]["shape_yx"] = list(res["mask"].shape)
        same = np.array_equal(res["mask"], np.load(V3_ROOT / v3_holes[vid]["mask_file"]))
        records[vid]["mask_equals_split_v3"] = bool(same)
        same_as_v3 += same
        log.info("[%2d/%d] %s: %d holes, same as split_v3: %s", i, len(vols), vid,
                 res["n_holes"], same)
    flagged = {v: r["n_holes"] for v, r in records.items()
               if r["n_holes"] != EXPECTED_HOLES}
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_zarr": str(ZARR_DST.relative_to(REPO)),
        "method": "poregen.dataset.holes.detect_holes",
        "params": {"z_fraction": Z_FRACTION, "min_area_px": MIN_AREA_PX,
                   "dilate_vox": DILATE_VOX,
                   "expected_holes_per_volume": EXPECTED_HOLES},
        "n_volumes": len(records),
        "volumes_with_unexpected_hole_count": flagged,
        "n_masks_equal_split_v3": int(same_as_v3),
        "wall_s": round(time.perf_counter() - t0, 1),
        "volumes": records,
    }
    _write_json(DST_ROOT / "holes.json", payload)
    log.info("holes.json written; %d volume(s) not at %d holes: %s; %d/%d masks equal split_v3",
             len(flagged), EXPECTED_HOLES, flagged, same_as_v3, len(vols))
    return payload


# ---------------------------------------------------------------------------
# Stage: splits
# ---------------------------------------------------------------------------

def stage_splits() -> dict:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    vols = volume_ids()
    per_vol = {v: {"panel_id": panel_id(v), "split": split_of(panel_id(v))}
               for v in vols}
    counts: dict[str, int] = {}
    for r in per_vol.values():
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    panels: dict[str, list[str]] = {}
    for v, r in per_vol.items():
        panels.setdefault(r["panel_id"], []).append(v)
    payload = {
        "version": ROOT_NAME[SEGMENTATION_NAME],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": SPLIT_RULE,
        "test_panels": list(TEST_PANELS),
        "val_panels": list(VAL_PANELS),
        "pegaso_single_panel_train_only": True,
        "excluded_volume_ids": list(EXCLUDED),
        "counts": counts,
        "panels": {p: sorted(v) for p, v in sorted(panels.items())},
        "volumes": {v: r["split"] for v, r in per_vol.items()},
        "panel_id": {v: r["panel_id"] for v, r in per_vol.items()},
    }
    _write_json(DST_ROOT / "splits.json", payload)
    log.info("splits.json written: %s", counts)
    return payload


# ---------------------------------------------------------------------------
# Stage: patch index
# ---------------------------------------------------------------------------

def bbox_3d(sample_mask: np.ndarray) -> list[int]:
    """[z_lo, z_hi, y_lo, y_hi, x_lo, x_hi] of the specimen, half-open."""
    out = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        prof = sample_mask.any(axis=others)
        nz = np.flatnonzero(prof)
        out += [int(nz[0]), int(nz[-1]) + 1]
    return out


def interior_flags(df: pd.DataFrame, bbox: list[int]) -> np.ndarray:
    """Patch entirely ``INTERIOR_MARGIN`` voxels clear of the specimen bbox."""
    zl, zh, yl, yh, xl, xh = bbox
    m, ps = INTERIOR_MARGIN, PATCH_SIZE
    return (
        (df["z0"].to_numpy() - zl >= m) & (zh - (df["z0"].to_numpy() + ps) >= m)
        & (df["y0"].to_numpy() - yl >= m) & (yh - (df["y0"].to_numpy() + ps) >= m)
        & (df["x0"].to_numpy() - xl >= m) & (xh - (df["x0"].to_numpy() + ps) >= m)
    )


def stage_index() -> dict:
    splits = json.loads((DST_ROOT / "splits.json").read_text())
    holes_meta = json.loads((DST_ROOT / "holes.json").read_text())
    g = zarr.open_group(str(ZARR_DST), mode="r")

    frames, per_vol, t0 = [], [], time.perf_counter()
    vols = volume_ids()
    for i, vid in enumerate(vols, 1):
        tv = time.perf_counter()
        split = splits["volumes"][vid]
        pid = splits["panel_id"][vid]

        mask = np.asarray(g[vid]["mask"]) > 0
        df = build_patch_index_for_volume(
            mask, vid, source_group(vid), split, PATCH_SIZE, STRIDE)
        del mask
        if df.empty:
            log.warning("%s: no patches", vid)
            continue

        smask = np.asarray(g[vid]["sample_mask"]) > 0
        bbox = bbox_3d(smask)
        coords = df[["z0", "y0", "x0"]].to_numpy(np.int64)
        df["air_fraction"] = patch_fractions(~smask, coords, PATCH_SIZE)
        del smask

        df["panel_id"] = pid
        n_before = len(df)
        interior_before = interior_flags(df, bbox)
        heavy_before = df["air_fraction"].to_numpy() > AIR_HEAVY

        hole_mask = np.load(DST_ROOT / "holes" / f"{vid}.npy")
        drop = patches_touching_holes(hole_mask, coords[:, 1], coords[:, 2],
                                      PATCH_SIZE)
        df = df[~drop].reset_index(drop=True)
        interior_after = interior_flags(df, bbox)
        heavy_after = df["air_fraction"].to_numpy() > AIR_HEAVY

        frames.append(df)
        per_vol.append({
            "volume_id": vid, "panel_id": pid, "split": split,
            "shape": list(g[vid]["xct"].shape),
            "bbox_zyx": bbox,
            "n_holes": holes_meta["volumes"][vid]["n_holes"],
            "n_patches_before": int(n_before),
            "n_dropped_for_holes": int(drop.sum()),
            "n_patches_after": int(len(df)),
            "mean_porosity": float(df["porosity"].mean()),
            "mean_air_fraction": float(df["air_fraction"].mean()),
            "n_air_heavy_before": int(heavy_before.sum()),
            "n_air_heavy_interior_before": int((heavy_before & interior_before).sum()),
            "n_air_heavy_after": int(heavy_after.sum()),
            "n_air_heavy_interior_after": int((heavy_after & interior_after).sum()),
            "wall_s": round(time.perf_counter() - tv, 1),
        })
        log.info("[%2d/%d] %s (%s, %s): %d -> %d patches (-%d holes), "
                 "phi=%.4f air=%.4f, %.0fs", i, len(vols), vid, pid, split,
                 n_before, len(df), int(drop.sum()),
                 per_vol[-1]["mean_porosity"], per_vol[-1]["mean_air_fraction"],
                 per_vol[-1]["wall_s"])

    full = pd.concat(frames, ignore_index=True)
    save_patch_index(full, DST_ROOT / "patch_index.parquet")

    # Same patches as split_v3?  Then patches_xct.bin is the same bytes.
    v3 = pd.read_parquet(V3_ROOT / "patch_index.parquet",
                         columns=["volume_id", "z0", "y0", "x0"])
    same_rows = v3.equals(full[["volume_id", "z0", "y0", "x0"]])

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "patch_size": PATCH_SIZE, "stride": STRIDE,
        "voxel_size_um": VOXEL_SIZE_UM,
        "interior_margin_vox": INTERIOR_MARGIN,
        "air_heavy_threshold": AIR_HEAVY,
        "hole_rule": (f"drop every patch whose (y, x) footprint intersects a "
                      f"detected hole dilated by {DILATE_VOX} voxels"),
        "split_rule": SPLIT_RULE,
        "n_volumes": len(per_vol),
        "n_patches": int(len(full)),
        "patch_rows_equal_split_v3": bool(same_rows),
        "wall_s": round(time.perf_counter() - t0, 1),
        "per_volume": per_vol,
    }
    _write_json(DST_ROOT / "index_report.json", report)
    log.info("patch_index.parquet: %d rows in %.0fs; rows equal split_v3: %s",
             len(full), report["wall_s"], same_rows)
    return report


# ---------------------------------------------------------------------------
# Stage: resplit — re-assign splits in place, without re-extracting
# ---------------------------------------------------------------------------

POROSITY_BINS = [0.0, 0.01, 0.03, 0.06, float("inf")]
BIN_LABELS = ["<1%", "1-3%", "3-6%", ">=6%"]


def panel_table(df: pd.DataFrame) -> list[dict]:
    """Per-panel porosity profile, for the record in splits.json and the docs."""
    rows = []
    for p, d in df.groupby("panel_id"):
        rows.append({
            "panel": p,
            "volumes": int(d.volume_id.nunique()),
            "patches": int(len(d)),
            "mean_porosity": float(d.porosity.mean()),
            "n_ge_6pct": int((d.porosity >= 0.06).sum()),
            "n_3_to_6pct": int(((d.porosity >= 0.03) & (d.porosity < 0.06)).sum()),
            "n_1_to_3pct": int(((d.porosity >= 0.01) & (d.porosity < 0.03)).sum()),
        })
    return sorted(rows, key=lambda r: -r["n_ge_6pct"])


def split_table(df: pd.DataFrame) -> dict:
    """Patches, volumes, panels, mean porosity and porosity bins per split."""
    out = {}
    for sp in ("train", "val", "test"):
        d = df[df.split == sp]
        cut = pd.cut(d.porosity, bins=POROSITY_BINS, right=False, labels=BIN_LABELS)
        out[sp] = {
            "n_patches": int(len(d)),
            "n_volumes": int(d.volume_id.nunique()),
            "panels": sorted(d.panel_id.unique().tolist()),
            "mean_porosity": float(d.porosity.mean()),
            "bins": {l: int(v) for l, v in cut.value_counts().reindex(BIN_LABELS).items()},
        }
    return out


def stage_resplit(dry_run: bool = False) -> dict:
    """Apply the current TEST_PANELS/VAL_PANELS to an already-built root.

    Only the ``split`` COLUMN of the parquet changes. Row order is untouched,
    so ``patches_xct.bin`` and ``patches_label.bin`` stay row-aligned and are
    not re-extracted.
    """
    idx = DST_ROOT / "patch_index.parquet"
    df = pd.read_parquet(idx)
    before = df["split"].value_counts().to_dict()
    order_before = df[["volume_id", "z0", "y0", "x0"]].copy()

    new_split = df["panel_id"].map(split_of)
    changed = int((new_split != df["split"]).sum())
    df["split"] = new_split
    after = df["split"].value_counts().to_dict()

    # The invariant the memmaps depend on.
    assert order_before.equals(df[["volume_id", "z0", "y0", "x0"]]), \
        "row order changed — the memmaps would no longer be row-aligned"

    per_split_bins = split_table(df)
    report = {
        "changed_rows": changed,
        "counts_before": before,
        "counts_after": after,
        "per_split": per_split_bins,
        "panel_table": panel_table(df),
    }
    if dry_run:
        log.info("DRY RUN — nothing written")
        for sp, v in per_split_bins.items():
            log.info(f"  {sp:5s} {v['n_volumes']:2d} vols {v['n_patches']:8d} patches "
                     f"mean phi {v['mean_porosity']:.5f} bins {v['bins']} panels {v['panels']}")
        log.info(f"  rows whose split changes: {changed}")
        return report

    save_patch_index(df, idx)
    splits = json.loads((DST_ROOT / "splits.json").read_text())
    splits.update({
        "resplit_created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": SPLIT_RULE,
        "test_panels": list(TEST_PANELS),
        "val_panels": list(VAL_PANELS),
        "panel_table": report["panel_table"],
        "per_split": per_split_bins,
        "volumes": {v: split_of(p) for v, p in splits["panel_id"].items()},
        "counts": {sp: per_split_bins[sp]["n_volumes"] for sp in per_split_bins},
    })
    _write_json(DST_ROOT / "splits.json", splits)

    meta_path = DST_ROOT / "patches_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["splits"] = {sp: per_split_bins[sp]["n_patches"] for sp in per_split_bins}
        meta["split_rule"] = SPLIT_RULE
        meta["parquet_sha256"] = _sha256(idx)
        _write_json(meta_path, meta)
        log.info(f"patches_meta.json updated; parquet sha {meta['parquet_sha256'][:16]}…")

    log.info(f"resplit applied: {changed} rows changed split; counts {after}")
    return report


def _sha256(path: Path, block: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage: class weights
# ---------------------------------------------------------------------------

def stage_weights(rule: str = "sqrt_inverse") -> dict:
    """Train-split voxel frequency of each label class, and the CE weights.

    The parquet already carries both minority fractions per patch — ``porosity``
    is the pore fraction and ``air_fraction`` the air fraction of a 64^3 patch —
    so the frequencies come straight off it and the label memmap is not needed.
    Every patch has the same voxel count, so a mean over patches IS the voxel
    frequency.

    Two rules, both normalised so ``sum_c f_c w_c = 1`` — the weighted
    cross-entropy then keeps the magnitude of an unweighted one instead of
    being inflated by the rare classes:

    ``inverse``       ``w_c ∝ 1 / f_c``. Full inverse-frequency. Used for
                      r08-run-0002, where it put pore at 16.1 and made a false
                      positive cheap enough that the model over-predicted pore
                      ~2.2x wherever porosity was appreciable — see
                      ``runs/campaigns/09-r08-latent-sweep/calibration_probe``.
    ``sqrt_inverse``  ``w_c ∝ 1 / sqrt(f_c)``. Tempered: it still lifts the
                      rare classes but by the square root, so the decision
                      boundary is not pushed as far. The default from
                      r08-run-0003 on.
    """
    if rule not in ("inverse", "sqrt_inverse"):
        raise ValueError(f"unknown weight rule {rule!r}")
    df = pd.read_parquet(DST_ROOT / "patch_index.parquet",
                         columns=["split", "porosity", "air_fraction"])
    tr = df[df.split == "train"]
    f_pore = float(tr.porosity.mean())
    f_air = float(tr.air_fraction.mean())
    f = [1.0 - f_pore - f_air, f_pore, f_air]
    raw = ([1.0 / x for x in f] if rule == "inverse"
           else [1.0 / (x ** 0.5) for x in f])
    norm = sum(fc * rc for fc, rc in zip(f, raw))   # so sum_c f_c w_c = 1
    w = [r / norm for r in raw]
    out = {
        "computed_from": f"{DST_ROOT.relative_to(REPO)}/patch_index.parquet, split == train",
        "n_train_patches": int(len(tr)),
        "class_names": ["material", "pore", "air"],
        "class_frequency": f,
        "weight_rule_name": rule,
        "weight_rule": ("w_c proportional to 1/f_c" if rule == "inverse"
                        else "w_c proportional to 1/sqrt(f_c)")
                       + ", normalised so sum_c f_c w_c = 1",
        "class_weights": w,
    }
    _write_json(DST_ROOT / "class_weights.json", out)
    log.info("rule %s | class frequencies material/pore/air = %.6f / %.6f / %.6f",
             rule, *f)
    log.info("class weights                      = %.4f / %.4f / %.4f", *w)
    return out


# ---------------------------------------------------------------------------
# Stage: build report
# ---------------------------------------------------------------------------

def stage_report() -> str:
    rep = json.loads((DST_ROOT / "index_report.json").read_text())
    holes = json.loads((DST_ROOT / "holes.json").read_text())
    splits = json.loads((DST_ROOT / "splits.json").read_text())
    labels = json.loads((DST_ROOT / "labels.json").read_text())
    pv = pd.DataFrame(rep["per_volume"])
    df = pd.read_parquet(DST_ROOT / "patch_index.parquet",
                         columns=["volume_id", "panel_id", "split", "porosity", "air_fraction"])
    v3 = pd.read_parquet(V3_ROOT / "patch_index.parquet",
                         columns=["split", "porosity"])
    params = labels["params"]
    name = DST_ROOT.name

    L = [f"# {name} build report", "",
         f"Built {rep['created']}. {rep['n_volumes']} volumes, {PATCH_SIZE}^3 "
         f"patches at stride {STRIDE}, voxel size {VOXEL_SIZE_UM} um.", "",
         "## Labels", "",
         f"Reference pipeline `{labels['procedure']}`, settings `{SEGMENTATION_NAME}`: "
         f"sauvola_radius {params['sauvola_radius']}, sauvola_k {params['sauvola_k']}, "
         f"min_size_filtering {params['min_size_filtering']}. "
         f"Read from the reference notebook outputs ({labels['source']}); "
         f"preprocess_tools commit(s) in their reports: "
         f"{', '.join(sorted({r['reference_commit'] for r in labels['volumes'].values()}))}.",
         "",
         "`volumes.zarr/<id>/xct` is a symlink to the split_v3 xct array (checked "
         "equal to the raw TIFF voxel for voxel on every volume); `mask` and "
         "`sample_mask` are new.", "",
         "| volume | panel | walls (front/back) | VVF % | split_v3 VVF % | sample voxels differing from split_v3 |",
         "|---|---|---|---|---|---|"]
    for vid, r in sorted(labels["volumes"].items()):
        L.append(f"| {vid} | {r['panel_id']} | {r['frontwall']}/{r['backwall']} "
                 f"| {100 * r['vvf']:.4f} | {100 * r['vvf_split_v3']:.4f} "
                 f"| {r['counts']['sample_diff']} |")
    L += ["", "## Per split", "",
          "| split | volumes | panels | patches before | dropped for holes "
          "| patches after | mean phi | split_v3 mean phi | mean air "
          "| air>0.5 & interior (after) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "val", "test"):
        s = pv[pv.split == sp]
        d = df[df.split == sp]
        after = int(s.n_patches_after.sum())
        ia = int(s.n_air_heavy_interior_after.sum())
        L.append(
            f"| {sp} | {len(s)} | {s.panel_id.nunique()} | {int(s.n_patches_before.sum())} "
            f"| {int(s.n_dropped_for_holes.sum())} | {after} "
            f"| {d.porosity.mean():.5f} | {v3[v3.split == sp].porosity.mean():.5f} "
            f"| {d.air_fraction.mean():.4f} "
            f"| {ia} ({100 * ia / max(after, 1):.3f}%) |")
    L += ["", f"Patch rows equal split_v3 (same volumes, same origins, same order): "
          f"{rep['patch_rows_equal_split_v3']}.", "",
          "## Porosity bins per split", "",
          "| split | <1% | 1-3% | 3-6% | >=6% |", "|---|---|---|---|---|"]
    for sp, v in split_table(df).items():
        b = v["bins"]
        L.append(f"| {sp} | {b['<1%']} | {b['1-3%']} | {b['3-6%']} | {b['>=6%']} |")
    L += ["", "## Panels", "",
          "| panel | volumes | patches | mean phi | n >=6% | n 3-6% | n 1-3% |",
          "|---|---|---|---|---|---|---|"]
    for r in panel_table(df):
        L.append(f"| {r['panel']} | {r['volumes']} | {r['patches']} | {r['mean_porosity']:.5f} "
                 f"| {r['n_ge_6pct']} | {r['n_3_to_6pct']} | {r['n_1_to_3pct']} |")
    L += ["", "## Splits", "", SPLIT_RULE, "",
          "| split | panels | volumes |", "|---|---|---|"]
    for sp in ("train", "val", "test"):
        ps = sorted({r["panel_id"] for r in rep["per_volume"] if r["split"] == sp})
        n = sum(r["split"] == sp for r in rep["per_volume"])
        L.append(f"| {sp} | {', '.join(ps)} | {n} |")
    flagged = holes["volumes_with_unexpected_hole_count"]
    L += ["", f"Excluded: {', '.join(splits['excluded_volume_ids'])}.", "",
          "## Holes", "",
          f"Detector: `poregen.dataset.holes.detect_holes` (z fraction {Z_FRACTION}, "
          f"min area {MIN_AREA_PX} px, dilation {DILATE_VOX} vox) on the new sample_mask. "
          f"{holes['n_masks_equal_split_v3']}/{holes['n_volumes']} hole masks equal split_v3's. "
          + ("All volumes have exactly 3 holes." if not flagged else
             f"**{len(flagged)} volume(s) not at 3 holes**: "
             + ", ".join(f"{k} ({v})" for k, v in flagged.items())), ""]
    text = "\n".join(L)
    (DST_ROOT / "build_report.md").write_text(text)
    log.info("build_report.md written")
    return text


# ---------------------------------------------------------------------------

STAGES = {"labels": stage_labels, "audit": stage_audit, "holes": stage_holes,
          "splits": stage_splits, "index": stage_index, "resplit": stage_resplit,
          "weights": stage_weights, "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation", required=True, choices=sorted(SEGMENTATION))
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    ap.add_argument("--volumes", nargs="+", metavar="SPECIMEN",
                    help="labels only: e.g. Na_04_2 Na_02_2")
    ap.add_argument("--dry-run", action="store_true",
                    help="resplit only: report what would change, write nothing")
    args = ap.parse_args()
    configure(args.segmentation)
    names = ([n for n in STAGES if n != "resplit"] if args.stage == "all"
             else [args.stage])
    for n in names:
        log.info("=== stage %s (%s) ===", n, DST_ROOT.relative_to(REPO))
        if n == "resplit":
            STAGES[n](dry_run=args.dry_run)
        elif n == "labels":
            STAGES[n](only=args.volumes)
        else:
            STAGES[n]()


if __name__ == "__main__":
    main()

"""Build ``data/split_v3`` — hole-free patches, panel-level splits, 3-class labels.

Three things change from ``split_v2``, and each fixes a defect that would
otherwise be baked into the r08 VAE and ldm06:

1. **Drilled holes removed.**  Every coupon has three ~200-voxel registration
   through-holes.  They are ``False`` in ``sample_mask`` and ``0`` in the pore
   mask, so they read as porosity 0 while being ~95 % air: 6 065 train patches
   of ``split_v2`` sit inside one.  They are the only interior source of large
   air in the dataset.  Any patch whose in-plane footprint touches a hole
   (dilated by 32 voxels) is dropped — see :mod:`poregen.dataset.holes`.
2. **Split by panel, not by coupon.**  ``split_v2`` scattered the five coupons
   of a panel across train/val/test, so val and test shared a panel with train.
   Here a whole panel goes to one split.
3. **Three-class voxel label** ``material / pore / air`` instead of the binary
   pore mask, so the model can represent exterior air as its own class.

``volumes.zarr`` is NOT copied: ``data/split_v3/volumes.zarr`` is a symlink to
the ``split_v2`` one.  ``data/split_v2`` is never written.

Stages (all idempotent, run with ``--stage``)::

    holes    holes/<volume_id>.npy + holes.json
    splits   splits.json  (rule, panel_id and split per volume)
    index    patch_index.parquet + index_report.json
    resplit  re-assign splits in place on an already-built root (--dry-run)
    weights  class_weights.json  (the r08 cross-entropy class weights)
    report   build_report.md

Usage
-----
    python scripts/build_split_v3.py --stage all
    python scripts/build_split_v3.py --stage index

The memmaps come afterwards, from ``scripts/extract_patches_memmap.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from poregen.dataset.patch_index import (  # noqa: E402
    build_patch_index_for_volume, generate_patch_coords, patch_fractions,
    save_patch_index,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_split_v3")

SRC_ROOT = REPO / "data" / "split_v2"
DST_ROOT = REPO / "data" / "split_v3"
ZARR_SRC = SRC_ROOT / "volumes.zarr"

PATCH_SIZE = 64
STRIDE = 32
VOXEL_SIZE_UM = 25.0
INTERIOR_MARGIN = 64            # distance to the specimen bbox faces
AIR_HEAVY = 0.5                 # "mostly air" threshold for the report

# Not in split_v2 either.
EXCLUDED = ("MedidasDB__Juan_Ignacio_probetas_11_volume_eq_aligned",)

# Whole panels, so no panel is split across train/val/test.
TEST_PANELS = ("Na_05", "Na_09", "JI_8")
VAL_PANELS = ("Na_08", "Na_01", "JI_12")
SPLIT_RULE = (
    "Split by PANEL, never by coupon. test = every coupon of panels Na_05 and "
    "Na_09 plus Juan_Ignacio_probetas_8; val = every coupon of panels Na_08 "
    "and Na_01 plus Juan_Ignacio_probetas_12; train = everything else. The 24 "
    "Airbus_Panel_Pegaso coupons are one single panel, so they can only ever "
    "be train — holding them out would remove a whole material family. Each "
    "Juan_Ignacio coupon is its own panel. "
    "Na_09 and Na_01 were added so the phi >= 6 % porosity bin is judgeable on "
    "both val and test: the first split (Na_05 + JI_8 / Na_08 + JI_12) left "
    "only 263 and 42 patches there, below any sensible floor. They were chosen "
    "over the panels with the MOST high-porosity material because the "
    "high-porosity regime is extremely concentrated — Na_02 alone holds 77 % "
    "of the train patches at phi >= 6 % and Na_02 + Na_10 hold 92 % — so "
    "holding those out would have left train with 8 % of its high-porosity "
    "data and tested a regime the model had barely seen. Na_09 + Na_01 give "
    "6055 and 1220 patches in that bin while train keeps 96 %."
)

# The high-porosity regime in TRAIN rests on two panels. Recorded because a
# reader judging a phi >= 6 % result needs to know the training support for it
# is not spread across the dataset.
HIGH_POROSITY_NOTE = (
    "Training support for phi >= 6 % is concentrated: Na_02 holds 77 % of it "
    "and Na_10 a further 15 %. Losing either panel from train would change "
    "what the model can represent at high porosity."
)


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
    g = zarr.open_group(str(ZARR_SRC), mode="r")
    return sorted(v for v in g.group_keys() if v not in EXCLUDED)


def source_group(volume_id: str) -> str:
    return volume_id.split("__")[0]


# ---------------------------------------------------------------------------
# Stage: holes
# ---------------------------------------------------------------------------

def stage_holes() -> dict:
    out_dir = DST_ROOT / "holes"
    out_dir.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(ZARR_SRC), mode="r")
    vols = volume_ids()
    records, t0 = {}, time.perf_counter()
    for i, vid in enumerate(vols, 1):
        npy = out_dir / f"{vid}.npy"
        res = detect_holes(g[vid]["sample_mask"], volume_id=vid)
        np.save(npy, res["mask"])
        records[vid] = {k: v for k, v in res.items() if k != "mask"}
        records[vid]["mask_file"] = f"holes/{vid}.npy"
        records[vid]["shape_yx"] = list(res["mask"].shape)
        log.info("[%2d/%d] %s: %d holes", i, len(vols), vid, res["n_holes"])
    flagged = {v: r["n_holes"] for v, r in records.items()
               if r["n_holes"] != EXPECTED_HOLES}
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_zarr": str(ZARR_SRC),
        "method": "poregen.dataset.holes.detect_holes",
        "params": {"z_fraction": Z_FRACTION, "min_area_px": MIN_AREA_PX,
                   "dilate_vox": DILATE_VOX,
                   "expected_holes_per_volume": EXPECTED_HOLES},
        "n_volumes": len(records),
        "volumes_with_unexpected_hole_count": flagged,
        "wall_s": round(time.perf_counter() - t0, 1),
        "volumes": records,
    }
    (DST_ROOT / "holes.json").write_text(json.dumps(payload, indent=2))
    log.info("holes.json written; %d volume(s) not at %d holes: %s",
             len(flagged), EXPECTED_HOLES, flagged)
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
        "version": "v3",
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
    (DST_ROOT / "splits.json").write_text(json.dumps(payload, indent=2))
    log.info("splits.json written: %s", counts)
    return payload


def link_zarr() -> None:
    """data/split_v3/volumes.zarr -> the split_v2 store (never a copy)."""
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    link = DST_ROOT / "volumes.zarr"
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(ZARR_SRC)
    log.info("symlinked %s -> %s", link, ZARR_SRC)


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
    link_zarr()
    splits = json.loads((DST_ROOT / "splits.json").read_text())
    holes_meta = json.loads((DST_ROOT / "holes.json").read_text())
    g = zarr.open_group(str(ZARR_SRC), mode="r")

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
        "wall_s": round(time.perf_counter() - t0, 1),
        "per_volume": per_vol,
    }
    (DST_ROOT / "index_report.json").write_text(json.dumps(report, indent=2))
    log.info("patch_index.parquet: %d rows in %.0fs", len(full),
             report["wall_s"])
    return report


# ---------------------------------------------------------------------------
# Stage: resplit — re-assign splits in place, without re-extracting
# ---------------------------------------------------------------------------

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


def stage_resplit(dry_run: bool = False) -> dict:
    """Apply the current TEST_PANELS/VAL_PANELS to an already-built root.

    Only the ``split`` COLUMN of the parquet changes. Row order is untouched,
    so ``patches_xct.bin`` and ``patches_label.bin`` stay row-aligned and are
    not re-extracted — that is the whole reason this stage exists rather than
    re-running ``--stage all``, which would cost 78 minutes of extraction to
    reach the same bytes.
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

    bins = [0.0, 0.01, 0.03, 0.06, float("inf")]
    labels = ["<1%", "1-3%", "3-6%", ">=6%"]
    per_split_bins = {}
    for sp in ("train", "val", "test"):
        d = df[df.split == sp]
        cut = pd.cut(d.porosity, bins=bins, right=False, labels=labels)
        per_split_bins[sp] = {
            "n_patches": int(len(d)),
            "n_volumes": int(d.volume_id.nunique()),
            "panels": sorted(d.panel_id.unique().tolist()),
            "mean_porosity": float(d.porosity.mean()),
            "bins": {l: int(v) for l, v in cut.value_counts().reindex(labels).items()},
        }

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
        "high_porosity_note": HIGH_POROSITY_NOTE,
        "test_panels": list(TEST_PANELS),
        "val_panels": list(VAL_PANELS),
        "panel_table": report["panel_table"],
        "per_split": per_split_bins,
        "volumes": {v: split_of(p) for v, p in splits["panel_id"].items()},
        "counts": {sp: per_split_bins[sp]["n_volumes"] for sp in per_split_bins},
    })
    (DST_ROOT / "splits.json").write_text(json.dumps(splits, indent=2))

    meta_path = DST_ROOT / "patches_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["splits"] = {sp: per_split_bins[sp]["n_patches"] for sp in per_split_bins}
        meta["split_rule"] = SPLIT_RULE
        meta["parquet_sha256"] = _sha256(idx)
        meta_path.write_text(json.dumps(meta, indent=2))
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

def stage_weights() -> dict:
    """Train-split voxel frequency of each label class, and the CE weights.

    The parquet already carries both minority fractions per patch — ``porosity``
    is the pore fraction and ``air_fraction`` the air fraction of a 64^3 patch —
    so the frequencies come straight off it and the label memmap is not needed.
    Every patch has the same voxel count, so a mean over patches IS the voxel
    frequency.

    Weight rule: ``w_c = 1 / (n_classes * f_c)``, which makes ``sum_c f_c w_c
    = 1`` — the weighted cross-entropy keeps the magnitude of an unweighted one
    instead of being inflated by the rare classes.
    """
    df = pd.read_parquet(DST_ROOT / "patch_index.parquet",
                         columns=["split", "porosity", "air_fraction"])
    tr = df[df.split == "train"]
    f_pore = float(tr.porosity.mean())
    f_air = float(tr.air_fraction.mean())
    f = [1.0 - f_pore - f_air, f_pore, f_air]
    n = len(f)
    w = [1.0 / (n * x) for x in f]
    out = {
        "computed_from": "data/split_v3/patch_index.parquet, split == train",
        "n_train_patches": int(len(tr)),
        "class_names": ["material", "pore", "air"],
        "class_frequency": f,
        "weight_rule": "w_c = 1 / (n_classes * f_c), so sum_c f_c w_c = 1",
        "class_weights": w,
    }
    (DST_ROOT / "class_weights.json").write_text(json.dumps(out, indent=2))
    log.info("class frequencies material/pore/air = %.6f / %.6f / %.6f",
             *f)
    log.info("class weights                      = %.4f / %.4f / %.4f", *w)
    return out


# ---------------------------------------------------------------------------
# Stage: build report
# ---------------------------------------------------------------------------

def stage_report() -> str:
    rep = json.loads((DST_ROOT / "index_report.json").read_text())
    holes = json.loads((DST_ROOT / "holes.json").read_text())
    splits = json.loads((DST_ROOT / "splits.json").read_text())
    pv = pd.DataFrame(rep["per_volume"])
    df = pd.read_parquet(DST_ROOT / "patch_index.parquet",
                         columns=["split", "porosity", "air_fraction"])

    L = ["# split_v3 build report", "",
         f"Built {rep['created']} from `{ZARR_SRC}` (symlinked, never "
         f"copied). {rep['n_volumes']} volumes, {PATCH_SIZE}^3 patches at "
         f"stride {STRIDE}, voxel size {VOXEL_SIZE_UM} um.", "",
         "## What changed from split_v2", "",
         "1. Every patch touching a drilled registration hole is dropped "
         f"({rep['hole_rule']}).",
         "2. " + SPLIT_RULE,
         "3. The stored voxel label is 3-class — 0 material, 1 pore, 2 air "
         "(`sample_mask == 0`) — instead of the binary pore mask.", "",
         "## Per split", "",
         "| split | volumes | panels | patches before | dropped for holes "
         "| patches after | mean phi | mean air | air>0.5 & interior (before) "
         "| air>0.5 & interior (after) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "val", "test"):
        s = pv[pv.split == sp]
        d = df[df.split == sp]
        before = int(s.n_patches_before.sum())
        after = int(s.n_patches_after.sum())
        ib = int(s.n_air_heavy_interior_before.sum())
        ia = int(s.n_air_heavy_interior_after.sum())
        L.append(
            f"| {sp} | {len(s)} | {s.panel_id.nunique()} | {before} "
            f"| {int(s.n_dropped_for_holes.sum())} | {after} "
            f"| {d.porosity.mean():.4f} | {d.air_fraction.mean():.4f} "
            f"| {ib} ({100 * ib / max(before, 1):.3f}%) "
            f"| {ia} ({100 * ia / max(after, 1):.3f}%) |")
    tb = int(pv.n_patches_before.sum())
    ta = int(pv.n_patches_after.sum())
    tib = int(pv.n_air_heavy_interior_before.sum())
    tia = int(pv.n_air_heavy_interior_after.sum())
    L += ["",
          f"Total: {tb} patches before, {int(pv.n_dropped_for_holes.sum())} "
          f"dropped for holes ({100 * pv.n_dropped_for_holes.sum() / tb:.2f}%), "
          f"{ta} after.", "",
          f"**Fully-interior air patches** (`air_fraction > {AIR_HEAVY}` and at "
          f"least {INTERIOR_MARGIN} voxels from every face of the specimen "
          f"bounding box) fall from {tib} ({100 * tib / tb:.3f}% of all "
          f"patches) to {tia} ({100 * tia / ta:.3f}%). Those patches can only "
          "be holes: the specimen has no other interior air.", "",
          "## Splits", "",
          "| split | panels | volumes |", "|---|---|---|"]
    for sp in ("train", "val", "test"):
        ps = sorted({r["panel_id"] for r in rep["per_volume"] if r["split"] == sp})
        n = sum(r["split"] == sp for r in rep["per_volume"])
        L.append(f"| {sp} | {', '.join(ps)} | {n} |")
    L += ["", f"Excluded: {', '.join(splits['excluded_volume_ids'])}.", "",
          "## Holes per volume", "",
          f"Detector: `poregen.dataset.holes.detect_holes` "
          f"(z fraction {Z_FRACTION}, min area {MIN_AREA_PX} px, dilation "
          f"{DILATE_VOX} vox). {EXPECTED_HOLES} expected per volume.", "",
          "| volume | panel | split | holes | diameters (px) | dropped patches |",
          "|---|---|---|---|---|---|"]
    for r in rep["per_volume"]:
        h = holes["volumes"][r["volume_id"]]
        dia = ", ".join(f"{x['equiv_diameter_px']:.0f}" for x in h["holes"])
        L.append(f"| {r['volume_id']} | {r['panel_id']} | {r['split']} "
                 f"| {r['n_holes']} | {dia} | {r['n_dropped_for_holes']} |")
    flagged = holes["volumes_with_unexpected_hole_count"]
    L += ["",
          ("All volumes have exactly 3 holes."
           if not flagged else
           f"**{len(flagged)} volume(s) not at 3 holes**: "
           + ", ".join(f"{k} ({v})" for k, v in flagged.items())), ""]
    text = "\n".join(L)
    (DST_ROOT / "build_report.md").write_text(text)
    log.info("build_report.md written")
    return text


# ---------------------------------------------------------------------------

STAGES = {"holes": stage_holes, "splits": stage_splits,
          "index": stage_index, "resplit": stage_resplit,
          "weights": stage_weights, "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=[*STAGES, "all"])
    ap.add_argument("--dry-run", action="store_true",
                    help="resplit only: report what would change, write nothing")
    args = ap.parse_args()
    # `all` rebuilds from scratch, which already produces the current split
    # rule; `resplit` is the in-place path for a root that is already built.
    names = ([n for n in STAGES if n != "resplit"] if args.stage == "all"
             else [args.stage])
    for n in names:
        log.info("=== stage %s ===", n)
        if n == "resplit":
            STAGES[n](dry_run=args.dry_run)
        else:
            STAGES[n]()


if __name__ == "__main__":
    main()

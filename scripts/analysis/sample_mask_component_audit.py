"""Is any stored ``sample_mask`` built on a dust speck?  (CPU, read-only)

``poregen.dataset.segmentation.material_mask`` takes the specimen bounding box
from the max-projection of the Otsu binary.  Until the F9 fix it used
``regionprops(labels)[0]`` — the FIRST label, which is raster order, not size.
A bright artefact above and left of the coupon therefore became "the specimen"
and the stored ``sample_mask`` collapsed to that artefact's bounding box.  The
dataset was built with the old code, so every stored mask has to be checked.

What this does
--------------
For each of the 80 dataset volumes it reads **one middle z-slice** of ``xct`` and
the same slice of the stored ``sample_mask`` — ~1 MB of decoded data per volume,
nothing bulk — and on that one slice reports:

* the projected areas of the components the specimen box is chosen from,
* which component the OLD rule (first label) and the NEW rule (largest area)
  each pick, and whether they disagree,
* whether the new ambiguity gate (second-largest > 10 % of largest) would fire,
* the bounding box of the stored mask against the bounding box the fixed
  function picks, and the Dice between the two masks.

Reading the decisive columns
----------------------------
``first_is_largest`` and ``stored_bbox`` are the load-bearing ones.  A volume
whose stored mask spans the same box as the largest component was not built on a
speck, whatever its Dice.

``dice`` is a weaker signal and is NOT expected to be 1.0.  Production runs a
GLOBAL Otsu over the whole cropped volume; one slice gives that slice's Otsu.
On these coupons the specimen fills almost the whole frame, so a single slice
carries little exterior-air mode and its threshold lands differently — the same
failure documented for 192³ interior crops in
``runs/campaigns/05-eval-v3-fixed-decode/onlypores/``.  Dice here measures the
single-slice approximation as much as it measures the stored mask.

A volume is flagged when the two rules disagree, when the ambiguity gate fires,
or when the stored box covers less than ``--bbox-frac`` of the recomputed one —
the signature of a mask built on an artefact.

Usage
-----
    python scripts/analysis/sample_mask_component_audit.py
    python scripts/analysis/sample_mask_component_audit.py --limit 5    # smoke test

Writes ``runs/campaigns/15-sample-mask-audit/`` (results.json, per_volume.csv,
findings.md).  Read-only with respect to ``data/``; safe to run beside a
training job (no GPU, ~1 s of I/O per volume).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage import filters, measure
from skimage.measure import regionprops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from poregen.dataset.segmentation import (  # noqa: E402
    AMBIGUOUS_COMPONENT_RATIO,
    material_mask,
)

ZARR_ROOT = REPO / "data" / "split_v3" / "volumes.zarr"
SPLITS = REPO / "data" / "split_v3" / "splits.json"
OUT_DIR = REPO / "runs" / "campaigns" / "15-sample-mask-audit"


def bbox_of(mask2d: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(minr, minc, maxr, maxc)`` of the True pixels, or None if empty."""
    rows = np.flatnonzero(mask2d.any(axis=1))
    cols = np.flatnonzero(mask2d.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def bbox_area(bb: tuple[int, int, int, int] | None) -> int:
    if bb is None:
        return 0
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / denom


def volume_ids() -> list[str]:
    """Every volume the dataset uses, in a stable order."""
    splits = json.loads(SPLITS.read_text())
    excluded = set(splits.get("excluded_volume_ids", []))
    return sorted(v for v in splits["volumes"] if v not in excluded)


def audit_volume(grp) -> dict:
    """Everything measurable from one middle z-slice of one volume."""
    depth = grp["xct"].shape[0]
    z = depth // 2
    xct = np.asarray(grp["xct"][z])
    stored = np.asarray(grp["sample_mask"][z]) > 0

    threshold = float(filters.threshold_otsu(xct))
    proj = xct > threshold
    props = regionprops(measure.label(proj))
    areas = sorted((int(p.area) for p in props), reverse=True)

    row: dict = {
        "z_slice": z,
        "depth": depth,
        "shape": list(xct.shape),
        "slice_otsu_threshold": threshold,
        "n_components": len(props),
        "largest_area": areas[0] if areas else 0,
        "second_area": areas[1] if len(areas) > 1 else 0,
        "stored_mask_fraction": float(stored.mean()),
    }
    row["second_over_largest"] = (
        row["second_area"] / row["largest_area"] if row["largest_area"] else 0.0
    )
    row["ambiguous"] = bool(row["second_over_largest"] > AMBIGUOUS_COMPONENT_RATIO)

    if props:
        by_area = sorted(props, key=lambda p: p.area, reverse=True)
        row["first_label_area"] = int(props[0].area)
        row["first_label_bbox"] = [int(v) for v in props[0].bbox]
        row["largest_bbox"] = [int(v) for v in by_area[0].bbox]
        row["first_is_largest"] = bool(props[0].label == by_area[0].label)
    else:
        row["first_label_area"] = 0
        row["first_label_bbox"] = None
        row["largest_bbox"] = None
        row["first_is_largest"] = True

    stored_bb = bbox_of(stored)
    row["stored_bbox"] = list(stored_bb) if stored_bb else None
    row["stored_bbox_area"] = bbox_area(stored_bb)

    if row["ambiguous"]:
        # The fixed function refuses this slice; that refusal IS the finding.
        row["recomputed_bbox"] = None
        row["recomputed_bbox_area"] = 0
        row["recomputed_mask_fraction"] = None
        row["dice"] = None
        row["stored_over_recomputed_bbox"] = None
        row["error"] = "material_mask raises: ambiguous specimen selection"
    else:
        recomputed = material_mask(xct[None])[0]
        rec_bb = bbox_of(recomputed)
        row["recomputed_bbox"] = list(rec_bb) if rec_bb else None
        row["recomputed_bbox_area"] = bbox_area(rec_bb)
        row["recomputed_mask_fraction"] = float(recomputed.mean())
        row["dice"] = dice(stored, recomputed)
        row["stored_over_recomputed_bbox"] = (
            row["stored_bbox_area"] / row["recomputed_bbox_area"]
            if row["recomputed_bbox_area"] else None
        )
        row["error"] = None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N volumes (smoke test)")
    ap.add_argument("--bbox-frac", type=float, default=0.5,
                    help="flag a volume whose stored box is below this fraction "
                         "of the recomputed box")
    args = ap.parse_args()

    logging.getLogger("poregen").setLevel(logging.WARNING)
    ids = volume_ids()
    if args.limit:
        ids = ids[:args.limit]

    store = zarr.open(str(ZARR_ROOT), mode="r")
    rows = []
    t0 = time.time()
    for i, vid in enumerate(ids, 1):
        row = {"volume_id": vid, **audit_volume(store[vid])}
        flags = []
        if not row["first_is_largest"]:
            flags.append("first_label_is_not_largest")
        if row["ambiguous"]:
            flags.append("ambiguous_components")
        frac = row["stored_over_recomputed_bbox"]
        if frac is not None and frac < args.bbox_frac:
            flags.append("stored_bbox_much_smaller")
        row["flag_reasons"] = ",".join(flags)
        rows.append(row)
        print(f"[{i:3d}/{len(ids)}] {vid[:58]:58s} "
              f"comps={row['n_components']:5d} "
              f"2nd/1st={row['second_over_largest']:.4f} "
              f"first_is_largest={str(row['first_is_largest']):5s} "
              f"dice={row['dice'] if row['dice'] is None else round(row['dice'], 4)} "
              f"{row['flag_reasons']}", flush=True)

    df = pd.DataFrame(rows)
    flagged = df[df["flag_reasons"] != ""]
    dices = df.dice.dropna()
    summary = {
        "n_volumes": len(df),
        "zarr_root": str(ZARR_ROOT),
        "ambiguity_ratio": AMBIGUOUS_COMPONENT_RATIO,
        "bbox_frac_gate": args.bbox_frac,
        "n_first_label_is_not_largest": int((~df.first_is_largest).sum()),
        "n_ambiguous": int(df.ambiguous.sum()),
        "n_flagged": int(len(flagged)),
        "flagged_volume_ids": flagged.volume_id.tolist(),
        "dice_min": float(dices.min()) if len(dices) else None,
        "dice_median": float(dices.median()) if len(dices) else None,
        "dice_mean": float(dices.mean()) if len(dices) else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("\n" + json.dumps(summary, indent=2))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_volume.csv", index=False)
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "findings.md").write_text(
        "# Stored `sample_mask` component audit\n\n"
        "Does any stored `sample_mask` come from a dust speck instead of the "
        "specimen?  One middle z-slice per volume; see the script docstring for "
        "why `dice` is an approximation and `first_is_largest` is not.\n\n"
        f"- volumes audited: **{summary['n_volumes']}**\n"
        f"- first label is not the largest component: "
        f"**{summary['n_first_label_is_not_largest']}**\n"
        f"- ambiguous (second-largest > "
        f"{AMBIGUOUS_COMPONENT_RATIO:.0%} of largest): "
        f"**{summary['n_ambiguous']}**\n"
        f"- flagged: **{summary['n_flagged']}** "
        f"{summary['flagged_volume_ids']}\n"
        f"- Dice stored vs single-slice recompute: "
        f"min {summary['dice_min']}, median {summary['dice_median']}\n\n"
        "Source: `scripts/analysis/sample_mask_component_audit.py`\n"
    )
    print(f"\nWrote {out}")
    return 1 if summary["n_flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

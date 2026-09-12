"""What is actually different at a chunk plane? (CPU, read-only)

Two eval-v4 numbers say something is wrong at the chunk planes and neither says
what:

* ``seam_chunk_pore_ratio`` is far below 1 (0.39 on the 1024 sampler cases,
  0.73 on the multichunk box) while ``seam_chunk_xct_ratio`` sits near the real
  floor.  The ratio is ``seam_mad / interior_mad`` over mean absolute
  slice-to-slice differences, so **below 1 means the field changes LESS across
  the chunk plane than it does inside a chunk** — smoother, not more broken.
  A discontinuity would read well above 1.
* ``pore_dice_across_chunk_planes`` is 0.27, which looks alarming and cannot be
  read at all on its own: it is the Dice between two ADJACENT SLICES, and
  adjacent slices of any porous medium disagree, because pores are finite along
  the axis.  Without the same quantity measured away from a chunk plane there
  is no scale for it.

This script supplies what both numbers are missing:

1. **The pore-fraction profile** — phi (pore over material) in 8-voxel slabs
   stepping across each chunk plane, against the same statistic in the
   interior.  If the model puts fewer pores near the plane, phi dips there and
   the low seam ratio is explained by there being less pore structure to
   differ.
2. **The interior baseline for the adjacent-slice Dice** — the identical
   statistic at every non-chunk, non-window plane.  Only the gap between the
   two means anything.

Usage:
    python scripts/analysis/chunk_plane_profile.py \
        --root runs/campaigns/12-eval-v4 \
        --case sampler/1024_ddim100_seed101 --case multichunk/box384_ddim200_seed101
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2

#: Half-width of the slab the profile is reported in.
SLAB = 8


def phi_of(label_slab: np.ndarray) -> float | None:
    """Pore over material+pore. None when the slab holds no specimen."""
    pore = int((label_slab == LABEL_PORE).sum())
    solid = int((label_slab != LABEL_AIR).sum())
    return pore / solid if solid else None


def profile_axis(label: np.ndarray, axis: int, period: int, slab: int = SLAB) -> dict:
    """phi in `slab`-thick slabs stepping across each chunk plane on one axis."""
    n = label.shape[axis]
    planes = [i for i in range(period, n, period)]
    offsets = list(range(-4 * slab, 4 * slab, slab))
    per_plane = []
    for p in planes:
        row = {"plane": p, "phi_by_offset": {}}
        for off in offsets:
            lo, hi = p + off, p + off + slab
            if lo < 0 or hi > n:
                continue
            sl = np.take(label, range(lo, hi), axis=axis)
            row["phi_by_offset"][str(off)] = phi_of(sl)
        per_plane.append(row)
    return {"axis": axis, "period": period, "planes": planes,
            "slab_voxels": slab, "per_plane": per_plane}


def adjacent_slice_dice(label: np.ndarray, axis: int, idx: int) -> float | None:
    """Dice between the pore masks of slices idx-1 and idx."""
    a = np.take(label, idx - 1, axis=axis) == LABEL_PORE
    b = np.take(label, idx, axis=axis) == LABEL_PORE
    n = float(a.sum() + b.sum())
    return 2.0 * float((a & b).sum()) / n if n else None


def dice_baseline(label: np.ndarray, axis: int, chunk_period: int,
                  window_period: int = 64, max_planes: int = 64) -> dict:
    """The adjacent-slice Dice AT chunk planes and AWAY from any plane.

    The interior set excludes both the chunk planes and the window planes, so
    the baseline is ordinary material and not some other seam.
    """
    n = label.shape[axis]
    at_chunk, interior = [], []
    for idx in range(1, n):
        if idx % chunk_period == 0:
            d = adjacent_slice_dice(label, axis, idx)
            if d is not None:
                at_chunk.append(d)
        elif idx % window_period != 0:
            interior.append(idx)
    rng = np.random.default_rng(0)
    if len(interior) > max_planes:
        interior = sorted(rng.choice(interior, max_planes, replace=False).tolist())
    vals = [adjacent_slice_dice(label, axis, i) for i in interior]
    vals = [v for v in vals if v is not None]
    return {
        "axis": axis,
        "at_chunk_planes": {"n": len(at_chunk),
                            "mean": float(np.mean(at_chunk)) if at_chunk else None,
                            "values": at_chunk},
        "interior_planes": {"n": len(vals),
                            "mean": float(np.mean(vals)) if vals else None,
                            "sd": float(np.std(vals)) if vals else None},
    }


def analyse(root: Path, rel: str) -> dict:
    import tifffile

    assessment, case = rel.split("/", 1)
    case_dir = root / assessment / "volumes" / case
    manifest = json.loads((case_dir / "manifest.json").read_text())
    label = tifffile.imread(case_dir / "label.tif")

    tiles = manifest.get("chunk_tiles") or [3, 3, 3]
    period = [int(t) * 64 for t in tiles]

    out = {
        "case": rel,
        "volume_shape": list(label.shape),
        "chunk_tiles": list(tiles),
        "chunk_period_voxels": period,
        "requested_global_phi": manifest.get("requested_global_phi"),
        "ddim_steps": manifest.get("ddim_steps"),
        "phi_whole_volume": phi_of(label),
        "profiles": [],
        "dice": [],
    }
    for axis in range(3):
        if label.shape[axis] <= period[axis]:
            out["profiles"].append(
                {"axis": axis, "period": period[axis],
                 "skipped": "no chunk plane on this axis — the volume is one chunk deep"})
            continue
        out["profiles"].append(profile_axis(label, axis, period[axis]))
        out["dice"].append(dice_baseline(label, axis, period[axis]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True,
                    help="<assessment>/<case>, repeatable")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = [analyse(args.root, c) for c in args.case]
    out = args.out or (args.root / "chunk_plane_profile.json")
    out.write_text(json.dumps(results, indent=2) + "\n")

    for r in results:
        print(f"\n=== {r['case']}  shape={r['volume_shape']}  "
              f"chunk_period={r['chunk_period_voxels']}  "
              f"phi_volume={r['phi_whole_volume']:.5f} "
              f"(requested {r['requested_global_phi']}) ===")
        for prof in r["profiles"]:
            if "skipped" in prof:
                print(f"  axis {prof['axis']}: {prof['skipped']}")
                continue
            print(f"  axis {prof['axis']}, planes {prof['planes']}, "
                  f"phi in {prof['slab_voxels']}-voxel slabs by offset from the plane:")
            offs = sorted({int(o) for pl in prof["per_plane"] for o in pl["phi_by_offset"]})
            head = "      offset " + " ".join(f"{o:+5d}" for o in offs)
            print(head)
            for pl in prof["per_plane"]:
                cells = " ".join(
                    (f"{pl['phi_by_offset'][str(o)]:.3f}"
                     if pl["phi_by_offset"].get(str(o)) is not None else "  -  ")
                    for o in offs)
                print(f"      z={pl['plane']:<5d} {cells}")
        for d in r["dice"]:
            ac, it = d["at_chunk_planes"], d["interior_planes"]
            print(f"  axis {d['axis']} adjacent-slice pore Dice: "
                  f"at chunk planes {ac['mean']:.3f} (n={ac['n']}) "
                  f"vs interior {it['mean']:.3f} +/- {it['sd']:.3f} (n={it['n']})")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

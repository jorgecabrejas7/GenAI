"""What is actually different at a chunk plane? (CPU, read-only)

Two eval-v4 numbers said something was wrong at the chunk planes and neither
said what:

* ``seam_chunk_pore_ratio`` is far below 1 (0.39 on the 1024 sampler cases)
  while ``seam_chunk_xct_ratio`` sits near the real floor.  The ratio is
  ``seam_mad / interior_mad`` over mean absolute slice-to-slice differences, so
  **below 1 means the field changes LESS across the chunk plane than inside a
  chunk** — smoother, not more broken.  A discontinuity reads well above 1.
* ``pore_dice_across_chunk_planes`` is 0.27, which cannot be read alone: it is
  the Dice between two ADJACENT SLICES, and adjacent slices of any porous
  medium disagree.  Real 1024-wide test material scores 0.25-0.34 at every
  plane, chunk or not.

This supplies what both are missing: the pore-fraction profile in slabs
stepping across each plane, and the adjacent-slice Dice at interior planes of
the same volume.  Run on ldm06 it found the cause — phi collapses from ~0.028
to ~0.004 in the last 16 voxels before EVERY chunk plane and recovers over ~24
after, while a real crop is flat across the same planes.

PLANES ARE GIVEN, NOT DERIVED, when ``--planes`` is passed.  The assembly-mode
arms have different chunk periods by construction — ``joint`` is a single
chunk, ``autoregressive`` uses 64-voxel chunks, ``hybrid`` and
``teacher_forced`` use 192 — so profiling each at its own period would compare
different places.  Fixing the planes compares the same coordinates across arms,
which is what separates "the band follows the chunk boundary" from "the band is
at these coordinates regardless".

Usage:
    python scripts/analysis/chunk_plane_profile.py \
        --root runs/campaigns/12-eval-v4 \
        --case sampler/1024_ddim100_seed101 --planes 192 --axes 1 2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2

#: Slab thickness the profile is reported in.
SLAB = 8
#: Offsets from the plane, in slabs, covering the band and clean material on
#: both sides.  -8 is the slab ENDING at the plane; +0 is the slab starting on it.
OFFSETS = (-32, -24, -16, -8, 0, 8, 16, 24)


def phi_of(label_slab: np.ndarray) -> float | None:
    """Pore over solid (material + pore). None when the slab holds no specimen."""
    pore = int((label_slab == LABEL_PORE).sum())
    solid = int((label_slab != LABEL_AIR).sum())
    return pore / solid if solid else None


def phi_profile(label: np.ndarray, axis: int, planes: list[int],
                slab: int = SLAB) -> list[dict]:
    """phi in `slab`-thick slabs at each offset from each plane, one row per plane."""
    n = label.shape[axis]
    out = []
    for p in planes:
        if p <= 0 or p > n:
            continue
        row = {"plane": int(p), "phi": {}}
        for off in OFFSETS:
            lo, hi = p + off, p + off + slab
            if lo < 0 or hi > n:
                continue
            row["phi"][str(off)] = phi_of(np.take(label, range(lo, hi), axis=axis))
        out.append(row)
    return out


def band_mass(profile: list[dict], phi_volume: float, axis_len: int,
              slab: int = SLAB) -> float:
    """How much volume-mean phi the depleted slabs account for.

    Sum over every slab that sits BELOW the volume mean of the shortfall times
    the fraction of the axis that slab occupies.  If the band explains the
    arm's porosity deficit, this is the size of that deficit.
    """
    total = 0.0
    for row in profile:
        for _off, val in row["phi"].items():
            if val is None or val >= phi_volume:
                continue
            total += (phi_volume - val) * (slab / axis_len)
    return total


def adjacent_slice_dice(pore: np.ndarray, axis: int, idx: int) -> float | None:
    a = np.take(pore, idx - 1, axis=axis)
    b = np.take(pore, idx, axis=axis)
    n = float(a.sum() + b.sum())
    return 2.0 * float((a & b).sum()) / n if n else None


def dice_vs_interior(label: np.ndarray, axis: int, planes: list[int],
                     window_period: int = 64, n_interior: int = 64,
                     seed: int = 0) -> dict:
    """Adjacent-slice pore Dice at the given planes, and away from any plane."""
    pore = label == LABEL_PORE
    n = label.shape[axis]
    at = [d for p in planes if 0 < p < n
          and (d := adjacent_slice_dice(pore, axis, int(p))) is not None]
    planeset = set(int(p) for p in planes)
    inner = [i for i in range(1, n)
             if i not in planeset and (not window_period or i % window_period)]
    rng = np.random.default_rng(seed)
    if len(inner) > n_interior:
        inner = rng.choice(inner, n_interior, replace=False).tolist()
    base = [d for i in inner if (d := adjacent_slice_dice(pore, axis, int(i))) is not None]
    mean = float(np.mean(at)) if at else None
    bmean = float(np.mean(base)) if base else None
    return {
        "axis": axis, "n_planes": len(at), "at_planes": mean,
        "interior": bmean, "interior_sd": float(np.std(base)) if base else None,
        "n_interior": len(base),
        "ratio": (mean / bmean) if mean is not None and bmean else None,
    }


def analyse(root: Path, rel: str, planes_arg: list[int] | None,
            axes: list[int] | None) -> dict:
    import tifffile

    assessment, case = rel.split("/", 1)
    case_dir = root / assessment / "volumes" / case
    manifest = json.loads((case_dir / "manifest.json").read_text())
    label = tifffile.imread(case_dir / "label.tif")
    notes = manifest.get("notes") or {}

    tiles = manifest.get("chunk_tiles") or [3, 3, 3]
    own_period = [int(t) * 64 for t in tiles]
    phi_vol = phi_of(label)

    out = {
        "case": rel,
        "arm": notes.get("arm") or case.split("_")[0],
        "seed": manifest.get("seed"),
        "volume_shape": list(label.shape),
        "chunk_tiles": list(tiles),
        "own_chunk_period": own_period,
        "neighbour_mode": notes.get("neighbour_mode") or manifest.get("neighbour_mode"),
        "s_nb": manifest.get("s_nb"),
        "requested_global_phi": manifest.get("requested_global_phi"),
        "phi_volume": phi_vol,
        "axes": {},
    }
    for axis in (axes if axes is not None else range(3)):
        n = label.shape[axis]
        period = planes_arg[0] if planes_arg else own_period[axis]
        planes = ([p for p in planes_arg if 0 < p <= n] if planes_arg and len(planes_arg) > 1
                  else list(range(period, n, period)))
        if not planes:
            out["axes"][str(axis)] = {"skipped": f"no plane at period {period} in {n} voxels"}
            continue
        prof = phi_profile(label, axis, planes)
        out["axes"][str(axis)] = {
            "planes": planes,
            "per_plane": prof,
            "band_mass": band_mass(prof, phi_vol, n) if phi_vol else None,
            "dice": dice_vs_interior(label, axis, planes),
        }
    return out


def aggregate(results: list[dict], axes: list[int]) -> dict:
    """Mean over seeds, per arm and offset."""
    # The cfg cases all carry the same `arm` note and differ only in s_nb, so
    # grouping on the arm alone would average the very pair under test.
    many_snb = len({r.get("s_nb") for r in results if r.get("s_nb") is not None}) > 1
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = r["arm"]
        if many_snb and r.get("s_nb") is not None:
            key = f"{key}={r['s_nb']:g}"
        by_arm[key].append(r)
    table = {}
    for arm, rows in by_arm.items():
        vals: dict[str, list[float]] = defaultdict(list)
        masses, phis, dice_at, dice_in = [], [], [], []
        for r in rows:
            phis.append(r["phi_volume"])
            for axis in axes:
                blk = r["axes"].get(str(axis))
                if not blk or "skipped" in blk:
                    continue
                for pl in blk["per_plane"]:
                    for off, v in pl["phi"].items():
                        if v is not None:
                            vals[off].append(v)
                if blk["band_mass"] is not None:
                    masses.append(blk["band_mass"])
                d = blk["dice"]
                if d["at_planes"] is not None:
                    dice_at.append(d["at_planes"])
                if d["interior"] is not None:
                    dice_in.append(d["interior"])
        table[arm] = {
            "n_volumes": len(rows),
            "seeds": sorted(r["seed"] for r in rows),
            "neighbour_mode": rows[0]["neighbour_mode"],
            "own_chunk_period": rows[0]["own_chunk_period"],
            "phi_volume": float(np.mean(phis)),
            "phi_by_offset": {o: float(np.mean(v)) for o, v in sorted(
                vals.items(), key=lambda kv: int(kv[0]))},
            "band_mass_per_axis": float(np.mean(masses)) if masses else None,
            "dice_at_planes": float(np.mean(dice_at)) if dice_at else None,
            "dice_interior": float(np.mean(dice_in)) if dice_in else None,
        }
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True,
                    help="<assessment>/<case>, repeatable")
    ap.add_argument("--planes", type=int, nargs="*", default=None,
                    help="explicit plane coordinates, or one value used as a "
                         "period. Omitted: the case's own chunk period.")
    ap.add_argument("--axes", type=int, nargs="*", default=None)
    ap.add_argument("--label", default="", help="tag for the output file")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    axes = args.axes if args.axes is not None else [0, 1, 2]
    results = [analyse(args.root, c, args.planes, axes) for c in args.case]
    table = aggregate(results, axes)

    out = args.out or (args.root / f"chunk_plane_profile{args.label}.json")
    out.write_text(json.dumps({"per_case": results, "by_arm": table}, indent=2) + "\n")

    offs = [str(o) for o in OFFSETS]
    print(f"\nphi by offset from the plane (slab {SLAB} voxels), mean over seeds "
          f"and over axes {axes}")
    print(f"{'arm':<16}{'nb_mode':<11}{'period':>8}{'phi_vol':>9}  "
          + "".join(f"{o:>8}" for o in offs) + f"{'band':>9}")
    for arm, t in sorted(table.items()):
        cells = "".join(
            (f"{t['phi_by_offset'][o]:>8.4f}" if o in t["phi_by_offset"] else f"{'-':>8}")
            for o in offs)
        per = t["own_chunk_period"][axes[0]] if t["own_chunk_period"] else 0
        bm = t["band_mass_per_axis"]
        print(f"{arm:<16}{str(t['neighbour_mode']):<11}{per:>8}{t['phi_volume']:>9.4f}  "
              f"{cells}{(f'{bm:>9.4f}' if bm is not None else f'{chr(45):>9}')}")
    print(f"\n{'arm':<16}{'dice@planes':>13}{'dice interior':>15}{'ratio':>8}")
    for arm, t in sorted(table.items()):
        a, i = t["dice_at_planes"], t["dice_interior"]
        r = (a / i) if a is not None and i else None
        fmt = lambda v: ("-" if v is None else f"{v:.3f}")  # noqa: E731
        print(f"{arm:<16}{fmt(a):>13}{fmt(i):>15}{fmt(r):>8}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Porosity of the REAL volumes, whole and by region. (CPU, index only.)

Every generated-volume porosity number in this project is read against a real
floor taken from crops. This is the floor for the volumes THEMSELVES: what the
80 dataset volumes deliver over their whole extent, and whether that number
moves between the front, middle and back third of each axis.

WHERE THE NUMBERS COME FROM, AND WHAT THIS DOES NOT TOUCH
--------------------------------------------------------
`data/split_v3/patch_index.parquet` alone, plus the per-volume shapes in
`index_report.json`. The zarr is never opened and nothing is streamed: a
training run holds the card and the page cache, and 195 GB of store traffic
beside it is how a CUDA allocation fails while `free` still reports tens of GB.

The index has no raw voxel counts — it has `porosity` (pore / 64**3) and
`air_fraction` (air / 64**3), both written by `patch_index.patch_fractions`
from an integral volume. The counts come back EXACTLY, not approximately:
64**3 is 2**18 and a patch holds at most 2**18 voxels, so `count / 2**18` is a
dyadic rational inside a float32's 24-bit mantissa and is stored without loss.
`count = round(fraction * 64**3)` therefore recovers the integer the integral
volume produced. That is the cheapest alternative to per-patch counts, and it
is not a degraded one.

phi IS MATERIAL POROSITY — pore / (pore + solid), air excluded from the
denominator — the same definition `metrics.phase_fractions` reports as
`phi_pore` and the same one every eval-v4 table uses. Air here is
`sample_mask == 0`: outside the specimen.

    phi = sum(pore) / sum(non-air)  over the patches in the region

NON-OVERLAPPING PATCHES ONLY. The index is built at stride 32 for a 64-voxel
patch, so every voxel appears in eight rows. Summing all of them would weight
the interior eight times and the border once, which is a weighted mean wearing
a pooled mean's name. Keeping `z0 % 64 == y0 % 64 == x0 % 64 == 0` takes a true
partition: every voxel counted once, or not at all.

Patches dropped for holes are absent from the index, so a region reports the
material that survived that filter. `n_patches` per region says how much that
was; a region with none reports null rather than 0.

Usage:
    python scripts/analysis/real_porosity_by_region.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("real_porosity_by_region")

#: Patch edge, voxels. Asserted against the index rather than assumed.
PS = 64
#: Voxels per patch. 2**18, which is why the float32 fractions are lossless.
PATCH_VOX = PS ** 3
#: The three regions, in order, along every axis.
REGIONS = ("front", "middle", "back")
#: (z, y, x). z is the THROUGH-THICKNESS axis: the volumes are about 224 deep
#: and 1600-3200 wide in the other two, and the plies stack along z. "front" is
#: the z = 0 face.
AXES = ("z", "y", "x")


def load_index(data_root: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(data_root / "patch_index.parquet")
    report = json.loads((data_root / "index_report.json").read_text())
    shapes = {v["volume_id"]: tuple(v["shape"]) for v in report["per_volume"]}
    panels = {v["volume_id"]: v["panel_id"] for v in report["per_volume"]}
    if not (df["ps"] == PS).all():
        raise ValueError(f"the index holds patch sizes {sorted(df['ps'].unique())}, expected {PS}")
    return df, {"shapes": shapes, "panels": panels}


def non_overlapping(df: pd.DataFrame) -> pd.DataFrame:
    """The stride-64 partition: every voxel in exactly one patch, or in none."""
    keep = (df["z0"] % PS == 0) & (df["y0"] % PS == 0) & (df["x0"] % PS == 0)
    return df.loc[keep].copy()


def with_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Exact pore and non-air voxel counts, recovered from the fractions."""
    out = df
    out["pore_vox"] = np.rint(out["porosity"].to_numpy(np.float64) * PATCH_VOX)
    out["air_vox"] = np.rint(out["air_fraction"].to_numpy(np.float64) * PATCH_VOX)
    out["solid_vox"] = PATCH_VOX - out["air_vox"]
    return out


def phi(pore: float, solid: float) -> float | None:
    """Material porosity, or None when the region holds no specimen at all."""
    return float(pore / solid) if solid > 0 else None


def third(centre: np.ndarray, extent: int) -> np.ndarray:
    """Region index 0/1/2 from a patch centre and the volume's own extent.

    Thirds of the TRUE volume shape, not of the span the patches happen to
    cover: a volume whose last patch row was dropped for a hole still has a
    back third, and measuring against the patch span would quietly move the
    boundary per volume and make two volumes' "back" different places.
    """
    edges = (extent / 3.0, 2.0 * extent / 3.0)
    return np.digitize(centre, edges)


def per_volume_rows(df: pd.DataFrame, meta: dict) -> list[dict]:
    rows = []
    for vid, g in df.groupby("volume_id", sort=True):
        shape = meta["shapes"].get(vid)
        if shape is None:
            raise KeyError(f"{vid} is in the patch index but not in index_report.json")
        centres = {"z": g["z0"].to_numpy() + PS / 2.0,
                   "y": g["y0"].to_numpy() + PS / 2.0,
                   "x": g["x0"].to_numpy() + PS / 2.0}
        pore = g["pore_vox"].to_numpy()
        solid = g["solid_vox"].to_numpy()
        regions = {}
        for ax_i, ax in enumerate(AXES):
            idx = third(centres[ax], shape[ax_i])
            regions[ax] = {
                name: {
                    "phi": phi(pore[idx == r].sum(), solid[idx == r].sum()),
                    "n_patches": int((idx == r).sum()),
                }
                for r, name in enumerate(REGIONS)
            }
        rows.append({
            "volume_id": vid,
            "panel": meta["panels"].get(vid),
            "split": str(g["split"].iloc[0]),
            "shape": list(shape),
            "n_patches": int(len(g)),
            "phi_whole": phi(pore.sum(), solid.sum()),
            "regions": regions,
        })
    return rows


def z_slab_profile(df: pd.DataFrame) -> dict:
    """phi per 64-voxel z SLAB, indexed from the z = 0 face.

    NOT what was asked for, and here because the thirds cannot answer the
    question on this axis. The volumes are 185-332 voxels deep, so the
    non-overlapping partition lays down only two to five slabs along z, and
    WHICH THIRD a slab falls in depends on the volume's own depth: at 224 the
    third slab's centre (160) is in the back third, at 246 the same slab is in
    the middle, and at 250 there is no back third at all. Ten volumes report an
    empty z back third for exactly that reason — the next slab would start at
    192 and need a 256-deep volume to fit.

    The slab index has none of that: slab 0 is the first 64 voxels from the
    z = 0 face in every volume, whatever its depth. This is the through-
    thickness profile the z thirds were asked for, on a unit that means the same
    thing in every volume. y and x are 1600-3200 wide and carry dozens of slabs,
    so their thirds are unaffected and are not repeated here.
    """
    out = {}
    for slab, g in df.groupby(df["z0"] // PS, sort=True):
        per_vol = [phi(gg["pore_vox"].sum(), gg["solid_vox"].sum())
                   for _, gg in g.groupby("volume_id", sort=True)]
        out[f"slab{int(slab)}"] = {
            "z_range": [int(slab) * PS, (int(slab) + 1) * PS],
            "n_volumes": len(per_vol),
            "voxel_pooled_phi": phi(g["pore_vox"].sum(), g["solid_vox"].sum()),
            **stats(per_vol),
        }
    return out


def stats(values) -> dict:
    """Across volumes. `n` counts the volumes that HAVE the quantity."""
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return {"mean": None, "median": None, "std": None,
                "min": None, "max": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def build_results(df: pd.DataFrame, meta: dict) -> dict:
    rows = per_volume_rows(df, meta)
    summary = {
        "whole": stats([r["phi_whole"] for r in rows]),
        "regions": {
            ax: {name: stats([r["regions"][ax][name]["phi"] for r in rows])
                 for name in REGIONS}
            for ax in AXES
        },
        # The volume-level statistics above weight every volume equally,
        # whatever its size. This one weights every VOXEL equally, and the two
        # answer different questions: a 3204-long panel and a 1600-long one
        # count the same above and not here.
        "voxel_pooled_phi": phi(df["pore_vox"].sum(), df["solid_vox"].sum()),
    }
    empty = {
        f"{ax}_{name}": [r["volume_id"] for r in rows
                         if r["regions"][ax][name]["n_patches"] == 0]
        for ax in AXES for name in REGIONS
    }
    empty = {k: v for k, v in empty.items() if v}
    return {
        "campaign": "21-real-porosity-by-region",
        "question": ("What porosity do the 80 real volumes deliver over their "
                     "whole extent, and does it move between the front, middle "
                     "and back third of each axis?"),
        "definition": {
            "phi": ("material porosity: pore / (pore + solid), air excluded from "
                    "the denominator. The same definition metrics.phase_fractions "
                    "reports as phi_pore."),
            "air": "sample_mask == 0 — outside the specimen.",
            "source": ("data/split_v3/patch_index.parquet and the per-volume "
                       "shapes in index_report.json. The zarr is never opened."),
            "counts": (f"the index stores porosity and air_fraction as fractions "
                       f"of {PATCH_VOX} = 2**18 voxels. A count/2**18 is a dyadic "
                       "rational inside float32's 24-bit mantissa, so multiplying "
                       "back recovers the integral volume's own integer exactly."),
            "patches": (f"NON-OVERLAPPING only: z0, y0 and x0 all divisible by "
                        f"{PS}. The index is built at stride 32, so every voxel "
                        "appears in eight rows; summing all of them would weight "
                        "the interior eight times and the border once."),
            "dropped_patches": ("patches dropped for holes are absent from the "
                                "index, so a region reports the material that "
                                "survived that filter — n_patches says how much."),
            "axes": ("(z, y, x). z is THROUGH-THICKNESS: the volumes are about "
                     "224 voxels deep and 1600-3200 wide in y and x, and the "
                     "plies stack along z. 'front' is the z = 0 face; 'back' is "
                     "the far face."),
            "regions": ("thirds of the TRUE volume shape, by patch CENTRE "
                        "(coordinate + 32). Thirds of the patch span instead "
                        "would move the boundary per volume and make two "
                        "volumes' 'back' different places."),
            "pooling": ("all 80 volumes together — train, val and test. No split "
                        "breakdown: the question is about the material, and the "
                        "split is a property of the experiment."),
        },
        "n_volumes": len(rows),
        "empty_regions": {
            "note": (
                "a region with no patch CENTRE in it. The partition lays 64-voxel "
                "slabs, so on the 185-332-voxel z axis only two to five of them "
                "fit and which third a slab's centre falls in depends on the "
                "volume's own depth. At 241-255 voxels deep the third slab's "
                "centre (160) is short of the back third's start, and the fourth "
                "slab would need a 256-deep volume; at 185-191 the same happens "
                "one slab earlier. y and x are 1600-3200 wide and carry dozens of "
                "slabs, so none of their regions is empty. Read `z_slabs` for the "
                "through-thickness profile instead."),
            "counts": {k: len(v) for k, v in empty.items()},
            "volumes": empty,
        },
        "z_slabs": z_slab_profile(df),
        "per_volume": rows,
        "summary": summary,
    }


def findings(res: dict) -> str:
    s = res["summary"]

    def row(label, d):
        if d["n"] == 0:
            return [label, "--", "--", "--", "--", "--", "0"]
        return [label, f"{d['mean']:.5f}", f"{d['median']:.5f}", f"{d['std']:.5f}",
                f"{d['min']:.5f}", f"{d['max']:.5f}", str(d["n"])]

    head = "| region | mean | median | sd | min | max | n |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep, "| " + " | ".join(row("WHOLE", s["whole"])[0:]) .replace("| ", "") + " |"]
    lines = [head, sep]
    lines.append("| " + " | ".join(row("**whole volume**", s["whole"])) + " |")
    for ax in AXES:
        for name in REGIONS:
            lines.append("| " + " | ".join(row(f"{ax} {name}", s["regions"][ax][name])) + " |")
    table = "\n".join(lines)

    slabs = res["z_slabs"]
    slab_rows = []
    for name in sorted(slabs, key=lambda k: int(k.replace("slab", ""))):
        b = slabs[name]
        slab_rows.append("| " + " | ".join([
            f"{name} (z {b['z_range'][0]}-{b['z_range'][1]})",
            f"{b['mean']:.5f}" if b["mean"] is not None else "--",
            f"{b['median']:.5f}" if b["median"] is not None else "--",
            f"{b['voxel_pooled_phi']:.5f}" if b["voxel_pooled_phi"] is not None else "--",
            str(b["n_volumes"]),
        ]) + " |")
    slab_table = "\n".join(
        ["| z slab | mean | median | voxel-pooled | n volumes |",
         "|---|---|---|---|---|"] + slab_rows)
    empty = res["empty_regions"]
    empty_note = ""
    if empty["counts"]:
        which = ", ".join(f"`{k}` ({n} volumes)" for k, n in empty["counts"].items())
        empty_note = (
            f"\n**{which} is empty** — no patch centre falls there. "
            + empty["note"] + "\n")

    d = res["definition"]
    return f"""# 21 — real porosity, whole volumes and by region

**Question.** {res["question"]}

Over **{res["n_volumes"]} volumes**, train, val and test pooled.

## Porosity across volumes

{table}

Every row is the spread ACROSS volumes of that volume's own pooled porosity.
Voxel-pooled phi over every volume at once: **{s["voxel_pooled_phi"]:.5f}** —
the volume-level mean weights each volume equally whatever its size, and this
weights each voxel equally. They answer different questions and the gap between
them is the size-porosity correlation.

{empty_note}
## Through the thickness, by slab

The z thirds above cannot answer the through-thickness question on their own,
for the reason in the note. This is the same material cut on a unit that means
the same thing in every volume: slab 0 is the first 64 voxels from the z = 0
face, whatever the volume's depth. It is an addition to what was asked for, not
a replacement for it.

{slab_table}

## What phi means here

- {d["phi"]}
- Air is {d["air"]}
- {d["axes"]}
- {d["regions"]}

## What was read, and what was not

- {d["source"]}
- {d["counts"]}
- {d["patches"]}
- {d["dropped_patches"]}
- {d["pooling"]}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "split_v3")
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "21-real-porosity-by-region")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    df, meta = load_index(args.data_root)
    logger.info("index: %d rows, %d volumes", len(df), df["volume_id"].nunique())
    part = with_counts(non_overlapping(df))
    logger.info("non-overlapping partition: %d patches (%.1f%% of the rows)",
                len(part), 100.0 * len(part) / len(df))
    res = build_results(part, meta)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    (args.out / "findings.md").write_text(findings(res))
    logger.info("wrote %s", args.out / "results.json")
    print(findings(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

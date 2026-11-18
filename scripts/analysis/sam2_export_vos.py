#!/usr/bin/env python
"""Export a labelled volume as DAVIS-style videos for SAM2 fine-tuning (campaign 40).

Input: a grey TIFF and an air TIFF with the same shape (from
``overlay_to_air_label.py``).  The volume is put thickness-first (T, A, B) and
percentile-normalised (1-99 % over the whole volume, as in ``fm_sam2_air.py``).
Per column, object 1 = upper air (air rows above the specimen run), object 2 =
lower air (below it); the specimen run is the longest non-air run.

Videos (tiles of ``--tile`` px, no overlap, last tile flush with the edge):
- time axis A: frames T x tile of B, one video per A range per B tile;
- time axis B: frames T x tile of A, tiles inside each A range.
Writes <out>/JPEGImages/<video>/<frame:05d>.jpg (quality 100) and
<out>/Annotations/<video>/<frame:05d>.png (palette PNG, 0/1/2), and
<out>/videos.txt (one video name per line).

    python scripts/analysis/sam2_export_vos.py --grey <grey.tif> --air <air.tif> --thickness-axis 1 \
        --ranges 0:750 1150:1450 --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

PALETTE = [0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (256 * 3 - 9)


def objects(air: np.ndarray) -> np.ndarray:
    """(T, A, B) bool air -> uint8 0 specimen, 1 upper air, 2 lower air (longest non-air run per column)."""
    T = air.shape[0]
    cur = np.zeros(air.shape[1:], np.int32)
    best = np.zeros_like(cur)
    end = np.full(cur.shape, -1, np.int32)
    for t in range(T):
        cur = np.where(air[t], 0, cur + 1)
        b = cur > best
        best = np.where(b, cur, best)
        end = np.where(b, t, end)
    top = end - best + 1
    t = np.arange(T)[:, None, None]
    lab = np.zeros(air.shape, np.uint8)
    lab[t < top[None]] = 1
    lab[t > end[None]] = 2
    return lab


def tiles(n: int, size: int) -> list[int]:
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, size))
    return s if s[-1] == n - size else s + [n - size]


def write_video(out: Path, name: str, frames: np.ndarray, labels: np.ndarray) -> None:
    (out / "JPEGImages" / name).mkdir(parents=True, exist_ok=True)
    (out / "Annotations" / name).mkdir(parents=True, exist_ok=True)
    for i, (f, m) in enumerate(zip(frames, labels)):
        Image.fromarray(np.repeat(f[..., None], 3, -1)).save(out / "JPEGImages" / name / f"{i:05d}.jpg", quality=100)
        p = Image.fromarray(m, mode="P")
        p.putpalette(PALETTE)
        p.save(out / "Annotations" / name / f"{i:05d}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grey", type=Path, required=True)
    ap.add_argument("--air", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--ranges", nargs="+", required=True, help="a0:a1 ranges on axis A")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--time-axes", nargs="+", default=["A", "B"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    g = np.moveaxis(tifffile.imread(a.grey), a.thickness_axis, 0)
    lo, hi = np.percentile(g, [1, 99])
    g8 = (np.clip((g.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    del g
    lab = objects(np.moveaxis(tifffile.imread(a.air) > 0, a.thickness_axis, 0))
    names = []
    for r in a.ranges:
        a0, a1 = map(int, r.split(":"))
        G, L = g8[:, a0:a1], lab[:, a0:a1]  # (T, A', B)
        if "A" in a.time_axes:
            for b0 in tiles(G.shape[2], a.tile):
                n = f"tA_a{a0}-{a1}_b{b0}"
                write_video(a.out, n, np.moveaxis(G[:, :, b0:b0 + a.tile], 1, 0), np.moveaxis(L[:, :, b0:b0 + a.tile], 1, 0))
                names.append(n)
        if "B" in a.time_axes:
            for t0 in tiles(G.shape[1], a.tile):
                n = f"tB_a{a0 + t0}-{a0 + t0 + min(a.tile, G.shape[1])}"
                write_video(a.out, n, np.moveaxis(G[:, t0:t0 + a.tile, :], 2, 0), np.moveaxis(L[:, t0:t0 + a.tile, :], 2, 0))
                names.append(n)
        print(r, "done", len(names), flush=True)
    (a.out / "videos.txt").write_text("\n".join(names) + "\n")
    (a.out / "export.json").write_text(json.dumps({"grey": str(a.grey), "air": str(a.air), "ranges": a.ranges,
                                                   "tile": a.tile, "time_axes": a.time_axes, "norm_p1_p99": [float(lo), float(hi)],
                                                   "videos": len(names)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

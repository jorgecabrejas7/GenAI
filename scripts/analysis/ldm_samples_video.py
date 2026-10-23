#!/usr/bin/env python
"""The LDM's in-training sample volumes beside a real crop, as an MP4 and a PNG. (CPU.)

The trainer writes one 192³ volume every ``sample_every`` steps
(``<run>/samples/step_N/{xct,label}.tif``: EMA weights, the production
``VolumeGenerator``, 3x3x3 tiles, DDIM-50, global porosity ``sample_global_por``).
This script puts the chosen steps next to a REAL 192³ crop from a val volume:
the air-free crop whose porosity is nearest the sample target, searched on a
64-voxel grid.

The video and the PNG share one layout: one column per volume, one row per
axis (z, y, x planes), grey with pores in red and air in blue. Frame i shows
slice i along each row's axis, so the video spans every slice; the PNG is the
central slice.

Usage:
    python scripts/analysis/ldm_samples_video.py --run runs/ldm/<run> \\
        --steps 19999 39999 59999 --out <dir>/name   (writes name.mp4, name.png)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tifffile
import yaml
import zarr

REPO = Path(__file__).resolve().parents[2]
PORE, AIR = 1, 2
GAP, HEAD = 6, 30


def real_crop(split: str, size: int, target: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Air-free size³ crop of a val volume with porosity nearest ``target``."""
    root = REPO / "data" / split
    idx = pd.read_parquet(root / "patch_index.parquet")
    vols = sorted(idx.loc[idx["split"] == "val", "volume_id"].unique())
    z = zarr.open(str(root / "volumes.zarr"), mode="r")
    best = None
    for v in vols:
        lab = np.asarray(z[v]["mask"])
        D, H, W = lab.shape
        if D < size:
            continue
        z0 = (D - size) // 2
        slab = lab[z0:z0 + size]
        for y0 in range(0, H - size + 1, 64):
            for x0 in range(0, W - size + 1, 64):
                c = slab[:, y0:y0 + size, x0:x0 + size]
                if (c == AIR).any():
                    continue
                d = abs(float((c == PORE).mean()) - target)
                if best is None or d < best[0]:
                    best = (d, v, z0, y0, x0)
    _, v, z0, y0, x0 = best
    s = np.s_[z0:z0 + size, y0:y0 + size, x0:x0 + size]
    name = v.split("probetas_")[-1].replace("_volume_eq_aligned", "")
    return np.asarray(z[v]["xct"][s]), np.asarray(z[v]["mask"][s]), f"real {name}"


def overlay(g: np.ndarray, lab: np.ndarray) -> np.ndarray:
    rgb = np.repeat(g[..., None], 3, axis=-1).astype(np.uint8)
    rgb[lab == PORE] = (230, 40, 40)
    rgb[lab == AIR] = (40, 90, 230)
    return rgb


def text(img: np.ndarray, s: str, x: int, y: int, size: float = 0.5) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO / a.run
    target = float(yaml.safe_load(open(run / "resolved_config.yaml"))["training"]["sample_global_por"])

    panels = []
    for s in a.steps:
        d = run / "samples" / f"step_{s:08d}"
        g, lab = tifffile.imread(d / "xct.tif"), tifffile.imread(d / "label.tif")
        panels.append((f"step {s + 1}  phi {float((lab == PORE).mean()):.4f}", g, lab))
    size = panels[0][1].shape[0]
    g, lab, name = real_crop(a.split, size, target)
    panels.insert(0, (f"{name}  phi {float((lab == PORE).mean()):.4f}", g, lab))
    print(json.dumps({"target": target, "panels": [p[0] for p in panels]}))

    def frame(i: int) -> np.ndarray:
        rows = []
        for ax in range(3):
            row = []
            for _, g, lb in panels:
                sl = [slice(None)] * 3
                sl[ax] = i
                row.append(overlay(g[tuple(sl)], lb[tuple(sl)]))
                row.append(np.zeros((size, GAP, 3), np.uint8))
            rows.append(np.concatenate(row[:-1], axis=1))
            rows.append(np.zeros((GAP, rows[-1].shape[1], 3), np.uint8))
        head = np.zeros((HEAD, rows[0].shape[1], 3), np.uint8)
        for k, p in enumerate(panels):
            text(head, p[0], k * (size + GAP) + 4, HEAD - 10, 0.42)
        foot = np.zeros((HEAD, rows[0].shape[1], 3), np.uint8)
        text(foot, f"slice {i + 1}/{size}  rows: z, y, x planes  target phi {target}  red pore, blue air", 4, 20, 0.42)
        img = np.concatenate([head] + rows[:-1] + [foot], axis=0)
        return np.pad(img, ((0, (-img.shape[0]) % 16), (0, (-img.shape[1]) % 16), (0, 0)))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out.with_suffix(".mp4"), fps=a.fps, codec="libx264", quality=9,
                                pixelformat="yuv420p", macro_block_size=16)
    for i in range(size):
        writer.append_data(frame(i))
    writer.close()
    imageio.imwrite(a.out.with_suffix(".png"), frame(size // 2))
    print("wrote", a.out.with_suffix(".mp4"), a.out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

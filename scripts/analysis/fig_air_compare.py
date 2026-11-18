#!/usr/bin/env python
"""Zoomed comparison figure of air masks on one volume (campaign 40).

Thickness-first grey (T, A, B).  For each requested slice (``A:<i>`` = plane
T x B at A=i, ``B:<j>`` = plane T x A at B=j) and window on the in-plane axis,
draws each mask's outline (1 px, own colour) on the grey, magnified by
``--zoom``, with a legend strip.  Also writes agreement.json: air fraction per
mask and per-column surface difference (median, p95, in voxels) of every mask
against the first one, using the longest non-air run per column.

    python scripts/analysis/fig_air_compare.py --grey <grey.tif> \
        --mask "SAM2 fine-tuned=<air.tif>" --mask "U-Net c39=<air.tif>" \
        --slice A:112:0:500 --slice B:785:0:225 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_air import surfaces  # noqa: E402

COLOURS = [(255, 40, 40), (0, 200, 255), (255, 210, 0), (60, 230, 90), (230, 80, 230)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grey", type=Path, required=True)
    ap.add_argument("--mask", action="append", required=True, help="label=path (thickness-first)")
    ap.add_argument("--slice", action="append", required=True, help="A|B:index:w0:w1")
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    g = tifffile.imread(a.grey)
    names, masks = [], []
    for s in a.mask:
        n, p = s.split("=", 1)
        names.append(n)
        masks.append(tifffile.imread(p) > 0)

    for s in a.slice:
        ax, i, w0, w1 = s.split(":")
        i, w0, w1 = int(i), int(w0), int(w1)
        sl = np.s_[:, i, w0:w1] if ax == "A" else np.s_[:, w0:w1, i]
        rgb = np.repeat(g[sl][..., None], 3, -1)
        rgb = np.kron(rgb, np.ones((a.zoom, a.zoom, 1), np.uint8))
        for k, m in enumerate(masks):
            mz = np.kron(m[sl].astype(np.uint8), np.ones((a.zoom, a.zoom), np.uint8)).astype(bool)
            rgb[mz & ~ndimage.binary_erosion(mz)] = COLOURS[k % len(COLOURS)]
        im = Image.fromarray(rgb)
        leg = Image.new("RGB", (im.width, 18 * len(names) + 6), (255, 255, 255))
        d = ImageDraw.Draw(leg)
        for k, n in enumerate(names):
            d.rectangle([6, 6 + 18 * k, 20, 18 + 18 * k], fill=COLOURS[k % len(COLOURS)])
            d.text((28, 6 + 18 * k), n, fill=(0, 0, 0))
        d.text((im.width - 220, 6), f"slice {ax}={i}, {w0}:{w1}, x{a.zoom}", fill=(0, 0, 0))
        out = Image.new("RGB", (im.width, im.height + leg.height), (255, 255, 255))
        out.paste(im, (0, 0))
        out.paste(leg, (0, im.height))
        iio.imwrite(a.out / f"compare_{ax}{i}_{w0}-{w1}.png", np.asarray(out))

    t0, b0 = surfaces(masks[0])
    agree = {"reference_for_differences": names[0], "masks": {}}
    for n, m in zip(names, masks):
        t, b = surfaces(m)
        dt, db = (t - t0).ravel(), (b - b0).ravel()
        agree["masks"][n] = {"air_fraction": float(m.mean()),
                             "top_median_signed": float(np.median(dt)), "bottom_median_signed": float(np.median(db)),
                             "abs_p95": float(np.percentile(np.abs(np.r_[dt, db]), 95))}
    (a.out / "agreement.json").write_text(json.dumps(agree, indent=1))
    print(json.dumps(agree, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

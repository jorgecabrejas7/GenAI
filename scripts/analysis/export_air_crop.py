#!/usr/bin/env python
"""Cut a thickness-first crop (grey + reference air) for the campaign-40 air tests.

Sources: a split_v4 coupon (thickness axis 0; air = sample_mask == 0, the
reference pipeline's mask) or a grey TIFF with an air TIFF (for example the
CuartoPulg overlay split).  The crop is on in-plane axis A (the first axis
after the thickness axis).  Writes <out>/grey.tif and <out>/air.tif (0/255),
shape (T, A, B).

    python scripts/analysis/export_air_crop.py --zarr Na_05_1 --crop 1500:1800 --out <dir>
    python scripts/analysis/export_air_crop.py --tif <grey.tif> --air <air.tif> --thickness-axis 1 \
        --crop 800:1100 --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", help="split_v4 volume key substring")
    ap.add_argument("--tif", type=Path)
    ap.add_argument("--air", type=Path)
    ap.add_argument("--thickness-axis", type=int, default=0)
    ap.add_argument("--crop", required=True, help="a0:a1 on in-plane axis A")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a0, a1 = map(int, a.crop.split(":"))
    a.out.mkdir(parents=True, exist_ok=True)
    if a.zarr:
        import zarr  # noqa: PLC0415
        z = zarr.open(str(REPO / "data/split_v4/volumes.zarr"), mode="r")
        key = [k for k in z.keys() if a.zarr in k][0]
        g = np.asarray(z[key]["xct"][:, a0:a1])
        air = np.asarray(z[key]["sample_mask"][:, a0:a1]) == 0
        src = key
    else:
        g = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0)[:, a0:a1]
        air = np.moveaxis(tifffile.imread(a.air) > 0, a.thickness_axis, 0)[:, a0:a1]
        src = str(a.tif)
    tifffile.imwrite(a.out / "grey.tif", np.ascontiguousarray(g), imagej=True)
    tifffile.imwrite(a.out / "air.tif", (air * 255).astype(np.uint8), imagej=True)
    rep = {"source": src, "crop_A": [a0, a1], "shape_TAB": list(g.shape), "air_fraction": float(air.mean())}
    (a.out / "crop.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

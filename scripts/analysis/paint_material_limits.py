#!/usr/bin/env python
"""Paint the upper and lower material surfaces on a grey volume (campaign 39).

From a 3-class label (0 material, 1 pore, 2 air), the specimen is every
non-air voxel.  Along the thickness axis, every column keeps only its LONGEST
contiguous run of specimen voxels, so a thin strip misclassified at the
volume edge (seen on the Teijin crop: row 0 labelled non-air) does not move
the surfaces.  The first voxel of that run is the UPPER limit (painted red)
and the last is the LOWER limit (painted green).  "Upper" means the
lower index, i.e. the top of the slice as ImageJ shows it when the thickness
axis is the image's vertical axis.

Output: an RGB TIFF stack (ImageJ hyperstack, same axis order as the input)
of the grey volume with the two lines, plus surfaces.json (mean, min, max of
each surface position and the specimen thickness).

    python scripts/analysis/paint_material_limits.py --grey <grey.tif> --label <label.tif> \
        --thickness-axis 1 --out <rgb.tif>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--grey", type=Path, required=True)
ap.add_argument("--label", type=Path, required=True)
ap.add_argument("--thickness-axis", type=int, required=True)
ap.add_argument("--width", type=int, default=1, help="line width in voxels, drawn inward")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

grey = tifffile.imread(a.grey)
lab = tifffile.imread(a.label)
assert grey.shape == lab.shape, (grey.shape, lab.shape)
ax = a.thickness_axis
s = np.moveaxis(lab != 2, ax, -1)                  # (..., T)
T = s.shape[-1]
cur_start = np.zeros(s.shape[:-1], np.int32); cur_len = np.zeros_like(cur_start)
best_start = np.zeros_like(cur_start); best_len = np.zeros_like(cur_start)
for t in range(T):                                  # longest run of specimen voxels per column
    on = s[..., t]
    cur_start = np.where(on & (cur_len == 0), t, cur_start)
    cur_len = np.where(on, cur_len + 1, 0)
    better = cur_len > best_len
    best_len = np.where(better, cur_len, best_len); best_start = np.where(better, cur_start, best_start)
has = best_len > 0
top = np.where(has, best_start, -1)
bot = np.where(has, best_start + best_len - 1, -1)

rgb = np.repeat(np.moveaxis(grey, ax, -1)[..., None], 3, -1).copy()   # (..., T, 3)
idx = np.nonzero(has)
for w in range(a.width):
    t = np.clip(top[idx] + w, 0, T - 1)
    b = np.clip(bot[idx] - w, 0, T - 1)
    rgb[(*idx, t)] = (255, 0, 0)
    rgb[(*idx, b)] = (0, 255, 0)
rgb = np.moveaxis(rgb, -2, ax)                      # restore the input axis order, channel last
tifffile.imwrite(a.out, rgb, imagej=True, photometric="rgb")
th = (bot - top + 1)[has]
info = {"grey": str(a.grey), "label": str(a.label), "thickness_axis": ax, "columns_with_material": int(has.sum()),
        "columns_total": int(has.size), "upper_mean": float(top[has].mean()), "upper_min": int(top[has].min()),
        "upper_max": int(top[has].max()), "lower_mean": float(bot[has].mean()), "lower_min": int(bot[has].min()),
        "lower_max": int(bot[has].max()), "thickness_voxels_mean": float(th.mean()),
        "thickness_voxels_min": int(th.min()), "thickness_voxels_max": int(th.max())}
(a.out.parent / "surfaces.json").write_text(json.dumps(info, indent=1))
print(info)

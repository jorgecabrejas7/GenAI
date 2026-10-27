#!/usr/bin/env python
"""Write the real crop behind each generated campaign-36 volume as TIFFs. (CPU.)

For every ``volumes/<case>/manifest.json`` under an assessment directory, reads
the real coupon and crop the case was built from (``geometry_from`` +
``geometry_crop_zyx`` for the bin volumes, ``real_volume_id`` +
``real_crop_zyx`` for the twins) and writes ``real_<coupon>_<crop>_volume.tif``
(grey, uint8) and ``..._label.tif`` (0 material, 1 pore, 2 air) into --out,
once per distinct crop. Same voxels and axis order as the generated volume.

    python scripts/analysis/export_real_crops.py \\
        --assessment runs/campaigns/36-pore-morphology/gen_production/pore_population \\
        --out runs/campaigns/36-pore-morphology/real_crops
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

from poregen.dataset.loader import build_label

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assessment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="split_v4")
    a = ap.parse_args()
    z = zarr.open(str(REPO / "data" / a.split / "volumes.zarr"), mode="r")
    a.out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for m in sorted(a.assessment.glob("volumes/*/manifest.json")):
        n = json.load(open(m)).get("notes", {})
        vid = n.get("geometry_from") or n.get("real_volume_id")
        crop = n.get("geometry_crop_zyx") or n.get("real_crop_zyx")
        if vid is None:
            continue
        short = vid.split("probetas_")[-1].replace("_volume_eq_aligned", "")
        tag = f"real_{short}_z{crop[0][0]}_y{crop[1][0]}_x{crop[2][0]}"
        seen.setdefault(tag, []).append(m.parent.name)
        if (a.out / f"{tag}_label.tif").exists():
            continue
        (z0, z1), (y0, y1), (x0, x1) = crop
        g = z[vid]
        sl = np.s_[z0:z1, y0:y1, x0:x1]
        tifffile.imwrite(a.out / f"{tag}_volume.tif", np.asarray(g["xct"][sl]), imagej=True)
        tifffile.imwrite(a.out / f"{tag}_label.tif",
                         build_label(np.asarray(g["mask"][sl]), np.asarray(g["sample_mask"][sl])),
                         imagej=True)
        print("wrote", tag, flush=True)
    (a.out / "cases_by_real_crop.json").write_text(json.dumps(seen, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

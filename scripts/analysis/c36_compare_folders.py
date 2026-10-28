#!/usr/bin/env python
"""One folder per real coupon with its twins beside it, for copying and viewing. (CPU.)

Campaign 36 (author 2026-10-28: every volume and its counterpart in the same
directory). For each twin found under ``gen_<sampler>/twins/volumes/`` writes
``compare/<coupon>/``:

    real_volume.tif, real_label.tif          the real coupon, full size
    twin_<sampler>_volume.tif / _label.tif   the twin, air padding cut off, so
                                             the SAME shape as the real files
    real_pores.parquet                       per-pore table of the real coupon
    twin_<sampler>_pores.parquet             per-pore table of the twin
    twin_<sampler>_manifest.json             the generation record

Labels: 0 material, 1 pore, 2 air. Existing files are kept; rerun after more
twins finish.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
import zarr

from poregen.dataset.loader import build_label

REPO = Path(__file__).resolve().parents[2]
C36 = REPO / "runs/campaigns/36-pore-morphology"


def main() -> int:
    z = zarr.open(str(REPO / "data/split_v4/volumes.zarr"), mode="r")
    real_tables = {p.stem: p for p in C36.glob("real_[ab]/*.parquet")}
    for man in sorted(C36.glob("gen_*/twins/volumes/*/manifest.json")):
        sampler = man.parts[-5].removeprefix("gen_")
        m = json.load(open(man))
        vid = m["notes"]["real_volume_id"]
        crop = m["notes"]["real_crop_zyx"]
        out = C36 / "compare" / man.parent.name.removeprefix("twin_")
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "real_label.tif").exists():
            g = z[vid]
            tifffile.imwrite(out / "real_volume.tif", np.asarray(g["xct"]), imagej=True)
            tifffile.imwrite(out / "real_label.tif",
                             build_label(np.asarray(g["mask"]), np.asarray(g["sample_mask"])), imagej=True)
            if vid in real_tables:
                shutil.copy2(real_tables[vid], out / "real_pores.parquet")
        cut = tuple(slice(0, hi - lo) for lo, hi in crop)
        for kind in ("volume", "label"):
            dst = out / f"twin_{sampler}_{kind}.tif"
            if not dst.exists():
                tifffile.imwrite(dst, tifffile.imread(man.parent / f"{kind}.tif")[cut], imagej=True)
        shutil.copy2(man, out / f"twin_{sampler}_manifest.json")
        tab = C36 / f"gen_{sampler}_twins" / f"{man.parent.name}.parquet"
        if tab.exists():
            shutil.copy2(tab, out / f"twin_{sampler}_pores.parquet")
        print("ok", out.name, sampler, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

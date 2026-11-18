#!/usr/bin/env python
"""Non-learned air baselines on a grey volume (campaign 40), for comparison only.

- ``reference_air.tif``: the complement (air) of
  ``preprocess_tools.onlypores.material_mask`` (UTvsXCT-preprocessing, global
  Otsu, max projection along axis 0, fill_voids), called unchanged on the
  volume as read.
- ``envelope_air.tif``: the hand texture envelope of campaign 39
  (``vae_segment_volume.air_envelope``: local std < 15, opening 5^3, face
  components, closing 3^3).

``--reference-src`` is ``preprocess_tools/onlypores.py`` at the pinned commit,
loaded as a lone module (the package ``__init__`` pulls in cv2):
    git -C ../UTvsXCT-preprocessing show 65c1eea:preprocess_tools/onlypores.py > /tmp/onlypores_65c1eea.py
    python scripts/analysis/teijin_air_baselines.py --tif <grey.tif> --thickness-axis 1 \
        --reference-src /tmp/onlypores_65c1eea.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_segment_volume import air_envelope  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--reference-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("onlypores_ref", a.reference_src)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    material_mask = ref.material_mask
    from skimage.filters import threshold_otsu  # noqa: PLC0415

    v = tifffile.imread(a.tif)
    ref_air = ~material_mask(v)
    env = np.moveaxis(air_envelope(np.moveaxis(v, a.thickness_axis, 0), 15.0), 0, a.thickness_axis)
    tifffile.imwrite(a.out / "reference_air.tif", (ref_air * 255).astype(np.uint8), imagej=True)
    tifffile.imwrite(a.out / "envelope_air.tif", (env * 255).astype(np.uint8), imagej=True)
    rep = {"otsu_threshold": float(threshold_otsu(v)), "reference_air_fraction": float(ref_air.mean()),
           "envelope_air_fraction": float(env.mean())}
    (a.out / "baselines.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PNG mid-slices of the SliceGAN baseline, for the notebook and the eye. (CPU.)

Uses `eval_v4_inspection_pack.render_case` — the SAME renderer the ldm06
inspection pack uses — so a baseline panel and an ldm06 panel are the same
figure with different content. A second renderer would differ in colour map,
aspect or slice choice, and a reader comparing the two would be reading the
difference between two plotting scripts.

The hourly training grids are raw `(4, 64, 64, 64)` arrays rather than eval_v4
cases, so they are written to a temporary case directory first and rendered
through the same path, for the same reason.

Usage:
    python scripts/analysis/slicegan_inspection.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

logger = logging.getLogger("slicegan_inspection")
C22 = REPO / "runs" / "campaigns" / "22-slicegan-baseline"


def grid_to_case(npy: Path, out: Path) -> Path:
    """A training sample grid, written as a case dir so render_case can read it."""
    import tifffile

    v = np.load(npy).astype(np.float32)
    grey = ((v[0].clip(-1.0, 1.0) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    label = v[1:].argmax(axis=0).astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out / "volume.tif"), grey)
    tifffile.imwrite(str(out / "label.tif"), label)
    # The renderer reads a few manifest fields for its title; a training grid
    # has no case identity, so it gets the honest minimum rather than a
    # fabricated one.
    (out / "manifest.json").write_text(json.dumps({
        "assessment": "slicegan_training",
        "case": npy.stem,
        "volume_shape": list(grey.shape),
        "sampler": "slicegan",
        "notes": {"source": str(npy),
                  "what": "a training-time sample grid, not an eval case"},
    }, indent=2) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=C22)
    ap.add_argument("--max-voxels", type=float, default=6e7,
                    help="skip cases larger than this; the renderer reads a column "
                         "through every page and a 1024-wide case is slow on CPU")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    import eval_v4_inspection_pack as pack

    out = args.root / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    rendered, skipped = [], []

    # 1. the generated eval cases
    vols = args.root / "slicegan" / "volumes"
    for d in sorted(vols.glob("*")):
        if not (d / "manifest.json").exists():
            continue
        shape = json.loads((d / "manifest.json").read_text()).get("volume_shape") or []
        n = int(np.prod(shape)) if shape else 0
        if n > args.max_voxels:
            skipped.append({"case": d.name, "why": f"{n/1e6:.0f} Mvox over the limit"})
            logger.info("skip %s (%.0f Mvox)", d.name, n / 1e6)
            continue
        info = pack.render_case(d, "slicegan", d.name,
                                "SliceGAN baseline — unconditional", out)
        if info:
            rendered.append(info["png"])

    # 2. the last training sample grid
    grids = sorted((args.root / "train").glob("sample_step*.npy"))
    if grids:
        with tempfile.TemporaryDirectory() as tmp:
            case = grid_to_case(grids[-1], Path(tmp) / grids[-1].stem)
            info = pack.render_case(case, "slicegan_training", grids[-1].stem,
                                    "training sample grid (not an eval case)", out)
            if info:
                rendered.append(info["png"])

    (out / "inspection_manifest.json").write_text(json.dumps({
        "campaign": str(args.root),
        "renderer": "eval_v4_inspection_pack.render_case (the same one ldm06 uses)",
        "n_rendered": len(rendered), "rendered": rendered, "skipped": skipped,
    }, indent=2) + "\n")
    logger.info("rendered %d panels -> %s", len(rendered), out)
    for row in skipped:
        logger.info("  skipped %s: %s", row["case"], row["why"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

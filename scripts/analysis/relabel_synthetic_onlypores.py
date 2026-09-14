"""Relabel synthetic volumes with the DATASET's pore segmentation. (CPU.)

WHY. Campaign 14's synthetic arm learns the ldm06 decoder's own 3-class head,
while the real arm learns the `onlypores` segmentation the dataset was built
with. Those are different labelling functions, so part of the synthetic arm's
gap is a label mismatch rather than image quality, and nothing in that campaign
separates the two. This writes a second label per case, produced by the
production pore segmentation, so an arm can be trained on the same images with
the same labelling function the real arm uses.

WHAT IS HELD FIXED. Exactly one thing changes: how PORE is decided. Air is
taken from the decoder's own class, unchanged, and material is everything that
is not air. Replacing the air class as well would move two variables at once
and the arm would no longer isolate the pore labelling function.

    air      = decoder label == 2                     (unchanged)
    material = not air
    pore     = production Sauvola (radius 30, k 0.125) inside material

`min_size_filtering` is -1, as `dataset.io.compute_mask` leaves it, so no
component filter is applied — the stored dataset labels had none either.

WHY NOT `onlypores()` ITSELF. That function derives the material mask with a
GLOBAL Otsu, which needs a whole scan and misfires on a crop — campaign 13's
docstring and `05-eval-v3-fixed-decode/onlypores/` both record it. Measured
here: called on a real crop of known 6 % porosity it returns 0.00000 pore and
calls half the crop non-material. Supplying the material mask instead of
letting Otsu guess it is what makes the production pore step usable on a
volume that is all specimen by construction.

Usage:
    python scripts/analysis/relabel_synthetic_onlypores.py --campaign runs/campaigns/18-eval-v4-final
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))

logger = logging.getLogger("relabel")

#: Production segmentation settings — `dataset.io.compute_mask` calls
#: `onlypores` with no arguments, so these are what every stored label used.
SAUVOLA_RADIUS = 30
SAUVOLA_K = 0.125

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2
#: Written beside label.tif; never overwrites it.
OUT_NAME = "label_onlypores.tif"


def relabel(xct: np.ndarray, decoder_label: np.ndarray) -> np.ndarray:
    from poregen.dataset.segmentation import sauvola_thresholding_nonconcurrent

    air = decoder_label == LABEL_AIR
    material = ~air
    binary = sauvola_thresholding_nonconcurrent(
        xct, window_size=2 * SAUVOLA_RADIUS + 1, k=SAUVOLA_K)
    pore = (~binary) & material
    out = np.full(decoder_label.shape, LABEL_MATERIAL, np.uint8)
    out[pore] = LABEL_PORE
    out[air] = LABEL_AIR
    return out


def pore_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    n = float(a.sum() + b.sum())
    return 2.0 * inter / n if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path,
                    default=REPO / "runs" / "campaigns" / "18-eval-v4-final")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="where to write the agreement report "
                         "(default: <campaign>/relabel_onlypores.json)")
    ap.add_argument("--force", action="store_true", help="rewrite labels that exist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logging.getLogger("poregen").setLevel(logging.WARNING)

    import tifffile
    from downstream_utility import required_synthetic_cases

    rows, t0 = [], time.time()
    cases = required_synthetic_cases()
    logger.info("%d synthetic cases to relabel", len(cases))
    for assessment, name in cases:
        d = args.campaign / assessment / "volumes" / name
        vol_f, lab_f = d / "volume.tif", d / "label.tif"
        if not (vol_f.exists() and lab_f.exists()):
            logger.warning("%s/%s: missing volume or label — skipped", assessment, name)
            continue
        out_f = d / OUT_NAME
        if out_f.exists() and not args.force:
            logger.info("%s/%s: already relabelled", assessment, name)
        xct = tifffile.imread(vol_f)
        dec = tifffile.imread(lab_f)
        t = time.time()
        new = relabel(xct, dec)
        dt = time.time() - t
        if not out_f.exists() or args.force:
            tifffile.imwrite(str(out_f), new)
        row = {
            "assessment": assessment, "case": name,
            "shape": list(xct.shape),
            "pore_decoder": float((dec == LABEL_PORE).mean()),
            "pore_onlypores": float((new == LABEL_PORE).mean()),
            "air_fraction": float((dec == LABEL_AIR).mean()),
            "pore_dice_relabel_vs_decoder": pore_dice(new == LABEL_PORE, dec == LABEL_PORE),
            "seconds": dt,
        }
        rows.append(row)
        logger.info("%s/%s  pore %.5f -> %.5f  Dice %.4f  (%.1f s)",
                    assessment, name, row["pore_decoder"], row["pore_onlypores"],
                    row["pore_dice_relabel_vs_decoder"], dt)
        del xct, dec, new

    dice = [r["pore_dice_relabel_vs_decoder"] for r in rows if np.isfinite(r["pore_dice_relabel_vs_decoder"])]
    pd_ = [r["pore_decoder"] for r in rows]
    po = [r["pore_onlypores"] for r in rows]
    report = {
        "campaign": str(args.campaign),
        "n_cases": len(rows),
        "settings": {"sauvola_radius": SAUVOLA_RADIUS, "sauvola_k": SAUVOLA_K,
                     "min_size_filtering": -1,
                     "air": "taken from the decoder's own class, unchanged"},
        "pore_dice_relabel_vs_decoder": {
            "mean": float(np.mean(dice)), "sd": float(np.std(dice)),
            "min": float(np.min(dice)), "max": float(np.max(dice)), "n": len(dice)},
        "pore_fraction_decoder_mean": float(np.mean(pd_)),
        "pore_fraction_onlypores_mean": float(np.mean(po)),
        "wall_seconds": time.time() - t0,
        "per_case": rows,
    }
    out_json = args.out_json or (args.campaign / "relabel_onlypores.json")
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("pore Dice relabel vs decoder: %.4f +/- %.4f over %d cases",
                report["pore_dice_relabel_vs_decoder"]["mean"],
                report["pore_dice_relabel_vs_decoder"]["sd"], len(dice))
    logger.info("pore fraction: decoder %.5f -> onlypores %.5f",
                report["pore_fraction_decoder_mean"], report["pore_fraction_onlypores_mean"])
    logger.info("wrote %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

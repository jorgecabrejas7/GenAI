"""Would the memorisation search find a copy if there were one? (GPU + store.)

THE NUMBER THE NULL RESULT DEPENDS ON. Campaign 18 reports that no generated
patch was memorised. That means nothing on its own: a search that could not
flag a copy would report exactly the same thing. This plants copies whose
answer is known and reports how many the 1/3 criterion actually catches.

FOUR PLANTS, each harder than the last:

  shift 0   a bank row used as its own query. Distance must be 0 and the ratio
            must be 0. If this is not caught the search is broken, and every
            other number in the assessment is void.
  shift 32  the store's own stride. The patch EXISTS in the store but not in
            the stride-64 BANK, so the nearest bank row is a genuinely
            different patch that overlaps it by half. This is the realistic
            worst case for a real copy and is the honest detection limit.
  shift 8   assembled from the stride-32 rows that cover it. No bank row can
  shift 16  match it exactly; the question is how close the ratio still gets.
  roundtrip a bank row pushed through the frozen VAE and back. A generated
            "copy" would arrive carrying decoder error, so this is what a real
            memorised sample would look like rather than a bit-exact one.

Usage:
    python scripts/analysis/memorisation_sensitivity.py --campaign <root>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("memorisation_sensitivity")

#: Planted queries per shift. The statistic is a detection RATE, so this buys
#: precision on that rate and nothing else; 256 gives it to about +/- 3 %.
N_PLANTED = 256


def assemble_shifted(store, pos: int, shift: tuple[int, int, int],
                     index, patch: int) -> np.ndarray | None:
    """One grey patch read at a VOXEL OFFSET from a bank row's origin.

    The store is built on stride 32 and a patch is 64 voxels, so the stored
    patches overlap by half and any 64-cube whose origin is a multiple of 8
    inside that lattice is covered by at most eight of them. This copies the
    overlapping parts out rather than re-reading the source volume, so the
    validation needs the patch file and not the original zarr.

    Returns None when the shifted cube runs off the volume's stored extent.
    """
    row = store.rows[pos]
    z0, y0, x0 = (int(index["z0"][row]), int(index["y0"][row]), int(index["x0"][row]))
    vid = index["volume_id"][row]
    tz, ty, tx = (z0 + shift[0], y0 + shift[1], x0 + shift[2])

    out = np.empty((patch, patch, patch), np.float32)
    filled = np.zeros((patch, patch, patch), bool)
    same_vol = np.flatnonzero(index["volume_id"] == vid)
    zs, ys, xs = (index["z0"][same_vol], index["y0"][same_vol], index["x0"][same_vol])
    # Every stored patch that overlaps the shifted cube.
    hit = np.flatnonzero(
        (zs < tz + patch) & (zs + patch > tz)
        & (ys < ty + patch) & (ys + patch > ty)
        & (xs < tx + patch) & (xs + patch > tx))
    if not hit.size:
        return None
    memmap = store.patch_memmap()
    for h in hit:
        r = same_vol[h]
        sz, sy, sx = int(zs[h]), int(ys[h]), int(xs[h])
        block = np.asarray(memmap[int(index["source_row"][r])], np.float32) / 255.0
        block = block.reshape(patch, patch, patch)
        lo = [max(tz, sz), max(ty, sy), max(tx, sx)]
        hi = [min(tz + patch, sz + patch), min(ty + patch, sy + patch),
              min(tx + patch, sx + patch)]
        dst = tuple(slice(lo[a] - [tz, ty, tx][a], hi[a] - [tz, ty, tx][a]) for a in range(3))
        src = tuple(slice(lo[a] - [sz, sy, sx][a], hi[a] - [sz, sy, sx][a]) for a in range(3))
        out[dst] = block[src]
        filled[dst] = True
    if not filled.all():
        return None
    return out.reshape(-1)


def main() -> int:
    import pandas as pd
    import torch

    from poregen.eval_v4 import memorisation as MEMO

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path,
                    default=REPO / "runs" / "campaigns" / "18-eval-v4-final")
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "27-memorisation-sensitivity")
    ap.add_argument("--n", type=int, default=N_PLANTED)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    busy = [] if args.allow_busy_gpu else MEMO.gpu_jobs_other_than(__import__("os").getpid())
    if busy:
        logger.warning("the card is busy with %s; this streams the whole store "
                       "and must not run beside it", busy)
        return 2

    store = MEMO.PatchStore.open(split="train")
    index = pd.read_parquet(store.root / store.split / "index.parquet")
    index = {c: index[c].to_numpy() for c in ("z0", "y0", "x0", "volume_id", "source_row")}
    rng = np.random.default_rng(0)
    take = rng.choice(len(store), size=min(args.n, len(store)), replace=False)
    logger.info("store: %d bank rows; planting %d", len(store), len(take))

    results = {}
    for shift in MEMO.PLANTED_SHIFTS:
        if shift == 0:
            q = store.grey_at(take)
            kept = take
        else:
            built, kept_list = [], []
            for pos in take:
                v = assemble_shifted(store, int(pos), (shift, shift, shift),
                                     index, MEMO.PATCH)
                if v is not None:
                    built.append(v); kept_list.append(pos)
            if not built:
                results[f"shift_{shift}"] = {"n": 0, "reason": "no shifted cube fitted"}
                continue
            q = np.stack(built); kept = np.asarray(kept_list)
        acc = MEMO.search(q, store, "grey")
        results[f"shift_{shift}"] = {
            **MEMO.detection_rate(acc),
            "shift_voxels": shift,
            "in_the_bank": shift == 0,
            "in_the_store_but_not_the_bank": shift == 32,
        }
        logger.info("shift %2d: detection %.3f  nn median %.4f",
                    shift, results[f"shift_{shift}"]["detection_rate"] or -1,
                    results[f"shift_{shift}"]["nn_distance_median"] or -1)

    out = {
        "question": "Would the memorisation search find a copy if there were one?",
        "n_planted": int(len(take)),
        "bank_rows": int(len(store)),
        "bank_coverage_note": MEMO.BANK_COVERAGE_NOTE,
        "ratio_threshold": MEMO.RATIO_THRESHOLD,
        "note": (
            "shift 0 is the sanity floor: a bank row queried against its own "
            "bank must score distance 0 and ratio 0, and a detection rate "
            "below 1.0 there invalidates the whole assessment. shift 32 is the "
            "honest detection limit — the patch is in the STORE but not in the "
            "stride-64 BANK, which is what a real 32-voxel-offset copy would "
            "look like. Absolute distances are reported beside every ratio, "
            "because a ratio cannot distinguish 'identical' from 'closest of "
            "many far-away rows'."
        ),
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    logger.info("-> %s/results.json", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

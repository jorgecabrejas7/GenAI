"""OOB / UNKNOWN / EXISTS fractions of the faces a latent store will serve.

Vault D57 / OQ-4. ``LatentDataset`` marks a face OOB only where the specimen
ends within one neighbour offset (``cond_dist6 < 1`` on that side) and UNKNOWN
where a stored neighbour is missing although the specimen continues. This
counts the three states over a split WITHOUT the store: the store's rows are
the ``patch_index.parquet`` rows of the split with ``porosity <= 1``
(``build_latent_dataset._source_rows``), and ``cond_dist6`` comes from the same
specimen box ``build_conditioning.build_scalars`` uses (``extent_foreground``
of the orientation field, box ``[lo, hi + 1)``).

    python scripts/analysis/face_availability_fractions.py \\
        --data-root data/split_v4 --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    dist6_from_box_array,
)
from poregen.diffusion.latents import face_availability

PATCH = 64
OFFSET = 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/split_v4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--orientation-field", default=None,
                    help="default <data-root>/orientation_field.json")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    root = Path(args.data_root)
    df = pd.read_parquet(root / "patch_index.parquet",
                         columns=["volume_id", "split", "z0", "y0", "x0", "porosity"])
    df = df[(df["split"] == args.split) & (df["porosity"] <= 1.0)].reset_index(drop=True)
    field = json.loads(Path(args.orientation_field or root / "orientation_field.json")
                       .read_text())["volumes"]
    missing = sorted(set(df["volume_id"]) - set(field))
    if missing:
        raise SystemExit(f"orientation field has no extent for {missing[:3]} "
                         f"({len(missing)} volumes)")

    ext = df["volume_id"].map(lambda v: field[v]["extent_foreground"])
    box_lo = np.stack([[e[a][0] for e in ext] for a in "zyx"], axis=1).astype(float)
    box_hi = np.stack([[e[a][1] + 1 for e in ext] for a in "zyx"], axis=1).astype(float)
    origins = df[["z0", "y0", "x0"]].to_numpy(np.int64)
    dist6 = dist6_from_box_array(origins.astype(float), PATCH, box_lo, box_hi)

    vol = pd.factorize(df["volume_id"], sort=True)[0].astype(np.int64)
    B = 8192
    def key(v, o):
        return ((v * B + o[:, 0]) * B + o[:, 1]) * B + o[:, 2]
    keys = key(vol, origins)
    order = np.argsort(keys)
    ks = keys[order]
    nb_rows = np.full((len(df), 6), -1, np.int64)
    for i, d in enumerate(NEIGHBOUR_DIRS):
        o = origins + np.asarray(d, np.int64) * OFFSET
        k = key(vol, o)
        pos = np.clip(np.searchsorted(ks, k), 0, len(ks) - 1)
        hit = (ks[pos] == k) & (o >= 0).all(axis=1)
        nb_rows[:, i] = np.where(hit, order[pos], -1)

    avail = face_availability(nb_rows, dist6)
    n_faces = avail.size
    missing_mask = avail != NB_EXISTS
    out = {
        "data_root": str(root), "split": args.split, "n_patches": int(len(df)),
        "n_faces": int(n_faces),
        "frac_exists": float(np.mean(avail == NB_EXISTS)),
        "frac_oob": float(np.mean(avail == NB_OOB)),
        "frac_unknown": float(np.mean(avail == NB_UNKNOWN)),
        "unknown_share_of_missing_faces": float(
            np.mean(avail[missing_mask] == NB_UNKNOWN)) if missing_mask.any() else 0.0,
        "frac_patches_with_any_unknown": float(np.mean((avail == NB_UNKNOWN).any(axis=1))),
        "per_face": {f"{d}": {"oob": float(np.mean(avail[:, i] == NB_OOB)),
                              "unknown": float(np.mean(avail[:, i] == NB_UNKNOWN))}
                     for i, d in enumerate(NEIGHBOUR_DIRS)},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

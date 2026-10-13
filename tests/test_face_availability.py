"""A missing neighbour is OOB only where the specimen ends (vault D57 / OQ-4).

The store has no row at a face for two reasons: the specimen ends there, or a
patch inside the specimen was dropped (a drilled hole, an index gap). Only the
first is OOB; the second is served as UNKNOWN, the encoding the per-face
dropout and the sampler use for "a neighbour exists but is not given".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _ldm06_store import DIST6_COLUMNS, SYN, build_store, dataset_kwargs
from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    dist6_from_box_array,
)
from poregen.diffusion.latents import LatentDataset, face_availability

PLUS_Z = NEIGHBOUR_DIRS.index((1, 0, 0))
PLUS_X = NEIGHBOUR_DIRS.index((0, 0, 1))


def _geometric_store(tmp_path):
    """The miniature store with REAL cond_dist6: specimen box = the volume,
    cap = one neighbour offset (the production cap is 64 = the offset)."""
    root, *_ = build_store(tmp_path)
    split = root / "train"
    idx = pd.read_parquet(split / "index.parquet")
    cond = pd.read_parquet(split / "cond.parquet")
    origins = idx[["z0", "y0", "x0"]].to_numpy(float)
    n = len(idx)
    d6 = dist6_from_box_array(origins, SYN["PATCH"], np.zeros((n, 3)),
                              np.full((n, 3), SYN["VOL"]), dist_cap=SYN["OFFSET"])
    for k, name in enumerate(DIST6_COLUMNS):
        cond[name] = d6[:, k]
    cond.to_parquet(split / "cond.parquet", index=False)
    return root, idx


def _row(idx, z, y, x):
    hit = idx.index[(idx.z0 == z) & (idx.y0 == y) & (idx.x0 == x)]
    assert len(hit) == 1
    return int(hit[0])


def test_a_missing_interior_neighbour_is_unknown_and_a_true_boundary_is_oob(tmp_path):
    root, idx = _geometric_store(tmp_path)
    # Drop the +x neighbour of an interior patch: move its row far away, on
    # the stride grid, so the position it held has no stored patch.
    moved = _row(idx, 32, 32, 48)
    idx.loc[moved, "x0"] = 8 * 500
    idx.to_parquet(root / "train" / "index.parquet", index=False)

    ds = LatentDataset(root, "train", **dataset_kwargs())
    interior = ds[_row(idx, 32, 32, 32)]["nb_avail"].numpy()
    assert interior[PLUS_X] == NB_UNKNOWN
    assert all(interior[i] == NB_EXISTS for i in range(6) if i != PLUS_X)

    # A patch whose +z face is the end of the specimen: no row there, OOB.
    edge = ds[_row(idx, 64, 32, 32)]
    assert edge["nb_avail"][PLUS_Z] == NB_OOB
    assert float(edge["nb_latents"][PLUS_Z].abs().max()) == 0.0
    assert float(ds[_row(idx, 32, 32, 32)]["nb_latents"][PLUS_X].abs().max()) == 0.0


def test_the_rule_reads_the_matching_face_of_cond_dist6():
    rows = np.array([-1, -1, 5, -1, -1, 7])
    # dist6 order (z-, z+, y-, y+, x-, x+): only z+ ends within a patch.
    d6 = np.array([1.0, 0.25, 1.0, 1.0, 1.0, 1.0])
    got = face_availability(rows, d6)
    # NEIGHBOUR_DIRS order (+z, -z, +y, -y, +x, -x).
    assert got.tolist() == [NB_OOB, NB_UNKNOWN, NB_EXISTS, NB_UNKNOWN,
                            NB_UNKNOWN, NB_EXISTS]

#!/usr/bin/env python
"""Campaign 41 (paper review A7 / Q8): calibrate the memorisation test with
PLANTED copies of training patches. (GPU, card must be otherwise idle.)

Reviewer (2026-11-24): "How was the 1/3 copy-detection threshold chosen, and
what fraction of planted, slightly shifted (non-stride-aligned) training
patches would the test catch?"

Planted queries. ``--n`` training patches are drawn from the stride-64 bank
rows of the train split (material fraction >= 0.5). Each is re-cut from the
training volume (data/split_v4/volumes.zarr, the source the bank rows were cut
from; label = 2 where sample_mask is 0, else 1 where mask, else 0) at its
origin shifted by s voxels on all three axes, s in ``--shifts``. s = 0 is the
bank row itself; s = 32 is half a bank stride, the hardest case for a
stride-64 bank. Then they are searched exactly as generated queries are
(``poregen.eval_v4.memorisation``: full stride-64 train bank, nearest and
second-nearest, Favero ratio, verdict ratio < 1/3):

    latent       VAE posterior mean of the planted patch / sigma_hat
    grey_raw     the planted grey as is (no decoder error)
    grey_rt      the planted latent decoded by the same VAE (the decoder error
                 a generated query carries)

Reported per shift and space: detection rate at 1/3, the ratio quantiles, the
share whose nearest bank row IS the source row, and the detection rate at
other thresholds (0.2 ... 0.8), so the 1/3 choice can be read against it.

    python scripts/analysis/c41_copy_calibration.py --out runs/campaigns/41-paper-review-checks/copy_calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from poregen.eval_v4 import memorisation as M

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data/split_v4/latents_r08-run-0020-s25000z4"
P = M.PATCH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--shifts", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    busy = M.gpu_jobs_other_than(__import__("os").getpid())
    if busy:
        raise SystemExit(f"GPU busy: {busy}")

    train = M.PatchStore.open(STORE, "train")
    idx = pd.read_parquet(STORE / "train/index.parquet")
    bank = idx.iloc[train.rows].reset_index(drop=True)          # aligned with bank positions
    air = np.fromfile(STORE / "train/air.bin", dtype=np.float32)[train.rows]
    ok = np.flatnonzero(air <= 0.5)
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(ok, a.n, replace=False)

    meta = json.loads((STORE / "metadata.json").read_text())
    from poregen.experiments.train_vae import load_vae_from_checkpoint
    vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), dev)
    vae.requires_grad_(False); vae.eval()

    vol = zarr.open(str(REPO / "data/split_v4/volumes.zarr"), mode="r")
    # Cut every planted patch for every shift first, then search each space
    # ONCE over all of them: a pass over the grey bank reads ~50 GB, so one
    # pass per space instead of one per shift and space.
    xs, ls, meta_q = [], [], []
    for s in a.shifts:
        for b in pick:
            r = bank.iloc[b]; g = vol[r.volume_id]
            z, y, x = int(r.z0) + s, int(r.y0) + s, int(r.x0) + s
            shp = g["xct"].shape
            if z + P > shp[0] or y + P > shp[1] or x + P > shp[2]:
                continue
            sl = np.s_[z:z + P, y:y + P, x:x + P]
            xs.append(np.asarray(g["xct"][sl]))
            m = np.asarray(g["mask"][sl]); sm = np.asarray(g["sample_mask"][sl])
            ls.append(np.where(sm == 0, 2, np.where(m > 0, 1, 0)).astype(np.uint8))
            meta_q.append((s, int(b)))
    xs, ls = np.stack(xs), np.stack(ls)
    mu = np.concatenate([M.encode_patches(vae, xs[i], ls[i], [(0, 0, 0)], device=dev) for i in range(len(xs))])
    # encode_patches returns (n, C*d*h*w); decode_grey wants the raw means as (n, C, d, h, w)
    mu5 = mu.reshape(len(mu), train.latent_channels, *train.latent_spatial)
    queries = {
        "latent": train.normalise_latents(mu),
        "grey_raw": xs.reshape(len(xs), -1).astype(np.float32) / 255.0,
        "grey_rt": M.decode_grey(vae, mu5, device=dev),
    }
    rows = []
    for space, q in queries.items():
        acc = M.search(q, train, "latent" if space == "latent" else "grey", device=dev)
        ratio = acc.ratio()
        for i, (s, b) in enumerate(meta_q):
            rows.append({"shift": s, "space": space, "bank_pos": b, "ratio": float(ratio[i]),
                         "d1": float(acc.d1[i]), "d2": float(acc.d2[i]),
                         "nearest_is_source": bool(acc.i1[i] == b)})
        print(f"{space}: detected {np.mean(ratio < M.RATIO_THRESHOLD):.3f} overall", flush=True)
        pd.DataFrame(rows).to_csv(a.out / "planted_per_query.csv", index=False)
    d = pd.DataFrame(rows)
    th = [0.2, 1 / 3, 0.5, 0.6, 0.7, 0.8]
    summ = d.groupby(["space", "shift"]).agg(
        n=("ratio", "size"), median_ratio=("ratio", "median"),
        p90_ratio=("ratio", lambda v: np.quantile(v, 0.9)),
        nearest_is_source=("nearest_is_source", "mean"),
        **{f"detected_at_{t:.2f}": ("ratio", (lambda t: lambda v: float(np.mean(v < t)))(t)) for t in th}).round(3)
    (a.out / "planted_summary.txt").write_text(
        "Campaign 41 - memorisation test on planted training patches shifted by s voxels on every axis\n"
        f"bank: stride-64 train rows of {STORE.name}; verdict ratio < 1/3\n\n" + summ.to_string() + "\n")
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

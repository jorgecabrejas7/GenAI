"""Is the chunk-plane band a BLUR or a CONTENT effect? (CPU, read-only)

The band is a fivefold porosity depletion in the last ~16 voxels of every
chunk.  Two mechanisms are still open and they predict different things about
the latent canvas that produced it:

**Blur.**  The fusion averages two predictions that disagree, and an average of
two sparse latents is not a sparse latent.  Pores are rare, so the mean of two
plausible pore fields decodes to none.  This shows as a DROP in the latent
standard deviation across the band — the canvas is flatter there — and, in the
decoded class probability, as probability spread over more voxels at a lower
peak: a smeared pore that never crosses the argmax threshold.

**Content.**  The conditional prediction itself asks for less porosity next to
a neighbour.  The latent std is then NORMAL in the band and the pore
probability is simply LOW there — no smearing, nothing near the threshold.

The two are told apart by the same volumes read two ways, so this reads the
saved `latents.npy` and `probs.npz` of volumes that already exist rather than
generating anything.

Everything is reported as a ratio to the volume's own interior, because the
arms differ in overall porosity and an absolute std is not comparable between
them.  The store's own sampled latents are included as the reference scale: the
canvas the model is imitating.

Usage:
    python scripts/analysis/chunk_band_latents.py --root runs/campaigns/12-eval-v4 \
        --case assembly_modes/hybrid_1024_seed101 --planes 192
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Latent cells per 64-voxel tile, i.e. the VAE downsampling factor.
DS = 4
#: Slab thickness in CELLS. 2 cells = 8 voxels, the slab the phi profile uses.
SLAB_CELLS = 2
OFFSETS_VOX = (-32, -24, -16, -8, 0, 8, 16, 24)


def slab_stats(z: np.ndarray, axis: int, lo: int, hi: int) -> dict:
    """Per-channel std and mean |value| of a latent slab, averaged over channels.

    `z` is (C, Z, Y, X). The std is taken per channel and then averaged, not
    over the pooled values: channels have different scales, and a pooled std
    would mostly measure that.
    """
    sl = [slice(None)] * 4
    sl[axis + 1] = slice(lo, hi)
    blk = z[tuple(sl)]
    if blk.size == 0:
        return {"std": None, "abs": None}
    flat = blk.reshape(blk.shape[0], -1)
    return {"std": float(np.mean(flat.std(axis=1))),
            "abs": float(np.mean(np.abs(flat).mean(axis=1)))}


def interior_stats(z: np.ndarray, axis: int, planes_cells: list[int],
                   guard_cells: int) -> dict:
    """The same statistics away from every plane, on the same volume."""
    n = z.shape[axis + 1]
    keep = np.ones(n, bool)
    for p in planes_cells:
        keep[max(0, p - guard_cells):min(n, p + guard_cells)] = False
    idx = np.flatnonzero(keep)
    sl = [slice(None)] * 4
    sl[axis + 1] = idx
    blk = z[tuple(sl)]
    flat = blk.reshape(blk.shape[0], -1)
    return {"std": float(np.mean(flat.std(axis=1))),
            "abs": float(np.mean(np.abs(flat).mean(axis=1))),
            "n_cells": int(idx.size)}


def pore_prob_profile(logit: np.ndarray, axis: int, planes_vox: list[int],
                      slab_vox: int = 8) -> dict:
    """Mean pore PROBABILITY (not argmax) by offset, and how often it is near
    the decision threshold.

    A blur puts probability mass on more voxels at a lower peak, so the mean
    can hold up while nothing crosses 0.5. `frac_mid` counts voxels in
    [0.2, 0.5) — smeared pore that never becomes a pore — and separates that
    from probability simply being absent.
    """
    p = 1.0 / (1.0 + np.exp(-logit.astype(np.float32)))
    n = p.shape[axis]
    out = {}
    for off in OFFSETS_VOX:
        vals = []
        mids = []
        for plane in planes_vox:
            lo, hi = plane + off, plane + off + slab_vox
            if lo < 0 or hi > n:
                continue
            sl = [slice(None)] * 3
            sl[axis] = slice(lo, hi)
            blk = p[tuple(sl)]
            vals.append(float(blk.mean()))
            mids.append(float(((blk >= 0.2) & (blk < 0.5)).mean()))
        if vals:
            out[str(off)] = {"mean_p": float(np.mean(vals)),
                             "frac_mid": float(np.mean(mids))}
    return out


def store_reference(store_root: Path, n_rows: int = 512, seed: int = 0) -> dict:
    """The scale the model is imitating: sampled latents from the train store.

    `mu + sigma*eps`, per-channel normalised exactly as the sampler's canvas
    is, so its std is comparable with a generated canvas cell for cell.
    """
    import pandas as pd

    meta = json.loads((store_root / "metadata.json").read_text())
    c, *spatial = (int(v) for v in meta["latent_shape"])
    norm = meta["normalization"]
    mean = np.asarray(norm["per_channel_mean"], np.float32).reshape(c, 1, 1, 1)
    std = np.asarray(norm["per_channel_std"], np.float32).reshape(c, 1, 1, 1)
    idx = pd.read_parquet(str(store_root / "train" / "index.parquet"),
                          columns=["source_row"])
    total = len(idx)
    dtype = np.dtype(meta["storage"]["dtype"])
    lat = np.memmap(str(store_root / "train" / "latents.bin"), dtype=dtype,
                    mode="r", shape=(total, 2 * c, *spatial))
    rng = np.random.default_rng(seed)
    rows = rng.choice(total, min(n_rows, total), replace=False)
    stds, abss = [], []
    for r in rows:
        packed = np.asarray(lat[int(r)], np.float32)
        mu, sd = packed[:c], packed[c:]
        z = (mu + sd * rng.standard_normal(mu.shape).astype(np.float32) - mean) / std
        flat = z.reshape(c, -1)
        stds.append(flat.std(axis=1).mean())
        abss.append(np.abs(flat).mean(axis=1).mean())
    return {"std": float(np.mean(stds)), "abs": float(np.mean(abss)),
            "n_rows": len(rows),
            "note": "train store, mu + sigma*eps, per-channel normalised"}


def analyse(root: Path, rel: str, planes_vox: list[int], axes: list[int]) -> dict:
    assessment, case = rel.split("/", 1)
    d = root / assessment / "volumes" / case
    manifest = json.loads((d / "manifest.json").read_text())
    z = np.load(d / "latents.npy").astype(np.float32)
    notes = manifest.get("notes") or {}

    out = {
        "case": rel,
        "arm": notes.get("arm") or case.split("_")[0],
        "s_nb": manifest.get("s_nb"),
        "chunk_tiles": manifest.get("chunk_tiles"),
        "latent_shape": list(z.shape),
        "axes": {},
    }
    logit = None
    p = d / "probs.npz"
    if p.exists():
        logit = np.load(p)["pore_logit"]

    for axis in axes:
        n_cells = z.shape[axis + 1]
        planes_c = [v // DS for v in planes_vox if 0 < v // DS < n_cells]
        if not planes_c:
            out["axes"][str(axis)] = {"skipped": "no plane on this axis"}
            continue
        inner = interior_stats(z, axis, planes_c, guard_cells=8)
        by_off = {}
        for off in OFFSETS_VOX:
            s_acc, a_acc = [], []
            for pc in planes_c:
                lo = pc + off // DS
                hi = lo + SLAB_CELLS
                if lo < 0 or hi > n_cells:
                    continue
                st = slab_stats(z, axis, lo, hi)
                if st["std"] is not None:
                    s_acc.append(st["std"])
                    a_acc.append(st["abs"])
            if s_acc:
                by_off[str(off)] = {
                    "std": float(np.mean(s_acc)),
                    "abs": float(np.mean(a_acc)),
                    "std_ratio": float(np.mean(s_acc)) / inner["std"],
                    "abs_ratio": float(np.mean(a_acc)) / inner["abs"],
                }
        block = {"planes_vox": [pc * DS for pc in planes_c],
                 "interior": inner, "by_offset": by_off}
        if logit is not None:
            block["pore_prob"] = pore_prob_profile(
                logit, axis, [pc * DS for pc in planes_c])
        out["axes"][str(axis)] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--planes", type=int, nargs="+", default=[192, 384, 576, 768, 960])
    ap.add_argument("--axes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--store", type=Path,
                    default=REPO / "data" / "split_v3" / "latents_r08z8")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ref = store_reference(args.store) if args.store.exists() else None
    results = [analyse(args.root, c, args.planes, args.axes) for c in args.case]
    out = args.out or (args.root / "chunk_band_latents.json")
    out.write_text(json.dumps({"store_reference": ref, "per_case": results},
                              indent=2) + "\n")

    offs = [str(o) for o in OFFSETS_VOX]
    print("\nLATENT std as a ratio to the volume's own interior "
          f"(slabs of {SLAB_CELLS} cells = {SLAB_CELLS * DS} voxels)")
    if ref:
        print(f"  store sampled-latent reference: std {ref['std']:.4f}  "
              f"mean|z| {ref['abs']:.4f}  ({ref['n_rows']} rows)")
    print(f"{'case':<34}{'s_nb':>5}{'interior std':>13}  "
          + "".join(f"{o:>8}" for o in offs))
    for r in results:
        for axis, blk in r["axes"].items():
            if "skipped" in blk:
                continue
            cells = "".join(
                (f"{blk['by_offset'][o]['std_ratio']:>8.3f}"
                 if o in blk["by_offset"] else f"{'-':>8}") for o in offs)
            print(f"{r['case'][-30:] + ' a' + axis:<34}{str(r['s_nb']):>5}"
                  f"{blk['interior']['std']:>13.4f}  {cells}")

    print("\nPORE PROBABILITY mean (top) and fraction in [0.2, 0.5) (bottom) by offset")
    for r in results:
        for axis, blk in r["axes"].items():
            if "skipped" in blk or "pore_prob" not in blk:
                continue
            pp = blk["pore_prob"]
            m = "".join((f"{pp[o]['mean_p']:>8.4f}" if o in pp else f"{'-':>8}")
                        for o in offs)
            f = "".join((f"{pp[o]['frac_mid']:>8.4f}" if o in pp else f"{'-':>8}")
                        for o in offs)
            print(f"{r['case'][-30:] + ' a' + axis:<34}{'p':>18}  {m}")
            print(f"{'':<34}{'mid':>18}  {f}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

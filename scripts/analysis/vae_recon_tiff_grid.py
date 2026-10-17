#!/usr/bin/env python
"""Four VAE checkpoints on the same val patches, as one uint8 TIFF volume. (CPU.)

For viewing in Fiji or any stack viewer: each z-slice of the output is a 2x2
grid of the four reconstructions of the same patch slice, in the order given
by ``--run`` (top-left, top-right, bottom-left, bottom-right). Patches follow
each other along z, separated by ``SEP`` white slices; cells are separated by
``SEP`` white voxels. Grey only (the XCT head, ``decode_xct_u8``), on the raw
scan scale. A second TIFF holds the input patches with the same z layout, so
slice k of both files is the same place.

Patches (split_v4 val, one per target, a different volume for each where the
200 nearest rows allow it):
  air-free rows at porosity quantiles 0.02, 0.02, 0.5, 0.9, 0.99, 0.999,
  rows nearest air fraction 0.2 and 0.5.

A CSV gives each patch's z range, dataset row, volume, porosity, air fraction
and per-model L1 and texture correlation (as in vae_recon_cross_sections.py).

Usage:
    python scripts/analysis/vae_recon_tiff_grid.py --out <dir>/name \\
        --run "rf-16 off=runs/vae/<run>:<ckpt>" --run ... (exactly four)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from vae_val_l1 import load_model_and_patches  # same dir
from vae_recon_cross_sections import corr  # same dir

REPO = Path(__file__).resolve().parents[2]
SEP = 2
TARGETS = [("porosity", 0.02), ("porosity", 0.02), ("porosity", 0.5), ("porosity", 0.9),
           ("porosity", 0.99), ("porosity", 0.999), ("air", 0.2), ("air", 0.5)]


def pick_rows(df, pool: int = 200) -> list[int]:
    """One row per target, nearest first, skipping volumes already used."""
    solid = df[df["air_fraction"] == 0.0]
    used_rows, used_vols = [], set()
    for kind, q in TARGETS:
        if kind == "porosity":
            dist = (solid["porosity"] - float(solid["porosity"].quantile(q))).abs()
        else:
            dist = (df["air_fraction"] - q).abs()
        cand = [r for r in dist.nsmallest(pool).index if r not in used_rows]
        fresh = [r for r in cand if df.at[r, "volume_id"] not in used_vols]
        row = int((fresh or cand)[0])
        used_rows.append(row); used_vols.add(df.at[row, "volume_id"])
    return used_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=path[:ckpt]", four times')
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.run) == 4, "the 2x2 grid needs exactly four --run"
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    rows, labels, recons, x_ref, df = None, [], [], None, None
    for spec in a.run:
        label, rest = spec.split("=", 1)
        parts = rest.split(":", 1)
        run, ckpt = REPO / parts[0], (parts[1] if len(parts) > 1 else "best.ckpt")
        model, ds, _, _, _ = load_model_and_patches(run, a.split, 1, 1, None, ckpt)
        if rows is None:
            df = ds.df
            rows = pick_rows(df)
        items = [ds[r] for r in rows]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        torch.manual_seed(0)
        batch_dev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            rec = decode_xct(model(*args).xct_out)[:, 0]
        x = batch_dev["xct"][:, 0]
        if x_ref is None:
            x_ref = x
        assert torch.equal(x, x_ref), "runs did not receive the same patches"
        labels.append(label); recons.append(rec)
        print(f"{label}: done")

    n, p = len(rows), x_ref.shape[-1]
    u8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    side = 2 * p + SEP
    grid = np.full((n * (p + SEP) - SEP, side, side), 255, np.uint8)
    inp = np.full((n * (p + SEP) - SEP, p, p), 255, np.uint8)
    offs = [(0, 0), (0, p + SEP), (p + SEP, 0), (p + SEP, p + SEP)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patch", "z_first", "z_last", "row", "volume_id", "z0", "y0", "x0", "porosity", "air_fraction"]
                   + [f"{l} {m}" for l in labels for m in ("L1", "corr")])
        for i, r in enumerate(rows):
            z = i * (p + SEP)
            inp[z:z + p] = u8(x_ref[i])
            for (oy, ox), rec in zip(offs, recons):
                grid[z:z + p, oy:oy + p, ox:ox + p] = u8(rec[i])
            meta = df.loc[r]
            vol = str(meta["volume_id"]).split("probetas_")[-1].replace("_volume_eq_aligned", "")
            w.writerow([i + 1, z, z + p - 1, r, vol, meta["z0"], meta["y0"], meta["x0"],
                        f"{meta['porosity']:.4f}", f"{meta['air_fraction']:.3f}"]
                       + [f"{v:.4f}" for rec in recons
                          for v in (float(F.l1_loss(rec[i], x_ref[i])), corr(rec[i], x_ref[i]))])
    meta = {"Info": "2x2 grid: " + " | ".join(labels)}
    tifffile.imwrite(out.with_name(out.name + "_grid.tif"), grid, imagej=True, metadata=meta)
    tifffile.imwrite(out.with_name(out.name + "_input.tif"), inp, imagej=True)
    print("wrote", out.with_name(out.name + "_grid.tif"), grid.shape, "and _input.tif", inp.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

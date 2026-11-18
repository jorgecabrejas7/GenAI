#!/usr/bin/env python
"""Score a trained U-Net segmenter on whole split_v4 splits (campaign 39).

Every patch of the split (or ``--max-patches`` drawn with a fixed seed), the
same metric as training validation (``poregen.metrics.seg.multiclass_metrics``
on the argmax: per-class Dice, porosity and air MAE/bias), aggregated three
ways: over all patches, per porosity bin of the patch (0-0.5, 0.5-1, 1-2, 2-5,
5-12, >12 %), and per source volume (coupon).  Output: <out>/<split>.json and
<split>_per_volume.csv.

    python scripts/analysis/unet_seg_eval.py --ckpt runs/unet_seg/<run>/best.ckpt --splits test val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from poregen.experiments.train_unet_seg import GreyLabelPatches
from poregen.models.unet3d import load_unet3d

REPO = Path(__file__).resolve().parents[2]
BINS = [0, 0.005, 0.01, 0.02, 0.05, 0.12, 1.01]
BIN_NAMES = ["0-0.5%", "0.5-1%", "1-2%", "2-5%", "5-12%", ">12%"]


@torch.no_grad()
def per_patch(model, loader, device) -> dict[str, np.ndarray]:
    cols: dict[str, list] = {k: [] for k in ("dice_material", "dice_pore", "dice_air",
                                             "pore_pred", "pore_true", "air_pred", "air_true")}
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        for c, name in enumerate(("material", "pore", "air")):
            p, t = (pred == c).flatten(1), (y == c).flatten(1)
            inter = (p & t).sum(1).float(); card = (p.sum(1) + t.sum(1)).float()
            cols[f"dice_{name}"].append(torch.where(card > 0, 2 * inter / card, torch.ones_like(card)).cpu())
        cols["pore_pred"].append((pred == 1).flatten(1).float().mean(1).cpu())
        cols["pore_true"].append((y == 1).flatten(1).float().mean(1).cpu())
        cols["air_pred"].append((pred == 2).flatten(1).float().mean(1).cpu())
        cols["air_true"].append((y == 2).flatten(1).float().mean(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in cols.items()}


def summarise(d: pd.DataFrame) -> dict:
    e = d.pore_pred - d.pore_true
    a = d.air_pred - d.air_true
    return {"n_patches": int(len(d)), "dice_material": float(d.dice_material.mean()),
            "dice_pore": float(d.dice_pore.mean()), "dice_air": float(d.dice_air.mean()),
            "porosity_mae": float(e.abs().mean()), "porosity_bias": float(e.mean()),
            "air_mae": float(a.abs().mean()), "air_bias": float(a.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=REPO / "data/split_v4")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--max-patches", type=int, default=0, help="0 = every patch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "runs/campaigns/39-unet-segmenter")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    model = load_unet3d(str(a.ckpt), device)
    for split in a.splits:
        ds = GreyLabelPatches(a.data_root, split, augment=False)
        idx = np.arange(len(ds))
        if a.max_patches and a.max_patches < len(ds):
            idx = np.sort(np.random.default_rng(0).choice(len(ds), a.max_patches, replace=False))
        dl = DataLoader(Subset(ds, idx.tolist()), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        d = pd.DataFrame(per_patch(model, dl, device))
        meta = pd.read_parquet(a.data_root / "patch_index.parquet", columns=["volume_id"]).iloc[ds.rows[idx]]
        d["volume_id"] = meta["volume_id"].to_numpy()
        d["bin"] = pd.cut(d.pore_true, BINS, labels=BIN_NAMES, right=False)
        res = {"ckpt": str(a.ckpt), "split": split, "all": summarise(d),
               "per_bin": {b: summarise(g) for b, g in d.groupby("bin", observed=True)}}
        pv = pd.DataFrame([{"volume_id": v, **summarise(g)} for v, g in d.groupby("volume_id")])
        pv.to_csv(a.out / f"{split}_per_volume.csv", index=False)
        (a.out / f"{split}.json").write_text(json.dumps(res, indent=1))
        s = res["all"]
        print(f"{split}: n={s['n_patches']} pore Dice {s['dice_pore']:.4f} air Dice {s['dice_air']:.4f} "
              f"material Dice {s['dice_material']:.4f} porosity MAE {s['porosity_mae']:.5f} "
              f"bias {s['porosity_bias']:+.5f} air MAE {s['air_mae']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

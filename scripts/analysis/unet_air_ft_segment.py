#!/usr/bin/env python
"""Outside-air mask of a grey volume with an air-only fine-tuned U-Net (campaign 40).

Input = grey / 255 as read (no envelope, no quantile map), the same input as
``poregen.experiments.train_unet_air_ft``.  Sliding windows of ``--patch``
(default 96, the training patch) with 50 % overlap and a Hann blend of the
2-class logits [logsumexp(material, pore), air]; air = p(air) > 0.5.  The
volume is padded by reflection to fit whole windows.

Outputs (<out>/): air.tif (0/255, input axis order), report.json.

    python scripts/analysis/unet_air_ft_segment.py --ckpt <last.ckpt> --tif <grey.tif> --thickness-axis 1 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from poregen.experiments.train_unet_air_ft import air_logits
from poregen.models.unet3d import load_unet3d


def hann3(p: int) -> np.ndarray:
    h = np.hanning(p + 2)[1:-1].astype(np.float32)
    return h[:, None, None] * h[None, :, None] * h[None, None, :]


@torch.no_grad()
def p_air(model, g: np.ndarray, p: int, device, batch: int = 8) -> np.ndarray:
    st = p // 2
    pad = [(0, max(p - s, (-(s - p)) % st)) for s in g.shape]
    gp = np.pad(g, pad, mode="reflect")
    Z, Y, X = gp.shape
    org = [(z, y, x) for z in range(0, Z - p + 1, st) for y in range(0, Y - p + 1, st) for x in range(0, X - p + 1, st)]
    w = hann3(p)
    acc = np.zeros((2, Z, Y, X), np.float32); wacc = np.zeros((Z, Y, X), np.float32)
    for i in range(0, len(org), batch):
        ob = org[i:i + batch]
        xs = torch.stack([torch.from_numpy(gp[z:z + p, y:y + p, x:x + p]) for z, y, x in ob])[:, None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lo = model(xs)
        lo = air_logits(lo.float()).cpu().numpy()
        for k, (z, y, x) in enumerate(ob):
            acc[:, z:z + p, y:y + p, x:x + p] += lo[k] * w; wacc[z:z + p, y:y + p, x:x + p] += w
    lo = acc / wacc[None]
    pa = 1 / (1 + np.exp(lo[0] - lo[1]))
    return pa[tuple(slice(0, s) for s in g.shape)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0).astype(np.float32) / 255.0
    pa = p_air(load_unet3d(str(a.ckpt), device), g, a.patch, device)
    air = np.moveaxis(((pa > 0.5) * 255).astype(np.uint8), 0, a.thickness_axis)
    tifffile.imwrite(a.out / "air.tif", air, imagej=True)
    rep = {"tif": str(a.tif), "ckpt": str(a.ckpt), "patch": a.patch, "air_fraction": float((air > 0).mean()),
           "seconds": time.time() - t0}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

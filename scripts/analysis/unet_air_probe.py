#!/usr/bin/env python
"""Run the campaign-39 U-Net on a volume WITHOUT the hand envelope (campaign 40).

Separates what the model does from what the texture envelope does.  Input
modes (``--map``):
- ``raw``: the grey as read, divided by 255.
- ``specimen-q``: every voxel quantile-matched to the split_v4 coupon
  specimen grey (Na_05_2, as in campaign 39), with no envelope; air can then
  only be found from texture, not from a dark grey.
``--zoom`` rescales the volume first (linear, ``scipy.ndimage.zoom``, order 1,
anti-alias Gaussian sigma = (1/zoom - 1)/2 when zoom < 1); the prediction is
rescaled back (order 0) so all outputs keep the input shape.  zoom 0.04 takes a
1 um scan to the 25 um training voxel.

Outputs (<out>/): label.tif (0 material, 1 pore, 2 air), air.tif (0/255),
report.json.

    python scripts/analysis/unet_air_probe.py --ckpt <ckpt> --tif <grey.tif> --thickness-axis 1 \
        --map raw --zoom 1 --out <dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet_segment_volume import segment  # noqa: E402
from vae_segment_volume import AIR_GREY, P, reference_quantiles  # noqa: E402

from poregen.models.unet3d import load_unet3d  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--map", choices=("raw", "specimen-q"), required=True)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)

    v = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0).astype(np.float32)
    shape = v.shape
    if a.map == "specimen-q":
        q = np.linspace(0, 100, 1001)
        v = np.interp(v, np.percentile(v, q), reference_quantiles(q)).astype(np.float32)
    if a.zoom != 1.0:
        if a.zoom < 1:
            v = ndimage.gaussian_filter(v, (1 / a.zoom - 1) / 2)
        v = ndimage.zoom(v, a.zoom, order=1)
    zshape = v.shape
    pad = [(0, max(P - s, (-s) % P)) for s in zshape]
    g = np.pad(np.clip(v, 0, 255), pad, constant_values=AIR_GREY) / 255.0

    model = load_unet3d(str(a.ckpt), device)
    logits = segment(model, g.astype(np.float32), device)
    pred = logits.argmax(0).astype(np.uint8)[tuple(slice(0, s) for s in zshape)]
    if a.zoom != 1.0:
        pred = ndimage.zoom(pred, [s / z for s, z in zip(shape, zshape)], order=0)
        pred = np.pad(pred, [(0, max(0, s - p)) for s, p in zip(shape, pred.shape)], mode="edge")
        pred = pred[tuple(slice(0, s) for s in shape)]
    lab = np.moveaxis(pred, 0, a.thickness_axis)
    tifffile.imwrite(a.out / "label.tif", lab, imagej=True)
    tifffile.imwrite(a.out / "air.tif", ((lab == 2) * 255).astype(np.uint8), imagej=True)
    rep = {"tif": str(a.tif), "ckpt": str(a.ckpt), "map": a.map, "zoom": a.zoom, "model_input_shape": list(zshape),
           "class_fractions": {n: float((pred == k).mean()) for k, n in enumerate(("material", "pore", "air"))}}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

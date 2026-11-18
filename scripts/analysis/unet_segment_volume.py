#!/usr/bin/env python
"""Segment a grey volume with the standalone 3-class U-Net (campaign 39).

The U-Net sees grey only.  The input grey is put on the training grey scale
first, the same way as ``vae_segment_volume.py``: a texture-based envelope
(local std below ``--air-std``, connected to the faces of the thickness axis)
splits specimen from air, the specimen grey is quantile-matched to a real
split_v4 coupon (Na_05_2), and the envelope is set to the air grey 41.  The
envelope only fixes the grey mapping: the three classes come from the U-Net.
``--no-grey-map`` feeds the raw grey instead (for split_v4 data, already on
the training scale).

Windows of 64^3 at stride 32, Hann^3 blend of the logits, padding cut off,
input axis order restored.  Outputs (<out>/): label.tif (0 material, 1 pore,
2 air), material.tif / pore.tif / air.tif (uint8 0/255), pore_prob.tif,
input_mapped.tif, report.json, preview_*.png.

    python scripts/analysis/unet_segment_volume.py --ckpt runs/unet_seg/<run>/best.ckpt \
        --tif <file> --thickness-axis 1 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_segment_volume import AIR_GREY, P, STRIDE, air_envelope, hann3, reference_quantiles  # noqa: E402

from poregen.models.unet3d import load_unet3d  # noqa: E402


@torch.no_grad()
def segment(model, grey01: np.ndarray, device, batch: int = 16) -> np.ndarray:
    """Blended logits ``(3, Z, Y, X)`` of a padded grey volume."""
    Z, Y, X = grey01.shape
    origins = [(z, y, x) for z in range(0, Z - P + 1, STRIDE)
               for y in range(0, Y - P + 1, STRIDE) for x in range(0, X - P + 1, STRIDE)]
    w = hann3(P)
    wt = torch.from_numpy(w)
    acc = np.zeros((3, Z, Y, X), np.float32)
    wacc = np.zeros((Z, Y, X), np.float32)
    for s in range(0, len(origins), batch):
        ob = origins[s:s + batch]
        xs = torch.stack([torch.from_numpy(grey01[z:z + P, y:y + P, x:x + P]) for z, y, x in ob])[:, None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lo = model(xs.to(device))
        lo = lo.float().cpu()
        for i, (z, y, x) in enumerate(ob):
            sl = np.s_[z:z + P, y:y + P, x:x + P]
            acc[(slice(None), *sl)] += (lo[i] * wt).numpy()
            wacc[sl] += w
    return acc / wacc[None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=0)
    ap.add_argument("--air-std", type=float, default=15.0)
    ap.add_argument("--no-grey-map", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)

    raw = tifffile.imread(a.tif)
    v = np.moveaxis(raw, a.thickness_axis, 0)
    shape = v.shape
    report = {"input": str(a.tif), "input_shape_as_read": list(raw.shape), "thickness_axis": a.thickness_axis,
              "ckpt": str(a.ckpt), "grey_map": not a.no_grey_map}
    if a.no_grey_map:
        mapped = v
    else:
        env = air_envelope(v, a.air_std)
        q = np.linspace(0, 100, 1001)
        mapped = np.interp(v.astype(np.float32), np.percentile(v[~env], q), reference_quantiles(q))
        mapped = np.where(env, AIR_GREY, mapped).clip(0, 255).astype(np.uint8)
        report.update(air_std_threshold=a.air_std, envelope_air_fraction=float(env.mean()))
    pad = [(0, (-s) % P) for s in shape]
    g = np.pad(mapped, pad, constant_values=AIR_GREY)

    model = load_unet3d(str(a.ckpt), device)
    logits = segment(model, g.astype(np.float32) / 255.0, device)
    cut = tuple(slice(0, s) for s in shape)
    pred = logits.argmax(0).astype(np.uint8)[cut]
    pore_p = torch.softmax(torch.from_numpy(logits), 0).numpy()[1][cut]
    spec = pred != 2
    report.update(porosity_of_specimen=float((pred == 1).sum() / max(spec.sum(), 1)),
                  air_fraction=float((pred == 2).mean()),
                  class_fractions={n: float((pred == k).mean()) for k, n in enumerate(("material", "pore", "air"))})
    print(report, flush=True)

    def back(arr):
        return np.moveaxis(arr, 0, a.thickness_axis)
    lab = back(pred)
    tifffile.imwrite(a.out / "label.tif", lab, imagej=True)
    for k, name in enumerate(("material", "pore", "air")):
        tifffile.imwrite(a.out / f"{name}.tif", ((lab == k) * 255).astype(np.uint8), imagej=True)
    tifffile.imwrite(a.out / "pore_prob.tif", back((pore_p * 255).astype(np.uint8)), imagej=True)
    tifffile.imwrite(a.out / "input_mapped.tif", back(np.asarray(mapped)), imagej=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    gm_full = np.asarray(mapped)
    for tag, sl in (("through_thickness", np.s_[shape[0] // 2]), ("cross_section", np.s_[:, shape[1] // 2, :])):
        gm, lb = gm_full[sl], pred[sl]
        rgb = np.repeat(gm[..., None], 3, -1).copy()
        rgb[lb == 1] = (230, 40, 40); rgb[lb == 2] = (40, 90, 230)
        iio.imwrite(a.out / f"preview_{tag}.png",
                    np.concatenate([np.repeat(gm[..., None], 3, -1), rgb], axis=0)[:, :1000])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

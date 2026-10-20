#!/usr/bin/env python
"""Scan and VAE reconstructions side by side, as an MP4 that steps through z.

For viewing on a phone: each frame is one z-slice of one val patch, the input
scan on the left and one panel per ``--run`` to its right, upscaled with
nearest neighbour and labelled. The video plays every slice of patch 1, then
patch 2, and so on, with a title card line naming the patch (porosity, air
fraction, volume). Grey only (``decode_xct``), on the raw scan scale.

Patches: the same eight split_v4 val rows as vae_recon_tiff_grid.py
(``pick_rows``), so the video and the TIFF grids show the same places.

Usage:
    python scripts/analysis/vae_recon_video.py --out <dir>/name.mp4 \\
        --run "rf-16 25k=runs/vae/<run>:<ckpt>" --run ... [--fps 8 --scale 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from vae_val_l1 import load_model_and_patches  # same dir
from vae_recon_tiff_grid import pick_rows  # same dir

REPO = Path(__file__).resolve().parents[2]
GAP, HEAD, FOOT = 8, 36, 30


def text(img: np.ndarray, s: str, x: int, y: int, size: float = 0.7) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, 255, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=path[:ckpt]"')
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    rows, labels, recons, x_ref, df = None, [], [], None, None
    for spec in a.run:
        label, rest = spec.split("=", 1)
        parts = rest.split(":", 1)
        run, ckpt = REPO / parts[0], (parts[1] if len(parts) > 1 else "best.ckpt")
        model, ds, _, _, _ = load_model_and_patches(run, a.split, 1, 1, None, ckpt)
        if rows is None:
            df, rows = ds.df, pick_rows(ds.df)
        items = [ds[r] for r in rows]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        torch.manual_seed(0)
        batch_dev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            rec = decode_xct(model(*args).xct_out)[:, 0]
        x = batch_dev["xct"][:, 0]
        x_ref = x if x_ref is None else x_ref
        assert torch.equal(x, x_ref), "runs did not receive the same patches"
        labels.append(label); recons.append(rec)
        print(f"{label}: done", flush=True)

    u8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).numpy()  # noqa: E731
    panels = [("scan", u8(x_ref))] + [(l, u8(r)) for l, r in zip(labels, recons)]
    p, s = x_ref.shape[-1], a.scale
    cell = p * s
    w = len(panels) * cell + (len(panels) - 1) * GAP
    w += (-w) % 16  # H.264 macroblocks
    h = HEAD + cell + FOOT
    h += (-h) % 16
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=9, pixelformat="yuv420p",
                                macro_block_size=16)
    for i, r in enumerate(rows):
        meta = df.loc[r]
        vol = str(meta["volume_id"]).split("probetas_")[-1].replace("_volume_eq_aligned", "")
        info = f"patch {i + 1}/{len(rows)}  porosity {meta['porosity']:.3f}  air {meta['air_fraction']:.2f}  {vol}"
        for z in range(p):
            fr = np.zeros((h, w), np.uint8)
            for k, (lab, vol_u8) in enumerate(panels):
                x0 = k * (cell + GAP)
                fr[HEAD:HEAD + cell, x0:x0 + cell] = np.kron(vol_u8[i, z], np.ones((s, s), np.uint8))
                text(fr, lab, x0 + 6, HEAD - 10)
            text(fr, f"{info}  z {z + 1}/{p}", 6, HEAD + cell + 22, 0.55)
            writer.append_data(np.repeat(fr[..., None], 3, axis=2))
    writer.close()
    print("wrote", out, f"{len(rows)} patches x {p} slices at {a.fps} fps, {w}x{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

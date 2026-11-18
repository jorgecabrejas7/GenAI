#!/usr/bin/env python
"""SAM2.1 video propagation of the outside air through a volume (campaign 40).

Runs in the separate ``segfm`` env.  Zero per-volume input: the prompts come
from one general rule, "the two thickness faces are air and the mid-thickness
plane is specimen", which holds for any scan of a plate with an air margin.

The volume is put thickness-first (T, A, B) and percentile-normalised (1-99 %).
``--time-axis`` picks the axis played as video:
- ``A`` or ``B``: frames are T x (other in-plane axis).  Two objects, upper
  and lower air: positive points on row ``--margin`` (upper) / T-1-margin
  (lower), negative points on the mid row, at ``--points`` evenly spaced
  columns, on every ``--key-every``-th frame.
- ``T``: frames are in-plane A x B; one air object, positive points on a grid
  of frame 0 and frame T-1 (all air), propagated forward from 0 and backward
  from T-1, union of the two.
Long frames are cut into tiles of ``--tile`` px with 50 % overlap (SAM2
resizes every frame to 1024^2); overlaps are averaged in logit space.

Outputs (<out>/): air.tif (0/255, input axis order), report.json.

    python scripts/analysis/fm_sam2_air.py --tif <grey.tif> --thickness-axis 1 --time-axis A \
        --model large --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import tifffile
import torch
from huggingface_hub import hf_hub_download

MODELS = {"tiny": ("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
          "small": ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
          "base+": ("facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt",
                    "configs/sam2.1/sam2.1_hiera_b+.yaml"),
          "large": ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}


def tiles(n: int, size: int) -> list[int]:
    if n <= size:
        return [0]
    step = size // 2
    s = list(range(0, n - size, step))
    return s + [n - size]


def run_video(pred, frames: np.ndarray, prompts: dict, device, reverse_from: int | None = None) -> np.ndarray:
    """frames (F, H, W) uint8; prompts {frame: {obj: (points xy, labels)}} -> logits (F, H, W) max over objects."""
    tmp = Path(tempfile.mkdtemp(prefix="sam2v_"))
    try:
        for i, f in enumerate(frames):
            iio.imwrite(tmp / f"{i:05d}.jpg", np.repeat(f[..., None], 3, -1), quality=100)
        out = np.full(frames.shape, -20.0, np.float32)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            st = pred.init_state(video_path=str(tmp), offload_video_to_cpu=True)
            for fi, objs in prompts.items():
                for oid, (pts, lab) in objs.items():
                    pred.add_new_points_or_box(st, frame_idx=fi, obj_id=oid, points=pts, labels=lab)
            starts = [(None, False), (None, True)] if reverse_from is None else [(0, False), (reverse_from, True)]
            for start, rev in starts:
                for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=start, reverse=rev):
                    lo = logits.float().max(0).values[0].cpu().numpy()
                    out[fi] = np.maximum(out[fi], lo)
            pred.reset_state(st)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--time-axis", choices=("T", "A", "B"), required=True)
    ap.add_argument("--model", choices=tuple(MODELS), default="large")
    ap.add_argument("--ckpt", type=Path, default=None, help="fine-tuned checkpoint (same architecture as --model)")
    ap.add_argument("--crop", default=None, help="a0:a1 crop on axis A after the thickness move (to test a part)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--key-every", type=int, default=0, help="0 = only the middle frame")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415
    repo, fn, cfg = MODELS[a.model]
    ck = str(a.ckpt) if a.ckpt else hf_hub_download(repo, fn)
    pred = build_sam2_video_predictor(cfg, ck, device=device)

    v = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0)
    full_shape = v.shape
    if a.crop:
        c0, c1 = map(int, a.crop.split(":"))
        v = v[:, c0:c1]
    lo, hi = np.percentile(v, [1, 99])
    v8 = (np.clip((v.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    T, A, B = v8.shape
    t0 = time.time()

    if a.time_axis == "T":
        vid = v8  # (T, A, B)
        logit = np.full(vid.shape, -20.0, np.float32)
        cnt = np.zeros(vid.shape, np.float32)
        for ta in tiles(A, a.tile):
            for tb in tiles(B, a.tile):
                fr = vid[:, ta:ta + a.tile, tb:tb + a.tile]
                h, w = fr.shape[1:]
                gy, gx = np.meshgrid(np.linspace(0.1, 0.9, a.points) * h, np.linspace(0.1, 0.9, a.points) * w,
                                     indexing="ij")
                pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
                lab = np.ones(len(pts), np.int32)
                pr = {0: {1: (pts, lab)}, T - 1: {1: (pts, lab)}}
                lg = run_video(pred, fr, pr, device, reverse_from=T - 1)
                sl = np.s_[:, ta:ta + a.tile, tb:tb + a.tile]
                logit[sl] = np.where(cnt[sl] == 0, lg, logit[sl] + lg)
                cnt[sl] += 1
        logit /= np.maximum(cnt, 1)
    else:
        vid = np.moveaxis(v8, 1 if a.time_axis == "A" else 2, 0)  # (F, T, W)
        F, _, W = vid.shape
        logit = np.zeros(vid.shape, np.float32)
        cnt = np.zeros(vid.shape, np.float32)
        keys = [F // 2] if a.key_every <= 0 else list(range(a.key_every // 2, F, a.key_every))
        for tw in tiles(W, a.tile):
            fr = vid[:, :, tw:tw + a.tile]
            w = fr.shape[2]
            xs = (np.linspace(0.1, 0.9, a.points) * w).astype(np.float32)
            neg = np.stack([xs, np.full_like(xs, T // 2)], 1)
            up = np.concatenate([np.stack([xs, np.full_like(xs, a.margin)], 1), neg]).astype(np.float32)
            dn = np.concatenate([np.stack([xs, np.full_like(xs, T - 1 - a.margin)], 1), neg]).astype(np.float32)
            lab = np.r_[np.ones(len(xs)), np.zeros(len(xs))].astype(np.int32)
            pr = {k: {1: (up, lab), 2: (dn, lab)} for k in keys}
            lg = run_video(pred, fr, pr, device)
            logit[:, :, tw:tw + a.tile] += lg
            cnt[:, :, tw:tw + a.tile] += 1
        logit = np.moveaxis(logit / np.maximum(cnt, 1), 0, 1 if a.time_axis == "A" else 2)

    air = logit > 0
    if a.crop:
        full = np.zeros(full_shape, bool)
        full[:, c0:c1] = air
        air = full
    tifffile.imwrite(a.out / "air.tif", (np.moveaxis(air, 0, a.thickness_axis) * 255).astype(np.uint8), imagej=True)
    rep = {"tif": str(a.tif), "model": a.model, "ckpt": ck, "time_axis": a.time_axis, "crop": a.crop, "tile": a.tile,
           "points": a.points, "margin": a.margin, "key_every": a.key_every, "seconds": round(time.time() - t0, 1),
           "air_fraction_in_crop": float(air[:, c0:c1].mean() if a.crop else air.mean()),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Frozen DINOv2 features + a linear air/specimen head from a few slices (campaign 40).

Runs in the separate ``segfm`` env (torch, transformers), not in poregen.

Each volume is put thickness-first (T, A, B) and cut into 2D slices T x B along
A.  A slice is percentile-normalised per volume (1-99 %, so no grey-level
prior), repeated to 3 channels, resized by ``--upscale`` to a multiple of the
14-px patch, and fed to DINOv2.  A logistic regression is fitted on the patch
tokens of ``--train-slices`` slices per training volume, using only patches
that are entirely air or entirely specimen in the training label.  Prediction
is per patch, bilinear-upsampled to the voxel grid; slices are stacked to 3D.
``*_face`` outputs keep only air components connected to the two thickness
faces (outside air touches the faces; a general topological rule, no
threshold).

Volume spec (``NAME=k=v,k=v``):
- ``tif=<path>,axis=<thickness axis>[,label=<air tif>,margin=<px>]``: label is
  an air mask; with margin > 0 only voxels farther than margin from its
  boundary are used for training (a stand-in for sparse scribbles).
- ``zarr=<split_v4 key substring>,crop=<a0>:<a1>``: split_v4 coupon, thickness
  axis 0, crop on axis 1; air = 1 - sample_mask (reference).
Outputs per applied volume (<out>/<name>/): air.tif, air_face.tif (0/255,
input axis order) and an entry in <out>/metrics.json (air fraction; air Dice
when a reference exists).

    python scripts/analysis/fm_dino_air.py --model facebook/dinov2-base \
        --train na=zarr=Na_04_2,crop=1500:1700 --apply teijin=tif=<grey.tif>,axis=1 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def parse(spec: str) -> tuple[str, dict]:
    name, rest = spec.split("=", 1)
    return name, dict(kv.split("=", 1) for kv in rest.split(","))


def load(d: dict):
    """Grey (T, A, B) uint8, training label (air bool or None), reference air or None, thickness axis."""
    if "tif" in d:
        ax = int(d["axis"])
        g = np.moveaxis(tifffile.imread(d["tif"]), ax, 0)
        lab = None
        if "label" in d:
            air = np.moveaxis(tifffile.imread(d["label"]) > 0, ax, 0)
            m = int(d.get("margin", 0))
            if m > 0:
                inner_air = ndimage.distance_transform_edt(air) > m
                inner_spec = ndimage.distance_transform_edt(~air) > m
                lab = np.where(inner_air, 1, np.where(inner_spec, 0, -1)).astype(np.int8)
            else:
                lab = air.astype(np.int8)
        return g, lab, None, ax
    import zarr  # noqa: PLC0415
    z = zarr.open(str(REPO / "data/split_v4/volumes.zarr"), mode="r")
    key = [k for k in z.keys() if d["zarr"] in k][0]
    a0, a1 = map(int, d["crop"].split(":"))
    g = np.asarray(z[key]["xct"][:, a0:a1])
    ref = np.asarray(z[key]["sample_mask"][:, a0:a1]) == 0
    return g, ref.astype(np.int8), ref, 0


class Dino:
    def __init__(self, name: str, device, upscale: float):
        from transformers import AutoModel  # noqa: PLC0415
        self.m = AutoModel.from_pretrained(name).eval().to(device)
        self.dev, self.up, self.p = device, upscale, 14

    @torch.no_grad()
    def tokens(self, img01: np.ndarray) -> torch.Tensor:
        """(h, w, C) patch tokens of a 2D slice in [0, 1]."""
        H, W = img01.shape
        h, w = max(1, round(H * self.up / self.p)), max(1, round(W * self.up / self.p))
        x = torch.from_numpy(img01).float()[None, None]
        x = F.interpolate(x, size=(h * self.p, w * self.p), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = (x - torch.tensor(MEAN).view(1, 3, 1, 1).float()) / torch.tensor(STD).view(1, 3, 1, 1).float()
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            out = self.m(pixel_values=x.to(self.dev)).last_hidden_state
        return out[0, 1:].float().reshape(h, w, -1).cpu()


def norm01(g: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def patch_labels(lab2d: np.ndarray, h: int, w: int) -> np.ndarray:
    """-1 where a patch is mixed or unlabelled, else its class."""
    t = torch.from_numpy(lab2d.astype(np.float32))[None, None]
    known = F.adaptive_avg_pool2d((t >= 0).float(), (h, w))[0, 0].numpy()
    frac = F.adaptive_avg_pool2d((t == 1).float(), (h, w))[0, 0].numpy()
    out = np.full((h, w), -1, np.int8)
    out[(known > 0.999) & (frac > 0.999)] = 1
    out[(known > 0.999) & (frac < 0.001)] = 0
    return out


def face_air(air: np.ndarray) -> np.ndarray:
    lab, _ = ndimage.label(air)
    keep = np.setdiff1d(np.union1d(np.unique(lab[0]), np.unique(lab[-1])), [0])
    return np.isin(lab, keep)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--train-slices", type=int, default=2)
    ap.add_argument("--apply", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    enc = Dino(a.model, torch.device(a.device), a.upscale)

    X, Y, used = [], [], {}
    for spec in a.train:
        name, d = parse(spec)
        g, lab, _, _ = load(d)
        g01 = norm01(g)
        idx = np.linspace(0, g.shape[1] - 1, a.train_slices + 2).round().astype(int)[1:-1]
        used[name] = idx.tolist()
        for i in idx:
            tok = enc.tokens(g01[:, i, :])
            pl = patch_labels(lab[:, i, :], *tok.shape[:2])
            X.append(tok[pl >= 0].numpy()); Y.append(pl[pl >= 0])
    X, Y = np.concatenate(X), np.concatenate(Y)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, Y)
    metrics = {"model": a.model, "upscale": a.upscale, "train": a.train, "train_slices_used": used,
               "n_train_patches": int(len(Y)), "train_air_patch_fraction": float(Y.mean()), "apply": {}}
    w = torch.from_numpy(clf.coef_[0]).float(); b = float(clf.intercept_[0])

    for spec in a.apply:
        name, d = parse(spec)
        g, _, ref, ax = load(d)
        g01 = norm01(g)
        t0 = time.time()
        logit = np.zeros(g.shape, np.float32)
        for i in range(g.shape[1]):
            tok = enc.tokens(g01[:, i, :])
            lo = (tok @ w + b)[None, None]
            logit[:, i, :] = F.interpolate(lo, size=(g.shape[0], g.shape[2]), mode="bilinear",
                                           align_corners=False)[0, 0].numpy()
        air = logit > 0
        airf = face_air(air)
        o = a.out / name
        o.mkdir(exist_ok=True)
        for tag, m in (("air", air), ("air_face", airf)):
            tifffile.imwrite(o / f"{tag}.tif", (np.moveaxis(m, 0, ax) * 255).astype(np.uint8), imagej=True)
        r = {"spec": spec, "seconds": round(time.time() - t0, 1), "air_fraction": float(air.mean()),
             "air_face_fraction": float(airf.mean())}
        if ref is not None:
            r.update(ref_air_fraction=float(ref.mean()), air_dice=dice(air, ref), air_face_dice=dice(airf, ref))
        metrics["apply"][name] = r
        print(name, r, flush=True)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

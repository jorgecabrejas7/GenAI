#!/usr/bin/env python
"""Segment a NEW grey volume with the production VAE, without reference masks.

The three-channel encoder needs a pore plane and an air plane beside the grey.
For a scan with no labels this script gives it (author 2026-11-17: "try to
segment with the VAE, outputting 3 volume masks"):

* AIR: a geometric envelope from local texture. Air in a scan is smooth and
  material is speckled, so the local standard deviation (box window) below a
  threshold, restricted to regions that reach the top or bottom face along the
  thickness axis, is air. No grey threshold and no reference segmentation.
* PORE: zeros in round 1, then the VAE's own pore prediction fed back for
  ``--rounds`` - 1 further passes.

Grey mapping: the production VAE saw equalised scans (specimen median 210,
air 41). The input's specimen voxels are quantile-matched to a real split_v4
coupon's specimen distribution (monotone map), air voxels are set to 41. The
thickness axis is moved to axis 0 (plies stacked along z, as in training).
Windows of 64^3 at stride 32 with a Hann^3 blend of logits and grey; the
outputs are cut back to the input extent and axis order.

Outputs (<out>/): label.tif (0 material, 1 pore, 2 air), material.tif /
pore.tif / air.tif (uint8 0/255), pore_prob.tif (uint8 p*255),
grey_decoded.tif, input_mapped.tif, report.json, preview_*.png.

    python scripts/analysis/vae_segment_volume.py --tif <file> --thickness-axis 1 --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import tifffile
import torch
import yaml
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
P, STRIDE = 64, 32
AIR_GREY = 41
N_CLASSES = 3


def local_std(v: np.ndarray, size: int = 7) -> np.ndarray:
    f = v.astype(np.float32)
    m = ndimage.uniform_filter(f, size)
    m2 = ndimage.uniform_filter(f * f, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def air_envelope(v: np.ndarray, thr: float) -> np.ndarray:
    """Smooth regions connected to the z faces (thickness axis = 0)."""
    smooth = local_std(v) < thr
    smooth = ndimage.binary_opening(smooth, structure=np.ones((5, 5, 5), bool))
    lab, n = ndimage.label(smooth)
    keep = set(np.unique(lab[0])) | set(np.unique(lab[-1]))
    keep.discard(0)
    air = np.isin(lab, list(keep))
    return ndimage.binary_closing(air, structure=np.ones((3, 3, 3), bool))


def reference_quantiles(q: np.ndarray) -> np.ndarray:
    import zarr  # noqa: PLC0415
    z = zarr.open(str(REPO / "data/split_v4/volumes.zarr"), mode="r")
    k = [x for x in z.keys() if "Na_05_2" in x][0]
    g = z[k]
    x = np.asarray(g["xct"][:, 800:1400, 500:1100])
    sm = np.asarray(g["sample_mask"][:, 800:1400, 500:1100]) == 1
    return np.percentile(x[sm], q)


def load_vae(run: Path, ckpt: str, device):
    from poregen.experiments.train_vae import build_model  # noqa: PLC0415
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    model = build_model(cfg, device)
    state = torch.load(run / ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def hann3(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    w = np.clip(w, 0.05, None)
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


@torch.no_grad()
def one_pass(model, grey01: np.ndarray, label: np.ndarray, device, batch: int = 32):
    Z, Y, X = grey01.shape
    origins = [(z, y, x) for z in range(0, Z - P + 1, STRIDE)
               for y in range(0, Y - P + 1, STRIDE) for x in range(0, X - P + 1, STRIDE)]
    w = hann3(P)
    logit_acc = np.zeros((N_CLASSES, Z, Y, X), np.float32)
    grey_acc = np.zeros((Z, Y, X), np.float32)
    w_acc = np.zeros((Z, Y, X), np.float32)
    wt = torch.from_numpy(w)
    for s in range(0, len(origins), batch):
        ob = origins[s:s + batch]
        xs = torch.stack([torch.from_numpy(grey01[z:z + P, y:y + P, x:x + P]) for z, y, x in ob])[:, None]
        ls = torch.stack([torch.from_numpy(label[z:z + P, y:y + P, x:x + P].astype(np.int64)) for z, y, x in ob])
        out = model(xs.to(device), ls.to(device))
        cl = out.class_logits.float().cpu()
        gr = out.xct_out.float().cpu()[:, 0].clamp(0, 1)
        for i, (z, y, x) in enumerate(ob):
            sl = np.s_[z:z + P, y:y + P, x:x + P]
            logit_acc[(slice(None), *sl)] += (cl[i] * wt).numpy()
            grey_acc[sl] += (gr[i] * wt).numpy()
            w_acc[sl] += w
    logit_acc /= w_acc[None]
    grey_acc /= w_acc
    return logit_acc, grey_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=0, help="axis of the input along which plies stack")
    ap.add_argument("--run", type=Path, default=None, help="VAE run dir (default: the paper's rf-16 run-0020)")
    ap.add_argument("--ckpt", default=None, help="checkpoint file name (default: step 25000)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--air-std", type=float, default=15.0, help="local-std threshold below which a voxel is smooth (air)")
    ap.add_argument("--no-air-channel", action="store_true", help="feed zeros as the air plane too (pure no-mask test)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    run = a.run or Path(glob.glob(str(REPO / "runs/vae/r08-run-0020-2026*"))[0])
    ckpt = a.ckpt or Path(glob.glob(str(run / "*_step00025000.ckpt"))[0]).name
    a.out.mkdir(parents=True, exist_ok=True)

    raw = tifffile.imread(a.tif)
    v = np.moveaxis(raw, a.thickness_axis, 0)
    orig_shape = v.shape
    air = air_envelope(v, a.air_std)
    spec = ~air
    # quantile matching of the specimen grey to the reference coupon
    q = np.linspace(0, 100, 1001)
    src_q = np.percentile(v[spec], q)
    ref_q = reference_quantiles(q)
    mapped = np.interp(v.astype(np.float32), src_q, ref_q)
    mapped = np.where(air, AIR_GREY, mapped).clip(0, 255).astype(np.uint8)
    # pad to the 64-voxel grid with air
    pad = [(0, (-s) % P) for s in orig_shape]
    g = np.pad(mapped, pad, constant_values=AIR_GREY)
    airp = np.pad(air, pad, constant_values=True)
    grey01 = g.astype(np.float32) / 255.0

    model = load_vae(run, ckpt, device)
    label = np.where(airp, 2, 0).astype(np.uint8)
    if a.no_air_channel:
        label[:] = 0
    report = {"input": str(a.tif), "input_shape_as_read": list(raw.shape), "thickness_axis": a.thickness_axis,
              "vae_run": run.name, "ckpt": ckpt, "air_std_threshold": a.air_std,
              "air_fraction_envelope": float(air.mean()), "grey_map": {"source_quantiles_1_50_99": [float(x) for x in np.percentile(v[spec], [1, 50, 99])],
              "reference_quantiles_1_50_99": [float(x) for x in np.percentile(ref_q, [1, 50, 99])]}, "rounds": []}
    prev = None
    for r in range(a.rounds):
        logits, grey = one_pass(model, grey01, label, device)
        pred = logits.argmax(0).astype(np.uint8)
        if not a.no_air_channel:
            pred[airp] = 2
        cut = tuple(slice(0, s) for s in orig_shape)
        pc = pred[cut]
        row = {"round": r + 1, "porosity_of_specimen": float((pc == 1).sum() / max((pc != 2).sum(), 1)),
               "air_fraction": float((pc == 2).mean()),
               "changed_voxels_vs_previous": None if prev is None else float((pc != prev).mean())}
        report["rounds"].append(row)
        print(row, flush=True)
        prev = pc
        label = np.where(airp, 2, np.where(pred == 1, 1, 0)).astype(np.uint8)
        if a.no_air_channel:
            label = np.where(pred == 1, 1, 0).astype(np.uint8)
    probs = torch.softmax(torch.from_numpy(logits), 0).numpy()

    def back(arr):  # cut the padding, restore the input axis order
        return np.moveaxis(arr[tuple(slice(0, s) for s in orig_shape)], 0, a.thickness_axis)
    lab = back(pred)
    tifffile.imwrite(a.out / "label.tif", lab, imagej=True)
    for k, name in enumerate(("material", "pore", "air")):
        tifffile.imwrite(a.out / f"{name}.tif", ((lab == k) * 255).astype(np.uint8), imagej=True)
    tifffile.imwrite(a.out / "pore_prob.tif", back((probs[1] * 255).astype(np.uint8)), imagej=True)
    tifffile.imwrite(a.out / "grey_decoded.tif", back((grey * 255).round().astype(np.uint8)), imagej=True)
    tifffile.imwrite(a.out / "input_mapped.tif", back(g), imagej=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    # previews: mid slice of the thickness axis and a cross-section, input | mapped | label overlay
    zc = orig_shape[0] // 2
    for tag, sl in (("through_thickness", np.s_[zc]), ("cross_section", np.s_[:, orig_shape[1] // 2, :])):
        gm = g[sl][: orig_shape[1] if tag == "through_thickness" else orig_shape[0], : orig_shape[2]]
        lb = pred[sl][: gm.shape[0], : gm.shape[1]]
        rgb = np.repeat(gm[..., None], 3, -1).copy()
        rgb[lb == 1] = (230, 40, 40); rgb[lb == 2] = (40, 90, 230)
        strip = np.concatenate([np.repeat(gm[..., None], 3, -1), rgb], axis=0)
        iio.imwrite(a.out / f"preview_{tag}.png", strip[:, :1000])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

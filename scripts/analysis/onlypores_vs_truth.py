"""Score onlypores, the model's own mask and the audit dark detector against a
hand-made ground-truth pore mask.

Two sub-commands:

``export``  Write the grayscale sub-region you are going to annotate, together
            with the three masks the region currently gets, an empty
            ``truth_template.tif`` to paint on, and a ``region_meta.json`` that
            records exactly which volume and offset the region came from.

``score``   Read your finished ground-truth TIFF and report Dice / IoU /
            recall / precision (plus the porosity each method implies) for
              (i)   the onlypores pore mask,
              (ii)  the model's own ``mask.tif`` (or the stored dataset mask
                    for a real volume),
              (iii) the audit's dark-voxel detector.

Ground-truth convention
-----------------------
* One TIFF, ``(dz, dy, dx)`` = exactly the shape of the region.  A single
  annotated slice may be saved as a plain 2-D ``(dy, dx)`` image; it is then
  scored against the one slice at ``--offset``.
* Any NON-ZERO voxel is a PORE.  Zero is not a pore.  8-bit, 16-bit or float
  all work; the file is read as ``> 0``.
* The region origin is ``--offset z,y,x`` in the ORIGINAL volume's index space
  (the same indices ImageJ shows).  With ``--size`` omitted the whole volume is
  used.

Both sub-commands run the segmentation on the WHOLE volume before cropping to
the region, because the material mask uses a global Otsu threshold over the
whole volume — scoring a crop in isolation would not be the same computation
that produced the numbers under ``runs/campaigns/04-measurement-limits/onlypores_generated/``.

Examples
--------
    # 1. cut out a region to annotate (64 x 256 x 256 at z=64, y=64, x=64)
    python scripts/analysis/onlypores_vs_truth.py export \\
        --volume runs/campaigns/03-eval-v2-buggy-decode/volumes/dose_response/joint_oob/target_0.02_seed_101 \\
        --offset 64,64,64 --size 64,256,256 --name t0.02_centre

    # 2. paint pores in Fiji on region_grayscale.tif, save as truth.tif in the
    #    same folder, then:
    python scripts/analysis/onlypores_vs_truth.py score \\
        --truth runs/campaigns/04-measurement-limits/onlypores_inspection/ground_truth/t0.02_centre/truth.tif

    # 3. same truth, re-tuned onlypores knobs
    python scripts/analysis/onlypores_vs_truth.py score \\
        --truth .../truth.tif --sauvola-radius 15 --sauvola-k 0.25 \\
        --material-approach fixed_t_best --tag retuned
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, savefig, set_style  # noqa: E402
from onlypores_inspection import (  # noqa: E402
    MIN_CC, SAUVOLA_K, SAUVOLA_RADIUS, T_BEST, ZARR_ROOT,
    detector_mask, material_mask_debug, sauvola_debug, to_native_u8,
)

sys.path.insert(0, str(REPO / "src"))
from poregen.dataset import segmentation as seg  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

GT_ROOT = REPO / "runs" / "campaigns" / "04-measurement-limits" / "onlypores_inspection" / "ground_truth"


# ---------------------------------------------------------------------------
# Volume access
# ---------------------------------------------------------------------------

def load_volume(volume: str) -> tuple[np.ndarray, np.ndarray, str]:
    """-> (decoder-native u8 volume, reference mask, description).

    ``volume`` is either a generated volume directory (holding ``volume.tif``
    and ``mask.tif``) or the id of a volume in ``data/split_v2/volumes.zarr``.
    """
    p = Path(volume)
    if not p.is_absolute():
        p = REPO / p
    if p.is_dir() and (p / "volume.tif").exists():
        native = to_native_u8(p / "volume.tif")
        ref = tifffile.imread(str(p / "mask.tif")) > 0
        return native, ref, f"generated volume {p}"

    g = zarr.open(str(ZARR_ROOT), mode="r")
    if volume in g:
        grp = g[volume]
        return np.asarray(grp["xct"]), np.asarray(grp["mask"]) > 0, f"real volume {volume}"

    raise SystemExit(
        f"--volume {volume!r} is neither a generated volume directory containing "
        f"volume.tif/mask.tif nor a volume id in {ZARR_ROOT}")


def segment(native: np.ndarray, sauvola_radius: int, sauvola_k: float,
            material_approach: str, material_component: str,
            min_size_filtering: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """onlypores with exposed knobs -> (pore_mask, sample_mask, info).

    With the default arguments this is bit-identical to
    ``poregen.dataset.segmentation.onlypores(native)``.
    """
    bbox = seg.content_bbox(native)
    if bbox is None:
        raise SystemExit("volume has no non-zero voxel")
    z0, z1, y0, y1, x0, x1 = bbox
    cropped = native[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]

    binary, _ = sauvola_debug(cropped, sauvola_radius, sauvola_k)
    mm = material_mask_debug(cropped, material_approach, material_component, True)

    pore = np.zeros(native.shape, bool)
    sample = np.zeros(native.shape, bool)
    pore[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = np.logical_and(~binary, mm["sample"])
    sample[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = mm["sample"]
    del binary, mm

    if min_size_filtering > 0:
        pore = seg.clean_pores(pore, min_size=min_size_filtering)

    info = {"content_bbox": [int(v) for v in bbox],
            "sauvola_radius": sauvola_radius, "sauvola_k": sauvola_k,
            "material_approach": material_approach,
            "material_component": material_component,
            "min_size_filtering": min_size_filtering}
    return pore, sample, info


def _region(shape: tuple[int, int, int], offset, size) -> tuple[slice, slice, slice]:
    o = [0, 0, 0] if offset is None else [int(v) for v in offset]
    s = list(shape) if size is None else [int(v) for v in size]
    for i in range(3):
        if o[i] < 0 or o[i] + s[i] > shape[i]:
            raise SystemExit(
                f"region axis {i}: offset {o[i]} + size {s[i]} = {o[i]+s[i]} "
                f"exceeds the volume extent {shape[i]}")
    return tuple(slice(o[i], o[i] + s[i]) for i in range(3))


def _parse_triple(s: str | None):
    if s is None:
        return None
    parts = [p for p in s.replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(f"expected three comma-separated integers, got {s!r}")
    return [int(p) for p in parts]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(a: argparse.Namespace) -> None:
    native, ref, desc = load_volume(a.volume)
    off = _parse_triple(a.offset)
    siz = _parse_triple(a.size)
    sl = _region(native.shape, off, siz)
    name = a.name or (Path(a.volume).name + "_" +
                      "-".join(str(s.start) for s in sl))
    out = GT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    pore, sample, info = segment(native, a.sauvola_radius, a.sauvola_k,
                                 a.material_approach, a.material_component,
                                 a.min_size_filtering)
    det = detector_mask(native, a.detector_threshold, a.detector_min_cc)

    gray = np.ascontiguousarray(native[sl])
    tifffile.imwrite(out / "region_grayscale.tif", gray,
                     imagej=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(out / "truth_template.tif", np.zeros_like(gray),
                     imagej=True, metadata={"axes": "ZYX"})
    for fname, arr in (("region_onlypores.tif", pore[sl]),
                       ("region_sample_mask.tif", sample[sl]),
                       ("region_model_mask.tif", ref[sl]),
                       ("region_detector.tif", det[sl])):
        tifffile.imwrite(out / fname, (np.ascontiguousarray(arr).astype(np.uint8) * 255),
                         imagej=True, metadata={"axes": "ZYX"})

    meta = {
        "volume": a.volume,
        "volume_description": desc,
        "volume_shape": [int(s) for s in native.shape],
        "offset_zyx": [int(s.start) for s in sl],
        "size_zyx": [int(s.stop - s.start) for s in sl],
        "segmentation": info,
        "detector": {"threshold": a.detector_threshold, "min_cc": a.detector_min_cc},
        "truth_convention": "any non-zero voxel is a pore; shape must equal size_zyx",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "region_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nRegion written to {out}")
    print(f"  region_grayscale.tif   {gray.shape}  <- annotate THIS")
    print(f"  truth_template.tif     empty stack of the same shape")
    print(f"  region_onlypores.tif / region_model_mask.tif / region_detector.tif / "
          f"region_sample_mask.tif   (current results, for reference)")
    print(f"\nSave your finished mask as {out / 'truth.tif'} and run:\n"
          f"  python scripts/analysis/onlypores_vs_truth.py score "
          f"--truth {out / 'truth.tif'}\n")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.count_nonzero(pred & truth))
    npred, ntru = int(pred.sum()), int(truth.sum())
    fp, fn = npred - tp, ntru - tp
    union = npred + ntru - tp
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "dice": (2.0 * tp / (npred + ntru)) if (npred + ntru) else float("nan"),
        "iou": (tp / union) if union else float("nan"),
        "recall": (tp / ntru) if ntru else float("nan"),
        "precision": (tp / npred) if npred else float("nan"),
        "n_predicted": npred,
    }


def cmd_score(a: argparse.Namespace) -> None:
    truth_path = Path(a.truth)
    if not truth_path.is_absolute():
        truth_path = REPO / truth_path
    if not truth_path.exists():
        raise SystemExit(f"ground-truth file not found: {truth_path}")

    meta_path = truth_path.parent / "region_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    volume = a.volume or meta.get("volume")
    if volume is None:
        raise SystemExit("no --volume given and no region_meta.json next to the truth file")
    off = _parse_triple(a.offset) or meta.get("offset_zyx")

    truth = np.squeeze(tifffile.imread(str(truth_path)))
    if truth.ndim == 2:            # a single annotated slice
        truth = truth[None]
    if truth.ndim != 3:
        raise SystemExit(f"truth must be a 2-D slice or a 3-D stack, got shape {truth.shape}")
    truth = truth > 0

    native, ref, desc = load_volume(volume)
    sl = _region(native.shape, off, truth.shape)
    print(f"volume : {desc}  shape {native.shape}")
    print(f"region : offset {[s.start for s in sl]}  size {truth.shape}")
    print(f"truth  : {int(truth.sum())} pore voxels "
          f"({truth.mean():.5f} of the region)\n")

    pore, sample, info = segment(native, a.sauvola_radius, a.sauvola_k,
                                 a.material_approach, a.material_component,
                                 a.min_size_filtering)
    det = detector_mask(native, a.detector_threshold, a.detector_min_cc)

    pore_r = np.ascontiguousarray(pore[sl])
    sample_r = np.ascontiguousarray(sample[sl])
    ref_r = np.ascontiguousarray(ref[sl])
    det_r = np.ascontiguousarray(det[sl])
    gray_r = np.ascontiguousarray(native[sl])
    del pore, sample, ref, det, native

    methods = {
        "onlypores": pore_r,
        "model_mask": ref_r,
        "audit_detector": det_r,
    }
    n_reg = int(truth.size)
    n_sm = int(sample_r.sum())
    res = {
        "truth_file": str(truth_path),
        "volume": volume,
        "volume_description": desc,
        "region_offset_zyx": [int(s.start) for s in sl],
        "region_size_zyx": [int(s) for s in truth.shape],
        "segmentation": info,
        "detector": {"threshold": a.detector_threshold, "min_cc": a.detector_min_cc},
        "region_sample_mask_fraction": n_sm / n_reg,
        "truth_porosity_over_region": int(truth.sum()) / n_reg,
        "truth_porosity_over_sample_mask": (int(np.count_nonzero(truth & sample_r)) / n_sm
                                            if n_sm else float("nan")),
        "over_region": {}, "over_sample_mask": {},
    }
    for name, pred in methods.items():
        res["over_region"][name] = {
            **_metrics(pred, truth),
            "porosity_over_region": int(pred.sum()) / n_reg,
        }
        res["over_sample_mask"][name] = {
            **_metrics(pred & sample_r, truth & sample_r),
            "porosity_over_sample_mask": (int(np.count_nonzero(pred & sample_r)) / n_sm
                                          if n_sm else float("nan")),
        }

    hdr = f"{'method':<16}{'Dice':>8}{'IoU':>8}{'recall':>9}{'precision':>11}{'porosity':>11}"
    for scope, key, denom in (("WHOLE REGION", "over_region", "porosity_over_region"),
                              ("INSIDE sample_mask", "over_sample_mask", "porosity_over_sample_mask")):
        print(f"--- {scope} ---")
        print(hdr)
        for name in methods:
            m = res[key][name]
            print(f"{name:<16}{m['dice']:>8.4f}{m['iou']:>8.4f}{m['recall']:>9.4f}"
                  f"{m['precision']:>11.4f}{m[denom]:>11.5f}")
        tp = (res["truth_porosity_over_region"] if key == "over_region"
              else res["truth_porosity_over_sample_mask"])
        print(f"{'GROUND TRUTH':<16}{'-':>8}{'-':>8}{'-':>9}{'-':>11}{tp:>11.5f}\n")
    print(f"sample_mask covers {res['region_sample_mask_fraction']:.4f} of the region")

    tag = f"_{a.tag}" if a.tag else ""
    out_json = truth_path.parent / f"score{tag}.json"
    out_json.write_text(json.dumps(res, indent=2))
    _score_figure(truth_path.parent, tag, gray_r, truth, methods, res)
    print(f"\nwrote {out_json}")


def _score_figure(out_dir: Path, tag: str, gray: np.ndarray, truth: np.ndarray,
                  methods: dict, res: dict) -> None:
    set_style()
    z = int(np.argmax(truth.reshape(truth.shape[0], -1).sum(axis=1)))
    base = (gray[z].astype(np.float32) / 255.0) * 0.75
    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 5.0))
    axes[0].imshow(gray[z], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title(f"grayscale, z = {z} of the region", fontsize=9)
    for ax, (name, pred) in zip(axes[1:], methods.items()):
        rgb = np.stack([base] * 3, -1)
        p, t = pred[z], truth[z]
        rgb[p & ~t] = [1.00, 0.12, 0.20]
        rgb[t & ~p] = [0.00, 0.78, 0.95]
        rgb[p & t] = [1.00, 0.90, 0.10]
        ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
        m = res["over_region"][name]
        ax.set_title(f"{name}\nDice {m['dice']:.3f}  IoU {m['iou']:.3f}  "
                     f"recall {m['recall']:.3f}  prec {m['precision']:.3f}", fontsize=9)
        ax.legend(handles=[Patch(facecolor="#ff1f33", label="method only (FP)"),
                           Patch(facecolor="#00c7f2", label="truth only (FN)"),
                           Patch(facecolor="#ffe61a", label="both (TP)")],
                  loc="upper right", fontsize=7, framealpha=0.85)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"{res['volume_description']} — region {res['region_size_zyx']} "
                 f"at {res['region_offset_zyx']} vs hand-made ground truth", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_dir, f"score{tag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sauvola-radius", type=int, default=SAUVOLA_RADIUS,
                   help=f"onlypores Sauvola window (production default {SAUVOLA_RADIUS})")
    p.add_argument("--sauvola-k", type=float, default=SAUVOLA_K,
                   help=f"onlypores Sauvola sensitivity (production default {SAUVOLA_K})")
    p.add_argument("--material-approach", default="otsu",
                   choices=["otsu", "li", "yen", "triangle", "fixed_t_best",
                            "fixed_t_cons", "mode_minus_24"],
                   help="global threshold used by the material-mask step")
    p.add_argument("--material-component", default="first", choices=["first", "largest"],
                   help="which max-projection component gives the material bounding box "
                        "(production uses 'first')")
    p.add_argument("--min-size-filtering", type=int, default=-1,
                   help="minimum pore size in voxels; <= 0 disables (production default)")
    p.add_argument("--detector-threshold", type=int, default=T_BEST,
                   help=f"audit dark-voxel detector threshold (default {T_BEST})")
    p.add_argument("--detector-min-cc", type=int, default=MIN_CC,
                   help=f"audit detector minimum component size (default {MIN_CC})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="cut out a region to annotate")
    e.add_argument("--volume", required=True,
                   help="generated volume directory (with volume.tif + mask.tif) "
                        "or a volume id in data/split_v2/volumes.zarr")
    e.add_argument("--offset", help="z,y,x origin of the region (default 0,0,0)")
    e.add_argument("--size", help="dz,dy,dx of the region (default: whole volume)")
    e.add_argument("--name", help="folder name under runs/campaigns/04-measurement-limits/onlypores_inspection/ground_truth/")
    add_common(e)
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("score", help="score the masks against your ground truth")
    s.add_argument("--truth", required=True, help="your ground-truth TIFF (non-zero = pore)")
    s.add_argument("--volume", help="overrides the volume recorded in region_meta.json")
    s.add_argument("--offset", help="z,y,x origin; overrides region_meta.json")
    s.add_argument("--tag", help="suffix for score<tag>.json / score<tag>.png")
    add_common(s)
    s.set_defaults(func=cmd_score)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

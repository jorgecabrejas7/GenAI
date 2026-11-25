#!/usr/bin/env python
"""Campaign 41 (paper review W2): does the generated label agree with the
generated grey image?

Reviewer W2 / Q1 (2026-11-24): "Run the Sauvola pipeline of Section III-B on
the decoded grey volumes. Report Dice and porosity agreement between the
decoded label and the re-segmented label, per porosity level, alongside the
same statistic for autoencoder reconstructions of real patches."

Re-segmentation (the split_v4 label rule, reference ``onlypores`` at
UTvsXCT-preprocessing 65c1eea, radius 30, k 0.125, min size 8):

    air      = the volume's own label == 2          (not re-decided)
    material = not air
    pore     = Sauvola(grey, window 30, k 0.125) says "not material", inside
               material, then ``clean_pores(min_size=8)``

The material mask is supplied, not derived with the reference's global Otsu:
Otsu needs a whole scan and misfires on a crop (relabel_synthetic_onlypores.py
records it: 0 pore on a 6 % crop). The same rule is applied to every set, so
the sets are comparable to each other; set ``real`` is the procedure's own
floor (the reference labels were made from whole scans with Otsu material).

Sets:
    real      real coupons (whole, cut to the specimen) with their dataset label
    vae_recon the same kind of real material encoded and decoded by the
              production VAE with its TRUE mask planes (``recon`` step, GPU)
    generated campaign-18 B microstructure volumes and campaign-36 a2s twins

Steps:
    python scripts/analysis/c41_label_agreement.py recon --out <dir>    (GPU)
        real_floor "large" crops of campaign 18 B -> <dir>/vae_recon/<case>/{volume,label}.tif
    python scripts/analysis/c41_label_agreement.py agree --out <dir> name=glob ...   (CPU)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
C18B = REPO / "runs/campaigns/18-eval-v4-final-v4-run0005-a2s"
SAUVOLA_R, SAUVOLA_K, MIN_SIZE = 30, 0.125, 8


def resegment(grey: np.ndarray, label: np.ndarray) -> np.ndarray:
    from preprocess_tools.onlypores import clean_pores, sauvola_thresholding_nonconcurrent
    air = label == 2
    binary = sauvola_thresholding_nonconcurrent(grey, window_size=SAUVOLA_R, k=SAUVOLA_K)
    pore = (~binary) & (~air)
    if pore.any():
        pore = clean_pores(pore, min_size=MIN_SIZE)
    return pore


def grey_of(label_f: Path) -> Path:
    n = label_f.name
    return label_f.with_name("volume.tif" if n == "label.tif" else n.replace("_label.tif", "_volume.tif"))


def level_of(path: str) -> str:
    m = re.search(r"phi(0\.\d+)", path)
    return m.group(1) if m else ""


def cmd_agree(a) -> int:
    rows = []
    out = a.out / "agreement_per_volume.csv"
    for spec in a.sets:
        name, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        print(f"{name}: {len(files)} volumes", flush=True)
        for f in files:
            f = Path(f)
            lab = tifffile.imread(f)
            grey = tifffile.imread(grey_of(f))
            dec = lab == 1
            seg = resegment(grey, lab)
            spec_n = max(int((lab != 2).sum()), 1)
            inter, n = int((dec & seg).sum()), int(dec.sum() + seg.sum())
            rows.append({"set": name, "volume": str(f), "case": f.parent.name, "level": level_of(str(f)),
                         "shape": list(lab.shape),
                         "porosity_label": float(dec.sum() / spec_n),
                         "porosity_resegmented": float(seg.sum() / spec_n),
                         "pore_dice": (2 * inter / n) if n else float("nan")})
            r = rows[-1]
            print(f"  {r['case']}: phi label {r['porosity_label']:.5f} reseg {r['porosity_resegmented']:.5f} "
                  f"Dice {r['pore_dice']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            del lab, grey, dec, seg
    d = pd.DataFrame(rows)
    d["porosity_diff"] = d.porosity_resegmented - d.porosity_label
    g = d.groupby(["set", "level"])[["pore_dice", "porosity_label", "porosity_resegmented", "porosity_diff"]]
    summ = g.agg(["mean", "std", "count"]).round(5)
    (a.out / "agreement_summary.txt").write_text(
        "Campaign 41 - label vs Sauvola re-segmentation of the grey (split_v4 rule r30 k0.125 min8, "
        "material = not air of the volume's own label)\n\n" + summ.to_string() + "\n")
    print(summ.to_string())
    return 0


def cmd_recon(a) -> int:
    import torch
    from vae_segment_volume import P, load_vae, one_pass
    dev = torch.device("cuda")
    run = Path(glob.glob(str(REPO / "runs/vae/r08-run-0020-2026*"))[0])
    ckpt = Path(glob.glob(str(run / "*_step00025000.ckpt"))[0]).name
    model = load_vae(run, ckpt, dev)
    cases = sorted(glob.glob(str(C18B / "real_floor/volumes/*__large")))
    print(f"{len(cases)} real crops, VAE {run.name} {ckpt}", flush=True)
    for c in cases:
        c = Path(c)
        dst = a.out / "vae_recon" / c.name
        if (dst / "label.tif").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        grey = tifffile.imread(c / "volume.tif")
        lab = tifffile.imread(c / "label.tif")
        pad = [(0, (-s) % P) for s in grey.shape]
        g = np.pad(grey, pad).astype(np.float32) / 255.0
        lp = np.pad(lab, pad, constant_values=2)
        logits, rgrey = one_pass(model, g, lp, dev)
        cut = tuple(slice(0, s) for s in grey.shape)
        tifffile.imwrite(dst / "volume.tif", (rgrey[cut] * 255).round().astype(np.uint8))
        tifffile.imwrite(dst / "label.tif", logits.argmax(0)[cut].astype(np.uint8))
        (dst / "source.json").write_text(json.dumps({"real_crop": str(c), "vae_run": run.name, "ckpt": ckpt,
                                                    "mask_planes": "true labels"}, indent=1))
        print("  recon", c.name, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recon"); r.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("agree"); g.add_argument("--out", type=Path, required=True)
    g.add_argument("sets", nargs="+", help="name=glob of label files")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    return cmd_recon(a) if a.cmd == "recon" else cmd_agree(a)


if __name__ == "__main__":
    raise SystemExit(main())

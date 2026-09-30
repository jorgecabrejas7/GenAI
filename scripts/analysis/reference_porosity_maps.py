#!/usr/bin/env python
"""Porosity maps and whole-volume VVF per specimen with the REFERENCE onlypores pipeline
(UTvsXCT-preprocessing), run exactly as its notebooks do, for two threshold configurations. (CPU, heavy.)

Per raw TIFF: io.load_tif -> reslicer.rotate_90(v, False) -> reslicer.reslice(., 'Right') -> aligner.crop_walls
(front/back wall slices) -> onlypores.onlypores(volume, fw, bw, sauvola_radius, sauvola_k, min_size_filtering)
for each configuration:
    ipynb : radius 30, k 0.125, min_size 8   (produccion/onlypores/onlypores.ipynb cell 6)
    batch : radius 15, k 0.2,   min_size 8   (produccion/onlypores/onlypores_batch.ipynb cell 5)
Outputs per specimen and configuration in <out>/<config>/: <specimen>_vvf_map.tif (float32 column porosity
over (y, x): pores / sample voxels along z), <specimen>_vvf_map.png, <specimen>_vvf_cells64.csv (64x64 cell map),
and one vvf.csv per configuration with the whole-volume VVF = pores / sample_mask (the notebook's number).
The reference package needs > 40 GB for one coupon; run one coupon at a time in a memory-capped scope.

    python scripts/analysis/reference_porosity_maps.py --pattern "Na_04_" --out runs/campaigns/30-reference-onlypores/maps
"""
from __future__ import annotations
import argparse, csv, re, time, gc
from pathlib import Path

import numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The installed, commit-pinned package. (Campaign 30's maps ran at GenAI cce9000
# from the clone at UTvsXCT-preprocessing 5d9da5b; the pin is bit-identical to it.)
from preprocess_tools import onlypores as ref_onlypores, aligner, reslicer, io as ref_io  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {"ipynb": dict(sauvola_radius=30, sauvola_k=0.125, min_size_filtering=8),
           "batch": dict(sauvola_radius=15, sauvola_k=0.2, min_size_filtering=8)}


def specimen(name):
    m = re.search(r"((?:Na|JI|Pegaso)_\d+(?:_\d+)?)_volume", name); return m.group(1) if m else name


def cell_map(pores2d_sum, sample2d_sum, cell=64):
    Y, X = pores2d_sum.shape; out = np.full(((Y + cell - 1) // cell, (X + cell - 1) // cell), np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            p = pores2d_sum[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].sum(); s = sample2d_sum[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].sum()
            if s > 0.1 * cell * cell * 64: out[i, j] = p / s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "raw_data/MedidasDB")); ap.add_argument("--pattern", default="Na_")
    ap.add_argument("--configs", default="ipynb,batch"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); configs = a.configs.split(",")
    files = sorted(p for p in Path(a.raw).glob("*.tif") if a.pattern in p.name)
    for cfg in configs: (out / cfg).mkdir(parents=True, exist_ok=True)
    for p in files:
        name = specimen(p.name)
        if all((out / cfg / f"{name}_vvf_map.tif").exists() for cfg in configs):
            continue
        t0 = time.time(); volume = ref_io.load_tif(str(p))
        resliced = reslicer.reslice(reslicer.rotate_90(volume, False), "Right")
        _, fw, bw = aligner.crop_walls(resliced); del resliced; gc.collect()
        print(f"{name}: shape {volume.shape} walls {fw}/{bw}", flush=True)
        for cfg in configs:
            if (out / cfg / f"{name}_vvf_map.tif").exists():
                continue
            pores, sample, _binary = ref_onlypores.onlypores(volume, fw, bw, **CONFIGS[cfg])
            pores = pores.astype(bool); sample = sample.astype(bool)
            n_p, n_s = int(pores.sum()), int(sample.sum()); vvf = n_p / n_s if n_s else float("nan")
            ps, ss = pores.sum(0).astype(np.float32), sample.sum(0).astype(np.float32)
            vmap = np.where(ss > 0, ps / np.maximum(ss, 1), np.nan).astype(np.float32)
            tifffile.imwrite(out / cfg / f"{name}_vvf_map.tif", vmap)
            cm = cell_map(ps, ss)
            np.savetxt(out / cfg / f"{name}_vvf_cells64.csv", 100 * cm, delimiter=",", fmt="%.4f")
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            im = ax[0].imshow(100 * vmap, cmap="inferno", vmin=0, vmax=np.nanpercentile(100 * vmap, 99)); ax[0].set_title(f"{name}  column VVF (%)  whole-volume VVF {100*vvf:.3f} %  [{cfg}]"); plt.colorbar(im, ax=ax[0], fraction=0.03)
            im2 = ax[1].imshow(100 * cm, cmap="inferno", vmin=0, vmax=np.nanmax(100 * cm)); ax[1].set_title("64x64 cell VVF (%)"); plt.colorbar(im2, ax=ax[1], fraction=0.03)
            for x in ax: x.set_xticks([]); x.set_yticks([])
            fig.tight_layout(); fig.savefig(out / cfg / f"{name}_vvf_map.png", dpi=110); plt.close(fig)
            row = {"specimen": name, "config": cfg, **CONFIGS[cfg], "frontwall": fw, "backwall": bw, "shape": "x".join(map(str, volume.shape)),
                   "pore_voxels": n_p, "sample_voxels": n_s, "vvf_percent": f"{100*vvf:.4f}"}
            csv_path = out / cfg / "vvf.csv"; new = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row)); (w.writeheader() if new else None); w.writerow(row)
            print(f"  {cfg}: VVF {100*vvf:.4f} %  ({time.time()-t0:.0f} s)", flush=True)
            del pores, sample, _binary, ps, ss, vmap; gc.collect()
        del volume; gc.collect()
    print("done", out)


if __name__ == "__main__":
    main()

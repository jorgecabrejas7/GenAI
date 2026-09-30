#!/usr/bin/env python
"""Porosity maps and whole-volume VVF per specimen from the reference onlypores outputs in raw_data. (CPU, minutes.)

Reads the masks that ``scripts/build_reference_onlypores.py`` stored beside the raw volumes
(``raw_data/MedidasDB/onlypores files/<stem>_{onlypores,samplemask}_<params>.tif``) and computes, per parameter set:

    <out>/<config>/<specimen>_vvf_map.tif      float32 column porosity over (y, x): pore voxels / sample voxels along z
    <out>/<config>/<specimen>_vvf_map.png      the column map and a 64x64 cell map
    <out>/<config>/<specimen>_vvf_cells64.csv  the 64x64 cell map in percent
    <out>/<config>/vvf.csv                     whole-volume VVF = pore voxels / sample voxels (the notebook's number)
    <out>/vvf_per_sample.csv                   one row per specimen, both parameter sets side by side

Nothing is segmented here; the numbers are those of the stored masks (and of the notebook reports next to them).

    python scripts/analysis/reference_porosity_maps.py --out runs/campaigns/30-reference-onlypores/maps
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from build_reference_onlypores import CONFIGS, RAW_ROOT, output_paths  # noqa: E402


def specimen(name: str) -> str:
    m = re.search(r"((?:Na|JI|Pegaso)_\d+(?:_\d+)?)_volume", name)
    return m.group(1) if m else name


def cell_map(pores2d_sum, sample2d_sum, cell=64):
    Y, X = pores2d_sum.shape
    out = np.full(((Y + cell - 1) // cell, (X + cell - 1) // cell), np.nan)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            p = pores2d_sum[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].sum()
            s = sample2d_sum[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].sum()
            if s > 0.1 * cell * cell * 64:
                out[i, j] = p / s
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_ROOT)
    ap.add_argument("--pattern", default="")
    ap.add_argument("--configs", default="ipynb,batch")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    configs = a.configs.split(",")
    files = sorted(p for p in a.raw.glob("*.tif") if a.pattern in p.name)
    rows = {cfg: [] for cfg in configs}
    for p in files:
        name = specimen(p.name)
        for cfg in configs:
            paths = output_paths(p, CONFIGS[cfg])
            if not (paths["onlypores"].exists() and paths["samplemask"].exists()):
                continue
            d = a.out / cfg
            d.mkdir(parents=True, exist_ok=True)
            pores = tifffile.imread(paths["onlypores"]) > 0
            sample = tifffile.imread(paths["samplemask"]) > 0
            n_p, n_s = int(pores.sum()), int(sample.sum())
            vvf = n_p / n_s if n_s else float("nan")
            ps, ss = pores.sum(0).astype(np.float32), sample.sum(0).astype(np.float32)
            del pores, sample
            vmap = np.where(ss > 0, ps / np.maximum(ss, 1), np.nan).astype(np.float32)
            tifffile.imwrite(d / f"{name}_vvf_map.tif", vmap)
            cm = cell_map(ps, ss)
            np.savetxt(d / f"{name}_vvf_cells64.csv", 100 * cm, delimiter=",", fmt="%.4f")
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            im = ax[0].imshow(100 * vmap, cmap="inferno", vmin=0, vmax=np.nanpercentile(100 * vmap, 99))
            ax[0].set_title(f"{name}  column VVF (%)  whole-volume VVF {100*vvf:.3f} %  [{cfg}]")
            plt.colorbar(im, ax=ax[0], fraction=0.03)
            im2 = ax[1].imshow(100 * cm, cmap="inferno", vmin=0, vmax=np.nanmax(100 * cm))
            ax[1].set_title("64x64 cell VVF (%)")
            plt.colorbar(im2, ax=ax[1], fraction=0.03)
            for x in ax:
                x.set_xticks([]); x.set_yticks([])
            fig.tight_layout(); fig.savefig(d / f"{name}_vvf_map.png", dpi=110); plt.close(fig)
            rows[cfg].append({"specimen": name, "file": p.name, "config": cfg, **CONFIGS[cfg],
                              "pore_voxels": n_p, "sample_voxels": n_s, "vvf_percent": f"{100*vvf:.4f}"})
            print(f"{name} {cfg}: VVF {100*vvf:.4f} %", flush=True)
    for cfg in configs:
        if rows[cfg]:
            with open(a.out / cfg / "vvf.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[cfg][0])); w.writeheader(); w.writerows(rows[cfg])
    # One row per specimen, both parameter sets side by side.
    by = {}
    for cfg in configs:
        for r in rows[cfg]:
            by.setdefault(r["file"], {"specimen": r["specimen"], "file": r["file"]})
            by[r["file"]][f"pore_voxels_{cfg}"] = r["pore_voxels"]
            by[r["file"]][f"sample_voxels_{cfg}"] = r["sample_voxels"]
            by[r["file"]][f"vvf_percent_{cfg}"] = r["vvf_percent"]
    if by:
        fields = ["specimen", "file"] + [f"{k}_{cfg}" for cfg in configs
                                         for k in ("pore_voxels", "sample_voxels", "vvf_percent")]
        with open(a.out / "vvf_per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader()
            w.writerows(by[k] for k in sorted(by))
    print("done", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

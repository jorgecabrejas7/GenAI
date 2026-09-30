#!/usr/bin/env python
"""Run the reference onlypores notebooks over raw_data and store their outputs beside the volumes. (CPU, heavy.)

This is produccion/onlypores/onlypores_batch.ipynb of UTvsXCT-preprocessing (the installed, commit-pinned
``preprocess_tools``), executed as written, for the two parameter sets the notebooks use:

    ipynb : sauvola_radius 30, sauvola_k 0.125, min_size_filtering 8   (onlypores.ipynb cell 6)
    batch : sauvola_radius 15, sauvola_k 0.2,   min_size_filtering 8   (onlypores_batch.ipynb cell 5)

and a third set that is exactly what PoreGen's old ``compute_mask`` did (GenAI cce9000
``src/poregen/dataset/io.py``: ``onlypores(xct)``, the function defaults, no wall detection), so the
split_v1-v3 labels can be rebuilt from raw_data:

    v3    : sauvola_radius 30, sauvola_k 0.125, min_size_filtering -1, no walls   (tag ..._min-1_nowalls)

The sets, tags and file names come from ``poregen.dataset.io`` (``SEGMENTATION``, ``NO_WALLS``,
``reference_outputs``), the same code ``scripts/build_split_v4.py`` reads them with.

Per raw TIFF: io.load_tif -> reslicer.rotate_90(v, False) -> reslicer.reslice(., 'Right') -> aligner.crop_walls
-> onlypores.onlypores(volume, frontwall, backwall, sauvola_radius, sauvola_k, min_size_filtering). Nothing is
added, wrapped or corrected: the size rule, the material mask, everything is what the reference package does at
its pinned scikit-image 0.26.0 (a component of exactly min_size voxels is removed, as in the notebooks' saved output).

Outputs go where the notebooks put them, ``<volume dir>/onlypores files/``, with the same suffixes and the
parameters appended to the file name so both parameter sets coexist:

    <stem>_onlypores_r30_k0.125_min8.tif    binary pore mask     (255 = pore)
    <stem>_samplemask_r30_k0.125_min8.tif   sample envelope      (255 = material)
    <stem>_binary_r30_k0.125_min8.tif       raw Sauvola result   (255 = material)
    <stem>_report_r30_k0.125_min8.txt       the notebook's processing report (walls, parameters, voxel counts, VVF)

The TIFFs are written deflate-compressed (same voxels, same dtype and axes as the notebooks' uncompressed files;
``preprocess_tools.io.load_tif`` reads both). Volumes with all four files present are skipped, so the script resumes.

    python scripts/build_reference_onlypores.py --config ipynb,batch --shard 0/2
"""
from __future__ import annotations

import argparse
import gc
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from preprocess_tools import aligner, io, onlypores as onlypores_module, reslicer

from poregen.dataset.io import NO_WALLS, SEGMENTATION as CONFIGS, reference_outputs

REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO / "raw_data" / "MedidasDB"


def output_paths(volume_path: Path, cfg: str) -> dict[str, Path]:
    return reference_outputs(volume_path, cfg)


def outputs_exist(volume_path: Path, cfg: str) -> bool:
    return all(p.exists() for p in output_paths(volume_path, cfg).values())


def save_tif(path: Path, volume: np.ndarray) -> None:
    tifffile.imwrite(path, volume, compression="zlib", compressionargs={"level": 6})


def reference_commit() -> str:
    try:
        from importlib.metadata import distribution
        direct = distribution("preprocess_tools").read_text("direct_url.json") or ""
        return direct
    except Exception:  # noqa: BLE001
        return "unknown"


def process_volume(volume_path: Path, configs: list[str]) -> None:
    """The notebook's process_volume, once per parameter set, walls computed once (they do not depend on it)."""
    todo = [c for c in configs if not outputs_exist(volume_path, c)]
    if not todo:
        return
    print(f"\n{'=' * 80}\nProcessing: {volume_path}\n{'=' * 80}", flush=True)
    t0 = time.time()
    volume = io.load_tif(str(volume_path))
    print(f"Loaded volume shape: {volume.shape}")

    if any(c not in NO_WALLS for c in todo):
        resliced_volume = reslicer.rotate_90(volume, False)
        resliced_volume = reslicer.reslice(resliced_volume, 'Right')
        resliced_shape = resliced_volume.shape
        print(f"Resliced volume shape: {resliced_shape}")

        _, frontwall, backwall = aligner.crop_walls(resliced_volume)
        del resliced_volume
        gc.collect()
        print(f"Front wall: {frontwall}, back wall: {backwall}")

    for cfg in todo:
        params = CONFIGS[cfg]
        sauvola_radius, sauvola_k, min_size_filtering = (params["sauvola_radius"], params["sauvola_k"],
                                                         params["min_size_filtering"])
        if cfg in NO_WALLS:
            # Exactly the old PoreGen call (GenAI cce9000 io.compute_mask): onlypores(xct), the defaults.
            pores, sample_mask, binary = onlypores_module.onlypores(volume)
        else:
            pores, sample_mask, binary = onlypores_module.onlypores(
                volume, frontwall, backwall,
                sauvola_radius=sauvola_radius, sauvola_k=sauvola_k,
                min_size_filtering=min_size_filtering
            )
        out = output_paths(volume_path, cfg)
        out["onlypores"].parent.mkdir(parents=True, exist_ok=True)
        save_tif(out["onlypores"], pores.astype(np.uint8) * 255)
        save_tif(out["samplemask"], sample_mask.astype(np.uint8) * 255)
        save_tif(out["binary"], binary.astype(np.uint8) * 255)

        total_voxels = int(np.prod(volume.shape))
        pore_voxels = int(np.sum(pores))
        material_voxels = int(np.sum(sample_mask))
        vvf = pore_voxels / material_voxels if material_voxels > 0 else float('nan')

        if cfg in NO_WALLS:
            walls_text = (
                "Wall detection: not run. onlypores is called as PoreGen's compute_mask called it until\n"
                "GenAI cce9000, onlypores(xct), so frontwall and backwall keep their defaults.\n"
                "Front wall slice: 0\nBack wall slice: 0\n(No slices are excluded.)")
            cleaning_text = f"Not applied (min_size_filtering = {min_size_filtering})."
        else:
            walls_text = (
                f"Front wall slice: {frontwall}\nBack wall slice: {backwall}\n"
                f"(Slices [0:{frontwall}] and [{backwall}:end] are excluded from pore detection as they "
                "belong to the sample walls.)")
            cleaning_text = (
                "Applied because min_size_filtering > 0.\n"
                f"   - min_size: {min_size_filtering} voxels (3D connected components, 26-connectivity, "
                "below this are discarded as noise)\n"
                "   - Dimensional filter: components must span >= 2 voxels in each of Z, Y, X "
                "(removes flat/linear artifacts)")
        step3_text = ("No wall slices are forced to material (frontwall = backwall = 0)." if cfg in NO_WALLS else
                      f"Front wall slices [0:{frontwall}] and back wall slices [{backwall}:end] are forced to "
                      "material (True) to exclude them from pore detection.")
        report = f"""Pore detection processing report
Generated: {datetime.now().isoformat(timespec='seconds')}
Reference package: preprocess_tools (UTvsXCT-preprocessing), installed from {reference_commit().strip()}
Parameter set: {cfg}

Input volume
------------
Path: {volume_path}
Original shape (Z, Y, X): {volume.shape}

Reslicing
---------
1. rotate_90(volume, clockwise=False)
2. reslice(rotated_volume, 'Right')
Resliced shape: {resliced_shape if cfg not in NO_WALLS else 'not resliced'}

Wall detection (aligner.crop_walls)
------------------------------------
{walls_text}

Pore detection (onlypores.onlypores)
-------------------------------------
1. Volume is cropped to its non-zero bounding box (+2 voxel margin) for efficiency.
2. Sauvola adaptive thresholding is applied slice-by-slice (per Y slice, over the Z-X plane):
   - window_size (sauvola_radius): {sauvola_radius}
   - k: {sauvola_k}
   - r (dynamic range): 128 (fixed)
   - Voxels above the local adaptive threshold are classified as material; below as pore/background.
3. {step3_text}
4. A material (sample) mask is generated independently via:
   - Global Otsu thresholding on the cropped volume.
   - Maximum-intensity projection along Z to find the sample's 2D footprint (largest connected component).
   - Internal voids filled (fill_voids) within that footprint, across all Z slices, to obtain a solid sample envelope.
5. Pores = (NOT binary) AND sample_mask, i.e. voxels below the Sauvola threshold that fall inside the solid sample envelope.

Pore cleaning (onlypores.clean_pores)
----------------------------------------
{cleaning_text}

Results
-------
Total volume voxels: {total_voxels}
Material (sample_mask) voxels: {material_voxels}
Pore voxels (after cleaning): {pore_voxels}
Void volumetric fraction (pore voxels / material voxels): {vvf:.6f} ({vvf*100:.4f}%)

Output files (in this folder)
------------------------------
{out['onlypores'].name}   - binary pore mask (255 = pore, 0 = not pore)
{out['samplemask'].name}  - binary sample/material envelope mask (255 = material, 0 = background)
{out['binary'].name}      - raw Sauvola thresholding result before pore/mask combination (255 = material, 0 = background)
"""
        out["report"].write_text(report)
        print(f"  {cfg}: VVF {100 * vvf:.4f} %  ({time.time() - t0:.0f} s)", flush=True)
        del pores, sample_mask, binary
        gc.collect()
    del volume
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_ROOT)
    ap.add_argument("--config", default="ipynb,batch", help="comma-separated parameter sets")
    ap.add_argument("--pattern", default="", help="only volumes whose file name contains this")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th volume starting at i (parallel workers)")
    a = ap.parse_args()
    configs = a.config.split(",")
    i, n = map(int, a.shard.split("/"))
    files = sorted(p for p in a.raw.glob("*.tif") if a.pattern in p.name)[i::n]
    for p in files:
        process_volume(p, configs)
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

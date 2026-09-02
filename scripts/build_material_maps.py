"""Build the ldm06 material maps + air fractions for an EXISTING latent store.

D40 §1: every training patch gets a material map (where the specimen's
material envelope is, from the onlypores ``sample_mask``) at latent
resolution, plus a scalar air fraction.  Nothing is re-encoded: the existing
``latents.bin`` / ``index.parquet`` / ``cond.parquet`` stay byte-identical.

Per split it writes two NEW sibling files next to ``latents.bin`` (D34
binary-store convention; see ``poregen.dataset.material`` for the format
rationale — uint8 fractions, half the size of float16):

    <store>/<split>/material.bin   uint8   (N, L, L, L)  value/255 = material fraction per cell
    <store>/<split>/air.bin        float32 (N,)          1 - sample_mask.mean() over the patch

Row i of both files is row i of that split's ``index.parquet``.

Per volume, the ``sample_mask`` itself is persisted into
``volumes.zarr/<volume_id>/sample_mask`` (compressed; beside the xct/mask
arrays, which is where per-volume masks already live).  On later runs it is
read back instead of recomputed.

The run is resumable: ``material_progress.json`` in the output root records
finished (split, volume) pairs, and each volume's rows are written into the
pre-sized memmaps at their final positions.  When every volume of every
requested split is done AND the output root is the store itself, a
``material`` block is added to the store's ``metadata.json`` (additive).

Usage
-----
python scripts/build_material_maps.py                       # full migration, live store
python scripts/build_material_maps.py \\
    --volumes Na_03_5 --output-root /path/to/scratch_store  # single-volume smoke test
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_STORE = REPO / "data" / "split_v2" / "latents_r07z4"
DEFAULT_DATA_ROOT = REPO / "data" / "split_v2"

MATERIAL_FILE = "material.bin"
AIR_FILE = "air.bin"
PROGRESS_FILE = "material_progress.json"


def _load_progress(path: Path) -> dict[str, list[str]]:
    if path.exists():
        return json.load(open(path))
    return {}


def _save_progress(path: Path, progress: dict[str, list[str]]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=1))
    tmp.replace(path)


def _open_memmaps(split_dir: Path, n: int, latent_size: int):
    """Create-or-open the two row-aligned sibling memmaps for one split."""
    shape_m = (n, latent_size, latent_size, latent_size)
    mat_path = split_dir / MATERIAL_FILE
    air_path = split_dir / AIR_FILE
    for path, dtype, shape in ((mat_path, np.uint8, shape_m), (air_path, np.float32, (n,))):
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if path.exists() and path.stat().st_size != expected:
            raise SystemExit(
                f"{path} exists with {path.stat().st_size} bytes, expected {expected} — "
                f"the store changed; delete the file (and {PROGRESS_FILE}) to rebuild."
            )
    mode_m = "r+" if mat_path.exists() else "w+"
    mode_a = "r+" if air_path.exists() else "w+"
    material = np.memmap(str(mat_path), dtype=np.uint8, mode=mode_m, shape=shape_m)
    air = np.memmap(str(air_path), dtype=np.float32, mode=mode_a, shape=(n,))
    return material, air


def _get_sample_mask(zroot, volume_id: str, data_root: Path, write: bool) -> np.ndarray:
    """Volume sample_mask: read the persisted zarr array, else compute + persist."""
    from poregen.dataset.io import save_sample_mask_zarr
    from poregen.dataset.segmentation import compute_sample_mask

    grp = zroot[volume_id]
    if "sample_mask" in grp:
        logger.info("  sample_mask found in volumes.zarr — reading")
        return np.asarray(grp["sample_mask"][:]).astype(bool)

    logger.info("  computing sample_mask from xct (Otsu + fill-voids, no Sauvola)")
    xct = np.asarray(grp["xct"][:])
    sm = compute_sample_mask(xct)
    if sm is None:
        raise RuntimeError(f"{volume_id}: empty volume — cannot compute sample_mask")
    if write:
        save_sample_mask_zarr(sm, data_root, volume_id)
    return sm


def _split_stats(air: np.ndarray, threshold: float = 0.999) -> dict:
    a = np.asarray(air, dtype=np.float64)
    return {
        "n": int(a.size),
        "air_fraction_mean": float(a.mean()),
        "frac_all_air": float((a >= threshold).mean()),
        "frac_no_air": float((a <= 1e-6).mean()),
        "all_air_threshold": threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=str(DEFAULT_STORE),
                    help="Latent store root (metadata.json + split dirs)")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="Directory holding volumes.zarr")
    ap.add_argument("--output-root", default=None,
                    help="Redirect the NEW files (material.bin, air.bin, progress) to a "
                         "sibling directory — for smoke tests. Default: the store itself.")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Splits to process (default: every split dir in the store)")
    ap.add_argument("--volumes", nargs="+", default=None,
                    help="Only volumes whose id contains one of these substrings (smoke test)")
    ap.add_argument("--no-write-sample-mask", action="store_true",
                    help="Do not persist computed sample_masks into volumes.zarr")
    args = ap.parse_args()

    store = Path(args.store).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else store
    out_root.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(store / "metadata.json"))
    latent_size = int(meta["latent_shape"][-1])
    patch_size = int(meta.get("patch_size", 64))
    factor = patch_size // latent_size
    if factor * latent_size != patch_size:
        raise SystemExit(f"patch_size {patch_size} is not a multiple of latent size {latent_size}")

    splits = args.splits or sorted(
        d.name for d in store.iterdir() if d.is_dir() and (d / "index.parquet").exists()
    )
    logger.info("Store: %s  splits=%s  latent_size=%d  factor=%d  output=%s",
                store, splits, latent_size, factor, out_root)

    from poregen.dataset.material import (
        encode_material_u8,
        patch_material_cells,
        pool_material_fractions,
    )

    zmode = "r" if args.no_write_sample_mask else "a"
    zroot = zarr.open_group(str(data_root / "volumes.zarr"), mode=zmode)

    progress_path = out_root / PROGRESS_FILE
    progress = _load_progress(progress_path)

    all_done = True
    for split in splits:
        df = pd.read_parquet(store / split / "index.parquet",
                             columns=["volume_id", "z0", "y0", "x0"])
        n = len(df)
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        material, air = _open_memmaps(split_out, n, latent_size)

        done = set(progress.get(split, []))
        vol_ids = df["volume_id"].unique().tolist()
        selected = [
            v for v in vol_ids
            if args.volumes is None or any(s in v for s in args.volumes)
        ]
        logger.info("[%s] %d rows, %d volumes (%d selected, %d already done)",
                    split, n, len(vol_ids), len(selected), len(done & set(selected)))

        for vid in selected:
            if vid in done:
                continue
            t0 = time.time()
            logger.info("[%s] %s", split, vid)
            sm = _get_sample_mask(zroot, vid, data_root, write=not args.no_write_sample_mask)
            pooled = pool_material_fractions(sm, factor)
            del sm

            rows = np.flatnonzero((df["volume_id"] == vid).to_numpy())
            z0 = df["z0"].to_numpy(np.int64)[rows]
            y0 = df["y0"].to_numpy(np.int64)[rows]
            x0 = df["x0"].to_numpy(np.int64)[rows]
            for r, z, y, x in zip(rows, z0, y0, x0):
                cells = patch_material_cells(pooled, int(z), int(y), int(x),
                                             factor, latent_size)
                material[r] = encode_material_u8(cells)
                # exact: mean of equal-size cell fractions == patch mean
                air[r] = np.float32(1.0 - cells.mean(dtype=np.float64))
            material.flush()
            air.flush()
            progress.setdefault(split, []).append(vid)
            _save_progress(progress_path, progress)
            logger.info("[%s] %s: %d patches in %.1fs  mean air=%.4f",
                        split, vid, len(rows), time.time() - t0,
                        float(air[rows].mean()))

        if set(vol_ids) - set(progress.get(split, [])):
            all_done = False
        del material, air

    if not all_done:
        logger.info("Some volumes remain — metadata not finalised (resumable run).")
        return
    if out_root != store:
        logger.info("Output root differs from the store — metadata not finalised (smoke run).")
        return

    # Finalise: additive `material` block in the store metadata.
    stats = {}
    for split in splits:
        n = len(pd.read_parquet(store / split / "index.parquet", columns=["z0"]))
        a = np.memmap(str(store / split / AIR_FILE), dtype=np.float32, mode="r", shape=(n,))
        stats[split] = _split_stats(a)
        del a
    meta_path = store / "metadata.json"
    meta = json.load(open(meta_path))
    meta["material"] = {
        "version": "ldm06-v1",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "builder": "scripts/build_material_maps.py",
        "files": {"material": MATERIAL_FILE, "air": AIR_FILE},
        "alignment": "row i of material.bin and air.bin is row i of index.parquet",
        "material_dtype": "uint8",
        "material_shape": [latent_size] * 3,
        "material_encoding": (
            f"value/255 = material fraction of the {factor}^3-voxel cell, "
            f"{factor}^3 block mean of the onlypores sample_mask"
        ),
        "air_dtype": "float32",
        "air_definition": "1 - sample_mask.mean() over the full-resolution patch",
        "sample_mask_source": "volumes.zarr/<volume_id>/sample_mask (compressed, additive)",
        "per_split_stats": stats,
    }
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)
    logger.info("Finalised: wrote `material` block into %s", meta_path)
    for split, st in stats.items():
        logger.info("  %s: n=%d  mean air=%.4f  all-air=%.2f%%",
                    split, st["n"], st["air_fraction_mean"], 100 * st["frac_all_air"])


if __name__ == "__main__":
    main()

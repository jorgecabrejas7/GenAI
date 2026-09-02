"""
extract_patches_memmap.py

One-time extraction: reads every patch listed in patch_index.parquet from the
Zarr volumes and writes them to two flat numpy memmaps.

Layout
------
    <data-root>/patches_xct.bin     (N, ps, ps, ps)  uint8   grey level 0-255
    <data-root>/patches_label.bin   (N, ps, ps, ps)  uint8   0/1/2
    <data-root>/patches_meta.json   metadata

Row i in both memmaps corresponds to row i in patch_index.parquet.
That 1-to-1 alignment is the sole invariant relied on by MemmapPatchDataset.

The label
---------
Three classes, in this precedence:

    2  air       ``sample_mask == 0`` — outside the specimen, or inside one of
                 the three drilled registration through-holes
    1  pore      ``mask != 0`` inside the specimen
    0  material

Air wins over pore, so a voxel can never be both.  In practice they do not
overlap: ``fill_voids`` leaves the holes empty in ``mask`` as well.  The binary
pore mask the VAE losses use is ``label == 1``.

Reconstruction with stride < patch_size
----------------------------------------
Because patch_index.parquet stores (volume_id, z0, y0, x0) for every row,
any patch can be placed back into its volume at any time:

    z0, y0, x0 = df.iloc[i][['z0','y0','x0']]
    patch = mmap_xct[i]          # (ps, ps, ps) uint8
    volume[z0:z0+ps, y0:y0+ps, x0:x0+ps] += patch   # or mean, or overwrite

Overlapping patches (stride=32, patch_size=64 → 50% overlap per axis) can be
averaged using float32 accumulation + count buffer — see
poregen.dataset.loader.reconstruct_volume().

Storage: ``2 x N x ps^3`` bytes.  For split_v3 that is ~0.5 TB per array; the
script refuses to start if the filesystem cannot hold both.

Usage
-----
    python scripts/extract_patches_memmap.py \\
        --data-root data/split_v3            \\
        [--chunk-size 256]                   \\
        [--force]                            \\
        [--verify]

Resumability
------------
Progress is tracked per volume in patches_progress.json.
Interrupt at any time; the script skips already-finished volumes on re-run.
Final files are written only when all volumes are done (atomic rename).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2
LABEL_NAMES = {LABEL_MATERIAL: "material", LABEL_PORE: "pore", LABEL_AIR: "air"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parquet_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def _open_partial(path: Path, shape: tuple, dtype: np.dtype) -> np.memmap:
    if path.exists():
        return np.memmap(str(path), dtype=dtype, mode="r+", shape=shape)
    return np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)


def build_label_volume(mask: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    """Three-class voxel label: 0 material, 1 pore, 2 air. Air wins."""
    label = np.zeros(mask.shape, dtype=np.uint8)
    label[mask != 0] = LABEL_PORE
    label[sample_mask == 0] = LABEL_AIR
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-extract VAE patches from Zarr into flat numpy memmaps. "
            "Row i in the output corresponds to row i in patch_index.parquet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, metavar="PATH",
                        help="Split root, e.g. data/split_v3")
    parser.add_argument("--chunk-size", type=int, default=256, metavar="INT",
                        help="Patches written per memmap flush (memory budget: "
                             "chunk_size × ps³ × 2 arrays × 1 byte).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing progress and restart from scratch.")
    parser.add_argument("--verify", action="store_true",
                        help="After extraction, compare 128 random patches "
                             "against zarr (seed 42).")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    zarr_root_path = data_root / "volumes.zarr"
    parquet_path   = data_root / "patch_index.parquet"
    index_report   = data_root / "index_report.json"

    for p in (zarr_root_path, parquet_path, index_report):
        if not p.exists():
            log.error("Not found: %s", p)
            raise SystemExit(1)
    report = json.loads(index_report.read_text())

    # ------------------------------------------------------------------
    # Load parquet (authoritative row order)
    # ------------------------------------------------------------------
    log.info("Loading %s …", parquet_path)
    df = pd.read_parquet(str(parquet_path))
    N  = len(df)
    ps = int(df["ps"].iloc[0])
    if df["ps"].nunique() != 1:
        log.error("Mixed patch sizes in parquet — not supported.")
        raise SystemExit(1)

    stride = int(df["stride"].iloc[0])
    shape = (N, ps, ps, ps)
    dtype = np.dtype("uint8")

    n_splits = {sp: int((df["split"] == sp).sum()) for sp in ("train", "val", "test")}
    bytes_per_arr = N * ps ** 3
    log.info(
        "Parquet: N=%d  patch_size=%d  stride=%d  splits=%s",
        N, ps, stride, n_splits,
    )
    log.info(
        "Output size: XCT %.1f GB  Label %.1f GB  Total %.1f GB",
        bytes_per_arr / 1e9, bytes_per_arr / 1e9, 2 * bytes_per_arr / 1e9,
    )

    free = shutil.disk_usage(data_root).free
    if free < 2 * bytes_per_arr:
        log.error("Need %.1f GB but only %.1f GB free on %s.",
                  2 * bytes_per_arr / 1e9, free / 1e9, data_root)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    xct_bin       = data_root / "patches_xct.bin"
    label_bin     = data_root / "patches_label.bin"
    xct_partial   = data_root / "patches_xct.bin.partial"
    label_partial = data_root / "patches_label.bin.partial"
    meta_path     = data_root / "patches_meta.json"
    meta_tmp      = data_root / "patches_meta.json.tmp"
    progress_path = data_root / "patches_progress.json"

    if xct_bin.exists() or label_bin.exists():
        if args.force:
            log.info("--force: removing existing output files.")
            for p in (xct_bin, label_bin, meta_path, progress_path):
                p.unlink(missing_ok=True)
        else:
            log.error(
                "Output already exists (%s). Use --force to overwrite.", xct_bin
            )
            raise SystemExit(1)

    if args.force:
        for p in (xct_partial, label_partial):
            p.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Progress tracking (per-volume granularity)
    # ------------------------------------------------------------------
    if progress_path.exists() and not args.force:
        with open(progress_path) as fh:
            progress: dict[str, bool] = json.load(fh)
        log.info("Resuming: %d / %d volumes already done.", sum(progress.values()), len(progress))
    else:
        progress = {}

    # ------------------------------------------------------------------
    # Open / create partial memmaps
    # ------------------------------------------------------------------
    mmap_xct   = _open_partial(xct_partial,   shape, dtype)
    mmap_label = _open_partial(label_partial, shape, dtype)

    zarr_root: zarr.Group = zarr.open_group(str(zarr_root_path), mode="r")

    # ------------------------------------------------------------------
    # Per-volume extraction (load full volume into RAM, then slice patches)
    # ------------------------------------------------------------------
    volume_ids = df["volume_id"].unique().tolist()
    class_counts = np.zeros(3, dtype=np.int64)
    volumes_done = 0
    t0 = time.perf_counter()

    with tqdm(total=N, unit="patches", desc="Extracting") as pbar:
        for vid in sorted(volume_ids):
            if progress.get(vid, False):
                n_vid = int((df["volume_id"] == vid).sum())
                pbar.update(n_vid)
                volumes_done += 1
                continue

            vol_mask = df["volume_id"] == vid
            vol_df   = df[vol_mask]
            row_indices = np.where(vol_mask.values)[0]   # global parquet indices

            # Load full volume into RAM (one big sequential zarr read)
            grp = zarr_root[vid]
            xct_vol   = np.asarray(grp["xct"], dtype=np.uint8)
            label_vol = build_label_volume(
                np.asarray(grp["mask"], dtype=np.uint8),
                np.asarray(grp["sample_mask"], dtype=np.uint8),
            )

            # Slice and write patches in chunks for bounded memory
            cs = args.chunk_size
            n_patches = len(row_indices)
            for chunk_start in range(0, n_patches, cs):
                chunk_end = min(chunk_start + cs, n_patches)
                for local_i in range(chunk_start, chunk_end):
                    row = vol_df.iloc[local_i]
                    z0, y0, x0 = int(row["z0"]), int(row["y0"]), int(row["x0"])
                    gi = row_indices[local_i]
                    patch = label_vol[z0:z0+ps, y0:y0+ps, x0:x0+ps]
                    mmap_xct  [gi] = xct_vol[z0:z0+ps, y0:y0+ps, x0:x0+ps]
                    mmap_label[gi] = patch
                    class_counts += np.bincount(patch.ravel(), minlength=3)

                mmap_xct.flush()
                mmap_label.flush()
                pbar.update(chunk_end - chunk_start)

            del xct_vol, label_vol

            volumes_done += 1
            progress[vid] = True
            with open(progress_path, "w") as fh:
                json.dump(progress, fh)

    elapsed = time.perf_counter() - t0
    del mmap_xct, mmap_label

    # ------------------------------------------------------------------
    # Atomic rename: partial → final
    # ------------------------------------------------------------------
    parquet_sha = _parquet_sha256(parquet_path)
    total_vox = int(class_counts.sum())
    meta = {
        "N": N,
        "patch_size": ps,
        "stride": stride,
        "voxel_size_um": report.get("voxel_size_um", 25.0),
        "dtype_xct":   "uint8",
        "dtype_label": "uint8",
        "shape": [N, ps, ps, ps],
        "splits": n_splits,
        "label_classes": {str(k): v for k, v in LABEL_NAMES.items()},
        "label_rule": ("2 = air (sample_mask == 0, exterior or a drilled "
                       "hole); 1 = pore (mask != 0 inside the specimen); "
                       "0 = material. Air takes precedence over pore."),
        "label_voxel_fraction": {
            LABEL_NAMES[i]: (float(class_counts[i]) / total_vox
                             if total_vox else 0.0) for i in range(3)},
        "hole_rule": report["hole_rule"],
        "split_rule": report["split_rule"],
        "parquet_sha256": parquet_sha,
    }
    with open(meta_tmp, "w") as fh:
        json.dump(meta, fh, indent=2)

    os.rename(meta_tmp,      meta_path)    # atomic on POSIX
    os.rename(xct_partial,   xct_bin)
    os.rename(label_partial, label_bin)
    progress_path.unlink(missing_ok=True)

    total_gb = 2 * bytes_per_arr / 1e9
    print(
        f"\nExtracted {N} patches in {elapsed:.1f}s  "
        f"({total_gb:.1f} GB at {total_gb/elapsed:.1f} GB/s)"
    )
    print("Label voxel fractions: "
          + ", ".join(f"{k} {v:.4f}"
                      for k, v in meta["label_voxel_fraction"].items()))
    print("patches_meta.json written with parquet SHA-256 for integrity checks.")

    # ------------------------------------------------------------------
    # Optional verification
    # ------------------------------------------------------------------
    if args.verify:
        print("\nRunning verification (128 random patches, seed 42) …")
        rng = random.Random(42)
        indices = rng.sample(range(N), min(128, N))

        mmap_xct_v   = np.memmap(str(xct_bin),   dtype=dtype, mode="r", shape=shape)
        mmap_label_v = np.memmap(str(label_bin), dtype=dtype, mode="r", shape=shape)
        zarr_root_v: zarr.Group = zarr.open_group(str(zarr_root_path), mode="r")
        _grp_cache: dict = {}

        n_errors = 0
        for idx in tqdm(indices, desc="Verifying"):
            row = df.iloc[idx]
            vid = row["volume_id"]
            if vid not in _grp_cache:
                _grp_cache[vid] = zarr_root_v[vid]
            grp = _grp_cache[vid]
            z0, y0, x0 = int(row["z0"]), int(row["y0"]), int(row["x0"])
            sl = np.s_[z0:z0+ps, y0:y0+ps, x0:x0+ps]

            exp_xct   = np.asarray(grp["xct"][sl], dtype=np.uint8)
            exp_label = build_label_volume(
                np.asarray(grp["mask"][sl], dtype=np.uint8),
                np.asarray(grp["sample_mask"][sl], dtype=np.uint8),
            )

            if not np.array_equal(mmap_xct_v[idx], exp_xct):
                print(f"  XCT mismatch at parquet row {idx} (volume {vid})")
                n_errors += 1
            if not np.array_equal(mmap_label_v[idx], exp_label):
                print(f"  Label mismatch at parquet row {idx} (volume {vid})")
                n_errors += 1

        del mmap_xct_v, mmap_label_v
        if n_errors == 0:
            print(f"Verification OK: {len(indices)} patches checked, 0 errors.")
        else:
            print(f"Verification FAILED: {n_errors} mismatches.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()

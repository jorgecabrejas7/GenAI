"""CLI entrypoint: build the dataset from raw TIFF volumes.

Usage
-----
::

    build_dataset --raw_root ./raw_data --out_root ./data/processed \\
        --n_train 40 --n_val 5 --n_test 5 --seed 123

Or via ``python -m poregen.dataset.build_dataset``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from poregen.dataset.io import (
    compute_mask,
    discover_volumes,
    load_volume,
    save_volume_zarr,
)
from poregen.dataset.patch_index import (
    build_patch_index_for_volume,
    save_patch_index,
)
from poregen.dataset.splits import assign_volume_splits, save_splits

logger = logging.getLogger(__name__)


def _parse_chunk_size(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"chunk_size must have 3 values, got {len(parts)}"
        )
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Zarr + Parquet dataset from raw TIFF volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_root", type=str, default="./raw_data")
    p.add_argument("--out_root", type=str, default="./data/processed")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument(
        "--chunk_size",
        type=_parse_chunk_size,
        default="32,32,32",
        help='Zarr chunk size as "D,H,W"',
    )
    p.add_argument("--clevel", type=int, default=3, help="Blosc zstd compression level")
    p.add_argument("--n_train", type=int, required=True)
    p.add_argument("--n_val", type=int, required=True)
    p.add_argument("--n_test", type=int, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--limit_volumes",
        type=int,
        default=None,
        help="Process at most N volumes (for quick smoke tests).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = build_parser().parse_args(argv)

    t0 = time.perf_counter()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. Discover volumes ------------------------------------------------
    all_vols = discover_volumes(args.raw_root)
    if args.limit_volumes is not None:
        all_vols = all_vols[: args.limit_volumes]
    if not all_vols:
        logger.error("No volumes found under %s", args.raw_root)
        return

    logger.info("Using %d volumes", len(all_vols))

    # ---- 2. Volume-level splits ---------------------------------------------
    vol_ids = [v.volume_id for v in all_vols]
    splits = assign_volume_splits(
        vol_ids,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
    splits_path = out_root / "splits.json"
    save_splits(splits, splits_path, seed=args.seed)

    # ---- 3. Process each assigned volume ------------------------------------
    dfs: list[pd.DataFrame] = []

    for vi in all_vols:
        if vi.volume_id not in splits:
            logger.info("Skipping unassigned volume %s", vi.volume_id)
            continue

        split = splits[vi.volume_id]
        logger.info(
            "Processing %s  split=%s  path=%s", vi.volume_id, split, vi.path
        )

        # Load
        xct = load_volume(vi.path)
        vi.shape = xct.shape
        logger.info("  shape=%s  dtype=%s", xct.shape, xct.dtype)

        # Mask
        mask = compute_mask(xct)

        # Save Zarr
        save_volume_zarr(
            xct,
            mask,
            out_root,
            vi.volume_id,
            chunk_size=args.chunk_size,
            clevel=args.clevel,
        )

        # Patch index (while mask is still in RAM)
        df = build_patch_index_for_volume(
            mask=mask,
            volume_id=vi.volume_id,
            source_group=vi.source_group,
            split=split,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        dfs.append(df)

    # ---- 4. Write combined patch index --------------------------------------
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        index_path = out_root / "patch_index.parquet"
        save_patch_index(combined, index_path)
    else:
        combined = pd.DataFrame()
        logger.warning("No patches generated — all volumes may have been skipped.")

    # ---- 5. Summary ---------------------------------------------------------
    elapsed = time.perf_counter() - t0
    n_assigned = sum(1 for v in all_vols if v.volume_id in splits)
    logger.info("=" * 60)
    logger.info("DONE  volumes=%d  patches=%d  time=%.1fs", n_assigned, len(combined), elapsed)
    for sp in ("train", "val", "test"):
        cnt = len(combined[combined["split"] == sp]) if len(combined) else 0
        logger.info("  %s patches: %d", sp, cnt)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""CLI entrypoint: build the dataset from raw TIFF volumes.

Usage
-----
::

    build_dataset --raw_root ./raw_data --out_root ./data/split_v1 \
        --n_train 40 --n_val 5 --n_test 5 --seed 123

    # Only compute per-volume intensity stats from existing zarr data:
    build_dataset --out_root ./data/split_v1 --stats_only

Or via ``python -m poregen.dataset.build_dataset``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import zarr

from poregen.dataset.io import (
    compute_mask,
    compute_volume_stats,
    compute_volume_stats_from_zarr,
    discover_volumes,
    load_volume,
    load_volume_stats,
    save_volume_stats,
    save_volume_zarr,
)
from poregen.dataset.patch_index import (
    build_patch_index_for_volume,
    save_patch_index,
)
from poregen.dataset.splits import assign_volume_splits, load_splits, save_splits

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
    p.add_argument("--out_root", type=str, default="./data/split_v1")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument(
        "--chunk_size",
        type=_parse_chunk_size,
        default="64,64,64",
        help='Zarr chunk size as "D,H,W" (default 64,64,64 aligns with patch size)',
    )
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_val", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--limit_volumes",
        type=int,
        default=None,
        help="Process at most N volumes (for quick smoke tests).",
    )
    p.add_argument(
        "--stats_only",
        action="store_true",
        help=(
            "Skip TIFF loading / zarr writing / patch indexing. "
            "Only compute per-volume intensity statistics from existing zarr data "
            "and save them to volume_stats.json."
        ),
    )
    return p


def _zarr_group_exists(out_root: Path, volume_id: str) -> bool:
    store = out_root / "volumes.zarr"
    if not store.exists():
        return False
    try:
        root = zarr.open_group(str(store), mode="r")
        return volume_id in root
    except Exception:
        return False


def _stats_only_mode(out_root: Path) -> None:
    """Compute missing per-volume stats from existing zarr data."""
    store_path = out_root / "volumes.zarr"
    if not store_path.exists():
        logger.error("volumes.zarr not found at %s — nothing to do.", store_path)
        return

    root = zarr.open_group(str(store_path), mode="r")
    all_vol_ids = list(root.group_keys())
    logger.info("Found %d zarr volumes to inspect.", len(all_vol_ids))

    stats = load_volume_stats(out_root)
    updated = 0

    for vol_id in sorted(all_vol_ids):
        if vol_id in stats:
            logger.info("Stats already present for %s — skipping.", vol_id)
            continue

        logger.info("Computing stats for %s …", vol_id)
        try:
            zarr_xct = root[vol_id]["xct"]
            vol_stats = compute_volume_stats_from_zarr(zarr_xct)
            stats[vol_id] = vol_stats
            updated += 1
            save_volume_stats(stats, out_root)
            logger.info(
                "  mean=%.2f  std=%.2f  n_fg=%d  otsu=%d",
                vol_stats["mean"],
                vol_stats["std"],
                vol_stats["n_foreground"],
                vol_stats.get("otsu_threshold", -1),
            )
        except Exception as exc:
            logger.error("Failed to compute stats for %s: %s", vol_id, exc)

    logger.info("Stats updated for %d volumes. Total in file: %d.", updated, len(stats))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = build_parser().parse_args(argv)

    t0 = time.perf_counter()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── stats_only shortcut ──────────────────────────────────────────────────
    if args.stats_only:
        _stats_only_mode(out_root)
        logger.info("stats_only run complete in %.1fs", time.perf_counter() - t0)
        return

    # ── Full build mode ──────────────────────────────────────────────────────

    if args.n_train is None or args.n_val is None or args.n_test is None:
        logger.error("--n_train, --n_val, and --n_test are required for a full build.")
        return

    # ---- 1. Discover volumes ------------------------------------------------
    all_vols = discover_volumes(args.raw_root)
    if args.limit_volumes is not None:
        all_vols = all_vols[: args.limit_volumes]
    if not all_vols:
        logger.error("No volumes found under %s", args.raw_root)
        return

    logger.info("Using %d volumes", len(all_vols))

    # ---- 2. Volume-level splits (load existing or create new) ---------------
    splits_path = out_root / "splits.json"
    if splits_path.exists():
        logger.info("Loading existing splits from %s", splits_path)
        splits = load_splits(splits_path)
    else:
        vol_ids = [v.volume_id for v in all_vols]
        splits = assign_volume_splits(
            vol_ids,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
        )
        save_splits(splits, splits_path, seed=args.seed)

    # ---- 3. Load existing patch index rows and stats -------------------------
    index_path = out_root / "patch_index.parquet"
    if index_path.exists():
        existing_df = pd.read_parquet(str(index_path))
        already_indexed = set(existing_df["volume_id"].unique())
        logger.info(
            "Existing patch index has %d volumes / %d patches.",
            len(already_indexed),
            len(existing_df),
        )
    else:
        existing_df = pd.DataFrame()
        already_indexed: set[str] = set()

    stats = load_volume_stats(out_root)

    # ---- 4. Process each assigned volume ------------------------------------
    new_dfs: list[pd.DataFrame] = []

    for vi in all_vols:
        if vi.volume_id not in splits:
            logger.info("Skipping unassigned volume %s", vi.volume_id)
            continue

        split = splits[vi.volume_id]

        zarr_exists = _zarr_group_exists(out_root, vi.volume_id)
        patch_indexed = vi.volume_id in already_indexed
        stats_computed = vi.volume_id in stats

        if zarr_exists and patch_indexed and stats_computed:
            logger.info("Already fully processed %s — skipping.", vi.volume_id)
            continue

        logger.info(
            "Processing %s  split=%s  zarr=%s  patches=%s  stats=%s",
            vi.volume_id,
            split,
            zarr_exists,
            patch_indexed,
            stats_computed,
        )

        # Load raw volume (needed if zarr or stats are missing)
        xct = load_volume(vi.path)
        vi.shape = xct.shape
        logger.info("  shape=%s  dtype=%s", xct.shape, xct.dtype)

        # Mask + stats from raw data (onlypores gives sample_mask for free)
        pore_mask, sample_mask = compute_mask(xct)

        if not zarr_exists:
            save_volume_zarr(
                xct,
                pore_mask,
                out_root,
                vi.volume_id,
                chunk_size=args.chunk_size,
            )

        if not stats_computed:
            vol_stats = compute_volume_stats(xct, sample_mask)
            stats[vi.volume_id] = vol_stats
            save_volume_stats(stats, out_root)
            logger.info(
                "  mean=%.2f  std=%.2f  n_fg=%d",
                vol_stats["mean"],
                vol_stats["std"],
                vol_stats["n_foreground"],
            )

        if not patch_indexed:
            df = build_patch_index_for_volume(
                mask=pore_mask,
                volume_id=vi.volume_id,
                source_group=vi.source_group,
                split=split,
                patch_size=args.patch_size,
                stride=args.stride,
            )
            new_dfs.append(df)

    # ---- 5. Write combined patch index (merge with existing) ----------------
    all_dfs = []
    if not existing_df.empty:
        all_dfs.append(existing_df)
    all_dfs.extend(new_dfs)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        save_patch_index(combined, index_path)
    else:
        combined = existing_df if not existing_df.empty else pd.DataFrame()
        if new_dfs:
            logger.warning("No new patches generated.")

    # ---- 6. Summary ---------------------------------------------------------
    elapsed = time.perf_counter() - t0
    n_assigned = sum(1 for v in all_vols if v.volume_id in splits)
    logger.info("=" * 60)
    logger.info(
        "DONE  volumes=%d  patches=%d  time=%.1fs",
        n_assigned,
        len(combined),
        elapsed,
    )
    for sp in ("train", "val", "test"):
        cnt = len(combined[combined["split"] == sp]) if len(combined) else 0
        logger.info("  %s patches: %d", sp, cnt)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

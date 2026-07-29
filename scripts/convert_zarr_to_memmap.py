"""
convert_zarr_to_memmap.py

One-time migration tool: converts an existing zarr latent store to numpy
memmap format.

Reads:
    <latents_root>/latents.zarr/latents   (N, n_channels, 16, 16, 16) float16

Writes:
    <latents_root>/latents.bin            flat float16 C-contiguous binary
    <latents_root>/latents_meta.json      metadata dict

Row i in the output maps to row i in the zarr (parquet alignment preserved).

Usage:
    python scripts/convert_zarr_to_memmap.py \\
        --latents-root <path>                  \\
        [--chunk-size 2048]                    \\
        [--z-channels INT]                     \\
        [--pack-scheme {none,mu_then_logvar}]  \\
        [--force]                              \\
        [--verify]
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pack_scheme auto-detection
# ---------------------------------------------------------------------------

def _detect_pack_scheme(latents_root, zarr_n_ch, cli_z_channels, cli_pack_scheme):
    """Return (pack_scheme, z_channels) following the documented priority rules.

    Priority (highest first):
      1. CLI --pack-scheme AND --z-channels both set  -> direct override
      2. latent_scale_stats.json auto-detection (with CLI partial override)
      3. CLI fallback heuristics
      4. Default: pack_scheme="none", z_channels=zarr_n_ch
    """
    # Step 1: both flags explicit -> unconditional override
    if cli_pack_scheme is not None and cli_z_channels is not None:
        log.info(
            "Using explicit CLI overrides: pack_scheme=%s z_channels=%d",
            cli_pack_scheme, cli_z_channels,
        )
        return cli_pack_scheme, cli_z_channels

    # Step 2: try latent_scale_stats.json
    stats_path = os.path.join(latents_root, "latent_scale_stats.json")
    z_channels_from_stats = None
    if os.path.isfile(stats_path):
        try:
            with open(stats_path) as fh:
                stats = json.load(fh)
            if "z_channels" in stats:
                z_channels_from_stats = int(stats["z_channels"])
                log.info(
                    "latent_scale_stats.json → z_channels=%d", z_channels_from_stats
                )
        except Exception as exc:
            log.warning("Could not read %s: %s", stats_path, exc)

    if z_channels_from_stats is not None:
        if zarr_n_ch == 2 * z_channels_from_stats:
            pack_scheme = "mu_then_logvar"
            z_channels = z_channels_from_stats
        else:
            pack_scheme = "none"
            z_channels = zarr_n_ch
        log.info(
            "Auto-detected from stats file: pack_scheme=%s z_channels=%d",
            pack_scheme, z_channels,
        )
        # Partial CLI overrides still apply
        if cli_pack_scheme is not None:
            pack_scheme = cli_pack_scheme
            log.info("CLI --pack-scheme overrides to: %s", pack_scheme)
        if cli_z_channels is not None:
            z_channels = cli_z_channels
            log.info("CLI --z-channels overrides to: %d", z_channels)
        return pack_scheme, z_channels

    # Step 3: stats file absent or missing key — fall back to CLI flags
    if cli_pack_scheme is not None:
        pack_scheme = cli_pack_scheme
    else:
        pack_scheme = "none"

    if cli_z_channels is not None:
        z_channels = cli_z_channels
        # Auto-upgrade pack_scheme when ratio matches and user did not pin it
        if cli_pack_scheme is None and zarr_n_ch == 2 * z_channels:
            pack_scheme = "mu_then_logvar"
            log.info(
                "zarr_n_ch=%d == 2 * --z-channels=%d → auto pack_scheme=mu_then_logvar",
                zarr_n_ch, z_channels,
            )
    else:
        z_channels = zarr_n_ch

    log.info("Final: pack_scheme=%s z_channels=%d", pack_scheme, z_channels)
    return pack_scheme, z_channels


# ---------------------------------------------------------------------------
# Resumability helper
# ---------------------------------------------------------------------------

def _chunk_already_written(partial_mmap, start, end):
    """Return True if all items in [start, end) appear already written (all non-zero)."""
    probe = np.array(partial_mmap[start:end, 0, 0, 0, 0])
    return bool(np.all(probe != 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert <latents-root>/latents.zarr/latents to numpy memmap "
            "(latents.bin + latents_meta.json)."
        )
    )
    parser.add_argument(
        "--latents-root",
        required=True,
        metavar="PATH",
        help="Directory containing latents.zarr/",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        metavar="INT",
        help="Number of items per zarr read (default: 2048).",
    )
    parser.add_argument(
        "--z-channels",
        type=int,
        default=None,
        metavar="INT",
        help="Explicit z_channels override for metadata and auto-detection.",
    )
    parser.add_argument(
        "--pack-scheme",
        choices=["none", "mu_then_logvar"],
        default=None,
        help="Override pack_scheme auto-detection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing latents.bin (and latents.bin.partial) if present.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, compare 64 random rows zarr vs memmap (seed 42).",
    )
    args = parser.parse_args()

    latents_root = os.path.abspath(args.latents_root)

    # ------------------------------------------------------------------
    # Locate and open zarr array
    # ------------------------------------------------------------------
    zarr_path = os.path.join(latents_root, "latents.zarr", "latents")
    if not os.path.exists(zarr_path):
        log.error(
            "zarr array not found at: %s\n"
            "Expected <latents-root>/latents.zarr/latents",
            zarr_path,
        )
        raise SystemExit(1)

    zarr_arr = zarr.open(zarr_path, mode="r")
    N = zarr_arr.shape[0]
    n_channels = zarr_arr.shape[1]
    spatial = list(zarr_arr.shape[2:])   # typically [16, 16, 16]
    dtype = np.float16
    shape = (N, n_channels) + tuple(spatial)

    log.info("zarr array: shape=%s dtype=%s", zarr_arr.shape, zarr_arr.dtype)

    # ------------------------------------------------------------------
    # Pack-scheme / z_channels detection
    # ------------------------------------------------------------------
    pack_scheme, z_channels = _detect_pack_scheme(
        latents_root, n_channels, args.z_channels, args.pack_scheme
    )
    log.info("pack_scheme=%s  z_channels=%d", pack_scheme, z_channels)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    bin_path = os.path.join(latents_root, "latents.bin")
    partial_path = os.path.join(latents_root, "latents.bin.partial")
    meta_path = os.path.join(latents_root, "latents_meta.json")
    meta_tmp_path = os.path.join(latents_root, "latents_meta.json.tmp")

    # ------------------------------------------------------------------
    # Guard: final latents.bin already exists
    # ------------------------------------------------------------------
    if os.path.isfile(bin_path):
        if args.force:
            log.info("--force set: removing existing %s", bin_path)
            os.remove(bin_path)
        else:
            log.error(
                "%s already exists. Use --force to overwrite.", bin_path
            )
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # Open or create the partial memmap
    # ------------------------------------------------------------------
    resuming = False
    if os.path.isfile(partial_path) and not args.force:
        print("Resuming from latents.bin.partial")
        resuming = True
        mmap = np.memmap(partial_path, dtype=dtype, mode="r+", shape=shape)
    else:
        if os.path.isfile(partial_path):
            log.info("--force set: removing existing latents.bin.partial")
            os.remove(partial_path)
        mmap = np.memmap(partial_path, dtype=dtype, mode="w+", shape=shape)

    # ------------------------------------------------------------------
    # Chunk-wise copy with progress bar
    # ------------------------------------------------------------------
    chunk_size = args.chunk_size
    n_chunks = (N + chunk_size - 1) // chunk_size
    spatial_prod = int(np.prod(spatial))
    bytes_per_item = n_channels * spatial_prod * np.dtype(dtype).itemsize
    total_bytes = N * bytes_per_item

    t0 = time.perf_counter()
    with tqdm(total=N, unit="items", desc="Converting") as pbar:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, N)

            if resuming and _chunk_already_written(mmap, start, end):
                pbar.update(end - start)
                continue

            chunk = zarr_arr[start:end]
            mmap[start:end] = np.asarray(chunk, dtype=dtype)
            pbar.update(end - start)

    mmap.flush()
    del mmap

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Atomic rename sequence
    # ------------------------------------------------------------------
    meta = {
        "N": N,
        "n_channels": n_channels,
        "spatial": spatial,
        "dtype": "float16",
        "pack_scheme": pack_scheme,
        "z_channels": z_channels,
    }

    with open(meta_tmp_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    os.rename(meta_tmp_path, meta_path)   # atomic on POSIX
    os.rename(partial_path, bin_path)     # atomic on POSIX

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_gb = total_bytes / (1024 ** 3)
    speed_gb_s = total_gb / elapsed if elapsed > 0 else float("inf")
    print(
        f"\nConverted {N} items in {elapsed:.1f}s "
        f"({total_gb:.1f} GB at {speed_gb_s:.1f} GB/s)"
    )
    print("latents_meta.json written.")
    print("zarr source preserved. To remove after verification:")
    print(f"  rm -rf {os.path.join(latents_root, 'latents.zarr')}")

    # ------------------------------------------------------------------
    # Optional verification
    # ------------------------------------------------------------------
    if args.verify:
        print("\nRunning verification (64 random rows, seed 42) ...")
        rng = random.Random(42)
        indices = rng.sample(range(N), min(64, N))

        zarr_check = zarr.open(zarr_path, mode="r")
        mmap_check = np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)

        n_checked = len(indices)
        for idx in indices:
            zarr_row = np.asarray(zarr_check[idx], dtype=dtype)
            mmap_row = np.array(mmap_check[idx])
            if not np.array_equal(zarr_row, mmap_row):
                print(f"MISMATCH at index {idx}!")
                raise SystemExit(1)

        del mmap_check
        print(f"Verification OK: {n_checked}/{n_checked} rows match.")


if __name__ == "__main__":
    main()

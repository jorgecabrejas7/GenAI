"""Pre-compute VAE latents for all patches and store them in a numpy memmap.

Usage
-----
python scripts/encode_latents.py \\
    --experiment r05/base \\
    --checkpoint runs/vae/r05/<run>/checkpoints/best.ckpt \\
    --data-root data/split_v2 \\
    --output data/split_v2/latents_s64 \\
    [--stride 64] \\
    [--batch-size 64] \\
    [--device cuda] \\
    [--splits train val test]

On-disk layout produced
-----------------------
<output>/
├── latents.bin           — float16, C-contiguous, shape (N, z_ch, 16, 16, 16)
├── latents_meta.json     — {"N": int, "n_channels": int, "spatial": [16,16,16],
│                            "dtype": "float16", "pack_scheme": "none",
│                            "z_channels": int}
└── latents_index.parquet
    columns: volume_id, z0, y0, x0, ps, stride, porosity, vol_porosity, split,
             source_group, vol_depth, vol_height, vol_width,
             grid_iz, grid_iy, grid_ix, parity

Row i of the parquet corresponds to latents.bin row i.

Parity
------
parity = (grid_iz + grid_iy + grid_ix) % 2, where grid_i* = coord // stride.
Parity-0 patches are "anchors" (generated first at inference with all in-bounds
neighbors UNKNOWN).  Parity-1 patches are "non-anchors" (generated after
anchors, seeing all in-bounds neighbors as EXISTS).

Restartability
--------------
If the memmap bin exists and the metadata matches, rows that already have
non-zero latents are skipped (detected by checking the first voxel of each
stored tensor).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return here.parent


def _load_vae(experiment: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    repo = _find_repo_root()
    sys.path.insert(0, str(repo / "src"))

    from poregen.configuration import resolve_experiment
    from poregen.models.vae import build_vae
    from poregen.training.checkpoint import load_checkpoint

    resolved = resolve_experiment(experiment, repo_root=repo)
    cfg = resolved.cfg
    model_cfg = cfg["model"]
    model = build_vae(
        model_cfg["name"],
        in_channels=model_cfg.get("in_channels", 2),
        z_channels=int(model_cfg["z_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        n_blocks=int(model_cfg["n_blocks"]),
        patch_size=int(model_cfg["patch_size"]),
    ).to(device)

    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    logger.info("Loaded VAE from %s", checkpoint)
    return model


def _build_patch_index(
    data_root: Path,
    stride: int,
    splits_json: Path,
) -> pd.DataFrame:
    """Build a fresh patch index at the requested stride from volumes.zarr.

    Patch coordinates and per-patch porosity are computed directly from the
    volume data — independent of any pre-existing patch_index.parquet.
    """
    import zarr as _zarr

    from poregen.dataset.patch_index import build_patch_index_for_volume
    from poregen.dataset.splits import load_splits

    split_map: dict[str, str] = load_splits(splits_json)

    volumes_root = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")

    frames = []
    for vid in sorted(volumes_root.keys()):
        split = split_map.get(vid)
        if split is None:
            logger.warning("Volume %s not in splits.json — skipping", vid)
            continue

        mask_arr = volumes_root[vid]["mask"]
        mask = np.array(mask_arr, dtype=np.uint8)

        # True volume VVF — computed directly from the full mask, not from
        # patch means, which are biased when stride < patch_size (overlapping).
        vol_porosity = float(mask.mean())

        # source_group: infer from splits.json key structure (best-effort)
        source_group = "unknown"

        df_vol = build_patch_index_for_volume(
            mask=mask,
            volume_id=vid,
            source_group=source_group,
            split=split,
            patch_size=64,
            stride=stride,
        )
        df_vol["vol_porosity"] = np.float32(vol_porosity)
        frames.append(df_vol)

    if not frames:
        raise RuntimeError("No volumes found in volumes.zarr — check data_root.")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Built patch index: %d patches at stride=%d", len(df), stride)
    return df


def _open_memmap_store(n_patches: int, output_dir: Path, z_channels: int) -> np.ndarray:
    """Open (or resume into) the float16 memmap output file."""
    shape     = (n_patches, z_channels, 16, 16, 16)
    bin_path  = output_dir / "latents.bin"
    meta_path = output_dir / "latents_meta.json"

    if bin_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        existing_shape = (meta["N"], meta["n_channels"]) + tuple(meta["spatial"])
        if existing_shape == shape:
            logger.info("Resuming into existing memmap %s", bin_path)
            return np.memmap(str(bin_path), dtype="float16", mode="r+", shape=shape)
        logger.warning(
            "Existing memmap shape %s does not match expected %s — recreating.",
            existing_shape, shape,
        )

    mmap = np.memmap(str(bin_path), dtype="float16", mode="w+", shape=shape)
    logger.info(
        "Created memmap: %s  shape=%s  %.1f GB",
        bin_path, shape, np.prod(shape) * 2 / 1e9,
    )
    return mmap


def _volume_shapes(data_root: Path) -> dict[str, tuple[int, int, int]]:
    import zarr as _zarr
    root = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")
    shapes: dict[str, tuple[int, int, int]] = {}
    for vid in root.keys():
        try:
            xct = root[vid]["xct"]
            shapes[vid] = (int(xct.shape[0]), int(xct.shape[1]), int(xct.shape[2]))
        except Exception:
            pass
    return shapes


def _add_vol_porosity(data_root: Path, output_dir: Path) -> None:
    """Patch an existing latents_index.parquet with the true per-volume VVF.

    Reads volumes.zarr masks, computes mask.mean() per volume, and writes the
    result back as a ``vol_porosity`` column.  The latent Zarr array is not
    touched.
    """
    import zarr as _zarr

    parquet_path = output_dir / "latents_index.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No latents_index.parquet found at {parquet_path}")

    df = pd.read_parquet(str(parquet_path))
    volumes_root = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")

    unique_vids = df["volume_id"].unique()
    logger.info("Computing vol_porosity for %d volumes …", len(unique_vids))

    vol_por: dict[str, float] = {}
    for vid in sorted(unique_vids):
        if vid not in volumes_root:
            logger.warning("Volume %s not found in volumes.zarr — skipping", vid)
            continue
        mask = np.array(volumes_root[vid]["mask"], dtype=np.uint8)
        vol_por[vid] = float(mask.mean())
        logger.info("  %s  vol_porosity=%.6f", vid, vol_por[vid])

    df["vol_porosity"] = df["volume_id"].map(vol_por).astype(np.float32)

    missing = df["vol_porosity"].isna().sum()
    if missing:
        logger.warning("%d rows have no vol_porosity (volume not in zarr) — filling with patch mean", missing)
        fallback = df.groupby("volume_id")["porosity"].transform("mean")
        df["vol_porosity"] = df["vol_porosity"].fillna(fallback.astype(np.float32))

    df.to_parquet(str(parquet_path), index=True)
    logger.info(
        "Updated %s with vol_porosity (min=%.4f  max=%.4f  mean=%.4f)",
        parquet_path,
        df["vol_porosity"].min(),
        df["vol_porosity"].max(),
        df["vol_porosity"].mean(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode VAE latents for all patches.")
    ap.add_argument("--experiment",  help="Experiment ref, e.g. 'r05/base'")
    ap.add_argument("--checkpoint",  help="Path to VAE .ckpt file")
    ap.add_argument("--data-root",   required=True, help="Data root dir (contains volumes.zarr, splits.json)")
    ap.add_argument("--output",      required=True, help="Output directory for latents")
    ap.add_argument("--stride",      type=int, default=64, help="Patch stride in voxels (default: 64)")
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--add-vol-porosity", action="store_true",
        help="Only patch an existing latents_index.parquet with the true per-volume "
             "VVF (mask.mean()).  No VAE loading or latent encoding is performed.",
    )
    args = ap.parse_args()

    data_root  = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.add_vol_porosity:
        _add_vol_porosity(data_root, output_dir)
        return

    if not args.experiment or not args.checkpoint:
        ap.error("--experiment and --checkpoint are required unless --add-vol-porosity is set")

    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    model = _load_vae(args.experiment, args.checkpoint, device)

    # Infer z_channels from a dummy forward pass
    with torch.no_grad():
        dummy_xct  = torch.zeros(1, 1, 64, 64, 64, device=device)
        dummy_mask = torch.zeros(1, 1, 64, 64, 64, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            out = model(dummy_xct, dummy_mask)
        z_channels = out.mu.shape[1]
    logger.info("z_channels=%d  (inferred from dummy forward pass)", z_channels)

    # Build patch index at the requested stride (independent of pre-built parquet)
    splits_json = data_root / "splits.json"
    df_full = _build_patch_index(data_root, args.stride, splits_json)

    mmap = _open_memmap_store(len(df_full), output_dir, z_channels)

    vol_shapes    = _volume_shapes(data_root)
    import zarr as _zarr
    volumes_root  = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")

    # Encode in row order to preserve parquet alignment
    bs      = args.batch_size
    n_total = len(df_full)

    with tqdm(total=n_total, desc="Encoding", unit="patch") as pbar:
        i = 0
        while i < n_total:
            j    = min(i + bs, n_total)
            rows = df_full.iloc[i:j]

            # Skip already-encoded blocks (first voxel non-zero heuristic)
            existing = np.array(mmap[i:j, 0, 0, 0, 0], dtype=np.float32)
            if np.all(existing != 0.0):
                pbar.update(j - i)
                i = j
                continue

            xct_list = []
            for _, row in rows.iterrows():
                vid = row["volume_id"]
                z0  = int(row["z0"])
                y0  = int(row["y0"])
                x0  = int(row["x0"])
                ps  = int(row["ps"])
                xct_patch = np.array(
                    volumes_root[vid]["xct"][z0:z0+ps, y0:y0+ps, x0:x0+ps],
                    dtype=np.float32,
                ) / 255.0
                xct_list.append(xct_patch)

            xct_batch  = torch.from_numpy(np.stack(xct_list)[:, None]).to(device)
            mask_batch = torch.zeros_like(xct_batch)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(xct_batch, mask_batch)
            mu = out.mu.float().cpu().numpy().astype(np.float16)

            mmap[i:j] = mu
            if (i // bs) % 256 == 0:
                mmap.flush()
            pbar.update(j - i)
            i = j

    mmap.flush()
    logger.info("Memmap write complete.")

    _meta = {
        "N":           n_total,
        "n_channels":  z_channels,
        "spatial":     [16, 16, 16],
        "dtype":       "float16",
        "pack_scheme": "none",
        "z_channels":  z_channels,
    }
    _meta_tmp = output_dir / "latents_meta.json.tmp"
    _meta_tmp.write_text(json.dumps(_meta, indent=2))
    _meta_tmp.rename(output_dir / "latents_meta.json")
    logger.info("Saved latents_meta.json → %s", output_dir / "latents_meta.json")

    logger.info("Encoded %d patches → %s", n_total, output_dir / "latents.bin")

    # Build latents_index.parquet
    df_idx = df_full.copy()

    # Volume shape columns
    df_idx["vol_depth"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[0])
    df_idx["vol_height"] = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[1])
    df_idx["vol_width"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[2])

    # Grid indices and parity for inference-faithful conditioning
    df_idx["grid_iz"] = (df_idx["z0"] // args.stride).astype(np.int32)
    df_idx["grid_iy"] = (df_idx["y0"] // args.stride).astype(np.int32)
    df_idx["grid_ix"] = (df_idx["x0"] // args.stride).astype(np.int32)
    df_idx["parity"]  = (
        (df_idx["grid_iz"] + df_idx["grid_iy"] + df_idx["grid_ix"]) % 2
    ).astype(np.int8)

    parity_counts = df_idx["parity"].value_counts().to_dict()
    logger.info("Parity distribution: %s", parity_counts)

    out_parquet = output_dir / "latents_index.parquet"
    df_idx.to_parquet(str(out_parquet), index=True)
    logger.info("Saved index → %s  (%d rows)", out_parquet, len(df_idx))

    # Latent scale stats (train split only to avoid leakage)
    logger.info("Computing latent stats from train split …")
    train_idxs = df_full.index[df_full["split"] == "train"].to_numpy()
    if len(train_idxs) == 0:
        raise RuntimeError("No 'train' split rows — cannot compute scale stats.")
    sample_idxs = np.random.choice(train_idxs, min(10000, len(train_idxs)), replace=False)
    sample = np.asarray(mmap[sample_idxs], dtype=np.float32)
    stats = {"mean": float(sample.mean()), "std": float(sample.std())}
    stats_path = output_dir / "latent_scale_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Latent stats (train-only): mean=%.4f  std=%.4f", stats["mean"], stats["std"])
    logger.info("Done.")


if __name__ == "__main__":
    main()

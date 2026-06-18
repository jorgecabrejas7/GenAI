"""Pre-compute VAE posterior parameters (mu, logvar) for all patches.

Stores mu and logvar packed into a SINGLE zarr array:
    latents.zarr/latents  (N, 2*z_ch, 16, 16, 16) float16
    first z_ch channels = mu
    last  z_ch channels = logvar

One zarr read per item at training time (same as ldm01) — halves I/O vs a
two-array design.  ldm01's mu-only store is NOT touched.

Usage
-----
python scripts/encode_latents_sampled.py \\
    --experiment r05/base \\
    --checkpoint runs/vae/r05/<run>/best.ckpt \\
    --data-root data/split_v2 \\
    --output data/split_v2/latents_s64_sampled \\
    [--stride 32] \\
    [--batch-size 64] \\
    [--device cuda] \\
    [--n-scale-samples 50000]

On-disk layout produced
-----------------------
<output>/
├── latents.zarr/
│   └── latents   (N, 2*z_ch, 16, 16, 16) float16
│                 channels 0..z_ch-1   = mu
│                 channels z_ch..2*z_ch-1 = logvar
├── latents_index.parquet
│   columns: volume_id, z0, y0, x0, ps, stride, porosity, vol_porosity, split,
│             source_group, vol_depth, vol_height, vol_width,
│             grid_iz, grid_iy, grid_ix, parity
└── latent_scale_stats.json
    {"z_channels": C, "mean": <mean of z_train>, "std": <std of z_train>}
    where z = mu + exp(0.5*logvar)*eps computed from train-split patches only.

Restartability
--------------
Encodes by batch; skips batches where latents[i, 0, 0, 0, 0] is already non-zero.
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

        mask = np.array(volumes_root[vid]["mask"], dtype=np.uint8)
        vol_porosity = float(mask.mean())

        df_vol = build_patch_index_for_volume(
            mask=mask,
            volume_id=vid,
            source_group="unknown",
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


def _open_zarr_store(n_patches: int, output_dir: Path, z_channels: int) -> zarr.Array:
    """Open (or create) single zarr array storing [mu | logvar] along channel axis."""
    zarr_path = output_dir / "latents.zarr"
    arr_path  = str(zarr_path / "latents")
    n_ch = 2 * z_channels   # first half = mu, second half = logvar

    try:
        arr = zarr.open_array(arr_path, mode="r+")
        if arr.shape == (n_patches, n_ch, 16, 16, 16):
            logger.info("Resuming into existing zarr array %s", arr_path)
            return arr
        logger.warning(
            "Existing zarr shape %s does not match expected (%d, %d, 16, 16, 16) — recreating.",
            arr.shape, n_patches, n_ch,
        )
    except Exception:
        pass

    arr = zarr.open_array(
        arr_path,
        mode="w",
        shape=(n_patches, n_ch, 16, 16, 16),
        dtype="float16",
        chunks=(256, n_ch, 16, 16, 16),
    )
    logger.info(
        "Created zarr array: shape=%s  dtype=float16  (ch 0..%d=mu, ch %d..%d=logvar)",
        arr.shape, z_channels - 1, z_channels, n_ch - 1,
    )
    return arr


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encode VAE posterior (mu+logvar packed) for all patches — ldm02 latent store."
    )
    ap.add_argument("--experiment",  required=True, help="Experiment ref, e.g. 'r05/base'")
    ap.add_argument("--checkpoint",  required=True, help="Path to VAE .ckpt file")
    ap.add_argument("--data-root",   required=True, help="Data root dir (contains volumes.zarr, splits.json)")
    ap.add_argument("--output",      required=True, help="Output directory (separate from ldm01 latent store)")
    ap.add_argument("--stride",      type=int, default=64, help="Patch stride in voxels (default: 64)")
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-scale-samples", type=int, default=50000,
                    help="Number of train-split patches sampled to compute z-std (default 50000)")
    args = ap.parse_args()

    data_root  = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    model = _load_vae(args.experiment, args.checkpoint, device)

    with torch.no_grad():
        dummy_xct  = torch.zeros(1, 1, 64, 64, 64, device=device)
        dummy_mask = torch.zeros(1, 1, 64, 64, 64, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            out = model(dummy_xct, dummy_mask)
        z_channels = out.mu.shape[1]
    logger.info("z_channels=%d  packed array will have %d channels", z_channels, 2 * z_channels)

    splits_json = data_root / "splits.json"
    df_full = _build_patch_index(data_root, args.stride, splits_json)

    arr = _open_zarr_store(len(df_full), output_dir, z_channels)

    vol_shapes   = _volume_shapes(data_root)
    import zarr as _zarr
    volumes_root = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")

    bs      = args.batch_size
    n_total = len(df_full)

    with tqdm(total=n_total, desc="Encoding (mu+logvar packed)", unit="patch") as pbar:
        i = 0
        while i < n_total:
            j    = min(i + bs, n_total)
            rows = df_full.iloc[i:j]

            # Skip already-encoded blocks (first voxel of mu channel non-zero heuristic)
            existing = np.array(arr[i:j, 0, 0, 0, 0], dtype=np.float32)
            if np.all(existing != 0.0):
                pbar.update(j - i)
                i = j
                continue

            xct_list = []
            for _, row in rows.iterrows():
                vid = row["volume_id"]
                z0  = int(row["z0"]); y0 = int(row["y0"]); x0 = int(row["x0"])
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

            # Pack [mu | logvar] along channel axis → (B, 2*C, 16, 16, 16)
            packed = torch.cat([out.mu, out.logvar], dim=1).float().cpu().numpy().astype(np.float16)
            arr[i:j] = packed
            pbar.update(j - i)
            i = j

    logger.info("Encoded %d patches (mu+logvar packed) → %s", n_total, output_dir)

    # Build latents_index.parquet (same structure as ldm01)
    df_idx = df_full.copy()
    df_idx["vol_depth"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[0])
    df_idx["vol_height"] = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[1])
    df_idx["vol_width"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[2])
    df_idx["grid_iz"] = (df_idx["z0"] // args.stride).astype(np.int32)
    df_idx["grid_iy"] = (df_idx["y0"] // args.stride).astype(np.int32)
    df_idx["grid_ix"] = (df_idx["x0"] // args.stride).astype(np.int32)
    df_idx["parity"]  = (
        (df_idx["grid_iz"] + df_idx["grid_iy"] + df_idx["grid_ix"]) % 2
    ).astype(np.int8)

    out_parquet = output_dir / "latents_index.parquet"
    df_idx.to_parquet(str(out_parquet), index=True)
    logger.info("Saved index → %s  (%d rows)", out_parquet, len(df_idx))

    # Compute scale stats on sampled z from train split only
    logger.info("Computing sampled-z stats from train split (n_samples=%d) …", args.n_scale_samples)
    train_idxs = df_full.index[df_full["split"] == "train"].to_numpy()
    if len(train_idxs) == 0:
        raise RuntimeError("No 'train' split rows — cannot compute scale stats.")

    rng = np.random.default_rng(seed=42)
    sample_idxs = rng.choice(train_idxs, min(args.n_scale_samples, len(train_idxs)), replace=False)

    all_z: list[np.ndarray] = []
    chunk = 2048
    for start in range(0, len(sample_idxs), chunk):
        batch_idxs = sample_idxs[start:start + chunk]
        packed_b = np.array(arr.get_orthogonal_selection(batch_idxs), dtype=np.float32)
        mu_b     = packed_b[:, :z_channels]
        logvar_b = packed_b[:, z_channels:]
        sigma_b  = np.exp(0.5 * logvar_b)
        eps      = rng.standard_normal(mu_b.shape).astype(np.float32)
        z_b      = mu_b + sigma_b * eps
        all_z.append(z_b.reshape(-1))

    z_flat = np.concatenate(all_z)
    stats = {
        "z_channels": int(z_channels),
        "mean": float(z_flat.mean()),
        "std":  float(z_flat.std()),
    }
    stats_path = output_dir / "latent_scale_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(
        "Sampled-z stats (train-only): mean=%.4f  std=%.4f  (ldm01 mu-std ≈ 0.4571)",
        stats["mean"], stats["std"],
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()

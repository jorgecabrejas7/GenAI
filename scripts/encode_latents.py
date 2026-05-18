"""Pre-compute VAE latents for all patches and store them in Zarr.

Usage
-----
python scripts/encode_latents.py \\
    --experiment r05/base \\
    --checkpoint runs/vae/r05/<run>/checkpoints/best.ckpt \\
    --data-root data/split_v2 \\
    --output data/split_v2/latents \\
    [--batch-size 64] \\
    [--device cuda] \\
    [--splits train val test]

On-disk layout produced
-----------------------
<output>/
├── latents.zarr/
│   └── latents    (N_total, z_ch, 16, 16, 16) float16  Blosc-zstd
└── latents_index.parquet
    columns: volume_id, z0, y0, x0, ps, porosity, split,
             source_group, vol_depth, vol_height, vol_width

Row i of the parquet corresponds to latents.zarr/latents[i].

Restartability
--------------
If the Zarr array already exists and matches the expected shape, rows that
already have non-zero latents are skipped (detected by checking the first
voxel of each stored tensor).
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
from numcodecs import Blosc
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


def _open_zarr_store(index_path: Path, output_dir: Path, z_channels: int) -> tuple[zarr.Array, pd.DataFrame]:
    df = pd.read_parquet(str(index_path))
    N  = len(df)
    logger.info("Parquet: %d total patches", N)

    zarr_path = output_dir / "latents.zarr"
    latents_path = str(zarr_path / "latents")

    try:
        arr = zarr.open_array(latents_path, mode="r+")
        if arr.shape == (N, z_channels, 16, 16, 16):
            logger.info("Resuming into existing Zarr array %s", latents_path)
            return arr, df
        logger.warning(
            "Existing Zarr shape %s does not match expected (%d, %d, 16, 16, 16) — recreating.",
            arr.shape, N, z_channels,
        )
    except Exception:
        pass

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arr = zarr.open_array(
        latents_path,
        mode="w",
        shape=(N, z_channels, 16, 16, 16),
        dtype="float16",
        chunks=(256, z_channels, 16, 16, 16),
        compressor=compressor,
    )
    logger.info("Created Zarr array: shape=%s, dtype=float16, compressor=zstd-5", arr.shape)
    return arr, df


def _volume_shapes(volumes_root: Path) -> dict[str, tuple[int, int, int]]:
    """Return {volume_id: (depth, height, width)} from the Zarr store."""
    import zarr
    root = zarr.open_group(str(volumes_root / "volumes.zarr"), mode="r")
    shapes: dict[str, tuple[int, int, int]] = {}
    for vid in root.keys():
        try:
            xct = root[vid]["xct"]
            shapes[vid] = (int(xct.shape[0]), int(xct.shape[1]), int(xct.shape[2]))
        except Exception:
            pass
    return shapes


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode VAE latents for all patches.")
    ap.add_argument("--experiment",  required=True, help="Experiment ref, e.g. 'r05/base'")
    ap.add_argument("--checkpoint",  required=True, help="Path to VAE .ckpt file")
    ap.add_argument("--data-root",   required=True, help="Data root dir (contains patch_index.parquet)")
    ap.add_argument("--output",      required=True, help="Output directory for latents")
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits",      nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    device     = torch.device(args.device)
    data_root  = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.bfloat16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    model = _load_vae(args.experiment, args.checkpoint, device)
    z_channels = model.to_mu.out_channels if hasattr(model, "to_mu") else 16

    # ── Discover z_channels from model ──────────────────────────────────────────
    # VAEOutput.mu shape is (B, z_channels, ...)
    with torch.no_grad():
        dummy_xct  = torch.zeros(1, 1, 64, 64, 64, device=device)
        dummy_mask = torch.zeros(1, 1, 64, 64, 64, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            out = model(dummy_xct, dummy_mask)
        z_channels = out.mu.shape[1]
    logger.info("z_channels=%d  (inferred from dummy forward pass)", z_channels)

    index_path = data_root / "patch_index.parquet"
    arr, df_full = _open_zarr_store(index_path, output_dir, z_channels)

    vol_shapes = _volume_shapes(data_root)
    logger.info("Found %d volume shapes", len(vol_shapes))

    # Open volumes Zarr
    import zarr as _zarr
    volumes_root = _zarr.open_group(str(data_root / "volumes.zarr"), mode="r")

    # ── Encode in row order to preserve parquet alignment ──────────────────────
    bs      = args.batch_size
    n_total = len(df_full)
    n_done  = 0

    with tqdm(total=n_total, desc="Encoding", unit="patch") as pbar:
        i = 0
        while i < n_total:
            j = min(i + bs, n_total)
            rows = df_full.iloc[i:j]

            # Check if all already encoded (first voxel non-zero heuristic)
            existing = np.array(arr[i:j, 0, 0, 0, 0], dtype=np.float32)
            if np.all(existing != 0.0):
                n_done += j - i
                pbar.update(j - i)
                i = j
                continue

            # Load XCT patches from Zarr
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

            xct_batch  = torch.from_numpy(np.stack(xct_list)[:, None]).to(device)   # (B,1,ps,ps,ps)
            mask_batch = torch.zeros_like(xct_batch)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(xct_batch, mask_batch)
            mu = out.mu.float().cpu().numpy().astype(np.float16)   # (B, z_ch, 16, 16, 16)

            arr[i:j] = mu
            n_done  += j - i
            pbar.update(j - i)
            i = j

    logger.info("Encoded %d patches → %s", n_done, output_dir / "latents.zarr")

    # ── Build latents_index.parquet ─────────────────────────────────────────────
    df_idx = df_full.copy()
    # Add volume shape columns
    df_idx["vol_depth"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[0])
    df_idx["vol_height"] = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[1])
    df_idx["vol_width"]  = df_idx["volume_id"].map(lambda v: vol_shapes.get(v, (0, 0, 0))[2])

    out_parquet = output_dir / "latents_index.parquet"
    df_idx.to_parquet(str(out_parquet), index=True)
    logger.info("Saved index → %s  (%d rows)", out_parquet, len(df_idx))

    # ── Save latent scale stats (train split only to avoid leakage) ─────────────
    logger.info("Computing latent stats from train split …")
    train_idxs = df_full.index[df_full["split"] == "train"].to_numpy()
    if len(train_idxs) == 0:
        raise RuntimeError("No 'train' split rows — cannot compute scale stats.")
    sample_idxs = np.random.choice(train_idxs, min(10000, len(train_idxs)), replace=False)
    sample = np.array(arr.get_orthogonal_selection(sample_idxs), dtype=np.float32)
    stats = {"mean": float(sample.mean()), "std": float(sample.std())}
    stats_path = output_dir / "latent_scale_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Latent stats (train-only): mean=%.4f  std=%.4f", stats["mean"], stats["std"])
    logger.info("Done.")


if __name__ == "__main__":
    main()

"""Build the latent dataset for LDM training from a trained VAE (ldm04 pipeline).

Encodes every patch of every split with the VAE encoder and stores the
posterior mean (``mu``) and posterior std (``std``) per patch, together with
the raw patch porosity ``phi = mask.mean()`` and an index that links each row
back to the source ``patch_index.parquet`` row.

Latents are stored RAW (un-normalised).  Per-channel normalisation stats are
computed over the train split only and stored in ``metadata.json``; the LDM
data pipeline applies them at load time (see
``poregen.diffusion.latents.LatentDataset``).

Usage
-----
python scripts/build_latent_dataset.py \\
    [--checkpoint runs/vae/<run>/best.ckpt] \\
    [--output data/split_v2/latents_r07z4] \\
    [--data-root data/split_v2] \\
    [--batch-size 256] \\
    [--num-workers 4] \\
    [--limit N]            # per-split cap, for smoke tests

Before encoding, the script prints the estimated total store size and aborts
if the output filesystem has less than 2x that estimate free.

On-disk layout produced (default output: <data_root>/latents_<exp>z<z>/)
-----------------------
data/split_v2/latents_r07z4/
├── metadata.json            — VAE checkpoint + resolved config copy & sha256,
│                              latent shape, per-channel train-split norm stats,
│                              creation date, per-split counts
├── train/
│   ├── latents.zarr/        — arrays "mu" and "std", each (N, C, 16, 16, 16) float16
│   └── index.parquet        — source_row (row in patch_index.parquet), volume_id,
│                              source_group, split, z0, y0, x0, ps, stride,
│                              porosity (from source index), phi (mask.mean())
├── val/   (same files)
└── test/  (same files)

Row i of ``index.parquet`` corresponds to row i of both zarr arrays.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "runs/vae/r07-run-0006-20260814-084936-archv2-conv_noattn_dualbranch"
    "-z4-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone/best.ckpt"
)

SPLITS = ("train", "val", "test")


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


def _default_tag(cfg: dict, run_dir: Path) -> str:
    """``r07`` + ``z=4`` → ``r07z4`` (falls back to the run dir prefix)."""
    exp = cfg.get("experiment", {}).get("name")
    if exp:
        return f"{exp}z{int(cfg['model']['z_channels'])}"
    return "-".join(run_dir.name.split("-")[:3])


def _build_split_dataset(data_root: Path, split: str):
    """Sequential patch dataset for one split — mirrors the backend selection
    of ``poregen.training.data.build_patch_dataloaders`` without its
    training-oriented shuffle/drop_last settings."""
    from poregen.dataset.loader import MemmapPatchDataset, PatchDataset

    index_path = data_root / "patch_index.parquet"
    if (data_root / "patches_meta.json").exists():
        return MemmapPatchDataset(index_path, data_root, split=split)
    return PatchDataset(index_path, data_root, split=split)


def _source_rows(df_full: pd.DataFrame, split: str) -> np.ndarray:
    """Original patch_index.parquet row indices for one split, in dataset order.

    Both patch dataset backends filter to the split and drop porosity > 1.0
    rows while preserving parquet order, so this aligns 1-to-1 with sequential
    (shuffle=False) iteration.
    """
    sel = (df_full["split"] == split) & (df_full["porosity"] <= 1.0)
    return df_full.index[sel].to_numpy(dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument(
        "--output", default=None,
        help="Output directory (default: <data_root>/latents_<exp>z<z>, e.g. data/split_v2/latents_r07z4)",
    )
    ap.add_argument("--data-root", default=None, help="Override patch data root (default: from VAE config)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Encode at most N patches per split (smoke test)")
    args = ap.parse_args()

    repo = _find_repo_root()
    sys.path.insert(0, str(repo / "src"))

    device = torch.device(args.device)
    checkpoint = (repo / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)

    from poregen.experiments.train_vae import load_vae_from_checkpoint, resolve_data_root

    model, cfg, cfg_text, run_dir = load_vae_from_checkpoint(checkpoint, device)

    data_root = Path(args.data_root).resolve() if args.data_root else resolve_data_root(cfg, repo)
    index_path = data_root / "patch_index.parquet"
    df_full = pd.read_parquet(str(index_path))
    logger.info("Data root: %s  (%d indexed patches)", data_root, len(df_full))

    if args.output:
        out_path = Path(args.output)
        out_root = out_path.resolve() if out_path.is_absolute() else (repo / out_path).resolve()
    else:
        out_root = data_root / f"latents_{_default_tag(cfg, run_dir)}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_root)

    z_channels = int(cfg["model"]["z_channels"])
    latent_size = int(cfg["model"]["patch_size"]) // 4  # two 2x downsampling stages
    latent_shape = (z_channels, latent_size, latent_size, latent_size)

    # Size estimate and free-space guard: 2 float16 arrays (mu, std) per patch.
    n_planned = 0
    for split in SPLITS:
        n_split = int(((df_full["split"] == split) & (df_full["porosity"] <= 1.0)).sum())
        if args.limit is not None:
            n_split = min(n_split, args.limit)
        n_planned += n_split
    est_bytes = n_planned * 2 * int(np.prod(latent_shape)) * 2
    free_bytes = shutil.disk_usage(out_root).free
    logger.info(
        "Planned: %d patches  estimated store size %.1f GB  (free: %.1f GB)",
        n_planned, est_bytes / 1e9, free_bytes / 1e9,
    )
    if free_bytes < 2 * est_bytes:
        raise SystemExit(
            f"Aborting: free space ({free_bytes / 1e9:.1f} GB) is below 2x the "
            f"estimated store size ({est_bytes / 1e9:.1f} GB). Free up disk space "
            f"or use --limit / a different --output filesystem."
        )

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Streaming per-channel moments over the TRAIN split (float64 accumulators).
    ch_sum = torch.zeros(z_channels, dtype=torch.float64, device=device)
    ch_sumsq = torch.zeros(z_channels, dtype=torch.float64, device=device)
    ch_count = 0

    split_counts: dict[str, int] = {}

    for split in SPLITS:
        ds = _build_split_dataset(data_root, split)
        source_rows = _source_rows(df_full, split)
        assert len(source_rows) == len(ds), (
            f"[{split}] source-row alignment broken: {len(source_rows)} vs {len(ds)}"
        )

        if args.limit is not None and args.limit < len(ds):
            ds = Subset(ds, range(args.limit))
            source_rows = source_rows[: args.limit]

        n = len(ds)
        split_counts[split] = n
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        store = zarr.open_group(str(split_dir / "latents.zarr"), mode="w")
        arr_kwargs = dict(
            shape=(n, *latent_shape),
            chunks=(1, *latent_shape),
            shards=(min(2048, max(n, 1)), *latent_shape),
            dtype="float16",
        )
        mu_arr = store.create_array("mu", **arr_kwargs)
        std_arr = store.create_array("std", **arr_kwargs)

        phi_all = np.empty(n, dtype=np.float32)

        i = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding {split}", unit="batch"):
                xct = batch["xct"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=device.type == "cuda"):
                    out = model(xct, mask)

                mu = out.mu.float()
                std = torch.exp(0.5 * out.logvar.float())
                phi = mask.mean(dim=(1, 2, 3, 4))

                if split == "train":
                    ch_sum += mu.double().sum(dim=(0, 2, 3, 4))
                    ch_sumsq += mu.double().pow(2).sum(dim=(0, 2, 3, 4))
                    ch_count += mu.shape[0] * mu.shape[2] * mu.shape[3] * mu.shape[4]

                b = mu.shape[0]
                mu_arr[i : i + b] = mu.cpu().numpy().astype(np.float16)
                std_arr[i : i + b] = std.cpu().numpy().astype(np.float16)
                phi_all[i : i + b] = phi.cpu().numpy()
                i += b

        assert i == n, f"[{split}] wrote {i} rows, expected {n}"
        rate = n / max(time.time() - t0, 1e-9)
        logger.info("[%s] encoded %d patches (%.0f patches/s)", split, n, rate)

        df_split = df_full.iloc[source_rows][
            ["volume_id", "source_group", "split", "z0", "y0", "x0", "ps", "stride", "porosity"]
        ].reset_index(drop=True)
        df_split.insert(0, "source_row", source_rows)
        df_split["phi"] = phi_all
        df_split.to_parquet(str(split_dir / "index.parquet"), index=False)

        max_dphi = float(np.abs(df_split["phi"] - df_split["porosity"]).max()) if n else 0.0
        logger.info("[%s] index.parquet written  max|phi - porosity| = %.2e", split, max_dphi)

    if ch_count == 0:
        raise RuntimeError("Train split produced no patches — cannot compute norm stats.")
    ch_mean = (ch_sum / ch_count).cpu().numpy()
    ch_var = (ch_sumsq / ch_count).cpu().numpy() - ch_mean**2
    ch_std = np.sqrt(np.maximum(ch_var, 0.0))
    logger.info("Per-channel mean: %s", np.array2string(ch_mean, precision=4))
    logger.info("Per-channel std:  %s", np.array2string(ch_std, precision=4))

    metadata = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "vae_checkpoint": str(checkpoint),
        "vae_run_dir": str(run_dir),
        "vae_config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "vae_config": cfg,
        "latent_shape": list(latent_shape),
        "dtype": "float16",
        "arrays": ["mu", "std"],
        "source_patch_index": str(index_path),
        "patch_size": int(cfg["model"]["patch_size"]),
        "splits": split_counts,
        "limit": args.limit,
        "normalization": {
            "computed_over": "train",
            "per_channel_mean": ch_mean.tolist(),
            "per_channel_std": ch_std.tolist(),
        },
    }
    meta_tmp = out_root / "metadata.json.tmp"
    meta_tmp.write_text(json.dumps(metadata, indent=2))
    meta_tmp.rename(out_root / "metadata.json")
    logger.info("Wrote %s", out_root / "metadata.json")
    logger.info("Done: %s", {s: split_counts[s] for s in SPLITS})


if __name__ == "__main__":
    main()

"""Build the ldm06 latent store from a trained 3-class VAE (r08 / split_v3).

Encodes every patch of every split with the VAE encoder and stores the
posterior mean (``mu``) and posterior std (``std``) per patch, together with
the raw patch porosity ``phi``, the MATERIAL MAP at latent resolution and the
patch air fraction, plus an index that links each row back to the source
``patch_index.parquet`` row.

The encoder input layout is read off the model: the r08 variant declares
``encoder_inputs = ("xct", "label")`` and internally feeds
``cat([xct, label == 1, label == 2])``.  The material map and the air fraction
come from the SAME label memmap in the same pass — one read, one alignment, no
second script to fall out of step with the store.

* ``material.bin`` — uint8 ``(N, L, L, L)``; ``value/255`` is the fraction of
  the ``f³``-voxel cell inside the specimen ENVELOPE (``label != 2``, so pores
  count as specimen).  The map says where the specimen IS; it is 1 throughout
  the interior and fractional only where the outer surface or a drilled hole
  cuts a cell.  Pooling ``label == 0`` instead would hand the denoiser the pore
  mask at 100 µm resolution, which it would learn to upsample instead of
  generating pores.
* ``air.bin`` — float32 ``(N,)``; the fraction of the patch labelled air
  (label 2), i.e. outside the envelope.  Equal to ``1 - material.mean()`` by
  construction — it is computed from the same pooled cells.

Latents are stored RAW (un-normalised).  Per-channel normalisation stats are
computed over the train split only and stored in ``metadata.json``; the LDM
data pipeline applies them at load time (see
``poregen.diffusion.latents.LatentDataset``).

Usage
-----
python scripts/build_latent_dataset.py \\
    [--checkpoint runs/vae/<r08 run>/best.ckpt] \\
    [--output data/split_v3/latents_r08z4] \\
    [--data-root data/split_v3] \\
    [--batch-size 256] \\
    [--num-workers 4] \\
    [--limit N]            # per-split cap, for smoke tests

Before encoding, the script prints the estimated total store size and aborts
if the output filesystem has less than 2x that estimate free.

On-disk layout produced (default output: <data_root>/latents_<exp>z<z>/)
-----------------------
data/split_v3/latents_r08z4/
├── metadata.json            — VAE checkpoint + resolved config copy & sha256,
│                              latent shape, storage record, per-channel
│                              train-split norm stats, the `material` block,
│                              creation date, per-split counts
├── train/
│   ├── latents.bin          — float16 C-contiguous memmap, shape
│   │                          (N, 2C, 16, 16, 16); channels 0..C-1 = mu,
│   │                          C..2C-1 = std ("mu_then_std" packing)
│   ├── material.bin         — uint8 (N, 16, 16, 16) envelope fraction per cell
│   ├── air.bin              — float32 (N,) air fraction per patch
│   └── index.parquet        — source_row (row in patch_index.parquet), volume_id,
│                              source_group, split, z0, y0, x0, ps, stride,
│                              porosity (from source index), phi
├── val/   (same files)
└── test/  (same files)

Row i of every file in a split is the same patch.

``scripts/build_conditioning.py`` is run AFTER this script; it adds
``cond.parquet`` and the ``conditioning`` metadata block.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Scanner resolution of every volume in the dataset.  Recorded here because
# its absence has cost time twice (D32 section 7).
VOXEL_SIZE_UM = 25.0

logger = logging.getLogger(__name__)

# The r08 run that supplies the ldm06 store.  Filled in when r08 finishes; the
# glob is resolved at launch and refuses to guess between two matches.
DEFAULT_CHECKPOINT = "runs/vae/r08-run-*/best.ckpt"

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


def _resolve_checkpoint(ref: str, repo: Path) -> Path:
    """Resolve a checkpoint reference, expanding a glob to its single match."""
    path = Path(ref)
    if not path.is_absolute():
        path = repo / path
    if "*" not in ref and "?" not in ref:
        return path.resolve()
    matches = sorted(repo.glob(ref))
    if len(matches) != 1:
        raise SystemExit(
            f"--checkpoint {ref!r} matched {len(matches)} paths "
            f"({[str(m) for m in matches[:4]]}) — pass the exact checkpoint."
        )
    return matches[0].resolve()


def _build_split_dataset(data_root: Path, split: str):
    """Sequential patch dataset for one split — mirrors the backend selection
    of ``poregen.training.data.build_patch_dataloaders`` without its
    training-oriented shuffle/drop_last settings.

    The memmap backend is required: the material map is pooled from the same
    ``label`` tensor the encoder consumes, so both must come from one loader.
    """
    from poregen.dataset.loader import MemmapPatchDataset

    if not (data_root / "patches_meta.json").exists():
        raise SystemExit(
            f"{data_root} has no patches_meta.json — the ldm06 store is built "
            f"from the memmap backend (patches_xct.bin + patches_label.bin). "
            f"Run scripts/extract_patches_memmap.py first."
        )
    return MemmapPatchDataset(data_root / "patch_index.parquet", data_root, split=split)


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
        help="Output directory (default: <data_root>/latents_<exp>z<z>, e.g. data/split_v3/latents_r08z4)",
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
    checkpoint = _resolve_checkpoint(args.checkpoint, repo)

    from poregen.dataset.material import (
        air_fraction,
        encode_material_u8,
        pool_material_fractions,
    )
    from poregen.experiments.train_vae import load_vae_from_checkpoint, resolve_data_root
    from poregen.models.vae.base import CLASS_AIR, CLASS_PORE
    from poregen.training.engine import encoder_input_keys

    model, cfg, cfg_text, run_dir = load_vae_from_checkpoint(checkpoint, device)
    enc_keys = encoder_input_keys(model)
    logger.info("Checkpoint: %s  encoder inputs: %s", checkpoint, enc_keys)

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
    patch_size = int(cfg["model"]["patch_size"])
    latent_size = patch_size // 4  # two 2x downsampling stages
    latent_shape = (z_channels, latent_size, latent_size, latent_size)
    pool_factor = patch_size // latent_size

    # Size estimate and free-space guard: 2 float16 arrays (mu, std) per patch.
    n_planned = 0
    for split in SPLITS:
        n_split = int(((df_full["split"] == split) & (df_full["porosity"] <= 1.0)).sum())
        if args.limit is not None:
            n_split = min(n_split, args.limit)
        n_planned += n_split
    est_bytes = n_planned * (2 * int(np.prod(latent_shape)) * 2      # mu + std, float16
                             + latent_size ** 3                       # material.bin, uint8
                             + 4)                                     # air.bin, float32
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

        # mu_then_std packing: channels 0..C-1 = mu, C..2C-1 = std.
        latents = np.memmap(
            str(split_dir / "latents.bin"), dtype=np.float16, mode="w+",
            shape=(n, 2 * z_channels, *latent_shape[1:]),
        )
        material = np.memmap(
            str(split_dir / "material.bin"), dtype=np.uint8, mode="w+",
            shape=(n, latent_size, latent_size, latent_size),
        )
        air = np.memmap(
            str(split_dir / "air.bin"), dtype=np.float32, mode="w+", shape=(n,),
        )

        phi_all = np.empty(n, dtype=np.float32)

        i = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding {split}", unit="batch"):
                # The label stays on the CPU as well: the material map and the
                # air fraction are pooled from the SAME tensor the encoder sees,
                # so the three arrays cannot fall out of alignment.
                label_np = batch["label"].numpy()
                inputs = tuple(batch[k].to(device, non_blocking=True) for k in enc_keys)

                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=device.type == "cuda"):
                    mu_raw, logvar_raw = model.encode_moments(*inputs)

                mu = mu_raw.float()
                std = torch.exp(0.5 * logvar_raw.float())

                if split == "train":
                    ch_sum += mu.double().sum(dim=(0, 2, 3, 4))
                    ch_sumsq += mu.double().pow(2).sum(dim=(0, 2, 3, 4))
                    ch_count += mu.shape[0] * mu.shape[2] * mu.shape[3] * mu.shape[4]

                b = mu.shape[0]
                latents[i : i + b, :z_channels] = mu.cpu().numpy().astype(np.float16)
                latents[i : i + b, z_channels:] = std.cpu().numpy().astype(np.float16)
                # The specimen ENVELOPE: pores are inside it, only air is out.
                cells = pool_material_fractions(label_np != CLASS_AIR, pool_factor)
                material[i : i + b] = encode_material_u8(cells)
                air[i : i + b] = air_fraction(cells)
                phi_all[i : i + b] = (label_np == CLASS_PORE).mean(axis=(1, 2, 3),
                                                                   dtype=np.float32)
                i += b

        assert i == n, f"[{split}] wrote {i} rows, expected {n}"
        for arr in (latents, material, air):
            arr.flush()
        mean_air = float(np.asarray(air).mean()) if n else 0.0
        mean_mat = float(np.asarray(material).mean()) / 255.0 if n else 0.0
        del latents, material, air
        rate = n / max(time.time() - t0, 1e-9)
        logger.info(
            "[%s] encoded %d patches (%.0f patches/s)  mean envelope=%.4f  mean air=%.4f",
            split, n, rate, mean_mat, mean_air,
        )

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
        "storage": {
            "format": "memmap",
            "file": "latents.bin",
            "dtype": "float16",
            "pack_scheme": "mu_then_std",
        },
        "source_patch_index": str(index_path),
        "patch_size": patch_size,
        "material": {
            "files": {"material": "material.bin", "air": "air.bin"},
            "alignment": "row i of material.bin and air.bin is row i of index.parquet",
            "material_dtype": "uint8",
            "material_shape": [latent_size] * 3,
            "material_encoding": (
                f"value/255 = fraction of the {pool_factor}^3-voxel cell inside the "
                f"specimen ENVELOPE (voxel label != 2, so pores count as specimen).  "
                f"1 throughout the interior, fractional only where the outer surface "
                f"or a drilled hole cuts a cell.  It says WHERE THE SPECIMEN IS, not "
                f"how much of it is solid: pooling label == 0 would be the pore mask "
                f"at {pool_factor}^3 resolution."
            ),
            "air_dtype": "float32",
            "air_definition": (
                "fraction of the patch labelled air (voxel label 2) == "
                "1 - material.mean(), computed from the same pooled cells"
            ),
            "phi_definition": "fraction of the patch labelled pore (voxel label 1)",
        },
        "voxel_size_um": VOXEL_SIZE_UM,
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

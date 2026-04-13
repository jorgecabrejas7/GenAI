# PoreGen — Latent Diffusion for Synthetic XCT Pore Generation

> Generate realistic 3-D pore structures in composite materials by training a
> latent diffusion model on X-ray computed tomography (XCT) volumes and their
> pore masks.

---

## Quick Start

```bash
# 1.  Install (editable, core only)
pip install -e ".[dev]"

# 2.  Install notebook extras (for Jupyter training)
pip install -e ".[notebook]"

# 3.  Build dataset (adjust counts to your volume inventory)
build_dataset \
    --raw_root ./raw_data \
    --out_root ./data/split_v1 \
    --patch_size 64 --stride 32 \
    --n_train 40 --n_val 5 --n_test 5 \
    --seed 123

# 4.  Run tests
pytest tests/ -v
```

---

## Architecture

```
GenAI/
├── onlypores.py                 # pore segmentation (used by dataset build)
├── pyproject.toml
├── src/poregen/
│   ├── dataset/
│   │   ├── io.py                # volume discovery, TIF loading, Zarr I/O
│   │   ├── splits.py            # volume-level train/val/test splits
│   │   ├── patch_index.py       # patch coords, integral-volume porosity
│   │   ├── build_dataset.py     # CLI pipeline
│   │   └── loader.py            # PyTorch PatchDataset
│   ├── models/
│   │   └── vae/
│   │       ├── base.py          # VAEConfig, VAEOutput dataclasses
│   │       ├── registry.py      # build_vae(name, **kwargs)
│   │       ├── conv_vae.py      # ConvVAE3D (simple baseline)
│   │       └── unet_vae.py      # UNetVAE3D (skip-connection decoder)
│   ├── losses/
│   │   ├── recon.py             # L1 / MSE / Charbonnier
│   │   ├── mask.py              # BCE + Dice / Tversky
│   │   ├── kl.py                # KL divergence + β schedule + free-bits
│   │   └── total.py             # compose total loss
│   ├── metrics/
│   │   ├── recon.py             # MSE / MAE / PSNR / sharpness proxy
│   │   ├── seg.py               # Dice / IoU / precision / recall / F1
│   │   └── latent.py            # active units, KL per channel, stats
│   ├── training/
│   │   ├── engine.py            # train_step / eval_step / train_loop
│   │   ├── checkpoint.py        # save / load checkpoints
│   │   ├── device.py            # GPU selection, AMP dtype, GradScaler
│   │   └── seed.py              # deterministic seeding
│   ├── configs/
│   │   └── example_vae.yaml     # default model/loss/training config
│   ├── vae/                     # thin re-export → poregen.models.vae
│   └── diffusion/               # placeholder (future)
├── notebooks/
│   ├── 10_train_vae.ipynb       # training skeleton
│   └── 11_eval_vae.ipynb        # eval + visualisation skeleton
├── scripts/
│   └── build_dataset.sh
├── tests/
│   ├── test_vae_output_shapes.py
│   ├── test_losses_smoke.py
│   ├── test_volume_split_counts.py
│   ├── test_patch_coords_count.py
│   ├── test_integral_porosity.py
│   └── test_dataset_loader_shapes.py
└── raw_data/                    # your volumes (not tracked in git)
```

---

## Dataset Pipeline

### Concepts

| Concept | Detail |
|---|---|
| **Storage** | Per-volume Zarr groups under `volumes.zarr/`, Blosc(zstd) compressed |
| **Index** | Single Parquet file (`patch_index.parquet`) with one row per patch |
| **Splits** | **Volume-level only** — you specify exact counts `--n_train`, `--n_val`, `--n_test` |
| **Porosity** | Computed using a 3-D integral volume (summed-area table) for O(1) per-patch queries |

### Output Structure

```
data/split_v1/
├── volumes.zarr/
│   ├── <volume_id>/
│   │   ├── xct   (uint8, chunked)
│   │   └── mask  (uint8 {0,1}, chunked)
│   └── ...
├── patch_index.parquet
└── splits.json
```

### CLI Reference

```
build_dataset --help
```

| Flag | Default | Description |
|---|---|---|
| `--raw_root` | `./raw_data` | Root of raw TIFF volumes |
| `--out_root` | `./data/split_v1` | Output directory |
| `--patch_size` | `64` | Cubic patch side length |
| `--stride` | `32` | Stride between patches |
| `--chunk_size` | `32,32,32` | Zarr chunk size |
| `--clevel` | `3` | Blosc zstd compression level |
| `--n_train` | *required* | Number of training volumes |
| `--n_val` | *required* | Number of validation volumes |
| `--n_test` | *required* | Number of test volumes |
| `--seed` | `123` | Random seed for split |
| `--limit_volumes` | all | Cap total volumes (for smoke tests) |

---

## VAE Experiments (notebook-first)

Training runs inside Jupyter notebooks on interactive HPC nodes. All logic
lives in small library functions; notebooks just wire them together.

### GPU selection

```bash
# Use a specific GPU
CUDA_VISIBLE_DEVICES=0 jupyter lab

# Or select in code
from poregen.training import select_device
device = select_device(gpu_id=0)
```

### Run training

1. Open `notebooks/10_train_vae.ipynb`
2. Edit `EXP_NAME` and optionally the YAML config
3. Run all cells — checkpoints are saved to `runs/vae/<exp_name>/`

### Available models

| Name | Registry key | Description |
|---|---|---|
| **ConvVAE3D** | `"conv"` | Simple stride-2 Conv3d encoder + ConvTranspose3d decoder. Fast, stable baseline. |
| **UNetVAE3D** | `"unet"` | Same encoder but decoder uses skip connections (UNet-style). Sharper reconstructions, diffusion-friendly. |

Both output **logits** for XCT and mask heads. Sigmoid is applied explicitly
in loss and metric functions for consistency.

### 2×A6000 note

> [!NOTE]
> The training engine supports AMP (bfloat16 on Ampere+) out of the box.
> For multi-GPU training, wrap your model with `torch.nn.DataParallel` or
> launch with `torchrun` + `DistributedDataParallel`. A full distributed
> trainer is not included — the codebase is designed for interactive,
> single-GPU notebook workflows with optional scale-up.

---

## HPC Workflow

1. **Copy raw data** to node-local NVMe / scratch storage.
2. Run `build_dataset` as a **batch job** — each volume is ~1 GB and
   `onlypores()` is CPU-heavy.
3. The output Zarr store + Parquet index stays on local storage for
   maximum I/O throughput during training.
4. Use Jupyter / interactive sessions for **training** and **evaluation**
   via the notebook skeletons in `notebooks/`.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

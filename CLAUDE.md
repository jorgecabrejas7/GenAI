# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (environment assumed to be set up with mamba/conda + PyTorch)
pip install -r requirements.txt
pip install -e ".[dev]"

# Tests
pytest tests/                          # full suite
pytest tests/test_losses_smoke.py      # single file
pytest tests/ -k "test_vae_output"     # filter by name

# Known pre-existing failures (do not fix unless explicitly asked):
# tests/test_losses_smoke.py::TestLossesSmoke::test_all_components_present
# tests/test_latent_metrics.py::test_active_units_counts_collapsed_channels
# tests/test_recon_metrics.py  (3 failures)

# Dataset construction
build_dataset --help                   # CLI entry point
# or: python -c "from poregen.dataset.build_dataset import main; main()"

# Training
python scripts/train_vae.py r05/base              # launch by experiment id
experiments list                                   # list all defined experiments
experiments clone r05/base r06/my_variant         # create new experiment extending r05

# TensorBoard
tensorboard --logdir runs/vae/
```

## Deployment target

**NVIDIA GB10 GPU (DGX Spark, 128 GB unified memory).** Code runs on that machine, not locally. Key implications:
- `torch.compile(mode="max-autotune", dynamic=False)` is the right mode — shapes are static (64³ patches)
- `autocast_dtype` will be `bfloat16` (Blackwell supports it natively)
- `num_workers=4` in `configs/machines/dgx_spark.yaml` — more workers cause heap copies in GB10 unified memory
- Discriminator intentionally runs in `float32` (spectral norm power iteration is less accurate in bfloat16) — do not wrap it in autocast

## Config system

Experiments are defined as hierarchical YAML under `configs/experiments/`. Resolution order:

1. `extends: parent/variant` — deep-merge parent config first
2. `components:` — merge named component YAMLs (from `configs/{models,losses,datasets,training}/`)
3. Body fields — overwrite what was inherited
4. `overrides:` — final deep-merge (highest priority)

Entry point: `poregen.configuration.resolve_experiment("r05/base")` returns a `ResolvedExperiment` with `.cfg` (the merged dict) and `.source_chain`. After resolution, `_normalise_cfg` fills `runtime.*` defaults (runs_root, checkpoint policy, resume mode, run-name format).

Component YAMLs live under `configs/` subdirectories. The resolved config always has four required top-level keys: `model`, `loss`, `training`, `data`.

## VAE model family

All VAEs are registered with `@register_vae("key")` and constructed via `build_vae("key", **kwargs)`. The registry is in `src/poregen/models/vae/registry.py`.

**Active architecture (R03–R05):** `v2.conv_noattn_dualbranch` (R04/R05) or `v2.conv_noattn` (R03).

Data flow for `v2.conv_noattn`:
```
XCT (B,1,64³) → encoder (2× down_block_v2) → (B,64,16³) → to_mu / to_logvar → z (B,16,16³)
z → decoder (2× up_block_v2) → (B,32,64³) → xct_head / mask_head → logits (B,1,64³)
```

The **encoder receives XCT only** (not the mask). The `mask` argument to `forward()` is a reconstruction target only. The mask head is a segmentation head that predicts pore density from XCT-latent features.

All model outputs are **logits** — sigmoid/threshold is applied in loss and metric functions, never inside the model.

`VAEOutput` (from `models/vae/base.py`) is a dataclass: `{xct_logits, mask_logits, mu, logvar, z}`.

## Training pipeline

`src/poregen/training/engine.py` contains three key functions:

- **`train_step`** — forward, adversarial generator pass (if discriminator enabled), backward, unscale, per-module grad norms, clip, optimizer step. Returns `(losses_dict, grad_norm, latent_moments, module_grad_norms)`. `latent_moments` contains GPU tensors (no `.cpu()` until the log guard fires).
- **`eval_step`** — returns `(losses_dict, VAEOutput, xct_dev, mask_dev)`. Returns device tensors for xct/mask to avoid double H→D transfer in the caller.
- **`train_loop`** — the main loop. When `compile=True`, compiles model + discriminator + loss_fn with `max-autotune, dynamic=False`. Calls `_run_eval` every `eval_every` steps and `_save_patch_samples` every `sample_every` steps. Checkpoints via async background thread.

Eval accumulators (`mae_acc`, `sharp_*_acc`, porosity lists) stay on GPU as `.detach()` tensors throughout the loop; `.item()` fires once post-loop.

## Loss composition

`compute_total_loss(output, batch, step, cfg)` in `src/poregen/losses/total.py`:

```
total = xct_weight × recon_loss(xct_logits, xct)
      + mask_bce_weight × (BCE or focal) + mask_dice_weight × (Dice or Tversky)
      + beta(step) × KL(mu, logvar, free_bits)
```

- `recon_loss` is selected by `cfg["loss"]["xct_loss_type"]` (`l1 | mse | charbonnier`)
- KL uses **free-bits** clamping: per-channel KL is clamped to `max(kl_ch, free_bits)` before summing — prevents posterior collapse without disabling the KL signal entirely
- `beta` ramps from 0 → `kl_max_beta` over `kl_warmup_steps` steps
- `pos_weight` for BCE should be pre-allocated once as a device tensor and passed every call
- `combined_mask_loss` and `focal_loss` accept an optional `sigmoid=` kwarg — pass `torch.sigmoid(logits)` to skip redundant computation when it's already available

## Data pipeline

Raw TIFFs → `build_dataset` script → `data/<split>/volumes.zarr/` + `patch_index.parquet`.

`PatchDataset` loads from Zarr via a Parquet index. Patches are `64³` float32, normalised to `[0, 1]` (XCT) or `{0, 1}` (mask). Zarr handles are opened once per dataset instance; `zarr_worker_init_fn` ensures each DataLoader worker gets an isolated copy after `fork()`.

`build_patch_dataloaders(cfg, data_root)` builds all three splits. The training DataLoader uses `shuffle=True, drop_last=True`; test uses `shuffle=False`.

## Experiment run structure

Runs land in `runs/vae/<experiment>/<timestamp>-<run_index>/`. Each run contains:
- `log.jsonl` — per-step training metrics
- `metrics.jsonl` — per-eval validation/test metrics  
- `resolved_config.yaml` — the fully merged config as executed
- `run_metadata.json` — git commit, machine info, environment
- `tensorboard/` — TensorBoard event files
- `checkpoints/latest.ckpt`, `checkpoints/best.ckpt`

On resume, `run_metadata.json` is updated and `log.jsonl` is pruned to the resume step to prevent duplicates.

## Key invariants

- **Porosity-MAE < 0.005** is the primary success metric (tracked as `val/porosity_mae`).
- `kl_collapsed_fraction` should stay low — if it spikes, the latent is collapsing.
- The **discriminator always runs in float32** — this is intentional, do not change.
- `latent_channel_moments` returns GPU tensors — downstream callers (`merge_latent_channel_moments`, `active_units_from_moments`) handle them correctly.
- `eval_step` returns `(losses, output, xct_dev, mask_dev)` — the device tensors should be reused rather than re-transferred in the calling `_run_eval`.

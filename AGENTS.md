# AGENTS.md

Engineering philosophy for this project. Applies to any agent working in this repo, on top of whatever project-specific rules live in `CLAUDE.md`.

## Code

- No backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, feature flags, or migrations.
- Simplest implementation that fully meets the current requirement. No speculative abstractions, config, or indirection for hypothetical futures.
- Grow in layers: get the smallest end-to-end version working, then add capabilities on top of something that already runs. Never trade a working product for unfinished complexity.
- Keep components modular with clearly separated concerns.
- Prefer established, well-maintained libraries over reimplementing common functionality. Check what's already a dependency, and check the library's actual capabilities/docs before assuming it can't do something.
- Decide architecture for the long term — don't ship a stopgap that's "meant to be replaced later."

## Working style

- Match scope to what was asked. Make routine judgment calls yourself; check in only when different readings of the request would produce materially different work. Don't quietly narrow, widen, or transform the task.
- Don't add verification steps beyond what the task needs (extra review passes, re-checks, "double-check this"). Do the work correctly the first time; verify with tests/build/lint when those exist, not with redundant re-reading.
- Delegate to a subagent only for large, genuinely independent, parallelizable work. Don't spawn one to double-check work you just did yourself.
- When you catch and fix your own mistake, just fix it — don't narrate the correction unless it changes what the user sees or decides.

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

# Training
python scripts/train_vae.py r05/base              # launch by experiment id
experiments list                                   # list all defined experiments
experiments clone r05/base r06/my_variant         # create new experiment extending r05

# TensorBoard
tensorboard --logdir runs/vae/
```

## Deployment target

**NVIDIA GB10 GPU (DGX Spark, 128 GB unified memory).** Code runs on that machine, not locally.

- `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)` — shapes are static (64³ patches); CUDA-graph trees break on the train/eval mode switch (BN buffers become graph outputs) and on the twice-per-step discriminator, so graph replay stays off
- `autocast_dtype` is `bfloat16` (Blackwell supports it natively)
- `num_workers=4` in `configs/machines/dgx_spark.yaml` — more workers cause heap copies in GB10 unified memory
- Discriminator intentionally runs in `float32` (spectral norm power iteration is less accurate in bfloat16) — do not wrap it in autocast

## Config system

Experiments are hierarchical YAML under `configs/experiments/`. Resolution order:

1. `extends: parent/variant` — deep-merge parent config first
2. `components:` — merge named component YAMLs (from `configs/{models,losses,datasets,training}/`)
3. Body fields — overwrite what was inherited
4. `overrides:` — final deep-merge (highest priority)

Entry point: `poregen.configuration.resolve_experiment("r05/base")` → `ResolvedExperiment` with `.cfg` and `.source_chain`. `_normalise_cfg` fills `runtime.*` defaults after resolution. Resolved config always has four top-level keys: `model`, `loss`, `training`, `data`.

## VAE model & training pipeline

See `docs/vae_architecture.md` for the encoder/decoder data flow and `train_step`/`eval_step`/`train_loop` internals.

## Loss composition gotchas

`compute_total_loss` in `src/poregen/losses/total.py` — see `docs/metrics_guide.md` for the full formula and health ranges per term.

- `recon_loss` type is picked by `cfg["loss"]["xct_loss_type"]` (`l1 | mse | charbonnier`)
- KL uses free-bits: per-channel KL is clamped to `max(kl_ch, free_bits)` before summing — prevents posterior collapse without disabling the KL signal
- `beta` ramps 0 → `kl_max_beta` over `kl_warmup_steps`
- `pos_weight` for BCE must be a pre-allocated device tensor, passed every call — don't reallocate per step
- `combined_mask_loss`/`focal_loss` take an optional `sigmoid=` kwarg — pass `torch.sigmoid(logits)` when already computed, to skip redundant work

## Data pipeline

Raw TIFFs → `build_dataset` → `data/<split>/volumes.zarr/` + `patch_index.parquet`. `PatchDataset` loads via the Parquet index; patches are `64³` float32, normalised to `[0, 1]` (XCT) or `{0, 1}` (mask). Zarr handles open once per dataset instance; `zarr_worker_init_fn` gives each DataLoader worker an isolated copy after `fork()`. `build_patch_dataloaders(cfg, data_root)` builds all three splits — train uses `shuffle=True, drop_last=True`, test uses `shuffle=False`.

## Experiment run structure

Runs land in `runs/vae/<experiment>/<timestamp>-<run_index>/`, each containing `log.jsonl`, `metrics.jsonl`, `resolved_config.yaml`, `run_metadata.json`, `tensorboard/`, `checkpoints/latest.ckpt` + `best.ckpt`. On resume, `run_metadata.json` is updated and `log.jsonl` is pruned to the resume step to prevent duplicates.

## Key invariants

- **Porosity-MAE < 0.005** is the primary success metric (`val/porosity_mae`).
- `kl_collapsed_fraction` should stay low — a spike means the latent is collapsing.
- The **discriminator always runs in float32** — intentional, do not change.
- `latent_channel_moments` returns GPU tensors — `merge_latent_channel_moments`/`active_units_from_moments` handle them correctly as-is.
- `eval_step` returns `(losses, output, xct_dev, mask_dev)` — reuse those device tensors in `_run_eval` rather than re-transferring.

# Architecture

How the project is built. For commands and the deployment target see
[DEVELOPMENT.md](DEVELOPMENT.md); for the file-by-file map see
[CODEMAP.md](CODEMAP.md). Design rationale and experiment history live in the
vault at `/home/jorgecabrejas/Dev/PhDTracker`.

## Config system

Experiments are hierarchical YAML under `configs/experiments/`. Resolution order:

1. `extends: parent/variant` — deep-merge parent config first
2. `components:` — merge named component YAMLs (from `configs/{models,losses,datasets,training}/`)
3. Body fields — overwrite what was inherited
4. `overrides:` — final deep-merge (highest priority)

Entry point: `poregen.configuration.resolve_experiment("r05/base")` →
`ResolvedExperiment` with `.cfg` and `.source_chain`. `_normalise_cfg` fills
`runtime.*` defaults after resolution. A resolved config always has four
top-level keys: `model`, `loss`, `training`, `data`.

## Data pipeline

Raw TIFFs → `build_dataset` → `data/<split>/volumes.zarr/` + `patch_index.parquet`.
`PatchDataset` loads via the Parquet index; patches are `64³` float32, normalised
to `[0, 1]` (XCT). Zarr handles open once per dataset instance;
`zarr_worker_init_fn` gives each DataLoader worker an isolated copy after `fork()`.
`build_patch_dataloaders(cfg, data_root)` builds all three splits — train uses
`shuffle=True, drop_last=True`, test uses `shuffle=False`.

### split_v3 — the current dataset

`data/split_v3` is built by `scripts/build_split_v3.py` and replaces `split_v2`
for the r08 VAE and ldm06. Its `volumes.zarr` is a **symlink** to the split_v2
store; nothing is copied and `data/split_v2` is never written. Three things
differ:

- **Drilled holes removed.** Every coupon has three ~200-voxel registration
  through-holes. They are `False` in `sample_mask` and `0` in `mask`, so they
  read as porosity 0 while being ~95 % air — 6 065 train patches of split_v2
  sit inside one, and they are the only interior source of large air in the
  dataset. `poregen.dataset.holes.detect_holes` finds them from a z-MINIMUM
  projection of `~sample_mask` (a maximum projection merges them into the
  exterior on the Pegaso coupons), dilates them by 32 voxels, and every patch
  whose `(y, x)` footprint touches one is dropped.
- **Split by panel, not by coupon.** test = every coupon of panel `Na_05` plus
  `Juan_Ignacio_probetas_8`; val = every coupon of `Na_08` plus
  `Juan_Ignacio_probetas_12`; train = the rest. The 24 Pegaso coupons are a
  single panel, so they can only ever be train. `patch_index.parquet` carries
  `panel_id` and `air_fraction` beside the split_v2 columns.
- **Three-class voxel label.** `patches_label.bin` stores 0 material, 1 pore,
  2 air (`sample_mask == 0`); air takes precedence over pore. There is no
  `patches_mask.bin` any more. Both loaders return `label` (int64, no channel
  dim, ready for cross-entropy) and the derived binary pore `mask`
  (`label == 1`, `(1, ps, ps, ps)` float32), which is what the current VAE
  losses consume.

Latents for the LDM stack live in a memmap binary store
(`data/split_v2/latents_r07z4/`) with sibling arrays for the material map and air
fraction; see `src/poregen/dataset/material.py`.

## VAE model & training pipeline

See [vae_architecture.md](vae_architecture.md) for the encoder/decoder data flow
and the `train_step` / `eval_step` / `train_loop` internals.

### Decoder output contract

The two decoder heads are **not** symmetric, despite both being called "heads":

- **XCT head** (`VAEOutput.xct_out`) emits the grey level in `[0, 1]` — the same
  scale as `xct / 255`. `compute_total_loss` regresses it directly
  (L1/MSE/Charbonnier). It is **not** a logit: decode it with
  `decode_xct()` / `decode_xct_u8()`, which clamp. Applying a sigmoid squashes
  the output into `[0.5, 0.731]` and destroys contrast.
- **Mask head** (`VAEOutput.mask_logits`) genuinely is a logit (BCE-with-logits);
  decode it with `decode_mask()`, which applies the sigmoid.

All three helpers live in `src/poregen/models/vae/base.py`. Every decode site
must use them so the behaviour cannot drift apart again.

## Loss composition gotchas

`compute_total_loss` in `src/poregen/losses/total.py` — see
[metrics_guide.md](metrics_guide.md) for the full formula and health ranges per term.

- `recon_loss` type is picked by `cfg["loss"]["xct_loss_type"]` (`l1 | mse | charbonnier`)
- KL uses free-bits: per-channel KL is clamped to `max(kl_ch, free_bits)` before
  summing — prevents posterior collapse without disabling the KL signal
- `beta` ramps 0 → `kl_max_beta` over `kl_warmup_steps`
- `pos_weight` for BCE must be a pre-allocated device tensor, passed every call —
  don't reallocate per step
- `combined_mask_loss`/`focal_loss` take an optional `sigmoid=` kwarg — pass
  `torch.sigmoid(logits)` when already computed, to skip redundant work

## Experiment run structure

Runs land in `runs/vae/<experiment>/<timestamp>-<run_index>/` (and `runs/ldm/…`
for diffusion runs), each containing `log.jsonl`, `metrics.jsonl`,
`resolved_config.yaml`, `run_metadata.json`, `tensorboard/`,
`checkpoints/latest.ckpt` + `best.ckpt`. On resume, `run_metadata.json` is
updated and `log.jsonl` is pruned to the resume step to prevent duplicates.

Analysis outputs live in `runs/campaigns/<NN>-<name>/`, one campaign per
question, each with a `README.md` and a vault note; see
`runs/campaigns/INDEX.md`.

## Key invariants

- **Porosity-MAE < 0.005** is the primary success metric (`val/porosity_mae`).
- `kl_collapsed_fraction` should stay low — a spike means the latent is collapsing.
- The **discriminator always runs in float32** — intentional, do not change.
- `latent_channel_moments` returns GPU tensors — `merge_latent_channel_moments`/
  `active_units_from_moments` handle them correctly as-is.
- `eval_step` returns `(losses, output, xct_dev, mask_dev)` — reuse those device
  tensors in `_run_eval` rather than re-transferring.
- The XCT head is not a logit (see the decoder output contract above).

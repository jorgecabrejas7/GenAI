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

How every root was produced, and how to rebuild each one, is in
[dataset_provenance.md](dataset_provenance.md). Each data root also carries a
short `BUILD.md` pointing there. **`volumes.zarr` is a symlink in both v2 and
v3** — `split_v3 -> split_v2 -> split_v1` — and `data/split_v1/volumes.zarr` is
the only real store, so any cleanup of a derived root must leave that link
alone.

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
(`data/split_v3/latents_r08z8/`) with sibling arrays for the material map and
air fraction, all written by one pass of `scripts/build_latent_dataset.py`; see
`src/poregen/dataset/material.py`.

## LDM conditioning contract

The denoiser is `UNet3DDenoiser`. Nothing in it is switchable: every signal
below is always present. An ldm05-era switch that no config ever flipped was
only a way for training and generation to disagree.

**Concatenated at the input** (155 channels at the live `z_channels = 8`; the rows below are in terms of `C`, so the total follows the chosen rung):

| Input | Channels | Meaning |
|---|---|---|
| `z_t` | `C` | the noisy latent at step `t` |
| `cond_orient` | 2 | `(cos2θ, sin2θ)` ply-orientation depth profile |
| `cond_material` | 1 | specimen-envelope fraction per latent cell (`label != 2`) |
| `nb_latents` | `6·C` | the six whole face-adjacent neighbour latents |
| availability | `6·8` | learned embedding of OOB / EXISTS / UNKNOWN per face |
| `nb_t` | `6·8` | sinusoidal embedding of each neighbour's own noise level |

**FiLM / AdaGN scalars**, summed with the timestep embedding: `cond_por`
(standardised `log(φ + 1e-3)` of the FULL-PATCH `φ = pore / patch_size³`, air
included — see "Material porosity in, full-patch φ out"; with a learned null
token for CFG),
`cond_depth`, `cond_dist6` through one MLP over the whole 6-vector, and the
availability-masked pool of the neighbour latents. There is no global-porosity
input — the per-patch porosity field carries volume-level control.

### Six per-face distances

`cond_dist6` is the gap from each of the patch's own six faces to the matching
face of the specimen box, capped at 64 voxels and normalised, ordered
`(z-, z+, y-, y+, x-, x+)` by `conditioning.DIST6_DIRS`. ldm05 used a single
`cond_dist` — the distance from the patch CENTRE to the NEAREST outer face over
all three axes — which cannot say *which* side the specimen ends on, so a patch
under the top surface and a patch against a side wall asked for the same thing.
`conditioning.dist6_from_box` is the one definition; the training builder and
the sampler both call it.

### Neighbours are sampled, then noised

The store serves each neighbour's CLEAN posterior — the mean AND the std, both
read from the neighbour's own store row and put through the same per-channel
affine map as the target. The batch therefore carries `nb_latents` (6,C,…) and
`nb_std` (6,C,…) next to `z` and `std`.

The training step draws each EXISTS neighbour before it noises it
(`ldm_engine.sample_neighbours`): `mu + sigma*eps`, fresh eps, exactly how the
target is drawn under `data.latent_mode: sampled`. Conditioning on posterior
MEANS while regressing a posterior SAMPLE was a real defect — the neighbour
input was short of `E[sigma^2]` per cell, a gap that does not exist at
generation time, where every neighbour is a real latent. OOB and UNKNOWN
neighbours carry no posterior and are left alone; `noise_neighbours` zeroes
them anyway.

Then `ldm_engine.noise_neighbours` draws a timestep per item and per face —
with probability `nb_t_mix` the target's own `t`, otherwise `Uniform{0..t}` —
and `q_sample`s each neighbour to it, passing `nb_t` to the denoiser. Separately,
with probability `drop_nb` an item loses all six at once: availability UNKNOWN,
latents zero, `nb_t` zero. That is exactly the neighbour null arm of the nested
CFG, so training and `DDIMSampler` share one definition of "no neighbour
information".

This is what makes joint sampling possible at all. At generation time a
neighbour is never clean — inside a chunk it is the canvas at the current
timestep, and in a finished chunk it is a clean latent re-noised to that same
timestep. A denoiser trained only on clean neighbours meets an input
distribution it has never seen on its very first sampling step, which is why
ldm05's joint mode had to drop neighbour conditioning entirely and its `s_nb`
guidance arm was inert.

Availability has three states: **OOB** (the specimen ends at that face),
**EXISTS**, and **UNKNOWN** (a neighbour exists but is not resolved yet). The
store only ever emits EXISTS and OOB; UNKNOWN comes from `drop_nb` and from the
chunks the sampler has not reached.

`VolumeGenerator.neighbour_mode` decides which of those a window may see. The
production value is `"canvas"`, described above. `"unknown"` forces every
in-bounds face to the CFG null, which with one chunk over the whole volume is
exactly the ldm05 joint sampler; `"reference"` reads every in-bounds face from a
canvas of real encoded material, which is teacher forcing. Both are measurement
arms of eval_v4's `assembly_modes` and neither is ever a production path. OOB
survives in all three: the canvas edge is geometry, and `cond_dist6` carries the
same fact, so removing it would change more than the neighbour content.

## Diffusion objective and the terminal step

`DDPMSchedule` owns what the denoiser regresses. Two keys under
`noise_schedule` decide it, and every entry point builds the schedule with
`DDPMSchedule.from_cfg(cfg, device)` so the two cannot disagree.

| Key | Values | Meaning |
|---|---|---|
| `objective` | `v` (ldm06/base) / `eps` (ldm06/eps ablation) | ε-prediction, or the velocity target `v = sqrt(ᾱ)·ε − sqrt(1−ᾱ)·x₀` (Salimans & Ho 2022) |
| `zero_terminal_snr` | bool | rescale sqrt(ᾱ) so its last entry is EXACTLY 0 (Lin et al. 2024, Alg. 1) |

Nothing outside the schedule branches on the objective. `training_target`
names what the step regresses, `predict_x0` / `predict_eps` convert a model
output back, and `ddim_step` is written on (x̂₀, ε̂) so there is one reverse
process rather than one per parameterisation.

**The terminal step is why ldm06 trains on v, and the usual justification does not
apply here.** Lin et al. attack a cosine schedule that derives ᾱ as
`cumprod(1 − clamp(β))`, which leaves a terminal `sqrt(ᾱ_T) ≈ 0.068` — real
signal the model still sees on its last training step but never on its first
sampling step. This schedule takes ᾱ straight from the cosine `f`, where
`f(T) = cos(π/2)²` is already zero to float precision. **Measured: the rescale
moves sqrt(ᾱ) by at most 6.12e-17 anywhere.** There is no leak to close.

What *is* broken is the ε form at that step. With `sqrt(ᾱ_T) = 6.12e-17`,
`x̂₀ = (x_t − sqrt(1−ᾱ)·ε̂)/sqrt(ᾱ)` divides by the `1e-8` guard clamp and
returns x̂₀ of order 1e8, which the ±10 clamp saturates — so the first DDIM
step of every ldm06 chain started from a clamp artefact, not a prediction.
The v form recovers `x̂₀ = sqrt(ᾱ)·x_t − sqrt(1−ᾱ)·v̂` with no division and
stays order 1. `DDPMSchedule` therefore refuses `zero_terminal_snr` with
`objective: eps`: there the division is by exactly zero.

### Where a DDIM step lands

**Index convention.** `alphas_cumprod[t]` is ᾱ_{t+1} and
`alphas_cumprod_prev[t]` is ᾱ_t. A network call at index `t` reads a state at
`alphas_cumprod[t]` — that is what `q_sample` builds for index `t` at training
time, and what `predict_x0` inverts. So the state `ddim_step` produces for the
call at `t_prev` must sit at `alphas_cumprod[t_prev]`, and the sampler grid's
final `t_prev = 0` closes the chain (no call follows) with the clean level
ᾱ = 1, returning x̂₀ exactly.

Until 2026-09-08 the step gathered `alphas_cumprod_prev[t_prev]` instead. That
is ᾱ_{t_prev}, one index of the 1000-step ladder below the ᾱ_{t_prev+1} the
next call assumes, so every intermediate state sat at a slightly wrong noise
level. Size on the ldm06 schedule: up to **1.6e-3 in sqrt(ᾱ)**, and — where it
hurts most — **3.9 % of sqrt(1−ᾱ) at `t_prev = 19`**, the bottom rung of a
50-step ladder, because there the remaining noise is small and a fixed absolute
error is a large relative one. The terminal step was already correct
(`alphas_cumprod_prev[0] = ᾱ_0 = 1`), which is why the "DDIM reaches x0" tests
never saw it: an oracle denoiser returns the same x̂₀ whatever level its input
is at. `tests/test_vpred_schedule.py` now drives the step with an oracle whose
answer PINS the level (x̂₀ = 1, ε̂ = 0) and reads the landing point back as a
noise level.

### Guidance in objective space

`DDIMSampler.predict_out` returns the model's RAW output and the nested CFG
combines the three arms there. That is valid for either objective: at a fixed
`t` the map v ↔ ε is affine with shared coefficients, so an affine combination
of arms commutes with it. The conversion happens once, in `ddim_step`.

`guidance.cfg_rescale` (φ, Lin et al. §3.4) is off by default. A guidance
scale above 1 is an extrapolation, and extrapolation inflates the standard
deviation of the combined prediction, which decodes over-exposed;
`rescale_guidance` scales the guided output back to the conditional arm's own
per-item standard deviation and interpolates by φ. At `s_por = s_nb = 1` it
would be a no-op, so it only earns its cost alongside a guidance sweep.

## Decoded-space auxiliary loss (ldm06/aux)

The latent objective is blind to any error that leaves the residual unchanged,
and three of the defects the diagnostics keep reporting are exactly that: air
placed inside the specimen envelope, a delivered porosity that ignores
`cond_por`, and dark regions the class head calls material. Two latents can
differ in all three and carry the same ε (or v) residual.

`loss.decoded.enabled` puts the frozen r08 VAE in the training graph and
scores the decode (precedent: Berrada et al. 2025, arXiv:2411.04873). Four
terms, in `src/poregen/losses/decoded.py`, each logged separately as
`train/aux_*` — one summed number could not say which defect moved:

| Term | What it measures |
|---|---|
| `air_outside_material` | mean p(air) where the upsampled `cond_material` is `> 0.99`; partially-filled cells are surface or drilled-hole cells and their air is legitimate |
| `pore_dice` | soft Dice of p(pore) against the source patch's real pore mask |
| `porosity_consistency` | \|mean p(pore) − φ\| on the raw scale — `gen/por_cond_mae` made differentiable |
| `grey_agreement` | p(pore)·relu(grey − 182/255) + p(material)·relu(182/255 − grey); both heads decode from ONE latent, so they cannot legitimately disagree. Air is unscored — it is dark AND correct |

Three constraints make it affordable and safe:

- **Lowest quartile of `t` only** (`t_max_frac` 0.25). x̂₀ from a high-t step is
  a blur with no pore structure to score, so every term would be measuring the
  schedule instead of the model.
- **A hard sub-batch cap** (`decoded_max_items` 32), taking the LOWEST-t items.
  Measured on the real r08 decoder, a 64³ decode with a live autograd graph
  holds **0.33 GB of intermediate activations per item in float32, ~0.20 GB
  under bf16 autocast** — decoding a whole 256 batch would need ~84 / ~51 GB on
  top of the denoiser.
- **A frozen VAE, in both senses.** Its parameters carry `requires_grad_(False)`
  so nothing accumulates on them, and the decode runs in `eval()`. The second
  is not hygiene: the r08 decoder is BatchNorm3d, and a decode in train mode
  would fold generated latents into the frozen VAE's running statistics.
  `requires_grad_(False)` does **not** prevent that.

x̂₀ is deliberately **not** clamped here. The ±10 clamp guards a reverse
process against one bad step; this loss only looks at low `t`, where
`predict_x0` is well conditioned, and a clamp would zero the gradient on
exactly the items that are most wrong.

The label the terms score against is not part of the latent store: it is
`patches_label.bin` in the data root recorded as
`metadata["source_patch_index"]`, addressed by `source_row`.
`LatentDataset(with_label=True)` serves it, and `build_latent_dataloaders`
sets that flag from `loss.decoded.enabled` — one place, so a run cannot ask
for the loss and be served batches without the label.

## Volume generation

One path: **hybrid chunked joint denoising** (`diffusion/sampler.py`).

- The volume is cut into CHUNKS of `chunk_tiles` 64-voxel tiles, generated in
  raster order.
- Inside a chunk, overlapping windows at `window_stride` voxels jointly denoise
  that chunk's latent canvas: every window predicts ε, the predictions are fused
  by a strictly-positive cosine weight, and ONE DDIM step is taken on the
  canvas. Fusing ε INSIDE the reverse process is not the same as blending
  finished samples, which halves the variance in the overlap and puts it
  off-manifold.
- A window's six faces come from the current chunk canvas at `t`, from a
  finished chunk re-noised to `t` with fresh noise, from OOB (the block leaves
  the volume), or from UNKNOWN (it reaches a chunk that does not exist yet).
  Both EXISTS sources sit at the same noise level, so a block straddling the
  current chunk and a finished one is still coherent.
- `chunk_tiles = (1, 1, 1)` is patch-at-a-time sequential generation; one chunk
  covering the volume is pure joint denoising.
- Every random draw is taken in the frame of the REQUEST, not of the canvas: one
  canvas-sized field per draw (the initial canvas once, the re-noising field at
  every timestep of every chunk), rolled by `request_offset` before use
  (`region_noise_field`). The value used at canvas cell `p` is the one the
  request sees at `p − offset`, so translating a request inside a bigger canvas
  translates its noise with it. That is what lets the assembly assessment hold
  the noise realisation fixed while the assembly grid moves; a canvas-anchored
  draw changed both at once. `request_offset` must be a whole number of latent
  cells.

Decoding is overlapped too: latent windows at `decode_stride` voxels, with the
decoded grey level and the raw 3-class logits blended under a tapered window
before the argmax. Direct stride-64 tiling left a decoder-side seam at every
patch face. `seam_discontinuity` is reported at BOTH the window period (64) and
the chunk period (`64 · chunk_tiles`), on the grey level and on the pore
log-odds, against one shared interior baseline.

### The requested porosity field over a window

`local_por_map` is defined on the TILE grid — one φ per 64-voxel tile — but a
window is placed every `window_stride` voxels, so at the default stride 32 it
straddles up to eight tiles. A window's requested φ is the tile field averaged
over the window's own voxel footprint, each tile weighted by the VOLUME of the
window it covers (`sampler.window_tile_mean`); the mean is then clamped to
`[POR_MIN, POR_MAX]` and transformed by `porosity_to_cond`. The average is on
RAW pore fractions and never on `cond_por`: that transform is a log, and the
mean of the transform is not the transform of the mean. Sampling the tile that
holds the window CENTRE instead — what the sampler did until this fix — handed a
window spanning a 0.01 tile and a 0.05 tile one of the two extremes, so the
50 %-overlap windows on either side of a field step both asked for the wrong
thing and the requested step was reproduced as a wider, offset one.

### Material porosity in, full-patch φ out

There are two porosities, and they are not the same number.

| | Definition | Where |
|---|---|---|
| **full-patch φ** | `pore / patch_size³` — the whole 64³ patch, air outside the specimen counted in the denominator | what the latent store records and what `cond_por` means |
| **material porosity** | `pore / material` — the specimen envelope only | what a user asks for, and what `eval_v4` measures |

`scripts/build_latent_dataset.py` stores `phi = (label == CLASS_PORE).mean()`
over the whole patch, so a patch half outside the specimen carries a φ about half
its material porosity. That is the right thing for training: the conditioning has
to describe the patch the encoder actually saw. It is the wrong thing to hand a
sampler a user's request in, and until this fix the sampler passed the request
straight through — a surface window asked for 0.03 got the full-patch 0.03 it
asked for, which is 0.06 material porosity, and eval scored the miss against a
target nobody requested.

`VolumeGenerator._window_conditioning` converts. Per window,

```
cond_phi = phi_request × (material fraction of the window)
```

where the material fraction is the mean of `material_map` over the window's
latent cells — every cell covers the same `downsample³` voxels, so that mean IS
the volume fraction of the window inside the specimen. The scale is applied to
the value `window_tile_mean` returns, **before** the `[POR_MIN, POR_MAX]` clip
and **before** `porosity_to_cond`: clipping first would turn a legal request
(0.2 material porosity at half material = 0.1 full-patch) into a clipped one, and
`porosity_to_cond` is a log, where a scale becomes an offset. A fully interior
window has fraction 1.0 and is untouched, so nothing changes for a volume with no
material map.

Nothing about training or the latent store moves: the store keeps full-patch φ,
and only the sampling-time request is converted.

## VAE model & training pipeline

See [vae_architecture.md](vae_architecture.md) for the encoder/decoder data flow
and the `train_step` / `eval_step` / `train_loop` internals.

### `training.freeze_modules` — frozen in both senses

The decoder fine-tune (`r08/decoder-ft`) names the encoder-side children in
`training.freeze_modules`, so the latent space the LDM was built on cannot move
underneath it. `apply_transfer` in `experiments/train_vae.py` applies it, and it
does **two** things per named child:

- `requires_grad_(False)` on its parameters, so no gradient reaches them, and
  `build_optimizer` never sees them.
- Holds the subtree in `eval()` for the rest of the run, by overriding `train()`
  on the model instance.

The second is not hygiene, for the same reason as the decoded auxiliary loss
above: the r08 VAE is BatchNorm3d, `train_step` calls `model.train()` on **every**
step, and a train-mode forward recomputes `running_mean` / `running_var` outside
autograd. `requires_grad_(False)` does **not** stop that. Without the eval hold
a "frozen" encoder's eval-mode `mu` drifts across the fine-tune and silently
invalidates the latent store.

The override sits on the model instance, so it covers every caller at once —
`train_step`, the `model.train()` after each sample export, the resume path
(which re-applies the freeze), and `torch.compile`, which wraps the model
afterwards and forwards `train()` down to it as a child. Do not "fix" this at
the call sites; there is more than one, and the next one added would miss it.

### Decoder output contract

The decoder heads are **not** symmetric, despite all being called "heads". Each
has its own decode helper in `src/poregen/models/vae/base.py`, and every decode
site must use them so the behaviour cannot drift apart again.

- **XCT head** (`VAEOutput.xct_out`) emits the grey level in `[0, 1]` — the same
  scale as `xct / 255`. `compute_total_loss` regresses it directly
  (L1/MSE/Charbonnier). It is **not** a logit: decode it with
  `decode_xct()` / `decode_xct_u8()`, which clamp. Applying a sigmoid squashes
  the output into `[0.5, 0.731]` and destroys contrast.
- **Mask head** (`VAEOutput.mask_logits`) genuinely is a logit
  (BCE-with-logits); decode it with `decode_mask()`, which applies the sigmoid.
  Emitted by every variant up to r07.
- **Class head** (`VAEOutput.class_logits`) is a 3-channel logit over the voxel
  label — 0 material, 1 pore, 2 air — trained with class-weighted cross-entropy
  plus soft Dice. Decode it with `decode_label()` (argmax) or
  `decode_class_probs()` (softmax). Emitted by the `*_cls` variants from r08 on.
  It carries the pore mask too: the qualitative sample export writes
  `decode_label(...) == CLASS_PORE` as `mask_recon`, next to the 3-class
  `label_recon`. Reaching for `mask_logits` there exports an empty volume.

**A variant emits `mask_logits` or `class_logits`, never both.** A two-valued
head and a three-valued one are different contracts; emitting both would let a
caller consume a pore mask that disagrees with the label.
`compute_total_loss` and the eval loop branch on which one is populated, so the
binary-mask config keys are simply inert for a 3-class variant.

Which input a variant's `forward()` takes is declared on the class as
`encoder_inputs` — `("xct", "mask")` historically, `("xct", "label")` for the
r08 3-class variant, whose encoder sees `cat([xct, pore, air])`. The training
engine reads that attribute rather than hard-coding the pair.

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
- For a 3-class head, `loss.class_weights` is **required** and
  `compute_total_loss` refuses to run without it: unweighted cross-entropy is
  dominated by the material class and both minority classes collapse. Compute
  them with `python scripts/build_split_v3.py --stage weights`
  (`w_c = 1 / (n_classes · f_c)`, so `Σ f_c w_c = 1`) and paste the result into
  the experiment config. Soft Dice is averaged over pore and air only —
  material is the background and its Dice sits near 1 regardless

## Experiment run structure

Runs land in `runs/vae/<experiment>/<timestamp>-<run_index>/` (and `runs/ldm/…`
for diffusion runs), each containing `log.jsonl`, `metrics.jsonl`,
`resolved_config.yaml`, `run_metadata.json`, `tensorboard/`,
`checkpoints/latest.ckpt` + `best.ckpt`. On resume, `run_metadata.json` is
updated and `log.jsonl` is pruned to the resume step to prevent duplicates.

A checkpoint carries the model, the optimizer, the scaler, the scheduler, the
RNG states, and — when the run has one — the discriminator with its own
optimizer. The resume path builds the discriminator *before* the load and
restores it: a fresh one would restart the GAN at every interruption, and
nothing in the loss curves would show it.

Analysis outputs live in `runs/campaigns/<NN>-<name>/`, one campaign per
question, each with a `README.md` and a vault note; see
`runs/campaigns/INDEX.md`.

## Key invariants

- **Porosity-MAE < 0.005** is the primary success metric (`val/porosity_mae`).
- `kl_collapsed_fraction` should stay low — a spike means the latent is collapsing.
- The **discriminator always runs in float32** — intentional, do not change.
- The **async checkpoint writer never sees a live tensor**. `save_checkpoint_async`
  copies the whole state to CPU on the calling thread. Passing the live state
  dicts lets the steps taken during serialisation land in a file labelled with
  an earlier step.
- `latent_channel_moments` returns GPU tensors — `merge_latent_channel_moments`/
  `active_units_from_moments` handle them correctly as-is.
- `eval_step` returns `(losses, output, xct_dev, mask_dev)` — reuse those device
  tensors in `_run_eval` rather than re-transferring.
- The XCT head is not a logit (see the decoder output contract above).
- A generated volume is written on the SAME scale as a real one: `volume.tif` is
  uint8 raw-scan grey and `label.tif` is uint8 `{0, 1, 2}`. Rescaling either
  makes generated and real volumes incomparable and cost a whole evaluation
  campaign once.
- `neighbour_offset >= patch_size` — face neighbours must TOUCH, never overlap.
  There is no flag to disable the guard.
- **A frozen module is in `eval()`, not only `requires_grad_(False)`.** BatchNorm
  running statistics are not gradients, and a train-mode forward moves them.
  This applies to `training.freeze_modules` and to any VAE put in a training
  graph (the decoded auxiliary loss).
- The specimen box in `material_mask` is the **largest** component of the
  Otsu max-projection, never the first-labelled one. Label ids follow raster
  order, so a bright dust speck above and left of the coupon is numbered first
  and used to become "the specimen", collapsing `sample_mask` to the speck's
  bounding box. A scan whose second-largest component exceeds
  `AMBIGUOUS_COMPONENT_RATIO` (10 %) of the largest raises `ValueError` — two
  comparable objects mean no single box is the specimen, and the scan must be
  inspected instead of silently segmented against half of itself.

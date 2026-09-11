# CODEMAP

Navigation map for the repo: where each thing lives and what it does. One line per file or group.
No tutorials, no API detail. For engineering rules see `AGENTS.md`. For subject depth see the
companion docs listed under [docs/](#docs).

**Keep this file current.** Add or remove a line here whenever a file is added or removed.

---

## End-to-end pipeline

| Stage | Entry point | Output |
|---|---|---|
| 1. Raw TIFF → volumes + patch index | `src/poregen/dataset/build_dataset.py` (CLI `build_dataset`) | `data/<split>/volumes.zarr/`, `patch_index.parquet`, `volume_stats.json`, `splits.json` |
| 2. Patch index → flat memmap (fast loading) | `scripts/extract_patches_memmap.py` | `data/split_v3/patches_{xct,label}.bin` + `patches_meta.json` (`patches_label.bin` is also what `LatentDataset(with_label=True)` reads for the decoded auxiliary loss) |
| 3. VAE training | `scripts/train_vae.py` → `poregen.cli.experiments:main` → `poregen.experiments.train_vae` → `poregen.training.engine.train_loop` | `runs/vae/<run>/` |
| 4. VAE → latent store | `scripts/build_latent_dataset.py` | `data/split_v3/latents_r08z8/` — per-split `latents.bin` + `index.parquet` + `material.bin` + `air.bin` |
| 4b. Per-patch conditioning | `scripts/build_conditioning.py` | per-split `cond.parquet` (`cond_depth`, six `cond_dist6_*`, `cond_por_raw`) + the `conditioning` block in the store `metadata.json`; reuses `data/split_v2/orientation_field.json` |
| 5. LDM training | `scripts/train_ldm.py` → `poregen.cli.experiments:main_ldm` → `poregen.experiments.train_ldm` → `poregen.training.ldm_engine.train_loop` | `runs/ldm/<run>/` |
| 6. Generation | `scripts/generate_volumes.py` (uses `poregen.diffusion.sampler.VolumeGenerator`) | `volume.tif` (uint8 grey) + `label.tif` (uint8 0/1/2) per porosity × layout |
| 7. Evaluation — VAE | `poregen.eval.runner.run_eval` (driven from the experiments CLI) | `runs/vae/<run>/eval/<timestamp>-<tier>/` |
| 7b. Evaluation — generated volumes | `poregen.eval_v4.cli` (CLI `eval_v4`) | `runs/campaigns/<NN>-<name>/<assessment>/` — volumes + manifests, `results.json`, `findings.md`, figures |

---

## src/poregen

### configuration + configs

| Path | What it does |
|---|---|
| `configuration/experiments.py` | Resolves an experiment id (`r07/base`) into a `ResolvedExperiment`: walks `extends`, merges `components`, body, `overrides`; also lists and clones definitions. |
| `configuration/__init__.py` | Public surface: `resolve_experiment`, `list_experiment_definitions`, `clone_experiment_definition`, `resolve_experiment_path`. |
| `configs/config.py` | `load_config` (plain dict from YAML) + typed `PoreGenConfig` dataclasses / `parse_config`, used by the resolver for validation. |
| `configs/*.yaml` (`vae_default`, `vae_v2_default`, `r03`, `r04`, `r05`, `example_vae`) | 🕰 Legacy flat per-experiment YAMLs, superseded by top-level `configs/experiments/`. Still referenced as defaults by `scripts/eval_checkpoint.py`, `scripts/debug_reconstruction.py`, `experiments/base.py`, `experiments/r03.py`. |

### cli

| Path | What it does |
|---|---|
| `cli/experiments.py` (~2k lines) | The whole experiment CLI: `main` (VAE) and `main_ldm` (LDM) — list / clone / launch / resume runs, plus a curses interactive menu and the eval integration point. |

### dataset

| Path | What it does |
|---|---|
| `dataset/build_dataset.py` | CLI: discover raw TIFFs → Zarr → masks → per-volume stats → patch index. Supports `--stats_only`. |
| `dataset/io.py` | Volume discovery, TIFF load, Zarr write, per-volume intensity stats (`VolumeInfo`). |
| `dataset/segmentation.py` | Pore/material segmentation (Sauvola + Otsu, fill-voids). Package home of the root `onlypores.py`. `compute_sample_mask` re-derives just the material envelope (no Sauvola). `material_mask` takes the specimen box from the LARGEST max-projection component (label order is raster order, so a corner dust speck used to win) and raises on an ambiguous scan — second-largest over `AMBIGUOUS_COMPONENT_RATIO` = 10 % of the largest. |
| `dataset/material.py` | ldm06 material maps: block-mean pooling of the specimen ENVELOPE (`label != 2`, pores included) to latent-resolution fraction cells, batched over a patch dimension, plus uint8 encode/decode and `air_fraction` (`1 - material.mean()`, exact). Pooling `label == 0` instead would be the pore mask at 100 µm — an input the model would upsample instead of generating pores. |
| `dataset/patch_index.py` | Patch coordinate generation, 3-D integral volume for O(1) patch fractions (`patch_fractions` — porosity and, in split_v3, air fraction), Parquet index writer. |
| `dataset/splits.py` | Volume-level splits: deterministic `v1` and stratified-by-porosity `v2`; writes lightweight split roots. The `v3` split is by PANEL and lives in `scripts/build_split_v3.py`. |
| `dataset/holes.py` | Finds the three drilled registration through-holes of a coupon from `sample_mask` (z-MINIMUM projection of the complement, border-touching components dropped, Euclidean dilation) and marks the patches that touch one. |
| `dataset/loader.py` | `PatchDataset` (Zarr backend), `MemmapPatchDataset` (`.bin` backend), `build_label` (3-class voxel label), `zarr_worker_init_fn` for fork-safe workers. Both backends return `label` (int64 `{0,1,2}`) and the derived binary pore `mask` (`label == 1`). |

### models

| Path | What it does |
|---|---|
| `models/nn/blocks.py` | Shared 3-D conv blocks: v1 family (GroupNorm/SiLU/ConvTranspose3d) and v2 family (BatchNorm3d/GELU/Upsample+Conv), plus `reparameterize`. |
| `models/vae/base.py` | `VAEConfig` and `VAEOutput` dataclasses, the class constants, and every decode helper: `decode_xct`, `decode_xct_u8`, `decode_mask`, `decode_label`, `decode_class_probs`. |
| `models/vae/registry.py` | `@register_vae(name)` / `build_vae(name, **kw)` / `list_vaes()`. |
| `models/vae/v1/{conv,conv_noattn,unet}.py` | 🕰 First-gen architectures `conv`, `conv_noattn`, `unet`. Kept for old-checkpoint compatibility; not used by current experiments. |
| `models/vae/v2/conv.py`, `v2/unet.py` | Second-gen attention and skip-connection variants (`v2.conv`, `v2.unet`). |
| `models/vae/v2/conv_noattn.py` | `v2.conv_noattn` — R03 baseline trunk; also the `v2.conv_noattn_xctonly` (mask head removed) used by r06. |
| `models/vae/v2/conv_noattn_dualbranch.py` | `v2.conv_noattn_dualbranch` — dual encoder branch + fusion. r04/r05/r07. **Still live**, not legacy: the r07 checkpoint behind `data/split_v2/latents_r07z4` is loaded by the campaign 05 and 08 analysis scripts. |
| `models/vae/v2/conv_noattn_dualbranch_cls.py` | `v2.conv_noattn_dualbranch_cls` — same trunk, encoder input `cat([xct, pore, air])` (`in_channels` 3) and a 3-class decoder head (material/pore/air) in place of the binary mask head. **Current VAE** (r08, the one ldm06 builds on). |
| `models/vae/v2/vrrae.py` | `v2.vrrae` — VRRAE-bottleneck VAE, XCT-only encoder and decoder (no mask head). |
| `models/vae/v2/vrrae_bottleneck.py` | The bottleneck itself: flatten → FC → truncated-SVD RR layer (vendored) → identity posterior mean. |
| `models/vae/v2/vrrae_linear.py` | `v2.vrrae_linear` — SVD ablation twin: same flat bottleneck, plain `Linear` heads. |
| `models/vae/v2/vrrae_finetune.py` | Fixed-basis extraction at inference + decoder-only fine-tune hook for VRRAE. |
| `models/discriminator.py` | 2-D multi-plane PatchGAN + LSGAN losses for adversarial XCT supervision (R04+). Runs in float32 — see `AGENTS.md`. |
| `models/diffusion/unet.py` | `UNet3DDenoiser` / `UNet3DConfig` — the LDM denoiser. Input channels: `z` + orientation (2) + material (1) + 6 whole neighbour latents + 6 availability embeddings + 6 sinusoidal `nb_t` embeddings = **127 at z=4**. FiLM/AdaGN scalars: `cond_por` (learned null token), `cond_depth`, `cond_dist6` (one MLP over the 6-vector), plus the availability-masked neighbour pool. Nothing is switchable and there is no global-porosity input. |
| `models/diffusion/blocks.py` | GroupNorm/SiLU res-blocks, attention, sinusoidal time embedding for the denoiser. |

### losses / metrics

| Path | What it does |
|---|---|
| `losses/recon.py` | XCT recon losses in z-score space: `l1`, `mse`, `charbonnier`, `get_recon_loss`. |
| `losses/mask.py` | Mask losses on logits: BCE, Dice, Tversky, focal, `combined_mask_loss`; and for the r08 3-class head `multiclass_ce_loss`, `multiclass_dice_loss`, `combined_class_loss`. |
| `losses/kl.py` | KL with free-bits (per-channel and flat) + `beta_schedule`. |
| `losses/total.py` | `compute_total_loss` — composes recon + mask + β·KL. Gotchas live in `AGENTS.md`. |
| `losses/decoded.py` | LDM decoded-space auxiliary loss (ldm06/aux): the four voxel-space terms (`air_outside_material`, `pore_dice`, `porosity_consistency`, `grey_agreement`), the lowest-t sub-batch selection (`select_decoded_items`), the ramp, and `DecodedAuxLoss`, which denormalises x̂₀ and decodes it through the FROZEN r08 VAE. `eval()` is mandatory there — the decoder is BatchNorm3d and `requires_grad_(False)` does not stop running-stat updates. `AIR_GREY_THRESHOLD` is 182/255, the air audit's real-calibrated void threshold. |
| `metrics/recon.py` | MAE, MSE, PSNR, sharpness proxy. |
| `metrics/seg.py` | Vectorised Dice / precision / recall, porosity metrics, binned porosity MAE; `multiclass_metrics` reads per-class Dice and porosity/air MAE off the 3-class argmax. |
| `metrics/latent.py` | KL per channel, active units, streaming channel moments and their merge. |

### training

| Path | What it does |
|---|---|
| `training/engine.py` (~1.3k lines) | VAE `train_step` / `eval_step` / `train_loop`: AMP, discriminator, EMA, eval cadence, early stopping, TensorBoard + JSONL logging, sample export. See `docs/vae_architecture.md`. |
| `training/ldm_engine.py` | LDM equivalent: EMA, the train/eval step (target from `schedule.training_target`, so ε and v share one step), eval loop, in-training generation eval and sample volumes. `sample_neighbours` draws each EXISTS neighbour from its stored posterior (`mu + sigma*eps`, like the target) and `noise_neighbours` then noises the draw (per-neighbour `t_nb`, `nb_t_mix`, the `drop_nb` CFG null). See `docs/ldm_training.md`. |
| `training/latent_bank.py` | `LatentBank`: every 16-cell window of every saved LDM latent canvas, memory-mapped, for the refiner decoder fine-tune (D43 option 2). Refuses a bank that mixes latent widths — the failure would otherwise surface as a decoder shape error thousands of steps in. It feeds the discriminator's FAKE branch only; the reconstruction and class losses stay on real patches, because a sampled latent carries no ground-truth label to score against. |
| `training/data.py` | `build_dataloader_kwargs`, `build_patch_dataloaders` (train/val/test from the Zarr or memmap backend). |
| `training/checkpoint.py` | Atomic checkpoint save/load (model, optimizer, scaler, scheduler, EMA, RNG, discriminator + its optimizer), sync + async variants. The async writer gets a CPU snapshot taken on the calling thread and re-raises a failed write from `join()`. |
| `training/device.py` | Device selection, autocast dtype, GradScaler. |
| `training/seed.py` | `seed_everything`. |
| `training/sample_export.py` | Writes saved 3-D patch samples as ImageJ-readable TIFF stacks; migrates legacy `.npz` archives. One TIFF per array handed in — `PATCH_SAMPLE_KEYS` are the compulsory four, and a 3-class run adds `label_recon`. |

### diffusion

| Path | What it does |
|---|---|
| `diffusion/latents.py` | Load side of the latent store: `LatentDataset` + `build_latent_dataloaders`; reads `metadata.json` + per-split `latents.bin`/`index.parquet`/`cond.parquet`/`material.bin`/`air.bin`, applies the train-split normalisation, and returns the fixed ldm06 batch contract (`cond_por`/`cond_depth`/`cond_dist6`/`cond_orient`/`cond_material`, the CLEAN posterior — mean AND std — of the unshifted touching neighbours + availability, `air_fraction`, provenance). With `with_label=True` (set from `loss.decoded.enabled`) it also serves the source patch's voxel label, read from `patches_label.bin` in the data root recorded as `metadata["source_patch_index"]` and addressed by `source_row` — the label is not copied into the store. Availability is only ever EXISTS or OOB — UNKNOWN comes from the training step and the sampler. Keeps `sample_stride` (32, data multiplier) / `generation_stride` (64) / `neighbour_offset` (64) separate and enforces `neighbour_offset >= patch_size`. |
| `diffusion/noise_schedule.py` | `DDPMSchedule` — cosine schedule, movable to device without being an `nn.Module`. Owns the prediction objective: `objective` (`eps` or `v`, Salimans & Ho 2022) and `zero_terminal_snr` (Lin et al. 2024 Alg. 1), with `training_target` / `predict_x0` / `predict_eps` so nothing downstream branches on which one is in force. `from_cfg` is the ONE builder every entry point uses. ᾱ comes straight from the cosine `f`, so its terminal value is already 6e-17 and the rescale only makes it exact — what that buys is a terminal step where x̂₀ needs no division. Refuses `zero_terminal_snr` with `eps`. |
| `diffusion/sampler.py` | `DDIMSampler` (nested-CFG `predict_out`, `sample_batch`) and `VolumeGenerator` — ONE generation path: hybrid chunked joint denoising. Chunks of `chunk_tiles` 64-voxel tiles in raster order; inside a chunk overlapping `window_stride` windows jointly denoise its latent canvas with cosine-weighted ε fusion; each window's six faces come from the chunk canvas at `t`, a finished chunk re-noised to `t`, OOB, or UNKNOWN. `neighbour_mode` selects WHERE those faces come from: `"canvas"` is that production path, `"unknown"` makes every in-bounds face the CFG null (the ldm05 joint sampler) and `"reference"` reads them all from a `reference_latents` canvas of real encoded material (teacher forcing); the last two exist only for eval_v4's `assembly_modes` and OOB is kept in all three, because the canvas edge is geometry and not neighbour content. Decode is overlapped at `decode_stride` and blends grey + 3-class logits with a tapered window before the argmax. The nested CFG combines RAW network outputs (ε or v — an affine combination commutes with the affine v↔ε map at fixed `t`) and `ddim_step` converts once; `rescale_guidance` is the optional Lin §3.4 over-exposure correction, off unless `guidance.cfg_rescale > 0`. A window's requested porosity is `window_tile_mean` — the tile-grid field averaged over the window's voxel footprint, volume-weighted, on RAW φ before the clamp and `porosity_to_cond`. With no `material_map` given, the default one is the EXACT specimen-box/latent-cell intersection fraction — the product of the three per-axis overlaps in closed form, so a surface cell the box crosses carries its true fraction instead of being rounded to 1 or 0. Every random draw is `region_noise_field` — one canvas-sized field per draw (the initial canvas once, the finished-chunk re-noising at every timestep), rolled by `request_offset` so the noise is anchored to the REQUEST and not to the canvas; translating a request inside a bigger canvas therefore translates its noise with it, which is what the assembly assessment needs to hold fixed. Also `window_origins`, `window_weight`, `theta_from_layup` and `seam_discontinuity` (per-axis period, so window and chunk planes are reported separately against one interior baseline). |
| `diffusion/conditioning.py` | Single source of truth for the shared conventions: neighbour direction order, availability states (OOB/EXISTS/UNKNOWN), the grid index, the six per-face distance directions and `dist6_from_box`/`dist6_from_box_array`, the porosity transform + clamp range, and the `neighbour_offset >= patch_size` leak guard (no escape hatch). |
| `diffusion/orientation.py` | `OrientationField` (reads `data/split_v2/orientation_field.json`) + the `(cos2θ, sin2θ)` encoding: pool the components, never the angle, and never renormalise the pooled vector. Shared by dataset and sampler. |
| `diffusion/generation_eval.py` | In-training LDM diagnostics: off-manifold latent moments, x0-clamp fraction, and decode-based pore/air fraction vs the real val split, sampled with real val conditioning and the CFG neighbour null (all-UNKNOWN at `nb_t` 0). |
| `diffusion/porosity_field.py` | Coherent per-patch porosity field for inference (D32 §4): T-E marginal draw per grid cell, anisotropic Gaussian smoothing with the T-D volume-mean-removed correlation lengths, mean rescale to the global target, clamp to the training phi range [0.002, 0.107]. Loaders for the T-E sampler and T-D lengths from their `results.json`. Used by `scripts/generate_volumes.py` distribution `"coherent"`. |

### experiments / runtime

| Path | What it does |
|---|---|
| `experiments/train_vae.py` | Config-driven VAE run: build model, dataloaders, run dir, `train_loop`; `run_experiment` and `resume_run`. `apply_transfer` is the transfer/freeze entry point — it holds every `training.freeze_modules` subtree in `eval()` as well as `requires_grad_(False)`, by overriding `train()` on the model, so a frozen encoder's BatchNorm cannot drift through a fine-tune. |
| `experiments/train_ldm.py` | Same for the LDM. Also pulls the porosity standardisation stats out of the latent store metadata and refuses to start when the store's VAE checkpoint is not the one the config names. |
| `experiments/base.py` | `ExperimentRuntime` (`from_checkpoint` factory), `build_patch_loader`, `find_repo_root`. |
| `experiments/r03.py` | R03 notebook import surface: auxiliary XCT decoder, its train/eval helpers, latent analyses. |
| `analysis/__init__.py` | 💀 Empty stub — only a pointer saying R03 analysis moved to `experiments/r03.py`. |
| `runtime/runs.py` | Run directory creation, run-name construction from config, run discovery, resolved-config saving. |
| `runtime/metadata.py` | Captures git / machine / environment metadata into `run_metadata.json`. |
| `runtime/preflight.py` | RAM estimation and auto-reduction of DataLoader workers before a run starts. |

### eval (VAE evaluation package)

| Path | What it does |
|---|---|
| `eval/config.py` | `EvalConfig` dataclass + `load_eval_config`; resolves ids like `eval/r03_base` and the metric tiers. |
| `eval/runner.py` | `run_eval` — the only module that wires checkpoint → reconstruction → metrics → visuals → outputs. |
| `eval/stochastic.py` | Volume reconstruction in three modes (`stoch_mean`, `stoch_single`, `mu`) in one tiling pass. |
| `eval/blended.py` | Overlapping-patch reconstruction with Tukey-window blending in logit space. |
| `eval/metrics.py` (~1.3k lines) | Authoritative metric implementations: patch-level, volume-level, latent audit, S2(r), PSD, FID. |
| `eval/visualise.py` | Slice grids, std images, S2/PSD plots, 3-D pore GIFs. No inference here. |
| `eval/outputs.py` | The single writer of eval artefacts + the self-describing `README.md` in each eval directory. |

### eval_v4 (generated-volume evaluation suite)

The rerunnable suite behind the `eval_v4` CLI. See
[eval_methodology.md](eval_methodology.md) for what each metric means and why.
Replaces `scripts/eval_generated_volumes.py` (deleted). Most of that script
scored distribution distances, which cannot express a *request*, and every
question ldm06 exists to answer is conditional. Its five statistics that ARE
worth having were ported into assessment 8, where they finally carry a
real-vs-real floor.

| Path | What it does |
|---|---|
| `eval_v4/manifest.py` | `Manifest` — what a volume claims about itself (model run, step, raw/EMA, objective, `cfg_rescale`, sampler geometry, the request, shape, commit, wall time, peak GPU memory) with a schema that refuses an unknown field or a self-contradiction. `requires(*fields)` is the decorator every metric declares itself with: it refuses a volume whose manifest lacks a field the metric reads, or whose positional array shape contradicts `volume_shape`. |
| `eval_v4/io.py` | Campaign layout, `save_case` (manifest written LAST, so a truncated case has none), `Case` with lazily-read arrays and the derived requests (`material_voxels`, `requested_phi_per_tile`). `load_u8` refuses anything not already uint8 — no rescaling branch, on purpose. |
| `eval_v4/metrics.py` | Every measurement: phase fractions, porosity error and the 0.005 gate, failure flags and degenerate cells, the WITHIN-volume local fit (plus `pooled_dose_fit`, named `..._not_obedience`), both seam periods via the sampler's own `seam_discontinuity`, pore Dice and the chunk-plane slab, the grey air detector (constants and calibration imported from `_eval_v3`, never forked), cross-head disagreement, layup recovery through the T-I readers, geometry Dice, and the campaign-08 layup floor / VAE tile-decode control read from their `results.json`. `chunk_blocks` / `chunk_seam_profile` / `chunk_porosity` / `chunk_s2` / `chunk_profile` resolve the same seams PER CHUNK along the generation order, on a reference chunk grid given by the assessment and never read from the volume's own manifest — each block owns the planes at its own origin, so every plane has exactly one owner and chunk 0 owns none. |
| `eval_v4/cases.py` | The eleven assessments as data — 177 `CaseSpec`s over seeds 101/202/303 — plus the requested-field painters (halves, checkerboard, coherent), the notch-and-hole material map, and the three layups (A and B16 from `data/layup_ground_truth.json`, C a fixed permutation of A). `CaseSpec.neighbour_mode` names where a case's neighbours come from; only `assembly_modes_cases` asks for anything but the production `"canvas"`. Nothing here knows the sampler. |
| `eval_v4/generate.py` | The ONLY module that knows the sampler API. `VolumeRunner` loads the LDM, VAE and latent stats once, then rebuilds sampler + generator per case. Reaches the model through `DDPMSchedule.from_cfg` and `VolumeGenerator.generate` only. `resolve_latent_store` takes the store from the run's own `data.latents_root` — there is no default — and refuses, before the first model call, a store whose latent width or VAE checkpoint disagrees with the run (checkpoints compared as RESOLVED ABSOLUTE paths: the store records one absolutely, a run config relative to the repo). Its `override` argument names another store by hand for a diagnostic and faces the same checks; `scripts/generate_volumes.py` is the only caller that uses it. Documents the three places the sampler's interface shapes the suite: seeding is global, size is in millimetres (and is checked to snap back), and the window phase has no parameter, so assessment 6 translates the request inside a larger canvas instead. |
| `eval_v4/measure.py` | One measurer per assessment, aggregating mean ± sd over seeds (`measure_microstructure` instead reports every distance three ways — generated vs real, the real-vs-real floor, and the ratio); plus `manifest_check`, which separates a manifest that does not parse, a volume that contradicts its manifest, and a case the assessment defines but the campaign lacks. Never re-reads the model. |
| `eval_v4/microstructure.py` | The five distribution statistics, ported from the deleted `scripts/eval_generated_volumes.py` and each validated against a structure whose answer is known: S2(r) by Hann-windowed FFT autocorrelation debiased with the window's own autocorrelation, the pore-size distribution of 6-connected components, border-corrected Ripley's K over a KD-tree (converges to the CSR `(4/3)pi r^3`, unlike the uncorrected estimator it replaces), FID on 64x64 native slice crops through torchvision's InceptionV3 pool3, and the latent nearest-neighbour memorisation check against the store the volumes were generated from (the manifest names it; nothing here guesses a store). `compare_sets` is the two-sample comparison the assessment runs twice — generated against real, and real against real. A missing torchvision or a missing latent store is a reported skip, not a failure. |
| `eval_v4/real_floor.py` | Cuts crops of the split_v3 TEST panels at the generated shapes and runs every request-free metric on them. Reduces the crop depth a tile at a time until a box fits entirely inside `sample_mask` (a 192-deep one does not exist on a real laminate), never below 128. `--shapes micro` additionally cuts the matched-porosity reference PAIRS assessment 8 is floored against: per level and per PANEL, two 128-cubed crops at that porosity that share no material. Candidates are scored exactly from a one-pass cell summary of the pore mask, so no candidate needs the volume read for it. |
| `eval_v4/teacher.py` | The teacher-forced ceiling for `assembly_modes`: a canvas of REAL encoded material, assembled by pasting one stored latent per 64-voxel tile from the RUN's own store (the rows whose origins are 64 apart tile a block with no overlap). Serves a posterior draw `mu + sigma*eps`, per-channel normalised and seeded by the case — not the mean, which is a distribution neither the model nor the sampler ever sees. Nothing here encodes anything. `find_reference_block` searches both phases of the 32-voxel lattice and RAISES when no test volume holds the shape: test patches reach z0 = 128, so a 384-deep canvas cannot be teacher forced, and repeating a block to fill it would put a fake join on a chunk plane. |
| `eval_v4/report.py` | `findings.md` + figures (PDF and PNG, 300 dpi) from `results.json` alone, with the real-floor row first in every table that has one. Touches no volume. |
| `eval_v4/cli.py` | `eval_v4 generate \| measure \| report \| manifest-check \| real-floor`. |

---

## scripts/

**Dataset building**

| Path | What it does |
|---|---|
| `build_split_v3.py` | Builds `data/split_v3/` in stages (`holes | splits | index | resplit | weights | report`): detects the drilled registration holes per volume, splits by PANEL (not by coupon — sibling coupons of one panel share microstructure and would leak), writes the patch index with hole-touching patches dropped, applies the manual re-split (`Na_09` -> test, `Na_01` -> val, which keeps 96 % of the high-porosity training patches instead of the 8 % the porosity-bin rule alone would have left), and derives the 3-class loss weights (`sqrt_inverse` by default, normalised so `sum_c f_c w_c = 1`). |
| `extract_patches_memmap.py` | One-time Zarr → flat `patches_{xct,label}.bin` memmap extraction, row-aligned with `patch_index.parquet`. `patches_label.bin` holds the 3-class voxel label (0 material, 1 pore, 2 air = `sample_mask == 0`); air takes precedence over pore. Refuses to start when the filesystem cannot hold both arrays. Resumable per volume; on resume the volumes finished earlier are counted again from the written label array, so `patches_meta.json` (and the 3-class weights derived from it) always describes the whole extraction. |
| `build_latent_dataset.py` | Encodes every patch with a trained VAE (encoder inputs read off the model's `encoder_inputs`) → raw `mu`/`std` memmap + train-split normalisation stats, AND `material.bin`/`air.bin` pooled from the same label tensor in the same pass. Builds the `latents_r08z8` store. |
| `build_conditioning.py` | Builds the per-split `cond.parquet` sidecar (`cond_depth`, six `cond_dist6_*`, `cond_por_raw`) and the `conditioning` metadata block, reusing `data/split_v2/orientation_field.json` after asserting every store volume has a record and a foreground extent. `--rebuild-orientation` re-derives that field from the T-I fit + expert ground truth — the provenance of the artefact. `--store` is REQUIRED and has no default — the store name follows the chosen rung, so a default annotates the wrong store the moment the rung changes — and the path is refused unless it holds a `metadata.json`, because this writes sidecars INTO the store. |

**Training entry points**

| Path | What it does |
|---|---|
| `train_vae.py`, `train_ldm.py`, `experiments.py` | Thin wrappers around `poregen.cli.experiments:main` / `:main_ldm`. Same as the installed `train_vae` / `train_ldm` / `experiments` console scripts. |

**Generation**

| Path | What it does |
|---|---|
| `generate_volumes.py` | Sweeps porosity × spatial layout with the hybrid chunked sampler (`--chunk-tiles`, `--window-stride`, `--decode-stride`, `--ddim-steps`, `--s-por`, `--s-nb`), writes `volume.tif` (uint8 grey) + `label.tif` (uint8 0/1/2) + `generation_stats.json`. Both arrays are NATIVE scale — nothing rescales them. The latent store is not a flag: it comes from the checkpoint's `resolved_config.yaml` through `eval_v4.generate.resolve_latent_store`, so the VAE and the normalisation belong to the run. `--latents-root` is a diagnostic override with no default, checked the same way. |
| `visualize_denoising.py` | MP4 of a DDIM chain for one interior patch, decoding every intermediate latent to grey / p(pore) / argmax label. |

**Diagnostics / profiling**

| Path | What it does |
|---|---|
| `diag_ldm_samples.py` | Standard LDM convergence health-check: samples with REAL val conditioning split by EXISTS-neighbour bucket (0..6, weighted by the OBSERVED frequency in the scanned rows), {raw, ema} × DDIM-50 latent moments + decoded pore/air fraction per bucket, conditioning-alive probe (porosity / orientation / material / neighbours-UNKNOWN vs the total-noise MAD scale), φ kill-switch, and a trend verdict (CONVERGING/STALLED/REGRESSING) appended to `<run_dir>/convergence_check.jsonl`. `--ddim200` adds the 200-step variants. |
| `diag_mask_sanity.py` | Does the r07 decoder emit sane masks from noise-perturbed latents? Writes `runs/diagnostics/mask_sanity_r07z4/`. |
| `diagnose_decode_regime.py` | Sweeps `z = mu + s·sigma·eps` to test clean-mu vs training-time-sample decoding. |
| `debug_reconstruction.py` | 🕰 Four-hypothesis probe of early reconstruction/orientation/z-score bugs. Defaults to the legacy `src/poregen/configs/vae_default.yaml`. |
| `bench_batch_size.py` | Throughput vs batch size (loader-only, train, eval) + peak GPU memory, using the real model and loaders. |
| `profile_vae_memory.py` | Single-config peak-memory profile of the real `train_step` on synthetic batches. |
| `investigate_vrrae_throughput_cliff.py` | 💀 Self-labelled THROWAWAY: reproduces the cuDNN 64-bit-indexing backward cliff on the GB10. |
| `rebuild_tensorboard.py` | Rebuilds TensorBoard event files from a run's `metrics.jsonl`. |
| `convert_patch_samples_to_tiff.py` | Migrates legacy patch-sample `.npz` archives under `runs/` to TIFF. |

**Analysis — conditioning design & microstructure (CPU only)**

Every script here writes under `runs/campaigns/<NN>-<name>/` — one campaign per
question, indexed in `runs/campaigns/INDEX.md`. `_common.py` holds the shared
plot style, JSON writers, zarr/data paths and 1-D signal helpers.

⚠ **The generation-side campaign scripts are pinned to the ldm05 sampler API**
(`DDPMSampler`, `mode="sequential"/"joint"`, `conditioning_semantics`,
`mask.tif`) and will NOT run against the ldm06 code. They are kept unchanged as
the provenance of the results already published from them — rewriting them onto
an API they never ran with would make those results untraceable. A new
generation campaign starts from `scripts/generate_volumes.py`.

Campaign map: `01-conditioning-design` (`t_*`, `viz_orientation_volume`) ·
`02-porosity-control-v1` (`dose_response`, `cfg_sweep`, `layup_roundtrip`,
`void_mask_audit`, `joint_vs_sequential`, `regen_layup_volumes`) ·
`03-eval-v2-buggy-decode` (`eval_v2_*`) · `04-measurement-limits`
(`air_audit_v2`, `onlypores_*`) · `05-eval-v3-fixed-decode` (`eval_v3_*`) ·
`06-ldm06-probe` (`ldm06_probe`) · `07-vae-metric-recompute`
(`recompute_vae_metrics`, `report_vae_metric_recompute`) ·
`08-pre-ldm06-diagnostics` (`vae_tile_seam`, `ddim200_1024`,
`ply_angle_structure_tensor`) · `09-r08-latent-sweep` (`r08_rung_report`,
`r08_calibration_probe`) · `12-sample-mask-audit`
(`sample_mask_component_audit`).

| Path | What it does |
|---|---|
| `analysis/_common.py` | Shared paths (`ZARR_ROOT`, `LATENT_INDEX_DIR`), plot style, `write_json`/`write_findings`, periodogram / autocorrelation / AR(1) surrogate helpers. |
| `analysis/t_a_periodicity.py` | Porosity periodicity on all three axes → the 19.4–19.8-voxel ply pitch in z. |
| `analysis/t_c_air_boundary.py` | Air/specimen boundary characterisation. |
| `analysis/t_d_autocorrelation.py` | Spatial autocorrelation of φ on all three axes. |
| `analysis/t_e_conditional.py` | p(local φ \| global φ) — can a volume target be painted uniformly? |
| `analysis/t_f_transform.py` | Choice of porosity transform for conditioning. |
| `analysis/t_g_orientation.py` | In-plane orientation vs depth from the doubled-angle spectral second moment. ⚠ Its "Nacho is unidirectional" conclusion is **wrong** — superseded by T-I. |
| `analysis/t_h_angular_shape.py` | Shape of the angular power density (modes m = 2, 4, 6 + lobe counting). ⚠ Its "Nacho is a 0/90 cross-ply" conclusion is **wrong** — superseded by T-I. |
| `analysis/t_i_layup_validation.py` | Validation of measured ply orientation against `data/layup_ground_truth.json`. Volume-mean-subtracted angular density + pore principal axes; recovers the true 4-class ply sequence at 84.7 % (88.6 % held out) vs a 49.6 % null. Diagnoses the T-G/T-H failure and emits `layup_field.json`. |
| `analysis/dose_response.py` | Porosity dose-response for the final ldm05 checkpoint (GPU): delivered vs requested φ over 7 targets × {sequential, joint} × 3 seeds at 192³, coherent local-φ field, per-volume seam metrics, OLS fits. Writes `runs/campaigns/02-porosity-control-v1/dose_response/`. |
| `analysis/cfg_sweep.py` | CFG porosity-guidance sweep for the final ldm05 checkpoint (GPU): delivered φ, seam metrics, and degenerate-block fraction over s_por {0, 0.5, 1, 1.5, 2, 3} × targets {0.02, 0.05} × {sequential, joint} × 3 seeds at 192³, coherent local-φ field. Writes `runs/campaigns/02-porosity-control-v1/cfg_sweep/`. |
| `analysis/layup_roundtrip.py` | Layup round-trip for the final ldm05 checkpoint (GPU): generates 1024×1024×192 volumes (full T-I window in-plane) for three requested stacking sequences (training layup A, a permutation, a simple repeat) × {sequential s_por 1.0, joint s_por 1.5} × 2 seeds, measures per-ply angles back with the imported T-I combined estimator, and scores direct/fitted angular error + 4-class accuracy vs the 84.7 % real-volume reference. Validates the adapted single-window path on real volumes first. Writes `runs/campaigns/02-porosity-control-v1/layup_roundtrip/`. |
| `analysis/void_mask_audit.py` | Validity audit of the generated grayscale-vs-mask disagreement: calibrates a grayscale void detector on real volumes (Dice vs real masks), de-sigmoids generated volumes to the decoder-native scale (the sampler's `expit` compresses intensities), then reports dark-but-unmasked fraction, component sizes, mask/detector/union porosity, per-64³-cell localisation, and two GPU VAE controls (latent std of worst cells; mask-head recall on the largest real pores). Writes `runs/campaigns/02-porosity-control-v1/void_mask_audit/`. |
| `analysis/joint_vs_sequential.py` | Sequential vs joint volume assembly at one ldm05 checkpoint: wall time, peak memory, seam ratios (xct + mask, per axis) and delivered porosity for one 192³ volume per mode. Was `compare_modes.py` living inside its own output directory. Writes `compare_modes_results_<step>.json` under `runs/campaigns/02-porosity-control-v1/joint_vs_sequential/`. 🕰 one seed per cell — seam metrics superseded by campaign 05's dose-response. |
| `analysis/regen_layup_volumes.py` | Regenerates the showcase subset of `layup_roundtrip.py` volumes (6 joint + the worst sequential cell) with identical settings and seeds and SAVES them as `volume.tif` + `mask.tif`, which the evaluation run did not. Writes `runs/campaigns/02-porosity-control-v1/layup_roundtrip/volumes/` — the input `void_mask_audit.py` needs. |
| `analysis/_eval_v2.py` | Shared infrastructure for the eval v2/v3 campaigns: the three generation arms (`seq`, `joint_legacy`, `joint_oob` — mode × conditioning semantics × operating s_por), generator construction, coherent-φ field, OOM-halving `run_one`, per-volume metrics (void-corrected porosity, per-64³-cell porosities, degenerate fraction), volume saving (`volume.tif` NATIVE uint8) + resume. Output root is `$POREGEN_EVAL_ROOT` (default `runs/campaigns/03-eval-v2-buggy-decode`); eval v3 points it at `runs/campaigns/05-eval-v3-fixed-decode`. |
| `analysis/eval_v2_dose_response.py` | Dose-response (GPU): 7 targets × 3 arms × 3 seeds at 192³; global AND local (per 64³ cell vs coherent-field target) OLS fits, mask vs void-corrected porosity, seam metrics. Writes under `$POREGEN_EVAL_ROOT/{volumes/dose_response, dose_response}/`. |
| `analysis/eval_v2_cfg_sweep.py` | CFG sweep v2 (GPU): s_por {0.5, 1, 1.5, 2} × targets {0.02, 0.05} × 3 arms × 3 seeds at 192³; per-volume mask/corrected porosity, seam ratios, degenerate-block fraction. Saves every volume under `runs/campaigns/03-eval-v2-buggy-decode/volumes/cfg_sweep/`; results to `runs/campaigns/03-eval-v2-buggy-decode/cfg_sweep/`. |
| `analysis/eval_v2_phase1.sh` | Phase-1 runner: dose-response v2 then CFG sweep v2, appending to `runs/campaigns/03-eval-v2-buggy-decode/phase1.log` (ends with `PHASE1_DONE`). |
| `analysis/eval_v2_layup.py` | Layup round-trip (GPU): 3 layups × 3 arms × `--seeds` (default 2 seeds) volumes of 1024×1024×192 at target φ 0.03; T-I angle measurement (imported from `layup_roundtrip.py`), delivered + void-corrected porosity, seam ratios. Writes under `$POREGEN_EVAL_ROOT/{volumes/layup, layup}/`. |
| `analysis/eval_v2_phase2.sh` | Phase-2 runner: layup round-trip v2, appending to `runs/campaigns/03-eval-v2-buggy-decode/phase2.log` (ends with `PHASE2_DONE`). |
| `analysis/eval_v2_audit.py` | Void/mask audit v2 (phase 3, CPU): recalibrates the grayscale void detector on real volumes (dice-optimal T_best + precision-≥0.95 conservative T_cons, decoder-native scale, ≥300-voxel connected-component filter), then audits all 153 saved campaign volumes — unmasked-air fraction (best + conservative + ±σ sensitivity), largest components, interior-vs-edge split, and per-64³-cell mask-collapse correlation. Writes `runs/campaigns/03-eval-v2-buggy-decode/audit/` (per_volume.csv, cells.csv.gz, results.json, findings.md, 4 figures). |
| `analysis/air_audit_v2.py` | Unmasked-air audit v2 (CPU) — redo of `eval_v2_audit.py` over all 162 saved volumes (153 eval_v2 + 9 ldm06_probe). Characterises every volume's intensity distribution properly (smoothed 256-bin histogram, prominence-based peak detection with FWHM, inter-mode valley and valley-depth bimodality index) instead of by argmax/percentiles, and derives the void threshold from each volume's own bimodal structure (Otsu between the two modes) with real-calibrated absolute and material-mode-referenced variants for comparison. Reports **both grey scales side by side** — raw `clip(v*255,0,255)` and decoder-native `clip(logit(v)*255,0,255)` — because the sampler's extra `expit` makes the dynamic-range reading scale-dependent; the two are linked by an area-preserving LUT so flagged voxel sets are identical. Keeps the earlier methodology (≥300-voxel CC filter, 32-voxel edge shell, per-64³-cell mask-collapse) and adds matched real-vs-generated interior comparison on 192³ boxes sampled inside `sample_mask`. Writes `runs/campaigns/04-measurement-limits/air_audit_v2/` (results.json, per_volume.csv, per_cell.csv, old_vs_new.csv, real_interior_boxes.json, histograms.npz, findings.md, 7 figures). |
| `analysis/ldm06_probe.py` | ldm06 design probe (GPU): (A) re-encodes high-air vs low-air 64³ cells from the saved dose-response volumes through the frozen r07-z4 VAE and compares per-channel latent moments against the train normalisation stats; (B) DDIM step sweep {50, 100, 200, 300} × 2 seeds at 192³ (joint, specimen semantics, s_por 1.5, target 0.03) measured with the audit's calibrated detector, plus an auto-gated 1024×1024×192 scale check at 200 steps. Saves volumes under `runs/campaigns/06-ldm06-probe/ldm06_probe/volumes/`; results to `runs/campaigns/06-ldm06-probe/ldm06_probe/`. |
| `analysis/onlypores_generated.py` | Ground-truth porosity of every generated volume (CPU): de-sigmoids each saved TIFF to the decoder-native u8 scale, then segments it with `poregen.dataset.segmentation.onlypores` at the exact defaults `dataset/io.compute_mask` uses for the REAL dataset — so generated and real volumes are measured the same way, independently of the model's mask head. Validates first by reproducing the stored zarr masks of three real volumes (Dice 1.000), then reports per volume and per 64³ cell: onlypores porosity (pore/sample-mask and pore/total), model-mask porosity, requested/conditioned target, and the sample-mask fraction; plus global + local dose-response OLS fits, mask-vs-onlypores discrepancy (joined to the audit's `cells.csv.gz` dark-air bins) and the DDIM-step effect. Covers all 162 volumes under `runs/campaigns/03-eval-v2-buggy-decode/volumes/` and `runs/campaigns/06-ldm06-probe/ldm06_probe/volumes/`. Writes `runs/campaigns/04-measurement-limits/onlypores_generated/` (results.json, per_volume.csv, per_cell.csv, real_validation.json, findings.md, run.log, 5 figures). |
| `analysis/onlypores_inspection.py` | Stage-by-stage inspection pack for `onlypores` on synthetic vs real grayscale (CPU). Re-implements the pipeline with every intermediate exposed (a debug path — the production functions in `dataset/segmentation.py` are untouched; the first volume is checked bit-identical against `sauvola_thresholding` and `onlypores`). For 8 generated volumes (dose_response joint_oob targets 0.005/0.02/0.05/0.10, seq 0.05, ldm06 DDIM 50/200, layup joint_oob A at 1024²) + 2 real volumes near φ 0.02 / 0.05 + their interior 192³ controls it writes: per-(volume, z-slice) 9-panel PNGs (raw / Sauvola / Otsu / material-mask before + after fill_voids / pore mask / model mask / overlay / audit dark detector), ImageJ hyperstacks of the same stages, real-vs-generated intensity histograms with the Otsu, Sauvola-local, material-mode and T_best/T_cons markers, and a sauvola_radius × sauvola_k × material-threshold-approach sensitivity sweep. Writes `runs/campaigns/04-measurement-limits/onlypores_inspection/`. |
| `analysis/onlypores_vs_truth.py` | Ground-truth harness for the inspection pack. `export` cuts a sub-region out of any generated or real volume (grayscale + current onlypores / model / detector masks + an empty template + `region_meta.json`) for hand annotation; `score` reads the finished TIFF (non-zero = pore) and reports Dice / IoU / recall / precision and implied porosity for the onlypores mask, the model's own `mask.tif` and the audit dark-voxel detector, both over the whole region and inside the sample mask. Segmentation knobs (`--sauvola-radius/-k`, `--material-approach`, `--material-component`, `--min-size-filtering`, detector threshold) are all exposed so a re-tuned onlypores can be scored against the same truth. Writes `runs/campaigns/04-measurement-limits/onlypores_inspection/ground_truth/<name>/`. |
| `analysis/_eval_v3.py` | Shared infrastructure for the eval v3 campaign (the correct-decode re-run): roots under `runs/campaigns/05-eval-v3-fixed-decode/`, the real-calibrated air detector loaded from the v2 audit (`T_abs` 182, Dice 0.842, min CC 300, edge shell 32), volume enumeration, small OLS/error helpers, and `load_u8` — the dtype-verifying loader (v3 `volume.tif` is NATIVE uint8; v2 files are float [0,1] and are rescaled, never guessed). Re-exports the v2 audit's `cc_filter` / `edge_shell` / `cellwise` / `analyse_modes` so the methodology is shared, not forked. |
| `analysis/eval_v3_ddim_probe.py` | DDIM-steps probe v3 (GPU, generation only): steps {50, 100, 200, 300} × seeds {101, 202}, joint + specimen semantics, s_por 1.5, target φ 0.03, 192³ = 8 volumes. Mirror of `ldm06_probe.py` part B on the fixed decode path. Writes `runs/campaigns/05-eval-v3-fixed-decode/volumes/ddim_probe/`. |
| `analysis/eval_v3_air_audit.py` | Air audit v3 (CPU): the calibrated ABSOLUTE threshold `T_abs = 182` applied directly — valid on generated volumes for the first time, because they now carry the real u8 grey scale. Per volume: detected/unmasked air, interior vs 32-voxel edge shell, mask capture, largest components; per 64³ cell: dark fraction vs mask collapse. Writes `runs/campaigns/05-eval-v3-fixed-decode/air_audit/`. |
| `analysis/eval_v3_onlypores.py` | onlypores porosity v3 (CPU): global + local, measured on the native u8 with NO logit inversion (verified and recorded per volume). Two controls — full real volumes reproducing the stored zarr masks (reused from `onlypores_generated.py`) and real 192³ interior crops, which expose the global-Otsu failure of onlypores' `material_mask` on small boxes (~0 porosity on known-good material). Writes `runs/campaigns/05-eval-v3-fixed-decode/onlypores/`. |
| `analysis/eval_v3_ddim_analysis.py` | DDIM-steps analysis v3 (CPU): joins the probe set to the air audit and the onlypores table; interior/edge air and porosity error vs step count, ldm06 monotone-reduction criterion. Writes `runs/campaigns/05-eval-v3-fixed-decode/ddim/`. |
| `analysis/eval_v3_compare.py` | OLD (eval v2, buggy decode) vs NEW (eval v3) head-to-head for every headline number — grey scale, dose-response fits and per-level tables, unmasked air, onlypores porosity, DDIM sweep, layup round trip. Also writes `runs/campaigns/05-eval-v3-fixed-decode/README.md`. |
| `analysis/eval_v3_run.sh` | eval v3 runner: dose-response → DDIM probe → first analysis pass → layup (last, ~4 h) → final analysis pass. Appends to `runs/campaigns/05-eval-v3-fixed-decode/run.log` (`STEP nn/NN <name> START|DONE rc=…`), ends with `EVAL_V3_DONE`. |
| `analysis/viz_orientation_volume.py` | Orientation colour volume for one XCT volume: RGB TIFF (hue = tow angle mod 180°, saturation = anisotropy, value = XCT grey), flat-colour ply-map TIFF, legend, y–z cut and ply-step verification. Writes `runs/campaigns/01-conditioning-design/orientation_viz/<volume_id>/`. |
| `analysis/recompute_vae_metrics.py` | Recompute of the two VAE eval metrics the XCT-sigmoid bug corrupted (GPU). Until 2026-09-01 `engine._run_eval` applied `torch.sigmoid()` to the XCT head's output, which regresses `xct / 255` directly — that put a ~0.135 artefact floor under `val/mae` and scaled `val/sharpness_recon_over_gt` by the sigmoid slope. Loads each run's `best.ckpt`, rebuilds model + val loader from that run's own `resolved_config.yaml`, and calls the real `engine._run_eval` on a seeded validation subset; a `_DualDecodeProbe` swapped in for `engine.decode_xct` / `engine.F` also computes the old sigmoid-path values from the same forward pass, so the buggy→fixed delta carries no eval noise. Controls (`xct_loss`, `porosity_mae`, KL, Dice) must reproduce their logged `val_full` values. Reads only; writes `runs/campaigns/07-vae-metric-recompute/vae_metric_recompute/results.json`. |
| `analysis/report_vae_metric_recompute.py` | Report generator for the above: per-run comparison tables, control verification, sweep-ranking assessment, `per_run_metrics.csv`, `findings.md` and the logged-vs-recomputed dumbbell figure (PDF + PNG 300 dpi). Writes `runs/campaigns/07-vae-metric-recompute/vae_metric_recompute/`. |
| `analysis/_real_windows.py` | Finds 64-aligned boxes lying entirely inside `sample_mask` — the shared control surface for every campaign-08/09 measurement. Because `sample_mask` is False inside the drilled holes, a box that passes is hole-free by construction rather than by a separate check. `find_window` / `find_window_best` / `find_region`, sliced over z to bound memory. |
| `analysis/vae_tile_seam.py` | Is the seam discontinuity the VAE's or the sampler's? Decodes the same posterior means three ways — A stride-64 tiled, B stride-32 Tukey-blended, C the real volume as a floor — and reports `seam_mad / interior_mad` at the 64-planes plus pore Dice and porosity. Head-agnostic: `pore_logit` returns the single logit for a binary head and `log p - log(1-p)` for the 3-class head, because a raw class logit carries a free additive constant per voxel (softmax is shift-invariant) and is not comparable across assemblies. `--checkpoint` / `--out` make it reusable per r08 rung. |
| `analysis/ddim200_1024.py` | Generates two 1024x1024x192 volumes at DDIM-200 and audits them by IMPORTING `eval_v3_air_audit.audit_one` unchanged, so the step-count answer is measured on the same instrument as the v3 campaign rather than a re-implementation of it. |
| `analysis/ply_angle_structure_tensor.py` | Structure-tensor ply-angle reading as a candidate replacement for the T-I estimators. `self_test` reads six synthetic tow bundles back to <0.5 deg before any real volume is touched. Angles are combined in the DOUBLED representation (`atan2(2*jyx, jxx - jyy)`) — averaging raw angles cancels tows at +/-90 deg, which is what an earlier version did. Conclusion: not competitive with `pore_axes` (4.2 deg / 86 %). |
| `analysis/sample_mask_component_audit.py` | Is any stored `sample_mask` built on a dust speck? (CPU, read-only.) The dataset was built before the F9 fix, when `material_mask` took the specimen box from `regionprops(labels)[0]` — raster order, not size. Reads ONE middle z-slice of `xct` + `sample_mask` per volume for all 80 volumes and reports, on that slice: the top-2 projected component areas, which component the OLD (first label) and NEW (largest area) rules pick and whether they disagree, whether the 10 % ambiguity gate fires, the stored vs recomputed bounding box, and Dice. `first_is_largest` and the bbox comparison are the decisive columns; `dice` is weaker because production Otsu is volume-global while one slice gets its own threshold. Exit code 1 if anything is flagged. Writes `runs/campaigns/12-sample-mask-audit/`. |
| `analysis/r08_calibration_probe.py` | Is a rung's dense-panel error calibration or capacity? One forward pass, scored three ways — `argmax`, `deweight` (divide the softmax by the training class weights and renormalise) and a `tau` sweep on p(pore). If de-weighting improves BOTH porosity MAE and pore Dice the loss weighting was mis-calibrated; if it hurts, the residual is capacity. Has a `--device cpu` path so it can overlap the next rung on the GPU. Writes `runs/campaigns/09-r08-latent-sweep/calibration_probe_r08_<variant>/`. |
| `analysis/decoder_ft_redecode.py` | The D43 decoder fine-tune gate: two decoders on the SAME latents. Val arm re-decodes real val patches through the frozen encoder (ground truth exists, so pore Dice and porosity MAE are real); generated arm re-decodes ldm06 latents saved by `--save-latents` (no ground truth, so sharpness is a ratio against real and segmentation is compared BETWEEN decoders). Refuses to report anything if the encoder moved between the two checkpoints — the latent store would be stale and the comparison meaningless. Assembles full volumes before measuring S2, because S2 needs 128-cubed windows of contiguous material that do not exist in a pile of 64-cubed patches, and scores both decoders inside the BASELINE's material mask so the same analysis windows are used for each. |
| `analysis/r08_rung_report.py` | Per-rung full-split report (`--run`), and the sweep decision table (`--compare`). `--compare` builds ONLY from artefacts that already exist — the run's own final `val_full` / `test_full` in `metrics.jsonl` (which are whole-split evaluations carrying the per-bin table and counts), the CPU calibration probe, the tile-seam and the latent-sanity summary — so the decision costs no GPU. A missing artefact renders as an em-dash and the footnote says a blank means 'not produced', not 'zero'. |

**Orchestration (long unattended chains)**

These exist because a missed process exit cost 3 h 31 m of GPU: rf-8 finished at 09:51 and the next rung was not launched until 13:22. A human or an agent noticing an exit is not a scheduling mechanism.

| Path | What it does |
|---|---|
| `r08_queue.sh` | Sequential r08 sweep runner. Starts the next rung the instant the previous process exits, pass or fail; a FAILED rung is logged and the chain continues, and only a rung that cannot be launched at all stops it. GPU end-of-run reports (tile-seam, mask sanity) run in the GAP between rungs with nothing else on the card; the calibration probe has a CPU path and is backgrounded to overlap the next rung. Appends every transition with an rc to `runs/campaigns/09-r08-latent-sweep/queue.log`. NOTE: bash reads a script by file offset, so editing this file while it runs is undefined — replace it by rename and RESTART the runner, then check `/proc/<pid>/fd/255` points at the new inode. |
| `ldm06_post.sh` | Everything the GPU does after ldm06/base exits, armed in advance: gated eval-v4 generation (`--save-latents`, original decoder) -> decoder fine-tune -> re-decode gate -> owed rung reports -> rf-2 -> rf-64. The eval-v4 block is gated on `runs/campaigns/ldm06_go`, read ONCE when ldm06 exits so a late touch cannot retro-insert hours of generation ahead of the fine-tune. |
| `r08_queue_tail.sh` | The second runner: rf-2 and rf-64, which bracket the sweep for the paper but do not feed the compressor decision, plus the full-split rung reports the first runner logged as OWED. Waits for the GPU to go quiet, so it must be started AFTER ldm06/base is training. `run_dir_for` resolves a rung by `experiment_id` and requires a `best.ckpt` — `base` has three runs (a crash, the untempered attempt, the tempered one) and the run index does not distinguish them. |
| `ldm06_bringup.sh` | Everything between 'the compressor is chosen' and 'ldm06 is training': latent store -> conditioning -> `LatentDataset` verification -> launch, each stage verified BEFORE the next begins, because every failure mode here is a missing file the next step would happily build around (a half-written store still loads, a stale conditioning sidecar still joins). Sets `ldm06/base`'s `data.latents_root` to the chosen rung's store as one explicit commit of that single file — so which store a run read stays answerable from git — and refuses on any other mismatch. |

**Evaluation**

| Path | What it does |
|---|---|
| `eval_r03.py` | 🕰 R03-specific post-training eval (full-volume recon, S2/PSD, GIFs, latent audit). Largely superseded by the `poregen.eval` package. |
| `eval_checkpoint.py` | 🕰 Older full-volume checkpoint eval; requires a legacy flat `--config` YAML. |

---

## configs/

Hierarchical YAML. Resolution order (`extends` → `components` → body → `overrides`) is documented in
`AGENTS.md`; the resolver is `src/poregen/configuration/experiments.py`.

**Building blocks**

| Path | What it does |
|---|---|
| `configs/models/vae/*.yaml` | Architecture presets: `v2_conv_noattn`, `v2_vrrae`, `v2_vrrae_linear`. |
| `configs/losses/*.yaml` | `vae_charbonnier_mask_kl` (with mask head), `vae_charbonnier_kl` (XCT-only). |
| `configs/datasets/split_v{1,2}.yaml` | Data root, split version, loader backend. |
| `configs/training/vae_default.yaml` | Step budget, eval/save cadence, optimiser and schedule defaults. |
| `configs/machines/dgx_spark.yaml` | GB10 worker/prefetch limits (see `AGENTS.md`). |
| `configs/profiles/{fast,repro}.yaml` | Throughput vs determinism profile (compile, preflight, seeding). |
| `configs/eval/{minimal,full,r03_base,r03_paper}.yaml` | Eval tiers consumed by `poregen.eval.config.load_eval_config`. |

**Experiment families**

| Family | State | What it is |
|---|---|---|
| `r03/` | 🕰 historical, still the root of the `extends` chain | Production baseline: `v2.conv_noattn`, z=16, split_v2. `split_v1.yaml` is the comparison run. |
| `r04/` | 🕰 historical | R03 + dual-branch encoder + focal mask loss + LSGAN discriminator (`disc_weight=0.01`). |
| `r05/` | **current sweep** | R04 with `free_bits=0.1`, `disc_weight=0.05`. `reduction-factor-{2,8,16,32,64}` sweep latent compression. |
| `r06/` | **current sweep** | XCT-only (`v2.conv_noattn_xctonly`, no mask head) compression sweep, same rungs plus `-4`. |
| `r07/` | 🕰 supplied the ldm05 latents — `latents_r07z4` stays live until the ldm06 store is built | Full AE (`in_channels=2`, both decoder heads) on the dual-branch trunk. `r07/reduction-factor-16` (z=4) produced `r07-run-0006-…-z4-c32-…`, the VAE behind `latents_r07z4`. The rung name is the TOTAL VOXEL reduction, not the channel count: `reduction-factor-4` is z=16, `-16` is z=4. |
| `r08/` | **current VAE** — will supply the ldm06 latents | The 3-class VAE for ldm06: `v2.conv_noattn_dualbranch_cls` on `data/split_v3`, encoder `cat([xct, pore, air])`, decoder emitting material/pore/air logits. Extends `r07/reduction-factor-16`, so every other hyperparameter is that run's. `loss.class_weights` comes from `data/split_v3/class_weights.json`. Full compression sweep mirroring r07 — `reduction-factor-{2,4,8,32,64}` at z={32,16,8,2,1}; **`base` IS the 16x rung** (z=4), so there is no `reduction-factor-16`. |
| `vrrae/`, `vrrae03/`, `vrrae04/` | 🕰 historical ablation | Flat SVD rank-reduction bottleneck (`vrrae03`) vs its plain-`Linear` twin (`vrrae04`). `vrrae/smoke100.yaml` is a self-labelled throwaway 100-step smoke test. |
| `ldm06/` | **current LDM rung** | The r08 3-class latent store (`data/split_v3/latents_r08z8`), six per-face distances, always-on material map, neighbours noised to their own timestep, hybrid chunked joint sampling with blended decode. 155 input channels at the live z=8 (the count follows the rung: 127 at z=4). `aux.yaml` extends it and turns ON the decoded-space auxiliary loss (`loss.decoded`): the lowest-quartile-t x̂₀ estimates are decoded through the frozen r08 VAE and scored on air placement, pore Dice, delivered porosity and grey/label agreement. |
| `ldm06/eps` | ablation | `ldm06/base` with `objective: eps`, `zero_terminal_snr: false`. Exists to MEASURE what the objective is worth, not to be trained on: the ε form divides by `sqrt(ᾱ_T) = 6.12e-17` at the terminal step and the ±10 clamp saturates x̂₀, which is the floor under `gen/x0_clamp_sat_frac`. There is no `ldm07/` — v-prediction was folded into `ldm06/base` because the defect is present from step one. |
| `ldm05/` | 🕰 historical | First conditional LDM (D32): orientation profile as input channels, porosity/depth/one distance scalar by FiLM, touching (offset-64) neighbours, eight-group parity assembly. Kept as the provenance of the `runs/ldm/ldm05-*` results; it names model switches and a latent store the current code no longer has, so it will not launch. |
| `ldm04/` | 🕰 historical | First LDM on the normalised r07z4 store: z=4, 16³ latents, **unconditional**. Same caveat as ldm05. ldm01–ldm03 configs were removed; only their `runs/ldm/` output remains. |

Reduction factor convention across r05/r06/r07: `X = 64³ / (z_channels · 16³) = 64 / z_channels`
at the fixed f=4 spatial downsampling.

---

## tests/

Run with `pytest tests/`. Known pre-existing failures are listed in `AGENTS.md`.

| Path | Covers |
|---|---|
| `test_config_loading.py`, `test_train_vae_migration.py` | Config loading, experiment resolution, run naming, cloning. |
| `test_volume_split_counts.py`, `test_split_dataset_roots.py` | Deterministic and stratified split assignment, split roots. |
| `test_patch_coords_count.py`, `test_integral_porosity.py`, `test_dataset_loader_shapes.py` | Patch coordinates, integral-volume porosity, dataset tensor shapes/ranges. |
| `test_extract_patches_resume.py` | `extract_patches_memmap.py` interrupted after one volume and resumed: the metadata label fractions must match a single uninterrupted run, and the patches themselves must be identical. |
| `test_checkpoint_async.py` | The async checkpoint writer and the discriminator round trip: a weight mutated after `save_async` returns must NOT reach the file (the writer gets a CPU clone taken on the calling thread), a failed background write re-raises on the next join, and resume restores the discriminator plus its optimizer state. A checkpoint predating that state WARNS rather than raises, so the running run stays resumable. |
| `test_segmentation_material_mask.py` | `material_mask` picks the LARGEST projected component, not the first-labelled one — a corner speck must not become the specimen — and raises when the second-largest exceeds `AMBIGUOUS_COMPONENT_RATIO` (10 %), because two comparable objects mean no single box is the specimen. |
| `test_vae_output_shapes.py`, `test_losses_smoke.py` | VAE forward shapes; loss finiteness. |
| `test_vrrae_vae.py`, `test_vrrae_linear.py`, `test_vrrae_bottleneck.py`, `test_vrrae_finetune.py` | VRRAE family: shapes, gradients, registry, fixed-basis round-trip. |
| `test_recon_metrics.py`, `test_latent_metrics.py` | Recon metrics + eval loop wiring; latent moment merging and active units. |
| `test_early_stopping.py`, `test_patch_sample_export.py` | `train_loop` early-stopping path; TIFF sample export, including the 3-class head exporting `argmax == pore` plus `label_recon` while a binary head exports neither. |
| `_ldm06_store.py` | Not a test — the miniature ldm06 latent store several test modules build on (patch 16, latent 4³, every patch a crop of one known field). |
| `test_material_maps.py` | The material-map arithmetic the encoding pass performs: batched block-mean pooling, uint8 round-trip, `material + pore + air = 1`, and `compute_sample_mask`. |
| `test_ldm06_conditioning.py` | The conditioning data side: orientation encoding/pooling rules, the touching-neighbour leak guard, `cond_dist6` (shared helper AND `build_conditioning.build_scalars` on a synthetic extent), and the `LatentDataset` batch contract. |
| `test_ldm06_training.py` | The 127-channel input accounting, one test per conditioning input proving it reaches the output, `noise_neighbours` (t_nb draw, mix rate, `drop_nb` null) and the train/eval step wiring. |
| `test_volume_generator_hybrid.py` | The hybrid sampler: `_neighbour_plan` source logic per face, the in-chunk / finished-chunk / OOB / UNKNOWN cases end to end, per-window conditioning, nested CFG passes, blended decode weight normalisation, and seams at the window vs chunk period. |
| `test_cfg_guidance.py` | The nested CFG decomposition: telescoping at s=1, both null arms carrying no neighbour information, and the `ldm06/base` config resolution. |
| `test_decoded_aux_loss.py` | The decoded auxiliary loss: each term zero when the decode is consistent and positive when one thing is broken, the lowest-t sub-batch selection and its hard cap, the ramp, gradient reaching the model output but NOT the VAE parameters or its BatchNorm buffers, the `label` served only when asked for, and `ldm_train_step` reporting every term separately. |
| `test_vpred_schedule.py` | The v objective: v↔x₀↔ε round-trips, the exactly-zero terminal ᾱ and the measured size of that rescale (a near no-op here — recorded so a result is not attributed to the wrong cause), the ε form blowing up at the terminal step while v stays order 1, DDIM reaching x₀ under both objectives, CFG telescoping in v-space, and `rescale_guidance`. |
| `test_blended_reconstruction.py` | Tukey-window blended reconstruction. |
| `test_eval_v4.py` | eval v4 on data whose answer is worked out by hand: the manifest contract (required fields, self-consistency, a metric refusing a volume it does not describe), the within-volume local metric on labels with an exactly-set porosity per tile — including the two-volume case that shows why the pooled R² is not obedience — seam plane selection at both periods, layup scoring on synthetic angles through the real T-I scoring maths, and geometry Dice. One test runs the metrics on a real campaign-05 volume. |
| `test_eval_v4_assembly_modes.py` | The `assembly_modes` assessment on volumes with a planted step at a known plane: that the step is charged to the right CHUNK INDEX and the right plane FAMILY (chunk vs tile), that the chunk-plane MAD recovers the step size, that tile planes are kept out of the interior baseline, that per-chunk porosity reads the block it names, that a destroyed join reads a larger S2 across-vs-inside distance, that the three neighbour modes reach a spy denoiser as the availability codes they claim, that the reference-block search covers both lattice phases and REFUSES a shape no real volume holds, and that measure -> results.json -> findings.md runs end to end (an assessment with no reporter is silently dropped from the findings). |
| `test_porosity_field.py` | Coherent porosity field: mean-to-target, clamp range, per-seed determinism, x-vs-z anisotropy, marginal spread, and every branch of `_build_local_por_map` in `generate_volumes.py` plus `_gaussian_por_grid` (both deliberately leave peaks outside the sampler's clamp). Uses the real T-E/T-D artefacts. |

---

## docs/

| Path | What it covers |
|---|---|
| `docs/CODEMAP.md` | This file. |
| `docs/ARCHITECTURE.md` | Config system, data pipeline, decoder output contract, run structure, key invariants — what the project *is*. |
| `docs/DEVELOPMENT.md` | Install, tests (including the known pre-existing failures), training commands, GPU deployment target. |
| `docs/vae_architecture.md` | VAE model family and the `engine.py` train/eval data flow. |
| `docs/ldm_training.md` | How to launch, resume, and babysit an LDM run (tmux recipes). |
| `docs/metrics_guide.md` | Every training/eval metric: formula, healthy range, what to do. (Spanish.) |
| `docs/eval_methodology.md` | eval v4: the manifest contract, every metric, the eight assessments and the real-volume floor — the paper methods reference. |

---

## data/ layout

```
data/
├── split_v1/                     🕰 v1 deterministic split, Zarr only
│   ├── volumes.zarr/  patch_index.parquet  splits.json  volume_stats.json
├── split_v3/                     ← current dataset (r08 VAE, ldm06)
│   ├── volumes.zarr/             → symlink to split_v2/volumes.zarr (never a copy)
│   ├── patch_index.parquet       ← split_v2 columns + panel_id + air_fraction,
│   │                               hole-touching patches removed
│   ├── splits.json               ← BY PANEL: test = Na_05 + JI_8, val = Na_08 + JI_12,
│   │                               train = the rest (Pegaso is one panel, train-only)
│   ├── holes/<volume_id>.npy  holes.json   ← per-volume dilated 2-D hole footprint
│   ├── patches_xct.bin  patches_label.bin  patches_meta.json  ← memmap backend
│   │                               (label: 0 material, 1 pore, 2 air)
│   ├── index_report.json  build_report.md
│   └── latents_r08z8/            ← LIVE latent store (memmap format, ldm06)
│       ├── metadata.json         latent shape, VAE provenance, per-channel train stats,
│       │                         voxel_size_um, `material` + `conditioning` blocks
│       │                         (`conditioning.geometry` carries the strides
│       │                          LatentDataset validates the run config against)
│       └── {train,val,test}/latents.bin (float16, mu_then_std) + index.parquet
│                                 + cond.parquet (cond_depth / cond_dist6_{zm,zp,ym,yp,
│                                   xm,xp} / cond_por_raw)
│                                 + material.bin (uint8 16³ envelope fractions, label != 2)
│                                 + air.bin (float32 air fraction, label 2)
│                                 — every file row-aligned with index.parquet
├── split_v2/                     🕰 previous dataset — the orientation field still lives here
│   ├── volumes.zarr/  patch_index.parquet  splits.json  volume_stats.json
│   ├── orientation_field.json    ← per-volume θ(z) from the NOMINAL layup, with
│   │                               confidence flags + provenance (build_conditioning.py).
│   │                               Per-VOLUME, so split_v3 reuses it unchanged.
│   └── latents_r07z4/            🕰 the ldm05 store; the current code cannot read it
│                                 (one cond_dist, no material sidecar)
└── real_test_volumes/            held-out volumes for generation evaluation
```
`raw_data/` holds the source TIFFs. `data/` and `raw_data/` are git-ignored.

## runs/ layout

```
runs/
├── vae/<experiment>-run-<idx>-<timestamp>-<config slug>/
│   ├── log.jsonl  metrics.jsonl  summary.json
│   ├── resolved_config.yaml  run_metadata.json
│   ├── tb/                      TensorBoard events
│   ├── samples/                 exported patch samples (TIFF)
│   ├── best.ckpt  latest.ckpt  <run>_stepNNNNNNNN.ckpt
│   └── eval/<timestamp>-<tier>/ produced by poregen.eval
├── ldm/<...>/                    same, with checkpoints under checkpoints/
├── diagnostics/                  monitoring output and data-build logs — NOT analyses
│   ├── mask_sanity_r07z4/        scripts/diag_mask_sanity.py (D33 decoder gate)
│   ├── ldm_samples/              scripts/diag_ldm_samples.py, per run/step sample grids
│   └── build_latent_dataset_full.log
└── campaigns/                    analysis output — ONE CAMPAIGN PER QUESTION
    ├── INDEX.md                  the campaign list + the convention. Read this first.
    ├── AUDIT.md                  read-only audit (2026-09-02) that produced this tree
    ├── 01-conditioning-design/   T-A…T-I, SUMMARY.md, orientation_viz/.
    │                             ⚠ T-G and T-H are wrong — superseded by T-I.
    │                             T-D/results.json + T-E/results.json are LIVE
    │                             production inputs (porosity_field.py); so are
    │                             T-I/layup_field.json and T-A/fine_profiles.npz.
    ├── 02-porosity-control-v1/   joint_vs_sequential/, dose_response/,
    │                             dose_response_spor15/, cfg_sweep/,
    │                             layup_roundtrip/ (+ run.log), void_mask_audit/.
    │                             🕰 superseded by 03 and 05 for every headline.
    ├── 03-eval-v2-buggy-decode/  eval v2 — BUGGY DECODE (sampler applied a spurious
    │                             expit; volume.tif is float [0.52, 0.73]). Kept as the
    │                             record: README.md, volumes/{cfg_sweep,layup}/,
    │                             dose_response/, cfg_sweep/, layup/, audit/
    ├── 04-measurement-limits/    air_audit_v2/ (real-volume detector calibration —
    │                             CURRENT, imported by _eval_v3.py), onlypores_generated/,
    │                             onlypores_inspection/
    ├── 05-eval-v3-fixed-decode/  eval v3 — correct decode, volume.tif NATIVE uint8 on the
    │                             raw-scan scale. AUTHORITATIVE. README.md,
    │                             volumes/{dose_response,ddim_probe,layup}/, dose_response/,
    │                             air_audit/, onlypores/, ddim/, comparison/, run.log
    ├── 06-ldm06-probe/           ldm06_probe/ — Part A provisional, Part B superseded by 05
    ├── 07-vae-metric-recompute/  vae_metric_recompute/ — results.json only; report not run
    ├── 08-pre-ldm06-diagnostics/ vae_tile_seam/, ddim200_1024/, ply_angle_structure_tensor/
    └── 09-r08-latent-sweep/      queue.log (every rung transition with an rc),
                                  r08_<variant>/tile_seam/, calibration_probe_r08_<variant>/,
                                  decision_table.md, ldm06_bringup.log
```
Every campaign carries a `README.md` (question, exact command, checkpoint and
settings, headline numbers, caveats, vault note) and a row in `INDEX.md`.

`runs/` is git-ignored — the committed script in `scripts/analysis/` is the only
provenance a result has. `runs/ldm/` still contains ldm01–ldm03 output whose
configs no longer exist.

## Other top-level directories

| Path | What it is |
|---|---|
| `third_party/RR_layer/` | Vendored truncated-SVD rank-reduction layer used by the VRRAE bottleneck (pinned commit). |
| `notebooks/` | Training / eval / R03 analysis notebooks. |
| `onlypores.py` | 🕰 Root copy of the segmentation code; the package version is `poregen/dataset/segmentation.py`. |
| `inference/`, `diagnostics/`, `eval_results/`, `best_ddim_300/`, `demo_render/`, `logs/` | Output and ad-hoc analysis artefacts from earlier rungs (mostly git-ignored). |

Markers: **⚠** stale/broken assumption · **🕰** historical, superseded but kept · **💀** dead or throwaway.

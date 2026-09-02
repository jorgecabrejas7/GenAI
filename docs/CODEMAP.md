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
| 2. Patch index → flat memmap (fast loading) | `scripts/extract_patches_memmap.py` | `data/split_v3/patches_{xct,label}.bin` + `patches_meta.json` |
| 3. VAE training | `scripts/train_vae.py` → `poregen.cli.experiments:main` → `poregen.experiments.train_vae` → `poregen.training.engine.train_loop` | `runs/vae/<run>/` |
| 4. VAE → latent store | `scripts/build_latent_dataset.py` | `data/split_v2/latents_r07z4/` (live store) |
| 4b. Per-patch conditioning (ldm05) | `scripts/build_conditioning.py` | `data/split_v2/orientation_field.json`, per-split `cond.parquet`, `conditioning`/`assembly` blocks in the store `metadata.json` |
| 4c. Per-patch material maps (ldm06) | `scripts/build_material_maps.py` | per-split `material.bin` + `air.bin` beside the latents, `sample_mask` arrays in `volumes.zarr`, `material` block in the store `metadata.json` |
| 5. LDM training | `scripts/train_ldm.py` → `poregen.cli.experiments:main_ldm` → `poregen.experiments.train_ldm` → `poregen.training.ldm_engine.train_loop` | `runs/ldm/<run>/` |
| 6. Generation | `scripts/generate_volumes.py` (uses `poregen.diffusion.sampler.VolumeGenerator`) | TIFF volume grid per porosity/layout |
| 7. Evaluation — VAE | `poregen.eval.runner.run_eval` (driven from the experiments CLI) | `runs/vae/<run>/eval/<timestamp>-<tier>/` |
| 7b. Evaluation — generated volumes | `scripts/eval_generated_volumes.py` ⚠ stale | `eval_results/` |

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
| `dataset/segmentation.py` | Pore/material segmentation (Sauvola + Otsu, fill-voids). Package home of the root `onlypores.py`. `compute_sample_mask` re-derives just the material envelope (no Sauvola). |
| `dataset/material.py` | ldm06 material maps: block-mean pooling of `sample_mask` to latent-resolution fraction cells, uint8 encode/decode, air fraction. |
| `dataset/patch_index.py` | Patch coordinate generation, 3-D integral volume for O(1) patch fractions (`patch_fractions` — porosity and, in split_v3, air fraction), Parquet index writer. |
| `dataset/splits.py` | Volume-level splits: deterministic `v1` and stratified-by-porosity `v2`; writes lightweight split roots. The `v3` split is by PANEL and lives in `scripts/build_split_v3.py`. |
| `dataset/holes.py` | Finds the three drilled registration through-holes of a coupon from `sample_mask` (z-MINIMUM projection of the complement, border-touching components dropped, Euclidean dilation) and marks the patches that touch one. |
| `dataset/loader.py` | `PatchDataset` (Zarr backend), `MemmapPatchDataset` (`.bin` backend), `build_label` (3-class voxel label), `zarr_worker_init_fn` for fork-safe workers. Both backends return `label` (int64 `{0,1,2}`) and the derived binary pore `mask` (`label == 1`). |

### models

| Path | What it does |
|---|---|
| `models/nn/blocks.py` | Shared 3-D conv blocks: v1 family (GroupNorm/SiLU/ConvTranspose3d) and v2 family (BatchNorm3d/GELU/Upsample+Conv), plus `reparameterize`. |
| `models/vae/base.py` | `VAEConfig` and `VAEOutput` dataclasses. |
| `models/vae/registry.py` | `@register_vae(name)` / `build_vae(name, **kw)` / `list_vaes()`. |
| `models/vae/v1/{conv,conv_noattn,unet}.py` | 🕰 First-gen architectures `conv`, `conv_noattn`, `unet`. Kept for old-checkpoint compatibility; not used by current experiments. |
| `models/vae/v2/conv.py`, `v2/unet.py` | Second-gen attention and skip-connection variants (`v2.conv`, `v2.unet`). |
| `models/vae/v2/conv_noattn.py` | `v2.conv_noattn` — R03 baseline trunk; also the `v2.conv_noattn_xctonly` (mask head removed) used by r06. |
| `models/vae/v2/conv_noattn_dualbranch.py` | `v2.conv_noattn_dualbranch` — dual encoder branch + fusion. **Current VAE** (r04/r05/r07). |
| `models/vae/v2/vrrae.py` | `v2.vrrae` — VRRAE-bottleneck VAE, XCT-only encoder and decoder (no mask head). |
| `models/vae/v2/vrrae_bottleneck.py` | The bottleneck itself: flatten → FC → truncated-SVD RR layer (vendored) → identity posterior mean. |
| `models/vae/v2/vrrae_linear.py` | `v2.vrrae_linear` — SVD ablation twin: same flat bottleneck, plain `Linear` heads. |
| `models/vae/v2/vrrae_finetune.py` | Fixed-basis extraction at inference + decoder-only fine-tune hook for VRRAE. |
| `models/discriminator.py` | 2-D multi-plane PatchGAN + LSGAN losses for adversarial XCT supervision (R04+). Runs in float32 — see `AGENTS.md`. |
| `models/diffusion/unet.py` | `UNet3DDenoiser` / `UNet3DConfig` — the LDM denoiser. Spatial conditioning (orientation profile + whole face-adjacent neighbour latents + availability embeddings) is concatenated at the input; scalars (`cond_por`, `cond_depth`, `cond_dist`) are FiLM/AdaGN. All switchable; there is no global-porosity input. |
| `models/diffusion/blocks.py` | GroupNorm/SiLU res-blocks, attention, sinusoidal time embedding for the denoiser. |

### losses / metrics

| Path | What it does |
|---|---|
| `losses/recon.py` | XCT recon losses in z-score space: `l1`, `mse`, `charbonnier`, `get_recon_loss`. |
| `losses/mask.py` | Mask losses on logits: BCE, Dice, Tversky, focal, `combined_mask_loss`. |
| `losses/kl.py` | KL with free-bits (per-channel and flat) + `beta_schedule`. |
| `losses/total.py` | `compute_total_loss` — composes recon + mask + β·KL. Gotchas live in `AGENTS.md`. |
| `metrics/recon.py` | MAE, MSE, PSNR, sharpness proxy. |
| `metrics/seg.py` | Vectorised Dice / precision / recall, porosity metrics, binned porosity MAE. |
| `metrics/latent.py` | KL per channel, active units, streaming channel moments and their merge. |

### training

| Path | What it does |
|---|---|
| `training/engine.py` (~1.3k lines) | VAE `train_step` / `eval_step` / `train_loop`: AMP, discriminator, EMA, eval cadence, early stopping, TensorBoard + JSONL logging, sample export. See `docs/vae_architecture.md`. |
| `training/ldm_engine.py` | LDM equivalent: EMA, ε-prediction step, eval loop, in-training generation eval. See `docs/ldm_training.md`. |
| `training/data.py` | `build_dataloader_kwargs`, `build_patch_dataloaders` (train/val/test from the Zarr or memmap backend). |
| `training/checkpoint.py` | Atomic checkpoint save/load (model, optimizer, scaler, scheduler, EMA, RNG), sync + async variants. |
| `training/device.py` | Device selection, autocast dtype, GradScaler. |
| `training/seed.py` | `seed_everything`. |
| `training/sample_export.py` | Writes saved 3-D patch samples as ImageJ-readable TIFF stacks; migrates legacy `.npz` archives. |

### diffusion

| Path | What it does |
|---|---|
| `diffusion/latents.py` | Load side of the latent store: `LatentDataset` + `build_latent_dataloaders`; reads `metadata.json` + per-split `latents.bin`/`index.parquet`/`cond.parquet`, applies the train-split normalisation, and returns the fixed ldm05 batch contract (scalars, `cond_orient`, unshifted touching neighbours + availability); behind `material=True` (ldm06, default off) also `cond_material`/`air_fraction`/`nb_air_fraction` and the optional all-air `air_patch_cap`. Keeps `sample_stride` (32, data multiplier) / `generation_stride` (64) / `neighbour_offset` (64) separate and enforces `neighbour_offset >= patch_size`. |
| `diffusion/noise_schedule.py` | `DDPMSchedule` — cosine schedule, ε-prediction, movable to device without being an `nn.Module`. |
| `diffusion/sampler.py` | `DDPMSampler`, `DDIMSampler`, `VolumeGenerator` with two denoising modes — sequential (eight-group parity schedule, whole unshifted neighbours) and joint (MultiDiffusion-style: overlapping stride-32 windows, per-timestep cosine-weighted ε fusion on one latent canvas, neighbours all-UNKNOWN) — both decoded on the stride-64 tiling grid, no blending; plus `porosity_to_cond`, `theta_from_layup` and `seam_discontinuity` (the assembly-quality metric: patch-to-patch face jump vs the interior slice-to-slice baseline). |
| `diffusion/conditioning.py` | Single source of truth for the shared conventions: neighbour direction order, availability states (OOB/EXISTS/UNKNOWN), the grid index, the eight-group parity schedule and its fixed ordering, the `neighbour_offset >= patch_size` leak guard, and the (ablation-only) neighbour→target-frame shift. Also the standalone availability embedding. |
| `diffusion/orientation.py` | `OrientationField` (reads `data/split_v2/orientation_field.json`) + the `(cos2θ, sin2θ)` encoding: pool the components, never the angle, and never renormalise the pooled vector. Shared by dataset and sampler. |
| `diffusion/generation_eval.py` | In-training LDM diagnostics: off-manifold latent moments, x0-clamp fraction, decode-based porosity vs the real val split. |
| `diffusion/porosity_field.py` | Coherent per-patch porosity field for inference (D32 §4): T-E marginal draw per grid cell, anisotropic Gaussian smoothing with the T-D volume-mean-removed correlation lengths, mean rescale to the global target, clamp to the training phi range [0.002, 0.107]. Loaders for the T-E sampler and T-D lengths from their `results.json`. Used by `scripts/generate_volumes.py` distribution `"coherent"`. |

### experiments / runtime

| Path | What it does |
|---|---|
| `experiments/train_vae.py` | Config-driven VAE run: build model, dataloaders, run dir, `train_loop`; `run_experiment` and `resume_run`. |
| `experiments/train_ldm.py` | Same for the LDM. Also pulls the porosity standardisation stats and the parity group ordering out of the latent store metadata. |
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

---

## scripts/

**Dataset building**

| Path | What it does |
|---|---|
| `extract_patches_memmap.py` | One-time Zarr → flat `patches_{xct,label}.bin` memmap extraction, row-aligned with `patch_index.parquet`. `patches_label.bin` holds the 3-class voxel label (0 material, 1 pore, 2 air = `sample_mask == 0`); air takes precedence over pore. Refuses to start when the filesystem cannot hold both arrays. |
| `build_latent_dataset.py` | Encodes every patch with a trained VAE → raw `mu`/`std` memmap + train-split normalisation stats. Builds the live `latents_r07z4` store. |
| `build_material_maps.py` | ldm06 (D40 §1): per-patch material maps (uint8, latent resolution) + air fractions for an EXISTING latent store, as sibling `material.bin`/`air.bin` memmaps; persists each volume's `sample_mask` into `volumes.zarr` (compressed); resumable per volume; no latents re-encoded. |
| `build_conditioning.py` | Builds the ldm05 conditioning data: `data/split_v2/orientation_field.json` (nominal θ(z) per volume, with T-I confidence flags and a full re-verification of the boundary↔ground-truth-angle correspondence) and the per-split `cond.parquet` sidecar (`cond_depth`, `cond_dist`, `cond_por_raw`), plus the `conditioning` / `assembly` metadata blocks. |

**Training entry points**

| Path | What it does |
|---|---|
| `train_vae.py`, `train_ldm.py`, `experiments.py` | Thin wrappers around `poregen.cli.experiments:main` / `:main_ldm`. Same as the installed `train_vae` / `train_ldm` / `experiments` console scripts. |

**Generation**

| Path | What it does |
|---|---|
| `generate_volumes.py` | Sweeps porosity × spatial layout, generates anisotropic volumes with stride-32 tiling, writes `volume.tif` / `mask.tif`. |
| `visualize_denoising.py` | MP4 of a DDIM chain for one patch, decoding every intermediate latent. |

**Diagnostics / profiling**

| Path | What it does |
|---|---|
| `diag_ldm_samples.py` | Standard LDM convergence health-check: samples with REAL val conditioning split by EXISTS-neighbour bucket (0/2/4/6), {raw, ema} × DDIM-50 latent moments + decoded porosity per bucket, conditioning-alive probe (FULL vs neutralised inputs on the total-noise MAD scale), trend verdict (CONVERGING/STALLED/REGRESSING) appended to `<run_dir>/convergence_check.jsonl`. Handles unconditional runs (single bucket). `--ddim200` adds the 200-step variants. |
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

Campaign map: `01-conditioning-design` (`t_*`, `viz_orientation_volume`) ·
`02-porosity-control-v1` (`dose_response`, `cfg_sweep`, `layup_roundtrip`,
`void_mask_audit`, `joint_vs_sequential`, `regen_layup_volumes`) ·
`03-eval-v2-buggy-decode` (`eval_v2_*`) · `04-measurement-limits`
(`air_audit_v2`, `onlypores_*`) · `05-eval-v3-fixed-decode` (`eval_v3_*`) ·
`06-ldm06-probe` (`ldm06_probe`) · `07-vae-metric-recompute`
(`recompute_vae_metrics`, `report_vae_metric_recompute`).

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

**Evaluation**

| Path | What it does |
|---|---|
| `eval_generated_volumes.py` (~1.8k lines) | Stage-4 generation quality suite: porosity MAE, PSD W1, S2(r), Ripley's K, FID, memorisation. ⚠ Its memorisation step expects the **old** latent layout (`latents_s64_sampled/latents_meta.json` + one flat `latents.bin`); the live store is `data/split_v2/latents_r07z4` with `metadata.json` and per-split files. It warns and skips rather than failing. |
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
| `r07/` | **current sweep — supplies the live latents** | Full AE (`in_channels=2`, both decoder heads) on the dual-branch trunk. `r07/reduction-factor-4` (z=4) produced the VAE behind `latents_r07z4`. |
| `vrrae/`, `vrrae03/`, `vrrae04/` | 🕰 historical ablation | Flat SVD rank-reduction bottleneck (`vrrae03`) vs its plain-`Linear` twin (`vrrae04`). `vrrae/smoke100.yaml` is a self-labelled throwaway 100-step smoke test. |
| `ldm05/` | **current LDM rung** | First conditional LDM (D32): orientation profile as input channels, porosity/depth/distance by FiLM, touching (offset-64) neighbours, eight-group assembly on a stride-64 tiling grid. Trained from scratch, not warm-started from ldm04. |
| `ldm04/` | previous rung | First LDM on the normalised r07z4 store: z=4, 16³ latents, **unconditional** (neighbour/position/porosity conditioning present in code but off). ldm01–ldm03 configs were removed; only their `runs/ldm/` output remains. |

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
| `test_vae_output_shapes.py`, `test_losses_smoke.py` | VAE forward shapes; loss finiteness. |
| `test_vrrae_vae.py`, `test_vrrae_linear.py`, `test_vrrae_bottleneck.py`, `test_vrrae_finetune.py` | VRRAE family: shapes, gradients, registry, fixed-basis round-trip. |
| `test_recon_metrics.py`, `test_latent_metrics.py` | Recon metrics + eval loop wiring; latent moment merging and active units. |
| `test_early_stopping.py`, `test_patch_sample_export.py` | `train_loop` early-stopping path; TIFF sample export. |
| `test_latent_dataset.py` | `LatentDataset` normalisation and memmap unpacking. |
| `test_material_maps.py` | ldm06 material maps: sample_mask pooling, uint8 store round-trip through `build_material_maps.py`, `LatentDataset` material flag/shapes, neighbour air fractions, all-air patch cap. |
| `test_ldm05_conditioning.py` | ldm05 conditioning data side: orientation encoding/pooling rules, the touching-neighbour geometry and its leak guards, the eight-group availability schedule, scalar ranges, and the batch contract (synthetic store + the built artefacts). |
| `test_cfg_guidance.py`, `test_conditioning_map_audit.py` | CFG implementation; porosity/conditioning map behaviour at train vs generation time (pins current behaviour). |
| `test_volume_generator_assembly.py`, `test_volume_generator_schedule.py` | Direct-tiling assembly (no blending); the eight-group schedule, the touching-neighbour guards and the seam diagnostic on the generation side. |
| `test_volume_generator_joint.py` | Joint (MultiDiffusion) mode: window/canvas math, weight normalisation, ε-fusion on predictions (not samples), all-UNKNOWN neighbours, mode dispatch. |
| `test_blended_reconstruction.py` | Tukey-window blended reconstruction. |
| `test_porosity_field.py` | Coherent porosity field: mean-to-target, clamp range, per-seed determinism, x-vs-z anisotropy, marginal spread, and the `"coherent"` branch of `_build_local_por_map` in `generate_volumes.py`. Uses the real T-E/T-D artefacts. |

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
| `docs/eval_methodology.md` | How each metric in `eval_generated_volumes.py` is computed — the paper methods reference. |

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
├── split_v2/                     🕰 previous dataset — latents and conditioning still live here
│   ├── volumes.zarr/  patch_index.parquet  splits.json  volume_stats.json
│   ├── orientation_field.json    ← per-volume θ(z) from the NOMINAL layup, with
│   │                               confidence flags + provenance (build_conditioning.py)
│   └── latents_r07z4/            ← LIVE latent store (memmap format)
│       ├── metadata.json         latent shape, VAE provenance, per-channel train stats,
│       │                         voxel_size_um, `conditioning` + `assembly` blocks
│       └── {train,val,test}/latents.bin (float16, mu_then_std) + index.parquet
│                                 + cond.parquet (cond_depth / cond_dist / cond_por_raw,
│                                   row-aligned with index.parquet)
│                                 + material.bin (uint8 16³ material fractions) + air.bin
│                                   (float32 air fraction) — ldm06, row-aligned sibling files
│   (volumes.zarr also gains a compressed `sample_mask` array per volume — ldm06)
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
│   ├── build_latent_dataset_full.log
│   └── material_migration.log
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
    └── 07-vae-metric-recompute/  vae_metric_recompute/ — results.json only; report not run
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

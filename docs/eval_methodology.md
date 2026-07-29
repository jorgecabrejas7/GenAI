# PoreGen Evaluation Methodology

This document describes exactly how every metric in `scripts/eval_generated_volumes.py`
is computed.  It serves as the authoritative reference for the paper methods section
and for comparing results against Naiff 2025, He 2024, and Pinaya 2022.

All metrics operate on **3D binary pore masks** (and, where noted, the corresponding
XCT greyscale volumes).  Masks are binarised from Sauvola-segmented TIFFs at load
time: uint8 → `> 0` (if max ≤ 1) or `>= 128` (if max = 255); float32 → `>= 0.5`.
XCT volumes are normalised to float32 in [0, 1] (uint8 → / 255.0).

**EDA ground truth** (held-out test set, used for sanity checks):

| Quantity | Value |
|:---------|------:|
| Mean porosity φ̄ | 0.055 |
| Std porosity | 0.027 |
| Median equivalent pore diameter | 1.79 voxels |
| P90 equivalent pore diameter | 2.92 voxels |
| S₂(r=30) / φ² | 4.73 |

---

## 1  Porosity fraction

**What it measures**: mean pore fraction φ = |pore voxels| / |total voxels| per volume.
Captures the primary scalar descriptor of pore structure.

**Computation** (`aggregate_comparison`):

```
φ_i = mask_i.mean()      # per volume; mask is {0,1} bool array
```

Aggregated scalars reported:

| Key | Formula |
|:----|:--------|
| `porosity_mean_gen` / `_real` | mean(φ) over all volumes in each set |
| `porosity_std_gen` / `_real` | std(φ) |
| `porosity_w1` | Wasserstein-1 between the two φ distributions (scipy `wasserstein_distance`) |
| `porosity_mae_paired` | mean\|φ_gen_i − φ_real_i\| per matched pair (only when sets are same size) |
| `porosity_dist_mae` | \|mean(φ_gen) − mean(φ_real)\| |

**Primary success metric**: `val/porosity_mae < 0.005`.

**Sanity check**: `porosity_mean_gen` must be within 3× of EDA reference 0.055.

---

## 2  Pore Size Distribution (PSD)

**What it measures**: the distribution of individual pore sizes via equivalent spherical
diameter.  Captures pore-scale morphology in a single scalar distribution.

**Computation** (`_pore_diameters`):

1. Label connected components of the binary mask with `scipy.ndimage.label`.
2. Compute voxel volume V of each component.
3. Convert to equivalent spherical diameter:

```
d_i = (6 · V_i / π)^(1/3)
```

All diameters from all volumes in a set are pooled into one flat array before the
comparison metrics are computed.

**Aggregate scalars** (`aggregate_comparison`):

| Key | Formula |
|:----|:--------|
| `psd_w1` | Wasserstein-1 between pooled generated and real diameter arrays |
| `psd_median_gen` / `_real` | median(d) |
| `psd_p90_gen` / `_real` | 90th percentile of d |
| `psd_mean_gen` / `_real` | mean(d) |
| `psd_n_gen` / `_n_real` | total pore count |

**Sanity checks**: `psd_median_gen` within 3× of 1.79 voxels; `psd_p90_gen` within 3×
of 2.92 voxels.

**Figure**: `figures/psd_histogram.png` — density histograms on linear and log scale,
one curve per directory, bins spanning [0, P99.5] capped at 20 voxels.

---

## 3  Two-Point Correlation S₂(r)

**What it measures**: the probability that two randomly chosen points separated by
distance r are both in pore space: S₂(r) = P(x ∈ pore, x+r ∈ pore).  Encodes
pore size, spatial arrangement, and connectivity simultaneously.  S₂(0) = φ,
S₂(∞) = φ².

**Computation** (`_s2_radial_crop`):

Because the assembled volumes are large (~128 × 3200 × 1280 voxels), the FFT is
computed on random 3D crops rather than the full volume to stay within memory:

1. Draw `n_s2_crops` (default 3) random cube crops of size `s2_crop_size`³
   (default 128, clamped to volume dims).
2. For each crop, apply a 3D Hann window before the FFT to suppress spectral leakage
   from finite crop boundaries, then normalise by the window's own autocorrelation to
   recover an unbiased estimate of S₂:
   ```
   hann_1d      = np.hanning(crop_size)
   hann_3d      = hann_1d[:, None, None] * hann_1d[None, :, None] * hann_1d[None, None, :]
   win_autocorr = real(ifftn(fftn(hann_3d) · conj(fftn(hann_3d))))   # computed once

   windowed = crop * hann_3d
   S₂_raw   = real(ifftn(fftn(windowed) · conj(fftn(windowed))))
   S₂_3D    = where(win_autocorr > 1e-10, S₂_raw / win_autocorr, 0.0)
   ```
   **Why dividing by crop.size is wrong after windowing**: `S₂_raw[r] ≈ S₂(r) · win_autocorr[r]`
   because the Hann window down-weights contributions at large lags.  Dividing by
   `win_autocorr` debiases this, recovering `S₂(r)`.  Dividing by `crop.size` instead
   leaves a lag-dependent suppression factor and causes `S₂(r=30) ~ 250× below φ²`.

   **Zero-lag sanity check**: `S₂_3D[0,0,0]` must equal the crop's porosity `φ`
   (within ~10% relative + 0.005 absolute; logged as a warning if violated).
3. **Safe-radius rule**: bins are capped at `r_max = min(r_max_arg, int(crop_size × 0.4))`.
   For the default 128³ crop this gives r_max = 51 voxels; beyond this radius, edge
   effects from the finite crop dominate the estimate even after windowing.
   The default `--r-max` is 50 (within the safe limit).
4. Bin `S₂_3D` by radial distance into `n_bins` (default 50, spanning [0, safe_r_max]):
   ```
   S₂(r_i) = mean of S₂_3D values where r_edges[i] ≤ ‖Δx‖ < r_edges[i+1]
   ```
5. Average the binned curves over all crops.

**Aggregate scalars** (`aggregate_comparison`):

| Key | Formula |
|:----|:--------|
| `s2_rmse` | RMSE between mean S₂ curves: √mean((S₂_gen(r) − S₂_real(r))²) over all bins. Matches the Gayon-Lombardo 2020 / Naiff 2025 convention of comparing raw curves |
| `s2_w1_per_vol_mean` / `_std` | per-volume W1 against pooled-real mean curve; std measures diversity (valid as a diversity signal, not a fidelity signal) |
| `s2_r30_check_ratio` | S₂_gen(r=30) / (EDA_S2_R30_COEFF × φ_mean²); expected ≈ 1.0; r=30 is within the safe range for 128³ crops |

**Sanity check**: `s2_r30_check_ratio` must be within [1/3, 3].

**Figure**: `figures/s2_curves.png` — mean S₂(r) curves per directory vs r (voxels).

---

## 4  Ripley's K(r)

**What it measures**: spatial clustering of pore centres relative to complete spatial
randomness (CSR).  K(r) = expected number of additional pore centres within distance r
of an arbitrary centre, scaled by volume / density.  For CSR, K(r) = (4/3)πr³.
Values above CSR indicate clustering; below indicate regularity.

**Computation** (`_ripleys_k` / `_ripleys_k_per_patch`):

1. Label connected components; compute centre-of-mass (`regionprops`).
2. If fewer than 3 components → skip (return None).
3. If more than `max_pores_ripley` (default **5000**) → random subsample to control O(N²)
   cost; a warning is logged when subsampling occurs.
4. Compute all pairwise Euclidean distances between centres using `scipy.spatial.distance.pdist`.
5. For each integer r from 1 to `r_max // 2` (default r_max = 50, so up to r = 25):
   ```
   K(r) = (V / N²) · 2 · #{(i,j) : i<j, d_ij < r}
   ```
   The factor of 2 converts unordered pairs (i<j, from `pdist`) to all ordered pairs (i≠j),
   which is the standard K(r) definition.  V is total volume (voxels³), N is pore count.
   For CSR: K(r) = (4/3)πr³.

**Per-patch fallback** (`_ripleys_k_per_patch`):

When `--ripley-per-patch` is set, **or** when the estimated pore count exceeds 50 000
(estimated as φ_mean × volume_voxels / median_pore_volume, with median_pore_volume = (π/6) × 1.79³ ≈ 3.0 voxels³),
Ripley's K is computed on individual non-overlapping 64³ patches extracted from the volume
and the per-patch curves are averaged.  This avoids O(N²) scaling on millions of pore
centres and is consistent with the spatial scale the model operates at.

**Aggregate scalars** (`aggregate_comparison`):

| Key | Formula |
|:----|:--------|
| `ripley_w1` | Wasserstein-1 between mean K(r) curves |
| `ripley_K_at_mean_spacing_gen` / `_real` | K(r) evaluated at r = mean pore diameter (interpolated) |

**Figure**: `figures/ripley_k.png` — mean K(r) curves per directory vs r, with CSR
reference (4/3)πr³ plotted as a dashed black line.

---

## 5  FID on 2D slices

### Why full-slice resize was rejected

PoreGen volumes are ~3 179 × 1 759 voxels.  The median equivalent pore diameter is
**1.79 voxels**.  Resizing a full slice to InceptionV3's 299 × 299 input requires a
~10.6× downscale; pores shrink to ~0.17 pixels and vanish before the network sees
them.  Full-slice FID measures macro-scale rock texture, not pore morphology.

### Protocol (`compute_fid_crop_based`)

1. **Crop extraction**: for each 3D volume, extract `crops_per_volume` (default 500)
   random 2D crops — approximately `crops_per_volume // 3` per axis (axial z, coronal y,
   sagittal x).  Each crop is **64 × 64 pixels** at native voxel resolution, drawn from
   a uniformly random slice with replacement.  If a slice dimension < 64, it is padded
   with reflection before cropping.

2. **Minimum sample guard**: total crops per set must be ≥ 5 000 before any computation;
   a `ValueError` is raised otherwise.

3. **Feature extraction**: crops are resized from 64 × 64 → 299 × 299 with **bilinear
   interpolation** (`torch.nn.functional.interpolate`, `align_corners=False`), replicated
   to 3 channels, and passed through **InceptionV3** (pretrained on ImageNet, eval mode,
   no gradient).  Features are the 2 048-dimensional `avgpool` (pool3) output.  Batch
   size: 32.

4. **FID formula**: let R, G be the (N, 2048) feature matrices for real and generated:
   ```
   FID = ‖μ_R − μ_G‖²  +  Tr(Σ_R + Σ_G − 2·(Σ_R·Σ_G + ε·I)^½)
   ```
   with ε = 1×10⁻⁶ added to the product matrix before `scipy.linalg.sqrtm` to prevent
   numerical instability.  Complex residuals from `sqrtm` are discarded (real part taken).

5. **Output**: FID is computed separately for axial, coronal, and sagittal planes; the
   mean of the three values is also reported.

### Comparison with prior work

| Work | Volume size | Protocol | Pore size after resize | Comparable to PoreGen? |
|:-----|:------------|:---------|:----------------------|:-----------------------|
| Naiff 2025 | 256³, ~6 µm/vox | full 256×256 slice → 299×299 (1.17× up) | ~21 px | No — different pore scale |
| He 2024 | ~128³ | full slice → 299×299 | tens of px | No |
| Pinaya 2022 | 3D brain MRI | full axial slice → 299×299 | N/A (not porous) | No |
| **PoreGen** | 3 179×1 759, ~25 µm/vox | 64×64 crop → 299×299 (4.7× up) | ~8 px | — |

**Internal comparisons** within a single PoreGen run (generated vs. baseline vs. real,
all evaluated with the same script) are valid and meaningful.  Cross-paper FID numbers
are not directly comparable.

---

## 6  Boundary inconsistency

**What it measures**: artefacts at patch assembly seams.  Because the LDM generates
non-overlapping 64³ patches that are stitched together, voxel-pair differences across
seam boundaries should be no larger than across interior positions.  A ratio > 1
indicates visible stitching artefacts.

**Computation** (`_boundary_inconsistency`):

Seam positions are multiples of `stride` (default 32) along each axis.  Interior
positions are all other positions, subsampled to ≤ 200 per axis to bound runtime
on large volumes.

For each of the two modalities (XCT, mask) and each axis:
```
seam_MAE   = mean |vol[i-1] - vol[i]|  for all seam i
interior_MAE = mean |vol[i-1] - vol[i]|  for all interior i
seam_ratio = seam_MAE / interior_MAE
```

Values are averaged across all three axes, then averaged across all volumes in the set.

**Reported keys**: `boundary_xct_seam_ratio_mean_gen` / `_real`,
`boundary_mask_seam_ratio_mean_gen` / `_real`.
Ratio ≈ 1 means no seam artefact.  Values substantially > 1 indicate stitching artefacts.

---

## 7  Pore morphology

**What it measures**: per-pore shape statistics — sphericity (how round each pore is)
and aspect ratio (elongation).

**Computation** (`_morphology_stats`):

Up to `max_pores_morph` (default 2 000) largest pores (by voxel count) are analysed.

**Sphericity** (Wadell definition):
```
Ψ = π^(1/3) · (6·V)^(2/3) / A
```
where V is voxel count and A is surface area from marching cubes
(`skimage.measure.marching_cubes`, level = 0.5).  Pores with V < 4 voxels are skipped
(too small for reliable surface reconstruction).  Ψ = 1 for a perfect sphere; < 1 for
elongated or rough pores.

**Aspect ratio**: ratio of longest to shortest 3D bounding-box dimension
(bbox extent along each spatial axis; skipped if shortest dimension = 0).

**Reported statistics**: mean, std, P5, P50, P95 for both quantities, separately for
generated and real sets.

**Keys**: `morph_sphericity_{mean,std,p5,p50,p95}_{gen,real}`,
`morph_aspect_ratio_{mean,std,p5,p50,p95}_{gen,real}`.

---

## 8  Diversity

**What it measures**: whether the generative model covers the variability in the real
distribution, not just the mean.  Low diversity indicates mode collapse.

**Computation** (`compute_diversity`):

Three diversity signals, each compared between generated and real:

| Key | Formula |
|:----|:--------|
| `diversity_phi_std_{gen,real}` | std(φ) across volumes |
| `diversity_psd_w1_mean_{gen,real}` | mean over volumes of W1(per-vol PSD, pooled-real PSD) |
| `diversity_psd_w1_std_{gen,real}` | std of the per-volume W1 values above |
| `diversity_ripley_K_std_{gen,real}` | std of mean(K(r)) over volumes |

Generated std values substantially below real values indicate mode collapse.

---

## 9  Memorisation check

**What it measures**: whether the LDM has memorised training patches.  Low nearest-
neighbour distance in latent space from generated patches to training patches indicates
overfitting; high values indicate genuine generalisation.

**Computation** (`compute_memorization`):

Requires `--vae-run` (directory with `resolved_config.yaml` and
`checkpoints/best.ckpt`) and `--latents-dir` (precomputed training latents).

1. Load the VAE encoder from checkpoint.
2. Slide a non-overlapping 64³ window over each generated XCT volume; encode each
   patch through the encoder to obtain μ (the mean of the posterior), shape (z_ch, 16, 16, 16).
3. Flatten each μ to a 1D vector of length z_ch × 16³.
4. Sample `n_train_sample` (default 10 000) training latents from the memmap file
   (μ channels only, first z_ch channels of the stored (mu ‖ logvar) representation).
5. For each generated latent, find the L2 nearest neighbour in the training set
   (chunked in groups of 256 to avoid OOM).

**Reported keys**:
- `memorization_nn_dist_mean`: mean min-L2 distance (higher = less memorisation)
- `memorization_nn_dist_std`: std of per-patch distances
- `memorization_n_gen_patches`: number of generated patches encoded
- `memorization_n_train_latents`: number of training latents sampled

Skipped automatically if `--vae-run` is not provided or checkpoint/latents are missing.

---

## Sanity checks

After computing all aggregate metrics, `sanity_check` flags any value that deviates by
more than **3×** from the EDA ground truth reference:

| Check | EDA reference | Flagged if |
|:------|:-------------|:-----------|
| `porosity_mean_gen` | 0.055 | outside (0.018, 0.165) |
| `psd_median_gen` | 1.79 vox | outside (0.60, 5.37) |
| `psd_p90_gen` | 2.92 vox | outside (0.97, 8.76) |
| `s2_r30_check_ratio` | 1.0 | outside (0.33, 3.0) |

Warnings appear in the log and in the `sanity_warnings` key of `eval_results.json`.

---

## Output files

| File | Contents |
|:-----|:---------|
| `eval_results.json` | All scalars, per-volume metrics, per-volume curves (S₂, K), FID per axis, diversity, memorisation |
| `eval_report.md` | Markdown report: FID table, summary metrics table, sanity warnings, figure links |
| `figures/s2_curves.png` | Mean S₂(r) curves, real vs generated (vs baseline) |
| `figures/psd_histogram.png` | PSD density histogram, linear and log scale |
| `figures/ripley_k.png` | Mean K(r) curves with CSR reference |
| `figures/fid_table.png` | FID table image (per axis + mean, per directory) |

---

## CLI reference

```
python scripts/eval_generated_volumes.py \
    --real-dir      path/to/real/volumes/ \
    --generated-dir inference/ldm03-run-0001-.../ \
    [--baseline-dir inference/baseline/] \
    [--out-dir      eval_results/stage4/] \
    [--vae-run      runs/vae/r05-run-0001-.../] \
    [--latents-dir  data/split_v2/latents_s64_sampled/] \
    [--stride 32] [--r-max 50] [--s2-crop 128] [--n-s2-crops 3] \
    [--crops-per-volume 500] [--device cuda] \
    [--skip-fid] [--skip-ripley] [--skip-morphology] \
    [--ripley-per-patch]

# FID-only (fast path):
python scripts/eval_generated_volumes.py --fid \
    --real-dir ... --generated-dir ... [--baseline-dir ...]
```

**Key defaults**:

| Flag | Default | Controls |
|:-----|:--------|:---------|
| `--r-max` | 50 | Max radius (voxels) for S₂ and Ripley K; safe limit for S₂ is `s2_crop × 0.4` (= 51 for default 128³ crop) |
| `--s2-crop` | 128 | Crop size³ for S₂ FFT |
| `--n-s2-crops` | 3 | Crops averaged per volume for S₂ |
| `--stride` | 32 | Patch stride for boundary inconsistency |
| `--crops-per-volume` | 500 | FID crops per volume (all 3 axes combined) |
| `--max-pores-ripley` | 5000 | Pore subsampling cap for O(N²) Ripley K; warning logged when hit |
| `--max-pores-morph` | 2000 | Largest-N pores for morphology |
| `--ripley-per-patch` | off | Force per-patch Ripley K; auto-enabled when est. pore count > 50 000 |

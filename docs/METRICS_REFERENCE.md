# Metrics reference

Every metric this project computes, taken from the code on 2026-11-12. Each entry
gives: where it is computed (file:function), the formula or algorithm, inputs and
units, how to read the value, and the caveats the code itself states.

Conventions used in this document:

- `label` is a uint8 volume with 0 = material, 1 = pore, 2 = air.
- `xct` is the grey volume. On disk it is uint8 (0-255). In the metrics it is
  usually `xct / 255` in [0, 1].
- `material` is the requested specimen envelope (true where the coupon is, pores
  included). For a real crop it is the crop's own `sample_mask`.
- "material porosity" is `pore voxels / material voxels`. Air is in neither the
  numerator nor the denominator.
- A **floor** is the value real material scores on the same metric. A **ratio**
  is `measured / floor`. A **gate** is a fixed pass/fail threshold.
- "mean +/- sd" over seeds is `metrics.mean_sd`: sample sd (ddof = 1).

---

## 1. Autoencoder (VAE) training and evaluation metrics

The trainer is `src/poregen/training/engine.py`. The periodic rows (`val/`,
`test/`) and the full-pass rows (`val_full/`, `test_full/`) all come from one
function, `engine._run_eval`. It runs `eval_step` on N batches, accumulates, and
averages. The `val_full` / `test_full` rows are the same numbers over the whole
loader (`train_loop`, "Final val" / "Final test", and the periodic
`full_val_every` pass). Every key in the loss dict is averaged over batches, so
loss terms and metrics share one table.

### 1.1 Loss terms (also reported as eval rows)

| metric | where | formula | reading |
|---|---|---|---|
| `total` | `losses/total.py:compute_total_loss` | `xct_weight * xct_loss + p_weight * lpips + mask_total + class_total + beta * kl_term` | the only scalar the optimiser minimises |
| `xct_loss` | `losses/total.py:compute_total_loss`, via `losses/recon.py:get_recon_loss` | one of `l1` (mean abs), `mse`, `charbonnier` (`mean sqrt((x_hat - x)^2 + 1e-12)`), chosen by `loss.xct_loss_type`. Target is `xct / 255` in [0, 1]; the decoder output is raw, with no activation. | per-voxel grey error in [0, 1] units |
| `lpips` | `losses/perceptual.py:SlicePerceptualLoss.forward` | LPIPS (package `lpips` 0.1.4, `net='vgg'`, version 0.1, frozen) on paired 2-D slices. Per patch: `n_slices_per_axis` (default 4) random slice positions along each of the three axes, the same positions for reconstruction and input. Grey in [0, 1] goes in as `2x - 1` expanded to 3 channels. The row is the mean over all slices of all patches. Only present when `loss.perceptual_weight > 0`. | lower = closer in VGG feature space; the slice positions are random every call |
| `kl` | `losses/kl.py:kl_divergence` (spatial latent) or `kl_divergence_flat` (flat VRRAE latent) | per element `0.5 (mu^2 + exp(logvar) - logvar - 1)`, logvar clamped to [-30, 20]; mean over batch and space gives `KL_c` per channel; `kl = sum_c max(KL_c, free_bits)` | nats per channel, summed; free bits clamp the term that enters `total` |
| `kl_raw` | `engine._run_eval` | `sum_c KL_c` without the free-bits clamp | the unclamped KL |
| `kl_per_channel` | `losses/kl.py` | the vector `KL_c` (raw, pre-clamp) | logged as a TensorBoard histogram; dropped from the JSON rows |
| `kl_collapsed_fraction` | `losses/kl.py` | `mean_c 1[KL_c < free_bits]` | share of channels the clamp is holding up; 0 when free bits are off |
| `kl_rombach` | `losses/kl.py:kl_rombach_sum` | `mean_b sum_elements KL / n_voxels_per_patch` | only under `loss.kl_reduction: rombach_sum`; the LDM-paper KL rescaled to a per-voxel loss |
| `beta` | `losses/kl.py:beta_schedule` | `min(max_beta, max_beta * step / warmup)` | the KL weight at this step |
| `mask_bce`, `mask_dice` / `mask_tversky` | `losses/mask.py:combined_mask_loss` | binary-head terms (pre-r08 variants) | not used by the 3-class head |
| `class_ce`, `class_dice`, `class_dice_<i>`, `class_total` | `losses/mask.py:combined_class_loss` | class-weighted cross-entropy plus soft Dice over the minority classes (pore, air by default); `class_total = ce_weight * ce + dice_weight * dice` | the 3-class head's loss (r08 and later) |

Caveat stated in the code: `losses/recon.py` still says the losses operate in
"z-score space". The engine and `models/vae/base.py:decode_xct` say the XCT head
regresses `xct / 255` directly. The code path is the same (a plain L1 / MSE /
Charbonnier on raw output); only the docstring is stale.

### 1.2 Reconstruction quality

**`mae`** - `engine._run_eval`. `F.l1_loss(decode_xct(xct_out), xct)` where
`decode_xct` is a clamp to [0, 1] and no sigmoid. Units: grey in [0, 1]
(multiply by 255 for u8 levels). Note: `metrics/recon.py:mae` is a different
function. It applies a sigmoid first. The trainer does not call it. Until
2026-09-01 the trainer did apply a sigmoid, which put a floor of about 0.135
under `val/mae`; rows before that date are not comparable.

**`sharpness_recon_over_gt`** - `engine._run_eval` using
`metrics/recon.py:sharpness_proxy`. `S(x) = (mean|dx/dz| + mean|dx/dy| + mean|dx/dx|) / 3`
by finite differences on the [0, 1] volume. Row value = `mean_batches S(recon) / mean_batches S(gt)`.
Reading: 1 = same mean gradient as the scan; < 1 = blur; > 1 = extra
high-frequency energy. Caveat: this counts the amount of gradient, not where it
is. Added noise of the right strength scores 1 (see `vae_texture_fidelity.py`
header, section 2).

**`mse`, `psnr`** - `metrics/recon.py`. Defined (`psnr = 10 log10(1 / mse)`,
sigmoid applied to the prediction) but not called by the trainer. They belong to
the eval-v3 offline path (`eval/metrics.py`).

### 1.3 Segmentation: 3-class head (r08 and later)

All computed by `metrics/seg.py:multiclass_metrics` on the **argmax** of the
class logits, never on the soft probabilities. The code states the reason: the
sampler writes the argmax, so these numbers match what a generated volume
delivers.

**`dice_material`, `dice_pore`, `dice_air`** - per patch and per class c:
`Dice_c = 2 |P_c & T_c| / (|P_c| + |T_c|)`; a patch with neither prediction nor
truth for c scores 1. The row is the mean over patches, then over batches.

**`porosity_mae`, `porosity_bias`, `porosity_pred_mean`** - per patch
`p = mean(pred == pore)`, `t = mean(label == pore)` over all 64^3 voxels (air
included in the denominator; this is the full-patch phi, not material
porosity). `porosity_mae = mean |p - t|`, `porosity_bias = mean (p - t)`,
`porosity_pred_mean = mean p`. **`air_mae`, `air_bias`, `air_pred_mean`** are
the same for the air class. Gate used throughout the project: `porosity_mae < 0.005`
(the docstring of `porosity_metrics`).

**`porosity_mae_bin_0..3`, `porosity_n_bin_0..3`** - `metrics/seg.py:porosity_binned_mae`,
over every patch of the whole eval pass (not per batch). Bins by **true**
porosity: [0, 0.01), [0.01, 0.03), [0.03, 0.06), [0.06, inf). A bin with no
patch gives NaN and count 0. The code says: read a bin's MAE only when its count
is large; the high bins are the ones a 20-batch eval can leave almost empty.

**`test/porosity_mae_per_volume`, `_vol_p50`, `_vol_p90`, `_vol_max`** -
`engine.train_loop`. Per source volume: the signed patch errors are averaged,
then the absolute value is taken. Histogram and the 50th / 90th percentile and
max over volumes. TensorBoard only.

### 1.4 Segmentation: binary mask head (pre-r08 variants)

`metrics/seg.py:segmentation_metrics` gives `dice_pos_only`, `precision_pos_only`,
`recall_pos_only` at threshold 0.5 on the sigmoid, over patches whose truth is
not empty (empty patches would inflate the overlap). Dice is averaged per patch;
precision and recall are pooled TP / FP / FN. `metrics/seg.py:porosity_metrics`
gives `porosity_mae`, `porosity_bias`, `mask_pred_mean` from the **soft** sigmoid
mean per patch. These rows only exist when the model has `mask_logits`.

### 1.5 Latent diagnostics

**`mu_active_fraction`, `mu_n_active`** - `metrics/latent.py:active_units_from_moments`,
fed by `latent_channel_moments` and `merge_latent_channel_moments` over every
eval batch. Per channel c, the sample variance of `mu_c` over batch and all
spatial positions, from pooled sums and sums of squares. A channel is active
when `Var(mu_c) > 0.01`. The fraction is `n_active / C`. In training the same
is logged from the moments of the current batch window.

**`mu_mean`, `mu_std`, `logvar_mean`, `logvar_std`, `std_mean`** -
`metrics/latent.py:latent_stats`: plain mean and std of `mu`, of `logvar`, and
the mean of `exp(0.5 logvar)`.

**`kl_per_channel`** as a diagnostic - `metrics/latent.py:kl_per_channel` is the
same formula as the loss version but without the logvar clamp. The trainer logs
the loss version.

---

## 2. Texture suite (campaign 33) and band fidelity (campaign 32)

Scripts: `scripts/analysis/vae_texture_suite.py` and
`scripts/analysis/vae_texture_fidelity.py`. CPU. Patches: the held-out val
patches of `vae_val_l1.load_model_and_patches` (20 batches of 32 on a fixed
stride), **interior only** (`vae_texture_fidelity.interior_patches` drops any
patch with an air voxel). The posterior is sampled with `torch.manual_seed(0)`.
These numbers track texture across checkpoints. The header states they do not
stop training and do not pick checkpoints.

**`grey_l1`** - `vae_texture_suite.suite`: mean over patches of
`mean |recon - input|`, grey in [0, 1].

### A. Band power ratio, coherence, relative error

`vae_texture_fidelity.shell_sums` and `summarise`. Each 64^3 patch pair
(reconstruction r, input t) is mean-subtracted, multiplied by a separable 3-D
Hann window, and transformed with `rfftn`. Every FFT bin is assigned to a radial
shell `k = round(|k_vec|)` in cycles per 64 voxels, k = 1..32; half-space bins
are weighted 2 to stand for their conjugates. Per shell the sums
`rt = Re sum F_r conj(F_t)`, `rr = sum |F_r|^2`, `tt = sum |F_t|^2`,
`err = sum |F_r - F_t|^2` are pooled over all patches. Bands by wavelength in
voxels: coarse 16-64 (k 1-3), mid 8-16 (k 4-7), fine 4-8 (k 8-15),
finest 2-4 (k 16-32). Per band:

- `power_ratio = rr / tt`: < 1 = detail lost (blur), > 1 = detail added
  (noise, invented texture). This is the quantity the suite's flat output
  reports (`A_<band>_P`).
- `coherence = rt / sqrt(rr tt)` (Fourier shell correlation): is the detail in
  the right place. Blind to a pure linear blur.
- `rel_error = err / tt`: one number that grows with blur, noise and
  displaced texture alike.
- `resolution_voxels = 64 / k*` where k* is the first shell with FSC < 0.5.

There is no floor: the comparison is each reconstruction against its own input.
The input's own shell power (`input_power`) is reported so the noise tail can be
seen. The header states the finest band is largely scan noise and is not
expected to reach 1.

### B. Sliced Wasserstein distance on a Laplacian pyramid

`vae_texture_suite.laplacian_pyramid`, `swd_descriptors`, `swd`. A 3-D
Laplacian pyramid with three levels (64, 32, 16; Gaussian sigma 1, decimate by
2, upsample linearly). From every level of every patch, 64 random 7^3
neighbourhoods are taken (343-d descriptors). Descriptors are standardised per
level by the **real** set's mean and sd. Then 4 repeats of 128 random unit
directions: project both sets, sort, and take `mean |sorted_a - sorted_b|`
(the sliced W1). Reported x 1e3, per level `lap64`, `lap32`, `lap16`. This is
ProGAN's SWD (Karras et al. 2018) in 3-D.

### C. Slice FID and KID

`vae_texture_suite.central_slices`, `kid`; `eval_v4.microstructure.inception_features`
and `frechet_distance`. The central slice of each patch in the three planes
(3 slices per patch) goes through the eval_v4 Inception pipeline (section 4.9).
FID is the Frechet distance on the pooled 2048-d features. KID is the unbiased
MMD^2 with the cubic polynomial kernel `k(x, y) = (x.y / d + 1)^3`, averaged
over 50 random subsets of 100 features (Binkowski et al. 2018), x 1e3.

### The half-split floor for B and C

`vae_texture_suite.suite`. The interior patches are split by index parity into
halves H1 and H2.

```
value = ( d(recon H1, real H2) + d(recon H2, real H1) ) / 2
floor = d(real H1, real H2)
```

Same set sizes on both sides; a patch is never compared with itself. Read
`value ~ floor` as "as close to the scan as another sample of the scan is".
For KID the code says to read the absolute difference, not a ratio, because the
KID floor sits near 0.

### D. Pore shape and size

`vae_texture_suite.pore_stats`. Pores = `argmax(class_logits) == pore` vs the
true label, on patches that hold true pores. `s2_shape_w1_vox` =
`eval_v4.microstructure.curve_w1` between the mean S2(r) of the reconstructed
pores and the mean S2(r) of the true pores (voxels; section 4.9). `pore_diam_w1_vox`
= `psd_w1` between pooled equivalent diameters (6-connected, voxels). No floor;
identity gives 0 by construction (`--validate` checks this).

### Validation

`vae_texture_suite.validate` replaces the reconstruction by known transforms of
the input (blur sigma 0.5 / 1 / 2, noise 0.1 / 0.3 sd; pore masks eroded,
dilated, speckled) and checks: identity sits at the floor, blur and noise
worsen B and C monotonically, blur lowers the fine power ratio below 1, noise
raises the finest band above 1, and D is 0 for identity and > 0 for every
mask change. `vae_texture_fidelity --validate` does the same for A with blur,
noise and phase scramble.

---

## 3. Pore morphology (campaign 36)

Module: `src/poregen/metrics/pore_morphology.py:measure(label, voxel_um)`.
Report: `scripts/analysis/pore_morphology_report.py`. A pore is a
**26-connected** component of `label == 1`, with no minimum size (the
segmentation as is). Coordinates are (x, y, z) = (axis 2, 1, 0) in micrometres;
z is the through-thickness axis. All sums are one `np.bincount` pass over pore
voxels. The table has 58 columns, confirmed by running `measure` on a toy
label. Volume summary beside it: `n_pores`, `pore_voxels`, `air_voxels`,
`nonair_voxels`, `porosity = pore / non-air`.

| column(s) | formula (v = voxel size in um, N = voxels in the pore, primed = centred coordinates) |
|---|---|
| `pore_id`, `n_voxels` | component index; N |
| `volume_um3` | `V = N v^3` |
| `r_eq_um` | `(3V / 4 pi)^(1/3)` |
| `area_um2` | Crofton-style, the external repo's rule: `n` = count of 0/1 face transitions of the pore's mask summed over the three axes with a one-voxel zero pad (the array edge counts), `A = (4/pi) n v^2` (`_transitions`) |
| `sphericity` | `min(1, pi^(1/3) (6V)^(2/3) / A)` |
| `area_corr_um2`, `sphericity_corr` | the same with factor `2/3` instead of `4/pi`. The docstring: 4/pi is the 2-D perimeter rule inverted; in 3-D an isotropic surface crosses (3/2) A voxel faces, so `A ~ (2/3) n`. A digital sphere of radius 20 scores sphericity 0.52 and A 1.9x too large with 4/pi, ~0.99 with 2/3 (`tests/test_pore_morphology.py`). Both are kept: the repo's for comparability, the corrected for absolute values. |
| `bary_{x,y,z}_um` | barycentre |
| `inertia_Ixx/Iyy/Izz/Ixy/Ixz/Iyz_um2` | per-voxel second moments: `Ixx = sum(y'^2 + z'^2)/N`, `Iyy = sum(x'^2 + z'^2)/N`, `Izz = sum(x'^2 + y'^2)/N`, `Ixy = -sum x'y'/N`, `Ixz = -sum x'z'/N`, `Iyz = -sum y'z'/N` |
| `inertia_eig{1,2,3}_um2`, `inertia_vec{k}_{x,y,z}` | eigen-decomposition (`_eig_desc`), eigenvalues **descending**. The largest inertia eigenvalue belongs to the **short** axis, so `inertia_vec1` is the minor axis. |
| `cov_eig{1,2,3}_um2`, `cov_vec{k}_{x,y,z}` | covariance with `np.cov` convention (N - 1), eigenvalues descending; `cov_vec1` is the **long** axis. NaN when N = 1. |
| `{cov,inertia}_anisotropy` | `1 - s3/s1` (s1 >= s2 >= s3 are the eigenvalues of that source) |
| `{cov,inertia}_elongation` | `s2/s1` |
| `{cov,inertia}_flatness` | `s3/s2` |
| `{cov,inertia}_elongation_avg` | `2 s1 / (s2 + s3)` |
| `{cov,inertia}_flatness_avg` | `s2 / s3` |
| `{cov,inertia}_phi_deg` | `arccos(|v1_z|)` in degrees, 0-90, from the z axis |
| `{cov,inertia}_theta_deg` | `atan2(v1_y, v1_x)` in degrees, -180..180; sign-ambiguous by 180 |
| `dist_surface_um` | shortest Euclidean distance from any voxel of the pore to the nearest air voxel: one EDT of the non-air region (`_dist_to_air`, computed in y-blocks of 512 with a 160-voxel margin, exact when the nearest air is within the margin, over-reads beyond it; NaN when the block window holds no air) |
| `touches_air` | a face neighbour (6-connected dilation) of the pore is air |
| `touches_edge` | the pore's bounding box reaches the array boundary (truncated by the crop) |

Caveats in the docstring: for the covariance source, elongation and flatness are
**squared** axis ratios. The sphericity clamp at 1 is applied to both versions.

### Report statistics (`pore_morphology_report.py`)

- **Normalised W1 with the real floor.** For each porosity bin (0-0.5 %,
  0.5-1 %, 1-2 %, 2-5 %, 5-12 %) and each property p in `PROPS` (every column
  except identifiers, in-plane barycentre and the two flags; eigenvector
  components compared as `|v|`): `D(v) = W1(sample of v, pooled real coupons of
  the bin) / IQR(pooled real of p)`. The pooled reference is an equal-size
  subsample per coupon (`max(200000 / n_coupons, 2000)` pores each). For a
  real coupon the reference leaves that coupon out (leave-one-coupon-out).
  The real coupons' D values are the floor; a generated volume reads "as real
  as a real coupon" when its D falls inside the real min-max range. Medians
  over volumes are tabled.
- **Twins.** Each twin (a regenerated real coupon) is also scored against its
  own coupon with the same normalised W1.
- **Small-pore error.** `small_pore_error`: pores with fewer than
  `MIN_REAL = 8` voxels (the split_v4 segmentation's minimum size; the smallest
  real pore is 9 voxels). Count, share of pores, share of pore volume. 0 for
  every real coupon by construction; a generated pore under 8 voxels is outside
  the training distribution and is counted as a model error.
- **Population per volume.** porosity, `pores_per_mm3` = pores / (non-air
  voxels x 0.025^3 mm^3), share of each flag, median of every property.

Note: the eval_v4 microstructure module uses **6-connectivity** for pores
(section 4.9). Campaign 36 uses 26. The two pore counts differ by design.

---

## 4. eval_v4

Package `src/poregen/eval_v4/`. Every metric is a function over numpy arrays
decorated with `@requires(...)`, which refuses a volume whose manifest does not
carry the fields the metric reads. Shared constants (`metrics.py`): `EDGE_VOX = 32`
(the "interior" is more than 32 voxels from every face), `TILE = 64`,
`POROSITY_GATE = 0.005`, `FAIL_PHI_LOW = 1e-4`, `FAIL_PHI_HIGH = 0.5`,
`FAIL_AIR_IN_MATERIAL = 0.2`. `measure.py:measure_core` runs on every generated
volume: `phase_fractions`, `failure_flags`, `degenerate_cells`, `seam_metrics`,
and `porosity_error` when a global porosity was requested. Real test crops go
through the same functions (`real_floor.py`) and give the floor rows.

### 4.1 Porosity and air

**`phase_fractions`** (`metrics.py`): `phi_pore = mean(label == pore)` inside
`material` (material porosity); `phi_pore_all` = over the whole volume, air
included (the definition the LDM is conditioned on); `air_fraction` inside
material; `air_fraction_all`; `air_fraction_interior` inside
`interior & material`; `material_voxel_fraction`.

**`porosity_error`** (`metrics.py`, needs `requested_global_phi`):
`delivered = mean(label[material] == pore)`, `error = delivered - requested`,
`abs_error = |error|`, `within_gate = abs_error < 0.005`. Both sides are
material porosity. The gate is decision D39, kept from eval v3.

**`err_stats`** (`metrics.py`): for a set of errors, `abs_error_mean`,
`_median`, `_p95`, `_max`, `frac_within_gate` (fraction with `|e| < 0.005`).

**`failure_flags`** (`metrics.py`): `phi_collapsed = phi < 1e-4` (suppressed
when the request itself is below 1e-4: delivering zero on a zero request is
obedience), `phi_saturated = phi > 0.5`, `air_in_material = air fraction inside
material > 0.2`, `failed = any`. `measure._failure_rate` gives the fraction of
volumes with `failed` and the count per flag.

**Dose response** (`measure.py:measure_porosity_global`): over every in-range
volume (the off-manifold 0.15 request is excluded), ordinary least squares of
`delivered_phi` on `requested_phi` with `metrics.fit_ols`: `slope`, `intercept`,
`r2 = 1 - SS_res / SS_tot`, `n`. Also `frac_within_gate`, and per level the
mean +/- sd of delivered, mean +/- sd of `abs_error`, the within-gate fraction and
the failure rate. Reading: slope 1 and intercept 0 is one-to-one control.

### 4.2 Local (within-volume) obedience

**`local_obedience`** (`metrics.py`): the label is summed per 64-voxel tile
(`block_sum`). `delivered[c] = pore_c / material_c` on tiles that are more than
50 % material (others are dropped). With a requested field `req[c]` (material
porosity per tile): subtract each side's own mean, then
`fit_ols(req - mean(req), delivered - mean(delivered))` gives
`within_volume_slope`, `within_volume_r2`, `within_volume_intercept`;
`per_cell = err_stats(delivered - req)` gives the per-cell absolute error and
`frac_within_gate` ("cells in gate", gate 0.005). For a real volume (no request)
only `delivered_cell_mean / sd / min / max` are returned: the cell-to-cell
spread is the noise floor the slope must beat. **`pooled_dose_fit`** is the fit
across all cells of all volumes; the code names it
`pooled_*_not_obedience` because most of its variance is the global dose
response. Reading: slope 1, R2 high = the pores land where the field asked.

### 4.3 Seams: window planes and chunk planes

**`seam_discontinuity`** (`src/poregen/diffusion/sampler.py`): the primary
assembly metric. On a continuous volume (grey `xct/255`, or the pore log-odds
from `probs.npz`), for each axis take `per_plane[k] = mean |v[k] - v[k-1]|`
over the whole slice, for every plane k = 1..n-1. A plane is a **seam** plane
when `k % period == 0`; it is an **interior** plane when `k % interior_exclude != 0`
(`interior_exclude` defaults to `period`; for the chunk period it is set to 64
so the chunk ratio and the window ratio share one baseline that excludes the
window planes). Per axis:

```
<prefix>_<axis>_mad          = mean of per_plane over seam planes      (the "MAD": mean absolute slice-to-slice difference)
<prefix>_<axis>_interior_mad = mean of per_plane over interior planes
<prefix>_<axis>_ratio        = seam_mad / interior_mad                 (NaN when the axis has no seam plane)
```

The combined values `<prefix>_mad`, `<prefix>_interior_mad`, `<prefix>_ratio`
are **voxel-weighted** across axes: each axis contributes `mad x n_planes x voxels_per_plane`
to the seam sum and to the interior sum separately, and the combined ratio is
`agg_seam / agg_interior`. An axis with **no** seam planes still contributes its
interior planes to `agg_interior`. That is why the combined ratio can read
below the per-axis ratios, and below 1: on a 192 x 1024 x 1024 volume with a
192-voxel chunk period the z axis has no chunk plane, but the z interior
(across plies) has a larger slice-to-slice change than the in-plane interior,
so the baseline is pulled up while the seam sum only holds y and x planes. Read
the per-axis ratios when the axes are not equivalent, and compare against the
real floor measured on the same shape (the real 128 x 1024 x 1024 crops score
about 0.90 on the chunk ratio for this reason).

Reading: ratio about 1 = the plane is indistinguishable from ordinary
slice-to-slice texture change; well above 1 = the two sides disagree where they
meet. Real material scores about 0.96 on the window ratio, so 0.96 is "nothing
left to fix", not "nearly perfect".

**`seam_metrics`** (`metrics.py`, needs `chunk_tiles`, `window_stride`):
`seam_xct_*` (period 64), `seam_chunk_xct_*` (period `64 x chunk_tiles`,
interior excludes the 64 planes), and the same on the pore log-odds
(`seam_pore_*`, `seam_chunk_pore_*`) when `probs.npz` exists
(`pore_logit_available`). The report's four headline ratios are
`seam_xct_ratio`, `seam_pore_ratio`, `seam_chunk_xct_ratio`,
`seam_chunk_pore_ratio` (`measure._seam_summary`, mean +/- sd over seeds). A
192^3 volume is one chunk and has no chunk plane; its chunk columns are NaN by
construction.

**`seam_discontinuity_material`** (`metrics.py`): the same statistic restricted,
plane by plane, to voxels that are material on **both** sides, so a requested
specimen boundary that lands on a chunk plane does not read as a seam. A plane
with fewer than `SEAM_MIN_SHARED = 256` shared material voxels is skipped. Keys
`seam_xct_material_*`, `seam_chunk_xct_material_*` (and `_pore_`). Two
differences from the plain version: it is computed one plane at a time, and its
combined ratio is the **plain mean of the finite per-axis ratios**, not
voxel-weighted. Only emitted when the case requested a material map
(`seam_material_available`); a full-material case emits no material keys on
purpose. The documented convention: for any case with air in the canvas, quote
the material-restricted column.

**`vae_tile_decode_control`** (`metrics.py`): reads campaign 08's real-data
tile-decode seam ratios (VAE only, no LDM) as the assembly-side control.

### 4.4 Per-chunk profile, chunk band, pore Dice across planes

**`chunk_blocks`, `chunk_seam_profile`, `chunk_profile`** (`metrics.py`): the
volume is cut on a **reference** chunk grid in raster (generation) order. Each
block owns the planes at its own origin ("chunk" family), the planes inside it
at multiples of 64 ("tile" family) and every other plane inside it
("interior"). Per block: `chunk_plane_mad`, `tile_plane_mad`, `interior_mad`
(voxel-weighted over axes), `chunk_plane_ratio = chunk / interior`,
`tile_plane_ratio = tile / interior`, plus `chunk_porosity` (material porosity
and air fraction of the block) and `chunk_s2`: S2(r) in a 128-window centred on
the block against S2(r) in a window centred on each lower plane,
`s2_relative_distance = mean |across - inside| / mean(inside)` (0 = identical
curves). `measure_assembly_modes` fits `trend` = OLS slope of each quantity
against the chunk index, to see whether errors compound along the generation
order.

**`chunk_band_profile`** (`metrics.py`): material porosity
`pore / (pore + material)` in the **8-voxel slab before each chunk end**
(`phi_-8`) and **after each chunk start** (`phi_+0`), with `chunk_bounds`
moving the planes with the overlap (chunks advance by `chunk - overlap`). The
last plane on each axis is reported separately (`phi_-8_terminal`,
`phi_+0_terminal`) because its successor's far face is the volume edge, the one
neighbour state the model was trained on. The reading is `ratio_-8`,
`ratio_+0`, `ratio_-8_worst`, `ratio_+0_worst`, `ratio_*_terminal` = slab phi /
volume phi; 1.0 is flat. Campaign 12's hybrid sampler read 0.19 at 1024.

**`pore_dice`** (`metrics.py`): Dice of the pore class between two volumes of
the same shape. Used by `measure._measure_window_phase`: the same region
generated at request offsets 0 / 16 / 32, Dice against offset 0, and the phi
difference.

**`pore_dice_across_planes`** (`metrics.py`, used by `measure_multichunk`): Dice
between the two **adjacent slices** either side of each chunk plane (`mean`,
`min`, per axis), and the same at up to 64 random interior planes that are
neither chunk nor window planes (`interior_mean`, `interior_sd`). The code says
the bare number must not be quoted: adjacent slices of any porous medium
disagree (real material scores 0.25-0.34 at every plane) and the value falls
with pore fraction. Read `ratio_to_interior`: ~1 = the plane is like ordinary
material; well below 1 = less pore agreement there.

### 4.5 Degenerate cells

**`degenerate_cells`** (`metrics.py`): per 64-voxel tile with more than half
material, `phi = pore / material`; a tile is degenerate when `phi < 1e-4` or
`phi > 0.5` (the same two numbers as the volume failure rule).
`degenerate_cell_fraction`, `n_cells`, `cell_phi_min`, `cell_phi_max`.

### 4.6 The grey air detector and cross-head disagreement

**`grey_air_detector`** (`metrics.py`) loads the eval-v3 calibration from
`runs/campaigns/04-measurement-limits/air_audit_v2/results.json` via
`scripts/analysis/_eval_v3.load_calibration`: `t_abs = 182` (the Dice-optimal
absolute u8 threshold inside `sample_mask` on 8 real volumes, Dice 0.842),
`min_cc = 300` voxels, `edge_vox = 32` (asserted equal to `EDGE_VOX`).
**`dark_components`**: `xct_u8 < 182`, connected components (default
6-connectivity of `ndimage.label`) smaller than 300 voxels dropped.

**`cross_head_disagreement`** (`metrics.py`): inside the requested material,
`disagreement_fraction` = share of material voxels that the grey detector calls
dark **and** the class head calls material; `disagreement_fraction_interior`
the same inside the 32-voxel interior; `dark_fraction_in_material`;
`label_air_fraction_in_material`; `grey_air_claimed_by_label` = share of dark
material voxels the label calls air. Reading: one head is wrong wherever the
two disagree. Real crops score 0.0000.

**Dark-but-material** (`metrics.surface_agreement` via `_dark_but_material`,
and `real_floor.surface_floor_stats` for the floor): share of voxels that the
map says are specimen and the label calls specimen (`label != air`) yet whose
grey is `< 182`. Reported over all material and again excluding a
`SURFACE_RIM_VOX = 2` band at each requested face (the partial-volume rim is
genuinely dark). Caveat: here the 182 is a hard-coded default argument
(`dark_threshold=182`), not read from the calibration file.

### 4.7 Requested geometry and the surface

**`geometry_agreement`** (`metrics.py`, needs a painted `requested_material`):
`dice_air = 2 |pred_air & req_air| / (|pred_air| + |req_air|)` where
`req_air = ~material`; `precision_air`, `recall_air`, `requested_air_fraction`,
`air_fraction_inside_material`, `air_fraction_outside_material`,
`phi_pore_inside_material`. Reading: obeying the map = high Dice, high air
outside, low air inside. No real floor exists (a real volume was not asked for
a hole).

**`height_map`** (`metrics.py`): per (y, x) column, the z of the interface of
one face (first material z for the lower face; last + 1 for the upper). Columns
without material are NaN. The "specimen" is `label != air` (pores count as
specimen).

**`surface_roughness`**: `Sa = mean |h - mean(h)|`, `Sq = sqrt(mean (h - mean(h))^2)`
in voxels, about the map's own mean plane.

**`surface_agreement`** (`metrics.py`): per face, against the **requested**
interface field (read back from the requested material map, so a rough request
is compared with its own height field): `error_mean`, `error_abs_mean`,
`error_max_abs` (the **surface position error**, voxels), `position_mean / sd`,
`roughness_sa / sq`, `requested_roughness_sa / sq`,
`roughness_ratio_to_requested = Sa_pred / Sa_requested`,
`columns_scored`, `columns_without_material`, and `outliers`: columns with
`|error| > SURFACE_OUTLIER_VOX = 4`, their count, fraction, 4-connected
clusters, singleton clusters, and how many have a pore within +/-2 voxels of the
predicted interface (`with_pore_at_face_fraction`: a surface-breaking pore is
legitimate geometry). Volume level: `air_fraction_outside_box`,
`air_fraction_inside_box`. `measure_surface` adds
`roughness_ratio_to_real_floor = Sa_pred / Sa_floor` and two gates: flat
request (controllability): both faces `error_abs_mean < 4` voxels; rough
request (realism): Sa ratio to the request **and** to the real floor both in
[0.5, 2].

**Real surface floor** (`real_floor.py:surface_floor_stats`): on real test
coupons, Sa and Sq of each face about the mean, and after removing a
least-squares plane (`_detrend_plane`; the tilt of a coupon in the scanner
frame is not roughness), plus the **correlation length**
(`_correlation_length`): per in-plane axis, the lag at which the pooled
Pearson correlation of the detrended height map
(`metrics.lag_correlation`, pairs k apart, means and variances taken over the
paired subsets at each lag, lags with fewer than 16 pairs dropped) first falls
below 1/e, linearly interpolated (`metrics.correlation_length_1_over_e`;
`None` when the curve never crosses). The floor's `detrended_sa_mean` is the
Sa the rough request is built to match and the denominator of the real-floor
ratio. The same two functions read the delivered porosity field in the
`field_stats` assessment.

**`sphere_agreement`** (`metrics.py`): exploratory, no gate or floor: air Dice,
air fraction inside / outside the sphere, phi inside, a radial profile in
2-voxel shells, and `r_at_air_half` (the radius where the shell air fraction
first reaches 0.5) with its error against the requested radius, optionally per
octant.

### 4.8 Layup readers

**`layup_recovery`** (`metrics.py`, needs `requested_layup`,
`requested_ply_thickness_vox`). The estimators are imported from
`scripts/analysis/layup_roundtrip.measure_volume`, the T-I code of campaign 08,
on one centred 1024 x 1024 window per slice:

- `fft_slice`: angular spectrum of the grey slices in a wavelength band tied to
  the ply pitch, averaged over the slices of each ply block, prepared with the
  axis excision; the angle per ply is the argmax of the 1-degree angular
  density (`ti.THETA`), weight = max - min of the density.
- `pore_axes`: the same from the principal axes of the **predicted** pore
  class (`ti.pore_axes_density`).

Scoring is **direct** (`layup_roundtrip.score_recovery`): errors
`wrap180(measured - requested)` with no offset, sign flip or face reversal
granted. Per reader: `median_abs_error_deg`, `max_abs_error_deg`,
`frac_within_10`, `strict_class_accuracy` (**4-class accuracy**: both angles
rounded to the nearest of 0 / 45 / 90 / 135 by `angle_class`, share of plies
that agree), `per_ply_hit`, and `recovered_ply_count` (`_run_count`: one plus
the number of class changes between consecutive ply blocks; two identical
adjacent plies read as one). **`layup_floor`** reads the real-scan reader
floor from campaign 08 (`angle_reader_floor/results.json`), never hard-coded.
Caveat in the code: `pore_axes` on real data read the **stored** mask, so its
floor is a ceiling no predicted-mask reader can beat.

### 4.9 Microstructure statistics (`microstructure.py`)

All four are set-to-set distances. Each is reported three ways
(`measure.measure_microstructure`): generated vs real, real vs real (two
disjoint 128^3 crops of **one** test panel matched to the requested porosity,
`real_floor.build_micro_reference`), and `ratio = generated / floor`
(`microstructure.ratio`; `None` when the floor is 0 or missing). Reading: ratio
1 = the generated set is as close to real material as real material is to
itself, which is as close as the measurement can tell. Connected components
are **6-connected** throughout.

**S2(r) radial W1** - `s2_radial`: on a 128^3 window (`S2_WINDOW`), the pore
mask is multiplied by a 3-D Hann window, autocorrelated by FFT, and divided by
the window's own autocorrelation so that `S2(0) = phi`. Radial shells of 1
voxel out to `S2_R_MAX = 48`. `s2_profile` averages over every 128-window on a
64-voxel stride that is at least 99 % material. `compare_sets` takes the mean
curve of each set and `curve_w1(r, a, b)`: both curves clipped at 0 and
normalised to unit mass, then the 1-D Wasserstein-1 over r. Units: voxels. The
amplitude (porosity) is discarded; this is a distance between curve **shapes**.

**Pore-size distribution W1** - `pore_diameters`: equivalent spherical diameter
`(6V / pi)^(1/3)` per 6-connected pore inside the material (a one-voxel pore is
1.24). `psd_w1 = wasserstein_distance(diameters_a, diameters_b)` on the pooled
diameters of each set. Units: voxels.

**Ripley's K log ratio** - `ripleys_k`: border-corrected (reduced-sample)
estimator on pore centroids, `K(r) = sum_{i: b_i >= r} n_i(r) / (lambda x #{i: b_i >= r})`,
where `b_i` is the distance of pore i to the nearest face, `n_i(r)` the other
pores within r, `lambda = N / V`; r = 1..24; NaN where no pore is far enough
from the faces; not computed under 20 pores. Under complete spatial randomness
`K(r) = (4/3) pi r^3` (`csr_k`), which the unit test checks; `k_over_csr > 1`
is clustering. The set distance is `log_ratio_distance = mean_r |log(K_a / K_b)|`
over the mean curves (0 = agree everywhere; 0.69 = a factor of two on average).

**FID on 2-D slices** - `fid_crops`: 64 x 64 crops at native resolution from
random slices along each axis (`FID_AXES` = axial, coronal, sagittal), kept
only when the crop lies entirely inside the requested material; 5000 crops per
axis per set, shared evenly over the cases (`collect_fid_crops`).
`fid_preprocess` / `fid_input_from_grey`: grey [0, 1] -> bilinear resize to
299 x 299 -> replicate to 3 channels -> ImageNet-normalise with the weights'
own preset mean / std; torchvision's `inception_v3(weights=IMAGENET1K_V1,
transform_input=True)` then maps that to [-1, 1] itself, which is the
`pytorch-fid` input range. Features: 2048-d `avgpool` (pool3).
`frechet_distance = |mu_a - mu_b|^2 + tr(Sa + Sb - 2 sqrt(Sa Sb))`. `fid_between`
reports the per-axis FID and their **mean**. Caveats stated in the code: a full
slice resized to 299 would shrink a 1.79-voxel pore to 0.18 pixels, hence the
native crops; the weights are torchvision's, not the TF-ported FID-Inception
weights `pytorch-fid` loads, so absolute values are comparable within this
implementation only; every FID before 2026-09-08 fed unnormalised input and is
void.

Also in the module: `s2_directional` / `s2_directional_profile` (S2 along each
axis separately, `anisotropy` = mean spread of the three curves over their
mean), and `_psd_histogram` on fixed bin edges for the figures.

### 4.10 Memorisation (`memorisation.py`)

**Statistic:** Favero's ratio `ratio = ||x - x'|| / ||x - x''||`
(`favero_ratio`): the query's distance to its nearest bank row over its
distance to the second-nearest. A copy sits on one training patch and a normal
distance from every other, so its ratio falls toward 0; a fresh sample sits a
typical distance from both, so its ratio approaches 1. Verdict:
`ratio < RATIO_THRESHOLD = 1/3` is memorised. `d2 = 0` gives ratio 0.

**Queries:** every 64^3 whole-material patch of the generated volumes of the
`sampler`, `porosity_global`, `multichunk` and `assembly_modes` assessments
(volumes above 256 tiles skipped on cost). In **latent space** the generated
volume is re-encoded through the VAE (the store holds posterior means, the LDM
samples draws), latents divided by the store's scalar sigma_hat. In **grey
space** the raw decoded patch is used.

**Bank:** every row of the **train** split whose origin lies on the
**stride-64** grid (`BANK_STRIDE = 64`, `PatchStore`). The store is built on a
32-voxel stride, so without the restriction the second-nearest neighbour would
be the same material shifted by 32 voxels. The search is exhaustive over the
bank (`search`, `Top2`), not a sample.

**Floor:** `FLOOR_PATCHES = 512` real **validation** patches, searched the same
way. In latent space one floor; in grey space two: `grey` = the val patch's own
stored posterior mean decoded by the same frozen VAE (like-for-like, carries
the decoder's error) and `grey_raw` = the untouched scan patch (what the ratio
would be with no decoder error). Read the generated grey row against `grey`.

**Outputs** (`summarise`): per set and space `n`, `nn_distance_mean / min`,
`second_nn_distance_mean`, `ratio_mean / median / min / p5`, `n_memorised`,
`frac_memorised`; also by requested phi and by neighbour-availability bucket
(`window_states` rebuilds, per window, whether a face looked into a chunk that
was already solved). `detection_rate` reports what fraction of planted copies
(shifts 0 / 8 / 16 / 32) the 1/3 rule flags. Caveat in the code
(`BANK_COVERAGE_NOTE`): the bank is one eighth of the store, so a patch shifted
by 32 voxels exists in the store but not in the bank.

### 4.11 LDM latent probes that are NOT in eval_v4

`x0 saturation` and the `std ratio` are not computed by eval_v4. They are
training-time probes (section 6). The eval_v4 manifests carry no such field.

### 4.12 Findings writers (`report.py`)

`report_porosity_global` prints the per-level table (requested, delivered,
|error|, within gate, fail) and the OLS line. `report_assembly` prints the
four seam ratios, cross-head disagreement (all and interior), the campaign-08
VAE tile-decode control, and the window-phase pore Dice. `report_microstructure`
prints each statistic as "generated vs real | real vs real (floor) | ratio"
with the fixed reading "1 is the floor, lower is better", warns when the
closest real crop misses the requested porosity by more than 0.005, and appends
`memorisation_section` (generated row, VAE round-trip floor, raw-scan floor,
by phi, by neighbour bucket). `report_surface` prints the gates of section 4.7.

---

## 5. Campaign 37: improved precision and recall

Script: `scripts/analysis/precision_recall.py`. Kynkaanniemi et al. 2019,
"Improved precision and recall".

**Feature space:** exactly the FID space of section 4.9: Inception pool3
(2048-d) on 64 x 64 grey crops that lie entirely inside material, along the
three axes (`MS.collect_fid_crops`, `MS.inception_features`). `--crops`
(default 10 000) per set, split evenly over the three axes (3333 per axis) and
over the cases of the set.

**Algorithm** (`knn_radii`, `inside`, `precision_recall`): for each point of a
set, the radius is the Euclidean distance to its k-th nearest **other** point
of the same set (`kthvalue(k + 1)` because the point itself is at distance 0),
k = 3. The manifold of a set is the union of the balls. `precision` = share of
generated features inside at least one real ball (fidelity); `recall` = share of
real features inside at least one generated ball (coverage). Both use the
pooled crops of the three axes. FID is reported beside them as the per-axis
mean (`score`).

**Sets and floor:** per porosity level 0.01 / 0.03 / 0.06 of campaign 18: the
real floor is `real_a` vs `real_b` (two disjoint crops of one panel); models
are scored against the union `real_a + real_b`. Per campaign-36 bin: the a2s
twins against their own real coupons (crops cut from the coupon arrays), floor
= real coupons odd vs even index. Reading: precision and recall near the
real-vs-real floor is as good as the measurement can tell; the baselines
(SliceGAN, pixel DDPM) are unconditional and are scored against every level.

---

## 6. LDM training-time probes

### 6.1 `generation_eval` (`src/poregen/diffusion/generation_eval.py`)

Run during LDM training on `n_samples = 64` latents sampled with DDIM (50
steps) and EMA weights, conditioned on random validation rows with every
neighbour UNKNOWN. Keys become `gen/<key>`.

- `x0_clamp_sat_frac`: fraction of x0-prediction elements that hit the +/-10
  clamp (`X0_CLAMP = 10`) during sampling, averaged over steps
  (`sampler.DDIMSampler.sample_batch` with `return_x0_saturation`).
- `ch<c>_mean`, `ch<c>_std_ratio`, `mean_abs_max`, `std_ratio_avg`: in the
  normalised latent space the real train distribution has mean 0 and std 1 per
  channel, so the generated per-channel std **is** the ratio to the real std.
  `std_ratio_avg` is the mean over channels. Target 1.
- `por_cond_mae`: the latents are denormalised, decoded by the frozen 3-class
  VAE, and the pore fraction of the argmax label is compared with the requested
  porosity of each row: `mean |phi_delivered - phi_requested|` (full-patch
  porosity). The direct conditional-adherence metric.
- `por_mean`, `por_std`, `air_mean`, `real_por_mean`, `real_por_std`.
- `por_w1`: `wasserstein_distance(delivered porosities, real val porosities)`.
- `degenerate_frac`: share of decoded volumes with pore fraction `< 1e-4` or
  `> 0.5`.

### 6.2 `diag_ldm_samples.py` (`scripts/`)

A read-only checkpoint health check. `bucket_metrics` reports per conditioning
bucket: `std_ratio` (mean per-channel std, as above), `std_ratio_vs_sampled`
(divided by a measured per-channel reference of the sampled, rescaled train
latents, `latents.sampled_std_reference`), `channels_outside_10pct` (channels
outside [0.90, 1.10] of that reference), `mean_abs_max`, `x0_sat`, `por_mean`,
`por_std`, `por_mae`, `air_mean`, `degen` (the plain degenerate count) and
`degen_unrequested` (saturated, or empty against a request above 1e-4; an
obeyed phi = 0 request is not a failure).

**Convergence verdict** (`print_verdict`): the change of the `ema_ddim50`
overall `std_ratio` since the previous entry of `convergence_check.jsonl`:
`<= -5 %` CONVERGING, `>= +5 %` REGRESSING, otherwise STALLED
(`VERDICT_BAND_PCT = 5`).

**Kill switch** (`killswitch_check`): the same rows and seed are asked for
phi = 0.005 and phi = 0.05 (`KILLSWITCH_ASKS`); `separation = por_hi - por_lo`,
`direction_ok = por_hi > por_lo`. A reversed kill switch or any non-finite
value is the only fatal failure (`fatal_failures`); unrequested degenerate cells
are reported loudly but do not stop the chain.

---

## 7. Campaign 14: downstream utility

Script: `scripts/analysis/downstream_utility.py`. Three arms (real, synthetic,
real + synthetic) train the same 3-D U-Net at a fixed patch budget and are
scored on the real held-out test panels with `SegmentationScore`:

- `dice_material`, `dice_pore`, `dice_air`, `porosity_mae`, `air_mae` (and
  the bias / pred-mean keys) from `metrics/seg.py:multiclass_metrics`, averaged
  per **patch** over the whole test set (patch-count-weighted, not a mean of
  batch means).
- `porosity_mae_bin_0..3` and counts from `porosity_binned_mae` with bins
  (0, 0.01, 0.03, 0.06, inf) over every test patch.
- `aggregate`: mean and sd over seeds per arm; headline keys `dice_pore`,
  `dice_air`, `porosity_mae`, `air_mae`.

Reading: the real-only arm is the reference row; the synthetic and mixed arms
read as a mixing curve at constant data budget.

---

## 8. Doc mismatches

Where `docs/metrics_guide.md` (Spanish, VAE-era) or `docs/eval_methodology.md`
disagree with the code as read today.

**`docs/metrics_guide.md`**

1. `val/mae` is given as `mean |sigmoid(x_hat) - x|`. The engine computes
   `mean |clamp(x_hat, 0, 1) - x|` with no sigmoid (since 2026-09-01).
   `metrics/recon.py:mae` still applies a sigmoid but the trainer does not call
   it.
2. `xct_loss` is described as Charbonnier with eps = 1e-6. In code it is
   `loss.xct_loss_type` (l1 / mse / charbonnier); the recipe decides.
3. The `total` formula lacks `class_total` (the 3-class head), the LPIPS term
   and the `rombach_sum` KL option.
4. `porosity_mae` is given with a sigmoid over a binary mask. For the 3-class
   head it is the argmax pore fraction (`multiclass_metrics`); `dice_material /
   pore / air`, `air_mae` and `air_bias` are not in the guide.
5. Section 7 (offline eval) describes `memorization_nn_dist` as a mean
   minimum L2 distance and `xct_boundary_mae` as a seam metric. These belong
   to the eval-v3 path (`eval/metrics.py`). eval_v4 replaced them with the
   Favero ratio and the seam ratio.
6. The guide's `s2_wasserstein` (eval v3) is a W1 between two curves as raw
   values; eval_v4's `curve_w1` normalises the curves to unit mass first and
   its support is the radius in voxels.

**`docs/eval_methodology.md`**

1. "Chunk-plane porosity band" describes `scripts/analysis/chunk_plane_profile.py`
   with slabs at offsets -32 ... +24 and a "+/-20 %" criterion.
   `metrics.chunk_band_profile` computes only the -8 and +0 slabs, separates
   the terminal planes, and carries no criterion. The script and the module are
   two implementations.
2. "A per-axis ratio is `null` when that axis carries no seam plane" - the
   code returns `float("nan")` in `seam_discontinuity` and
   `seam_discontinuity_material`; it reads as null only after JSON writing.
3. The doc says the plain and material-restricted ratios "reproduce the plain
   statistic exactly" with nothing excluded. Per axis that is true; the
   **combined** ratios differ in form (voxel-weighted vs plain mean of per-axis
   ratios, section 4.3).
4. The doc does not state that the combined seam ratio includes the interior
   planes of an axis that has no seam planes (the reason the real 1024-wide
   floor reads 0.90 on the chunk ratio while the window ratio reads 0.96).
5. The dark-but-material threshold is described through the calibrated
   detector; in `surface_agreement` and `surface_floor_stats` it is a
   hard-coded default of 182, not read from the calibration file.
6. The FID section says 5000 crops per axis per set. Campaign 37's
   precision / recall uses 10 000 crops per set split over three axes
   (about 3333 per axis), so its FID column is not the eval_v4 FID at the same
   sample size.
7. `docs/eval_methodology.md` does not describe campaigns 32, 33, 36, 37 or
   the LDM training probes; they are only in the scripts' headers.

---

## 9. Is the precision / recall the one in Rombach et al. 2022?

The statistic is the same: Kynkaanniemi et al. 2019's improved precision and
recall, k-NN manifolds with k = 3, precision = share of generated points inside
the real manifold, recall = share of real points inside the generated manifold,
in Inception pool3 feature space. The use differs in every other respect.
Rombach et al. report it through torch-fidelity on about 50 000 generated
images at full image size against the training images, with the TF-ported
FID-Inception weights that torch-fidelity loads, and with no real-vs-real
floor. Ours scores 10 000 crops per set, each a 64 x 64 grey slice of a 3-D
volume cut inside the material and upsampled bilinearly to 299 x 299, using
torchvision's IMAGENET1K_V1 Inception weights, pooled over three slice
orientations, and every row is read beside a real-vs-real floor measured with
the same crops and the same k. So the numbers are the same kind of number and
are read the same way (precision = fidelity, recall = coverage), but they are
not comparable in absolute value to a table in that paper: the inputs are
crops of volumes rather than images, the extractor weights differ, the sample
size is five times smaller, and the reference set is held-out real material
rather than the training set.

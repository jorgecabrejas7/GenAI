# PoreGen — Model Architecture

**Joint XCT + Pore Mask VAE • Modular Architectures • Diffusion-Ready Latents**

---

## 0. Key Design Choices

- **All model heads output logits** — sigmoid is applied explicitly in loss
  and metric functions for numerical stability and consistency.
- **Diffusion patch size:** `64×64×64`
- **Latent shape:** `(z_channels, 16, 16, 16)` with default `z_channels=8`
- **Downsample factor:** 4× (2 stride-2 conv stages)
- **Overlap stride:** 32 voxels → 8 latent cells per axis

---

## 1. VAE Architectures

### `ConvVAE3D` (registry key: `"conv"`)

Baseline model — no skip connections, no attention.

```
Encoder:
  (B,  2, 64, 64, 64)  →  stride-2 Conv3d + GN + SiLU  →  (B, 32, 32, 32, 32)
                        →  stride-2 Conv3d + GN + SiLU  →  (B, 64, 16, 16, 16)
                        →  1×1 Conv3d                   →  mu, logvar  (B, 8, 16, 16, 16)

Decoder:
  z (B, 8, 16,16,16)   →  ConvTranspose3d + GN + SiLU  →  (B, 32, 32, 32, 32)
                        →  ConvTranspose3d + GN + SiLU  →  (B, 32, 64, 64, 64)

Heads:
  1×1 Conv3d → xct_logits   (B, 1, 64, 64, 64)
  1×1 Conv3d → mask_logits  (B, 1, 64, 64, 64)
```

**Use case:** fast iteration, stable training, sufficient for initial experiments.

### `UNetVAE3D` (registry key: `"unet"`)

Same encoder topology, but the decoder receives skip features from
corresponding encoder stages (concatenated before 3×3 refinement).

```
Encoder:
  Same as ConvVAE3D, but skip features are stored at each stage.

Decoder:
  z → from_z projection → concat skip[1] → UpBlock → concat skip[0] → UpBlock → heads
```

**Use case:** sharper reconstructions (especially mask boundaries), better
foundation for downstream diffusion training where reconstruction sharpness
matters.

### Adding new architectures

1. Create `src/poregen/models/vae/<name>.py`
2. Decorate your class with `@register_vae("name")`
3. Import it in `src/poregen/models/vae/__init__.py`
4. Use `build_vae("name")` in notebooks

---

## 2. Standardised API

```python
forward(xct: Tensor, mask: Tensor) -> VAEOutput
```

- **Input:** `xct` (B, 1, 64, 64, 64) float32 [0,1], `mask` (B, 1, 64, 64, 64) float32 {0,1}
- **Model concatenates** both channels → (B, 2, 64, 64, 64)

### `VAEOutput` dataclass

| Field | Shape | Description |
|---|---|---|
| `xct_logits` | (B, 1, 64, 64, 64) | Raw XCT output — apply `sigmoid` before comparison |
| `mask_logits` | (B, 1, 64, 64, 64) | Raw mask output — use with `BCEWithLogitsLoss` |
| `mu` | (B, 8, 16, 16, 16) | Posterior mean |
| `logvar` | (B, 8, 16, 16, 16) | Posterior log-variance |
| `z` | (B, 8, 16, 16, 16) | Sampled latent (reparameterised) |

---

## 3. Training Objective

### Reconstruction Loss (XCT channel)

| Loss | Function | Notes |
|---|---|---|
| L1 | `l1_loss(logits, target)` | Default; robust to outliers |
| MSE | `mse_loss(logits, target)` | Smoother gradients |
| Charbonnier | `charbonnier_loss(logits, target)` | Smooth L1 approximation |

All apply `sigmoid(logits)` internally.

### Mask Loss

| Component | Function | Notes |
|---|---|---|
| BCE | `bce_logits_loss(logits, target)` | Numerically stable |
| Dice | `dice_loss(logits, target)` | Overlap-based, handles class imbalance |
| Tversky | `tversky_loss(logits, target, α=0.3, β=0.7)` | Recall-biased for sparse pores |

Default: BCE + Dice with configurable weights.
Optional: switch to BCE + Tversky for highly sparse masks.

### KL Divergence

$$KL(q(z|x) \| \mathcal{N}(0,I))$$

- **β warm-up:** linear ramp from 0 → `max_beta` over `warmup_steps`
- **Free-bits:** per-channel minimum KL (default 0.25) — prevents posterior collapse
  while allowing the model to use latent capacity efficiently

### Total Loss

```
total = xct_weight × xct_loss + mask_total + β × KL
```

Returns a dict: `total`, `xct_loss`, `mask_bce`, `mask_dice` (or `mask_tversky`),
`kl`, `beta`, `freebits_used`.

---

## 4. Evaluation Metrics

### Reconstruction (XCT)

| Metric | Notes |
|---|---|
| MSE / MAE | Standard pixel-level errors |
| PSNR | Signal-to-noise ratio (dB) |
| Sharpness proxy | Mean absolute gradient (finite differences) |

### Segmentation (Mask)

| Metric | Notes |
|---|---|
| Dice, IoU | Overlap metrics |
| Precision, Recall, F1 | Classification-style |

All reported as `*_all` and `*_pos_only` (excluding samples with empty GT).

### Latent Diagnostics

| Metric | Notes |
|---|---|
| Active units | Fraction with Var(μ_c) > threshold |
| KL per channel | Identifies collapsed or dominant channels |
| μ/logvar stats | Mean, std for monitoring posterior health |

---

## 5. Future: Diffusion (Latent Space)

The diffusion model will operate on latent patches sampled from the VAE encoder:

- **Clean latent target:** `z₀ = μ + σ ⊙ ε`, `ε ~ N(0, I)`
- **Shape:** `(8, 16, 16, 16)`
- **Context:** 6-connected neighbors (±x, ±y, ±z)
- **Conditioning:** global porosity + normalised spatial position
- **Inference:** anchor-first strategy with latent-space overlap clamping

### Key ideas

- **Anchor-first inference:** generate sparse anchor patches first, then fill
  remaining patches conditioned on anchors, limiting error propagation.
- **Overlap clamping:** at each diffusion step, force overlap regions to match
  reference latents with noise-consistent rescaling.
- **Reconstruction:** decode each latent patch, combine overlaps for mask
  (threshold once) and XCT (light blending).

---

## 6. Reproducibility

- All architectural decisions documented here and in `example_vae.yaml`
- Latent interfaces are stable and decoupled from diffusion
- Deterministic seeding via `seed_everything()`
- Checkpoints contain model + optimizer + scaler + step + metadata
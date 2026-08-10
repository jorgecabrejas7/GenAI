# VAE Architecture & Training Pipeline

Reference for the VAE model family and `src/poregen/training/engine.py`. See `docs/metrics_guide.md` for loss/metric formulas and health ranges.

## Model family

Registered with `@register_vae("key")`, built via `build_vae("key", **kwargs)` (`src/poregen/models/vae/registry.py`).

**Active architecture (R03–R05):** `v2.conv_noattn_dualbranch` (R04/R05) or `v2.conv_noattn` (R03).

Data flow for `v2.conv_noattn`:

```
XCT (B,1,64³) → encoder (2× down_block_v2) → (B,64,16³) → to_mu / to_logvar → z (B,16,16³)
z → decoder (2× up_block_v2) → (B,32,64³) → xct_head / mask_head → logits (B,1,64³)
```

The encoder receives **XCT only** — `mask` is a reconstruction target, never an encoder input. The mask head is a segmentation head predicting pore density from XCT-latent features. All outputs are logits; sigmoid/threshold happens in loss/metric code, never inside the model.

`VAEOutput` (`models/vae/base.py`): `{xct_logits, mask_logits, mu, logvar, z}`.

## Training pipeline

`src/poregen/training/engine.py`:

- **`train_step`** — forward, adversarial generator pass (if discriminator enabled), backward, unscale, per-module grad norms, clip, optimizer step. Returns `(losses_dict, grad_norm, latent_moments, module_grad_norms)`. `latent_moments` are GPU tensors — no `.cpu()` until the log guard fires.
- **`eval_step`** — returns `(losses_dict, VAEOutput, xct_dev, mask_dev)`. `xct_dev`/`mask_dev` are device tensors, reused by the caller to avoid a double H→D transfer.
- **`train_loop`** — main loop. With `compile=True`, compiles model + discriminator + loss_fn (`max-autotune, dynamic=False`). Calls `_run_eval` every `eval_every` steps, `_save_patch_samples` every `sample_every` steps. Checkpoints via an async background thread.

Eval accumulators (`mae_acc`, `sharp_*_acc`, porosity lists) stay on GPU as `.detach()` tensors through the loop; `.item()` fires once, post-loop.

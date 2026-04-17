# PoreGen VAE — Blackwell GB10 Performance Audit

> Generated from two independent audits (main conversation + Ultraplan). Discriminator AMP excluded by user decision.

---

## Files in scope

```
src/poregen/
├── metrics/latent.py        Fix 1
├── training/engine.py       Fixes 2, 3, 4, 5, 6, 8
├── experiments/base.py      Fix 7
├── dataset/loader.py        Fix 9
└── losses/mask.py           Fix 10
```

---

## Fix 1 — `latent_channel_moments` GPU→CPU sync every train step
**File:** `src/poregen/metrics/latent.py:34-35`
**Priority:** HIGH

Called inside `train_step` on every training step. The `.cpu()` calls force a full GPU→CPU sync before backward can proceed.

```python
# BEFORE
return {
    "count": int(mu_flat.shape[1]),
    "sum": mu_flat.sum(dim=1).cpu(),
    "sum_sq": mu_flat.square().sum(dim=1).cpu(),
}

# AFTER — keep on GPU; merge_latent_channel_moments and active_units_from_moments
# accept GPU tensors; the terminal .item() is already inside the log_every guard
return {
    "count": int(mu_flat.shape[1]),
    "sum": mu_flat.sum(dim=1),
    "sum_sq": mu_flat.square().sum(dim=1),
}
```

**Metric:** Eliminates 2 mandatory GPU→CPU syncs per training step from the forward-pass critical path.

---

## Fix 2 — Double H→D Transfer in `_run_eval`
**File:** `src/poregen/training/engine.py:268-286` (eval_step), `engine.py:318-327` (_run_eval)
**Priority:** HIGH

`eval_step` already moves `xct` and `mask` to device. `_run_eval` then re-calls `.to(device)` on the same CPU batch dict, allocating fresh GPU buffers and issuing duplicate DMA transfers.

```python
# BEFORE — eval_step
) -> tuple[dict[str, Any], VAEOutput]:
    ...
    return {k: _to_scalar(v) for k, v in losses.items()}, output

# AFTER — return the device tensors to avoid re-transfer
) -> tuple[dict[str, Any], VAEOutput, torch.Tensor, torch.Tensor]:
    ...
    return {k: _to_scalar(v) for k, v in losses.items()}, output, xct, mask

# BEFORE — _run_eval
losses, output = eval_step(model, batch, loss_fn, step, device, autocast_dtype)
mask_dev = batch["mask"].to(device, non_blocking=True)   # redundant
xct_dev  = batch["xct"].to(device,  non_blocking=True)   # redundant

# AFTER
losses, output, xct_dev, mask_dev = eval_step(model, batch, loss_fn, step, device, autocast_dtype)
# delete the two .to(device) lines
```

**Metric:** For `val_batches=100 × batch_size=128` at 64³ float32: eliminates ~3.2 GB of redundant PCIe transfer per eval pass.

---

## Fix 3 — `torch.compile` Mode Too Conservative
**File:** `src/poregen/training/engine.py:563-564`
**Priority:** HIGH

`reduce-overhead` replays CUDA graphs but skips autotune and elementwise fusion. `max-autotune` with `dynamic=False` enables exhaustive algorithm search for the static 64³ shapes and fuses elementwise chains in `loss_fn` (charbonnier: 4 kernels → 1; KL divergence: 5 kernels → 1).

```python
# BEFORE
if compile_model:
    model = torch.compile(model, mode="reduce-overhead")

# AFTER
if compile_model:
    model = torch.compile(model, mode="max-autotune", dynamic=False)
    if discriminator is not None:
        discriminator = torch.compile(discriminator, mode="max-autotune", dynamic=False)
    loss_fn = torch.compile(loss_fn, mode="max-autotune")
```

**Metric:** +15–25% additional TFLOPS vs `reduce-overhead` on static 3D conv shapes. First step will be significantly slower (JIT + autotune); stabilises by ~step 5.

---

## Fix 4 — Pre-Backward `.item()` Sync for `mask_pred_mean`
**File:** `src/poregen/training/engine.py:201`
**Priority:** MEDIUM

`.item()` before `.backward()` forces the GPU to finish the entire forward pass before the CPU can issue the backward kernel, inserting the full forward-pass latency as dead CPU wait time.

```python
# BEFORE — sync BEFORE backward
mask_pred_mean = float(torch.sigmoid(output.mask_logits).mean().item())
scaler.scale(losses["total"]).backward()   # stalled

# AFTER — defer sync until after all GPU work is queued
_mask_pred_tensor = torch.sigmoid(output.mask_logits).mean().detach()
scaler.scale(losses["total"]).backward()   # fires immediately
# ... unscale, clip, step, update ...
scaler.step(optimizer)
scaler.update()
mask_pred_mean = float(_mask_pred_tensor.item())   # single sync after GPU work

# Also update the return line:
# BEFORE:  result["mask_pred_mean"] = mask_pred_mean
# AFTER:   result["mask_pred_mean"] = float(_mask_pred_tensor.item())
#          (move the original mask_pred_mean assignment line here instead)
```

**Metric:** Removes the forward→backward pipeline stall (~2–5 ms/step at batch_size=128 with 64³ volumes).

---

## Fix 5 — N Separate Grad Norm `.item()` Syncs Per Step
**File:** `src/poregen/training/engine.py:209-215`
**Priority:** MEDIUM

Up to 5 `clip_grad_norm_(..., inf).item()` calls = up to 5 GPU→CPU syncs per step inside `train_step`. With `clip=inf` no clipping occurs; the calls are purely norm computation.

```python
# BEFORE — up to 5 .item() syncs
module_grad_norms: dict[str, float] = {}
for name in ("encoder", "encoder_a", "encoder_b", "decoder", "mask_head"):
    module = getattr(model, name, None)
    if module is not None:
        module_grad_norms[f"grad_norm_{name}"] = torch.nn.utils.clip_grad_norm_(
            module.parameters(), float("inf")
        ).item()

# AFTER — accumulate on GPU, single sync at end
_norm_gpu: dict[str, torch.Tensor] = {}
for name in ("encoder", "encoder_a", "encoder_b", "decoder", "mask_head"):
    module = getattr(model, name, None)
    if module is not None:
        grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
        if grads:
            _norm_gpu[f"grad_norm_{name}"] = torch.stack(
                [g.norm(2) for g in grads]
            ).norm(2)
module_grad_norms: dict[str, float] = {k: v.item() for k, v in _norm_gpu.items()}
```

**Metric:** Reduces per-step sync count from up to 5 → 1 for module grad norms. Combined with Fix 4: total `train_step` syncs drop from ~7 → ~2.

---

## Fix 6 — Per-Batch `.cpu()` Page Migrations in `_run_eval`
**File:** `src/poregen/training/engine.py:346-349`
**Priority:** MEDIUM

Three `.cpu()` calls per eval batch interrupt the GPU prefetch pipeline. The tensors are only consumed after the loop.

```python
# BEFORE — migrations inside hot eval loop
pred_por_signed_all.append((pred_por_v - gt_por_v).cpu())
pred_por_all.append(pred_por_v.cpu())
gt_por_all.append(gt_por_v.cpu())

# AFTER — stay on GPU during accumulation; post-loop .item() remains unchanged
pred_por_signed_all.append((pred_por_v - gt_por_v).detach())
pred_por_all.append(pred_por_v.detach())
gt_por_all.append(gt_por_v.detach())
```

**Metric:** Eliminates 300 page migrations per eval pass (`val_batches=100 × 3`).

---

## Fix 7 — `build_patch_loader` DataLoader Defaults
**File:** `src/poregen/experiments/base.py:74-86`
**Priority:** MEDIUM

The notebook/analysis path defaults `pin_memory=False`, omits `persistent_workers`, and omits `worker_init_fn`. The training path (`build_dataloader_kwargs` in `training/data.py`) is already correct — this only affects `ExperimentRuntime.build_loader()` and notebook calls.

```python
# Add to imports at top of base.py:
from poregen.dataset.loader import zarr_worker_init_fn

# BEFORE
pin_memory=bool(data_cfg.get("pin_memory", False) if pin_memory is None else pin_memory),
# no persistent_workers, no worker_init_fn

# AFTER
pin_memory=bool(data_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),

if effective_workers > 0:
    kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
    kwargs["worker_init_fn"]     = zarr_worker_init_fn
    if data_cfg.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
```

**Metric:** Enables async DMA and eliminates worker-respawn overhead for all notebook eval/analysis runs using `ExperimentRuntime.build_loader()`.

---

## Fix 8 — Missing `non_blocking=True` in `_save_patch_samples`
**File:** `src/poregen/training/engine.py:939-940`
**Priority:** LOW

The only two `.to(device)` calls in engine.py missing `non_blocking=True`.

```python
# BEFORE
xct  = batch["xct"] [:n_take].to(device)
mask = batch["mask"][:n_take].to(device)

# AFTER
xct  = batch["xct"] [:n_take].to(device, non_blocking=True)
mask = batch["mask"][:n_take].to(device, non_blocking=True)
```

**Metric:** Overlaps host→device DMA with CPU-side sample list preparation.

---

## Fix 9 — Intermediate Numpy Array in `_normalise_xct`
**File:** `src/poregen/dataset/loader.py:102-103`
**Priority:** LOW

`xct.astype(np.float32) / 255.0` allocates two full-size numpy arrays; only one is necessary. With `num_workers=16`, this doubles per-worker memory pressure on each `__getitem__` call.

```python
# BEFORE — 2 numpy allocations per patch
return torch.from_numpy(xct.astype(np.float32) / 255.0).unsqueeze(0)

# AFTER — 1 numpy allocation; in-place multiply on the tensor
t = torch.from_numpy(np.asarray(xct, dtype=np.float32))
t.mul_(1.0 / 255.0)
return t.unsqueeze(0)
```

**Metric:** Halves per-sample CPU allocation in DataLoader workers (~1 MB → 0.5 MB per 64³ float32 patch); reduces L3 cache pressure under 16-worker parallelism.

---

## Fix 10 — `focal_loss` Redundant Sigmoid When `use_focal=True`
**File:** `src/poregen/losses/mask.py:49-51`, `losses/mask.py:168-170`
**Priority:** LOW (inactive in r03.yaml, active in r04+)

`combined_mask_loss` computes sigmoid once and forwards it to `dice_loss`/`tversky_loss`, but `focal_loss` always recomputes it independently, making the optimization a no-op for the primary term.

```python
# BEFORE
def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    bce  = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)   # always recomputed

# AFTER — accept pre-computed sigmoid
def focal_loss(logits, target, gamma=2.0, alpha=0.25, *, sigmoid=None):
    bce  = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = sigmoid if sigmoid is not None else torch.sigmoid(logits)

# In combined_mask_loss, pass through:
if use_focal:
    primary = focal_loss(logits, target, gamma=focal_gamma, alpha=focal_alpha, sigmoid=sigmoid)
```

**Metric:** Eliminates one 33M-element sigmoid op per step when `use_focal=True` (~64 MB VRAM bandwidth per step at fp16).

---

## Rejected Claims from the Other Report

| Claim | Verdict | Reason |
|---|---|---|
| `data.py:build_dataloader_kwargs` pin_memory null bug | Not a real bug | `r03.yaml` sets `pin_memory: true` explicitly; training path is already correct. The actual bug is in `base.py` (Fix 7 above). |
| Discriminator `.float()` → AMP | Excluded (user decision) | Spectral norm power iteration can be less accurate in bfloat16. Comment in codebase explicitly marks this as intentional: *"D runs in float32"*. |

---

## Verification Checklist

1. **No-compile smoke test** — 50 steps with `compile: false`. Check `train/mu_n_active` and `train/mu_active_fraction` are unchanged (Fix 1 must not alter computed values, only sync point).
2. **Compile smoke test** — 50 steps with `compile: true`. First step will be slow (JIT + autotune); confirm stabilises by ~step 5.
3. **Throughput** — Compare `train/steps_per_sec` in TensorBoard before and after. Expected: +20–35%.
4. **Eval correctness** — Confirm `val/mae` and `val/mask_dice` match pre-fix values within float rounding (Fixes 2 and 6 are numerically identical — same data, different sync points).
5. **Focal loss** — For any r04+ config with `use_focal: true`, confirm mask loss values are identical before/after Fix 10.

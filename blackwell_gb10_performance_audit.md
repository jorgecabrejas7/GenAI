# Deep Code Audit Report: ML Training Performance on NVIDIA Blackwell GB10

## Executive Summary
This report details the findings of a performance code audit conducted to identify throughput and utilization bottlenecks on NVIDIA Blackwell GB10 architecture (128GB Unified Memory). The audit targeted PyTorch training loops, data pipelines, hardware acceleration utilization, and memory management. Four critical areas were identified that currently degrade performance: pipeline data stalls, sub-optimal precision falling back to standard CUDA cores, redundant memory allocation (trashing), and unoptimized kernel launch overhead.

---

## 1. Data Stalls & Pipeline Synchronization

### Location
- [src/poregen/training/data.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/training/data.py#L18-L23) (lines 18-23)
- [src/poregen/metrics/latent.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/metrics/latent.py#L31-L35) (lines 31-35)

### Issue Overview
1. **Disabled PCIe Pinning**: In the DataLoader configurations, parsing `null` values from YAML configs results in `pin_memory` evaluating as `None`, which `bool()` silently converts to `False`. This disables page-locked host memory and renders `non_blocking=True` commands useless, forcing rigid CPU memory paging blocks.
2. **Synchronous Host-Device Blocks**: Within the main training step, the metric evaluator pulls intermediate calculation data using `.cpu()`. Because this is executed in the inner training loop, it completely stalls the pipeline as the CPU waits for the GPU's command queue to finish.

### Required Code Fixes

**Missing Pin Memory Evaluation:**
```diff
# src/poregen/training/data.py:L18-L23
     kwargs: dict[str, Any] = {
         "batch_size": int(data_cfg["batch_size"]),
         "num_workers": num_workers,
-        "pin_memory": bool(data_cfg.get("pin_memory", True)),
+        "pin_memory": data_cfg.get("pin_memory") is not False,
         "worker_init_fn": zarr_worker_init_fn if num_workers > 0 else None,
     }
```

**Synchronous Pipeline Stalls:**
```diff
# src/poregen/metrics/latent.py:L31-L35
     return {
         "count": int(mu_flat.shape[1]),
-        "sum": mu_flat.sum(dim=1).cpu(),
-        "sum_sq": mu_flat.square().sum(dim=1).cpu(),
+        "sum": mu_flat.sum(dim=1),
+        "sum_sq": mu_flat.square().sum(dim=1),
     }
```

---

## 2. Precision & Hardware Utilization (Tensor Cores)

### Location
- [src/poregen/training/engine.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/training/engine.py#L188-L198) (lines 188-198)
- [src/poregen/training/engine.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/training/engine.py#L229-L245) (lines 229-245)

### Issue Overview
The GB10 relies on specialized FP8 and FP16/BF16 Tensor Cores. Current Adversarial loss models forcefully strip these benefits by casting to `.float()` (FP32) and running outside of PyTorch's Automatic Mixed Precision (`torch.autocast`) boundaries, which kicks execution out to standard, slower CUDA cores.

### Required Code Fixes

**Generator Adversarial Operations:**
```diff
# src/poregen/training/engine.py:L188-L198
     if discriminator is not None and disc_optimizer is not None and disc_weight > 0.0:
-        # Cast from AMP dtype (fp16/bf16) to float32 — D runs in float32
-        _fake_slices = extract_multiplane_slices(output.xct_logits).float()   
-        _real_slices = extract_multiplane_slices(xct).float()                  
-
-        d_fake_gen = discriminator(_fake_slices)
-        _gen_adv_loss = lsgan_gen_loss(d_fake_gen)
+        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
+            _fake_slices = extract_multiplane_slices(output.xct_logits)   
+            _real_slices = extract_multiplane_slices(xct)                  
+
+            d_fake_gen = discriminator(_fake_slices)
+            _gen_adv_loss = lsgan_gen_loss(d_fake_gen)

         losses["total"] = losses["total"] + disc_weight * _gen_adv_loss
```

**Discriminator Fallbacks:**
```diff
# src/poregen/training/engine.py:L229-L245
     if (
         discriminator is not None
         and disc_optimizer is not None
         and _fake_slices is not None
         and _gen_adv_loss is not None
     ):
         disc_optimizer.zero_grad(set_to_none=True)
 
-        d_real = discriminator(_real_slices)                       
-        d_fake = discriminator(_fake_slices.detach())              
-
-        disc_loss = lsgan_disc_loss(d_real, d_fake)
-        disc_loss.backward()
-        disc_optimizer.step()
+        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
+            d_real = discriminator(_real_slices)                       
+            d_fake = discriminator(_fake_slices.detach())              
+            disc_loss = lsgan_disc_loss(d_real, d_fake)
+        
+        scaler.scale(disc_loss).backward()
+        scaler.step(disc_optimizer)
```

---

## 3. Memory Management (VRAM Trashing)

### Location
- [src/poregen/training/engine.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/training/engine.py#L268-L281) (lines 268-281)
- [src/poregen/training/engine.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/training/engine.py#L318-L327) (lines 318-327)

### Issue Overview
The evaluation loop (`_run_eval`) unnecessarily fragments the 128GB Unified Memory by forcing twin memory allocations on identical batched tensors. After `eval_step` allocates the incoming CPU batches onto the GPU, the parent loop blindly repeats the `.to(device)` mapping on the initial CPU copies, triggering double allocations and GC thrashing. 

### Required Code Fixes

**Returning Cached Device Tensors:**
```diff
# src/poregen/training/engine.py:L268-L281
 @torch.no_grad()
 def eval_step(
     model: nn.Module,
     batch: dict[str, torch.Tensor],
     loss_fn: Callable[..., dict[str, Any]],
     step: int,
     device: torch.device,
     autocast_dtype: torch.dtype = torch.float16,
-) -> tuple[dict[str, Any], VAEOutput]:
+) -> tuple[dict[str, Any], VAEOutput, dict[str, torch.Tensor]]:
     model.eval()
     xct  = batch["xct"].to(device, non_blocking=True)
     mask = batch["mask"].to(device, non_blocking=True)
     batch_dev = {**batch, "xct": xct, "mask": mask}
 
     with torch.autocast(device_type=device.type, dtype=autocast_dtype):
         output: VAEOutput = model(xct, mask)
         losses = loss_fn(output, batch_dev, step)
 
-    return {k: _to_scalar(v) for k, v in losses.items()}, output
+    return {k: _to_scalar(v) for k, v in losses.items()}, output, batch_dev
```

**Evading Redundant DMA Re-allocations:**
```diff
# src/poregen/training/engine.py:L318-L327
     for batch_idx in tqdm(range(n_batches), desc=desc, leave=False, unit="batch"):
         batch = next(data_iter)
-        losses, output = eval_step(
+        losses, output, batch_dev = eval_step(
             model, batch, loss_fn, step, device, autocast_dtype
         )
         _accumulate(loss_acc, losses)
 
-        mask_dev = batch["mask"].to(device, non_blocking=True)
-        xct_dev  = batch["xct"].to(device,  non_blocking=True)
+        mask_dev = batch_dev["mask"]
+        xct_dev  = batch_dev["xct"]
```

---

## 4. Execution Efficiency (Kernel Launch Overhead)

### Location
- [src/poregen/losses/mask.py](file:///home/jorgecabrejas/Dev/GenAI/src/poregen/losses/mask.py#L25-L30) (lines 25-30)

### Issue Overview
Calculations comprising many continuous element-wise operations suffer heavily from Kernel Launch Overhead (where every operand triggers an independent CUDA command logic flow). During the focal loss module, each operator is continually triggering expensive memory swaps back into VRAM rather than fusing the algebraic computation. 

### Required Code Fixes

**Fusing Element-Wise Kernels via Triton:**
```diff
# src/poregen/losses/mask.py:L25-L30
+@torch.compile(mode="reduce-overhead")
 def focal_loss(
     logits: torch.Tensor,
     target: torch.Tensor,
     gamma: float = 2.0,
     alpha: float = 0.25,
 ) -> torch.Tensor:
```

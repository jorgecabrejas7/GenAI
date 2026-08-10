"""THROWAWAY diagnostic (read-only): confirming experiment.

Per-t loss measurement (investigate_ldm03_per_t_loss.py) showed:
  - RAW (non-EMA) weights: eps MSE tracks the theoretical Gaussian-data optimum
    almost exactly at EVERY t in [0,999]; eps_pred_std correctly tracks noise_std
    up to ~1.0 at high t.
  - EMA weights: eps_pred_std plateaus at ~0.55-0.60 for t>=600 instead of tracking
    to 1.0 -- severely miscalibrated (under-scaled) eps predictions in the high-noise
    regime, with MSE 30-300x worse than raw at t>=700.

This script runs the SAME DDIM generation as diagnose_offmanifold.py (200 steps,
unguided s_por=s_nb=1) but with RAW (non-EMA) weights instead of EMA, on the same
val conditioning, and reports normalized-space std + clamp saturation -- the exact
metrics that were 7.1x / 24.8% with EMA weights. If raw weights fix it (std ~1,
clamp ~0), that confirms EMA staleness is the generation-time root cause.

No source modified; ad-hoc use of DDIMSampler with a manually-loaded raw model.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

LDM_CKPT = REPO / "runs/ldm/ldm03-run-0001-20260622-093226-z16-c128-s32-bs128-lr1e-04/checkpoints/best.ckpt"
SAMPLED_LATENTS_ROOT = REPO / "data/split_v2/latents_s64_sampled"
SAMPLED_STATS = SAMPLED_LATENTS_ROOT / "latent_scale_stats.json"


def load_raw_model(checkpoint, device):
    import yaml
    from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser

    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    model_cfg = UNet3DConfig.from_cfg(cfg)
    model = UNet3DDenoiser(model_cfg).to(device)

    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    msd = raw["model"]
    if any(k.startswith("_orig_mod.") for k in msd):
        msd = {k.removeprefix("_orig_mod."): v for k, v in msd.items()}
    model.load_state_dict(msd)
    model.eval()
    return model, cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16

    from poregen.diffusion.noise_schedule import DDPMSchedule
    from poregen.diffusion.sampler import DDIMSampler
    from poregen.diffusion.sampled_latent_dataset import SampledLatentPatchDataset

    latent_std = float(json.loads(SAMPLED_STATS.read_text())["std"])

    model, cfg = load_raw_model(LDM_CKPT, device)
    sched_cfg = cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(T=int(sched_cfg.get("T", 1000)), s=float(sched_cfg.get("s", 0.008)),
                             device=device)

    ds = SampledLatentPatchDataset(str(SAMPLED_LATENTS_ROOT), "val", patch_stride=32)
    rng = np.random.default_rng(0)
    n_patches = 256
    sel = np.sort(rng.choice(len(ds), size=min(n_patches, len(ds)), replace=False))
    items = [ds[int(i)] for i in sel]

    nb_latents = torch.stack([it["nb_latents"] for it in items]).to(device)
    nb_avail = torch.stack([it["nb_avail"] for it in items]).to(device)
    pos_frac = torch.stack([it["pos_frac"] for it in items]).to(device)
    global_por = torch.stack([it["global_por"] for it in items]).squeeze(1).to(device)
    local_por = torch.stack([it["local_por"] for it in items]).squeeze(1).to(device)

    sampler = DDIMSampler(model, schedule, device, n_steps=200, s_por=1.0, s_nb=1.0)

    outs = []
    gen_batch = 128
    N = nb_latents.shape[0]
    with torch.no_grad():
        for s in range(0, N, gen_batch):
            e = min(s + gen_batch, N)
            z = sampler.sample_batch(
                nb_latents[s:e], nb_avail[s:e], pos_frac[s:e], global_por[s:e], local_por[s:e],
                autocast_dtype=autocast_dtype,
            )
            outs.append(z.cpu())
            print(f"  generated {e}/{N}")
    z_gen_norm = torch.cat(outs)

    gen_std = float(z_gen_norm.std())
    gen_clamp_frac = float((z_gen_norm.abs() > 9.0).float().mean())
    print()
    print(f"RAW-weights DDIM generation (200 steps, unguided): "
          f"normalized-space std = {gen_std:.4f}  (target ~1.0)  "
          f"clamp-saturation(|z|>9) = {100*gen_clamp_frac:.3f}%")
    print(f"[for reference: EMA-weights run in eval_results/phase0_offmanifold gave "
          f"std=7.115  clamp_frac=24.8%]")

    # quick porosity sanity via the VAE mask head, reusing encode_latents' _load_vae
    from encode_latents import _load_vae
    VAE_EXPERIMENT = "r05/base"
    VAE_CKPT = REPO / "runs/vae/r05-run-0001-20260420-142643-archv2-conv_noattn_dualbranch-z16-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone/best.ckpt"
    vae = _load_vae(VAE_EXPERIMENT, str(VAE_CKPT), device)
    vae.requires_grad_(False)

    z_decode = (z_gen_norm * latent_std).to(device)
    preds = []
    with torch.no_grad():
        for s in range(0, N, 64):
            e = min(s + 64, N)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                dec = vae.decoder(z_decode[s:e])
                mask_logits = vae.mask_head(dec)
            p = torch.sigmoid(mask_logits.float())
            preds.append(p.mean(dim=(1, 2, 3, 4)).cpu())
    pred_por = torch.cat(preds)
    intended_por = torch.stack([it["local_por"] for it in items]).squeeze(1)
    mae = (pred_por - intended_por).abs().mean().item()
    print(f"RAW-weights decoded porosity: mean={pred_por.mean().item():.4f} "
          f"std={pred_por.std().item():.4f}  MAE-vs-intended={mae:.4f}")
    print(f"[for reference: EMA-weights run gave porosity mean=0.4593 MAE-vs-GT=0.4365]")


if __name__ == "__main__":
    main()

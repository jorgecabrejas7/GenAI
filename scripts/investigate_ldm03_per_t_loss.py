"""THROWAWAY diagnostic (read-only): per-timestep eps-MSE loss for the real ldm03
checkpoint (both EMA and raw weights) on real held-out val latents, compared against
the theoretical optimal-Gaussian-baseline loss E[abar_t] per bin.

Motivation: training loss plateaus at ~0.475-0.49 (uniform-random-t average) for the
ENTIRE run (step ~1000 through 31499) -- suspiciously close to the theoretical floor
for a model that has learned NOTHING beyond the trivial data-independent Gaussian
predictor eps*(x_t) = sqrt(1-abar_t)*x_t, whose t-averaged loss is E_t[abar_t]=0.4956.

The round-trip test in eval_results/phase0_offmanifold showed DDIM-reversal is stable
for t<=400 but diverges for t in {700,900}. If the model is at-or-below the trivial
floor in AGGREGATE but specifically worse than trivial at high t (where the trivial
floor itself is near 0, so any real error is a large *relative* miss even if it barely
moves the aggregate average), that would explain: (a) misleadingly OK aggregate/val
loss used for early-stopping & best-ckpt selection, (b) stable low-t round trip,
(c) catastrophic high-t divergence in full generation (which starts at t=T-1).

No source is modified. Uses real DGX GPU + real checkpoint + real val split.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

LDM_CKPT = REPO / "runs/ldm/ldm03-run-0001-20260622-093226-z16-c128-s32-bs128-lr1e-04/checkpoints/best.ckpt"
LATENTS_ROOT = REPO / "data/split_v2/latents_s64_sampled"


def load_both_weight_sets(checkpoint, device):
    """Load raw (non-EMA) AND EMA weights into two separate model instances."""
    import yaml
    from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
    from poregen.training.checkpoint import load_checkpoint

    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    model_cfg = UNet3DConfig.from_cfg(cfg)

    raw_ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    model_raw = UNet3DDenoiser(model_cfg).to(device)
    msd = raw_ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in msd):
        msd = {k.removeprefix("_orig_mod."): v for k, v in msd.items()}
    model_raw.load_state_dict(msd)
    model_raw.eval()

    model_ema = UNet3DDenoiser(model_cfg).to(device)
    esd = raw_ckpt["ema"]
    if any(k.startswith("_orig_mod.") for k in esd):
        esd = {k.removeprefix("_orig_mod."): v for k, v in esd.items()}
    model_ema.load_state_dict({k: v.to(device) for k, v in esd.items()})
    model_ema.eval()

    return model_raw, model_ema, cfg


@torch.no_grad()
def per_t_loss(model, ds, schedule, device, autocast_dtype, t_bins, n_patches=512, batch=128, seed=0):
    """For each t bin (a specific t value), sample n_patches items, add noise at
    exactly that t, run eps_pred, compute MSE(eps_pred, noise), and eps_pred/noise
    std for sanity."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_patches, len(ds)), replace=False)
    items = [ds[int(i)] for i in idx]

    z = torch.stack([it["z"] for it in items]).to(device)             # (N,C,16,16,16) normalized
    nb_latents = torch.stack([it["nb_latents"] for it in items]).to(device)
    nb_avail = torch.stack([it["nb_avail"] for it in items]).to(device)
    pos_frac = torch.stack([it["pos_frac"] for it in items]).to(device)
    global_por = torch.stack([it["global_por"] for it in items]).squeeze(1).to(device)
    local_por = torch.stack([it["local_por"] for it in items]).squeeze(1).to(device)

    N = z.shape[0]
    results = {}
    for t_val in t_bins:
        losses, eps_stds, noise_stds, biases = [], [], [], []
        for s in range(0, N, batch):
            e = min(s + batch, N)
            zb = z[s:e]
            t = torch.full((zb.shape[0],), t_val, dtype=torch.long, device=device)
            torch.manual_seed(seed + t_val)
            noise = torch.randn_like(zb)
            z_t = schedule.q_sample(zb, t, noise)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                eps_pred = model(z_t, t, nb_latents[s:e], nb_avail[s:e], pos_frac[s:e],
                                 global_por[s:e], local_por[s:e])
            eps_pred = eps_pred.float()
            loss = F.mse_loss(eps_pred, noise).item()
            losses.append(loss)
            eps_stds.append(eps_pred.std().item())
            noise_stds.append(noise.std().item())
            biases.append((eps_pred - noise).mean().item())
        results[t_val] = {
            "mse": float(np.mean(losses)),
            "eps_pred_std": float(np.mean(eps_stds)),
            "noise_std": float(np.mean(noise_stds)),
            "bias": float(np.mean(biases)),
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.bfloat16

    from poregen.diffusion.noise_schedule import DDPMSchedule
    from poregen.diffusion.sampled_latent_dataset import SampledLatentPatchDataset

    model_raw, model_ema, cfg = load_both_weight_sets(LDM_CKPT, device)
    sched_cfg = cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(T=int(sched_cfg.get("T", 1000)), s=float(sched_cfg.get("s", 0.008)),
                             device=device)

    ds = SampledLatentPatchDataset(str(LATENTS_ROOT), "val", patch_stride=32)

    t_bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 975, 990, 999]
    abar = schedule.alphas_cumprod.cpu().numpy()

    print(f"{'t':>5} {'abar_t':>8} {'opt_loss':>9} | "
          f"{'RAW mse':>9} {'RAW eps_std':>11} {'RAW bias':>9} | "
          f"{'EMA mse':>9} {'EMA eps_std':>11} {'EMA bias':>9}")

    res_raw = per_t_loss(model_raw, ds, schedule, device, autocast_dtype, t_bins)
    res_ema = per_t_loss(model_ema, ds, schedule, device, autocast_dtype, t_bins)

    for t in t_bins:
        a = float(abar[t])
        opt = a  # theoretical optimal MSE at this t for N(0,I) data
        rr = res_raw[t]
        re = res_ema[t]
        print(f"{t:5d} {a:8.5f} {opt:9.5f} | "
              f"{rr['mse']:9.4f} {rr['eps_pred_std']:11.4f} {rr['bias']:9.4f} | "
              f"{re['mse']:9.4f} {re['eps_pred_std']:11.4f} {re['bias']:9.4f}")

    # overall aggregate (uniform sample of t like training) for cross-check against
    # the logged val loss (~0.475-0.49)
    print()
    for tag, model in (("RAW", model_raw), ("EMA", model_ema)):
        rng = np.random.default_rng(123)
        idx = rng.choice(len(ds), size=512, replace=False)
        items = [ds[int(i)] for i in idx]
        z = torch.stack([it["z"] for it in items]).to(device)
        nb_latents = torch.stack([it["nb_latents"] for it in items]).to(device)
        nb_avail = torch.stack([it["nb_avail"] for it in items]).to(device)
        pos_frac = torch.stack([it["pos_frac"] for it in items]).to(device)
        global_por = torch.stack([it["global_por"] for it in items]).squeeze(1).to(device)
        local_por = torch.stack([it["local_por"] for it in items]).squeeze(1).to(device)
        losses = []
        with torch.no_grad():
            for s in range(0, 512, 128):
                e = min(s + 128, 512)
                t = torch.randint(0, schedule.T, (e - s,), device=device)
                noise = torch.randn_like(z[s:e])
                z_t = schedule.q_sample(z[s:e], t, noise)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    eps_pred = model(z_t, t, nb_latents[s:e], nb_avail[s:e], pos_frac[s:e],
                                     global_por[s:e], local_por[s:e])
                losses.append(F.mse_loss(eps_pred.float(), noise).item())
        print(f"{tag} uniform-random-t aggregate MSE (matches training/val loss metric): {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()

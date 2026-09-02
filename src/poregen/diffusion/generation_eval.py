"""In-training generation diagnostics for the LDM (ldm04+).

Two eval blocks, computed on latents freshly sampled with DDIM + EMA weights:

1. Off-manifold latent diagnostics — the LDM trains in *normalised* latent
   space, where the real train distribution has per-channel mean 0 and std 1.
   We report the generated latents' per-channel mean (target 0) and the ratio
   of their per-channel std to the real std (target 1), plus the fraction of
   x0-prediction elements that hit the ±10 clamp during DDIM sampling.

2. Decode-based porosity eval — generated latents are denormalised and pushed
   through the frozen VAE decoder; decoded-mask porosities are compared to the
   real val-split porosity distribution (mean/std, Wasserstein-1) and the
   fraction of degenerate masks is reported.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from poregen.diffusion.conditioning import NB_UNKNOWN
from poregen.diffusion.sampler import DDIMSampler

logger = logging.getLogger(__name__)

# Degenerate decoded mask: essentially no pores, or majority-pore (the real
# porosity distribution tops out around 0.107).
_DEGENERATE_LO = 1e-4
_DEGENERATE_HI = 0.5


@torch.no_grad()
def generation_eval(
    model: torch.nn.Module,
    schedule,
    vae: torch.nn.Module,
    *,
    latent_shape: tuple[int, int, int, int],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    val_phi: np.ndarray,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.bfloat16,
    n_samples: int = 64,
    ddim_steps: int = 50,
    batch_size: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Sample n_samples latents with DDIM, run both diagnostic blocks.

    *model* must already carry the weights to evaluate (e.g. EMA weights
    swapped in by the caller).  Porosity conditioning values are drawn from
    the real val phi distribution — a no-op for unconditional models.

    Returns a flat metric dict (keys become ``gen/<key>`` in TensorBoard).
    """
    model.eval()
    C, d, h, w = latent_shape
    sampler = DDIMSampler(model, schedule, device, n_steps=ddim_steps)

    rng = np.random.default_rng(seed)
    phi_draw = rng.choice(val_phi, size=n_samples).astype(np.float32)

    chunks: list[torch.Tensor] = []
    sat_fracs: list[float] = []
    for i in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - i)
        nb_latents = torch.zeros(b, 6, C, d, h, w, device=device)
        nb_avail   = torch.full((b, 6), NB_UNKNOWN, dtype=torch.long, device=device)
        pos_frac   = torch.full((b, 3), 0.5, device=device)
        por        = torch.from_numpy(phi_draw[i : i + b]).to(device)
        z, sat = sampler.sample_batch(
            nb_latents, nb_avail, pos_frac, por, por,
            autocast_dtype=autocast_dtype, return_x0_saturation=True,
        )
        chunks.append(z)
        sat_fracs.append(sat)
    gen_z = torch.cat(chunks, dim=0)  # (N, C, d, h, w) normalised space

    metrics: dict[str, float] = {
        "x0_clamp_sat_frac": float(np.mean(sat_fracs)),
    }

    # ── off-manifold latent moments (normalised space: real mean=0, std=1) ──
    ch_mean = gen_z.mean(dim=(0, 2, 3, 4))
    ch_std  = gen_z.std(dim=(0, 2, 3, 4))
    for c in range(C):
        metrics[f"ch{c}_mean"]      = ch_mean[c].item()
        metrics[f"ch{c}_std_ratio"] = ch_std[c].item()
    metrics["mean_abs_max"]  = ch_mean.abs().max().item()
    metrics["std_ratio_avg"] = ch_std.mean().item()

    # ── decode-based porosity eval ──────────────────────────────────────────
    mean_dev = latent_mean.to(device)
    std_dev  = latent_std.to(device)
    vae.eval()
    porosities: list[torch.Tensor] = []
    degenerate = 0
    for i in range(0, n_samples, batch_size):
        z_denorm = gen_z[i : i + batch_size] * std_dev + mean_dev
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            dec         = vae.decoder(z_denorm)
            mask_logits = vae.mask_head(dec)
        mask = (mask_logits.float() > 0.0)                      # sigmoid(x) > 0.5
        por  = mask.float().mean(dim=(1, 2, 3, 4))              # (b,)
        degenerate += int(((por < _DEGENERATE_LO) | (por > _DEGENERATE_HI)).sum().item())
        porosities.append(por)
    por_all = torch.cat(porosities).cpu().numpy()

    metrics["por_mean"]        = float(por_all.mean())
    metrics["por_std"]         = float(por_all.std())
    metrics["real_por_mean"]   = float(val_phi.mean())
    metrics["real_por_std"]    = float(val_phi.std())
    metrics["degenerate_frac"] = degenerate / max(len(por_all), 1)

    from scipy.stats import wasserstein_distance
    metrics["por_w1"] = float(wasserstein_distance(por_all, val_phi))

    logger.info(
        "generation_eval: n=%d  std_ratio_avg=%.3f  mean_abs_max=%.3f  "
        "x0_sat=%.4f  por=%.4f±%.4f (real %.4f±%.4f)  W1=%.5f  degen=%.3f",
        n_samples, metrics["std_ratio_avg"], metrics["mean_abs_max"],
        metrics["x0_clamp_sat_frac"], metrics["por_mean"], metrics["por_std"],
        metrics["real_por_mean"], metrics["real_por_std"],
        metrics["por_w1"], metrics["degenerate_frac"],
    )
    return metrics

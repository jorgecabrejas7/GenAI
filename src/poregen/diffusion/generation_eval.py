"""In-training generation diagnostics for the LDM (ldm06).

Two eval blocks, computed on latents freshly sampled with DDIM + EMA weights:

1. Off-manifold latent diagnostics — the LDM trains in *normalised* latent
   space, where the real train distribution has per-channel mean 0 and std 1.
   We report the generated latents' per-channel mean (target 0) and the ratio
   of their per-channel std to the real std (target 1), plus the fraction of
   x0-prediction elements that hit the +/-10 clamp during DDIM sampling.

2. Decode-based porosity eval — generated latents are denormalised and pushed
   through the frozen 3-class VAE decoder; the decoded label's pore fraction is
   compared to the real val-split porosity distribution (mean/std,
   Wasserstein-1) and the fraction of degenerate volumes is reported.

Conditioning is drawn from real validation rows (cond_por / cond_depth /
cond_dist6 / cond_orient / cond_material exactly as the model saw them in
training), with every neighbour set to UNKNOWN at ``nb_t = 0`` — the CFG
neighbour null, and the hardest context the sampler ever runs in.  Because each
sample carries a known requested porosity, the eval also reports
``por_cond_mae``, the direct conditional-adherence metric.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from poregen.diffusion.conditioning import NB_UNKNOWN, N_NEIGHBOURS
from poregen.diffusion.sampler import DDIMSampler
from poregen.models.vae.base import CLASS_PORE, decode_label

logger = logging.getLogger(__name__)

# Degenerate decoded label: essentially no pores, or majority-pore (the real
# porosity distribution tops out around 0.107).
_DEGENERATE_LO = 1e-4
_DEGENERATE_HI = 0.5


@torch.no_grad()
def generation_eval(
    model: torch.nn.Module,
    schedule,
    vae: torch.nn.Module,
    *,
    val_dataset,
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
    swapped in by the caller).  Conditioning is taken verbatim from random
    validation rows of *val_dataset*, so it is always in-distribution.

    Returns a flat metric dict (keys become ``gen/<key>`` in TensorBoard).
    """
    model.eval()
    C, d, h, w = tuple(val_dataset.latent_shape)
    sampler = DDIMSampler(model, schedule, device, n_steps=ddim_steps)

    rng = np.random.default_rng(seed)
    row_idx = rng.integers(0, len(val_dataset), size=n_samples)
    items = [val_dataset[int(i)] for i in row_idx]
    phi_draw = np.asarray(
        [float(val_dataset.df["phi"].iloc[int(i)]) for i in row_idx], dtype=np.float64
    )

    def _scalars(key: str, sl: slice) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(it[key]).float().reshape(()) for it in items[sl]]
        ).to(device)

    def _tensors(key: str, sl: slice) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(it[key]).float() for it in items[sl]]
        ).to(device)

    chunks: list[torch.Tensor] = []
    sat_fracs: list[float] = []
    for i in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - i)
        sl = slice(i, i + b)
        nb_latents = torch.zeros(b, N_NEIGHBOURS, C, d, h, w, device=device)
        nb_avail   = torch.full((b, N_NEIGHBOURS), NB_UNKNOWN, dtype=torch.long,
                                device=device)
        nb_t       = torch.zeros((b, N_NEIGHBOURS), dtype=torch.long, device=device)
        z, sat = sampler.sample_batch(
            nb_latents, nb_avail, nb_t,
            _scalars("cond_por", sl), _scalars("cond_depth", sl),
            _tensors("cond_dist6", sl), _tensors("cond_orient", sl),
            _tensors("cond_material", sl),
            autocast_dtype=autocast_dtype, return_x0_saturation=True,
        )
        chunks.append(z)
        sat_fracs.append(sat)
    gen_z = torch.cat(chunks, dim=0)  # (N, C, d, h, w) normalised space

    metrics: dict[str, float] = {
        "x0_clamp_sat_frac": float(np.mean(sat_fracs)),
    }

    # -- off-manifold latent moments (normalised space: real mean=0, std=1) --
    ch_mean = gen_z.mean(dim=(0, 2, 3, 4))
    ch_std  = gen_z.std(dim=(0, 2, 3, 4))
    for c in range(C):
        metrics[f"ch{c}_mean"]      = ch_mean[c].item()
        metrics[f"ch{c}_std_ratio"] = ch_std[c].item()
    metrics["mean_abs_max"]  = ch_mean.abs().max().item()
    metrics["std_ratio_avg"] = ch_std.mean().item()

    # -- decode-based porosity eval ------------------------------------------
    mean_dev = latent_mean.to(device)
    std_dev  = latent_std.to(device)
    vae.eval()
    porosities: list[torch.Tensor] = []
    airs: list[torch.Tensor] = []
    degenerate = 0
    for i in range(0, n_samples, batch_size):
        z_denorm = gen_z[i : i + batch_size] * std_dev + mean_dev
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            dec = vae.decoder(z_denorm)
            class_logits = vae.class_head(dec)
        label = decode_label(class_logits.float())                # (b, D, H, W)
        por = (label == CLASS_PORE).float().mean(dim=(1, 2, 3))   # (b,)
        airs.append((label == 2).float().mean(dim=(1, 2, 3)))
        degenerate += int(((por < _DEGENERATE_LO) | (por > _DEGENERATE_HI)).sum().item())
        porosities.append(por)
    por_all = torch.cat(porosities).cpu().numpy()
    air_all = torch.cat(airs).cpu().numpy()

    # Conditional adherence: each sample carries a known requested porosity.
    metrics["por_cond_mae"]    = float(np.abs(por_all - phi_draw).mean())
    metrics["por_mean"]        = float(por_all.mean())
    metrics["por_std"]         = float(por_all.std())
    metrics["air_mean"]        = float(air_all.mean())
    metrics["real_por_mean"]   = float(val_phi.mean())
    metrics["real_por_std"]    = float(val_phi.std())
    metrics["degenerate_frac"] = degenerate / max(len(por_all), 1)

    from scipy.stats import wasserstein_distance
    metrics["por_w1"] = float(wasserstein_distance(por_all, val_phi))

    logger.info(
        "generation_eval: n=%d  std_ratio_avg=%.3f  mean_abs_max=%.3f  "
        "x0_sat=%.4f  por=%.4f+-%.4f (real %.4f+-%.4f)  air=%.4f  W1=%.5f  "
        "cond_mae=%.5f  degen=%.3f",
        n_samples, metrics["std_ratio_avg"], metrics["mean_abs_max"],
        metrics["x0_clamp_sat_frac"], metrics["por_mean"], metrics["por_std"],
        metrics["real_por_mean"], metrics["real_por_std"], metrics["air_mean"],
        metrics["por_w1"], metrics["por_cond_mae"], metrics["degenerate_frac"],
    )
    return metrics

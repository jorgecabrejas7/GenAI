"""THROWAWAY diagnostic (read-only): does the DDIM schedule itself blow up variance
under the analytically-optimal Gaussian denoiser?

If the target latent distribution is ~N(0, I) (which the eval shows: sampled-z std
~1.0, per-channel std ~1.0, eps-MSE loss ~0.48 ~= optimal 0.5), then the Bayes-optimal
eps predictor at training-index t is:

    eps*(x_t) = sqrt(1 - abar_t) * x_t          (for N(0,I) data)

where abar_t is the cumulative-alpha the MODEL associates with index t, i.e. the same
one q_sample uses = schedule.alphas_cumprod[t] (the [1:] array).

We plug that closed-form denoiser into the repo's OWN schedule.ddim_step and watch the
std of x evolve across a 200-step DDIM chain from pure noise.  No model, no checkpoint.

- If std stays ~1  -> the schedule math is fine; the blow-up must come from the model.
- If std blows up  -> the schedule/ddim_step indexing is the culprit (model-independent).

We also run a 'corrected' ddim_step that uses alphas_cumprod[t_prev] (consistent index)
for the previous-step cumulative alpha, to isolate the off-by-one.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.noise_schedule import DDPMSchedule


def optimal_eps(schedule, x, t):
    # eps*(x_t) = sqrt(1 - abar_t) * x_t  for N(0,I) data, using the model's index convention
    a = schedule.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    return (1.0 - a).sqrt() * x


def ddim_step_corrected(schedule, x_t, t, t_prev, eps_pred):
    """Same as schedule.ddim_step but uses alphas_cumprod[t_prev] (index-consistent with
    the training convention) instead of alphas_cumprod_prev[t_prev]. For t_prev==0 we fall
    back to abar_prev=1 (clean) to match the intended final-step behaviour."""
    at = schedule.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    # index-consistent previous cumulative alpha
    at_m1 = schedule.alphas_cumprod[t_prev].view(-1, 1, 1, 1, 1).clone()
    at_m1[t_prev == 0] = 1.0
    x0 = (x_t - (1.0 - at).sqrt() * eps_pred) / at.sqrt().clamp(min=1e-8)
    x0 = x0.clamp(-10.0, 10.0)
    return at_m1.sqrt() * x0 + (1.0 - at_m1).sqrt() * eps_pred


def run(schedule, n_steps, B=4096, C=16, D=4, step_fn=None, tag=""):
    """step_fn(schedule, x, t, t_prev, eps) -> x_prev. Default = schedule.ddim_step."""
    if step_fn is None:
        step_fn = lambda s, x, t, tp, e: s.ddim_step(x, t, tp, e)
    T = schedule.T
    ts = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long).flip(0).tolist()
    x = torch.randn(B, C, D, D, D)
    stds = []
    clampfracs = []
    for i, t_val in enumerate(ts[:-1]):
        t_prev_val = ts[i + 1]
        t = torch.full((B,), t_val, dtype=torch.long)
        t_prev = torch.full((B,), t_prev_val, dtype=torch.long)
        eps = optimal_eps(schedule, x, t)
        # measure x0 clamp saturation exactly as ddim_step would see it
        at = schedule.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        x0raw = (x - (1.0 - at).sqrt() * eps) / at.sqrt().clamp(min=1e-8)
        clampfracs.append(((x0raw.abs() > 9.0).float().mean().item()))
        x = step_fn(schedule, x, t, t_prev, eps)
        stds.append(x.std().item())
    print(f"\n[{tag}] n_steps={n_steps}")
    idxs = [0, 1, 2, 5, 10, 20, 50, 100, 150, len(stds) - 1]
    print("  step :  t   -> t_prev   std(x)     x0_clamp_frac")
    for k in idxs:
        if k < len(stds):
            print(f"  {k:4d} : {ts[k]:4d} -> {ts[k+1]:4d}    {stds[k]:8.4f}   {clampfracs[k]:.4f}")
    print(f"  FINAL std(x) = {stds[-1]:.4f}   (target ~1.0 for N(0,I) data)")
    return stds[-1]


def main():
    torch.manual_seed(0)
    schedule = DDPMSchedule(T=1000, s=0.008, device="cpu")

    # sanity on the schedule endpoints
    ac = schedule.alphas_cumprod
    acp = schedule.alphas_cumprod_prev
    print("schedule sanity:")
    print(f"  alphas_cumprod[0]      (abar_1)   = {ac[0].item():.6f}")
    print(f"  alphas_cumprod[-1]     (abar_T)   = {ac[-1].item():.8f}")
    print(f"  alphas_cumprod_prev[0] (abar_0)   = {acp[0].item():.6f}")
    print(f"  alphas_cumprod_prev[1] (abar_1)   = {acp[1].item():.6f}")
    print(f"  => alphas_cumprod_prev[t] == alphas_cumprod[t-1] ? "
          f"{torch.allclose(acp[1:], ac[:-1])}")

    for n_steps in (200, 50, 20, 1000):
        run(schedule, n_steps, step_fn=None, tag=f"repo ddim_step (as shipped)")
    print("\n" + "=" * 70)
    for n_steps in (200, 50, 20, 1000):
        run(schedule, n_steps, step_fn=ddim_step_corrected,
            tag=f"index-consistent ddim_step")


if __name__ == "__main__":
    main()

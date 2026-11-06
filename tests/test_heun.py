"""Heun sampler: known answer on Gaussian data, and the generator's call pattern."""
import math

import pytest
import torch

from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler, HeunSampler

from test_volume_generator_hybrid import P, LAT, VOX_MM, _ConstVAE, _SpyModel, theta_from_layup, VolumeGenerator


def _exact_gaussian(s: float, n_steps: int, order: int):
    """Data N(0, s^2): the posterior eps is closed-form and the probability-flow
    ODE maps x_T = xi to x_0 = xi * s (the marginal std at t is sqrt(a s^2 + 1 - a))."""
    sch = DDPMSchedule(T=1000, objective="eps")
    a_all = sch.alphas_cumprod

    def predict(x, t):
        a = a_all[t].view(-1, 1, 1, 1, 1)
        return (1 - a).sqrt() * x / (a * s * s + 1 - a)

    ts = DDIMSampler(torch.nn.Identity(), sch, torch.device("cpu"), n_steps=n_steps).timesteps
    g = torch.Generator().manual_seed(0)
    xi = torch.rand(64, 1, 2, 2, 2, generator=g) * 2 - 1   # |x0| <= s, clear of the x0 clamp
    # start at the ladder's top: x_T is the marginal at t = T-1, std sqrt(a s^2 + 1 - a)
    a_T = a_all[ts[0]]
    x_T = xi * math.sqrt(float(a_T) * s * s + 1 - float(a_T))
    x0 = sch.sample_ode(predict, x_T, ts, order=order)
    return float((x0 - xi * s).abs().mean())


def test_heun_beats_euler_on_gaussian_data():
    e_ddim10, e_heun10, e_ddim20 = (_exact_gaussian(3.0, 10, 1), _exact_gaussian(3.0, 10, 2),
                                     _exact_gaussian(3.0, 20, 1))
    assert e_heun10 < 0.5 * e_ddim10          # second order: a clear win at equal steps
    assert e_heun10 < e_ddim20                # and still ahead of Euler with twice the steps


def test_heun_converges_to_the_exact_solution():
    # The first and last ladder steps stay Euler, so the error is not monotone
    # in the step count; it is small from 20 steps on (float32 loop).
    assert _exact_gaussian(3.0, 20, 2) < 0.01
    assert _exact_gaussian(3.0, 80, 2) < 0.01


def test_generator_call_pattern_and_equal_result_for_a_constant_model():
    dev = torch.device("cpu")
    sch = DDPMSchedule(T=100, device=dev)
    theta = theta_from_layup(2 * P, [0, 45, 90], 19.6)
    out = {}
    for cls in (DDIMSampler, HeunSampler):
        model = _SpyModel(0.0)
        gen = VolumeGenerator(cls(model, sch, dev, n_steps=4), _ConstVAE(), dev, patch_size=P,
                              latent_size=LAT, latent_scale=1.0, voxel_size_mm=VOX_MM,
                              por_log_stats=(-3.0, 1.0), theta_deg=theta, chunk_tiles=(2, 2, 2),
                              chunk_overlap=0)
        xct, label, stats, *_ = gen.generate(tuple(2 * P * VOX_MM for _ in range(3)),
                                             target_porosity=0.02, seed=1)
        out[cls.solver] = (len(model.calls), xct, stats)
    n_ddim, n_heun = out["ddim"][0], out["heun"][0]
    # 4 ladder steps: Heun adds one evaluation on the 2 intermediate steps only
    assert n_heun == n_ddim * (4 + 2) // 4
    assert out["heun"][2]["solver"] == "heun" and out["heun"][2]["model_calls_per_step"] == 2
    # a constant eps makes both slopes equal, so the correction is exactly zero
    assert (out["heun"][1] == out["ddim"][1]).all()

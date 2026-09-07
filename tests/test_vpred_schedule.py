"""v-prediction and zero-terminal-SNR — the ldm06 objective.

The four things that can silently go wrong here, and the tests that pin them:

* a conversion that is *nearly* right — v -> x0 -> eps must reproduce the
  forward process exactly, or the model trains against one target and samples
  against another;
* a "zero" terminal SNR that is only small — ``sqrt(alpha_bar_T)`` must be
  EXACTLY 0 and nothing at t = 0 may move;
* the size of that rescale being misremembered.  This schedule takes alpha_bar
  straight from the cosine f, so its terminal value is already 6e-17 and the
  Lin rescale changes nothing measurable — the win is that the eps form's
  division by ~0 disappears, not that a signal leak was closed.  The test says
  so, so the next reader does not attribute an ldm06 result to the wrong
  cause;
* a CFG decomposition that stops telescoping once the network emits v instead
  of eps, which would make s_por = s_nb = 1 quietly non-neutral.
"""

from __future__ import annotations

import math

import pytest
import torch

from poregen.diffusion.conditioning import NB_EXISTS, N_DIST6, N_NEIGHBOURS
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler, rescale_guidance
from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser

T = 200
B, C, D = 4, 2, 4


def _eps_schedule(**kw) -> DDPMSchedule:
    return DDPMSchedule(T=T, **kw)


def _v_schedule(zero_terminal_snr: bool = True) -> DDPMSchedule:
    return DDPMSchedule(T=T, objective="v", zero_terminal_snr=zero_terminal_snr)


def _x0_noise_t(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(B, C, D, D, D, generator=g)
    noise = torch.randn(B, C, D, D, D, generator=g)
    t = torch.tensor([0, T // 3, 2 * T // 3, T - 1])
    return x0, noise, t


# ── objective plumbing ───────────────────────────────────────────────────────

class TestObjective:

    def test_unknown_objective_is_rejected(self):
        with pytest.raises(ValueError, match="objective"):
            DDPMSchedule(T=T, objective="x0")

    def test_zero_terminal_snr_refuses_the_eps_objective(self):
        """At alpha_bar_T = 0 the eps parameterisation divides by zero."""
        with pytest.raises(ValueError, match="zero_terminal_snr"):
            DDPMSchedule(T=T, objective="eps", zero_terminal_snr=True)

    def test_eps_objective_returns_the_noise_as_its_target(self):
        x0, noise, t = _x0_noise_t()
        assert torch.equal(_eps_schedule().training_target(x0, noise, t), noise)

    def test_v_target_matches_the_salimans_ho_definition(self):
        sch = _v_schedule(zero_terminal_snr=False)
        x0, noise, t = _x0_noise_t()
        a = sch.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = sch.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        assert torch.allclose(sch.training_target(x0, noise, t), a * noise - b * x0,
                              atol=1e-6)

    def test_forward_process_is_objective_independent(self):
        """q_sample is the same draw whichever target the network regresses."""
        x0, noise, t = _x0_noise_t()
        assert torch.allclose(
            _eps_schedule().q_sample(x0, t, noise),
            DDPMSchedule(T=T, objective="v").q_sample(x0, t, noise),
            atol=1e-6,
        )


# ── conversions round-trip ───────────────────────────────────────────────────

class TestConversions:

    @pytest.mark.parametrize("ztsnr", [False, True])
    def test_v_recovers_x0_and_eps_exactly(self, ztsnr):
        sch = _v_schedule(zero_terminal_snr=ztsnr)
        x0, noise, t = _x0_noise_t()
        x_t = sch.q_sample(x0, t, noise)
        v = sch.training_target(x0, noise, t)
        assert torch.allclose(sch.predict_x0(x_t, t, v), x0, atol=1e-4)
        assert torch.allclose(sch.predict_eps(x_t, t, v), noise, atol=1e-4)

    def test_eps_objective_recovers_x0_and_passes_eps_through(self):
        """Away from the terminal step, where sqrt(alpha_bar) is not ~0."""
        sch = _eps_schedule()
        x0, noise, t = _x0_noise_t()
        t = t.clamp(max=T - 2)
        x_t = sch.q_sample(x0, t, noise)
        assert torch.allclose(sch.predict_x0(x_t, t, noise), x0, atol=1e-3)
        assert torch.equal(sch.predict_eps(x_t, t, noise), noise)

    def test_the_two_objectives_describe_the_same_denoiser(self):
        """A perfect v-predictor and a perfect eps-predictor give one x0."""
        eps_sch, v_sch = _eps_schedule(), _v_schedule(zero_terminal_snr=False)
        x0, noise, t = _x0_noise_t()
        t = t.clamp(max=T - 2)
        x_t = eps_sch.q_sample(x0, t, noise)
        v = v_sch.training_target(x0, noise, t)
        assert torch.allclose(
            eps_sch.predict_x0(x_t, t, noise), v_sch.predict_x0(x_t, t, v), atol=1e-3
        )

    def test_the_v_form_is_the_only_usable_one_at_the_terminal_step(self):
        """The motivation for the v objective, stated as a measurement.

        The eps form recovers x0 by dividing by sqrt(alpha_bar_T), which this
        cosine schedule already puts at 6.1e-17.  The 1e-8 guard clamp turns
        that into an x0 of order 1e8, which the +/-10 clamp saturates — so the
        first DDIM step of every ldm06 chain started from a clamp artefact.
        The v form needs no division and stays order 1.
        """
        x_t = torch.randn(B, C, D, D, D)
        out = torch.randn_like(x_t)
        t = torch.full((B,), T - 1, dtype=torch.long)

        eps_x0 = _eps_schedule().predict_x0(x_t, t, out)
        v_x0 = _v_schedule().predict_x0(x_t, t, out)

        assert float(eps_x0.abs().max()) > 1e6      # saturates the +/-10 clamp
        assert torch.isfinite(v_x0).all()
        assert float(v_x0.abs().max()) < 100.0


# ── zero terminal SNR ────────────────────────────────────────────────────────

class TestZeroTerminalSNR:

    def test_terminal_alpha_bar_is_exactly_zero(self):
        sch = _v_schedule()
        assert float(sch.alphas_cumprod[-1]) == 0.0
        assert float(sch.sqrt_alphas_cumprod[-1]) == 0.0
        assert float(sch.sqrt_one_minus_alphas_cumprod[-1]) == 1.0

    def test_the_rescale_is_a_near_no_op_on_this_cosine_schedule(self):
        """Pin the honest magnitude — this repo has no terminal signal leak.

        Lin et al.'s 0.068 terminal sqrt(alpha_bar) comes from deriving
        alpha_bar as cumprod(1 - clamp(beta)).  DDPMSchedule takes alpha_bar
        straight from the cosine f, whose last entry is already zero to float
        precision, so the rescale only removes that 6e-17 — it is the
        exactness that matters here, not a signal leak.  A future change that
        makes this assertion fail has changed the schedule itself.
        """
        plain, rescaled = _eps_schedule(), _v_schedule()
        assert float(plain.sqrt_alphas_cumprod[-1]) < 1e-10
        delta = (plain.sqrt_alphas_cumprod - rescaled.sqrt_alphas_cumprod).abs().max()
        assert float(delta) < 1e-10

    def test_the_first_step_is_left_alone(self):
        """Only the tail moves — alpha_bar_0 = 1 and t = 0 is untouched."""
        plain, rescaled = _eps_schedule(), _v_schedule()
        assert float(rescaled.alphas_cumprod_prev[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(rescaled.alphas_cumprod[0]) == pytest.approx(
            float(plain.alphas_cumprod[0]), abs=1e-9
        )

    def test_the_schedule_stays_monotone_and_bounded(self):
        sch = _v_schedule()
        ac = sch.alphas_cumprod
        assert bool((ac[1:] <= ac[:-1] + 1e-7).all())
        assert float(ac.min()) >= 0.0 and float(ac.max()) <= 1.0
        assert bool((sch.betas >= 0.0).all()) and bool((sch.betas <= 1.0).all())
        assert float(sch.betas[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_q_sample_at_the_terminal_step_is_pure_noise(self):
        sch = _v_schedule()
        x0, noise, _ = _x0_noise_t()
        t = torch.full((B,), T - 1, dtype=torch.long)
        assert torch.allclose(sch.q_sample(x0, t, noise), noise, atol=1e-6)

    def test_from_cfg_reads_the_objective_off_the_config(self):
        sch = DDPMSchedule.from_cfg(
            {"noise_schedule": {"T": T, "objective": "v", "zero_terminal_snr": True}}
        )
        assert sch.objective == "v" and sch.zero_terminal_snr and sch.T == T
        plain = DDPMSchedule.from_cfg({})
        assert plain.objective == "eps" and not plain.zero_terminal_snr


# ── DDIM reaches x0 ──────────────────────────────────────────────────────────

class _LinearVModel(torch.nn.Module):
    """A denoiser that is exactly right about one fixed latent ``x0``.

    Given ``x_t`` and ``t`` it reconstructs the noise the forward process must
    have used to reach ``x_t`` from ``x0``, and returns the matching v.  It is
    the analytic optimum, so a correct reverse process must land on ``x0``.
    """

    def __init__(self, schedule: DDPMSchedule, x0: torch.Tensor) -> None:
        super().__init__()
        self.schedule = schedule
        self.x0 = x0

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        noise = (x_t - a * self.x0) / b.clamp(min=1e-8)
        return a * noise - b * self.x0


@pytest.mark.parametrize("ztsnr", [False, True])
def test_ddim_with_the_v_objective_lands_on_x0(ztsnr):
    sch = _v_schedule(zero_terminal_snr=ztsnr)
    torch.manual_seed(0)
    x0 = torch.randn(1, C, D, D, D)
    model = _LinearVModel(sch, x0)

    n_steps = 25
    grid = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long).flip(0).tolist()
    assert grid[0] == T - 1 and grid[-1] == 0

    x = torch.randn(1, C, D, D, D)
    for i, t_val in enumerate(grid[:-1]):
        t = torch.full((1,), t_val, dtype=torch.long)
        t_prev = torch.full((1,), grid[i + 1], dtype=torch.long)
        x = sch.ddim_step(x, t, t_prev, model(x, t))
    assert torch.allclose(x, x0, atol=1e-3)


def test_the_eps_objective_lands_on_x0_too():
    """Same reverse process, other parameterisation — no regression."""
    sch = _eps_schedule()
    torch.manual_seed(1)
    x0 = torch.randn(1, C, D, D, D)

    def eps_model(x_t, t):
        a = sch.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = sch.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return (x_t - a * x0) / b.clamp(min=1e-8)

    n_steps = 25
    grid = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long).flip(0).tolist()
    x = torch.randn(1, C, D, D, D)
    for i, t_val in enumerate(grid[:-1]):
        t = torch.full((1,), t_val, dtype=torch.long)
        t_prev = torch.full((1,), grid[i + 1], dtype=torch.long)
        x = sch.ddim_step(x, t, t_prev, eps_model(x, t))
    assert torch.allclose(x, x0, atol=1e-3)


# ── the sampler under a v schedule ───────────────────────────────────────────

@pytest.fixture()
def denoiser() -> UNet3DDenoiser:
    cfg = UNet3DConfig(z_channels=C, base_channels=8, channel_mult=(1, 2),
                       n_res_blocks=1, cond_embed_dim=16, nb_avail_embed_dim=4,
                       nb_t_embed_dim=4, n_attn_heads=2)
    m = UNet3DDenoiser(cfg)
    torch.manual_seed(0)
    torch.nn.init.normal_(m.out_conv.weight, std=0.1)
    for mod in m.modules():
        if type(mod).__name__ == "AdaGN":
            torch.nn.init.normal_(mod.proj.weight, std=0.1)
    return m.eval()


def _cond_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(2, C, D, D, D, generator=g),
        t=torch.zeros(2, dtype=torch.long),
        nb_latents=torch.randn(2, N_NEIGHBOURS, C, D, D, D, generator=g),
        nb_avail=torch.full((2, N_NEIGHBOURS), NB_EXISTS),
        nb_t=torch.randint(0, 10, (2, N_NEIGHBOURS), generator=g),
        cond_por=torch.randn(2, generator=g),
        cond_depth=torch.rand(2, generator=g),
        cond_dist6=torch.rand(2, N_DIST6, generator=g),
        cond_orient=torch.randn(2, 2, D, D, D, generator=g),
        cond_material=torch.rand(2, 1, D, D, D, generator=g),
    )


def _v_sampler(model, **kw):
    return DDIMSampler(model, _v_schedule(), torch.device("cpu"), n_steps=2, **kw)


def test_cfg_in_v_space_telescopes_at_scale_one(denoiser):
    """s_por = s_nb = 1 must be EXACTLY the full-conditional v prediction."""
    b = _cond_batch()
    with torch.no_grad():
        reference = denoiser(b["x"], b["t"], b["nb_latents"], b["nb_avail"], b["nb_t"],
                             b["cond_por"], b["cond_depth"], b["cond_dist6"],
                             b["cond_orient"], b["cond_material"])
    s = _v_sampler(denoiser)
    s.guided = True                            # force the 3-pass path
    with torch.no_grad():
        three_pass = s.predict_out(**b, autocast_dtype=torch.float32)
    assert torch.allclose(reference.float(), three_pass.float(), atol=1e-5)


def test_the_sampler_starts_at_the_terminal_timestep(denoiser):
    """Under zero terminal SNR that is the only index where alpha_bar = 0."""
    s = _v_sampler(denoiser)
    assert s.timesteps[0] == T - 1
    assert s.timesteps[-1] == 0
    assert float(s.schedule.alphas_cumprod[s.timesteps[0]]) == 0.0


def test_cfg_rescale_is_off_by_default_and_acts_when_asked(denoiser):
    b = _cond_batch()
    with torch.no_grad():
        plain = _v_sampler(denoiser, s_por=3.0).predict_out(**b, autocast_dtype=torch.float32)
        rescaled = _v_sampler(denoiser, s_por=3.0, cfg_rescale=0.7).predict_out(
            **b, autocast_dtype=torch.float32)
    assert _v_sampler(denoiser).cfg_rescale == 0.0
    assert not torch.allclose(plain, rescaled, atol=1e-6)


class TestRescaleGuidance:

    def test_phi_zero_is_the_identity(self):
        g = torch.randn(3, 2, 4, 4, 4)
        c = torch.randn(3, 2, 4, 4, 4)
        assert torch.allclose(rescale_guidance(g, c, 0.0), g, atol=1e-6)

    def test_phi_one_matches_the_conditional_standard_deviation(self):
        torch.manual_seed(0)
        g = torch.randn(3, 2, 4, 4, 4) * 5.0
        c = torch.randn(3, 2, 4, 4, 4)
        out = rescale_guidance(g, c, 1.0)
        dims = (1, 2, 3, 4)
        assert torch.allclose(out.std(dim=dims), c.std(dim=dims), rtol=1e-4)

    def test_it_shrinks_an_inflated_prediction(self):
        torch.manual_seed(0)
        c = torch.randn(3, 2, 4, 4, 4)
        g = c * 4.0
        dims = (1, 2, 3, 4)
        out = rescale_guidance(g, c, 0.7)
        assert bool((out.std(dim=dims) < g.std(dim=dims)).all())


def test_ldm06_base_resolves_to_a_v_schedule():
    """ldm06 trains on v from the start; ldm07 was folded into it.

    The objective is not a later rung because the defect it fixes is present
    from the first step: on this schedule the epsilon form of x0_hat divides by
    sqrt(alpha_bar_T) = 6.12e-17 at the terminal step.
    """
    from poregen.configuration import resolve_experiment

    cfg = resolve_experiment("ldm06/base").cfg
    assert cfg["noise_schedule"]["objective"] == "v"
    assert cfg["noise_schedule"]["zero_terminal_snr"] is True
    # Everything else is ldm06/base: same store, same denoiser, same budget.
    assert cfg["data"]["latents_root"] == (
        f"data/split_v3/latents_r08z{cfg['model']['z_channels']}")
    _z = cfg["model"]["z_channels"]
    assert UNet3DConfig.from_cfg(cfg).in_channels == _z + 2 + 1 + 6 * _z + 48 + 48
    assert cfg["training"]["total_steps"] == 130000

    sch = DDPMSchedule.from_cfg(cfg)
    assert sch.objective == "v"
    assert float(sch.sqrt_alphas_cumprod[-1]) == 0.0
    # The cosine offset survived the merge — the tail moved, not the shape.
    assert sch.s == pytest.approx(0.008)
    assert math.isclose(float(sch.alphas_cumprod_prev[0]), 1.0, abs_tol=1e-6)


def test_ldm06_eps_ablation_is_the_only_eps_config():
    """The eps variant exists to MEASURE the defect, so it must stay eps.

    zero_terminal_snr has to be false there: the schedule rejects the flag with
    the epsilon objective, because then the terminal division is by exactly
    zero rather than by 6.12e-17.
    """
    from poregen.configuration import resolve_experiment

    cfg = resolve_experiment("ldm06/eps").cfg
    assert cfg["noise_schedule"]["objective"] == "eps"
    assert cfg["noise_schedule"]["zero_terminal_snr"] is False
    # Everything else is base: the ablation isolates the objective alone.
    base = resolve_experiment("ldm06/base").cfg
    assert cfg["data"]["latents_root"] == base["data"]["latents_root"]
    assert cfg["training"]["total_steps"] == base["training"]["total_steps"]
    assert DDPMSchedule.from_cfg(cfg).objective == "eps"

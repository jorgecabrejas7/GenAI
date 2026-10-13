"""The Rombach-aligned first-stage loss switches (vault D68, r08/rombach-rf16).

Covers: the rombach_sum KL reduction value, the adaptive adversarial weight,
the paired LPIPS slice extraction, and that configs without the new keys keep
the historical loss exactly. None of these build the LPIPS network (that needs
the VGG16 ImageNet weights).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from poregen.losses.kl import kl_divergence, kl_rombach_sum
from poregen.losses.perceptual import build_perceptual, grey_to_lpips_input, paired_axis_slices
from poregen.losses.total import compute_total_loss
from poregen.models.discriminator import adaptive_disc_weight
from poregen.models.vae.base import VAEOutput


def _cfg(**loss_overrides):
    loss = {
        "xct_loss_type": "l1",
        "xct_weight": 1.0,
        "kl_free_bits": 0.0,
        "kl_warmup_steps": 0,
        "kl_max_beta": 1e-6,
    }
    loss.update(loss_overrides)
    return {"loss": loss}


def _xct_only_output(B=2, P=8, C=2, s=2, *, mu=None, logvar=None):
    return VAEOutput(
        xct_out=torch.rand(B, 1, P, P, P),
        mu=torch.randn(B, C, s, s, s) if mu is None else mu,
        logvar=torch.randn(B, C, s, s, s) * 0.1 if logvar is None else logvar,
        z=torch.zeros(B, C, s, s, s),
    )


# ── KL reduction ──────────────────────────────────────────────────────────────

def test_kl_rombach_sum_value_on_a_tiny_tensor():
    # mu = 1, logvar = 0: every element's KL is 0.5 * (1 + 1 - 0 - 1) = 0.5.
    # Per sample: 0.5 * C*d*h*w = 0.5 * 1*2*2*2 = 4. Batch mean 4. / 512 voxels.
    mu = torch.ones(3, 1, 2, 2, 2)
    logvar = torch.zeros(3, 1, 2, 2, 2)
    assert kl_rombach_sum(mu, logvar, n_voxels=512).item() == pytest.approx(4.0 / 512)


def test_kl_rombach_sum_sums_per_sample_then_averages_the_batch():
    mu = torch.zeros(2, 1, 1, 1, 2)
    mu[0, 0, 0, 0, 0] = 2.0          # sample 0: one element with KL 0.5 * 4 = 2
    logvar = torch.zeros_like(mu)    # sample 1: KL 0
    assert kl_rombach_sum(mu, logvar, n_voxels=1).item() == pytest.approx(1.0)


def test_kl_rombach_is_the_per_channel_kl_rescaled_without_free_bits():
    # sum over elements per sample / N == (sum_c mean_{b,s} KL) * (d*h*w) / N.
    mu = torch.randn(4, 3, 2, 2, 2)
    logvar = torch.randn(4, 3, 2, 2, 2) * 0.3
    kl, _, _ = kl_divergence(mu, logvar, free_bits=0.0)
    assert kl_rombach_sum(mu, logvar, n_voxels=64).item() == pytest.approx(kl.item() * 8 / 64, rel=1e-5)


def test_compute_total_loss_uses_the_rombach_kl_term():
    out = _xct_only_output()
    batch = {"xct": torch.rand(2, 1, 8, 8, 8)}
    r = compute_total_loss(out, batch, 0, _cfg(kl_reduction="rombach_sum"))
    expected_kl = kl_rombach_sum(out.mu, out.logvar, n_voxels=8 ** 3)
    assert r["kl_rombach"].item() == pytest.approx(expected_kl.item())
    assert r["total"].item() == pytest.approx((r["xct_loss"] + 1e-6 * expected_kl).item())
    # The per-channel KL is still reported for monitoring.
    assert r["kl_per_channel"].shape == (2,)


def test_rombach_sum_refuses_free_bits():
    out = _xct_only_output()
    batch = {"xct": torch.rand(2, 1, 8, 8, 8)}
    with pytest.raises(ValueError, match="free bits"):
        compute_total_loss(out, batch, 0, _cfg(kl_reduction="rombach_sum", kl_free_bits=0.1))


def test_perceptual_weight_without_a_module_refuses():
    out = _xct_only_output()
    batch = {"xct": torch.rand(2, 1, 8, 8, 8)}
    with pytest.raises(ValueError, match="perceptual"):
        compute_total_loss(out, batch, 0, _cfg(perceptual_weight=1.0))


def test_perceptual_term_is_added_to_the_reconstruction():
    class Const(nn.Module):
        def forward(self, recon, target):
            return torch.tensor(0.25)

    out = _xct_only_output()
    batch = {"xct": torch.rand(2, 1, 8, 8, 8)}
    r = compute_total_loss(out, batch, 0, _cfg(perceptual_weight=2.0, disc_adaptive=True),
                           perceptual=Const())
    assert r["lpips"].item() == pytest.approx(0.25)
    assert r["rec_total"].item() == pytest.approx(r["xct_loss"].item() + 0.5)


# ── adaptive adversarial weight ───────────────────────────────────────────────

def test_adaptive_weight_on_a_tiny_model():
    torch.manual_seed(0)
    trunk = nn.Linear(4, 4)
    last = nn.Linear(4, 1)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    out = last(trunk(x))
    rec = (out - y).abs().mean()
    gen = 0.5 * (out - 1.0).pow(2).mean()

    g_rec = torch.autograd.grad(rec, last.weight, retain_graph=True)[0]
    g_gen = torch.autograd.grad(gen, last.weight, retain_graph=True)[0]
    expected = g_rec.norm() / (g_gen.norm() + 1e-4)

    lam = adaptive_disc_weight(rec, gen, last.weight)
    assert lam.item() == pytest.approx(expected.item(), rel=1e-6)
    assert not lam.requires_grad
    # The graph is retained: the caller still backprops the total.
    (rec + lam * gen).backward()
    assert last.weight.grad is not None


def test_adaptive_weight_is_clamped():
    w = nn.Parameter(torch.ones(3))
    rec = (w * 1e3).sum()
    gen = (w * 0.0).sum()           # zero generator gradient -> ratio 3e3*sqrt(3)/1e-4
    assert adaptive_disc_weight(rec, gen, w).item() == pytest.approx(1e4)


# ── LPIPS slice extraction ────────────────────────────────────────────────────

def test_paired_axis_slices_shapes_and_pairing():
    a = torch.rand(2, 1, 8, 8, 8)
    b = 2.0 * a
    sa, sb = paired_axis_slices(a, b, n_per_axis=4)
    assert sa.shape == (3 * 4 * 2, 1, 8, 8)
    assert sb.shape == sa.shape
    # Same positions in both volumes.
    assert torch.allclose(sb, 2.0 * sa)


def test_paired_axis_slices_are_real_slices_of_each_axis():
    B, D, n = 2, 6, 3
    vol = torch.rand(B, 1, D, D, D)
    s, _ = paired_axis_slices(vol, vol, n_per_axis=n)
    per_axis = s.view(3, B * n, 1, D, D)
    for axis, block in enumerate(per_axis):
        moved = vol.movedim(2 + axis, 1)          # (B, D, 1, D, D): slices along axis
        for i, sl in enumerate(block):
            sample = i // n                        # repeat_interleave order
            assert any(torch.equal(sl, moved[sample, k]) for k in range(D))


def test_paired_axis_slices_refuses_a_non_cubic_patch():
    with pytest.raises(ValueError, match="cubic"):
        paired_axis_slices(torch.rand(1, 1, 4, 4, 8), torch.rand(1, 1, 4, 4, 8), 1)


def test_grey_to_lpips_input_maps_to_three_channels_in_minus_one_one():
    x = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3)
    y = grey_to_lpips_input(x)
    assert y.shape == (1, 3, 1, 3)
    assert torch.equal(y[0, :, 0], torch.tensor([[-1.0, 0.0, 1.0]] * 3))


# ── defaults keep the historical loss ─────────────────────────────────────────

def test_default_loss_is_the_historical_formula_exactly():
    torch.manual_seed(1)
    out = _xct_only_output()
    batch = {"xct": torch.rand(2, 1, 8, 8, 8)}
    cfg = _cfg(kl_free_bits=0.1, kl_max_beta=0.05)
    r = compute_total_loss(out, batch, 0, cfg)
    kl, _, _ = kl_divergence(out.mu, out.logvar, free_bits=0.1)
    zero = torch.zeros(())
    expected = 1.0 * r["xct_loss"] + zero + zero + 0.05 * kl
    assert torch.equal(r["total"], expected)
    assert not {"lpips", "kl_rombach", "rec_total"} & set(r)


@pytest.mark.parametrize("experiment", [
    "r08/base",
    "r08/reduction-factor-16-disc0.2",
    "r08/reduction-factor-16-beta0.02",
    "r07/reduction-factor-16",
])
def test_existing_configs_resolve_to_the_old_behaviour(experiment):
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import adversarial_schedule

    cfg = resolve_experiment(experiment).cfg
    assert not {"kl_reduction", "perceptual_weight", "disc_adaptive", "disc_start"} & set(cfg["loss"])
    assert adversarial_schedule(cfg) == {"disc_adaptive": False, "disc_start": 0}
    assert build_perceptual(cfg, torch.device("cpu")) is None


def test_rombach_config_resolves_to_the_recipe():
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import adversarial_schedule

    cfg = resolve_experiment("r08/rombach-rf16").cfg
    c = cfg["loss"]
    assert c["kl_reduction"] == "rombach_sum"
    assert c["kl_max_beta"] == pytest.approx(1e-6)
    assert c["kl_free_bits"] == 0.0 and c["kl_warmup_steps"] == 0
    assert c["perceptual_weight"] == 1.0 and c["perceptual_net"] == "vgg"
    assert c["perceptual_slices_per_axis"] == 4
    assert cfg["discriminator"]["disc_weight"] == 0.5
    assert adversarial_schedule(cfg) == {"disc_adaptive": True, "disc_start": 4500}
    assert cfg["model"]["z_channels"] == 4 and cfg["model"]["in_channels"] == 3

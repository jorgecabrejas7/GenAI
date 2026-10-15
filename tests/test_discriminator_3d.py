"""The 3D patch discriminator (vault D69) and the advperc rung configs."""

from __future__ import annotations

import pytest
import torch

from poregen.models.discriminator import (
    DISCRIMINATORS,
    PatchDiscriminator2D,
    PatchDiscriminator3D,
    hinge_disc_loss,
    lsgan_disc_loss,
)


def test_3d_scores_one_value_per_volume():
    d = PatchDiscriminator3D(base_channels=8)
    vol = torch.rand(2, 1, 64, 64, 64)
    assert d.prepare(vol) is vol
    assert d(d.prepare(vol)).shape == (2,)


def test_2d_prepare_still_gives_three_slices_per_sample():
    d = PatchDiscriminator2D(base_channels=8)
    x = d.prepare(torch.rand(2, 1, 64, 64, 64))
    assert x.shape == (6, 1, 64, 64)
    assert d(x).shape == (6,)


def test_3d_discriminator_can_learn_to_separate_real_from_blurred():
    """A working discriminator moves real and fake scores apart in a few steps."""
    torch.manual_seed(0)
    d = PatchDiscriminator3D(base_channels=8)
    opt = torch.optim.AdamW(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real = torch.rand(4, 1, 32, 32, 32)
    fake = torch.nn.functional.avg_pool3d(real, 5, stride=1, padding=2)
    for _ in range(60):
        opt.zero_grad()
        s = d(torch.cat([real, fake]))
        lsgan_disc_loss(s[:4], s[4:]).backward()
        opt.step()
    with torch.no_grad():
        assert (d(real).mean() - d(fake).mean()) > 0.3


def test_build_discriminator_picks_the_kind():
    from poregen.experiments.train_vae import build_discriminator

    for kind, cls in DISCRIMINATORS.items():
        d, opt, w = build_discriminator(
            {"discriminator": {"kind": kind, "base_channels": 8, "disc_weight": 0.5}},
            torch.device("cpu"))
        assert isinstance(d, cls) and w == 0.5
    d, _, _ = build_discriminator({"discriminator": {"base_channels": 8}}, torch.device("cpu"))
    assert isinstance(d, PatchDiscriminator2D)
    with pytest.raises(ValueError):
        build_discriminator({"discriminator": {"kind": "nope"}}, torch.device("cpu"))


@pytest.mark.parametrize("rf,z", [(2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)])
def test_advperc_rungs_are_rombach_rf16_with_a_3d_discriminator(rf, z):
    from poregen.configuration import resolve_experiment

    ref = resolve_experiment("r08/rombach-rf16").cfg
    cfg = resolve_experiment(f"r08/advperc-rf{rf}").cfg
    assert cfg["discriminator"]["kind"] == "volume3d"
    assert cfg["model"]["z_channels"] == z
    assert cfg["loss"] == ref["loss"]
    assert {k: v for k, v in cfg["discriminator"].items() if k != "kind"} == ref["discriminator"]
    assert cfg["training"] == ref["training"]
    assert {k: v for k, v in cfg["model"].items() if k != "z_channels"} == \
        {k: v for k, v in ref["model"].items() if k != "z_channels"}


# ── nlayer3d: the LDM autoencoder's discriminator in 3D (vault OQ-69) ─────────

def test_nlayer3d_matches_the_taming_layout():
    from poregen.models.discriminator import NLayerDiscriminator3D

    d = NLayerDiscriminator3D()
    assert d.joint_pass is False
    convs = [m for m in d.modules() if isinstance(m, torch.nn.Conv3d)]
    bns = [m for m in d.modules() if isinstance(m, torch.nn.BatchNorm3d)]
    assert [c.out_channels for c in convs] == [64, 128, 256, 512, 1]
    assert [c.stride[0] for c in convs] == [2, 2, 2, 1, 1]
    assert len(bns) == 3
    assert not any("weight_orig" in n for n, _ in d.named_parameters())  # no spectral norm
    out = d(d.prepare(torch.rand(2, 1, 64, 64, 64)))
    assert out.shape == (2, 1, 6, 6, 6)                                   # a logit per location


def test_nlayer3d_prepare_maps_grey_to_minus_one_one():
    from poregen.models.discriminator import NLayerDiscriminator3D

    x = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1)
    assert torch.equal(NLayerDiscriminator3D.prepare(x).flatten(), torch.tensor([-1.0, 0.0, 1.0]))


def test_hinge_losses_are_the_taming_formulas():
    from poregen.models.discriminator import ADV_LOSSES, hinge_disc_loss, hinge_gen_loss

    real = torch.tensor([2.0, 0.0, -1.0]); fake = torch.tensor([-2.0, 0.5, 1.0])
    expect = 0.5 * (torch.relu(1 - real).mean() + torch.relu(1 + fake).mean())
    assert torch.isclose(hinge_disc_loss(real, fake), expect)
    assert torch.isclose(hinge_gen_loss(fake), -fake.mean())
    assert ADV_LOSSES["hinge"][2] == 0.0 and ADV_LOSSES["lsgan"][2] == 0.5


def test_nlayer3d_hinge_learns_to_separate_real_from_blurred():
    from poregen.models.discriminator import NLayerDiscriminator3D

    torch.manual_seed(0)
    d = NLayerDiscriminator3D(base_channels=8)
    opt = torch.optim.AdamW(d.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0)
    real = torch.rand(4, 1, 32, 32, 32)
    fake = torch.nn.functional.avg_pool3d(real, 5, stride=1, padding=2)
    for _ in range(40):
        opt.zero_grad()
        hinge_disc_loss(d(d.prepare(real)), d(d.prepare(fake))).backward()
        opt.step()
    d.eval()
    with torch.no_grad():
        assert d(d.prepare(real)).mean() - d(d.prepare(fake)).mean() > 0.3


@pytest.mark.parametrize("rf,z", [(2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)])
def test_advperc_ldm_rungs_carry_the_official_recipe(rf, z):
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import adversarial_schedule, build_discriminator, build_optimizer
    from poregen.models.discriminator import NLayerDiscriminator3D

    cfg = resolve_experiment(f"r08/advperc-ldm-rf{rf}").cfg
    assert cfg["model"]["z_channels"] == z
    assert (cfg["loss"]["xct_loss_type"], cfg["loss"]["xct_weight"]) == ("l1", 2.0)
    assert cfg["loss"]["kl_reduction"] == "rombach_sum" and cfg["loss"]["kl_max_beta"] == 1e-6
    assert cfg["loss"]["perceptual_weight"] == 1.0
    sched = adversarial_schedule(cfg)
    assert sched == {"disc_adaptive": True, "disc_start": 15000, "adv_loss": "hinge"}
    assert cfg["training"]["early_stopping_warmup_steps"] == 15000
    d, dopt, w = build_discriminator(cfg, torch.device("cpu"))
    assert isinstance(d, NLayerDiscriminator3D) and w == 0.5
    g = dopt.param_groups[0]
    assert g["betas"] == (0.5, 0.9) and g["weight_decay"] == 0.0 and g["lr"] == pytest.approx(5.76e-4)
    g = build_optimizer(cfg, torch.nn.Linear(2, 2)).param_groups[0]
    assert g["betas"] == (0.5, 0.9) and g["weight_decay"] == 0.0 and g["lr"] == pytest.approx(5.76e-4)

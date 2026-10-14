"""The 3D patch discriminator (vault D69) and the advperc rung configs."""

from __future__ import annotations

import pytest
import torch

from poregen.models.discriminator import (
    DISCRIMINATORS,
    PatchDiscriminator2D,
    PatchDiscriminator3D,
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

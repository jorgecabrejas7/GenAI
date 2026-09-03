"""The nested CFG decomposition in ``DDIMSampler.predict_out``.

The null arms are structural, not numerical: the porosity null is the learned
``null_por`` token and the neighbour null is all-UNKNOWN at ``nb_t = 0`` — the
exact state ``ldm_engine.noise_neighbours`` produces under ``drop_nb``.  If the
two definitions ever drift the guidance silently guides towards something the
model was never trained on, so both are pinned here.

All tests run on CPU with a tiny model so they complete in seconds.
"""

from __future__ import annotations

import pytest
import torch

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_UNKNOWN,
    N_DIST6,
    N_NEIGHBOURS,
)
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler
from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser

B, Z, D = 2, 2, 4
DEV = torch.device("cpu")


@pytest.fixture()
def model() -> UNet3DDenoiser:
    cfg = UNet3DConfig(z_channels=Z, base_channels=8, channel_mult=(1, 2),
                       n_res_blocks=1, cond_embed_dim=16, nb_avail_embed_dim=4,
                       nb_t_embed_dim=4, n_attn_heads=2)
    m = UNet3DDenoiser(cfg)
    torch.manual_seed(0)
    torch.nn.init.normal_(m.out_conv.weight, std=0.1)
    for mod in m.modules():
        if type(mod).__name__ == "AdaGN":
            torch.nn.init.normal_(mod.proj.weight, std=0.1)
    return m.eval()


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(B, Z, D, D, D, generator=g),
        t=torch.zeros(B, dtype=torch.long),
        nb_latents=torch.randn(B, N_NEIGHBOURS, Z, D, D, D, generator=g),
        nb_avail=torch.full((B, N_NEIGHBOURS), NB_EXISTS),
        nb_t=torch.randint(0, 10, (B, N_NEIGHBOURS), generator=g),
        cond_por=torch.randn(B, generator=g),
        cond_depth=torch.rand(B, generator=g),
        cond_dist6=torch.rand(B, N_DIST6, generator=g),
        cond_orient=torch.randn(B, 2, D, D, D, generator=g),
        cond_material=torch.rand(B, 1, D, D, D, generator=g),
    )


def _sampler(model, s_por=1.0, s_nb=1.0):
    return DDIMSampler(model, DDPMSchedule(T=10, device=DEV), DEV,
                       n_steps=1, s_por=s_por, s_nb=s_nb)


def test_unguided_scales_take_the_single_pass(model):
    assert _sampler(model).guided is False
    assert _sampler(model, s_por=1.5).guided is True
    assert _sampler(model, s_nb=0.5).guided is True


def test_three_pass_formula_telescopes_at_scale_one(model):
    """s_por = s_nb = 1 must be EXACTLY the full-conditional pass."""
    b = _batch()
    with torch.no_grad():
        reference = model(b["x"], b["t"], b["nb_latents"], b["nb_avail"], b["nb_t"],
                          b["cond_por"], b["cond_depth"], b["cond_dist6"],
                          b["cond_orient"], b["cond_material"])
    s = _sampler(model)
    s.guided = True                       # force the 3-pass path
    with torch.no_grad():
        three_pass = s.predict_out(**b, autocast_dtype=torch.float32)
    assert torch.allclose(reference.float(), three_pass.float(), atol=1e-5)


def test_guidance_scales_change_the_prediction(model):
    b = _batch()
    with torch.no_grad():
        plain = _sampler(model).predict_out(**b, autocast_dtype=torch.float32)
        por_guided = _sampler(model, s_por=3.0).predict_out(
            **b, autocast_dtype=torch.float32)
        nb_guided = _sampler(model, s_nb=3.0).predict_out(
            **b, autocast_dtype=torch.float32)
    assert not torch.allclose(plain, por_guided, atol=1e-5)
    assert not torch.allclose(plain, nb_guided, atol=1e-5)


def test_the_neighbour_arm_is_inert_when_there_are_no_neighbours(model):
    """With every face already UNKNOWN, eps_full == eps_por, so s_nb cannot act."""
    b = _batch()
    b["nb_avail"] = torch.full((B, N_NEIGHBOURS), NB_UNKNOWN)
    b["nb_t"] = torch.zeros_like(b["nb_t"])
    with torch.no_grad():
        plain = _sampler(model).predict_out(**b, autocast_dtype=torch.float32)
        guided = _sampler(model, s_nb=4.0).predict_out(
            **b, autocast_dtype=torch.float32)
    assert torch.allclose(plain, guided, atol=1e-5)


def test_the_null_arms_carry_no_neighbour_information(model):
    """Both null passes must be all-UNKNOWN at nb_t 0 — the training null."""
    seen: list[tuple] = []
    real_forward = model.forward

    def spy(x, t, nb_latents, nb_avail, nb_t, *args, **kw):
        seen.append((nb_avail.clone(), nb_t.clone(),
                     args[-1] if args and torch.is_tensor(args[-1]) else kw.get("drop_por")))
        return real_forward(x, t, nb_latents, nb_avail, nb_t, *args, **kw)

    model.forward = spy                                   # type: ignore[method-assign]
    try:
        with torch.no_grad():
            _sampler(model, s_por=2.0, s_nb=2.0).predict_out(
                **_batch(), autocast_dtype=torch.float32)
    finally:
        model.forward = real_forward                      # type: ignore[method-assign]

    assert len(seen) == 3
    uncond, por_only, full = seen
    assert bool((uncond[0] == NB_UNKNOWN).all()) and bool((uncond[1] == 0).all())
    assert bool((por_only[0] == NB_UNKNOWN).all()) and bool((por_only[1] == 0).all())
    assert bool((full[0] == NB_EXISTS).all())


def test_ldm06_config_resolves() -> None:
    """ldm06/base resolves and declares the geometry the code enforces."""
    from poregen.configuration import resolve_experiment

    cfg = resolve_experiment("ldm06/base").cfg

    assert UNet3DConfig.from_cfg(cfg).in_channels == 127
    # Three distinct strides.  sample_stride stays 32 as a training-data
    # multiplier; the tiling grid and the neighbour relation are 64 so that
    # neighbours share no voxel with the target.
    assert cfg["data"]["sample_stride"] == 32
    assert cfg["data"]["generation_stride"] == 64
    assert cfg["data"]["neighbour_offset"] == 64
    assert cfg["data"]["latent_mode"] == "sampled"
    assert cfg["data"]["latents_root"] == "data/split_v3/latents_r08z4"
    # The ldm05 switches must not come back through an inherited config.
    for gone in ("use_por_cond", "use_neighbor_cond", "use_orient_cond",
                 "use_pos_cond", "use_por_null", "use_avail_embedding",
                 "use_nb_global_cond"):
        assert gone not in cfg["model"]
    for gone in ("neighbour_shift", "allow_neighbour_overlap", "material",
                 "air_patch_cap"):
        assert gone not in cfg["data"]
    assert float(cfg["training"]["drop_por"]) > 0
    assert float(cfg["training"]["drop_nb"]) > 0
    assert 0.0 <= float(cfg["training"]["nb_t_mix"]) <= 1.0
    assert cfg["generation"]["chunk_tiles"] == [3, 3, 3]
    assert cfg["generation"]["window_stride"] == 32
    assert cfg["generation"]["decode_stride"] == 32
    # The in-training sample volume is exactly one chunk.
    assert cfg["training"]["sample_grid"] == cfg["generation"]["chunk_tiles"]
    assert cfg["vae"]["checkpoint"]
    assert int(cfg["training"]["gen_eval_every"]) > 0

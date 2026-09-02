"""Sanity checks for the CFG implementation (ldm03).

All tests run on CPU with tiny model configs so they complete in seconds.
"""

from __future__ import annotations

import pytest
import torch

from poregen.diffusion.conditioning import NB_EXISTS, NB_UNKNOWN
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler
from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser


# ── Shared tiny model fixtures ────────────────────────────────────────────────

_MINI_CFG_KWARGS = dict(
    z_channels=2,
    base_channels=8,
    channel_mult=(1, 2),
    n_res_blocks=1,
    cond_embed_dim=16,
    nb_avail_embed_dim=4,
    n_attn_heads=2,
    dropout=0.0,
)


@pytest.fixture()
def mini_cfg_no_null() -> UNet3DConfig:
    """ldm01/ldm02-style config — no null_por."""
    return UNet3DConfig(**_MINI_CFG_KWARGS, use_por_null=False)


@pytest.fixture()
def mini_cfg_with_null() -> UNet3DConfig:
    """ldm03-style config — with learned null_por."""
    return UNet3DConfig(**_MINI_CFG_KWARGS, use_por_null=True)


@pytest.fixture()
def model_with_null(mini_cfg_with_null: UNet3DConfig) -> UNet3DDenoiser:
    return UNet3DDenoiser(mini_cfg_with_null)


@pytest.fixture()
def model_no_null(mini_cfg_no_null: UNet3DConfig) -> UNet3DDenoiser:
    return UNet3DDenoiser(mini_cfg_no_null)


def _make_batch(B: int = 2, z_ch: int = 2, D: int = 4):
    """Return a small synthetic batch following the D32 §5 contract."""
    z_t    = torch.randn(B, z_ch, D, D, D)
    t      = torch.zeros(B, dtype=torch.long)
    nb_l   = torch.randn(B, 6, z_ch, D, D, D)
    nb_a   = torch.ones(B, 6, dtype=torch.long)   # all EXISTS
    por    = torch.randn(B)
    depth  = torch.rand(B)
    dist   = torch.rand(B)
    orient = torch.randn(B, 2, D, D, D)
    return z_t, t, nb_l, nb_a, por, depth, dist, orient


# ── Sanity check 1: null_por receives gradient ────────────────────────────────

def test_null_por_in_graph_and_receives_gradient(model_with_null: UNet3DDenoiser) -> None:
    """null_por is a leaf parameter that gets a non-zero gradient when
    drop_por=True for at least one sample in a forward+backward pass.

    Two zero-initialization choices in the UNet conspire to kill the gradient at
    fresh initialization: out_conv (output head) and AdaGN.proj (conditioning
    injection in each residual block) are both zero-initialized for training
    stability.  At initialisation, AdaGN.proj(cond)=0 so the feature maps don't
    depend on cond at all, and the gradient of any output-derived loss w.r.t.
    cond (and hence null_por) is identically zero.

    We un-zero both so the full gradient chain from output → features → cond →
    null_por is active for this test.  The test checks the real property:
    null_por is in the computation graph and accumulates gradient.
    """
    model_with_null.train()
    torch.manual_seed(42)
    # Un-zero output head so gradient flows out of the network.
    torch.nn.init.normal_(model_with_null.out_conv.weight, std=0.01)
    # Un-zero all AdaGN projections so cond is not detached from the computation graph.
    for module in model_with_null.modules():
        name = type(module).__name__
        if name == "AdaGN" and hasattr(module, "proj"):
            torch.nn.init.normal_(module.proj.weight, std=0.01)
            torch.nn.init.zeros_(module.proj.bias)   # keep bias zero, weight non-zero

    assert hasattr(model_with_null, "null_por"), "null_por parameter missing"
    assert model_with_null.null_por.requires_grad, "null_por.requires_grad should be True"

    z_t, t, nb_l, nb_a, por, depth, dist, orient = _make_batch()
    # Drop porosity for ALL samples — guarantees null_por is in the compute graph.
    drop_all = torch.ones(z_t.shape[0], dtype=torch.bool)

    out  = model_with_null(z_t, t, nb_l, nb_a, por, depth, dist, orient, drop_por=drop_all)
    loss = out.sum()
    loss.backward()

    assert model_with_null.null_por.grad is not None, "null_por.grad is None after backward"
    assert model_with_null.null_por.grad.abs().sum().item() > 0, (
        "null_por.grad is all-zero — not in the compute graph. "
        "Check that _build_cond uses torch.where with null_por on the True branch "
        "and that AdaGN.proj weights are non-zero."
    )


# ── Sanity check 2: UNKNOWN availability zeroes spatial + pooled summary ─────

def test_unknown_avail_zeros_neighbour_contributions(model_with_null: UNet3DDenoiser) -> None:
    """After forcing nb_avail=UNKNOWN:
    - the spatial neighbour input to input_proj is zero for ALL neighbour slots, and
    - the pooled neighbour summary fed into the cond vector is zero.
    The output must be identical regardless of what nb_latents contains.
    """
    model_with_null.eval()
    B, z_ch, D = 2, 2, 4

    z_t    = torch.randn(B, z_ch, D, D, D)
    t      = torch.zeros(B, dtype=torch.long)
    por    = torch.randn(B)
    depth  = torch.rand(B)
    dist   = torch.rand(B)
    orient = torch.randn(B, 2, D, D, D)

    # Two different non-zero neighbour tensors
    nb_l_a = torch.randn(B, 6, z_ch, D, D, D)
    nb_l_b = torch.randn(B, 6, z_ch, D, D, D)
    nb_a_all_unk = torch.full((B, 6), NB_UNKNOWN, dtype=torch.long)

    with torch.no_grad():
        out_a = model_with_null(z_t, t, nb_l_a, nb_a_all_unk, por, depth, dist, orient)
        out_b = model_with_null(z_t, t, nb_l_b, nb_a_all_unk, por, depth, dist, orient)

    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "Outputs differ when nb_avail=UNKNOWN but nb_latents differ — "
        "the EXISTS mask in _build_nb_spatial is not applied correctly."
    )


# ── Sanity check 3: s_por=s_nb=1 reproduces un-guided full-conditional pass ──

def test_guided_at_scale_1_equals_full_conditional(model_with_null: UNet3DDenoiser) -> None:
    """At s_por=s_nb=1.0 the 3-pass formula telescopes to eps_full.
    We verify by running DDIMSampler with 1 step (n_steps=1) in guided vs
    un-guided mode and comparing the sampled latent tensors element-wise.
    """
    torch.manual_seed(0)
    B, z_ch, D = 2, 2, 4

    schedule = DDPMSchedule(T=10, device="cpu")
    model_with_null.eval()

    # Shared inputs
    nb_l  = torch.randn(B, 6, z_ch, D, D, D)
    nb_a   = torch.ones(B, 6, dtype=torch.long)   # real EXISTS neighbours
    por    = torch.randn(B)
    depth  = torch.rand(B)
    dist   = torch.rand(B)
    orient = torch.randn(B, 2, D, D, D)

    # To compare deterministically: use a fixed z_t as starting point, run one
    # denoising step with both samplers and compare intermediate eps predictions.
    z_t    = torch.randn(B, z_ch, D, D, D)
    t_full = torch.zeros(B, dtype=torch.long)

    # Reference: direct model call (full conditional, no drop)
    with torch.no_grad():
        eps_reference = model_with_null(z_t, t_full, nb_l, nb_a, por, depth, dist, orient, None)

    # Guided sampler at s_por=s_nb=1.0 — should produce the same eps
    sampler_guided = DDIMSampler(model_with_null, schedule, torch.device("cpu"),
                                 n_steps=1, s_por=1.0, s_nb=1.0)
    # Confirm the guided flag is OFF (short-circuit path)
    assert not sampler_guided.guided, (
        "DDIMSampler.guided should be False when s_por=s_nb=1.0"
    )

    # Confirm that manually calling _guided_eps reproduces the reference
    # (this is independent of the short-circuit; tests the formula itself)
    sampler_always_guided = DDIMSampler(model_with_null, schedule, torch.device("cpu"),
                                        n_steps=1, s_por=1.0, s_nb=1.0)
    # Force guided=True to exercise the 3-pass code path
    sampler_always_guided.guided = True
    with torch.no_grad():
        eps_3pass = sampler_always_guided._guided_eps(
            z_t, t_full, nb_l, nb_a, por, depth, dist, orient,
            autocast_dtype=torch.float32,
        )

    assert torch.allclose(eps_reference.float(), eps_3pass.float(), atol=1e-5), (
        "3-pass CFG formula at s_por=s_nb=1.0 does not reproduce the single full-conditional "
        "pass.  Max deviation: {:.2e}".format(
            (eps_reference.float() - eps_3pass.float()).abs().max().item()
        )
    )


# ── Sanity check 4: backward compatibility (drop_por=None leaves output identical) ──

def test_drop_por_none_is_backward_compatible(
    model_no_null: UNet3DDenoiser,
    model_with_null: UNet3DDenoiser,
) -> None:
    """forward(..., drop_por=None) is identical to forward(...) for both
    use_por_null=False (ldm01/02) and use_por_null=True (ldm03) models."""
    z_t, t, nb_l, nb_a, por, depth, dist, orient = _make_batch()

    for model in (model_no_null, model_with_null):
        model.eval()
        with torch.no_grad():
            out_no_arg  = model(z_t, t, nb_l, nb_a, por, depth, dist, orient)
            out_none    = model(z_t, t, nb_l, nb_a, por, depth, dist, orient, drop_por=None)
        assert torch.allclose(out_no_arg, out_none, atol=1e-7), (
            f"drop_por=None changes output for {model.cfg.use_por_null=}"
        )


# ── Config resolution smoke test ─────────────────────────────────────────────

def test_ldm05_config_resolves() -> None:
    """ldm05/base resolves: the fully conditional rung (D32)."""
    try:
        from poregen.configuration import resolve_experiment
    except ImportError:
        pytest.skip("poregen configuration not importable in this environment")

    resolved = resolve_experiment("ldm05/base")
    cfg = resolved.cfg

    assert cfg["model"]["use_por_cond"] is True
    assert cfg["model"]["use_neighbor_cond"] is True
    assert cfg["model"]["use_pos_cond"] is True
    assert cfg["model"]["use_orient_cond"] is True
    assert cfg["model"]["use_por_null"] is True
    # Three distinct strides (D32 §3.1).  sample_stride stays 32 as a
    # training-data multiplier; the generation grid and the neighbour relation
    # are 64 so that patches tile and neighbours share no voxel with the target.
    assert cfg["data"]["sample_stride"] == 32
    assert cfg["data"]["generation_stride"] == 64
    assert cfg["data"]["neighbour_offset"] == 64
    assert cfg["data"]["neighbour_shift"] is False
    assert cfg["data"].get("allow_neighbour_overlap", False) is False
    assert cfg["data"]["latent_mode"] == "sampled"
    assert cfg["data"]["latents_root"] == "data/split_v2/latents_r07z4"
    assert cfg["vae"]["checkpoint"]
    assert int(cfg["training"]["gen_eval_every"]) > 0

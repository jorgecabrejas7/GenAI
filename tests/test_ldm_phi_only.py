"""The porosity-only baseline: same trunk as ldm06, blind to everything else.

THE TRAP THIS SUITE IS BUILT AROUND. Three things in this network are
ZERO-INITIALISED on purpose — the output head, the residual blocks' output
projection, and the FiLM projection that every conditioning vector reaches the
trunk through. On a freshly constructed model the output is therefore
identically zero, and the whole `cond` path has no effect at all. Every "is it
blind to X?" question answers YES on such a model, for both networks, whatever
they actually do. The first version of this check passed on exactly that and
proved nothing.

So `live()` undoes those three initialisations before anything is asserted, and
`test_the_liveness_fixture_is_doing_its_job` fails if it ever stops working.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from poregen.baselines.ldm_phi_only import PhiOnlyDenoiser
from poregen.baselines.ldm_phi_only.networks import REMOVED_MODULES
from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser

CONFIG = "configs/experiments/ldm06/base.yaml"
S = 8


def small_cfg() -> UNet3DConfig:
    """ldm06's real settings, narrowed so the suite runs on a CPU in seconds."""
    cfg = UNet3DConfig.from_cfg(yaml.safe_load(open(CONFIG)))
    cfg.base_channels = 16
    cfg.cond_embed_dim = 32
    cfg.n_res_blocks = 1
    return cfg


def live(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Undo every zero-initialisation, so the model actually uses its inputs."""
    g = torch.Generator().manual_seed(seed)

    # An all-zero parameter IS the signature of a deliberate zero-init here:
    # the output head, each residual block's output projection, and the FiLM
    # projection every conditioning vector reaches the trunk through. Filling
    # them by that test needs no list of names to fall out of date.
    with torch.no_grad():
        for p in model.parameters():
            if not p.any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def inputs(cfg: UNet3DConfig, b: int = 2, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    z = cfg.z_channels
    return dict(
        z_t=torch.randn(b, z, S, S, S, generator=g),
        t=torch.randint(0, 1000, (b,), generator=g),
        nb=torch.randn(b, 6, z, S, S, S, generator=g),
        avail=torch.ones(b, 6, dtype=torch.long),
        nbt=torch.zeros(b, 6, dtype=torch.long),
        por=torch.randn(b, generator=g),
        depth=torch.rand(b, generator=g),
        d6=torch.rand(b, 6, generator=g),
        orient=torch.randn(b, 2, S, S, S, generator=g),
        mat=torch.ones(b, 1, S, S, S),
    )


def call(model, d: dict, **over):
    d = {**d, **over}
    with torch.no_grad():
        return model(d["z_t"], d["t"], d["nb"], d["avail"], d["nbt"], d["por"],
                     d["depth"], d["d6"], d["orient"], d["mat"])


@pytest.fixture(scope="module")
def cfg():
    return small_cfg()


@pytest.fixture(scope="module")
def phi(cfg):
    torch.manual_seed(0)
    return live(PhiOnlyDenoiser(cfg))


@pytest.fixture(scope="module")
def ldm06(cfg):
    torch.manual_seed(0)
    return live(UNet3DDenoiser(cfg))


class TestTheFixtureItself:
    """If these fail, every other test in the file is passing for free."""

    def test_the_liveness_fixture_is_doing_its_job(self, cfg):
        """A fresh model outputs zeros; a live one must not."""
        torch.manual_seed(0)
        fresh = PhiOnlyDenoiser(cfg).eval()
        d = inputs(cfg)
        assert float(call(fresh, d).abs().max()) == 0.0, (
            "a fresh model should output exactly zero — if this changes, the "
            "zero-init trap is gone and `live()` may no longer be needed")
        torch.manual_seed(0)
        assert float(call(live(PhiOnlyDenoiser(cfg)), d).std()) > 1e-3

    def test_a_live_model_uses_its_conditioning_vector(self, cfg, phi):
        """The FiLM path must be live, or 'blind to X' means nothing."""
        d = inputs(cfg)
        assert not torch.allclose(call(phi, d), call(phi, d, por=d["por"] + 3.0))
        assert not torch.allclose(call(phi, d), call(phi, d, t=d["t"] * 0 + 900))


class TestItIsTheSameTrunk:
    """The comparison is only about conditioning if the trunk is shared."""

    def test_every_trunk_module_is_shared_with_ldm06(self, cfg, phi, ldm06):
        removed = set(REMOVED_MODULES)
        a = {n for n, _ in ldm06.named_modules() if n}
        b = {n for n, _ in phi.named_modules() if n}
        missing = {n for n in a - b if n.split(".")[0] not in removed}
        assert not missing, f"phi-only is missing trunk modules: {sorted(missing)}"
        assert not (b - a), f"phi-only has modules ldm06 does not: {sorted(b - a)}"

    def test_it_drops_only_the_conditioning_it_says_it_drops(self, phi, ldm06):
        for name in REMOVED_MODULES:
            assert hasattr(ldm06, name)
            assert not hasattr(phi, name)

    def test_the_trunk_is_literally_ldm06s_code(self):
        assert PhiOnlyDenoiser._trunk is UNet3DDenoiser._trunk

    def test_only_the_stem_differs_in_width(self, cfg, phi, ldm06):
        assert ldm06.input_proj.in_channels == cfg.in_channels == 155
        assert phi.input_proj.in_channels == cfg.z_channels

    def test_capacity_is_not_the_variable(self, cfg):
        """A gap between the two must be the conditioning, not the size."""
        full = UNet3DConfig.from_cfg(yaml.safe_load(open(CONFIG)))
        torch.manual_seed(0)
        a = sum(p.numel() for p in UNet3DDenoiser(full).parameters())
        torch.manual_seed(0)
        b = sum(p.numel() for p in PhiOnlyDenoiser(full).parameters())
        assert a == pytest.approx(83.00e6, rel=0.01)
        assert b == pytest.approx(81.67e6, rel=0.01)
        assert b / a > 0.98


class TestWhatItIsBlindTo:

    @pytest.mark.parametrize("field,value", [
        ("nb", torch.randn(2, 6, 8, S, S, S)),
        ("avail", torch.zeros(2, 6, dtype=torch.long)),
        ("nbt", torch.full((2, 6), 500, dtype=torch.long)),
        ("depth", torch.rand(2) * 9),
        ("d6", torch.rand(2, 6) * 9),
        ("orient", torch.randn(2, 2, S, S, S) * 7),
        ("mat", torch.zeros(2, 1, S, S, S)),
    ])
    def test_it_ignores_every_input_but_the_latent_porosity_and_t(
            self, cfg, phi, field, value):
        d = inputs(cfg)
        assert torch.equal(call(phi, d), call(phi, d, **{field: value}))

    @pytest.mark.parametrize("field", ["nb", "orient", "mat"])
    def test_ldm06_does_NOT_ignore_them(self, cfg, ldm06, field):
        """The other half of the claim: these inputs do reach ldm06."""
        d = inputs(cfg)
        other = {"nb": torch.randn(2, 6, cfg.z_channels, S, S, S),
                 "orient": torch.randn(2, 2, S, S, S) * 7,
                 "mat": torch.zeros(2, 1, S, S, S)}[field]
        assert not torch.equal(call(ldm06, d), call(ldm06, d, **{field: other}))

    def test_it_still_uses_the_latent_the_porosity_and_the_timestep(self, cfg, phi):
        d = inputs(cfg)
        ref = call(phi, d)
        assert not torch.allclose(ref, call(phi, d, z_t=torch.randn_like(d["z_t"])))
        assert not torch.allclose(ref, call(phi, d, por=d["por"] + 3.0))
        assert not torch.allclose(ref, call(phi, d, t=d["t"] * 0 + 900))


class TestCfgDropout:

    def test_the_learned_null_replaces_the_porosity_embedding(self, cfg, phi):
        d = inputs(cfg)
        with torch.no_grad():
            dropped = phi(d["z_t"], d["t"], d["nb"], d["avail"], d["nbt"], d["por"],
                          d["depth"], d["d6"], d["orient"], d["mat"],
                          torch.ones(2, dtype=torch.bool))
        assert not torch.allclose(call(phi, d), dropped)

    def test_a_dropped_sample_does_not_depend_on_its_porosity(self, cfg, phi):
        """The null must be a constant, not a function of the value dropped."""
        d = inputs(cfg)
        drop = torch.ones(2, dtype=torch.bool)
        with torch.no_grad():
            a = phi(d["z_t"], d["t"], d["nb"], d["avail"], d["nbt"], d["por"],
                    d["depth"], d["d6"], d["orient"], d["mat"], drop)
            b = phi(d["z_t"], d["t"], d["nb"], d["avail"], d["nbt"], d["por"] + 5.0,
                    d["depth"], d["d6"], d["orient"], d["mat"], drop)
        assert torch.equal(a, b)


class TestSNbIsInert:

    def test_the_neighbour_guidance_term_is_identically_zero(self, cfg, phi):
        """Why the cfg assessment runs the s_por arm only.

        The guided sampler forms out_uncond, out_por (neighbours all UNKNOWN)
        and out_full (real neighbours). This model ignores neighbours, so
        out_full EQUALS out_por and s_nb multiplies an exact zero. An s_nb
        sweep on it would draw a flat line that looks like a measurement.
        """
        d = inputs(cfg)
        all_unk = torch.full_like(d["avail"], 2)
        with torch.no_grad():
            out_por = phi(d["z_t"], d["t"], d["nb"], all_unk, d["nbt"] * 0, d["por"],
                          d["depth"], d["d6"], d["orient"], d["mat"], None)
            out_full = phi(d["z_t"], d["t"], d["nb"], d["avail"], d["nbt"], d["por"],
                           d["depth"], d["d6"], d["orient"], d["mat"], None)
        assert torch.equal(out_por, out_full)

"""The ldm06 model input contract and the neighbour-noising training step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _ldm06_store import build_store, dataset_kwargs
from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    N_AVAIL_STATES,
    N_DIST6,
    N_NEIGHBOURS,
)
from poregen.diffusion.latents import LatentDataset
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
from poregen.training.ldm_engine import (
    ldm_eval_step,
    ldm_train_step,
    noise_neighbours,
    sample_neighbours,
)

T = 1000
Z, L, B = 4, 8, 3


def _cfg(**kw) -> UNet3DConfig:
    base = dict(z_channels=Z, base_channels=8, channel_mult=(1, 2),
                n_res_blocks=1, cond_embed_dim=16)
    base.update(kw)
    return UNet3DConfig(**base)


def _wake(model: UNet3DDenoiser, seed: int = 0) -> UNet3DDenoiser:
    """Break the identity initialisation so conditioning can reach the output.

    ``out_conv`` and every ``AdaGN.proj`` are zero-initialised on purpose (the
    denoiser starts as an identity and the FiLM path starts neutral), so a
    freshly built model ignores every scalar by construction.  A wiring test
    has to perturb them or it proves nothing.
    """
    torch.manual_seed(seed)
    torch.nn.init.normal_(model.out_conv.weight, std=0.1)
    for name, module in model.named_modules():
        if type(module).__name__ == "AdaGN":
            torch.nn.init.normal_(module.proj.weight, std=0.1)
            torch.nn.init.normal_(module.proj.bias, std=0.1)
    return model


def _inputs(cfg: UNet3DConfig, b: int = B, l: int = L, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return dict(
        z_t=torch.randn(b, cfg.z_channels, l, l, l, generator=g),
        t=torch.randint(0, T, (b,), generator=g),
        nb_latents=torch.randn(b, N_NEIGHBOURS, cfg.z_channels, l, l, l, generator=g),
        nb_avail=torch.full((b, N_NEIGHBOURS), NB_EXISTS),
        nb_t=torch.randint(0, T, (b, N_NEIGHBOURS), generator=g),
        cond_por=torch.randn(b, generator=g),
        cond_depth=torch.rand(b, generator=g),
        cond_dist6=torch.rand(b, N_DIST6, generator=g),
        cond_orient=torch.randn(b, 2, l, l, l, generator=g),
        cond_material=torch.rand(b, 1, l, l, l, generator=g),
    )


# ── input channel accounting ─────────────────────────────────────────────────

class TestUNetInputs:

    def test_channel_count_is_the_sum_of_every_conditioning_path(self):
        cfg = _cfg(nb_avail_embed_dim=8, nb_t_embed_dim=8)
        expected = (Z                       # noisy latent
                    + 2                     # (cos2t, sin2t) orientation
                    + 1                     # material fraction
                    + N_NEIGHBOURS * Z      # neighbour latents
                    + N_NEIGHBOURS * 8      # availability embeddings
                    + N_NEIGHBOURS * 8)     # nb_t embeddings
        assert cfg.in_channels == expected == 127

    def test_production_config_is_127_channels(self):
        """z=4, 8-wide availability and nb_t embeddings — the ldm06 rung."""
        assert UNet3DConfig.from_cfg({"model": {"z_channels": 4}}).in_channels == 127

    def test_embedding_widths_move_the_count(self):
        assert _cfg(nb_t_embed_dim=16).in_channels == 127 + 6 * 8
        assert _cfg(nb_avail_embed_dim=4).in_channels == 127 - 6 * 4

    def test_input_projection_matches_the_declared_count(self):
        cfg = _cfg()
        m = UNet3DDenoiser(cfg)
        assert m.input_proj.in_channels == cfg.in_channels

    def test_availability_embedding_is_per_neighbour_and_per_state(self):
        m = UNet3DDenoiser(_cfg())
        assert m.nb_avail_emb.num_embeddings == N_NEIGHBOURS * N_AVAIL_STATES

    def test_forward_shape(self):
        cfg = _cfg()
        m = UNet3DDenoiser(cfg).eval()
        out = m(**_inputs(cfg))
        assert out.shape == (B, Z, L, L, L)

    @pytest.mark.parametrize("key", ["cond_dist6", "cond_material", "cond_orient",
                                     "cond_por", "cond_depth", "nb_t"])
    def test_every_conditioning_input_reaches_the_output(self, key):
        """A silent no-signal input looks exactly like a working one."""
        cfg = _cfg()
        m = _wake(UNet3DDenoiser(cfg)).eval()
        a = _inputs(cfg)
        base = m(**a)
        b = dict(a)
        if key == "nb_t":
            b[key] = torch.zeros_like(a[key])
        else:
            b[key] = a[key] + 1.0
        assert not torch.allclose(base, m(**b), atol=1e-6)

    def test_non_exists_neighbours_cannot_influence_the_output(self):
        """CFG reuses one nb_latents tensor and flips availability only."""
        cfg = _cfg()
        m = _wake(UNet3DDenoiser(cfg)).eval()
        a = _inputs(cfg)
        a["nb_avail"] = torch.full((B, N_NEIGHBOURS), NB_UNKNOWN)
        base = m(**a)
        b = dict(a, nb_latents=torch.randn_like(a["nb_latents"]))
        assert torch.allclose(base, m(**b), atol=1e-5)

    def test_null_porosity_replaces_the_embedding_only_where_dropped(self):
        cfg = _cfg()
        m = _wake(UNet3DDenoiser(cfg)).eval()
        a = _inputs(cfg)
        none = m(**a)
        drop_none = m(**a, drop_por=torch.zeros(B, dtype=torch.bool))
        drop_all = m(**a, drop_por=torch.ones(B, dtype=torch.bool))
        assert torch.allclose(none, drop_none, atol=1e-6)
        assert not torch.allclose(none, drop_all, atol=1e-6)

    def test_null_porosity_token_receives_gradient(self):
        cfg = _cfg()
        m = _wake(UNet3DDenoiser(cfg))
        out = m(**_inputs(cfg), drop_por=torch.ones(B, dtype=torch.bool))
        out.sum().backward()
        assert m.null_por.grad is not None
        assert float(m.null_por.grad.abs().sum()) > 0.0


# ── neighbour posterior draw ─────────────────────────────────────────────────

class TestSampleNeighbours:
    """The training target is a posterior DRAW, so the neighbours must be too.

    Conditioning on posterior MEANS while regressing a SAMPLE hands the model
    an input whose per-cell variance is short by E[sigma^2] — a mismatch that
    does not exist at generation time, where every neighbour is a real latent.
    """

    @staticmethod
    def _fixture(b: int = 64, std: float = 0.0, seed: int = 0):
        torch.manual_seed(seed)
        nb = torch.randn(b, N_NEIGHBOURS, Z, 4, 4, 4)
        nb_std = torch.full_like(nb, std)
        avail = torch.full((b, N_NEIGHBOURS), NB_EXISTS)
        avail[:, 1] = NB_OOB
        return nb, nb_std, avail

    def test_zero_std_reproduces_the_stored_mean_exactly(self):
        """A degenerate posterior must be bit-identical to the mean it is."""
        nb, nb_std, avail = self._fixture(std=0.0)
        out = sample_neighbours(nb, nb_std, avail)
        assert torch.equal(out, nb)

    def test_variance_is_the_mean_variance_plus_the_expected_square_std(self):
        """Var[mu + sigma*eps] = Var[mu] + E[sigma^2] — the missing term."""
        sigma = 0.4
        nb, nb_std, avail = self._fixture(b=128, std=sigma, seed=3)
        mu_var = float(nb[:, 0].var(unbiased=False))
        draws = torch.stack([
            sample_neighbours(nb, nb_std, avail)[:, 0] for _ in range(16)
        ])
        assert float(draws.var(unbiased=False)) == pytest.approx(
            mu_var + sigma ** 2, rel=0.02
        )

    def test_non_exists_neighbours_are_left_untouched(self):
        """OOB carries no posterior; noise_neighbours zeroes it anyway."""
        nb, nb_std, avail = self._fixture(std=0.5)
        out = sample_neighbours(nb, nb_std, avail)
        assert torch.equal(out[:, 1], nb[:, 1])
        assert not torch.allclose(out[:, 0], nb[:, 0])

    def test_the_draw_is_fresh_on_every_call(self):
        nb, nb_std, avail = self._fixture(std=0.5)
        a = sample_neighbours(nb, nb_std, avail)
        b = sample_neighbours(nb, nb_std, avail)
        assert not torch.allclose(a, b)


# ── neighbour noising ────────────────────────────────────────────────────────

class TestNoiseNeighbours:

    @staticmethod
    def _fixture(b: int = 64, seed: int = 0):
        torch.manual_seed(seed)
        sch = DDPMSchedule(T=T)
        nb = torch.randn(b, N_NEIGHBOURS, Z, 4, 4, 4)
        avail = torch.full((b, N_NEIGHBOURS), NB_EXISTS)
        avail[:, 1] = NB_OOB                        # one face always OOB
        t = torch.randint(1, T, (b,))
        return sch, nb, avail, t

    def test_nb_t_never_exceeds_the_target_timestep(self):
        sch, nb, avail, t = self._fixture()
        _, _, nb_t = noise_neighbours(sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=0.0)
        assert bool((nb_t <= t.view(-1, 1)).all())
        assert bool((nb_t >= 0).all())

    def test_mix_one_pins_every_neighbour_to_the_target_timestep(self):
        """The sampler's own case: canvas and re-noised chunk both sit at t."""
        sch, nb, avail, t = self._fixture()
        _, _, nb_t = noise_neighbours(sch, nb, avail, t, nb_t_mix=1.0, drop_nb_p=0.0)
        ex = avail == NB_EXISTS
        assert bool((nb_t[ex] == t.view(-1, 1).expand_as(nb_t)[ex]).all())

    def test_mix_zero_draws_strictly_below_the_target_most_of_the_time(self):
        sch, nb, avail, t = self._fixture(b=256)
        _, _, nb_t = noise_neighbours(sch, nb, avail, t, nb_t_mix=0.0, drop_nb_p=0.0)
        ex = avail == NB_EXISTS
        below = (nb_t < t.view(-1, 1).expand_as(nb_t))[ex].float().mean()
        assert 0.9 < float(below) <= 1.0

    def test_the_draw_is_per_neighbour_not_per_item(self):
        sch, nb, avail, t = self._fixture(b=64)
        _, _, nb_t = noise_neighbours(sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=0.0)
        ex_rows = nb_t[:, [0, 2, 3, 4, 5]]
        assert int((ex_rows != ex_rows[:, :1]).any(dim=1).sum()) > 0

    def test_mix_is_honoured_on_average(self):
        sch, nb, avail, t = self._fixture(b=512, seed=1)
        _, _, nb_t = noise_neighbours(sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=0.0)
        ex = avail == NB_EXISTS
        at_t = (nb_t == t.view(-1, 1).expand_as(nb_t))[ex].float().mean()
        # p(t_nb == t) = mix + (1 - mix)/(t + 1); with t large the second term
        # is negligible, so the observed rate must sit near the mix.
        assert 0.45 < float(at_t) < 0.58

    def test_non_exists_neighbours_are_zeroed_and_carry_timestep_zero(self):
        sch, nb, avail, t = self._fixture()
        noisy, out_avail, nb_t = noise_neighbours(
            sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=0.0)
        assert float(noisy[:, 1].abs().max()) == 0.0
        assert int(nb_t[:, 1].abs().max()) == 0
        assert bool((out_avail == avail).all())

    def test_exists_neighbours_are_actually_noised(self):
        sch, nb, avail, t = self._fixture()
        noisy, _, _ = noise_neighbours(sch, nb, avail, t, nb_t_mix=1.0, drop_nb_p=0.0)
        assert not torch.allclose(noisy[:, 0], nb[:, 0], atol=1e-3)

    def test_noise_is_fresh_on_every_call(self):
        sch, nb, avail, t = self._fixture()
        a, _, _ = noise_neighbours(sch, nb, avail, t, nb_t_mix=1.0, drop_nb_p=0.0)
        b, _, _ = noise_neighbours(sch, nb, avail, t, nb_t_mix=1.0, drop_nb_p=0.0)
        assert not torch.allclose(a, b)

    def test_dropout_sends_every_face_of_an_item_to_unknown(self):
        """The CFG neighbour null: zero latents, nb_t 0, all six UNKNOWN."""
        sch, nb, avail, t = self._fixture()
        noisy, out_avail, nb_t = noise_neighbours(
            sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=1.0)
        assert bool((out_avail == NB_UNKNOWN).all())
        assert bool((nb_t == 0).all())
        assert float(noisy.abs().max()) == 0.0

    def test_dropout_is_all_or_nothing_per_item(self):
        sch, nb, avail, t = self._fixture(b=256, seed=2)
        _, out_avail, _ = noise_neighbours(
            sch, nb, avail, t, nb_t_mix=0.5, drop_nb_p=0.5)
        dropped = (out_avail == NB_UNKNOWN)
        assert bool((dropped.all(dim=1) | (~dropped).all(dim=1)).all())
        rate = float(dropped[:, 0].float().mean())
        assert 0.4 < rate < 0.6


# ── train / eval step wiring ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    root, *_ = build_store(tmp_path_factory.mktemp("train_store"))
    ds = LatentDataset(root, "train", normalize=True, **dataset_kwargs())
    items = [ds[i] for i in range(0, 4 * 37, 37)]
    keys = ("z", "std", "cond_por", "cond_depth", "cond_dist6", "cond_orient",
            "cond_material", "nb_latents", "nb_std", "nb_avail")
    return {k: torch.stack([it[k] for it in items]) for k in keys}


def _model_for(batch):
    cfg = _cfg(z_channels=batch["z"].shape[1], base_channels=8,
               channel_mult=(1, 2), n_res_blocks=1, cond_embed_dim=16)
    return UNet3DDenoiser(cfg)


class TestSteps:

    def test_train_step_runs_and_moves_the_weights(self, batch):
        model = _wake(_model_for(batch))
        before = model.out_conv.weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out = ldm_train_step(
            model, batch, opt, torch.amp.GradScaler(enabled=False),
            DDPMSchedule(T=T), step=0, device=torch.device("cpu"),
            autocast_dtype=torch.float32, drop_por_p=0.5, drop_nb_p=0.5,
            nb_t_mix=0.5,
        )
        assert np.isfinite(out["loss"]) and np.isfinite(out["grad_norm"])
        assert not torch.allclose(before, model.out_conv.weight)

    def test_eval_step_runs(self, batch):
        model = _model_for(batch)
        out = ldm_eval_step(model, batch, DDPMSchedule(T=T),
                            torch.device("cpu"), autocast_dtype=torch.float32)
        assert np.isfinite(out["loss"])

    def test_a_batch_missing_dist6_is_rejected(self, batch):
        model = _model_for(batch)
        broken = {k: v for k, v in batch.items() if k != "cond_dist6"}
        with pytest.raises(KeyError, match="cond_dist6"):
            ldm_eval_step(model, broken, DDPMSchedule(T=T),
                          torch.device("cpu"), autocast_dtype=torch.float32)

    def test_a_batch_missing_material_is_rejected(self, batch):
        model = _model_for(batch)
        broken = {k: v for k, v in batch.items() if k != "cond_material"}
        with pytest.raises(KeyError, match="cond_material"):
            ldm_eval_step(model, broken, DDPMSchedule(T=T),
                          torch.device("cpu"), autocast_dtype=torch.float32)

    def test_a_batch_missing_neighbour_std_is_rejected(self, batch):
        """Without it the step would silently fall back to mean neighbours."""
        model = _model_for(batch)
        broken = {k: v for k, v in batch.items() if k != "nb_std"}
        with pytest.raises(KeyError, match="nb_std"):
            ldm_eval_step(model, broken, DDPMSchedule(T=T),
                          torch.device("cpu"), autocast_dtype=torch.float32)

    @pytest.mark.parametrize("step_fn", ["train", "eval"])
    def test_both_steps_noise_a_posterior_draw_not_the_stored_mean(
        self, batch, step_fn, monkeypatch
    ):
        """The defect this guards: mean neighbours reaching ``noise_neighbours``."""
        import poregen.training.ldm_engine as engine

        seen: list[torch.Tensor] = []
        real = engine.noise_neighbours

        def spy(schedule, nb_latents, nb_avail, t, **kw):
            seen.append(nb_latents.clone())
            return real(schedule, nb_latents, nb_avail, t, **kw)

        monkeypatch.setattr(engine, "noise_neighbours", spy)
        model = _model_for(batch)
        if step_fn == "train":
            engine.ldm_train_step(
                model, batch, torch.optim.AdamW(model.parameters(), lr=1e-4),
                torch.amp.GradScaler(enabled=False), DDPMSchedule(T=T), step=0,
                device=torch.device("cpu"), autocast_dtype=torch.float32,
                nb_t_mix=1.0,
            )
        else:
            engine.ldm_eval_step(model, batch, DDPMSchedule(T=T),
                                 torch.device("cpu"), autocast_dtype=torch.float32)

        assert len(seen) == 1
        exists = batch["nb_avail"] == NB_EXISTS
        assert bool(exists.any())
        drawn, mu = seen[0][exists], batch["nb_latents"][exists]
        assert not torch.allclose(drawn, mu)
        # The draw is mu shifted by sigma*eps, so it stays close to mu.
        spread = float((drawn - mu).std())
        rms_sigma = float((batch["nb_std"][exists] ** 2).mean().sqrt())
        assert spread == pytest.approx(rms_sigma, rel=0.1)

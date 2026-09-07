"""D43 decoder fine-tune: transfer, freezing, and the refiner's fake branch.

The failure these guard against is silent. A fine-tune whose freeze did not
take still trains, still improves its loss, and still writes a checkpoint — it
just invalidates the 272 GB latent store and the LDM built on it, without
saying so. Likewise a refiner whose bank never reached the discriminator trains
the option-1 objective and reports it under option 2's name.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.configuration import resolve_experiment
from poregen.experiments.train_vae import apply_transfer, build_latent_bank, build_optimizer
from poregen.training.latent_bank import LatentBank


class _Tiny(torch.nn.Module):
    """Same child-module names as the r08 3-class VAE, four orders smaller."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_a = torch.nn.Conv3d(3, 4, 1)
        self.encoder_b = torch.nn.Conv3d(3, 4, 1)
        self.fusion = torch.nn.Conv3d(8, 4, 1)
        self.to_mu = torch.nn.Conv3d(4, 2, 1)
        self.to_logvar = torch.nn.Conv3d(4, 2, 1)
        self.decoder = torch.nn.Conv3d(2, 4, 1)
        self.xct_head = torch.nn.Conv3d(4, 1, 1)
        self.class_head = torch.nn.Conv3d(4, 3, 1)


def _cfg(**training):
    return {"model": {"z_channels": 2}, "training": {"lr": 1e-4, "weight_decay": 0.0, **training}}


class TestFreezing:
    def test_absent_keys_change_nothing(self, tmp_path):
        m = _Tiny()
        assert apply_transfer(_cfg(), m, tmp_path) == {}
        assert all(p.requires_grad for p in m.parameters())

    def test_freezing_leaves_only_the_decoder_trainable(self, tmp_path):
        m = _Tiny()
        info = apply_transfer(
            _cfg(freeze_modules=["encoder_a", "encoder_b", "fusion", "to_mu",
                                 "to_logvar", "class_head"]), m, tmp_path)
        assert not any(p.requires_grad for p in m.encoder_a.parameters())
        assert not any(p.requires_grad for p in m.class_head.parameters())
        assert all(p.requires_grad for p in m.decoder.parameters())
        assert info["trainable_params"] == sum(
            p.numel() for p in list(m.decoder.parameters()) + list(m.xct_head.parameters()))

    def test_a_misspelled_module_raises(self, tmp_path):
        # The dangerous outcome is freezing NOTHING and training the encoder,
        # which invalidates the latent store without any visible symptom.
        with pytest.raises(ValueError, match="no such submodule"):
            apply_transfer(_cfg(freeze_modules=["encoder"]), _Tiny(), tmp_path)

    def test_freezing_everything_raises(self, tmp_path):
        with pytest.raises(ValueError, match="nothing trainable"):
            apply_transfer(_cfg(freeze_modules=[
                "encoder_a", "encoder_b", "fusion", "to_mu", "to_logvar",
                "decoder", "xct_head", "class_head"]), _Tiny(), tmp_path)

    def test_the_optimizer_never_sees_a_frozen_parameter(self, tmp_path):
        m = _Tiny()
        cfg = _cfg(freeze_modules=["encoder_a", "encoder_b", "fusion", "to_mu", "to_logvar"])
        apply_transfer(cfg, m, tmp_path)
        seen = {id(p) for g in build_optimizer(cfg, m).param_groups for p in g["params"]}
        assert not (seen & {id(p) for p in m.encoder_a.parameters()})
        assert {id(p) for p in m.decoder.parameters()} <= seen

    def test_a_frozen_encoder_does_not_move_across_an_optimiser_step(self, tmp_path):
        m = _Tiny()
        cfg = _cfg(freeze_modules=["encoder_a", "encoder_b", "fusion", "to_mu", "to_logvar"])
        apply_transfer(cfg, m, tmp_path)
        opt = build_optimizer(cfg, m)
        before = m.encoder_a.weight.detach().clone()
        x = torch.randn(2, 3, 4, 4, 4)
        h = m.fusion(torch.cat([m.encoder_a(x), m.encoder_b(x)], dim=1))
        m.xct_head(m.decoder(m.to_mu(h))).sum().backward()
        opt.step()
        assert torch.equal(m.encoder_a.weight, before)


class TestInitFromCheckpoint:
    def test_weights_are_loaded(self, tmp_path):
        from poregen.training.checkpoint import save_checkpoint
        src = _Tiny()
        with torch.no_grad():
            src.decoder.weight.fill_(0.5)
        ck = tmp_path / "src.ckpt"
        save_checkpoint(str(ck), model=src,
                        optimizer=torch.optim.SGD(src.parameters(), lr=0.0),
                        scaler=torch.amp.GradScaler(enabled=False), step=7)
        dst = _Tiny()
        info = apply_transfer(_cfg(init_from_checkpoint=str(ck)), dst, tmp_path)
        assert torch.allclose(dst.decoder.weight, src.decoder.weight)
        assert info["init_from_step"] == 7

    def test_a_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            apply_transfer(_cfg(init_from_checkpoint=str(tmp_path / "nope.ckpt")),
                           _Tiny(), tmp_path)

    def test_resume_reapplies_the_freeze_but_not_the_init(self, tmp_path):
        from poregen.training.checkpoint import save_checkpoint
        src = _Tiny()
        ck = tmp_path / "src.ckpt"
        save_checkpoint(str(ck), model=src,
                        optimizer=torch.optim.SGD(src.parameters(), lr=0.0),
                        scaler=torch.amp.GradScaler(enabled=False), step=3)
        m = _Tiny()
        with torch.no_grad():
            m.decoder.weight.fill_(9.0)          # stands in for resumed progress
        info = apply_transfer(
            _cfg(init_from_checkpoint=str(ck), freeze_modules=["encoder_a"]),
            m, tmp_path, init=False)
        assert torch.all(m.decoder.weight == 9.0), "resume must not reload the parent"
        assert not any(p.requires_grad for p in m.encoder_a.parameters())
        assert "init_from_checkpoint" not in info


class TestAdversarialSource:
    def test_an_unknown_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a known source"):
            apply_transfer(_cfg(adversarial_source="whatever"), _Tiny(), tmp_path)

    def test_ldm_latents_without_a_bank_raises(self, tmp_path):
        # Without this the refiner would train option 1 under option 2's name.
        with pytest.raises(ValueError, match="latent_bank_root"):
            apply_transfer(_cfg(adversarial_source="ldm_latents"), _Tiny(), tmp_path)

    def test_recon_is_the_default_and_builds_no_bank(self, tmp_path):
        assert build_latent_bank(_cfg(), tmp_path) is None

    def test_a_bank_of_the_wrong_width_raises(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        np.save(tmp_path / "a" / "b" / "latents.npy", np.zeros((4, 16, 16, 16), np.float16))
        cfg = _cfg(adversarial_source="ldm_latents", latent_bank_root=str(tmp_path))
        with pytest.raises(ValueError, match="channel but the model is z=2"):
            build_latent_bank(cfg, tmp_path)


class TestLatentBank:
    def _bank(self, tmp_path, shape=(2, 32, 32, 32)):
        (tmp_path / "c" / "d").mkdir(parents=True)
        f = tmp_path / "c" / "d" / "latents.npy"
        np.save(f, np.random.randn(*shape).astype(np.float16))
        return f

    def test_windows_are_the_decoder_patch(self, tmp_path):
        b = LatentBank([self._bank(tmp_path)])
        assert len(b) == 8                       # 2x2x2 non-overlapping
        assert tuple(b[0].shape) == (2, 16, 16, 16)
        assert b[0].dtype is torch.float32

    def test_mixed_latent_widths_are_refused(self, tmp_path):
        a = self._bank(tmp_path)
        other = tmp_path / "other.npy"
        np.save(other, np.zeros((8, 16, 16, 16), np.float16))
        with pytest.raises(ValueError, match="must come from the same VAE"):
            LatentBank([a, other])

    def test_a_canvas_too_small_for_one_window_yields_an_empty_bank(self, tmp_path):
        small = tmp_path / "s.npy"
        np.save(small, np.zeros((2, 4, 4, 4), np.float16))
        with pytest.raises(ValueError, match="no canvas held a full"):
            LatentBank([small])


class TestRefinerFakeBranch:
    """The one behaviour that separates option 2 from option 1."""

    def test_the_fake_branch_comes_from_the_bank_not_the_reconstruction(self):
        from poregen.training.engine import train_step
        seen: dict[str, torch.Tensor] = {}

        class _D(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                seen.setdefault("first", x.detach().clone())
                return x.mean(dim=(1, 2, 3), keepdim=True) + self.p

        class _M(torch.nn.Module):
            encoder_inputs = ("xct",)

            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Conv3d(2, 4, 1)
                self.xct_head = torch.nn.Conv3d(4, 1, 1)
                with torch.no_grad():
                    self.xct_head.weight.fill_(0.0)
                    self.xct_head.bias.fill_(-3.0)   # a value no real patch has

            def forward(self, xct):
                from poregen.models.vae.base import VAEOutput
                b = xct.shape[0]
                mu = torch.zeros(b, 2, 1, 1, 1, requires_grad=True)
                return VAEOutput(xct_out=torch.zeros_like(xct), class_logits=None,
                                 mu=mu, logvar=torch.zeros_like(mu),
                                 mask_logits=None, z=mu)

        m = _M()
        d = _D()
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        dopt = torch.optim.SGD(d.parameters(), lr=0.0)
        batch = {"xct": torch.ones(2, 1, 4, 4, 4)}
        z = torch.zeros(2, 2, 4, 4, 4)
        train_step(
            m, batch, opt, torch.amp.GradScaler(enabled=False),
            lambda o, b, s: {"total": o.mu.sum()},
            step=0, device=torch.device("cpu"), autocast_dtype=torch.bfloat16,
            discriminator=d, disc_optimizer=dopt, disc_weight=0.1,
            adv_fake_latents=z,
        )
        # xct_head is all-zero weight with bias -3, so anything routed through
        # the DECODER is -3. The reconstruction path would be 0.0 (xct_out is
        # zeros) and the real patch is 1.0. Seeing -3 proves the bank was used.
        assert seen["first"].mean().item() == pytest.approx(-3.0, abs=1e-3)


def test_the_shipped_configs_resolve_and_agree_with_the_store():
    ft = resolve_experiment("r08/decoder-ft").cfg
    assert ft["model"]["z_channels"] == 8
    assert ft["training"]["freeze_modules"][:2] == ["encoder_a", "encoder_b"]
    # The class losses MUST stay on: xct_head and class_head share the decoder
    # trunk, so a frozen class_head does not hold segmentation still by itself.
    assert ft["loss"]["class_ce_weight"] > 0
    assert ft["loss"]["class_dice_weight"] > 0
    assert ft["discriminator"]["disc_weight"] == pytest.approx(0.2)

    ref = resolve_experiment("r08/decoder-ft-refiner").cfg
    assert ref["training"]["adversarial_source"] == "ldm_latents"
    assert ref["training"]["latent_bank_root"]
    assert ref["training"]["freeze_modules"] == ft["training"]["freeze_modules"]

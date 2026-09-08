"""Checkpoint snapshotting, writer failures, and discriminator resumption.

Three failures guarded here are all silent. An async save handed the LIVE state
dict writes a file that mixes the step it is named after with whatever the
optimizer did while the writer was serialising. A writer thread that dies takes
the checkpoint with it and says nothing. A resume that rebuilds a fresh
discriminator restarts the GAN on every interruption, and the reconstruction
losses look exactly the same either way.
"""

from __future__ import annotations

import threading

import pytest
import torch

from poregen.experiments.train_vae import _prepare_resume_state, build_discriminator
from poregen.training.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_async


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def _scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler(enabled=False)


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """One optimizer step — moves the weights and fills the Adam moments."""
    optimizer.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()


@pytest.fixture
def gated_save(monkeypatch):
    """Hold the background writer inside ``torch.save`` until released."""
    gate = threading.Event()
    entered = threading.Event()
    real_save = torch.save

    def _save(obj, f, *args, **kwargs):
        entered.set()
        assert gate.wait(30), "writer was never released"
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", _save)
    gate.entered = entered
    return gate


class TestAsyncSnapshot:
    def test_weights_changed_after_the_call_do_not_reach_the_file(
        self, tmp_path, gated_save
    ):
        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        holder: list = [None]

        save_checkpoint_async(
            tmp_path / "ckpt.ckpt", model, optimizer, _scaler(),
            step=10, thread_holder=holder,
        )
        assert gated_save.entered.wait(30)
        with torch.no_grad():                      # a later training step
            model.weight.fill_(7.0)
        gated_save.set()
        holder[0].join()

        state = torch.load(tmp_path / "ckpt.ckpt", weights_only=False)
        assert torch.all(state["model"]["weight"] == 1.0), (
            "the writer saw the live tensor, so step 10's file holds a later state"
        )

    def test_optimizer_moments_are_snapshotted_too(self, tmp_path, gated_save):
        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        _step(model, optimizer)                    # gives Adam its first moments
        saved_moment = optimizer.state_dict()["state"][0]["exp_avg"].clone()
        holder: list = [None]

        save_checkpoint_async(
            tmp_path / "ckpt.ckpt", model, optimizer, _scaler(),
            step=1, thread_holder=holder,
        )
        assert gated_save.entered.wait(30)
        _step(model, optimizer)                    # moments move under the writer
        gated_save.set()
        holder[0].join()

        state = torch.load(tmp_path / "ckpt.ckpt", weights_only=False)
        assert torch.allclose(state["optimizer"]["state"][0]["exp_avg"], saved_moment)


class TestWriterFailures:
    def test_a_failed_write_raises_on_join(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            torch, "save",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no space left")),
        )
        model = _model()
        holder: list = [None]
        save_checkpoint_async(
            tmp_path / "ckpt.ckpt", model,
            torch.optim.Adam(model.parameters(), lr=0.1), _scaler(),
            step=1, thread_holder=holder,
        )
        with pytest.raises(RuntimeError, match="no space left"):
            holder[0].join()

    def test_a_failed_write_raises_from_the_next_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            torch, "save",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no space left")),
        )
        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        holder: list = [None]
        save_checkpoint_async(
            tmp_path / "a.ckpt", model, optimizer, _scaler(),
            step=1, thread_holder=holder,
        )
        with pytest.raises(RuntimeError, match="no space left"):
            save_checkpoint_async(
                tmp_path / "b.ckpt", model, optimizer, _scaler(),
                step=2, thread_holder=holder,
            )


def _disc_cfg(total_steps: int = 100) -> dict:
    return {
        "discriminator": {"enabled": True, "base_channels": 4},
        "training": {"total_steps": total_steps},
        "runtime": {"resume": {"prune_jsonl": False}},
    }


class TestDiscriminatorRoundTrip:
    def test_save_and_load_restore_the_discriminator(self, tmp_path):
        cfg = _disc_cfg()
        device = torch.device("cpu")
        disc, disc_opt, _ = build_discriminator(cfg, device)
        disc_opt.zero_grad()
        disc(torch.randn(2, 1, 16, 16)).sum().backward()
        disc_opt.step()

        model = _model()
        ckpt = save_checkpoint(
            tmp_path / "ckpt.ckpt", model,
            torch.optim.Adam(model.parameters(), lr=0.1), _scaler(),
            step=5, discriminator=disc, disc_optimizer=disc_opt,
        )

        fresh, fresh_opt, _ = build_discriminator(cfg, device)
        assert not all(
            torch.equal(a, b) for a, b in zip(fresh.state_dict().values(),
                                              disc.state_dict().values())
        ), "a fresh discriminator must differ, or the test proves nothing"

        load_checkpoint(
            ckpt, model=_model(), discriminator=fresh, disc_optimizer=fresh_opt,
            restore_rng=False,
        )
        for key, value in disc.state_dict().items():
            assert torch.equal(fresh.state_dict()[key], value)
        assert fresh_opt.state_dict()["state"], "optimizer moments were not restored"
        for pid, entry in disc_opt.state_dict()["state"].items():
            assert torch.allclose(fresh_opt.state_dict()["state"][pid]["exp_avg"],
                                  entry["exp_avg"])

    def test_resume_restores_the_discriminator(self, tmp_path):
        cfg = _disc_cfg()
        device = torch.device("cpu")
        disc, disc_opt, _ = build_discriminator(cfg, device)
        disc_opt.zero_grad()
        disc(torch.randn(2, 1, 16, 16)).sum().backward()
        disc_opt.step()

        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        save_checkpoint(
            tmp_path / "latest.ckpt", model, optimizer, _scaler(),
            step=5, discriminator=disc, disc_optimizer=disc_opt,
        )

        resumed_disc, resumed_opt, _ = build_discriminator(cfg, device)
        resumed_model = _model()
        start_step, remaining = _prepare_resume_state(
            cfg=cfg,
            run_dir=tmp_path,
            checkpoint_path=tmp_path / "latest.ckpt",
            model=resumed_model,
            optimizer=torch.optim.Adam(resumed_model.parameters(), lr=0.1),
            scaler=_scaler(),
            scheduler=None,
            device=device,
            discriminator=resumed_disc,
            disc_optimizer=resumed_opt,
        )
        assert (start_step, remaining) == (5, 95)
        for key, value in disc.state_dict().items():
            assert torch.equal(resumed_disc.state_dict()[key], value), (
                f"{key} restarted from scratch on resume"
            )
        assert resumed_opt.state_dict()["state"], "disc optimizer restarted from scratch"

    def test_a_checkpoint_without_discriminator_state_warns(self, tmp_path, caplog):
        model = _model()
        ckpt = save_checkpoint(
            tmp_path / "ckpt.ckpt", model,
            torch.optim.Adam(model.parameters(), lr=0.1), _scaler(), step=1,
        )
        disc, disc_opt, _ = build_discriminator(_disc_cfg(), torch.device("cpu"))
        with caplog.at_level("WARNING"):
            load_checkpoint(ckpt, model=_model(), discriminator=disc,
                            disc_optimizer=disc_opt, restore_rng=False)
        assert "no discriminator state" in caplog.text

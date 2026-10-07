"""A latent store and an LDM run are bound to the VAE WEIGHTS, not its path."""
from __future__ import annotations

import json

import pytest
import torch

from poregen.eval_v4.generate import check_run_store_binding
from poregen.experiments.train_ldm import check_vae_weights
from poregen.training.checkpoint import checkpoint_identity


def _ckpt(path, step, value):
    torch.save({"step": step, "model": {"w": torch.full((3,), float(value))}}, path)
    return path


def test_identity_is_the_file_contents_and_its_step(tmp_path):
    a = checkpoint_identity(_ckpt(tmp_path / "best.ckpt", 26620, 1.0))
    assert a["step"] == 26620 and len(a["sha256"]) == 64
    # same path, new weights: a resumed run rewrites best.ckpt in place
    b = checkpoint_identity(_ckpt(tmp_path / "best.ckpt", 36720, 2.0))
    assert b["sha256"] != a["sha256"] and b["step"] == 36720


def test_a_store_refuses_the_rewritten_checkpoint(tmp_path):
    ck = _ckpt(tmp_path / "best.ckpt", 26620, 1.0)
    meta = {"vae_checkpoint": str(ck), **{f"vae_checkpoint_{k}": v
            for k, v in checkpoint_identity(ck).items()}}
    check_vae_weights(ck, meta)                       # the weights that built it
    _ckpt(ck, 36720, 2.0)
    with pytest.raises(RuntimeError, match="weights mismatch"):
        check_vae_weights(ck, meta)
    with pytest.raises(RuntimeError, match="records no vae_checkpoint_sha256"):
        check_vae_weights(ck, {"vae_checkpoint": str(ck)})


def test_an_ldm_run_refuses_a_store_rebuilt_under_it(tmp_path):
    run = tmp_path / "ldm25-run"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"vae_checkpoint_sha256": "a" * 64}))
    check_run_store_binding(run, {"vae_checkpoint_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="cannot be decoded from this store"):
        check_run_store_binding(run, {"vae_checkpoint_sha256": "b" * 64,
                                      "vae_checkpoint_step": 36720})
    (run / "run_metadata.json").write_text("{}")
    with pytest.raises(RuntimeError, match="records no vae_checkpoint_sha256"):
        check_run_store_binding(run, {"vae_checkpoint_sha256": "a" * 64})

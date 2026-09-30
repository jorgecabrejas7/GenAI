"""The smoke step must call a VAE the way the trainer does.

It called model(xct) directly, and the paper's own VAE — the r08 3-class model,
whose encoder takes (xct, label) — raised before measuring anything. The v4
rebuild's first stage died on it. These tests run the smoke's batch and call
through every VAE family the chains smoke, on CPU, so that cannot recur.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location(
        "vae_smoke_step", REPO / "scripts" / "analysis" / "vae_smoke_step.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("experiment", [
    "r08/reduction-factor-8",          # the paper's VAE: (xct, label)
    "r08_grey/reduction-factor-8",     # grey only
    "vrrae/conv_k8",                   # (xct, mask)
])
def test_one_step_runs_through_the_trainers_contract(smoke, experiment):
    import copy
    import logging

    logging.disable(logging.WARNING)
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import build_model
    from poregen.losses import compute_total_loss
    from poregen.training.engine import to_device_inputs

    cfg = copy.deepcopy(resolve_experiment(experiment).cfg)
    dev = torch.device("cpu")
    model = build_model(cfg, dev)
    model.train()
    batch = smoke.synthetic_batch(2, int(cfg["model"]["patch_size"]), dev)
    _, args = to_device_inputs(model, batch, dev)
    out = model(*args)
    loss = compute_total_loss(out, batch, 0, cfg)["total"]
    assert torch.isfinite(loss)
    loss.backward()


def test_the_batch_carries_every_key_a_forward_or_loss_reads(smoke):
    b = smoke.synthetic_batch(2, 64, torch.device("cpu"))
    assert set(b) == {"xct", "mask", "label"}
    assert b["label"].dtype == torch.long and b["label"].shape == (2, 64, 64, 64)
    assert int(b["label"].max()) <= 2

"""A VAE resume can continue a run past the rule it stopped on, on record."""
from __future__ import annotations

import copy

from poregen.experiments.train_vae import (
    apply_resume_overrides,
    resolve_early_stopping_patience,
)

CFG = {"training": {"total_steps": 71690, "eval_every": 121,
                    "early_stopping_patience": 0,
                    "early_stopping_patience_steps": 3000}}


def test_no_override_changes_nothing():
    cfg = copy.deepcopy(CFG)
    assert apply_resume_overrides(cfg) == {}
    assert cfg == CFG
    assert resolve_early_stopping_patience(cfg) == 25


def test_overrides_set_the_target_and_turn_early_stopping_off():
    cfg = copy.deepcopy(CFG)
    changed = apply_resume_overrides(cfg, total_steps=36720, no_early_stopping=True)
    assert cfg["training"]["total_steps"] == 36720
    # 0 patience is "off" in the engine (early_stopping_patience <= 0)
    assert resolve_early_stopping_patience(cfg) == 0
    assert changed == {
        "training.total_steps": [71690, 36720],
        "training.early_stopping_patience_steps": [3000, 0],
        "training.early_stopping_patience": [0, 0],
    }

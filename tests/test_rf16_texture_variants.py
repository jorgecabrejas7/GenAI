"""The two rf-16 texture variants change one lever each, on split_v4."""
from __future__ import annotations

import pytest

from poregen import paths
from poregen.configuration.experiments import resolve_experiment
from poregen.runtime.runs import build_run_name

BASE = "r08/base"
DISC = "r08/reduction-factor-16-disc0.2"
BETA = "r08/reduction-factor-16-beta0.02"


@pytest.fixture
def v4(monkeypatch):
    if not (paths.data_root("split_v4") / "class_weights.json").exists():
        pytest.skip("split_v4 not present")
    monkeypatch.setenv(paths.SPLIT_ENV, "split_v4")


def _diff(a, b, path=""):
    out = {}
    for k in set(a) | set(b):
        p = f"{path}.{k}" if path else k
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out.update(_diff(a[k], b[k], p))
        elif a.get(k) != b.get(k):
            out[p] = (a.get(k), b.get(k))
    return out


def test_each_variant_changes_its_one_lever(v4):
    base = resolve_experiment(BASE).cfg
    d = _diff(base, resolve_experiment(DISC).cfg)
    d.pop("experiment.variant"); d.pop("experiment.description")
    assert d.pop("discriminator.disc_weight") == (0.05, 0.2)
    assert set(d) == {"runtime.run_name.fields"}          # the name only
    b = _diff(base, resolve_experiment(BETA).cfg)
    b.pop("experiment.variant"); b.pop("experiment.description")
    assert b == {"loss.kl_max_beta": (0.05, 0.02)}


@pytest.mark.parametrize("exp", [DISC, BETA])
def test_both_run_on_split_v4_as_the_rungs_do(v4, exp):
    c = resolve_experiment(exp).cfg
    assert c["data"]["dataset_root"] == "split_v4" and c["data"]["pin_memory"] is True
    assert c["model"]["z_channels"] == 4
    assert c["loss"]["class_weights_source"]["file"] == "data/split_v4/class_weights.json"


def test_the_run_names_tell_the_variants_from_run_0014(v4):
    base = build_run_name(resolve_experiment(BASE).cfg, run_index=14)
    disc = build_run_name(resolve_experiment(DISC).cfg, run_index=20)
    beta = build_run_name(resolve_experiment(BETA).cfg, run_index=21)
    assert "-dw0.2-" in disc and "-dw" not in base
    assert "-b0.020-" in beta and "-b0.050-" in base
    assert disc.endswith("-dsv4") and beta.endswith("-dsv4")

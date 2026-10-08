"""One switch decides which dataset split the pipeline reads.

The project has been rebuilt three times and is about to be rebuilt again.
Each rebuild, the split name lived in configs, script defaults, analysis
constants and test fixtures separately — and a missed one does not fail, it
quietly measures the old data. These tests are what stops that.
"""
from __future__ import annotations

import importlib
import os

import pytest

from poregen import paths


@pytest.fixture
def forced(monkeypatch):
    def _set(name: str):
        monkeypatch.setenv(paths.SPLIT_ENV, name)
    return _set


class TestTheSwitchItself:

    def test_the_default_is_the_published_split(self):
        """It does NOT follow the newest build: a rerun of a published number
        must reproduce that number, not re-measure on new data."""
        assert paths.DEFAULT_SPLIT == "split_v3"

    def test_the_environment_redirects_it(self, forced):
        forced("split_v4")
        assert paths.default_split() == "split_v4"
        assert paths.data_root().name == "split_v4"

    def test_an_explicit_argument_beats_the_environment(self, forced):
        """A caller that names a split means it."""
        forced("split_v4")
        assert paths.data_root("split_v2").name == "split_v2"

    def test_an_empty_variable_is_not_a_split(self, monkeypatch):
        monkeypatch.setenv(paths.SPLIT_ENV, "  ")
        assert paths.default_split() == paths.DEFAULT_SPLIT
        assert paths.split_override() is None

    def test_a_latent_store_follows_its_split(self, forced):
        forced("split_v4")
        assert paths.latents_root("latents_r08z8").parts[-2:] == (
            "split_v4", "latents_r08z8")


class TestConfigsFollowTheSwitch:
    """A switch a config could silently opt out of would not be a switch."""

    @staticmethod
    def _data(exp):
        import poregen.configuration.experiments as E
        return E.resolve_experiment(exp).cfg["data"]

    def test_a_vae_config_moves_its_dataset_root(self, forced):
        assert self._data("r08/base")["dataset_root"] == "split_v3"
        forced("split_v4")
        assert self._data("r08/base")["dataset_root"] == "split_v4"

    def test_an_ldm_config_moves_its_latent_store(self, forced):
        """LDM configs have no dataset_root at all — they are keyed on the
        store, and a store belongs to the split it was encoded from."""
        before = self._data("ldm06/base")["latents_root"]
        assert "split_v3" in before
        forced("split_v4")
        assert self._data("ldm06/base")["latents_root"] == \
            before.replace("split_v3", "split_v4")

    def test_nothing_is_invented(self, forced):
        """An LDM config must not acquire a dataset_root it never reads."""
        forced("split_v4")
        assert "dataset_root" not in self._data("ldm06/base")

    def test_the_default_is_untouched(self):
        assert self._data("r08/base")["dataset_root"] == "split_v3"
        assert "split_v3" in self._data("ldm06/base")["latents_root"]


class TestNoScriptKeepsItsOwnCopy:

    def test_the_analysis_constants_follow_the_switch(self, forced):
        """These were `REPO / "data" / "split_v3"` written out by hand."""
        forced("split_v4")
        for mod in ("scripts.analysis.label_uncertainty",):
            pass  # scripts/ is not a package; covered by the source scan below
        import poregen.baselines.slicegan.data as sg
        importlib.reload(sg)
        assert sg.DATA_ROOT.name == "split_v4"

    def test_no_default_names_the_split_directly(self):
        """Prose may mention split_v3 — it records what was measured, and a
        script whose whole job is one split may name it. What may NOT name it
        is a DEFAULT: a module-level constant or an argparse default is a copy
        of the switch, and a copy is what gets missed at the next rebuild.
        """
        import ast
        from pathlib import Path

        # Scripts whose subject IS a particular split. Naming it is the point.
        ALLOWED = {
            "paths.py", "build_split_v3.py", "build_split_v4.py",
            "verify_split_v3_reference.py", "dataset_v4_vvf_table.py",
            "split_v3_reproducibility.py",
            "reference_onlypores_audit.py",
        }
        repo = paths.repo_root()
        offenders = []
        for f in list((repo / "src").rglob("*.py")) + list((repo / "scripts").rglob("*.py")):
            if f.name in ALLOWED:
                continue
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue

            def names_split(node) -> bool:
                return any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                           and "split_v3" in n.value for n in ast.walk(node))

            for node in tree.body:                       # module level only
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and names_split(node):
                    offenders.append(f"{f.relative_to(repo)}:{node.lineno}")
            for node in ast.walk(tree):                  # argparse defaults
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "add_argument":
                    for kw in node.keywords:
                        if kw.arg == "default" and names_split(kw.value):
                            offenders.append(f"{f.relative_to(repo)}:{node.lineno}")
        assert not offenders, (
            "these DEFAULTS name the split directly instead of asking "
            "poregen.paths: " + ", ".join(sorted(set(offenders))))


class TestARunOnAnotherSplitSaysSo:
    """A split_v4 rf-8 and the split_v3 rf-8 behind the paper would otherwise
    have names differing only in index and timestamp, and a glob for "the rf-8
    run" would take whichever is newer."""

    @staticmethod
    def _name(exp):
        import poregen.configuration.experiments as E
        from poregen.runtime.runs import build_run_name
        return build_run_name(E.resolve_experiment(exp).cfg, run_index=1)

    def test_a_published_split_run_is_named_as_before(self):
        """No existing run is renamed, so every resume still parses."""
        assert "dsv" not in self._name("r08/reduction-factor-8")
        assert "dsv" not in self._name("ldm06/base")

    def test_a_vae_on_split_v4_carries_it(self, forced):
        forced("split_v4")
        assert self._name("r08/reduction-factor-8").endswith("-dsv4")

    def test_an_ldm_on_split_v4_carries_it_from_its_latent_store(self, forced):
        """An LDM has no dataset_root; its split is inside latents_root."""
        forced("split_v4")
        assert self._name("ldm06/facedrop_from_start").endswith("-dsv4")

    def test_the_family_prefix_does_not_move(self, forced):
        """tb-vae and tb-ldm group by the leading experiment name."""
        forced("split_v4")
        assert self._name("r08/reduction-factor-8").startswith("r08-run-0001-")


class TestTheVaeFollowsTheRebuild:
    """train_ldm REFUSES to start when cfg['vae']['checkpoint'] differs from
    the store's recorded encoder. A rebuild trains a new VAE, so without this
    the v4 LDM stage would stop ~36 hours in, after the VAE and the store."""

    @staticmethod
    def _vae(exp):
        import poregen.configuration.experiments as E
        return E.resolve_experiment(exp).cfg.get("vae", {}).get("checkpoint")

    def test_unset_leaves_the_published_vae(self):
        assert "r08-run-0004" in self._vae("ldm06/facedrop_from_start")

    def test_set_redirects_every_ldm(self, monkeypatch):
        monkeypatch.setenv("POREGEN_VAE_CHECKPOINT", "runs/vae/new/best.ckpt")
        assert self._vae("ldm06/facedrop_from_start") == "runs/vae/new/best.ckpt"
        assert self._vae("ldm25/phi_only") == "runs/vae/new/best.ckpt"

    def test_a_vae_config_acquires_no_vae_key(self, monkeypatch):
        monkeypatch.setenv("POREGEN_VAE_CHECKPOINT", "runs/vae/new/best.ckpt")
        assert self._vae("r08/reduction-factor-8") is None

    def test_it_works_without_the_split_switch(self, monkeypatch):
        """The two are independent: one must not need the other to be set."""
        monkeypatch.delenv("POREGEN_SPLIT", raising=False)
        monkeypatch.setenv("POREGEN_VAE_CHECKPOINT", "runs/vae/new/best.ckpt")
        assert self._vae("ldm06/facedrop_from_start") == "runs/vae/new/best.ckpt"


# ── analysis outputs carry the split, as run names do ────────────────────────

def test_split_tag_is_empty_only_for_the_published_split():
    from poregen.paths import DEFAULT_SPLIT, split_tag
    assert split_tag(DEFAULT_SPLIT) == ""
    assert split_tag("split_v4") == "dsv4"
    assert split_tag("data/split_v5") == "dsv5"


def test_std_reference_is_one_file_per_split():
    from poregen.paths import repo_root, std_reference_path
    camp = repo_root() / "runs/campaigns/09-r08-latent-sweep"
    assert std_reference_path("data/split_v3/latents_r08z8") \
        == camp / "latent_std_reference.json"
    assert std_reference_path("data/split_v4/latents_r08z8") \
        == camp / "latent_std_reference-dsv4.json"


def test_rung_report_directory_keeps_split_v3_names():
    import importlib.util
    from poregen.paths import repo_root
    spec = importlib.util.spec_from_file_location(
        "r08_rung_report", repo_root() / "scripts/analysis/r08_rung_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.out_name("r08_reduction-factor-8", "split_v3") == "r08_reduction-factor-8"
    assert mod.out_name("r08_reduction-factor-8", "split_v4") \
        == "r08_reduction-factor-8-dsv4"


# ── the loss weights are the active split's own ──────────────────────────────

class TestClassWeightsFollowTheSplit:
    """sqrt-inverse weights of the split's OWN train frequencies, never a copy.

    r08/base once carried split_v3's weights as a literal, and the first
    split_v4 VAE trained with them (found 2026-10-08).
    """

    @pytest.mark.parametrize("split", ["split_v3", "split_v4"])
    def test_resolved_weights_are_the_split_file(self, monkeypatch, split):
        import json
        from poregen.configuration.experiments import resolve_experiment

        f = paths.data_root(split) / "class_weights.json"
        if not f.exists():
            pytest.skip(f"{f} not present")
        monkeypatch.setenv(paths.SPLIT_ENV, split)
        loss = resolve_experiment("r08/reduction-factor-8").cfg["loss"]
        assert loss["class_weights"] == json.loads(f.read_text())["class_weights"]
        assert loss["class_weights_source"]["file"] == f"data/{split}/class_weights.json"

    def test_no_experiment_config_carries_a_weight_literal(self):
        import re
        offenders = []
        for f in (paths.repo_root() / "configs").rglob("*.yaml"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if re.match(r"\s*class_weights:\s*\[", line):
                    offenders.append(f"{f.relative_to(paths.repo_root())}:{i}")
        assert not offenders, "class weights must come from the split: " + ", ".join(offenders)

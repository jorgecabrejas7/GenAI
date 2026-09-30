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
        assert self._name("r08/reduction-factor-8_v4").endswith("-dsv4")

    def test_an_ldm_on_split_v4_carries_it_from_its_latent_store(self, forced):
        """An LDM has no dataset_root; its split is inside latents_root."""
        forced("split_v4")
        assert self._name("ldm06/facedrop_from_start").endswith("-dsv4")

    def test_the_family_prefix_does_not_move(self, forced):
        """tb-vae and tb-ldm group by the leading experiment name."""
        forced("split_v4")
        assert self._name("r08/reduction-factor-8_v4").startswith("r08-run-0001-")

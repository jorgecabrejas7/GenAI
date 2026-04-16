from __future__ import annotations

from datetime import datetime

from poregen.configuration import clone_experiment_definition, resolve_experiment
from poregen.runtime.runs import build_run_name


def test_resolve_r03_base_experiment() -> None:
    resolved = resolve_experiment("r03/base")

    assert resolved.experiment_id == "r03/base"
    assert resolved.cfg["experiment"]["name"] == "r03"
    assert resolved.cfg["experiment"]["variant"] == "base"
    assert resolved.cfg["model"]["name"] == "v2.conv_noattn"
    assert resolved.cfg["model"]["z_channels"] == 16
    assert resolved.cfg["data"]["dataset_root"] == "split_v2"
    assert resolved.cfg["runtime"]["checkpoints"]["best_metric"] == "val_full.total"


def test_resolve_extended_experiment_variant() -> None:
    resolved = resolve_experiment("r03/split_v1")

    assert resolved.cfg["experiment"]["variant"] == "split_v1"
    assert resolved.cfg["data"]["dataset_root"] == "split_v1"
    assert resolved.cfg["training"]["val_batches"] == 100
    assert resolved.cfg["model"]["z_channels"] == 16


def test_build_run_name_uses_configured_tokens() -> None:
    resolved = resolve_experiment("r03/base")
    run_name = build_run_name(
        resolved.cfg,
        run_index=7,
        when=datetime(2026, 4, 15, 14, 30, 0),
    )

    assert run_name.startswith("r03-run-0007-20260415-143000-")
    assert "archv2-conv_noattn" in run_name
    assert "z16" in run_name
    assert "c32" in run_name
    assert "schednone" in run_name


def test_clone_experiment_creates_extending_yaml(tmp_path) -> None:
    repo_root = tmp_path
    (repo_root / "src" / "poregen").mkdir(parents=True)
    configs_dir = repo_root / "configs" / "experiments" / "r03"
    configs_dir.mkdir(parents=True)
    source_path = configs_dir / "base.yaml"
    source_path.write_text(
        "experiment:\n"
        "  name: r03\n"
        "  variant: base\n"
        "overrides: {}\n"
    )

    target_path = clone_experiment_definition(
        source_path,
        "r03/z24",
        repo_root=repo_root,
    )

    text = target_path.read_text()
    assert "extends: r03/base" in text
    assert "variant: z24" in text

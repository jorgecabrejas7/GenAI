"""Config-driven experiment discovery and resolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from poregen.configs.config import parse_config
from poregen.experiments.base import find_repo_root


@dataclass(frozen=True)
class ResolvedExperiment:
    """Fully resolved experiment definition."""

    experiment_id: str
    experiment_path: Path
    repo_root: Path
    configs_root: Path
    cfg: dict[str, Any]
    source_chain: list[str]
    component_paths: dict[str, str]


def find_configs_root(start: str | Path | None = None) -> Path:
    """Locate the top-level ``configs`` directory."""
    repo_root = find_repo_root(start)
    configs_root = repo_root / "configs"
    if not configs_root.exists():
        raise FileNotFoundError(f"Could not find configs directory at {configs_root}.")
    return configs_root


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _experiment_id_from_path(path: Path, configs_root: Path) -> str:
    rel = path.relative_to(configs_root / "experiments")
    return str(rel.with_suffix("")).replace("\\", "/")


def resolve_experiment_path(
    experiment_ref: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve an experiment id like ``r03/base`` or an explicit YAML path."""
    repo = find_repo_root(repo_root)
    configs_root = find_configs_root(repo)

    candidate = Path(experiment_ref)
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                (repo / candidate),
                (configs_root / candidate),
                (configs_root / "experiments" / candidate),
                (configs_root / "experiments" / candidate).with_suffix(".yaml"),
                (repo / candidate).with_suffix(".yaml"),
            ]
        )

    for path in search_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not resolve experiment reference '{experiment_ref}'.")


def _resolve_component_path(
    component_ref: str | Path,
    *,
    configs_root: Path,
    repo_root: Path,
) -> Path:
    candidate = Path(component_ref)
    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                repo_root / candidate,
                configs_root / candidate,
                (configs_root / candidate).with_suffix(".yaml"),
            ]
        )

    for path in search_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not resolve config component '{component_ref}'.")


def _normalise_cfg(cfg: dict[str, Any], *, experiment_id: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", experiment_id.split("/")[0])
    cfg["experiment"].setdefault("variant", experiment_id.split("/")[-1])

    cfg.setdefault("runtime", {})
    runtime = cfg["runtime"]
    runtime.setdefault("runs_root", "runs/vae")
    runtime.setdefault("device", {})
    runtime["device"].setdefault("gpu_id", None)

    runtime.setdefault("checkpoints", {})
    runtime["checkpoints"].setdefault("save_latest", True)
    runtime["checkpoints"].setdefault("save_best", True)
    runtime["checkpoints"].setdefault("best_metric", "val_full.total")
    runtime["checkpoints"].setdefault("best_mode", "min")

    runtime.setdefault("resume", {})
    runtime["resume"].setdefault("mode", "exact")
    runtime["resume"].setdefault("prune_jsonl", True)
    runtime["resume"].setdefault("clear_tensorboard", False)

    runtime.setdefault("metadata", {})
    runtime["metadata"].setdefault("capture_git", True)
    runtime["metadata"].setdefault("capture_machine", True)
    runtime["metadata"].setdefault("capture_environment", True)

    runtime.setdefault("preflight", {})
    runtime["preflight"].setdefault("enabled", True)
    runtime["preflight"].setdefault("loader_warmup_batches", 1)
    runtime["preflight"].setdefault("warmup_splits", ["train"])
    runtime["preflight"].setdefault("auto_reduce_workers", True)
    runtime["preflight"].setdefault("worker_retry_min", 0)

    runtime.setdefault("run_name", {})
    runtime["run_name"].setdefault("timestamp_format", "%Y%m%d-%H%M%S")
    runtime["run_name"].setdefault("run_index_width", 4)
    runtime["run_name"].setdefault("fields", [])

    if runtime["checkpoints"]["best_mode"] not in {"min", "max"}:
        raise ValueError(
            "runtime.checkpoints.best_mode must be 'min' or 'max', "
            f"got {runtime['checkpoints']['best_mode']!r}."
        )

    core_cfg = {
        "model": cfg["model"],
        "loss": cfg["loss"],
        "training": cfg["training"],
        "data": cfg["data"],
    }
    parse_config(core_cfg)
    return cfg


def resolve_experiment(
    experiment_ref: str | Path,
    *,
    repo_root: str | Path | None = None,
    _seen: tuple[Path, ...] = (),
) -> ResolvedExperiment:
    """Resolve an experiment YAML into a final merged config."""
    repo = find_repo_root(repo_root)
    configs_root = find_configs_root(repo)
    experiment_path = resolve_experiment_path(experiment_ref, repo_root=repo)

    if experiment_path in _seen:
        chain = " -> ".join(str(path) for path in (*_seen, experiment_path))
        raise ValueError(f"Cycle detected while resolving experiments: {chain}")

    raw = _load_yaml_dict(experiment_path)
    experiment_id = _experiment_id_from_path(experiment_path, configs_root)

    merged: dict[str, Any] = {}
    source_chain: list[str] = []
    component_paths: dict[str, str] = {}

    extends_ref = raw.get("extends")
    if extends_ref:
        parent = resolve_experiment(extends_ref, repo_root=repo, _seen=(*_seen, experiment_path))
        merged = copy.deepcopy(parent.cfg)
        source_chain.extend(parent.source_chain)
        component_paths.update(parent.component_paths)

    components = raw.get("components", {})
    if components is not None and not isinstance(components, dict):
        raise TypeError(f"'components' in {experiment_path} must be a mapping.")

    for key, component_ref in (components or {}).items():
        component_path = _resolve_component_path(
            component_ref,
            configs_root=configs_root,
            repo_root=repo,
        )
        merged = _deep_merge(merged, _load_yaml_dict(component_path))
        component_paths[key] = str(component_path)

    body = {
        key: value
        for key, value in raw.items()
        if key not in {"extends", "components", "overrides"}
    }
    merged = _deep_merge(merged, body)

    overrides = raw.get("overrides", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise TypeError(f"'overrides' in {experiment_path} must be a mapping.")
        merged = _deep_merge(merged, overrides)

    merged = _normalise_cfg(merged, experiment_id=experiment_id)
    source_chain.append(str(experiment_path))

    return ResolvedExperiment(
        experiment_id=experiment_id,
        experiment_path=experiment_path,
        repo_root=repo,
        configs_root=configs_root,
        cfg=merged,
        source_chain=source_chain,
        component_paths=component_paths,
    )


def list_experiment_definitions(
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return lightweight metadata for all experiment YAMLs."""
    repo = find_repo_root(repo_root)
    configs_root = find_configs_root(repo)
    experiments_dir = configs_root / "experiments"
    results: list[dict[str, Any]] = []
    for path in sorted(experiments_dir.rglob("*.yaml")):
        raw = _load_yaml_dict(path)
        exp_block = raw.get("experiment", {}) if isinstance(raw.get("experiment"), dict) else {}
        results.append(
            {
                "id": _experiment_id_from_path(path, configs_root),
                "path": str(path),
                "name": exp_block.get("name"),
                "variant": exp_block.get("variant"),
                "description": exp_block.get("description", ""),
                "extends": raw.get("extends"),
            }
        )
    return results


def clone_experiment_definition(
    source_ref: str | Path,
    target_ref: str | Path,
    *,
    repo_root: str | Path | None = None,
    description: str | None = None,
) -> Path:
    """Create a new experiment YAML that extends an existing one."""
    repo = find_repo_root(repo_root)
    configs_root = find_configs_root(repo)
    source_path = resolve_experiment_path(source_ref, repo_root=repo)

    target = Path(target_ref)
    if target.is_absolute():
        target_path = target
    else:
        target_path = (configs_root / "experiments" / target).with_suffix(".yaml")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"Target experiment already exists: {target_path}")

    target_id = _experiment_id_from_path(target_path, configs_root)
    target_parts = target_id.split("/")
    experiment_name = target_parts[0]
    variant = target_parts[-1]

    doc = {
        "extends": _experiment_id_from_path(source_path, configs_root),
        "experiment": {
            "name": experiment_name,
            "variant": variant,
            "description": description or f"Variant cloned from {source_ref}.",
        },
        "overrides": {},
    }
    target_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target_path

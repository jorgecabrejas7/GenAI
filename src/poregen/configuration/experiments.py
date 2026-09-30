"""Config-driven experiment discovery and resolution."""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

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

    # parse_config validates VAE-specific field schemas; skip silently for
    # experiment types (e.g. LDM) that use a different model/data structure.
    try:
        core_cfg = {
            "model": cfg["model"],
            "loss": cfg.get("loss", {}),
            "training": cfg["training"],
            "data": cfg["data"],
        }
        parse_config(core_cfg)
    except (KeyError, TypeError):
        pass

    # THE SPLIT NAMES ARE CHECKED SEPARATELY, because the block above is
    # allowed to fail. parse_config carries a dataset_root/split_version
    # consistency rule, but any TypeError from an unrelated field — r08/base's
    # loss.class_ce_weight, for one — skips the whole validator, so that rule
    # has not been running on the very configs it matters most for. Warned and
    # not raised: r08/base ships with the two disagreeing and raising here
    # would refuse the production VAE config.
    data = cfg.get("data")
    if isinstance(data, dict):
        version, root = data.get("split_version"), data.get("dataset_root")
        if version is not None and root is not None and root != f"split_{version}":
            _logger.warning(
                "%s names its split twice and they disagree: dataset_root=%r "
                "but split_version=%r. The dataset_root is what is read.",
                experiment_id, root, version,
            )
    return cfg


def _apply_split_override(cfg: dict[str, Any], experiment_id: str, *,
                          outermost: bool) -> dict[str, Any]:
    """Let POREGEN_SPLIT redirect a config's dataset, latent store and version.

    A switch every config could silently opt out of would not be a switch, so
    the environment wins over ``data.dataset_root`` — and says so, loudly,
    because a run that reads different data from the one its config names must
    not do it quietly.

    THREE KEYS NAME THE SPLIT and they must move together or not at all:

    * ``dataset_root`` — the VAE configs' key. LDM configs do not have it.
    * ``latents_root`` — the LDM configs' key, a path with the split inside it.
      A latent store belongs to the split it was encoded from; moving the
      dataset while the latents stay behind produces a plausible wrong number.
    * ``split_version`` — a second name for the same split, which
      ``parse_config`` refuses to let disagree with ``dataset_root``.

    Nothing is ADDED: a config without a key keeps not having it, because
    inventing one would give an LDM config a dataset root it never reads.
    """
    from poregen.paths import split_override  # noqa: PLC0415

    # ONLY ON THE OUTERMOST RESOLUTION. Applying it to each ancestor as the
    # chain is walked would redirect configs the caller never named and log a
    # warning per level; the answer that matters is the merged one.
    forced = split_override() if outermost else None
    if not forced:
        return cfg
    data = cfg.get("data")
    if not isinstance(data, dict):
        return cfg

    was = data.get("dataset_root")
    if was is not None and was != forced:
        data["dataset_root"] = forced
        _logger.warning("POREGEN_SPLIT=%s overrides %s's data.dataset_root (%s)",
                        forced, experiment_id, was)

    latents = data.get("latents_root")
    if isinstance(latents, str):
        moved = re.sub(r"(^|/)split_[^/]+(/|$)", rf"\1{forced}\2", latents)
        if moved != latents:
            data["latents_root"] = moved
            _logger.warning("POREGEN_SPLIT=%s moves %s's latents_root to %s",
                            forced, experiment_id, moved)

    # Only when it AGREED with the root it is paired with. A config whose
    # split_version already disagrees is a separate problem and this is not
    # the place to paper over it.
    version = data.get("split_version")
    if version is not None and was == f"split_{version}":
        data["split_version"] = (forced[len("split_"):]
                                 if forced.startswith("split_") else forced)
        _logger.warning("POREGEN_SPLIT=%s moves %s's split_version %s -> %s",
                        forced, experiment_id, version, data["split_version"])
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

    merged = _apply_split_override(merged, experiment_id, outermost=not _seen)
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

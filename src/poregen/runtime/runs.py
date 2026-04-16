"""Run directory creation, discovery, and naming."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from poregen.configuration import ResolvedExperiment


@dataclass(frozen=True)
class RunContext:
    """Resolved run directory metadata."""

    experiment_id: str
    run_index: int
    run_name: str
    run_dir: Path


def _slugify(value: Any) -> str:
    text = str(value).strip().replace(".", "-")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-").lower() or "value"


def _deep_get(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing config key '{path}' while building run metadata.")
        current = current[part]
    return current


def _format_field_value(value: Any, transform: str | None) -> str:
    if transform == "slug":
        return _slugify(value)
    if transform == "sci":
        return f"{float(value):.0e}"
    if transform == "fixed3":
        return f"{float(value):.3f}"
    if value is None:
        return "none"
    return str(value)


def next_run_index(experiment_name: str, runs_root: Path) -> int:
    """Return the next sequential run index for *experiment_name*."""
    pattern = re.compile(rf"^{re.escape(experiment_name)}-run-(\d+)-")
    max_index = 0
    if not runs_root.exists():
        return 1
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_run_name(
    cfg: dict[str, Any],
    *,
    run_index: int,
    when: datetime | None = None,
) -> str:
    """Build the run directory name from config metadata."""
    experiment_name = cfg["experiment"]["name"]
    run_name_cfg = cfg["runtime"]["run_name"]
    timestamp = (when or datetime.now()).strftime(run_name_cfg["timestamp_format"])
    index_width = int(run_name_cfg.get("run_index_width", 4))

    suffix_parts = []
    for item in run_name_cfg.get("fields", []):
        label = item["label"]
        path = item["path"]
        transform = item.get("transform")
        value = _format_field_value(_deep_get(cfg, path), transform)
        suffix_parts.append(f"{label}{value}")

    suffix = "-".join(suffix_parts)
    base = f"{experiment_name}-run-{run_index:0{index_width}d}-{timestamp}"
    return f"{base}-{suffix}" if suffix else base


def create_run_context(
    resolved: ResolvedExperiment,
    *,
    existing_run_dir: str | Path | None = None,
) -> RunContext:
    """Create a fresh run context or attach to an existing run directory."""
    runs_root = (resolved.repo_root / resolved.cfg["runtime"]["runs_root"]).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    if existing_run_dir is not None:
        run_dir = Path(existing_run_dir).resolve()
        run_name = run_dir.name
        match = re.match(rf"^{re.escape(resolved.cfg['experiment']['name'])}-run-(\d+)-", run_name)
        run_index = int(match.group(1)) if match else 0
        return RunContext(
            experiment_id=resolved.experiment_id,
            run_index=run_index,
            run_name=run_name,
            run_dir=run_dir,
        )

    run_index = next_run_index(resolved.cfg["experiment"]["name"], runs_root)
    run_name = build_run_name(resolved.cfg, run_index=run_index)
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunContext(
        experiment_id=resolved.experiment_id,
        run_index=run_index,
        run_name=run_name,
        run_dir=run_dir,
    )


def save_resolved_config(run_dir: Path, cfg: dict[str, Any]) -> Path:
    """Write the frozen resolved config for this run."""
    path = run_dir / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False))
    return path


def list_run_directories(
    *,
    repo_root: str | Path,
    runs_root: str | Path = "runs/vae",
) -> list[Path]:
    """List run directories that contain metadata."""
    root = Path(repo_root).resolve() / runs_root
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "run_metadata.json").exists()],
        key=lambda path: path.name,
    )


def resolve_run_directory(
    run_ref: str | Path,
    *,
    repo_root: str | Path,
    runs_root: str | Path = "runs/vae",
) -> Path:
    """Resolve a run directory by path or run name."""
    repo = Path(repo_root).resolve()
    candidate = Path(run_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    search_paths = [
        repo / candidate,
        repo / runs_root / candidate,
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not resolve run reference '{run_ref}'.")


def format_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary from training history."""
    train_records = [record for record in history if record.get("split") == "train"]
    val_full_records = [record for record in history if record.get("split") == "val_full"]
    val_records = [record for record in history if record.get("split") == "val"]

    summary: dict[str, Any] = {}
    if train_records:
        summary["final_train_total"] = train_records[-1].get("total")
        summary["final_train_step"] = train_records[-1].get("step")
    if val_full_records:
        best = min(
            (record for record in val_full_records if "total" in record),
            key=lambda record: record["total"],
            default=None,
        )
        if best is not None:
            summary["best_val_full_total"] = best["total"]
            summary["best_val_full_step"] = best["step"]
    elif val_records:
        best = min(
            (record for record in val_records if "total" in record),
            key=lambda record: record["total"],
            default=None,
        )
        if best is not None:
            summary["best_val_total"] = best["total"]
            summary["best_val_step"] = best["step"]
    return summary

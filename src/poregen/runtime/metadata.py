"""Run metadata capture utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def collect_runtime_metadata(
    *,
    repo_root: Path,
    capture_git: bool,
    capture_machine: bool,
    capture_environment: bool,
) -> dict[str, Any]:
    """Collect lightweight runtime metadata for a run."""
    metadata: dict[str, Any] = {}
    if capture_git:
        metadata["git"] = {
            "commit": _run_git(repo_root, "rev-parse", "HEAD"),
            "branch": _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run_git(repo_root, "status", "--porcelain")),
        }
    if capture_machine:
        metadata["machine"] = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            metadata["machine"]["gpu_name"] = device_props.name
            metadata["machine"]["gpu_total_memory_gb"] = round(
                device_props.total_memory / (1024 ** 3),
                2,
            )
    if capture_environment:
        metadata["environment"] = {
            "cwd": os.getcwd(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        }
    return metadata


def save_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> Path:
    """Persist run metadata as JSON."""
    path = run_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def update_run_metadata(run_dir: Path, updates: dict[str, Any]) -> Path:
    """Merge and rewrite the run metadata JSON file."""
    path = run_dir / "run_metadata.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text())
    merged = {**current, **updates}
    path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    return path

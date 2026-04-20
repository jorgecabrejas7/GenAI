"""Experiment management CLI and guided interactive UI."""

from __future__ import annotations

import argparse
import copy
import curses
import json
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from poregen.configuration import (
    clone_experiment_definition,
    list_experiment_definitions,
    resolve_experiment,
)
from poregen.experiments.base import find_repo_root
from poregen.experiments.train_vae import resume_run, run_experiment
from poregen.runtime import build_run_name, list_run_directories
from poregen.runtime.runs import next_run_index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeferredAction:
    """Action to execute after leaving the curses UI."""

    kind: str
    primary: str
    secondary: str | None = None
    tertiary: str | None = None


@dataclass(frozen=True)
class BrowserEntry:
    """Single row in the experiment or run browser."""

    identifier: str
    title: str
    subtitle: str = ""
    path: str | None = None


@dataclass(frozen=True)
class ActionOption:
    """Visible action in the guided action pane."""

    label: str
    description: str
    action_id: str


@dataclass(frozen=True)
class DetailLine:
    """Renderable line in the preview pane."""

    text: str
    style: str = "body"


@dataclass(frozen=True)
class DetailDocument:
    """Structured preview plus full-screen text."""

    title: str
    lines: list[DetailLine]
    full_text: str


@dataclass
class DashboardState:
    """Interactive dashboard state."""

    mode: str = "experiments"
    focus: str = "browser"
    browser_index: dict[str, int] = field(
        default_factory=lambda: {"experiments": 0, "runs": 0}
    )
    action_index: dict[str, int] = field(
        default_factory=lambda: {"experiments": 0, "runs": 0}
    )
    browser_offset: dict[str, int] = field(
        default_factory=lambda: {"experiments": 0, "runs": 0}
    )
    action_offset: dict[str, int] = field(
        default_factory=lambda: {"experiments": 0, "runs": 0}
    )
    detail_offset: dict[str, int] = field(
        default_factory=lambda: {"experiments": 0, "runs": 0}
    )
    status: str = "Ready."


SECTION_DESCRIPTIONS: dict[str, str] = {
    "model": "Architecture and latent-space configuration.",
    "loss": "Loss weights and KL-related settings.",
    "data": "Dataset and dataloader settings.",
    "training": "Optimizer, schedule, logging, and evaluation cadence.",
    "runtime": "Checkpointing, preflight, metadata capture, and run naming.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage and launch config-driven PoreGen experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List available experiments.")
    subparsers.add_parser("runs", help="List available run directories.")

    show_parser = subparsers.add_parser("show", help="Show the resolved config for an experiment.")
    show_parser.add_argument("experiment", help="Experiment id like r03/base or a YAML path.")

    run_parser = subparsers.add_parser("run", help="Run an experiment definition.")
    run_parser.add_argument("experiment", help="Experiment id like r03/base or a YAML path.")

    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted run.")
    resume_parser.add_argument("run_ref", help="Run name or full run directory path.")
    resume_parser.add_argument(
        "checkpoint",
        nargs="?",
        default="latest.ckpt",
        help="Checkpoint filename or path to resume from.",
    )

    clone_parser = subparsers.add_parser("clone", help="Clone an experiment into a new variant YAML.")
    clone_parser.add_argument("source", help="Source experiment id or YAML path.")
    clone_parser.add_argument("target", nargs="?", default=None, help="Target experiment id.")

    subparsers.add_parser("menu", help="Open the interactive keyboard UI.")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the full evaluation suite on a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "run_ref",
        help="Run directory name (e.g. r03-run-0002-…) or full path.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        default="best.ckpt",
        help="Checkpoint filename (relative to run dir) or absolute path.",
    )
    eval_parser.add_argument(
        "--eval-config",
        default="eval/r03_base",
        dest="eval_config",
        help="Eval config id (e.g. 'eval/r03_base' or 'eval/r03_paper') or YAML path.",
    )

    return parser


def _print_experiments(repo_root: Path) -> list[dict[str, Any]]:
    experiments = list_experiment_definitions(repo_root=repo_root)
    if not experiments:
        print("No experiment YAMLs found.")
        return experiments

    for idx, experiment in enumerate(experiments, 1):
        description = experiment.get("description") or ""
        print(f"{idx:2d}. {experiment['id']}")
        if description:
            print(f"    {description}")
    return experiments


def _print_runs(repo_root: Path) -> list[Path]:
    runs = list_run_directories(repo_root=repo_root)
    if not runs:
        print("No runs found.")
        return runs

    for idx, run_dir in enumerate(runs, 1):
        print(f"{idx:2d}. {run_dir.name}")
    return runs


def _prompt(prompt: str) -> str:
    return input(prompt).strip()


def _safe_add_line(window: Any, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = window.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    try:
        window.addnstr(y, x, text, max(0, width - x - 1), attr)
    except curses.error:
        return


def _wrap_text(text: str, width: int) -> list[str]:
    wrapped: list[str] = []
    effective_width = max(10, width)
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped.append("")
            continue
        pieces = textwrap.wrap(
            raw_line,
            width=effective_width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped.extend(pieces or [""])
    return wrapped or [""]


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _relative_path(path: str | Path | None, repo_root: Path) -> str:
    if path is None:
        return "-"
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(repo_root))
    except ValueError:
        return str(candidate)


def _get_nested(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else None


def _default_variant_id(source_id: str) -> str:
    if "/" not in source_id:
        return f"{source_id}/variant"
    prefix, leaf = source_id.rsplit("/", 1)
    return f"{prefix}/{leaf}-variant"


def _action_options(mode: str) -> list[ActionOption]:
    if mode == "experiments":
        return [
            ActionOption("Run selected experiment", "Launch a new run from the highlighted experiment.", "run"),
            ActionOption("Create new variant", "Clone the highlighted experiment, edit common fields, and save it.", "clone"),
            ActionOption("Open full config", "View the resolved configuration in a larger window.", "view"),
            ActionOption("Refresh experiments", "Reload experiment YAMLs from disk.", "refresh"),
            ActionOption("Switch to runs", "Browse previous runs and resume checkpoints.", "switch_runs"),
            ActionOption("Help", "Show the navigation guide.", "help"),
        ]

    return [
        ActionOption("Resume latest checkpoint", "Continue the selected run from latest.ckpt.", "resume_latest"),
        ActionOption("Choose checkpoint", "Pick a specific checkpoint before resuming.", "resume_pick"),
        ActionOption("Run evaluation", "Evaluate best.ckpt with the default eval config (best_checkpoint tier).", "eval_best"),
        ActionOption("Run evaluation (choose config)", "Pick a checkpoint and eval config tier, then run evaluation.", "eval_custom"),
        ActionOption("Open full run details", "View the run metadata, summary, and resolved config.", "view"),
        ActionOption("Refresh runs", "Reload available runs from disk.", "refresh"),
        ActionOption("Switch to experiments", "Go back to experiment definitions.", "switch_experiments"),
        ActionOption("Help", "Show the navigation guide.", "help"),
    ]


def _section(lines: list[DetailLine], title: str) -> None:
    if lines:
        lines.append(DetailLine(""))
    lines.append(DetailLine(title, "heading"))


def _kv(lines: list[DetailLine], label: str, value: Any, *, style: str = "body") -> None:
    lines.append(DetailLine(f"{label}: {_format_value(value)}", style))


def _preview_run_name(cfg: dict[str, Any], repo_root: Path) -> str:
    runs_root = (repo_root / cfg["runtime"]["runs_root"]).resolve()
    try:
        run_index = next_run_index(cfg["experiment"]["name"], runs_root)
    except Exception:
        run_index = 1
    return build_run_name(cfg, run_index=run_index)


def _build_experiment_entries(repo_root: Path) -> list[BrowserEntry]:
    entries: list[BrowserEntry] = []
    for experiment in list_experiment_definitions(repo_root=repo_root):
        subtitle = experiment.get("description") or ""
        if not subtitle and experiment.get("extends"):
            subtitle = f"extends {experiment['extends']}"
        if not subtitle:
            subtitle = _relative_path(experiment.get("path"), repo_root)
        entries.append(
            BrowserEntry(
                identifier=experiment["id"],
                title=experiment["id"],
                subtitle=subtitle,
                path=experiment.get("path"),
            )
        )
    return entries


def _build_run_entries(repo_root: Path) -> list[BrowserEntry]:
    entries: list[BrowserEntry] = []
    for run_dir in list_run_directories(repo_root=repo_root):
        summary = _load_json(run_dir / "summary.json") or {}
        subtitle_parts: list[str] = []
        if "best_val_full_total" in summary:
            subtitle_parts.append(f"best val {summary['best_val_full_total']:.4f}")
        elif "best_val_total" in summary:
            subtitle_parts.append(f"best val {summary['best_val_total']:.4f}")

        checkpoint_count = sum(1 for _ in run_dir.glob("*.ckpt"))
        if checkpoint_count:
            subtitle_parts.append(f"{checkpoint_count} ckpt")

        modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        subtitle_parts.append(modified)
        entries.append(
            BrowserEntry(
                identifier=run_dir.name,
                title=run_dir.name,
                subtitle=" | ".join(subtitle_parts),
                path=str(run_dir),
            )
        )
    return entries


def _build_experiment_document(experiment_id: str, repo_root: Path) -> DetailDocument:
    resolved = resolve_experiment(experiment_id, repo_root=repo_root)
    cfg = resolved.cfg
    lines: list[DetailLine] = []

    _section(lines, "Overview")
    _kv(lines, "id", resolved.experiment_id)
    _kv(lines, "description", _get_nested(cfg, "experiment.description", "-"))
    _kv(lines, "definition", _relative_path(resolved.experiment_path, repo_root))
    _kv(lines, "extends", " -> ".join(_relative_path(path, repo_root) for path in resolved.source_chain))

    _section(lines, "Core Settings")
    _kv(lines, "model", _get_nested(cfg, "model.name"))
    _kv(lines, "latent channels", _get_nested(cfg, "model.z_channels"))
    _kv(lines, "base channels", _get_nested(cfg, "model.base_channels"))
    _kv(lines, "patch size", _get_nested(cfg, "model.patch_size"))
    _kv(lines, "blocks", _get_nested(cfg, "model.n_blocks"))
    _kv(lines, "dataset root", _get_nested(cfg, "data.dataset_root"))
    _kv(lines, "split", _get_nested(cfg, "data.split_version"))
    _kv(lines, "batch size", _get_nested(cfg, "data.batch_size"))
    _kv(lines, "workers", _get_nested(cfg, "data.num_workers"))
    _kv(lines, "prefetch factor", _get_nested(cfg, "data.prefetch_factor"))
    _kv(lines, "persistent workers", _get_nested(cfg, "data.persistent_workers"))

    _section(lines, "Training")
    _kv(lines, "total steps", _get_nested(cfg, "training.total_steps"))
    _kv(lines, "learning rate", _get_nested(cfg, "training.lr"))
    _kv(lines, "scheduler", _get_nested(cfg, "training.scheduler"))
    _kv(lines, "compile model", _get_nested(cfg, "training.compile"))
    _kv(lines, "deterministic", _get_nested(cfg, "training.deterministic"))
    _kv(lines, "log every", _get_nested(cfg, "training.log_every"))
    _kv(lines, "eval every", _get_nested(cfg, "training.eval_every"))
    _kv(lines, "val batches", _get_nested(cfg, "training.val_batches"))
    _kv(lines, "save every", _get_nested(cfg, "training.save_every"))
    _kv(lines, "montecarlo every", _get_nested(cfg, "training.montecarlo_every"))
    _kv(lines, "final full eval", _get_nested(cfg, "training.final_full_eval"))

    _section(lines, "Runtime")
    _kv(lines, "runs root", _get_nested(cfg, "runtime.runs_root"))
    _kv(lines, "best checkpoint", _get_nested(cfg, "runtime.checkpoints.best_metric"))
    _kv(lines, "best mode", _get_nested(cfg, "runtime.checkpoints.best_mode"))
    _kv(lines, "save latest", _get_nested(cfg, "runtime.checkpoints.save_latest"))
    _kv(lines, "save best", _get_nested(cfg, "runtime.checkpoints.save_best"))
    _kv(lines, "resume mode", _get_nested(cfg, "runtime.resume.mode"))
    _kv(lines, "preflight", _get_nested(cfg, "runtime.preflight.enabled"))
    _kv(lines, "machine", _get_nested(cfg, "runtime.machine.name", "-"))
    _kv(lines, "run name preview", _preview_run_name(cfg, repo_root))

    _section(lines, "Components")
    for key, path in resolved.component_paths.items():
        _kv(lines, key, _relative_path(path, repo_root), style="dim")

    _section(lines, "Resolved YAML")
    yaml_text = yaml.safe_dump(cfg, sort_keys=False).rstrip()
    for raw_line in yaml_text.splitlines():
        lines.append(DetailLine(raw_line, "code"))

    full_text = "\n".join(line.text for line in lines)
    return DetailDocument(
        title=f"Experiment {resolved.experiment_id}",
        lines=lines,
        full_text=full_text,
    )


def _build_run_document(run_name: str, repo_root: Path) -> DetailDocument:
    run_dir = next((path for path in list_run_directories(repo_root=repo_root) if path.name == run_name), None)
    if run_dir is None:
        return DetailDocument(
            title=f"Run {run_name}",
            lines=[DetailLine(f"Run not found: {run_name}", "heading")],
            full_text=f"Run not found: {run_name}",
        )

    metadata = _load_json(run_dir / "run_metadata.json") or {}
    summary = _load_json(run_dir / "summary.json") or {}
    resolved_cfg = _load_yaml(run_dir / "resolved_config.yaml")
    checkpoints = sorted(path.name for path in run_dir.glob("*.ckpt"))

    lines: list[DetailLine] = []
    _section(lines, "Overview")
    _kv(lines, "run", run_dir.name)
    _kv(lines, "path", _relative_path(run_dir, repo_root))
    _kv(lines, "modified", datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))

    if resolved_cfg is not None:
        _kv(
            lines,
            "experiment",
            f"{_get_nested(resolved_cfg, 'experiment.name', '?')}/"
            f"{_get_nested(resolved_cfg, 'experiment.variant', '?')}",
        )
        _kv(lines, "model", _get_nested(resolved_cfg, "model.name"))
        _kv(lines, "latent channels", _get_nested(resolved_cfg, "model.z_channels"))
        _kv(lines, "batch size", _get_nested(resolved_cfg, "data.batch_size"))
        _kv(lines, "workers", _get_nested(resolved_cfg, "data.num_workers"))
        _kv(lines, "compile model", _get_nested(resolved_cfg, "training.compile"))

    _section(lines, "Summary")
    if summary:
        for key, value in sorted(summary.items()):
            _kv(lines, key, value)
    else:
        lines.append(DetailLine("No summary.json found yet.", "dim"))

    _section(lines, "Checkpoints")
    if checkpoints:
        _kv(lines, "count", len(checkpoints))
        for checkpoint in checkpoints[:8]:
            lines.append(DetailLine(checkpoint, "code"))
        if len(checkpoints) > 8:
            lines.append(DetailLine(f"... {len(checkpoints) - 8} more", "dim"))
    else:
        lines.append(DetailLine("No checkpoints found.", "dim"))

    _section(lines, "Metadata")
    if metadata:
        git_meta = metadata.get("git", {})
        machine_meta = metadata.get("machine", {})
        if git_meta:
            _kv(lines, "git commit", git_meta.get("commit"))
            _kv(lines, "git branch", git_meta.get("branch"))
            _kv(lines, "git dirty", git_meta.get("dirty"))
        if machine_meta:
            _kv(lines, "gpu", machine_meta.get("gpu_name", "-"))
            _kv(lines, "gpu memory", machine_meta.get("gpu_total_memory_gb", "-"))
            _kv(lines, "torch", machine_meta.get("torch", "-"))
            _kv(lines, "python", machine_meta.get("python", "-"))
        known_keys = {"git", "machine", "environment"}
        for key, value in sorted(metadata.items()):
            if key in known_keys:
                continue
            _kv(lines, key, value)
    else:
        lines.append(DetailLine("No run_metadata.json found.", "dim"))

    if resolved_cfg is not None:
        _section(lines, "Resolved Config")
        yaml_text = yaml.safe_dump(resolved_cfg, sort_keys=False).rstrip()
        for raw_line in yaml_text.splitlines():
            lines.append(DetailLine(raw_line, "code"))
    else:
        _section(lines, "Raw Metadata")
        for raw_line in json.dumps(metadata, indent=2, sort_keys=True).splitlines():
            lines.append(DetailLine(raw_line, "code"))

    full_blocks = [
        f"Run: {run_dir.name}",
        "",
        "[summary.json]",
        json.dumps(summary, indent=2, sort_keys=True) if summary else "{}",
        "",
        "[run_metadata.json]",
        json.dumps(metadata, indent=2, sort_keys=True) if metadata else "{}",
    ]
    if resolved_cfg is not None:
        full_blocks.extend(
            [
                "",
                "[resolved_config.yaml]",
                yaml.safe_dump(resolved_cfg, sort_keys=False).rstrip(),
            ]
        )

    return DetailDocument(
        title=f"Run {run_dir.name}",
        lines=lines,
        full_text="\n".join(full_blocks),
    )


def _show_message(stdscr: Any, title: str, message: str) -> None:
    lines = _wrap_text(message, max(20, stdscr.getmaxyx()[1] - 8))
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        box_width = min(width - 4, max(40, max(len(title) + 6, *(len(line) + 4 for line in lines))))
        box_height = min(height - 4, max(7, len(lines) + 4))
        start_y = max(1, (height - box_height) // 2)
        start_x = max(2, (width - box_width) // 2)
        window = stdscr.derwin(box_height, box_width, start_y, start_x)
        window.erase()
        window.box()
        _safe_add_line(window, 0, 2, f" {title} ", curses.A_BOLD)
        for idx, line in enumerate(lines[: box_height - 4], start=2):
            _safe_add_line(window, idx, 2, line)
        _safe_add_line(window, box_height - 2, 2, "Press any key to continue.", curses.A_DIM)
        stdscr.refresh()
        window.refresh()
        stdscr.getch()
        return


def _view_text(stdscr: Any, title: str, text: str) -> None:
    offset = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        window = stdscr.derwin(max(3, height - 2), max(10, width - 4), 1, 2)
        window.erase()
        window.box()
        _safe_add_line(window, 0, 2, f" {title} ", curses.A_BOLD)
        _safe_add_line(
            window,
            1,
            2,
            "Arrows or PgUp/PgDn to scroll, q or Esc to close.",
            curses.A_DIM,
        )
        lines = _wrap_text(text, max(10, window.getmaxyx()[1] - 4))
        inner_height = max(1, window.getmaxyx()[0] - 4)
        offset = max(0, min(offset, max(0, len(lines) - inner_height)))
        for row, line in enumerate(lines[offset : offset + inner_height], start=3):
            _safe_add_line(window, row, 2, line)
        progress = f"{min(len(lines), offset + inner_height)}/{len(lines)}"
        _safe_add_line(window, window.getmaxyx()[0] - 2, max(2, window.getmaxyx()[1] - len(progress) - 3), progress, curses.A_DIM)
        stdscr.refresh()
        window.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return
        if key in (curses.KEY_UP, ord("k")):
            offset = max(0, offset - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            offset = min(max(0, len(lines) - inner_height), offset + 1)
        elif key == curses.KEY_PPAGE:
            offset = max(0, offset - inner_height)
        elif key == curses.KEY_NPAGE:
            offset = min(max(0, len(lines) - inner_height), offset + inner_height)
        elif key == curses.KEY_HOME:
            offset = 0
        elif key == curses.KEY_END:
            offset = max(0, len(lines) - inner_height)


def _select_item(
    stdscr: Any,
    title: str,
    items: list[tuple[str, str]],
    *,
    empty_message: str,
) -> int | None:
    if not items:
        _show_message(stdscr, title, empty_message)
        return None

    index = 0
    offset = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        window = stdscr.derwin(max(3, height - 2), max(10, width - 4), 1, 2)
        window.erase()
        window.box()
        _safe_add_line(window, 0, 2, f" {title} ", curses.A_BOLD)
        _safe_add_line(window, 1, 2, "Arrows to move, Enter to select, q or Esc to cancel.", curses.A_DIM)

        inner_height = max(1, window.getmaxyx()[0] - 4)
        if index < offset:
            offset = index
        elif index >= offset + inner_height:
            offset = index - inner_height + 1

        for row, (label, sublabel) in enumerate(items[offset : offset + inner_height], start=3):
            actual_index = offset + row - 3
            attr = curses.A_REVERSE if actual_index == index else curses.A_NORMAL
            _safe_add_line(window, row, 2, label, attr | curses.A_BOLD)
            if sublabel:
                _safe_add_line(window, row, min(window.getmaxyx()[1] // 2, len(label) + 4), sublabel, attr | curses.A_DIM)

        stdscr.refresh()
        window.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (curses.KEY_UP, ord("k")):
            index = max(0, index - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            index = min(len(items) - 1, index + 1)
        elif key == curses.KEY_PPAGE:
            index = max(0, index - inner_height)
        elif key == curses.KEY_NPAGE:
            index = min(len(items) - 1, index + inner_height)
        elif key in (10, 13, curses.KEY_ENTER):
            return index


def _prompt_text(stdscr: Any, title: str, prompt: str, default: str = "") -> str | None:
    buffer = default
    try:
        curses.curs_set(1)
    except curses.error:
        pass

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        box_width = min(width - 4, max(50, len(prompt) + 8))
        box_height = 8
        start_y = max(1, (height - box_height) // 2)
        start_x = max(2, (width - box_width) // 2)
        window = stdscr.derwin(box_height, box_width, start_y, start_x)
        window.erase()
        window.box()
        _safe_add_line(window, 0, 2, f" {title} ", curses.A_BOLD)
        _safe_add_line(window, 2, 2, prompt)
        _safe_add_line(window, 4, 2, buffer, curses.A_REVERSE)
        _safe_add_line(window, 6, 2, "Enter confirms. Esc cancels.", curses.A_DIM)
        cursor_x = min(len(buffer), max(0, box_width - 5))
        window.move(4, 2 + cursor_x)
        stdscr.refresh()
        window.refresh()

        key = stdscr.getch()
        if key == 27:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            return None
        if key in (10, 13, curses.KEY_ENTER):
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            return buffer.strip()
        if key in (curses.KEY_BACKSPACE, 127, 8):
            buffer = buffer[:-1]
            continue
        if 32 <= key <= 126:
            buffer += chr(key)


def _choose_checkpoint_interactive(stdscr: Any, run_dir: Path) -> str | None:
    checkpoints = sorted(run_dir.glob("*.ckpt"))
    items = [(path.name, datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")) for path in checkpoints]
    selected = _select_item(
        stdscr,
        f"Resume {run_dir.name}",
        items,
        empty_message="No checkpoints found in this run directory.",
    )
    if selected is None:
        return None
    return items[selected][0]


_NO_DIFF = object()


def _yaml_inline(value: Any) -> str:
    rendered = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    lines = [line for line in rendered.splitlines() if line.strip() != "..."]
    return " ".join(lines).strip() or "null"


def _preview_editor_value(value: Any, *, limit: int = 48) -> str:
    if isinstance(value, dict):
        summary = f"<section: {len(value)} keys>"
    else:
        summary = _yaml_inline(value)
    return summary if len(summary) <= limit else f"{summary[: limit - 3]}..."


def _count_changes(source_value: Any, draft_value: Any) -> int:
    if isinstance(source_value, dict) and isinstance(draft_value, dict):
        total = 0
        for key in draft_value:
            total += _count_changes(source_value.get(key), draft_value[key])
        for key in source_value:
            if key not in draft_value:
                total += 1
        return total
    return 0 if source_value == draft_value else 1


def _deep_diff(source_value: Any, draft_value: Any) -> Any:
    if isinstance(source_value, dict) and isinstance(draft_value, dict):
        diff: dict[str, Any] = {}
        for key, draft_child in draft_value.items():
            source_child = source_value.get(key, _NO_DIFF)
            child_diff = _deep_diff(source_child, draft_child)
            if child_diff is not _NO_DIFF:
                diff[key] = child_diff
        return diff if diff else _NO_DIFF
    return _NO_DIFF if source_value == draft_value else copy.deepcopy(draft_value)


def _build_variant_overrides(source_cfg: dict[str, Any], draft_cfg: dict[str, Any]) -> dict[str, Any]:
    diff = _deep_diff(source_cfg, draft_cfg)
    return {} if diff is _NO_DIFF or not isinstance(diff, dict) else diff


def _editable_section_keys(draft_cfg: dict[str, Any]) -> list[str]:
    return [key for key, value in draft_cfg.items() if key != "experiment" and isinstance(value, dict)]


def _edit_leaf_value(
    stdscr: Any,
    *,
    title: str,
    key_path: str,
    current_value: Any,
    source_value: Any,
) -> Any | None:
    default = _yaml_inline(current_value)
    inherited = _preview_editor_value(source_value)
    raw = _prompt_text(
        stdscr,
        title,
        f"{key_path} | inherited: {inherited}",
        default=default,
    )
    if raw is None:
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        _show_message(
            stdscr,
            title,
            f"Could not parse that value as YAML.\n\n{exc}\n\nExamples: 0.1, true, null, [1, 2, 3]",
        )
        return _edit_leaf_value(
            stdscr,
            title=title,
            key_path=key_path,
            current_value=current_value,
            source_value=source_value,
        )


def _edit_config_mapping(
    stdscr: Any,
    *,
    title: str,
    path_prefix: str,
    source_mapping: dict[str, Any],
    draft_mapping: dict[str, Any],
) -> None:
    while True:
        items: list[tuple[str, str]] = []
        keys = list(draft_mapping.keys())
        for key in keys:
            current_value = draft_mapping[key]
            source_value = source_mapping.get(key)
            change_count = _count_changes(source_value, current_value)
            changed_marker = f" [{change_count} changed]" if change_count else ""
            if isinstance(current_value, dict):
                description = SECTION_DESCRIPTIONS.get(key, "Nested configuration section.")
                items.append((f"{key}{changed_marker}", description))
            else:
                items.append(
                    (
                        f"{key}: {_preview_editor_value(current_value)}{changed_marker}",
                        f"inherited: {_preview_editor_value(source_value)}",
                    )
                )

        items.append(("Back", "Return to the previous menu."))
        selected = _select_item(
            stdscr,
            title,
            items,
            empty_message="No configurable values available.",
        )
        if selected is None or selected == len(items) - 1:
            return

        key = keys[selected]
        current_value = draft_mapping[key]
        source_value = source_mapping.get(key)
        full_path = f"{path_prefix}.{key}" if path_prefix else key
        if isinstance(current_value, dict) and isinstance(source_value, dict):
            _edit_config_mapping(
                stdscr,
                title=f"{title}: {key}",
                path_prefix=full_path,
                source_mapping=source_value,
                draft_mapping=current_value,
            )
            continue

        updated_value = _edit_leaf_value(
            stdscr,
            title=title,
            key_path=full_path,
            current_value=current_value,
            source_value=source_value,
        )
        if updated_value is not None:
            draft_mapping[key] = updated_value


def _save_variant_yaml(
    *,
    repo_root: Path,
    source_id: str,
    target_id: str,
    description: str,
    source_cfg: dict[str, Any],
    draft_cfg: dict[str, Any],
) -> Path:
    target_path = clone_experiment_definition(
        source_id,
        target_id,
        repo_root=repo_root,
        description=description or None,
    )
    document = _load_yaml(target_path) or {}
    document.setdefault("experiment", {})
    target_parts = target_id.split("/")
    document["experiment"]["name"] = target_parts[0]
    document["experiment"]["variant"] = target_parts[-1]
    document["experiment"]["description"] = description
    document["overrides"] = _build_variant_overrides(source_cfg, draft_cfg)
    target_path.write_text(yaml.safe_dump(document, sort_keys=False))
    return target_path


def _variant_preview_text(
    *,
    source_id: str,
    target_id: str,
    description: str,
    source_cfg: dict[str, Any],
    draft_cfg: dict[str, Any],
) -> str:
    preview_doc = {
        "extends": source_id,
        "experiment": {
            "name": target_id.split("/")[0],
            "variant": target_id.split("/")[-1],
            "description": description,
        },
        "overrides": _build_variant_overrides(source_cfg, draft_cfg),
    }
    return yaml.safe_dump(preview_doc, sort_keys=False)


def _guided_clone_experiment(
    stdscr: Any,
    repo_root: Path,
    source_id: str,
) -> tuple[str, str] | None:
    resolved = resolve_experiment(source_id, repo_root=repo_root)
    source_cfg = copy.deepcopy(resolved.cfg)
    draft_cfg = copy.deepcopy(resolved.cfg)
    target_id = _default_variant_id(source_id)
    description = f"Variant cloned from {source_id}."

    while True:
        section_keys = _editable_section_keys(draft_cfg)
        items = [
            (f"Variant id: {target_id}", "Choose the new experiment name and folder id."),
            (f"Description: {description}", "Human-readable summary shown in experiment lists."),
        ]
        section_start = len(items)
        for section_key in section_keys:
            change_count = _count_changes(source_cfg.get(section_key), draft_cfg.get(section_key))
            changed_marker = f" [{change_count} changed]" if change_count else ""
            section_title = section_key.replace("_", " ").title()
            items.append(
                (
                    f"Edit {section_title}{changed_marker}",
                    SECTION_DESCRIPTIONS.get(section_key, "Open this config section and edit any value."),
                )
            )
        items.extend(
            [
                ("Preview YAML to save", "Review the exact variant file before writing it."),
                ("Save variant", "Write a new YAML file for this edited experiment."),
                ("Cancel", "Leave without creating a new experiment."),
            ]
        )
        selected = _select_item(
            stdscr,
            f"Create Variant From {source_id}",
            items,
            empty_message="No options available.",
        )
        if selected is None or selected == len(items) - 1:
            return None
        if selected == 0:
            updated = _prompt_text(
                stdscr,
                "Variant Id",
                "New experiment id:",
                default=target_id,
            )
            if updated:
                target_id = updated
            continue
        if selected == 1:
            updated = _prompt_text(
                stdscr,
                "Variant Description",
                "Experiment description:",
                default=description,
            )
            if updated is not None:
                description = updated
            continue
        if section_start <= selected < section_start + len(section_keys):
            section_key = section_keys[selected - section_start]
            section_title = section_key.replace("_", " ").title()
            _edit_config_mapping(
                stdscr,
                title=f"{section_title} Editor",
                path_prefix=section_key,
                source_mapping=source_cfg.get(section_key, {}),
                draft_mapping=draft_cfg[section_key],
            )
            continue

        preview_index = section_start + len(section_keys)
        save_index = preview_index + 1
        cancel_index = preview_index + 2

        if selected == preview_index:
            _view_text(
                stdscr,
                f"Variant Preview: {target_id}",
                _variant_preview_text(
                    source_id=source_id,
                    target_id=target_id,
                    description=description,
                    source_cfg=source_cfg,
                    draft_cfg=draft_cfg,
                ),
            )
            continue
        if selected == save_index:
            target_path = _save_variant_yaml(
                repo_root=repo_root,
                source_id=source_id,
                target_id=target_id,
                description=description,
                source_cfg=source_cfg,
                draft_cfg=draft_cfg,
            )
            return target_id, str(target_path)
        if selected == cancel_index:
            return None


def _init_theme() -> dict[str, int]:
    theme = {
        "header": curses.A_BOLD,
        "tab_active": curses.A_BOLD | curses.A_REVERSE,
        "tab_inactive": curses.A_DIM,
        "panel_title": curses.A_BOLD,
        "panel_title_focus": curses.A_BOLD,
        "panel_border": curses.A_NORMAL,
        "panel_border_focus": curses.A_BOLD,
        "selected": curses.A_REVERSE | curses.A_BOLD,
        "selected_subtitle": curses.A_REVERSE | curses.A_DIM,
        "subtitle": curses.A_DIM,
        "heading": curses.A_BOLD,
        "dim": curses.A_DIM,
        "status": curses.A_BOLD,
        "code": curses.A_NORMAL,
        "accent": curses.A_BOLD,
    }

    if not curses.has_colors():
        return theme

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_BLUE, -1)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)

    theme["header"] = curses.A_BOLD | curses.color_pair(1)
    theme["tab_active"] = curses.A_BOLD | curses.color_pair(5)
    theme["tab_inactive"] = curses.A_DIM | curses.color_pair(4)
    theme["panel_title"] = curses.A_BOLD | curses.color_pair(1)
    theme["panel_title_focus"] = curses.A_BOLD | curses.color_pair(2)
    theme["panel_border"] = curses.color_pair(4)
    theme["panel_border_focus"] = curses.A_BOLD | curses.color_pair(2)
    theme["heading"] = curses.A_BOLD | curses.color_pair(2)
    theme["status"] = curses.A_BOLD | curses.color_pair(3)
    theme["code"] = curses.color_pair(4)
    theme["accent"] = curses.A_BOLD | curses.color_pair(1)
    return theme


def _draw_panel(
    stdscr: Any,
    y: int,
    x: int,
    height: int,
    width: int,
    title: str,
    *,
    theme: dict[str, int],
    focused: bool,
) -> Any:
    window = stdscr.derwin(height, width, y, x)
    window.erase()
    border_attr = theme["panel_border_focus"] if focused else theme["panel_border"]
    title_attr = theme["panel_title_focus"] if focused else theme["panel_title"]
    window.attron(border_attr)
    window.box()
    window.attroff(border_attr)
    _safe_add_line(window, 0, 2, f" {title} ", title_attr)
    return window


def _draw_header(
    stdscr: Any,
    state: DashboardState,
    counts: dict[str, int],
    current_label: str | None,
    theme: dict[str, int],
) -> None:
    height, width = stdscr.getmaxyx()
    _safe_add_line(stdscr, 0, 2, "PoreGen Experiment Control Center", theme["header"])
    count_text = f"{counts['experiments']} experiments | {counts['runs']} runs"
    _safe_add_line(stdscr, 0, max(2, width - len(count_text) - 2), count_text, curses.A_DIM)

    tabs = [("experiments", " Experiments "), ("runs", " Runs ")]
    x = 2
    for mode, label in tabs:
        attr = theme["tab_active"] if state.mode == mode else theme["tab_inactive"]
        _safe_add_line(stdscr, 1, x, label, attr)
        x += len(label) + 1

    selected_text = current_label or "No selection"
    focus_label = {
        "browser": "List",
        "actions": "Options",
        "details": "Details",
    }.get(state.focus, state.focus)
    focus_text = f"Selected pane: {focus_label}"
    _safe_add_line(stdscr, 2, 2, selected_text, theme["accent"])
    _safe_add_line(stdscr, 2, max(2, width - len(focus_text) - 2), focus_text, curses.A_DIM)


def _draw_footer(stdscr: Any, state: DashboardState, theme: dict[str, int]) -> None:
    height, width = stdscr.getmaxyx()
    help_text = (
        "Arrows move in the focused pane | Left/Right or Tab changes pane | "
        "Enter selects the highlighted option | q quits"
    )
    _safe_add_line(stdscr, height - 2, 2, help_text[: max(0, width - 4)], curses.A_DIM)
    _safe_add_line(stdscr, height - 1, 2, state.status[: max(0, width - 4)], theme["status"])


def _draw_browser_panel(
    window: Any,
    entries: list[BrowserEntry],
    *,
    selected_index: int,
    offset: int,
    focus: str,
    theme: dict[str, int],
) -> int:
    inner_height = max(1, window.getmaxyx()[0] - 2)
    inner_width = max(10, window.getmaxyx()[1] - 4)
    row_height = 2
    visible_items = max(1, inner_height // row_height)
    selected_index = max(0, min(selected_index, max(0, len(entries) - 1)))

    if selected_index < offset:
        offset = selected_index
    elif selected_index >= offset + visible_items:
        offset = selected_index - visible_items + 1

    if not entries:
        _safe_add_line(window, 2, 2, "Nothing to show yet.", theme["dim"])
        return 0

    visible = entries[offset : offset + visible_items]
    for slot, entry in enumerate(visible):
        actual_index = offset + slot
        row_y = 1 + slot * row_height
        is_selected = actual_index == selected_index
        title_attr = theme["selected"] if is_selected else curses.A_BOLD
        subtitle_attr = theme["selected_subtitle"] if is_selected else theme["subtitle"]
        if is_selected and focus == "browser":
            title_attr |= theme["accent"]
        _safe_add_line(window, row_y, 2, entry.title[:inner_width], title_attr)
        _safe_add_line(window, row_y + 1, 2, entry.subtitle[:inner_width], subtitle_attr)

    progress = f"{selected_index + 1}/{len(entries)}"
    _safe_add_line(
        window,
        window.getmaxyx()[0] - 2,
        max(2, window.getmaxyx()[1] - len(progress) - 2),
        progress,
        curses.A_DIM,
    )
    return offset


def _draw_action_panel(
    window: Any,
    actions: list[ActionOption],
    *,
    selected_index: int,
    offset: int,
    focus: str,
    theme: dict[str, int],
) -> int:
    inner_height = max(1, window.getmaxyx()[0] - 2)
    inner_width = max(10, window.getmaxyx()[1] - 4)
    row_height = 2
    visible_items = max(1, inner_height // row_height)
    selected_index = max(0, min(selected_index, max(0, len(actions) - 1)))

    if selected_index < offset:
        offset = selected_index
    elif selected_index >= offset + visible_items:
        offset = selected_index - visible_items + 1

    if not actions:
        _safe_add_line(window, 2, 2, "No actions available.", theme["dim"])
        return 0

    visible = actions[offset : offset + visible_items]
    for slot, action in enumerate(visible):
        actual_index = offset + slot
        row_y = 1 + slot * row_height
        is_selected = actual_index == selected_index
        title_attr = theme["selected"] if is_selected else curses.A_BOLD
        subtitle_attr = theme["selected_subtitle"] if is_selected else theme["subtitle"]
        if is_selected and focus == "actions":
            title_attr |= theme["accent"]
        _safe_add_line(window, row_y, 2, action.label[:inner_width], title_attr)
        _safe_add_line(window, row_y + 1, 2, action.description[:inner_width], subtitle_attr)

    progress = f"{selected_index + 1}/{len(actions)}"
    _safe_add_line(
        window,
        window.getmaxyx()[0] - 2,
        max(2, window.getmaxyx()[1] - len(progress) - 2),
        progress,
        curses.A_DIM,
    )
    return offset


def _wrap_detail_lines(lines: list[DetailLine], width: int) -> list[DetailLine]:
    wrapped: list[DetailLine] = []
    for line in lines:
        if line.text == "":
            wrapped.append(line)
            continue
        for wrapped_line in _wrap_text(line.text, width):
            wrapped.append(DetailLine(wrapped_line, line.style))
    return wrapped


def _detail_attr(style: str, theme: dict[str, int]) -> int:
    if style == "heading":
        return theme["heading"]
    if style == "dim":
        return theme["dim"]
    if style == "code":
        return theme["code"]
    return curses.A_NORMAL


def _draw_detail_panel(
    window: Any,
    document: DetailDocument,
    *,
    offset: int,
    focus: str,
    theme: dict[str, int],
) -> int:
    inner_height = max(1, window.getmaxyx()[0] - 2)
    inner_width = max(10, window.getmaxyx()[1] - 4)
    lines = _wrap_detail_lines(document.lines, inner_width)
    offset = max(0, min(offset, max(0, len(lines) - inner_height)))

    if not lines:
        _safe_add_line(window, 2, 2, "No details available.", theme["dim"])
        return 0

    for row, line in enumerate(lines[offset : offset + inner_height], start=1):
        attr = _detail_attr(line.style, theme)
        if focus == "details":
            attr |= 0
        _safe_add_line(window, row, 2, line.text, attr)

    progress = f"{min(len(lines), offset + inner_height)}/{len(lines)}"
    _safe_add_line(
        window,
        window.getmaxyx()[0] - 2,
        max(2, window.getmaxyx()[1] - len(progress) - 2),
        progress,
        curses.A_DIM,
    )
    return offset


def _show_help(stdscr: Any) -> None:
    help_text = "\n".join(
        [
            "PoreGen Experiment Control Center",
            "",
            "Main navigation",
            "  Up / Down              Move inside the highlighted pane",
            "  Left / Right           Move between List, Options, and Details",
            "  Tab                    Cycle between panes",
            "  PgUp / PgDn            Move faster in the highlighted pane",
            "  Enter                  Execute the highlighted option",
            "",
            "Experiments flow",
            "  1. Pick an experiment in the List pane.",
            "  2. Move to Options and choose Create new variant.",
            "  3. Walk through naming, opening any config section, editing values, previewing, and saving.",
            "",
            "Runs flow",
            "  1. Use the Options pane to switch to runs.",
            "  2. Pick a run in the List pane.",
            "  3. Resume latest or choose a checkpoint from Options.",
            "",
            "  q                      Quit the dashboard",
        ]
    )
    _view_text(stdscr, "Keyboard Help", help_text)


def _move_browser_selection(state: DashboardState, delta: int, entries: list[BrowserEntry]) -> None:
    if not entries:
        state.browser_index[state.mode] = 0
        return
    state.browser_index[state.mode] = max(0, min(len(entries) - 1, state.browser_index[state.mode] + delta))
    state.detail_offset[state.mode] = 0


def _scroll_detail(state: DashboardState, delta: int) -> None:
    state.detail_offset[state.mode] = max(0, state.detail_offset[state.mode] + delta)


def _move_action_selection(state: DashboardState, delta: int, actions: list[ActionOption]) -> None:
    if not actions:
        state.action_index[state.mode] = 0
        return
    state.action_index[state.mode] = max(0, min(len(actions) - 1, state.action_index[state.mode] + delta))


def _cycle_focus(state: DashboardState, delta: int) -> None:
    order = ["browser", "actions", "details"]
    current = order.index(state.focus)
    state.focus = order[(current + delta) % len(order)]


def _selected_entry(state: DashboardState, entries_by_mode: dict[str, list[BrowserEntry]]) -> BrowserEntry | None:
    entries = entries_by_mode[state.mode]
    if not entries:
        return None
    index = max(0, min(state.browser_index[state.mode], len(entries) - 1))
    state.browser_index[state.mode] = index
    return entries[index]


def _current_document(
    state: DashboardState,
    repo_root: Path,
    entries_by_mode: dict[str, list[BrowserEntry]],
    cache: dict[str, dict[str, DetailDocument]],
) -> DetailDocument:
    entry = _selected_entry(state, entries_by_mode)
    if entry is None:
        return DetailDocument(
            title="Nothing Selected",
            lines=[DetailLine("Select an experiment or run to preview it.", "heading")],
            full_text="Select an experiment or run to preview it.",
        )

    mode_cache = cache[state.mode]
    if entry.identifier not in mode_cache:
        if state.mode == "experiments":
            mode_cache[entry.identifier] = _build_experiment_document(entry.identifier, repo_root)
        else:
            mode_cache[entry.identifier] = _build_run_document(entry.identifier, repo_root)
    return mode_cache[entry.identifier]


def _refresh_dashboard_data(repo_root: Path) -> dict[str, list[BrowserEntry]]:
    return {
        "experiments": _build_experiment_entries(repo_root),
        "runs": _build_run_entries(repo_root),
    }


def _select_entry_by_id(entries: list[BrowserEntry], identifier: str) -> int:
    for index, entry in enumerate(entries):
        if entry.identifier == identifier:
            return index
    return 0


def _refresh_after_clone(
    *,
    repo_root: Path,
    state: DashboardState,
    target_id: str,
) -> tuple[dict[str, list[BrowserEntry]], dict[str, dict[str, DetailDocument]]]:
    entries_by_mode = _refresh_dashboard_data(repo_root)
    cache: dict[str, dict[str, DetailDocument]] = {"experiments": {}, "runs": {}}
    state.mode = "experiments"
    state.focus = "browser"
    state.browser_index["experiments"] = _select_entry_by_id(entries_by_mode["experiments"], target_id)
    state.browser_offset["experiments"] = 0
    state.detail_offset["experiments"] = 0
    state.action_index["experiments"] = 0
    state.action_offset["experiments"] = 0
    return entries_by_mode, cache


def _select_eval_config_interactive(stdscr: Any, repo_root: Path) -> str | None:
    """Let the user pick an eval config from configs/eval/ using the item picker."""
    from poregen.eval.config import list_eval_configs
    configs = list_eval_configs(repo_root)
    if not configs:
        _show_message(stdscr, "Eval Config", "No eval configs found in configs/eval/.")
        return None
    items = [(c["id"], f"tier: {c['tier']}") for c in configs]
    selected = _select_item(
        stdscr,
        "Select Eval Config",
        items,
        empty_message="No eval configs found.",
    )
    if selected is None:
        return None
    return configs[selected]["id"]


def _execute_action(
    stdscr: Any,
    *,
    repo_root: Path,
    state: DashboardState,
    action: ActionOption,
    entries_by_mode: dict[str, list[BrowserEntry]],
    cache: dict[str, dict[str, DetailDocument]],
) -> tuple[DeferredAction | None, dict[str, list[BrowserEntry]], dict[str, dict[str, DetailDocument]]]:
    entry = _selected_entry(state, entries_by_mode)

    if action.action_id == "help":
        _show_help(stdscr)
        state.status = "Help closed."
        return None, entries_by_mode, cache

    if action.action_id == "view":
        document = _current_document(state, repo_root, entries_by_mode, cache)
        _view_text(stdscr, document.title, document.full_text)
        state.status = f"Viewing {document.title}."
        return None, entries_by_mode, cache

    if action.action_id == "refresh":
        entries_by_mode = _refresh_dashboard_data(repo_root)
        cache = {"experiments": {}, "runs": {}}
        state.status = (
            f"Refreshed: {len(entries_by_mode['experiments'])} experiments, "
            f"{len(entries_by_mode['runs'])} runs."
        )
        return None, entries_by_mode, cache

    if action.action_id == "switch_runs":
        state.mode = "runs"
        state.focus = "browser"
        state.status = "Browsing runs."
        return None, entries_by_mode, cache

    if action.action_id == "switch_experiments":
        state.mode = "experiments"
        state.focus = "browser"
        state.status = "Browsing experiments."
        return None, entries_by_mode, cache

    if entry is None:
        state.status = "Select an item first."
        return None, entries_by_mode, cache

    if action.action_id == "run":
        return DeferredAction("run", entry.identifier), entries_by_mode, cache

    if action.action_id == "resume_latest":
        return DeferredAction("resume", entry.identifier, "latest.ckpt"), entries_by_mode, cache

    if action.action_id == "resume_pick":
        run_dir = next((path for path in list_run_directories(repo_root=repo_root) if path.name == entry.identifier), None)
        if run_dir is None:
            state.status = f"Run not found: {entry.identifier}"
            return None, entries_by_mode, cache
        checkpoint_name = _choose_checkpoint_interactive(stdscr, run_dir)
        if checkpoint_name is None:
            state.status = "Checkpoint selection cancelled."
            return None, entries_by_mode, cache
        return DeferredAction("resume", entry.identifier, checkpoint_name), entries_by_mode, cache

    if action.action_id == "clone":
        try:
            clone_result = _guided_clone_experiment(stdscr, repo_root, entry.identifier)
        except Exception as exc:
            _show_message(stdscr, "Create Variant", f"Failed to create variant:\n{exc}")
            state.status = f"Variant creation failed for {entry.identifier}."
            return None, entries_by_mode, cache
        if clone_result is None:
            state.status = "Variant creation cancelled."
            return None, entries_by_mode, cache

        target_id, target_path = clone_result
        entries_by_mode, cache = _refresh_after_clone(repo_root=repo_root, state=state, target_id=target_id)
        state.status = f"Saved {target_id} to {Path(target_path).name}."
        return None, entries_by_mode, cache

    if action.action_id == "eval_best":
        run_dir = next(
            (path for path in list_run_directories(repo_root=repo_root) if path.name == entry.identifier),
            None,
        )
        if run_dir is None:
            state.status = f"Run not found: {entry.identifier}"
            return None, entries_by_mode, cache
        # Use best.ckpt if it exists, otherwise fall back to latest.ckpt
        ckpt = "best.ckpt" if (run_dir / "best.ckpt").exists() else "latest.ckpt"
        return DeferredAction("eval", entry.identifier, ckpt, "eval/r03_base"), entries_by_mode, cache

    if action.action_id == "eval_custom":
        run_dir = next(
            (path for path in list_run_directories(repo_root=repo_root) if path.name == entry.identifier),
            None,
        )
        if run_dir is None:
            state.status = f"Run not found: {entry.identifier}"
            return None, entries_by_mode, cache
        checkpoint_name = _choose_checkpoint_interactive(stdscr, run_dir)
        if checkpoint_name is None:
            state.status = "Checkpoint selection cancelled."
            return None, entries_by_mode, cache
        eval_config_id = _select_eval_config_interactive(stdscr, repo_root)
        if eval_config_id is None:
            state.status = "Eval config selection cancelled."
            return None, entries_by_mode, cache
        return DeferredAction("eval", entry.identifier, checkpoint_name, eval_config_id), entries_by_mode, cache

    state.status = f"Unhandled action: {action.label}"
    return None, entries_by_mode, cache


def _dashboard_impl(stdscr: Any, repo_root: Path) -> DeferredAction | None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    theme = _init_theme()
    state = DashboardState()
    entries_by_mode = _refresh_dashboard_data(repo_root)
    cache: dict[str, dict[str, DetailDocument]] = {"experiments": {}, "runs": {}}
    state.status = (
        f"Loaded {len(entries_by_mode['experiments'])} experiments and "
        f"{len(entries_by_mode['runs'])} runs."
    )

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 24 or width < 100:
            _safe_add_line(stdscr, 2, 2, "The dashboard needs at least 100x24 terminal space.", theme["heading"])
            _safe_add_line(stdscr, 4, 2, "Resize the terminal or use non-interactive subcommands.", curses.A_DIM)
            _safe_add_line(stdscr, 6, 2, "Press q to quit.", curses.A_DIM)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return None
            continue

        current_entry = _selected_entry(state, entries_by_mode)
        current_label = current_entry.identifier if current_entry is not None else None
        counts = {mode: len(entries) for mode, entries in entries_by_mode.items()}
        _draw_header(stdscr, state, counts, current_label, theme)

        content_y = 4
        content_height = height - 7
        browser_width = max(34, min(48, width // 3))
        right_x = browser_width + 3
        right_width = width - browser_width - 5
        detail_min_height = 10
        action_min_height = 7
        action_height = max(action_min_height, min(13, content_height // 3))
        action_height = min(action_height, max(action_min_height, content_height - detail_min_height))
        detail_height = content_height - action_height
        browser_window = _draw_panel(
            stdscr,
            content_y,
            2,
            content_height,
            browser_width,
            "Experiments" if state.mode == "experiments" else "Runs",
            theme=theme,
            focused=state.focus == "browser",
        )
        detail_window = _draw_panel(
            stdscr,
            content_y,
            right_x,
            detail_height,
            right_width,
            "Details",
            theme=theme,
            focused=state.focus == "details",
        )
        action_window = _draw_panel(
            stdscr,
            content_y + detail_height,
            right_x,
            action_height,
            right_width,
            "Options",
            theme=theme,
            focused=state.focus == "actions",
        )

        entries = entries_by_mode[state.mode]
        actions = _action_options(state.mode)
        state.browser_offset[state.mode] = _draw_browser_panel(
            browser_window,
            entries,
            selected_index=state.browser_index[state.mode],
            offset=state.browser_offset[state.mode],
            focus=state.focus,
            theme=theme,
        )
        state.action_offset[state.mode] = _draw_action_panel(
            action_window,
            actions,
            selected_index=state.action_index[state.mode],
            offset=state.action_offset[state.mode],
            focus=state.focus,
            theme=theme,
        )

        document = _current_document(state, repo_root, entries_by_mode, cache)
        detail_title = document.title
        _safe_add_line(
            detail_window,
            0,
            2,
            f" {detail_title} ",
            theme["panel_title_focus"] if state.focus == "details" else theme["panel_title"],
        )
        state.detail_offset[state.mode] = _draw_detail_panel(
            detail_window,
            document,
            offset=state.detail_offset[state.mode],
            focus=state.focus,
            theme=theme,
        )

        _draw_footer(stdscr, state, theme)
        stdscr.refresh()
        browser_window.refresh()
        detail_window.refresh()
        action_window.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return None
        if key in (9, curses.KEY_BTAB):
            _cycle_focus(state, 1)
            continue
        if key == curses.KEY_LEFT:
            _cycle_focus(state, -1)
            continue
        if key == curses.KEY_RIGHT:
            _cycle_focus(state, 1)
            continue
        if key in (curses.KEY_UP, ord("k")):
            if state.focus == "browser":
                _move_browser_selection(state, -1, entries)
            elif state.focus == "actions":
                _move_action_selection(state, -1, actions)
            else:
                _scroll_detail(state, -1)
            continue
        if key in (curses.KEY_DOWN, ord("j")):
            if state.focus == "browser":
                _move_browser_selection(state, 1, entries)
            elif state.focus == "actions":
                _move_action_selection(state, 1, actions)
            else:
                _scroll_detail(state, 1)
            continue
        if key == curses.KEY_PPAGE:
            if state.focus == "browser":
                _move_browser_selection(state, -8, entries)
            elif state.focus == "actions":
                _move_action_selection(state, -4, actions)
            else:
                _scroll_detail(state, -12)
            continue
        if key == curses.KEY_NPAGE:
            if state.focus == "browser":
                _move_browser_selection(state, 8, entries)
            elif state.focus == "actions":
                _move_action_selection(state, 4, actions)
            else:
                _scroll_detail(state, 12)
            continue
        if key == curses.KEY_HOME:
            if state.focus == "browser":
                state.browser_index[state.mode] = 0
                state.detail_offset[state.mode] = 0
            elif state.focus == "actions":
                state.action_index[state.mode] = 0
            else:
                state.detail_offset[state.mode] = 0
            continue
        if key == curses.KEY_END:
            if state.focus == "browser" and entries:
                state.browser_index[state.mode] = len(entries) - 1
                state.detail_offset[state.mode] = 0
            elif state.focus == "actions" and actions:
                state.action_index[state.mode] = len(actions) - 1
            else:
                state.detail_offset[state.mode] = 10**9
            continue
        if key in (10, 13, curses.KEY_ENTER):
            if state.focus == "browser":
                state.focus = "actions"
                state.status = "Choose an option for the selected item."
                continue
            if state.focus == "details":
                _view_text(stdscr, document.title, document.full_text)
                continue
            action = actions[state.action_index[state.mode]] if actions else None
            if action is None:
                state.status = "No action available."
                continue
            deferred, entries_by_mode, cache = _execute_action(
                stdscr,
                repo_root=repo_root,
                state=state,
                action=action,
                entries_by_mode=entries_by_mode,
                cache=cache,
            )
            if deferred is not None:
                return deferred
            continue
        if key == ord("?"):
            _show_help(stdscr)
            continue


def _run_eval_action(
    *,
    run_name: str,
    checkpoint_name: str,
    eval_config_ref: str,
    repo_root: Path,
) -> None:
    """Resolve paths, load eval config, and call ``run_eval``.

    This helper is shared between the TUI deferred-action path and the
    ``eval`` CLI subcommand so that both surfaces use identical logic.

    Parameters
    ----------
    run_name : str
        Run directory name (e.g. ``r03-run-0002-…``).
    checkpoint_name : str
        Checkpoint filename (e.g. ``best.ckpt``) or absolute path.
    eval_config_ref : str
        Eval config id (e.g. ``"eval/r03_base"``) or YAML path.
    repo_root : Path
    """
    from poregen.eval.config import load_eval_config
    from poregen.eval.runner import run_eval

    # Resolve run directory
    run_dirs = list_run_directories(repo_root=repo_root)
    run_dir = next((d for d in run_dirs if d.name == run_name), None)
    if run_dir is None:
        # Maybe the caller passed a full path
        candidate = Path(run_name)
        if candidate.is_dir():
            run_dir = candidate
        else:
            raise FileNotFoundError(
                f"Run directory not found: {run_name!r}. "
                f"Available: {[d.name for d in run_dirs]}"
            )

    # Resolve checkpoint path
    ckpt_path = Path(checkpoint_name)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load eval config
    eval_cfg = load_eval_config(eval_config_ref, repo_root=repo_root)
    logger.info(
        "Starting eval: run=%s  checkpoint=%s  tier=%s",
        run_name, ckpt_path.name, eval_cfg.tier,
    )

    out_dir = run_eval(run_dir, ckpt_path, eval_cfg, repo_root)
    print(f"Eval complete. Results: {out_dir}")


def interactive_menu(repo_root: Path) -> None:
    """Open the interactive dashboard UI."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "The interactive menu requires a TTY. Use explicit subcommands like "
            "'list', 'show', 'run', or 'resume' in non-interactive contexts."
        )

    action = curses.wrapper(lambda stdscr: _dashboard_impl(stdscr, repo_root))
    if action is None:
        return
    if action.kind == "run":
        run_dir = run_experiment(action.primary, repo_root=repo_root)
        print(run_dir)
        return
    if action.kind == "resume":
        run_dir = resume_run(
            action.primary,
            checkpoint_name=action.secondary or "latest.ckpt",
            repo_root=repo_root,
        )
        print(run_dir)
        return
    if action.kind == "eval":
        _run_eval_action(
            run_name=action.primary,
            checkpoint_name=action.secondary or "best.ckpt",
            eval_config_ref=action.tertiary or "eval/r03_base",
            repo_root=repo_root,
        )
        return
    raise ValueError(f"Unknown deferred action: {action.kind}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    repo_root = find_repo_root(__file__)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.command == "menu":
        interactive_menu(repo_root)
        return

    if args.command == "list":
        _print_experiments(repo_root)
        return

    if args.command == "runs":
        _print_runs(repo_root)
        return

    if args.command == "show":
        resolved = resolve_experiment(args.experiment, repo_root=repo_root)
        print(yaml.safe_dump(resolved.cfg, sort_keys=False))
        return

    if args.command == "run":
        run_dir = run_experiment(args.experiment, repo_root=repo_root)
        print(run_dir)
        return

    if args.command == "resume":
        run_dir = resume_run(args.run_ref, checkpoint_name=args.checkpoint, repo_root=repo_root)
        print(run_dir)
        return

    if args.command == "clone":
        target = args.target
        if target is None:
            target = _prompt("New experiment id (example: r03/z24): ")
        if not target:
            raise ValueError("A target experiment id is required to clone an experiment.")
        path = clone_experiment_definition(args.source, target, repo_root=repo_root)
        print(path)
        return

    if args.command == "eval":
        _run_eval_action(
            run_name=args.run_ref,
            checkpoint_name=args.checkpoint,
            eval_config_ref=args.eval_config,
            repo_root=repo_root,
        )
        return

    raise ValueError(f"Unhandled command: {args.command}")

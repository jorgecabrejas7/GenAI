#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


META_KEYS = frozenset({"step", "split", "elapsed", "full_eval", "n_batches"})
SKIP_KEYS_BY_SPLIT = {
    "val": frozenset({"sharpness_gt"}),
    "val_full": frozenset({"sharpness_gt"}),
    "test": frozenset({"sharpness_gt"}),
    "test_full": frozenset({"sharpness_gt"}),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _iter_records(run_dir: Path) -> list[dict[str, Any]]:
    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for path in (run_dir / "log.jsonl", run_dir / "metrics.jsonl"):
        for record in _load_jsonl(path):
            merged[(int(record["step"]), str(record["split"]))] = record
    return sorted(merged.values(), key=lambda record: (int(record["step"]), str(record["split"])))


def _restore_record(writer: SummaryWriter, record: dict[str, Any]) -> None:
    split = str(record["split"])
    step = int(record["step"])
    skip_keys = SKIP_KEYS_BY_SPLIT.get(split, frozenset())

    for key, value in record.items():
        if key in META_KEYS or key in skip_keys:
            continue
        if isinstance(value, bool):
            writer.add_scalar(f"{split}/{key}", int(value), step)
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{split}/{key}", value, step)
            continue
        if key == "kl_per_channel" and isinstance(value, list) and value:
            tensor = torch.tensor(value, dtype=torch.float32)
            writer.add_histogram(f"{split}/kl_per_channel", tensor, step)
            if split != "train":
                for i, channel_value in enumerate(value):
                    writer.add_scalar(f"{split}/kl_ch{i:02d}", channel_value, step)


def rebuild_tensorboard(
    run_dir: Path,
    output_dir: Path,
    *,
    min_step: int | None = None,
    max_step: int | None = None,
) -> tuple[Counter[str], int]:
    if not (run_dir / "log.jsonl").exists():
        raise FileNotFoundError(f"Missing log file: {run_dir / 'log.jsonl'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = _iter_records(run_dir)
    if min_step is not None:
        records = [record for record in records if int(record["step"]) >= min_step]
    if max_step is not None:
        records = [record for record in records if int(record["step"]) <= max_step]
    counts: Counter[str] = Counter()
    writer = SummaryWriter(str(output_dir), max_queue=1, flush_secs=1)
    try:
        for record in records:
            _restore_record(writer, record)
            counts[str(record["split"])] += 1
    finally:
        writer.flush()
        writer.close()
    return counts, len(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild TensorBoard scalar history from a run's JSONL logs."
    )
    parser.add_argument("run_dir", help="Run directory containing log.jsonl.")
    parser.add_argument(
        "--output-dir",
        help="Destination for rebuilt TensorBoard events. Defaults to <run_dir>/tb_restored.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        help="Only restore records with step >= this value.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="Only restore records with step <= this value.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_dir / "tb_restored").resolve()
    )
    counts, total = rebuild_tensorboard(
        run_dir,
        output_dir,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    print(f"Rebuilt {total} log records into {output_dir}")
    for split, count in sorted(counts.items()):
        print(f"  {split}: {count}")
    print("Limitations:")
    print("  - Restores scalar history from log.jsonl.")
    print("  - Does not restore TensorBoard-only artifacts that were never stored in JSONL,")
    print("    such as previous image panels, Monte Carlo images, and some histograms.")


if __name__ == "__main__":
    main()

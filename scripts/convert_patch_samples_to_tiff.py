#!/usr/bin/env python
"""Convert legacy patch-sample npz archives into ImageJ-readable TIFF stacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from poregen.training.sample_export import convert_patch_sample_archives_under


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="runs",
        help="Root directory to scan for legacy sample archives (default: runs).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the original .npz and *_meta.json files after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    converted = convert_patch_sample_archives_under(
        root,
        delete_source=not args.keep_source,
    )

    print(f"Converted {len(converted)} archive(s) under {root}.")
    for path in converted:
        print(path)


if __name__ == "__main__":
    main()

"""Launch or resume an LDM training run.

Usage
-----
# New run:
python scripts/train_ldm.py ldm01/base

# Resume:
python scripts/train_ldm.py --resume ldm01/base
python scripts/train_ldm.py --resume runs/ldm/ldm01-run-001-...
python scripts/train_ldm.py --resume runs/ldm/ldm01-run-001-... --checkpoint checkpoints/ldm_step00010000.ckpt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a PoreGen LDM.")
    ap.add_argument("experiment", help="Experiment ref (e.g. 'ldm01/base') or run dir when resuming")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run")
    ap.add_argument(
        "--checkpoint", default="checkpoints/latest.ckpt",
        help="Checkpoint name within run dir when resuming (default: checkpoints/latest.ckpt)",
    )
    args = ap.parse_args()

    # Ensure src is on the path when run directly
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))

    from poregen.experiments.train_ldm import resume_ldm_run, run_ldm_experiment

    if args.resume:
        run_dir = resume_ldm_run(args.experiment, checkpoint_name=args.checkpoint)
        print(f"Resumed run completed: {run_dir}")
    else:
        run_dir = run_ldm_experiment(args.experiment)
        print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()

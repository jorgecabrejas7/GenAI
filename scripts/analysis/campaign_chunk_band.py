#!/usr/bin/env python
"""The chunk-plane band of an eval-v4 campaign's 1024 sampler volumes. (CPU, seconds.)

Campaign 18's headline band row (the -8 slab and the +0 slab as a fraction of
the volume's porosity, mean and worst plane) was filled by hand from the
campaign-17 scorer. This reads it the same way for any campaign root, so the
row is traceable: each 1024 case under ``<root>/sampler/volumes`` is scored by
``chunk_band_trial_report.measure`` — the campaign-17 metric, itself
``poregen.eval_v4.metrics.chunk_band_profile`` — and summarised over the cases.

Validated on split_v3 campaign 18: it returns the README's numbers exactly
(-8 slab 1.025, worst 0.834-0.974; +0 slab 0.837, worst 0.663-0.771).

Usage:
    python scripts/analysis/campaign_chunk_band.py runs/campaigns/18-eval-v4-final \\
        runs/campaigns/18-eval-v4-final-v4
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "chunk_band_trial_report", HERE / "chunk_band_trial_report.py")
_cbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cbt)

KEYS = ("ratio_-8", "ratio_-8_worst", "ratio_+0", "ratio_+0_worst",
        "grey_chunk_ratio", "phi_error")


def summarise(root: Path) -> dict[str, dict[str, float]]:
    cases = sorted((root / "sampler" / "volumes").glob("1024_*"))
    if not cases:
        raise SystemExit(f"no 1024 sampler cases under {root}")
    rows = [_cbt.measure(d) for d in cases]
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "min": float(np.min([r[k] for r in rows])),
                "max": float(np.max([r[k] for r in rows])),
                "n": len(rows)} for k in KEYS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    for root in ap.parse_args().roots:
        print(root)
        for k, s in summarise(root).items():
            print(f"  {k:17s} mean {s['mean']:.3f}  min {s['min']:.3f}  "
                  f"max {s['max']:.3f}  n {s['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

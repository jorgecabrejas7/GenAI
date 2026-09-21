"""The VRRAE family on one validation set, re-rendered. (CPU, minutes.)

Runs `vae_val_l1.py`'s measurement over every VRRAE run and the spatial
baseline, on the SAME held-out patches, and writes campaign 28's table. Called
after each new VRRAE run finishes so the collaborator gets one growing table
rather than a series of numbers that have to be reconciled.

Every row is on the 5 split_v3 validation volumes that split_v2 never trained
on, because six of the eleven are in split_v2's TRAIN set and scoring a
split_v2-trained model on all eleven flatters it.

Usage:
    python scripts/analysis/vrrae_family_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The spatial baseline every VRRAE row is read against, and the reason the
#: harness can be trusted: it reproduces this run's own logged value.
BASELINE_GLOB = "runs/vae/r08-run-0004-*"
FAMILY_GLOBS = ("runs/vae/vrrae0*-run-*", "runs/vae/vrrae-run-*")


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    from vae_val_l1 import evaluate

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "28-vrrae-family")
    ap.add_argument("--n-batches", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for g in (*FAMILY_GLOBS, BASELINE_GLOB):
        for d in sorted(REPO.glob(g)):
            if (d / "best.ckpt").exists() and (d / "resolved_config.yaml").exists():
                runs.append(d)

    rows = []
    for d in runs:
        try:
            rows.append(evaluate(d, "split_v3", args.n_batches, 32,
                                 exclude_train_of="split_v2"))
        except Exception as exc:                          # noqa: BLE001
            # A run that cannot be measured is reported as such rather than
            # dropped: a missing row reads as "not run yet", which is a
            # different thing from "would not load".
            rows.append({"run": d.name, "error": str(exc)[:200]})

    (args.out / "family_table.json").write_text(json.dumps(rows, indent=2) + "\n")

    hdr = (f"| run | evaluated on | L1 | predicting the mean | error removed |\n"
           f"|---|---|---|---|---|\n")
    body = ""
    for r in sorted(rows, key=lambda r: r.get("fraction_of_baseline_error_removed", -1)):
        if "error" in r:
            body += f"| `{r['run'][:44]}` | — | — | — | **could not load**: {r['error'][:60]} |\n"
            continue
        body += (f"| `{r['run'][:44]}` | {r['evaluated_on']} | {r['l1']:.4f} | "
                 f"{r['constant_prediction_baseline']:.4f} | "
                 f"**{r['fraction_of_baseline_error_removed']:.1%}** |\n")
    (args.out / "family_table.md").write_text(
        "# The VRRAE family on one validation set\n\n"
        f"{len(rows)} runs, {args.n_batches * 32} patches of split_v3, restricted "
        "to the 5 validation volumes split_v2 never trained on.\n\n"
        "`error removed` is the fraction of the constant-prediction baseline's "
        "error the model removes. It is the column to read: an L1 of 0.066 "
        "against 0.034 sounds like a factor of two until you see that "
        "predicting the dataset mean already scores 0.076.\n\n" + hdr + body)
    print(hdr + body)
    print(f"-> {args.out}/family_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

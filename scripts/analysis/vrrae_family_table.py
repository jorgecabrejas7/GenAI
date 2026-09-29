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


def _needs_latest(run: Path) -> bool:
    """True when best.ckpt lacks the RR inference basis that latest.ckpt carries."""
    import torch
    key = "bottleneck.rr.inference_basis"
    best = torch.load(run / "best.ckpt", map_location="cpu", weights_only=False)["model"]
    if key in best or not (run / "latest.ckpt").exists():
        return False
    latest = torch.load(run / "latest.ckpt", map_location="cpu", weights_only=False)["model"]
    return key in latest


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
            # The RR finalisation pass stamps its inference basis on latest.ckpt,
            # after best.ckpt was written; an RR run's row therefore comes from
            # latest.ckpt (the finalised model), everything else from best.ckpt.
            ckpt = "latest.ckpt" if _needs_latest(d) else "best.ckpt"
            row = evaluate(d, "split_v3", args.n_batches, 32,
                           exclude_train_of="split_v2", ckpt=ckpt)
            row["checkpoint"] = ckpt
            rows.append(row)
        except Exception as exc:                          # noqa: BLE001
            # A run that cannot be measured is reported as such rather than
            # dropped: a missing row reads as "not run yet", which is a
            # different thing from "would not load".
            rows.append({"run": d.name, "error": str(exc)[:200]})

    (args.out / "family_table.json").write_text(json.dumps(rows, indent=2) + "\n")

    def name(run: str) -> str:
        """Keep the TAIL as well as the head. Truncating at 44 characters made
        the three vrrae-run-0008 rows — as trained, and the two decoder
        fine-tunes — print identically, which is a table that cannot be read."""
        return run if len(run) <= 46 else f"{run[:30]}...{run[-13:]}"

    hdr = ("| run | L1 | error removed | texture, interior | texture, all "
           "| sharpness, interior | sharpness, all |\n"
           "|---|---|---|---|---|---|---|\n")
    body = ""
    # SORTED BY TEXTURE, not by error removed. On this family the two orders
    # disagree: V0 and A0 beat the spatial baseline on error removed while
    # reproducing no texture at all, so sorting by that column would present
    # the flat bottleneck as the better one.
    for r in sorted(rows, key=lambda r: (r.get("texture_corr_interior") is None,
                                         r.get("texture_corr_interior") or -2)):
        if "error" in r:
            body += (f"| `{name(r['run'])}` | — | — | — | — | — | "
                     f"**could not load**: {r['error'][:60]} |\n")
            continue
        ti = r.get("texture_corr_interior")
        si = r.get("sharpness_interior")
        body += (f"| `{name(r['run'])}` | {r['l1']:.4f} | "
                 f"{r['fraction_of_baseline_error_removed']:.1%} | "
                 f"**{'—' if ti is None else f'{ti:+.3f}'}** | "
                 f"{r.get('texture_corr', float('nan')):+.3f} | "
                 f"{'—' if si is None else f'{si:.3f}'} | "
                 f"{r.get('sharpness', float('nan')):.3f} |\n")
    (args.out / "family_table.md").write_text(
        "# The VRRAE family on one validation set\n\n"
        f"{len(rows)} runs, {args.n_batches * 32} patches of split_v3, restricted "
        "to the 5 validation volumes split_v2 never trained on.\n\n"
        "`error removed` RANKS THE MEAN GREY LEVEL. `texture` RANKS THE "
        "RECONSTRUCTION. The two disagree, and that is the point of this "
        "table: V0 and A0 remove MORE of the constant-prediction error than "
        "the spatial baseline while returning a flat grey block wherever "
        "there is no specimen edge. L1 is dominated by a patch's mean "
        "brightness and by the air/material boundary, so it cannot referee "
        "this family.\n\n"
        "`texture` is the correlation between reconstruction and input after "
        "each patch's own mean is removed — the same quantity the recon "
        "figure prints, computed here on every patch. `interior` is the mean "
        "over patches with NO air voxel (by the dataset's own label, class "
        "2), which is the honest one: a patch touching the specimen edge "
        "carries the single structure every model in this family reproduces, "
        "and including those patches lifts a grey model far above what it "
        "earns in the material. Rows are sorted by it.\n\n"
        "`sharpness` is a THIRD question, and it is not implied by the other "
        "two: the standard deviation of the reconstruction's 3-D Laplacian "
        "over the input's. 1.0 is as sharp as the input; 0.6 means a third of "
        "the fine contrast has gone. A reconstruction can put the pattern in "
        "the right place, and so correlate well, and still be a smoothed "
        "version of it.\n\n"
        "READ SHARPNESS ONLY NEXT TO TEXTURE. On its own it is not a quality "
        "measure at all: the 400-step a0 trial scores the HIGHEST sharpness "
        "in this table, 0.909, with a texture correlation of -0.001. It emits "
        "fine contrast at almost exactly the input's level and in none of the "
        "right places. Sharpness says how much detail is there; texture says "
        "whether it is the input's detail.\n\n" + hdr + body)
    print(hdr + body)
    print(f"-> {args.out}/family_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

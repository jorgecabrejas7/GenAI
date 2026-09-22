"""Move aside the cases a code fix invalidated, so they are regenerated. (CPU.)

`eval_v4 generate` SKIPS any case whose `manifest.json` already exists — which
is what makes a long campaign resumable, and which silently turned a
regeneration stage into a four-second no-op: the chain logged rc=0 and nothing
was rebuilt.

So a regeneration has to MOVE THE OLD CASES ASIDE FIRST. They are moved, never
deleted: they are the record of what the published numbers were built from, and
a superseded result in this project is labelled rather than destroyed.

Two code fixes invalidate volumes:
  * the window porosity was `mean(phi) * mean(m)` instead of `mean(phi * m)`,
    which affects any case whose requested field VARIES across a window;
  * the coherent field was smoothed to twice its requested correlation length,
    and its priors were refitted on the train split.

Both touch only cases with a painted or coherent FIELD. A uniform request is
unaffected — `mean(phi0 * m)` is `phi0 * mean(m)` exactly — so this moves the
field-carrying cases and leaves the rest, and prints which.

Usage:
    python scripts/analysis/archive_stale_cases.py --root <campaign> --assessment <a>
    python scripts/analysis/archive_stale_cases.py --root <campaign> --assessment <a> --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: Where the superseded cases go. One directory per assessment, beside the
#: live one, so the pair is obvious in a listing.
ARCHIVE = "volumes_prefix_v1"


def main() -> int:
    from poregen.eval_v4.cases import build_cases

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--assessment", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="actually move; without it, list and change nothing")
    args = ap.parse_args()

    # The case list is the authority on which cases carry a field, not the
    # manifests: a manifest records what was requested, and what we need is
    # which cases the CURRENT code would build with a field.
    stale = {c.name for c in build_cases(args.assessment) if c.field_fn is not None}
    vols = args.root / args.assessment / "volumes"
    if not vols.is_dir():
        print(f"no volumes under {vols}")
        return 0

    present = sorted(d for d in vols.iterdir()
                     if (d / "manifest.json").exists())
    move = [d for d in present if d.name in stale]
    keep = [d for d in present if d.name not in stale]

    print(f"{args.assessment} under {args.root.name}:")
    print(f"  {len(present)} cases present, {len(move)} carry a field and are stale")
    for d in keep:
        print(f"    keep    {d.name}")
    for d in move:
        print(f"    ARCHIVE {d.name}")
    if not move:
        print("  nothing to move")
        return 0
    if not args.apply:
        print("\n  dry run; pass --apply to move them")
        return 0

    dest = args.root / args.assessment / ARCHIVE
    dest.mkdir(parents=True, exist_ok=True)
    for d in move:
        target = dest / d.name
        if target.exists():
            print(f"  REFUSING: {target} already exists; move or remove it first")
            return 1
        shutil.move(str(d), str(target))
    (dest / "WHY.md").write_text(
        "# Superseded cases\n\n"
        "Moved here so `eval_v4 generate` would rebuild them: it skips any case "
        "whose `manifest.json` exists, which turned a regeneration stage into a "
        "no-op.\n\n"
        "These were generated before two fixes:\n\n"
        "1. **The window porosity was wrong.** `cond_por` is pore over the whole "
        "patch, and converting a material-porosity request to it is a per-cell "
        "product `mean(phi*m)`. The sampler used `mean(phi)*mean(m)`, which drops "
        "the covariance between the requested field and the specimen shape. On a "
        "window half exterior air at 0.10 and half material at 0.01 the true "
        "value is 0.005 and the old formula gave 0.0275.\n\n"
        "2. **The coherent field was twice as smooth as requested**, because "
        "smoothed white noise reaches 1/e at 2*sigma and the generator used "
        "sigma = L/stride. Its priors were also refitted on the train split "
        "(the originals were fitted on all 80 volumes, which the generator "
        "reads at sampling time).\n\n"
        "Only cases carrying a painted or coherent FIELD are here. A uniform "
        "request is unaffected: `mean(phi0*m)` equals `phi0*mean(m)` exactly.\n")
    print(f"\n  moved {len(move)} case(s) -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

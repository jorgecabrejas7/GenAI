"""What each conditioning input is worth, measured by removing it. (GPU.)

PREPARED, NOT RUN. Queue it after rf-64.

ldm06 is conditioned on four things: a porosity request, a specimen-envelope
map, a ply-orientation profile, and the six neighbour faces. Every eval-v4
table says the model OBEYS them. None says what any one of them CONTRIBUTES —
a model that ignored the envelope entirely would still score well on a case
whose envelope is a full box.

So each input is suppressed AT INFERENCE on the same requests, with the same
weights and the same seed, and the request-bearing metrics are read against the
intact run. The difference is what that input was doing.

  intact            everything as production
  drop_material     the painted envelope replaced by "full" — the model is no
                    longer told where the specimen is
  drop_layup        a CONSTANT orientation instead of the ply profile, so
                    theta carries no ply structure
  drop_neighbours   neighbour_mode "unknown": every window sees six UNKNOWN
                    faces, as if nothing had been solved around it

WHY INFERENCE-TIME AND NOT RETRAINING. Retraining without an input answers a
different question — what a model trained without it would learn — and costs a
run per input. This one asks what the trained model USES, which is the question
the eval tables raise and cannot answer.

WHAT IT CANNOT SHOW. An input the model ignores looks identical to an input
whose effect the metric cannot see. A null result here is evidence about the
pair (input, metric), not about the input alone, and the report says so per row.

Usage:
    python scripts/analysis/conditioning_ablation.py --model <run> --ckpt latest \\
        --out runs/campaigns/23-conditioning-ablation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("cond_ablation")

#: The ablations. Each is a change to the CaseSpec, applied to every base case.
ABLATIONS = ("intact", "drop_material", "drop_layup", "drop_neighbours")

#: Base cases, chosen so every suppressed input has a metric that can see it:
#: a painted envelope for drop_material, a real layup for drop_layup, a
#: multi-chunk volume for drop_neighbours, and a painted field for the porosity
#: request. A full-box 192-cubed case would be blind to three of the four.
BASE_CASES: tuple[tuple[str, str], ...] = (
    ("geometry", "notch_hole_seed101"),
    ("surface", "rough_192_ddim50_seed101"),
    ("layup", "A_seed101"),
    ("porosity_local", "coherent_ddim50_seed101"),
    ("multichunk", "box384_ddim50_seed101"),
)

#: A single angle, so theta_deg is constant through the depth and carries no
#: ply structure. 0 rather than a mean: an average of the layup would still be
#: an orientation the model can use.
FLAT_LAYUP = (0,)


def ablate(spec, kind: str):
    """A copy of ``spec`` with one conditioning input suppressed."""
    if kind == "intact":
        return spec
    if kind == "drop_material":
        # No painted envelope: the manifest then records requested_material
        # "full" and the model is not told where the specimen is.
        return dataclasses.replace(spec, material_fn=None)
    if kind == "drop_layup":
        return dataclasses.replace(spec, layup=FLAT_LAYUP)
    if kind == "drop_neighbours":
        return dataclasses.replace(spec, neighbour_mode="unknown")
    raise KeyError(f"unknown ablation {kind!r}; choose from {ABLATIONS}")


def is_noop(base, kind: str) -> str | None:
    """Why this ablation changes nothing for this case, or None if it does.

    THE DRY RUN CAUGHT THIS. `drop_material` on a case that already requests
    "full" produces a spec identical to `intact`: three of the five base cases
    are full-box, so three of the twelve planned drops were duplicates of their
    own control. Generating them would have spent GPU on nothing and — worse —
    put three rows in the table that look like measurements of an input's
    contribution and are actually the same volume twice.
    """
    if kind == "drop_material" and base.material_fn is None:
        return "the case already requests 'full' material, so there is no envelope to remove"
    if kind == "drop_layup" and len(base.layup) <= 1:
        return "the case already has a single-angle layup, so there is no ply profile to remove"
    if kind == "drop_neighbours" and base.neighbour_mode != "canvas":
        return f"the case already uses neighbour_mode {base.neighbour_mode!r}"
    return None


def build_specs(repo: Path):
    """(ablation, spec) for every base case, renamed so cases never collide.

    Ablations that would be no-ops for a case are SKIPPED and reported, never
    generated: an ablation identical to its control is not a measurement.
    """
    from poregen.eval_v4.cases import build_cases

    out, skipped = [], []
    for assessment, case in BASE_CASES:
        specs = [s for s in build_cases(assessment, repo) if s.name == case]
        if not specs:
            raise KeyError(f"{assessment}/{case} is not in the case list")
        base = specs[0]
        for kind in ABLATIONS:
            why = is_noop(base, kind) if kind != "intact" else None
            if why:
                skipped.append({"base": f"{assessment}/{case}", "ablation": kind,
                                "why": why})
                continue
            s = ablate(base, kind)
            out.append((kind, dataclasses.replace(
                s,
                name=f"{assessment}__{case}__{kind}",
                assessment="conditioning_ablation",
                notes={**dict(base.notes or {}),
                       "ablation": kind,
                       "base_assessment": assessment,
                       "base_case": case,
                       "exploratory": True,
                       "off_gates_because":
                           "a conditioning input was suppressed at inference, so "
                           "this volume is a probe of what that input contributes "
                           "and not a gated result"},
            )))
    return out, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "23-conditioning-ablation")
    ap.add_argument("--only", nargs="*", default=None, help="ablation names to run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    specs, skipped = build_specs(REPO)
    if args.only:
        specs = [(k, s) for k, s in specs if k in set(args.only)]
    logger.info("%d cases from %d base cases; %d ablations skipped as no-ops",
                len(specs), len(BASE_CASES), len(skipped))
    for row in skipped:
        logger.info("  SKIP %-44s %s", f"{row['base']} {row['ablation']}", row["why"])
    for kind, s in specs:
        logger.info("  %-52s nb=%-8s material=%-7s layup=%d plies",
                    s.name, s.neighbour_mode,
                    "painted" if s.material_fn else "full", len(s.layup))
    if args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(
            {"n_cases": len(specs),
             "ablations": list(ABLATIONS),
             "base_cases": [f"{a}/{c}" for a, c in BASE_CASES],
             "cases": [s.name for _, s in specs],
             "skipped_as_noop": skipped}, indent=2) + "\n")
        logger.info("--dry-run: plan written, nothing generated")
        return 0

    from poregen.eval_v4.generate import VolumeRunner

    runner = VolumeRunner(args.model, args.ckpt, weights="ema", repo=REPO)
    for kind, s in specs:
        logger.info("GENERATE %s", s.name)
        runner.run(s, args.out)
    logger.info("done -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

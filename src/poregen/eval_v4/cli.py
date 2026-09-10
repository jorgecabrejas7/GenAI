"""``eval_v4`` - one command per stage of the suite.

    eval_v4 generate <assessment> --model <run_dir> --ckpt <step> --out <campaign>
    eval_v4 measure  <assessment> --root <campaign>
    eval_v4 report                --root <campaign>
    eval_v4 manifest-check        --root <campaign>
    eval_v4 real-floor            --root <campaign>

The stages are separate on purpose.  Generation needs a GPU and hours;
measurement needs neither and must be repeatable from the volumes alone; the
report needs only the results file.  Run ``real-floor`` first: every table is
read against it, and a table with no floor row says only that a number exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from poregen.eval_v4.cases import ASSESSMENTS, build_cases
from poregen.eval_v4.io import case_dir, repo_root

log = logging.getLogger("eval_v4")


def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, type=Path,
                   help="campaign directory, e.g. runs/campaigns/10-eval-v4/")
    p.add_argument("--repo", type=Path, default=None,
                   help="repository root (default: found from this file)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eval_v4", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate one assessment's volumes")
    g.add_argument("assessment", choices=sorted(ASSESSMENTS))
    g.add_argument("--model", required=True, type=Path, help="LDM run directory")
    g.add_argument("--ckpt", required=True,
                   help="training step, or 'best' / 'latest'")
    g.add_argument("--out", required=True, type=Path, help="campaign directory")
    g.add_argument("--weights", choices=("raw", "ema"), default="ema")
    # No --latents-root: the store is the run's own data.latents_root, checked
    # against the run's z_channels and VAE checkpoint before anything is
    # generated.  A flag here would be a second, unverified answer to the one
    # question that decides whether the output means anything.
    g.add_argument("--repo", type=Path, default=None)
    g.add_argument("--only", nargs="*", default=None,
                   help="generate only these case names")
    g.add_argument("--dry-run", action="store_true",
                   help="list the cases and what each one asks for, generate nothing")
    g.add_argument("--save-latents", action="store_true",
                   help="also write latents.npy per case — the finished latent canvas the "
                        "decoder consumed. Required by the decoder fine-tune gate, which "
                        "must compare two decoders on identical latents.")

    m = sub.add_parser("measure", help="run the metrics over one assessment")
    m.add_argument("assessment", choices=sorted(ASSESSMENTS))
    _add_root(m)

    r = sub.add_parser("report", help="write findings.md and the figures")
    _add_root(r)
    r.add_argument("--assessment", nargs="*", default=None)

    c = sub.add_parser("manifest-check", help="validate every manifest in a campaign")
    _add_root(c)

    f = sub.add_parser("real-floor", help="cut and measure the real test crops")
    _add_root(f)
    f.add_argument("--data-root", type=Path, default=None,
                   help="dataset root (default data/split_v3)")
    f.add_argument("--shapes", nargs="*", default=["small", "large", "micro", "surface"],
                   choices=("small", "large", "micro", "surface"),
                   help="'micro' cuts the matched-porosity reference PAIRS the "
                        "microstructure assessment is floored against")
    f.add_argument("--max-volumes", type=int, default=None)
    f.add_argument("--rebuild", action="store_true",
                   help="re-cut the crops even when they already exist")
    return ap


def cmd_generate(args) -> int:
    from tqdm import tqdm  # noqa: PLC0415

    from poregen.eval_v4.generate import VolumeRunner  # noqa: PLC0415

    repo = args.repo or repo_root()
    specs = build_cases(args.assessment, repo)
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise SystemExit(f"no such case in {args.assessment}: {sorted(missing)}")

    if args.dry_run:
        for s in specs:
            print(
                f"{s.name:34s} shape={s.volume_shape} ddim={s.ddim_steps} "
                f"chunk={s.chunk_tiles} s_por={s.s_por} s_nb={s.s_nb} "
                f"seed={s.seed} phi={s.target_phi} "
                f"field={'yes' if s.field_fn else 'no'} "
                f"material={'painted' if s.material_fn else 'full'}"
            )
        print(f"\n{len(specs)} cases")
        return 0

    runner = VolumeRunner(
        args.model, args.ckpt, weights=args.weights, repo=repo,
        save_latents=args.save_latents,
    )
    todo = [s for s in specs
            if not (case_dir(args.out, args.assessment, s.name) / "manifest.json").exists()]
    log.info("%d of %d cases still to generate", len(todo), len(specs))
    with tqdm(total=len(todo), unit="vol") as bar, tqdm(unit="step", leave=False) as steps:
        for spec in todo:
            bar.set_description(spec.name)
            n_chunks = 1
            for n, c in zip(spec.tile_grid, spec.chunk_tiles):
                n_chunks *= -(-n // c)
            steps.reset(total=n_chunks * spec.ddim_steps)
            runner.run(spec, case_dir(args.out, args.assessment, spec.name), progress=steps)
            bar.update(1)
    return 0


def cmd_measure(args) -> int:
    from poregen.eval_v4.measure import measure  # noqa: PLC0415

    res = measure(args.root, args.assessment, args.repo)
    print(f"{args.assessment}: {res['n_cases_measured']} of {res['n_cases_expected']} "
          f"cases -> {args.root}/{args.assessment}/results.json")
    return 0


def cmd_report(args) -> int:
    from poregen.eval_v4.report import report  # noqa: PLC0415

    for p in report(args.root, args.assessment):
        print(p)
    return 0


def cmd_manifest_check(args) -> int:
    from poregen.eval_v4.measure import manifest_check  # noqa: PLC0415

    rep = manifest_check(args.root, args.repo)
    for name, entry in rep["assessments"].items():
        status = "ok" if entry["ok"] else "FAIL"
        print(f"{status:4s} {name:18s} {entry['n_cases']:3d} cases")
        for kind in ("invalid", "mismatched"):
            for item in entry[kind]:
                print(f"       {kind}: {item['case']}: {item['error']}")
        if entry.get("missing"):
            print(f"       missing: {', '.join(entry['missing'])}")
        if entry.get("unexpected"):
            print(f"       unexpected: {', '.join(entry['unexpected'])}")
    return 0 if rep["ok"] else 1


def cmd_real_floor(args) -> int:
    from poregen.eval_v4.real_floor import run  # noqa: PLC0415

    res = run(args.root, data_root=args.data_root, repo=args.repo,
              shapes=tuple(args.shapes), max_volumes=args.max_volumes,
              rebuild=args.rebuild)
    for tag, s in res["by_shape"].items():
        print(f"{tag}: {s['n_volumes']} crops  phi={s['phi_pore']['mean']:.4f}  "
              f"seam_xct={s['seam_xct_ratio']['mean']:.3f}")
    return 0


COMMANDS = {
    "generate": cmd_generate,
    "measure": cmd_measure,
    "report": cmd_report,
    "manifest-check": cmd_manifest_check,
    "real-floor": cmd_real_floor,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

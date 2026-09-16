"""Sample the pixel-space DDPM and write eval_v4 cases. (GPU.)

Same adapter as the SliceGAN baseline: request-free manifests with
`sampler: "ddpm3d"`, into campaign 23, so `eval_v4 measure` scores it on the
same tables.

LARGE VOLUMES ARE FUSED FROM INDEPENDENT WINDOWS. This model has no
conditioning of any kind, so a 192-cubed sample is overlapping 64-cubed windows
denoised separately and blended. They will disagree where they meet and the
seam metrics are meant to see it: what neighbour conditioning buys is exactly
that difference, and hiding it would delete the comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("ddpm3d_sample")

#: EVERY case at DDIM-200 — the step count the microstructure and seam tables
#: of this project use. A DDIM-50 shortcut would have made the baseline's rows
#: incomparable with the rows they sit beside, which is the one thing a
#: baseline may not be.
DDIM_STEPS = 200

#: 64-cubed is the model's native shape; 192-cubed is the smallest that has a
#: fused seam to measure.
#:
#: THE NATIVE SHAPE CANNOT BE SCORED ON MICROSTRUCTURE AT ALL. Every
#: microstructure statistic in this project is defined on a 128-cubed analysis
#: window, and `analysis_windows` refuses a 64-cubed volume outright. So the
#: microstructure set is 192-cubed and every microstructure number for this
#: baseline comes from a FUSED volume and carries fusion seams. That is not a
#: choice of this script; it is a consequence of the 64-cubed ceiling, and the
#: README says so where the numbers are read.
CASES = (
    ("64_seed101", (64, 64, 64), 101),
    ("64_seed202", (64, 64, 64), 202),
    ("64_seed303", (64, 64, 64), 303),
    ("192_seed101", (192, 192, 192), 101),
    ("192_seed202", (192, 192, 192), 202),
    ("192_seed303", (192, 192, 192), 303),
    ("micro_192_seed404", (192, 192, 192), 404),
    ("micro_192_seed505", (192, 192, 192), 505),
    ("micro_192_seed606", (192, 192, 192), 606),
    ("micro_192_seed707", (192, 192, 192), 707),
)

#: ONE wide case, at the same DDIM-200, chosen by MEASUREMENT. 512 if it fits
#: the budget, else 384.
#:
#: 192x1024x1024 is NOT a candidate — at 200 steps it is out of reach — but its
#: cost is EXTRAPOLATED from the measured per-window time and recorded. The
#: affordability argument for a latent space is stronger with a measured number
#: than with an omission: "the pixel-space model needs N hours for the volume
#: ldm06 makes in M" is a result, and "we did not run it" is not.
WIDE_CANDIDATES = ((192, 512, 512), (192, 384, 384))
#: Costed but never generated, so the headline comparison has a number.
WIDE_EXTRAPOLATE = ((192, 1024, 1024),)


def window_count(shape, patch: int, stride: int) -> int:
    def starts(n: int) -> list[int]:
        out = list(range(0, n - patch + 1, stride))
        if out[-1] != n - patch:
            out.append(n - patch)
        return out
    z, y, x = (len(starts(n)) for n in shape)
    return z * y * x


def time_one_window(model, sched, device, steps: int, batch: int) -> float:
    """Seconds per WINDOW, measured on this card with this model.

    One warm-up batch first: the first CUDA call of a shape pays allocation and
    autotune, and timing that would inflate every estimate built on it.
    """
    from poregen.baselines.ddpm3d.train import sample_window

    sample_window(model, sched, batch, device, steps=2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    sample_window(model, sched, batch, device, steps=steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / batch


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return "0" * 40


def main() -> int:
    from poregen.baselines.ddpm3d.data import decode_sample
    from poregen.baselines.ddpm3d.train import DDPMConfig, build, sample_volume
    from poregen.eval_v4.io import case_dir, save_case
    from poregen.eval_v4.manifest import Manifest

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path,
                    default=REPO / "runs" / "campaigns" / "23-ddpm3d-baseline")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4,
                    help="windows denoised per forward pass")
    ap.add_argument("--wide-budget-hours", type=float, default=6.0,
                    help="the widest case that fits this is generated; the cost of "
                         "the ones that do not is recorded instead")
    ap.add_argument("--no-wide", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = DDPMConfig(**{k: v for k, v in (ck.get("config") or {}).items()
                        if k in DDPMConfig.__dataclass_fields__})
    model, sched = build(cfg, device)
    # The EMA weights, because they are what a diffusion model is sampled with.
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.eval()
    logger.info("model from %s (step %s), EMA weights", args.checkpoint, ck.get("step"))

    # ── the wide case, decided by measurement ──────────────────────────────
    wide = None
    wide_report = {}
    if not args.no_wide:
        per_window = time_one_window(model, sched, device, DDIM_STEPS, args.batch)
        wide_report["seconds_per_window_measured"] = per_window
        wide_report["ddim_steps"] = DDIM_STEPS
        wide_report["measured_on"] = str(device)
        est = {}
        for shape in (*WIDE_CANDIDATES, *WIDE_EXTRAPOLATE):
            n = window_count(shape, 64, args.stride)
            hours = n * per_window / 3600.0
            est[str(list(shape))] = {"windows": n, "estimated_hours": hours,
                                     "generated": False}
            logger.info("%s: %d windows, %.1f h estimated at DDIM-%d",
                        shape, n, hours, DDIM_STEPS)
        for shape in WIDE_CANDIDATES:
            if wide is None and est[str(list(shape))]["estimated_hours"] <= args.wide_budget_hours:
                wide = shape
        if wide is None:
            wide = WIDE_CANDIDATES[-1]
            logger.info("no candidate fits %.1f h; taking the smallest, %s",
                        args.wide_budget_hours, wide)
        est[str(list(wide))]["generated"] = True
        wide_report["estimates"] = est
        wide_report["chosen"] = list(wide)
        wide_report["note"] = (
            "192x1024x1024 is costed from the measured per-window time and NOT "
            "generated: at DDIM-200 it is out of reach for this model. That "
            "number is the comparison — it is what a pixel-space diffusion "
            "model would need for the volume ldm06 produces routinely.")
        logger.info("WIDE CASE: %s at DDIM-%d", wide, DDIM_STEPS)

    commit, written = git_commit(), []
    todo = list(CASES)
    if wide is not None:
        todo.append((f"wide{wide[1]}_seed101", wide, 101))
    for name, shape, seed in todo:
        if args.only and name not in args.only:
            continue
        g = torch.Generator(device=device).manual_seed(seed)
        t = time.time()
        steps = DDIM_STEPS
        vol = sample_volume(model, sched, shape, device, steps=steps,
                            stride=args.stride, batch=args.batch, generator=g)
        grey, label = decode_sample(vol)
        wall = time.time() - t
        mf = Manifest(
            assessment="ddpm3d", case=name, volume_shape=tuple(shape),
            git_commit=commit, sampler="ddpm3d",
            model_run=str(args.checkpoint.parent), checkpoint_step=int(ck.get("step", 0)),
            weights="ema", ddim_steps=steps, seed=seed,
            requested_material="full", wall_time_s=wall,
            peak_gpu_memory_bytes=(int(torch.cuda.max_memory_allocated(device))
                                   if device.type == "cuda" else None),
            notes={"baseline": "3-D pixel-space DDPM",
                   "unconditional": True,
                   "native_shape": [64, 64, 64],
                   "fusion": f"independent 64-cubed windows, stride {args.stride}, "
                             "Tukey-tapered; NO neighbour conditioning exists in "
                             "this model, so windows disagree where they meet and "
                             "the seam metrics are meant to measure that",
                   "no_request_because":
                       "the model has no conditioning path, so every conditional "
                       "assessment is inapplicable"},
        )
        d = case_dir(args.root, "ddpm3d", name)
        save_case(d, mf, grey, label)
        phi = float((label == 1).sum() / max(int((label != 2).sum()), 1))
        logger.info("%-14s %s  phi %.4f  %.1f s -> %s", name, shape, phi, wall, d)
        written.append({"case": name, "shape": list(shape), "phi_material": phi,
                        "wall_s": wall, "path": str(d)})
    (args.root / "generation.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "step": ck.get("step"),
         "ddim_steps": DDIM_STEPS, "stride": args.stride,
         "parameters_M": sum(p.numel() for p in model.parameters()) / 1e6,
         "wide_case": wide_report,
         "n_cases": len(written), "cases": written}, indent=2) + "\n")
    logger.info("wrote %d cases -> %s", len(written), args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

#: 64-cubed is the model's native shape; 192-cubed is the smallest that has a
#: fused seam to measure.
CASES = (
    ("64_seed101", (64, 64, 64), 101),
    ("64_seed202", (64, 64, 64), 202),
    ("64_seed303", (64, 64, 64), 303),
    ("192_seed101", (192, 192, 192), 101),
    ("192_seed202", (192, 192, 192), 202),
    ("192_seed303", (192, 192, 192), 303),
)

#: ONE wide case, at DDIM-50, chosen by MEASUREMENT rather than by guess. The
#: full 1024 is generated if it fits the budget; otherwise 512, and the
#: extrapolated cost of the 1024 is recorded.
#:
#: The affordability argument for a latent space is stronger with a measured
#: number than with an omission — "the pixel-space model would need N hours for
#: the volume ldm06 makes in M" is a result; "we did not run it" is not.
WIDE_CANDIDATES = ((192, 1024, 1024), (192, 512, 512))
WIDE_STEPS = 50


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
    ap.add_argument("--steps", type=int, default=200)
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
        per_window = time_one_window(model, sched, device, WIDE_STEPS, args.batch)
        wide_report["seconds_per_window_measured"] = per_window
        wide_report["ddim_steps"] = WIDE_STEPS
        for shape in WIDE_CANDIDATES:
            n = window_count(shape, 64, args.stride)
            hours = n * per_window / 3600.0
            wide_report[str(list(shape))] = {"windows": n, "estimated_hours": hours}
            logger.info("wide candidate %s: %d windows, %.1f h estimated",
                        shape, n, hours)
            if wide is None and hours <= args.wide_budget_hours:
                wide = shape
        if wide is None:
            wide = WIDE_CANDIDATES[-1]
            logger.info("no candidate fits %.1f h; taking the smallest, %s",
                        args.wide_budget_hours, wide)
        wide_report["chosen"] = list(wide)
        logger.info("WIDE CASE: %s at DDIM-%d", wide, WIDE_STEPS)

    commit, written = git_commit(), []
    todo = list(CASES)
    if wide is not None:
        todo.append((f"wide{wide[1]}_seed101", wide, 101))
    for name, shape, seed in todo:
        if args.only and name not in args.only:
            continue
        g = torch.Generator(device=device).manual_seed(seed)
        t = time.time()
        steps = WIDE_STEPS if name.startswith("wide") else args.steps
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
         "ddim_steps": args.steps, "stride": args.stride,
         "parameters_M": sum(p.numel() for p in model.parameters()) / 1e6,
         "wide_case": wide_report,
         "n_cases": len(written), "cases": written}, indent=2) + "\n")
    logger.info("wrote %d cases -> %s", len(written), args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample the trained SliceGAN and write eval_v4 cases. (GPU.)

Writes into campaign 22 in the layout `eval_v4 measure` reads, so the baseline
meets the same request-free tables as ldm06: phase fractions, seams,
microstructure, memorisation, failure flags.

    <root>/slicegan/volumes/<case>/{volume.tif,label.tif,manifest.json}

MANIFESTS CARRY NO REQUESTS. SliceGAN is unconditional — it cannot be asked for
a porosity, a layup, an envelope or a shape — so `requested_global_phi`,
`requested_layup` and the rest are absent rather than invented.
`requested_material` is "full", which here is a MEASUREMENT CONVENTION and not
a request: it tells the metrics to treat the whole canvas as specimen, so
porosity is pore over non-air exactly as for every other volume.

Shapes come from the paper's own scaling: the generator is fully convolutional,
so a bigger latent gives a bigger volume with no retraining. `out = 32n - 64`,
so 192 needs an 8-cell latent and 1024 needs 34.

Usage:
    python scripts/analysis/slicegan_sample.py --checkpoint <run>/latest.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("slicegan_sample")

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2

#: (case name, shape, seed). The 192-cubed trio and the 1024-wide pair mirror
#: eval_v4's own sampler cases so the tables line up row for row. The extra
#: 192-cubed volumes are for microstructure, which needs several independent
#: samples at one porosity to estimate a distribution rather than a point.
CASES: tuple[tuple[str, tuple[int, int, int], int], ...] = (
    ("192_seed101", (192, 192, 192), 101),
    ("192_seed202", (192, 192, 192), 202),
    ("192_seed303", (192, 192, 192), 303),
    ("1024_seed101", (192, 1024, 1024), 101),
    ("1024_seed202", (192, 1024, 1024), 202),
    ("micro_192_seed404", (192, 192, 192), 404),
    ("micro_192_seed505", (192, 192, 192), 505),
    ("micro_192_seed606", (192, 192, 192), 606),
    ("micro_192_seed707", (192, 192, 192), 707),
)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return "0" * 40


@torch.no_grad()
def generate(gen, shape, seed: int, device, tile_batch: int = 1):
    """One volume: grey as uint8 and the 3-class label.

    The generator is run in ONE forward pass over the whole latent, not tiled.
    Tiling would put a seam at every tile face and the seam metric would then be
    measuring this script rather than the method.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = gen.sample_latent(shape, n=1, generator=g).to(device)
    out = gen(z)[0]
    grey = ((out[0].clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    label = out[1:].argmax(dim=0).to(torch.uint8)
    return grey.to(torch.uint8).cpu().numpy(), label.cpu().numpy()


def main() -> int:
    import tifffile

    from poregen.baselines.slicegan.networks import Generator3D
    from poregen.eval_v4.io import case_dir, save_case
    from poregen.eval_v4.manifest import Manifest

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path,
                    default=REPO / "runs" / "campaigns" / "22-slicegan-baseline")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get("config") or {}
    gen = Generator3D(nz=cfg.get("nz", 64), ngf=cfg.get("ngf", 64)).to(device)
    gen.load_state_dict(ck["generator"])
    gen.eval()
    logger.info("generator from %s (step %s)", args.checkpoint, ck.get("step"))

    commit = git_commit()
    written = []
    for name, shape, seed in CASES:
        if args.only and name not in args.only:
            continue
        t = time.time()
        grey, label = generate(gen, shape, seed, device)
        wall = time.time() - t
        mf = Manifest(
            assessment="slicegan", case=name, volume_shape=tuple(shape),
            git_commit=commit, sampler="slicegan",
            model_run=str(args.checkpoint.parent), checkpoint_step=int(ck.get("step", 0)),
            weights="raw", seed=seed,
            # "full" is a measurement convention, not a request: it tells the
            # metrics to treat the whole canvas as specimen. SliceGAN was asked
            # for nothing, which is why every requested_* field is absent.
            requested_material="full",
            wall_time_s=wall,
            peak_gpu_memory_bytes=(int(torch.cuda.max_memory_allocated(device))
                                   if device.type == "cuda" else None),
            notes={
                "baseline": "SliceGAN (Kench & Cooper 2021, arXiv:2102.07708)",
                "unconditional": True,
                "no_request_because":
                    "SliceGAN has no conditioning path: it cannot be asked for a "
                    "porosity, a layup, an envelope or a shape, so every "
                    "conditional assessment is inapplicable and this volume is "
                    "scored only on the request-free metrics",
                "latent_shape": list(gen.sample_latent(shape, 1).shape[2:]),
                "train_checkpoint": str(args.checkpoint),
            },
        )
        d = case_dir(args.root, "slicegan", name)
        save_case(d, mf, grey, label)
        phi = float((label == LABEL_PORE).sum() /
                    max(int((label != LABEL_AIR).sum()), 1))
        logger.info("%-18s %s  phi %.4f  air %.4f  %.1f s -> %s",
                    name, shape, phi, float((label == LABEL_AIR).mean()), wall, d)
        written.append({"case": name, "shape": list(shape), "phi_material": phi,
                        "air_fraction": float((label == LABEL_AIR).mean()),
                        "wall_s": wall, "path": str(d)})
        del grey, label
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (args.root / "generation.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "step": ck.get("step"),
         "n_cases": len(written), "cases": written}, indent=2) + "\n")
    logger.info("wrote %d cases -> %s", len(written), args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

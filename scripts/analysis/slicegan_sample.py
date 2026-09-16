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


#: Latent cells of halo around each tile. The generator is fully convolutional,
#: so a tile with enough halo reproduces the single-pass result EXACTLY in its
#: interior — this is not a blend and there is no seam.
#:
#: MEASURED, not assumed: a latent tile starting at cell `a` produces output
#: starting at voxel 32a, a tile of n cells produces 32n - 64 voxels, and only
#: the 2 outermost voxels differ from the full pass. The arithmetic then forces
#: h >= 2 (the core must fit inside 32n - 64), and at exactly 2 the cores tile
#: with nothing wasted. `tests/test_slicegan.py` asserts the reproduction.
HALO_CELLS = 2
#: Core cells per tile. 8 cores + 2x2 halo = 12 cells -> a 320-voxel cube per
#: forward pass, which is the size that keeps a 1024-wide volume inside memory.
CORE_CELLS = 8


@torch.no_grad()
def generate(gen, shape, seed: int, device, core: int = CORE_CELLS):
    """One volume: grey as uint8 and the 3-class label.

    A SINGLE FORWARD PASS WOULD BE IDEAL AND IS NOT AFFORDABLE. At
    192x1024x1024 the last layer alone holds ngf x 2.0e8 activations; the first
    attempt at this took the machine to 119 GB of its 121 GB unified pool
    before it was stopped.

    So the latent is tiled with a halo instead. Because the generator is fully
    convolutional, each tile's interior is BIT-EXACT against the single pass —
    this introduces no seam and is not a blend. The halo is what buys that, and
    its size was measured rather than guessed.
    """
    from poregen.baselines.slicegan.networks import latent_for_shape

    g = torch.Generator(device="cpu").manual_seed(seed)
    z_full = gen.sample_latent(shape, n=1, generator=g)
    n_cells = latent_for_shape(shape)

    grey = torch.empty(shape, dtype=torch.uint8)
    label = torch.empty(shape, dtype=torch.uint8)

    def cores(n: int) -> list[tuple[int, int]]:
        """(start, length) in CELLS whose outputs tile the volume.

        ONLY n - 2 CELLS PRODUCE OUTPUT. A latent of n cells gives 32n - 64
        voxels, which is 32(n - 2): cell a maps to voxel 32a, and the last two
        cells fall off the end. Tiling all n cells asks for output that does not
        exist, and the guard below caught exactly that on the first attempt.
        """
        producing = n - 2
        out, a = [], 0
        while a < producing:
            c = min(core, producing - a)
            out.append((a, c))
            a += c
        return out

    for z0, cz in cores(n_cells[0]):
        for y0, cy in cores(n_cells[1]):
            for x0, cx in cores(n_cells[2]):
                sl, off, size = [], [], []
                for a0, c, n in ((z0, cz, n_cells[0]), (y0, cy, n_cells[1]),
                                 (x0, cx, n_cells[2])):
                    lo = max(0, a0 - HALO_CELLS)
                    hi = min(n, a0 + c + HALO_CELLS)
                    sl.append(slice(lo, hi))
                    off.append(32 * a0 - 32 * lo)
                    size.append(32 * c)
                tile = z_full[:, :, sl[0], sl[1], sl[2]].to(device)
                out = gen(tile)[0]
                cut = out[:, off[0]:off[0] + size[0],
                          off[1]:off[1] + size[1],
                          off[2]:off[2] + size[2]]
                if tuple(cut.shape[1:]) != tuple(size):
                    raise RuntimeError(
                        f"tile at cells ({z0},{y0},{x0}) gave {tuple(cut.shape[1:])} "
                        f"voxels where {tuple(size)} were needed — the halo is too "
                        f"small for this geometry, which would silently truncate "
                        f"the volume.")
                g_t = ((cut[0].clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
                dst = (slice(32 * z0, 32 * z0 + size[0]),
                       slice(32 * y0, 32 * y0 + size[1]),
                       slice(32 * x0, 32 * x0 + size[2]))
                grey[dst] = g_t.to(torch.uint8).cpu()
                label[dst] = cut[1:].argmax(dim=0).to(torch.uint8).cpu()
                del tile, out, cut
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return grey.numpy(), label.numpy()


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

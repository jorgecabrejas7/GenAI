"""Does the model under-predict pores in the rim of a window next to a neighbour?

The chunk-plane porosity band is caused by the neighbour conditioning: turning
`s_nb` to 0 removes it, and the teacher-forced arm — which never has an UNKNOWN
face — still shows it.  The proposed mechanism is a RIM effect.  A window is 64
voxels; inside a chunk the windows step by 32, so every voxel is also covered by
a window that holds it in its interior and the fusion averages any rim bias
away.  At a chunk boundary the last voxels are covered by rim predictions only,
and whatever the rim does survives into the volume.

This tests the mechanism directly, with no fusion to average anything:

* A canvas of REAL latents, 6x6x6 tiles.  Each tile is its own chunk AND its
  own single window (`chunk_tiles=(1,1,1)`, `window_stride=64`), so nothing
  overlaps and every window is decided once.
* `neighbour_mode="reference"` makes every in-bounds face EXISTS, fed real
  material re-noised to t exactly as in training.  The 4x4x4 = 64 INTERIOR
  tiles are the windows with all six faces EXISTS; the 152 boundary tiles have
  at least one OOB face and are excluded.
* The same canvas decoded directly is the control: what the real material in
  those windows actually looks like, through the same VAE.

Prediction under the rim hypothesis: at ``s_nb=1`` phi is depleted in the 0-16
voxel rim of the window on every face and flat in the interior; at ``s_nb=0``
it is flat throughout.  Generated neighbours (``--neighbours <latents.npy>``)
should deepen it relative to real ones.

CPU-impossible: needs the GPU.  It must not run while a CUDA job holds the card
(see docs/DEVELOPMENT.md on unified memory) — it refuses if one does.

Usage:
    python scripts/analysis/window_rim_test.py --model runs/ldm/ldm06-run-... \
        --out runs/campaigns/16-window-rim
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

TILE = 64
LABEL_PORE, LABEL_AIR = 1, 2
#: Tiles per axis.  NOT cubic: no val or test volume holds a 384-deep block —
#: the store's patches reach z0 = 128 — so depth is 3 tiles and the width
#: carries the count.  (3, 10, 10) leaves 1 x 8 x 8 = 64 interior tiles, which
#: are the windows with all six faces in bounds and so all six EXISTS.
TILES = (3, 10, 10)
SLAB = 8


def shape_of(tiles) -> tuple[int, int, int]:
    return tuple(TILE * int(t) for t in tiles)


def interior_range(tiles):
    """Tile indices whose window has all six faces in bounds, per axis."""
    return [range(1, int(t) - 1) for t in tiles]


def n_interior(tiles) -> int:
    return int(np.prod([max(0, int(t) - 2) for t in tiles]))


def phi_of(block: np.ndarray) -> float | None:
    pore = int((block == LABEL_PORE).sum())
    solid = int((block != LABEL_AIR).sum())
    return pore / solid if solid else None


def rim_profile(label: np.ndarray, tiles=TILES) -> dict[str, float | None]:
    """phi by distance from the window face, pooled over the 64 interior tiles.

    Shell `d` is every voxel whose distance to the NEAREST face of its own
    tile is in [d, d+SLAB).  A rim effect shows as a low phi in the first
    shells and a flat tail; anything that is simply a porosity difference moves
    every shell together.
    """
    n = TILE
    idx = np.arange(n)
    dist1d = np.minimum(idx, n - 1 - idx)
    d3 = np.minimum(np.minimum(dist1d[:, None, None], dist1d[None, :, None]),
                    dist1d[None, None, :])
    shells = {}
    for d in range(0, n // 2, SLAB):
        mask = (d3 >= d) & (d3 < d + SLAB)
        pore = solid = 0
        rz, ry, rx = interior_range(tiles)
        for tz in rz:
            for ty in ry:
                for tx in rx:
                    blk = label[tz * n:(tz + 1) * n,
                                ty * n:(ty + 1) * n,
                                tx * n:(tx + 1) * n]
                    sel = blk[mask]
                    pore += int((sel == LABEL_PORE).sum())
                    solid += int((sel != LABEL_AIR).sum())
        shells[str(d)] = (pore / solid) if solid else None
    return shells


def interior_phi(label: np.ndarray, tiles=TILES) -> float | None:
    n = TILE
    rz, ry, rx = interior_range(tiles)
    blocks = [label[tz * n:(tz + 1) * n, ty * n:(ty + 1) * n, tx * n:(tx + 1) * n]
              for tz in rz for ty in ry for tx in rx]
    return phi_of(np.concatenate([b.ravel() for b in blocks]))


FACES = (("-z", 0, -1), ("+z", 0, +1), ("-y", 1, -1), ("+y", 1, +1),
         ("-x", 2, -1), ("+x", 2, +1))


def face_profile(label: np.ndarray, tiles=TILES, core: int = 16) -> dict:
    """phi by distance from EACH face separately, pooled over interior tiles.

    Distance is measured from one face at a time, and the other two axes are
    restricted to the tile's central `core`..`TILE-core` band so a voxel near
    two faces cannot be counted for both.  Without that restriction every
    corner would appear in three face profiles and a dip on one face would
    leak into the other two.
    """
    n = TILE
    rz, ry, rx = interior_range(tiles)
    out: dict[str, dict[str, float | None]] = {}
    for name, axis, sign in FACES:
        shells: dict[str, list[int]] = {}
        for d in range(0, n // 2, SLAB):
            pore = solid = 0
            for tz in rz:
                for ty in ry:
                    for tx in rx:
                        blk = label[tz * n:(tz + 1) * n,
                                    ty * n:(ty + 1) * n,
                                    tx * n:(tx + 1) * n]
                        sl = [slice(core, n - core)] * 3
                        sl[axis] = (slice(d, d + SLAB) if sign < 0
                                    else slice(n - d - SLAB, n - d))
                        sel = blk[tuple(sl)]
                        pore += int((sel == LABEL_PORE).sum())
                        solid += int((sel != LABEL_AIR).sum())
            shells[str(d)] = (pore / solid) if solid else None
        out[name] = shells
    return out


def face_neighbour_split(label: np.ndarray, tiles=TILES, core: int = 16,
                         shell: int = SLAB) -> dict:
    """Does a window's dip at a face follow what lies ON THE OTHER SIDE of it?

    A window whose neighbour face is itself pore-poor may be depleted because
    it is conditioned on pore-poor material — inherited, not generated. That
    would mean fixing one strip lifts the other, and it is separable: pair each
    window's own phi in the `shell` voxels at a face with the NEIGHBOUR tile's
    phi in its own `shell` voxels across the same plane, then split the windows
    at the median neighbour value.

    If the dip is inherited, the windows facing pore-poor neighbours are much
    more depleted than those facing pore-rich ones. If it is a property of the
    face's availability alone, both halves look the same.
    """
    n = TILE
    rz, ry, rx = interior_range(tiles)
    out: dict[str, dict] = {}
    for name, axis, sign in FACES:
        pairs = []
        for tz in rz:
            for ty in ry:
                for tx in rx:
                    t = [tz, ty, tx]
                    nb = list(t)
                    nb[axis] += sign
                    if not 0 <= nb[axis] < tiles[axis]:
                        continue

                    def block(ti):
                        return label[ti[0] * n:(ti[0] + 1) * n,
                                     ti[1] * n:(ti[1] + 1) * n,
                                     ti[2] * n:(ti[2] + 1) * n]

                    own_sl = [slice(core, n - core)] * 3
                    nb_sl = [slice(core, n - core)] * 3
                    # The two `shell`-thick slabs either side of the shared plane.
                    own_sl[axis] = (slice(0, shell) if sign < 0
                                    else slice(n - shell, n))
                    nb_sl[axis] = (slice(n - shell, n) if sign < 0
                                   else slice(0, shell))
                    a, b = block(t)[tuple(own_sl)], block(nb)[tuple(nb_sl)]
                    pa, pb = phi_of(a), phi_of(b)
                    if pa is not None and pb is not None:
                        pairs.append((pb, pa))
        if not pairs:
            continue
        nb_phi = np.array([p[0] for p in pairs])
        own = np.array([p[1] for p in pairs])
        med = float(np.median(nb_phi))
        poor, rich = nb_phi <= med, nb_phi > med
        out[name] = {
            "n": len(pairs),
            "neighbour_phi_median": med,
            "own_phi_facing_pore_poor": float(own[poor].mean()) if poor.any() else None,
            "own_phi_facing_pore_rich": float(own[rich].mean()) if rich.any() else None,
            "neighbour_phi_poor": float(nb_phi[poor].mean()) if poor.any() else None,
            "neighbour_phi_rich": float(nb_phi[rich].mean()) if rich.any() else None,
        }
    return out

def build(runner, reference, s_nb: float, ddim: int, theta,
          neighbour_mode: str = "reference"):
    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator
    from poregen.eval_v4.generate import LATENT_SIZE, PATCH_SIZE, VOXEL_SIZE_MM

    sampler = DDIMSampler(
        runner.model, runner.schedule, runner.device,
        n_steps=ddim, s_por=1.0, s_nb=s_nb, cfg_rescale=runner.cfg_rescale,
    )
    return VolumeGenerator(
        sampler=sampler, vae=runner.vae, device=runner.device,
        patch_size=PATCH_SIZE, latent_size=LATENT_SIZE,
        latent_mean=runner.latent_mean, latent_std=runner.latent_std,
        voxel_size_mm=VOXEL_SIZE_MM, por_log_stats=runner.por_log_stats,
        theta_deg=theta,
        # One tile per chunk and a window per tile: NO overlap, so nothing
        # averages the rim away and each window is decided exactly once.
        chunk_tiles=(1, 1, 1), window_stride=TILE, decode_stride=32,
        neighbour_mode=neighbour_mode,
        reference_latents=reference if neighbour_mode == "reference" else None,
    )


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the ldm run directory")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--target-phi", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tiles", type=int, nargs=3, default=list(TILES),
                    help="tiles per axis; the interior ones are the windows "
                         "with all six faces EXISTS")
    ap.add_argument("--s-nb", type=float, nargs="+", default=[1.0, 0.0])
    ap.add_argument("--neighbours", type=Path, default=None,
                    help="a latents.npy to use as the neighbour canvas instead "
                         "of real material — the generated-neighbour arm")
    ap.add_argument("--neighbour-mode", default="reference",
                    choices=("reference", "canvas", "unknown"),
                    help="'reference' gives every window six real neighbours. "
                         "'canvas' with one tile per chunk gives the raster "
                         "order's MIXED set — the three trailing faces EXISTS, "
                         "the three leading faces UNKNOWN — which is the set a "
                         "chunk-frontier window actually has.")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()

    from poregen.eval_v4.cases import build_cases
    from poregen.eval_v4.generate import VOXEL_SIZE_MM, VolumeRunner, theta_for_canvas
    from poregen.eval_v4.memorisation import gpu_jobs_other_than

    import os
    busy = [] if args.allow_busy_gpu else gpu_jobs_other_than(os.getpid())
    if busy:
        print("REFUSING: the card is busy with "
              + ", ".join(f"{p} ({n})" for p, n in busy), flush=True)
        return 3

    tiles = tuple(args.tiles)
    shape = shape_of(tiles)
    if n_interior(tiles) < 1:
        print(f"REFUSING: tiles {tiles} leave no interior window", flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    runner = VolumeRunner(args.model, args.ckpt, weights="ema", repo=REPO)
    # The same layup the sampler assessment uses, so the orientation
    # conditioning is the production one and not a constant this test invented.
    spec0 = build_cases("sampler")[0]
    theta = theta_for_canvas(shape[0], spec0.layup, spec0.ply_thickness_vox, 0)

    if args.neighbours:
        canvas = np.load(args.neighbours).astype(np.float32)
        note = {"source": str(args.neighbours), "kind": "generated"}
    else:
        from poregen.eval_v4.teacher import reference_latent_canvas
        canvas, note = reference_latent_canvas(
            runner.latents_root, shape, seed=args.seed, split=args.split)
        note = {**note, "kind": "real"}
    reference = torch.from_numpy(np.ascontiguousarray(canvas)).to(runner.device)

    results = {"shape": list(shape), "tiles": list(tiles),
               "neighbour_mode": args.neighbour_mode,
               "n_interior_windows": n_interior(tiles),
               "ddim": args.ddim, "target_phi": args.target_phi,
               "checkpoint_step": runner.checkpoint_step,
               "neighbours": note, "arms": {}}

    # The control: the SAME real latents through the same decoder, so the
    # comparison is window-vs-window and not model-vs-scanner.
    gen0 = build(runner, reference, args.s_nb[0], args.ddim, theta, "reference")
    with torch.no_grad():
        xct, probs = gen0._decode_canvas(
            reference, shape, runner.autocast_dtype, 64)
    # `_decode_canvas` hands back CLASS PROBABILITIES (3, D, H, W), not a
    # label: taking it for a label indexes the 3-channel axis as if it were z.
    lab = np.asarray(probs).argmax(0).astype(np.uint8)
    results["arms"]["reference_decoded"] = {
        "phi_interior_windows": interior_phi(lab, tiles),
        "phi_by_shell": rim_profile(lab, tiles),
        "note": "the neighbour canvas itself, decoded — no denoising",
    }

    for s_nb in args.s_nb:
        gen = build(runner, reference, s_nb, args.ddim, theta, args.neighbour_mode)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = gen.generate(
                volume_size_mm=tuple(v * VOXEL_SIZE_MM for v in shape),
                target_porosity=args.target_phi,
                autocast_dtype=runner.autocast_dtype,
                window_batch=32, decode_batch_size=64,
                return_class_probs=False, seed=args.seed,
            )
        lab = np.asarray(out[1])
        results["arms"][f"s_nb={s_nb:g}"] = {
            "phi_interior_windows": interior_phi(lab, tiles),
            "phi_by_shell": rim_profile(lab, tiles),
            "phi_by_face": face_profile(lab, tiles),
            "face_neighbour_split": face_neighbour_split(lab, tiles),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        np.save(args.out / f"label_snb{s_nb:g}.npy", lab)
        print(f"  s_nb={s_nb:g} done in {results['arms'][f's_nb={s_nb:g}']['wall_s']} s",
              flush=True)

    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    shells = sorted({k for a in results["arms"].values() for k in a["phi_by_shell"]},
                    key=int)
    print(f"\nphi by distance from the window face, pooled over "
          f"{results['n_interior_windows']} interior windows "
          f"(all six faces EXISTS), neighbours = {note['kind']}")
    print(f"{'arm':<22}{'phi window':>11}" + "".join(f"{'d=' + s:>9}" for s in shells))
    for arm, a in results["arms"].items():
        cells = "".join(
            (f"{a['phi_by_shell'][s]:>9.4f}" if a["phi_by_shell"].get(s) is not None
             else f"{'-':>9}") for s in shells)
        print(f"{arm:<22}{a['phi_interior_windows']:>11.4f}{cells}")
    for arm, a in results["arms"].items():
        sp = a.get("face_neighbour_split")
        if not sp:
            continue
        print(f"\nis the dip INHERITED? own phi in the 8 voxels at a face, "
              f"split by the neighbour's own phi across it — {arm}")
        print(f"{'face':<8}{'n':>5}{'nb phi poor':>13}{'own phi':>10}"
              f"{'nb phi rich':>13}{'own phi':>10}")
        for fc, v in sp.items():
            def f(x):
                return "-" if x is None else f"{x:.4f}"
            print(f"{fc:<8}{v['n']:>5}{f(v['neighbour_phi_poor']):>13}"
                  f"{f(v['own_phi_facing_pore_poor']):>10}"
                  f"{f(v['neighbour_phi_rich']):>13}"
                  f"{f(v['own_phi_facing_pore_rich']):>10}")

    faces = [f[0] for f in FACES]
    for arm, a in results["arms"].items():
        if "phi_by_face" not in a:
            continue
        print(f"\nphi by distance from EACH face — {arm} "
              f"(neighbours {args.neighbour_mode})")
        print(f"{'face':<8}" + "".join(f"{'d=' + s:>9}" for s in shells))
        for fc in faces:
            row = a["phi_by_face"][fc]
            print(f"{fc:<8}" + "".join(
                (f"{row[s]:>9.4f}" if row.get(s) is not None else f"{'-':>9}")
                for s in shells))
    print(f"\nWrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

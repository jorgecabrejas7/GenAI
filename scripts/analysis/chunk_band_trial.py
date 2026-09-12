"""Does anything remove the chunk-plane porosity band? (GPU)

The band is a fivefold porosity depletion in the last ~16 voxels of every
chunk.  Its cause is narrowed but not settled: the neighbour guidance produces
it (``s_nb=0`` removes it, ``joint`` never shows it), yet a single window beside
a real neighbour shows only a 16 % dip in its outermost 8 voxels, so it is not
simply a window's rim.  See `chunk_plane_profile.py` and `window_rim_test.py`.

This regenerates the hybrid arm under candidate changes and measures the
outcome rather than the mechanism, because the outcome is what the paper needs:

  b  s_nb = 0.5          the only intervention that has moved the band
  a  chunk overlap 32, blended   each chunk's rim replaced by the other's interior
  c  both
  f  drop the neighbour arm for MIXED-set windows only, at production cost
  g  f AND the overlap: the two clear opposite strips
  e  chunk overlap 32, PINNED    kept to show that pinning alone does NOT work

Every arm runs the production path — `VolumeRunner` with a `CaseSpec` — so a
trial volume carries a manifest naming the settings that produced it and cannot
be mistaken for a production one.

Success: the -8 AND +0 slab phi within +/-20 % of the volume mean — the
depletion sits on both single-covered strips, 32 voxels each side of a plane,
so clearing only the deep one is not clearing the band — AND the grey chunk
seam still at the real floor, AND delivered phi inside the gate.  Wall time per
volume is reported with them: the regeneration decision needs the cost.

Usage:
    python scripts/analysis/chunk_band_trial.py --model runs/ldm/ldm06-run-... \
        --out runs/campaigns/17-chunk-band-trial --arms b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TILE = 64
#: The production chunk, so the planes sit where campaign 12's do.
CHUNK_TILES = (3, 3, 3)
#: Half a tile: the overlap must be a multiple of window_stride (32), not of
#: the 64-voxel tile.  See `VolumeGenerator.chunk_overlap`.
OVERLAP = 32

#: (label, s_nb, chunk_overlap, chunk_overlap_pinned, chunk_overlap_write,
#:  drop_neighbours_when_mixed)
ARMS = {
    "b":  ("s_nb 0.5", 0.5, 0, None, "blend", False),
    "a":  ("overlap 32 blended", 1.0, OVERLAP, None, "blend", False),
    "c":  ("both", 0.5, OVERLAP, None, "blend", False),
    # (f) keeps s_nb at 1, so the sampler stays on its ONE-pass path and the
    # neighbour arm is dropped per window instead of globally: production cost.
    "f":  ("drop nb on mixed sets", 1.0, 0, None, "blend", True),
    "e":  ("overlap 32 pinned only", 1.0, OVERLAP, None, "pin", False),
    # (g) = (f) + (a). They fix opposite strips: (f) drops the neighbour arm on
    # mixed-set windows and clears the TRAILING strip, the overlap makes the
    # successor's LEADING strip double-covered and clears that. Neither alone
    # does both, and both together cost 1.36x production rather than 3.8x.
    "g":  ("drop nb on mixed + overlap 32", 1.0, OVERLAP, None, "blend", True),
    # (a2): overlap 64 with only the leading 32 pinned, so the predecessor's own
    # rim is FREE for the successor to redraw against a pore-normal context.
    # `a2b` blends the free part, `a2s` takes the successor outright — together
    # with (e) they give A_final, B_final and the mix over the same strip.
    "a2b": ("overlap 64, pin 32, blend", 1.0, 2 * OVERLAP, OVERLAP, "blend", False),
    "a2s": ("overlap 64, pin 32, successor", 1.0, 2 * OVERLAP, OVERLAP, "successor", False),
    # Run against a facedrop checkpoint, not ldm06. `fd_baseline` is the
    # PRODUCTION sampler with no mitigation at all: if per-face dropout fixed
    # the cause, that row alone should clear the band, and every inference-side
    # arm becomes unnecessary. `fd_a2s` is the best inference arm on the new
    # weights, to see whether the two are additive or redundant.
    "fd_baseline": ("facedrop, production sampler", 1.0, 0, None, "blend", False),
    "fd_a2s": ("facedrop + overlap 64 pin 32 successor", 1.0, 2 * OVERLAP,
               OVERLAP, "successor", False),
}

#: The hybrid cases campaign 12 already has, so every arm has a like-for-like
#: row to sit beside.  1024 carries the y/x chunk planes, 384 carries all three.
CASES = [((192, 1024, 1024), 101), ((192, 1024, 1024), 202),
         ((384, 384, 384), 101), ((384, 384, 384), 202), ((384, 384, 384), 303)]
#: Arm (e) is a demonstration, not a measurement: one seed at 1024 is enough.
CASES_E = [((192, 1024, 1024), 101)]


def specs(arm: str, only_1024: bool = False):
    from poregen.eval_v4.cases import CaseSpec, build_cases

    _, s_nb, overlap, pinned, write, drop_mixed = ARMS[arm]
    base = build_cases("sampler")[0]
    out = []
    for shape, seed in (CASES_E if arm == "e" else CASES):
        if only_1024 and shape[0] != 192:
            continue
        tag = "1024" if shape[1] == 1024 else "384"
        out.append(CaseSpec(
            name=f"{tag}_seed{seed}",
            assessment=f"arm_{arm}",
            volume_shape=shape,
            seed=seed,
            layup=base.layup,
            ply_thickness_vox=base.ply_thickness_vox,
            target_phi=0.03,
            ddim_steps=50,
            chunk_tiles=CHUNK_TILES,
            chunk_overlap=overlap,
            chunk_overlap_pinned=pinned,
            chunk_overlap_write=write,
            drop_neighbours_when_mixed=drop_mixed,
            s_nb=s_nb,
            notes={"trial_arm": arm, "trial_label": ARMS[arm][0]},
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["b", "a", "c", "f", "g", "e"],
                    choices=sorted(ARMS))
    ap.add_argument("--only-1024", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()

    from poregen.eval_v4.generate import VolumeRunner
    from poregen.eval_v4.io import case_dir
    from poregen.eval_v4.memorisation import gpu_jobs_other_than

    busy = [] if args.allow_busy_gpu else gpu_jobs_other_than(os.getpid())
    if busy:
        print("REFUSING: the card is busy with "
              + ", ".join(f"{p} ({n})" for p, n in busy), flush=True)
        return 3

    runner = VolumeRunner(args.model, args.ckpt, weights="ema",
                          repo=REPO, save_latents=True)
    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    for arm in args.arms:
        label, s_nb, overlap, pinned, write, drop_mixed = ARMS[arm]
        for spec in specs(arm, args.only_1024):
            d = case_dir(args.out, spec.assessment, spec.name)
            if (Path(d) / "manifest.json").exists():
                print(f"  arm {arm} {spec.name}: already there, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"  arm {arm} ({label}) {spec.name}: s_nb={s_nb} "
                  f"overlap={overlap} pinned={pinned} write={write} "
                  f"drop_mixed={drop_mixed}",
                  flush=True)
            m = runner.run(spec, d)
            wall = time.perf_counter() - t0
            log.append({"arm": arm, "label": label, "case": spec.name,
                        "shape": list(spec.volume_shape), "seed": spec.seed,
                        "s_nb": s_nb, "chunk_overlap": overlap,
                        "chunk_overlap_pinned": pinned, "chunk_overlap_write": write,
                        "drop_neighbours_when_mixed": drop_mixed,
                        "wall_s": round(wall, 1),
                        "checkpoint_step": m.checkpoint_step})
            print(f"    done in {wall / 60:.1f} min", flush=True)
            (args.out / "trial_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nWrote {args.out / 'trial_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

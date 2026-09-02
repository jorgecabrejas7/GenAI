"""DDIM-steps probe v3 — generation only (GPU).

Mirror of ``scripts/analysis/ldm06_probe.py`` part B, re-run on the fixed
decode path: DDIM steps {50, 100, 200, 300} x seeds {101, 202}, joint mode
with specimen (OOB) semantics, s_por 1.5, target porosity 0.03 as a coherent
local field, 192^3 voxels = 8 volumes.  Identical settings and seeds to the
originals, so the numbers are directly comparable.

Volumes land under ``runs/campaigns/05-eval-v3-fixed-decode/volumes/ddim_probe/steps_<n>_seed_<s>/``
(volume.tif uint8 on the raw-scan grey scale, mask.tif uint8 0/255,
stats.json), so a crashed run resumes by skipping completed cells.

The analysis lives in ``scripts/analysis/eval_v3_ddim_analysis.py``, which
joins this set to the air audit and the onlypores measurement.

Usage:
    python scripts/analysis/eval_v3_ddim_probe.py [--joint-window-batch 16]
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402
from _eval_v3 import (  # noqa: E402
    DDIM_ARM, DDIM_S_POR, DDIM_SEEDS, DDIM_STEP_COUNTS, DDIM_TARGET,
    DDIM_VOL_ROOT,
)
from _eval_v2 import (  # noqa: E402
    ARMS, CKPT, DECODE_BATCH, GEN_BATCH, GRID, JOINT_WINDOW_STRIDE, VOLUME_MM,
    build_generator, cell_porosities, coherent_por_map, load_existing,
    save_volume,
)

from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import DDIMSampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-window-batch", type=int, default=16)
    args = ap.parse_args()

    mode, semantics, _ = ARMS[DDIM_ARM]
    device = torch.device("cuda")
    gen, model, schedule, step = build_generator(device)
    gen.conditioning_semantics = semantics
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    n_total = len(DDIM_STEP_COUNTS) * len(DDIM_SEEDS)
    i = 0
    for steps in DDIM_STEP_COUNTS:
        gen.sampler = DDIMSampler(model, schedule, device, n_steps=steps,
                                  s_por=DDIM_S_POR, s_nb=1.0)
        for seed in DDIM_SEEDS:
            i += 1
            vol_dir = DDIM_VOL_ROOT / f"steps_{steps}_seed_{seed}"
            if load_existing(vol_dir) is not None:
                print(f"[{i}/{n_total}] steps={steps} seed={seed} (cached)",
                      flush=True)
                continue
            torch.manual_seed(seed)
            por_map, field = coherent_por_map(DDIM_TARGET, seed, te_sampler,
                                              corr_lengths)
            jwb = args.joint_window_batch
            while True:
                try:
                    torch.cuda.empty_cache()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        xct, mask, stats = gen.generate(
                            volume_size_mm=VOLUME_MM,
                            local_por_map=por_map,
                            autocast_dtype=torch.bfloat16,
                            gen_batch_size=GEN_BATCH,
                            decode_batch_size=DECODE_BATCH,
                            mode=mode,
                            joint_window_stride=JOINT_WINDOW_STRIDE,
                            joint_window_batch=jwb,
                        )
                    wall = time.perf_counter() - t0
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if jwb <= 1:
                        raise
                    jwb //= 2
                    print(f"  OOM — retrying with joint_window_batch={jwb}",
                          flush=True)
            delivered = float(stats["actual_mask_porosity"])
            rec = {
                "experiment": "ddim_probe",
                "arm": DDIM_ARM,
                "mode": mode,
                "conditioning_semantics": semantics,
                "s_por": DDIM_S_POR,
                "ddim_steps": steps,
                "target": DDIM_TARGET,
                "seed": seed,
                "checkpoint": str(CKPT),
                "checkpoint_step": step,
                "weights": "raw (non-EMA)",
                "delivered_mask_porosity": delivered,
                "abs_error": abs(delivered - DDIM_TARGET),
                "field_mean": float(field.mean()),
                "cell_targets": {f"{iz},{iy},{ix}": por_map[(iz, iy, ix)]
                                 for iz in range(GRID[0])
                                 for iy in range(GRID[1])
                                 for ix in range(GRID[2])},
                "cell_delivered": cell_porosities(mask),
                "seam_xct_ratio": stats["seam_xct_ratio"],
                "seam_mask_ratio": stats["seam_mask_ratio"],
                "wall_s": round(wall, 1),
                "joint_window_batch": jwb,
            }
            save_volume(vol_dir, xct, mask, rec)
            del xct, mask
            print(f"[{i}/{n_total}] steps={steps} seed={seed} "
                  f"delivered={delivered:.4f} |err|={rec['abs_error']:.4f} "
                  f"wall={wall:.1f}s -> {vol_dir}", flush=True)

    print(f"DDIM probe generation complete: {DDIM_VOL_ROOT}", flush=True)


if __name__ == "__main__":
    main()

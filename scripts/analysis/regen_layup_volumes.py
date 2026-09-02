"""Regenerate selected layup-roundtrip volumes and SAVE them as TIFFs.

The evaluation run measured 12 volumes in memory and kept only metrics.
This regenerates the showcase subset with identical settings and seeds
(checkpoint 130k, DDIM-50, coherent phi field, target 0.03) and writes
volume.tif (float32 [0,1]) + mask.tif (uint8) per cell, following the
generate_volumes.py convention.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import layup_roundtrip as lr  # noqa: E402
from layup_roundtrip import (  # noqa: E402
    LAYUPS, PLY_VOX, TARGET_POR, VOL_SHAPE, JOINT_WINDOW_BATCH,
    build_generator, coherent_por_map, generate_one,
)
from poregen.diffusion.sampler import theta_from_layup  # noqa: E402
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS,
    load_corr_lengths_voxels, load_sampler,
)

# (layup, mode, s_por, seed) — the 6 joint volumes + the worst sequential
CELLS = [
    ("A_training", "joint", 1.5, 101),
    ("A_training", "joint", 1.5, 202),
    ("B_permuted", "joint", 1.5, 101),
    ("B_permuted", "joint", 1.5, 202),
    ("C_simple", "joint", 1.5, 101),
    ("C_simple", "joint", 1.5, 202),
    ("C_simple", "sequential", 1.0, 101),
]

OUT_ROOT = lr.REPO / ("inference/ldm05-run-0001-20260827-114902-z4-c128-bs256-lr1e-04"
                      "/layup_roundtrip_volumes")
REF = json.loads((lr.REPO / "runs/analysis/layup_roundtrip/results.json").read_text())


def main() -> None:
    device = torch.device("cuda")
    gen, step = build_generator(device)
    te_sampler = load_sampler(lr.REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(lr.REPO / DEFAULT_TD_RESULTS)

    for i, (layup_name, mode, s_por, seed) in enumerate(CELLS, 1):
        out_dir = OUT_ROOT / f"{layup_name}_{mode}_seed{seed}"
        if (out_dir / "mask.tif").exists():
            print(f"[{i}/{len(CELLS)}] {out_dir.name} exists — skipping", flush=True)
            continue
        gen.sampler.s_por = s_por
        gen.sampler.guided = not (s_por == 1.0 and gen.sampler.s_nb == 1.0)
        gen.theta_deg = theta_from_layup(VOL_SHAPE[0], LAYUPS[layup_name], PLY_VOX)
        torch.manual_seed(seed)
        por_map = coherent_por_map(TARGET_POR, seed, te_sampler, corr_lengths)
        xct, mask, stats, wall, _, _ = generate_one(gen, mode, por_map,
                                                    JOINT_WINDOW_BATCH)
        por = float(stats["actual_mask_porosity"])
        ref = next(r for r in REF["records"]
                   if r["layup"] == layup_name and r["mode"] == mode
                   and r["seed"] == seed)
        drift = abs(por - ref["delivered_porosity"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out_dir / "volume.tif"),
                         np.asarray(xct, dtype=np.float32))
        tifffile.imwrite(str(out_dir / "mask.tif"),
                         (np.asarray(mask) > 0).astype(np.uint8) * 255)
        (out_dir / "stats.json").write_text(json.dumps(
            {"layup": layup_name, "angles": LAYUPS[layup_name], "mode": mode,
             "s_por": s_por, "seed": seed, "step": step,
             "target_porosity": TARGET_POR, "delivered_porosity": por,
             "reference_porosity": ref["delivered_porosity"],
             "reproduction_drift": drift, "wall_s": round(wall, 1)}, indent=1))
        print(f"[{i}/{len(CELLS)}] {out_dir.name}  por={por:.4f} "
              f"(ref {ref['delivered_porosity']:.4f}, drift {drift:.1e})  "
              f"wall={wall:.0f}s", flush=True)
        del xct, mask
    print("ALL_SAVED", flush=True)


if __name__ == "__main__":
    main()

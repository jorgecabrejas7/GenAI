"""Sequential vs joint generation comparison at the current ldm05 checkpoint.

192³ voxels (4.8 mm)³, porosity 0.02 uniform, layup A, DDIM-50, RAW weights.
Small batches: shared GPU with an active training run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path("/home/jorgecabrejas/Dev/GenAI")
sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.conditioning import resolve_group_order
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator, theta_from_layup
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
from poregen.training.checkpoint import load_checkpoint
from poregen.experiments.train_vae import load_vae_from_checkpoint

CKPT = REPO / "runs/ldm/ldm05-run-0001-20260827-114902-z4-c128-bs256-lr1e-04/checkpoints/latest.ckpt"
LATENTS_ROOT = REPO / "data/split_v2/latents_r07z4"
OUT = Path(__file__).parent / "compare_modes_results_130k.json"

VOLUME_MM = (4.8, 4.8, 4.8)          # 192 vox per axis
POROSITY = 0.02
LAYUP = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_VOX = 19.6
DDIM_STEPS = 50
BATCH = 4                             # shared GPU: keep every batch small


def main() -> None:
    device = torch.device("cuda")
    run_dir = CKPT.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    model = UNet3DDenoiser(UNet3DConfig.from_cfg(cfg)).to(device)
    step, _ = load_checkpoint(str(CKPT), model=model, map_location=device,
                              restore_rng=False)   # RAW weights, not EMA
    model.eval()
    print(f"Loaded RAW weights at step {step}")

    meta = json.loads((LATENTS_ROOT / "metadata.json").read_text())
    norm = meta["normalization"]
    c = len(norm["per_channel_mean"])
    latent_mean = torch.tensor(norm["per_channel_mean"], dtype=torch.float32).view(c, 1, 1, 1)
    latent_std = torch.tensor(norm["per_channel_std"], dtype=torch.float32).view(c, 1, 1, 1)
    vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), device)
    vae.requires_grad_(False)

    schedule = DDPMSchedule(T=1000, s=0.008, device=device)
    sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS,
                          s_por=1.0, s_nb=1.0)

    _st = meta["conditioning"]["por_standardisation"]
    theta = theta_from_layup(192, LAYUP, PLY_VOX)

    gen = VolumeGenerator(
        sampler=sampler, vae=vae, device=device,
        patch_size=64, generation_stride=64, neighbour_offset=64,
        latent_size=16, latent_mean=latent_mean, latent_std=latent_std,
        voxel_size_mm=0.025,
        por_log_stats=(float(_st["mean"]), float(_st["std"])),
        theta_deg=theta,
        group_order=resolve_group_order(meta),
    )

    results = {"checkpoint": str(CKPT), "step": step, "porosity": POROSITY,
               "ddim_steps": DDIM_STEPS, "volume_vox": 192, "batch": BATCH}
    for mode in ("sequential", "joint"):
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            xct, mask, stats = gen.generate(
                volume_size_mm=VOLUME_MM,
                target_porosity=POROSITY,
                autocast_dtype=torch.bfloat16,
                gen_batch_size=BATCH,
                decode_batch_size=BATCH,
                mode=mode,
                joint_window_stride=32,
                joint_window_batch=BATCH,
            )
        wall = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        results[mode] = {
            "wall_s": round(wall, 1),
            "peak_mem_gb": round(peak_gb, 2),
            "seam_xct_ratio": stats["seam_xct_ratio"],
            "seam_xct_z_ratio": stats["seam_xct_z_ratio"],
            "seam_xct_y_ratio": stats["seam_xct_y_ratio"],
            "seam_xct_x_ratio": stats["seam_xct_x_ratio"],
            "seam_mask_ratio": stats["seam_mask_ratio"],
            "seam_xct_mad": stats["seam_xct_mad"],
            "seam_xct_interior_mad": stats["seam_xct_interior_mad"],
            "actual_mask_porosity": stats["actual_mask_porosity"],
        }
        print(f"[{mode}] wall={wall:.1f}s  peak={peak_gb:.2f}GB  "
              f"seam_xct={stats['seam_xct_ratio']:.3f}  "
              f"seam_mask={stats['seam_mask_ratio']:.3f}  "
              f"por={stats['actual_mask_porosity']:.4f}")
        del xct, mask

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

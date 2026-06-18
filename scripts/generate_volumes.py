"""Generate a grid of synthetic XCT volumes across porosity and spatial distribution.

Generates anisotropic volumes using a 50×20×2 (y×x×z) patch grid with
no-overlap tiling (stride = patch_size = 64 voxels).  At 25 µm voxel size
each 64³ patch is 1.6 mm³, so the assembled volume is:
  z:  2 patches ×  64 vox =  128 vox  =  3.2 mm  (depth)
  y: 50 patches × 64 vox = 3200 vox  = 80.0 mm  (height / 8 cm)
  x: 20 patches ×  64 vox = 1280 vox  = 32.0 mm  (width / 3.2 cm)

Sweeps porosity × spatial distribution.

Usage
-----
python scripts/generate_volumes.py \\
    --checkpoint runs/ldm/ldm01-run-0002-.../checkpoints/best.ckpt \\
    --vae-run    runs/vae/r05-run-0001-... \\
    [--ddim-steps 50] \\
    [--latent-std 0.4571]

Output tree (relative to cwd)
------------------------------
<checkpoint_stem>/
  por_0.005/
    center/   volume.tif  mask.tif
    edges/    volume.tif  mask.tif
    uniform/  volume.tif  mask.tif
  por_0.01/
    ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import tifffile
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── sweep parameters ──────────────────────────────────────────────────────────
# Patch grid per axis (z=depth, y=height, x=width).
# stride = patch_size → no overlap → exact physical dimensions at 25 µm/vox:
#   z:  2 × 64 =  128 vox =  3.2 mm
#   y: 50 × 64 = 3200 vox = 80.0 mm (8 cm)
#   x: 20 × 64 = 1280 vox = 32.0 mm (3.2 cm)
GRID_Z          = 2
GRID_Y          = 50
GRID_X          = 20
PATCH_SIZE      = 64
PATCH_STRIDE    = PATCH_SIZE                            # no overlap (stride = patch_size)
VOLUME_Z        = GRID_Z * PATCH_SIZE                  # 128 voxels
VOLUME_Y        = GRID_Y * PATCH_SIZE                  # 3200 voxels
VOLUME_X        = GRID_X * PATCH_SIZE                  # 1280 voxels
POROSITY_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
DISTRIBUTIONS   = ["center", "edges", "uniform"]
DEFAULT_DDIM_STEPS = 250
# ─────────────────────────────────────────────────────────────────────────────


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
    return here.parent


def _build_local_por_map(
    gz: int,
    gy: int,
    gx: int,
    target_por: float,
    distribution: str,
    sigma_norm: float = 0.3,
) -> dict[tuple[int, int, int], float]:
    """Build per-patch local porosity dict keyed by (iz, iy, ix).

    center  — 3D Gaussian weight map; pores concentrated at grid midpoint.
    edges   — 1 − Gaussian; pores concentrated at corners/shell.
    uniform — every patch conditioned to target_por exactly (calibration baseline).
    """
    if distribution == "uniform":
        val = float(np.clip(target_por, 0.001, 0.999))
        return {(iz, iy, ix): val for iz in range(gz) for iy in range(gy) for ix in range(gx)}

    import math
    weights = np.empty((gz, gy, gx), dtype=np.float64)
    denom = 2.0 * sigma_norm ** 2
    for iz in range(gz):
        dz = iz / max(gz - 1, 1) - 0.5
        for iy in range(gy):
            dy = iy / max(gy - 1, 1) - 0.5
            for ix in range(gx):
                dx = ix / max(gx - 1, 1) - 0.5
                weights[iz, iy, ix] = math.exp(-(dz * dz + dy * dy + dx * dx) / denom)

    if distribution == "edges":
        weights = 1.0 - weights

    mean_w = float(weights.mean())
    if mean_w < 1e-8:
        weights = np.full_like(weights, target_por)
    else:
        weights = weights * (target_por / mean_w)

    weights = np.clip(weights, 0.001, 0.999)
    return {
        (iz, iy, ix): float(weights[iz, iy, ix])
        for iz in range(gz)
        for iy in range(gy)
        for ix in range(gx)
    }


def _load_ldm(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
    from poregen.training.checkpoint import load_checkpoint

    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    model_cfg = UNet3DConfig.from_cfg(cfg)
    model = UNet3DDenoiser(model_cfg).to(device)

    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    if "ema" in raw:
        ema_state = raw["ema"]
        if any(k.startswith("_orig_mod.") for k in ema_state):
            ema_state = {k.removeprefix("_orig_mod."): v for k, v in ema_state.items()}
        model.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
        logger.info("Loaded EMA weights from %s", checkpoint)
    else:
        load_checkpoint(str(checkpoint), model=model, map_location=device)
        logger.warning("No EMA key in checkpoint — using training weights")

    model.eval()
    return model, cfg


def _load_vae(vae_run_dir: str | Path, device: torch.device) -> torch.nn.Module:
    from poregen.models.vae import build_vae

    vae_run_dir = Path(vae_run_dir)
    resolved = yaml.safe_load((vae_run_dir / "resolved_config.yaml").read_text())
    model_cfg = dict(resolved["model"])
    vae_name = model_cfg.pop("name")
    vae = build_vae(vae_name, **model_cfg)

    for candidate in [
        vae_run_dir / "checkpoints" / "best.ckpt",
        vae_run_dir / "checkpoints" / "latest.ckpt",
        vae_run_dir / "best.ckpt",
        vae_run_dir / "latest.ckpt",
    ]:
        if candidate.exists():
            ckpt_path = candidate
            break
    else:
        raise FileNotFoundError(f"No best.ckpt or latest.ckpt found in {vae_run_dir}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    vae.load_state_dict(state)
    vae.to(device).eval().requires_grad_(False)
    logger.info("Loaded VAE from %s", ckpt_path)
    return vae


def _load_latent_std(repo_root: Path, override: float | None) -> float:
    if override is not None:
        return override
    stats = repo_root / "data" / "split_v2" / "latents_s64" / "latent_scale_stats.json"
    if stats.exists():
        val = float(json.loads(stats.read_text())["std"])
        logger.info("latent_std=%.6f (from %s)", val, stats)
        return val
    logger.warning("latent_scale_stats.json not found — using latent_std=1.0")
    return 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a sweep of synthetic XCT volumes.")
    ap.add_argument("--checkpoint",  required=True, help="Path to LDM checkpoint (.ckpt)")
    ap.add_argument("--vae-run",     required=True, help="Path to VAE run directory")
    ap.add_argument("--sampler",     choices=["ddim", "ddpm"], default="ddim",
                    help="Sampler to use (default: ddim)")
    ap.add_argument("--ddim-steps",  type=int, default=DEFAULT_DDIM_STEPS,
                    help="Number of DDIM steps (ignored when --sampler ddpm)")
    ap.add_argument("--latent-std",  type=float, default=None,
                    help="Latent scale std (read from latent_scale_stats.json if omitted)")
    args = ap.parse_args()

    repo = _find_repo_root()
    sys.path.insert(0, str(repo / "src"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        autocast_dtype = torch.bfloat16

    from poregen.diffusion.noise_schedule import DDPMSchedule
    from poregen.diffusion.sampler import DDIMSampler, DDPMSampler, VolumeGenerator

    ldm, ldm_cfg = _load_ldm(args.checkpoint, device)
    vae = _load_vae(args.vae_run, device)
    latent_std = _load_latent_std(repo, args.latent_std)

    sched_cfg = ldm_cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(
        T=int(sched_cfg.get("T", 1000)),
        s=float(sched_cfg.get("s", 0.008)),
        device=device,
    )

    if args.sampler == "ddpm":
        sampler = DDPMSampler(ldm, schedule, device)
        logger.info("Using DDPM sampler (T=%d steps)", schedule.T)
    else:
        sampler = DDIMSampler(ldm, schedule, device, n_steps=args.ddim_steps)
        logger.info("Using DDIM sampler (%d steps)", args.ddim_steps)
    generator = VolumeGenerator(
        sampler=sampler,
        vae=vae,
        device=device,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        latent_size=PATCH_SIZE // 4,
        latent_std=latent_std,
    )

    vol_shape = (VOLUME_Z, VOLUME_Y, VOLUME_X)
    out_root  = Path(Path(args.checkpoint).stem)
    combinations = list(product(POROSITY_LEVELS, DISTRIBUTIONS))
    n_patches_per_vol = GRID_Z * GRID_Y * GRID_X

    logger.info(
        "Generating %d volumes  shape=%s  grid=%dx%dx%d  patches_per_vol=%d"
        "  device=%s  ddim_steps=%d  latent_std=%.5f",
        len(combinations), vol_shape, GRID_Z, GRID_Y, GRID_X, n_patches_per_vol,
        device, args.ddim_steps, latent_std,
    )

    vol_pbar   = tqdm(total=len(combinations), unit="vol", position=0)
    patch_pbar = tqdm(unit="patch", position=1, leave=False)

    with vol_pbar, patch_pbar:
        for por_level, dist in combinations:
            vol_pbar.set_description(f"por={por_level:.3f} {dist}")

            local_por_map = _build_local_por_map(GRID_Z, GRID_Y, GRID_X, por_level, dist)

            patch_pbar.reset(total=n_patches_per_vol)
            patch_pbar.set_description("patches")

            with torch.no_grad():
                xct_u8, mask_u8 = generator.generate(
                    volume_shape=vol_shape,
                    target_porosity=por_level,
                    autocast_dtype=autocast_dtype,
                    local_por_map=local_por_map,
                    patch_pbar=patch_pbar,
                )

            xct_f32 = xct_u8.astype(np.float32) / 255.0

            out_dir = out_root / f"por_{por_level}" / dist
            out_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out_dir / "volume.tif"), xct_f32)
            tifffile.imwrite(str(out_dir / "mask.tif"), mask_u8)

            vol_pbar.update(1)

    logger.info("Done. Outputs in %s/", out_root)


if __name__ == "__main__":
    main()

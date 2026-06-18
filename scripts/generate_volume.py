"""Generate synthetic TIFF volumes using a trained LDM + VAE.

Usage
-----
python scripts/generate_volume.py \\
    --ldm-checkpoint  runs/ldm/ldm01-run-001-.../checkpoints/best.ckpt \\
    --vae-checkpoint  runs/vae/r05-run-001-.../checkpoints/best.ckpt \\
    --ldm-experiment  ldm01/base \\
    --vae-experiment  r05/base \\
    --size 256 256 256 \\
    [--porosity 0.05] \\
    --output-dir outputs/ \\
    [--n-volumes 1] \\
    [--seed 42] \\
    [--patch-size 64] \\
    [--patch-stride 32]

Outputs
-------
outputs/vol_000_xct.tiff
outputs/vol_000_mask.tiff
... (repeated for each volume)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
    return here.parent


def _load_ldm(experiment: str, checkpoint: str, device: torch.device) -> "torch.nn.Module":
    from poregen.configuration import resolve_experiment
    from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
    from poregen.training.checkpoint import load_checkpoint

    repo = _find_repo_root()
    resolved = resolve_experiment(experiment, repo_root=repo)
    cfg = resolved.cfg
    model_cfg = UNet3DConfig.from_cfg(cfg)
    model = UNet3DDenoiser(model_cfg).to(device)

    # Prefer EMA weights for generation; fall back to training weights
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict({k: v.to(device) for k, v in ckpt["ema"].items()})
        logger.info("Loaded EMA weights from %s", checkpoint)
    else:
        load_checkpoint(checkpoint, model=model, map_location=device)
        logger.warning("No EMA weights found in checkpoint — using training weights")

    model.eval()
    return model


def _load_vae(experiment: str, checkpoint: str, device: torch.device) -> "torch.nn.Module":
    from poregen.configuration import resolve_experiment
    from poregen.models.vae import build_vae
    from poregen.training.checkpoint import load_checkpoint

    repo = _find_repo_root()
    resolved = resolve_experiment(experiment, repo_root=repo)
    cfg = resolved.cfg
    mc = cfg["model"]
    model = build_vae(
        mc["name"],
        in_channels=mc.get("in_channels", 2),
        z_channels=int(mc["z_channels"]),
        base_channels=int(mc["base_channels"]),
        n_blocks=int(mc["n_blocks"]),
        patch_size=int(mc["patch_size"]),
    ).to(device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    logger.info("Loaded VAE from %s", checkpoint)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic 3D TIFF volumes with PoreGen LDM.")
    ap.add_argument("--ldm-checkpoint",  required=True)
    ap.add_argument("--vae-checkpoint",  required=True)
    ap.add_argument("--ldm-experiment",  required=True, help="LDM experiment ref, e.g. 'ldm01/base'")
    ap.add_argument("--vae-experiment",  required=True, help="VAE experiment ref, e.g. 'r05/base'")
    ap.add_argument("--size", type=int, nargs=3, default=[256, 256, 256],
                    metavar=("D", "H", "W"), help="Volume shape in voxels")
    ap.add_argument("--porosity", type=float, default=None,
                    help="Target VVF for conditioning (default: 0.05)")
    ap.add_argument("--output-dir",  default="outputs/generated")
    ap.add_argument("--n-volumes",   type=int, default=1)
    ap.add_argument("--seed",        type=int, default=None)
    ap.add_argument("--patch-size",  type=int, default=64)
    ap.add_argument("--patch-stride", type=int, default=32)
    ap.add_argument("--sampler",     choices=["ddpm", "ddim"], default="ddim",
                    help="Reverse sampler: ddim (fast, 50 steps) or ddpm (1000 steps)")
    ap.add_argument("--ddim-steps",  type=int, default=50,
                    help="Number of DDIM denoising steps (only used with --sampler ddim)")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    repo = _find_repo_root()
    sys.path.insert(0, str(repo / "src"))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)

    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    from poregen.configuration import resolve_experiment
    from poregen.diffusion.noise_schedule import DDPMSchedule
    from poregen.diffusion.sampler import DDIMSampler, DDPMSampler, VolumeGenerator

    ldm_model = _load_ldm(args.ldm_experiment, args.ldm_checkpoint, device)
    vae_model = _load_vae(args.vae_experiment, args.vae_checkpoint, device)

    # Read noise schedule params from LDM config
    _ldm_resolved = resolve_experiment(args.ldm_experiment, repo_root=repo)
    _sched_cfg    = _ldm_resolved.cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(
        T=int(_sched_cfg.get("T", 1000)),
        s=float(_sched_cfg.get("s", 0.008)),
        device=device,
    )

    # Load latent std for unscaling before VAE decode
    _latents_root = Path(_ldm_resolved.cfg.get("data", {}).get("latents_root", "data/split_v2/latents"))
    if not _latents_root.is_absolute():
        _latents_root = repo / _latents_root
    _stats_path = _latents_root / "latent_scale_stats.json"
    latent_std = 1.0
    if _stats_path.exists():
        latent_std = float(json.loads(_stats_path.read_text())["std"])
        logger.info("latent_std=%.4f (from %s)", latent_std, _stats_path)
    else:
        logger.warning("latent_scale_stats.json not found — using latent_std=1.0")

    latent_size = args.patch_size // 4

    if args.sampler == "ddim":
        sampler = DDIMSampler(ldm_model, schedule, device, n_steps=args.ddim_steps)
        logger.info("Using DDIM sampler with %d steps", args.ddim_steps)
    else:
        sampler = DDPMSampler(ldm_model, schedule, device)
        logger.info("Using DDPM sampler with T=%d steps", schedule.T)

    generator = VolumeGenerator(
        sampler, vae_model, device,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        latent_size=latent_size,
        latent_std=latent_std,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_shape = tuple(args.size)
    logger.info(
        "Generating %d volume(s) of shape %s, porosity=%s, device=%s",
        args.n_volumes, volume_shape, args.porosity, device,
    )

    for i in range(args.n_volumes):
        logger.info("Volume %d / %d …", i + 1, args.n_volumes)
        xct, mask = generator.generate(
            volume_shape,
            target_porosity=args.porosity,
            autocast_dtype=autocast_dtype,
        )
        xct_path  = output_dir / f"vol_{i:03d}_xct.tiff"
        mask_path = output_dir / f"vol_{i:03d}_mask.tiff"
        VolumeGenerator.save_tiff(xct, mask, xct_path, mask_path)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()

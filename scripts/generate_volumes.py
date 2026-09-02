"""Generate a grid of synthetic XCT volumes across porosity and spatial distribution.

Generates anisotropic volumes at 25 µm/voxel with the ldm06 hybrid chunked
sampler.  Physical volume:
  z:  3.2 mm  (128 vox)
  y: 80.0 mm  (3200 vox)
  x: 32.0 mm  (1280 vox)

Sweeps porosity × spatial distribution.

Usage
-----
python scripts/generate_volumes.py \\
    --checkpoint   runs/ldm/ldm06-run-0001-.../checkpoints/best.ckpt \\
    [--latents-root data/split_v3/latents_r08z4] \\
    [--ddim-steps 50] [--chunk-tiles 3 3 3]

The latent store's metadata.json supplies both the per-channel
denormalisation stats and the VAE checkpoint the latents were built with.

Output tree (relative to cwd)
------------------------------
inference/<ldm_run>/<run_tag>/
  por_0.005/
    center/   volume.tif  label.tif  generation_stats.json
    edges/    ...
    uniform/  ...
  por_0.01/
    ...

``volume.tif`` is uint8 on the RAW-SCAN grey scale and ``label.tif`` is uint8
{0 material, 1 pore, 2 air} — both native, so a generated volume is measured
with exactly the tools a real one is.  Rescaling either of them cost a whole
evaluation campaign once already.
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
# Physical volume size (z, y, x) in mm and voxel pitch.
VOLUME_SIZE_MM     = (3.2, 80.0, 32.0)   # z, y, x in mm
VOXEL_SIZE_MM      = 0.025               # 25 µm / voxel
PATCH_SIZE         = 64                  # one tile; also the neighbour offset
TILE_SIZE          = PATCH_SIZE          # the porosity field's grid unit
# Requested layup (D32 §4): θ(z) is written directly, never estimated.
LAYUP_ANGLES_DEG   = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_THICKNESS_VOX  = 19.6
POROSITY_LEVELS    = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
DISTRIBUTIONS      = ["center", "edges", "uniform", "coherent"]
DEFAULT_DDIM_STEPS = 200
# Batch sizes calibrated to ≤75% of 128 GB unified memory (GB10 DGX Spark).
# Measured on CPU (conservative upper bound; GPU bfloat16 autocast uses ~half):
#   UNet forward:  ~42 MB/patch  (empirical: B=2048 → 85 GB RSS)
#   VAE decode:    ~40 MB/patch  (marginal slope B=32→64)
# Fixed overhead: ~10.3 GB (models + OS + latent dict + 3 accumulators)
# Budget: 128 × 0.75 = 96 GB → (96 - 10.3) GB / 42 MB ≈ 2040 → round to 2048
WINDOW_BATCH       = 32
DECODE_BATCH_SIZE  = 64
DEFAULT_CHUNK_TILES = (3, 3, 3)
DEFAULT_WINDOW_STRIDE = 32
DEFAULT_DECODE_STRIDE = 32
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

    center   — 3D Gaussian weight map; pores concentrated at grid midpoint.
    edges    — 1 − Gaussian; pores concentrated at corners/shell.
    uniform  — every patch conditioned to target_por exactly (calibration baseline).
    coherent — one spatially coherent field over the volume (D32 §4): T-E
               marginal draw per patch, smoothed with the T-D correlation
               lengths, mean rescaled to target_por.  Seeded from the
               porosity level so repeated runs are reproducible.
    """
    if distribution == "uniform":
        val = float(np.clip(target_por, 0.001, 0.999))
        return {(iz, iy, ix): val for iz in range(gz) for iy in range(gy) for ix in range(gx)}

    if distribution == "coherent":
        from poregen.diffusion.porosity_field import (
            DEFAULT_TD_RESULTS,
            DEFAULT_TE_RESULTS,
            build_porosity_field,
            load_corr_lengths_voxels,
            load_sampler,
        )

        repo_root = _find_repo_root()
        field = build_porosity_field(
            grid_shape=(gz, gy, gx),
            target=target_por,
            sampler=load_sampler(repo_root / DEFAULT_TE_RESULTS),
            corr_lengths_voxels=load_corr_lengths_voxels(repo_root / DEFAULT_TD_RESULTS),
            stride_voxels=TILE_SIZE,
            # The script has no global seed; derive one from the porosity
            # level so each sweep cell is deterministic across runs.
            seed=int(round(target_por * 1e6)),
        )
        return {
            (iz, iy, ix): float(field[iz, iy, ix])
            for iz in range(gz)
            for iy in range(gy)
            for ix in range(gx)
        }

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


def _load_latent_store_meta(
    latents_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict]:
    """Load the frozen VAE, per-channel denormalisation stats and the store metadata."""
    from poregen.experiments.train_vae import load_vae_from_checkpoint

    meta = json.loads((latents_root / "metadata.json").read_text())
    norm = meta["normalization"]
    c = len(norm["per_channel_mean"])
    mean = torch.tensor(norm["per_channel_mean"], dtype=torch.float32).view(c, 1, 1, 1)
    std  = torch.tensor(norm["per_channel_std"],  dtype=torch.float32).view(c, 1, 1, 1)

    vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), device)
    vae.requires_grad_(False)
    return vae, mean, std, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a sweep of synthetic XCT volumes.")
    ap.add_argument("--checkpoint",  required=True, help="Path to LDM checkpoint (.ckpt)")
    ap.add_argument("--latents-root", default="data/split_v3/latents_r08z4",
                    help="Latent store root (metadata.json supplies VAE checkpoint + norm stats)")
    ap.add_argument("--ddim-steps",  type=int, default=DEFAULT_DDIM_STEPS,
                    help="Number of DDIM steps")
    ap.add_argument("--s-por", type=float, default=None,
                    help="Porosity guidance scale (default: from resolved_config.yaml guidance.s_por, else 1.0)")
    ap.add_argument("--s-nb",  type=float, default=None,
                    help="Neighbour guidance scale (default: from resolved_config.yaml guidance.s_nb, else 1.0)")
    ap.add_argument("--chunk-tiles", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                    help="Tiles per jointly denoised chunk (default: config "
                         "generation.chunk_tiles, else 3 3 3).  1 1 1 = patch-at-a-time.")
    ap.add_argument("--window-stride", type=int, default=None,
                    help="Voxels between denoising window origins inside a chunk "
                         "(default: config generation.window_stride, else 32)")
    ap.add_argument("--decode-stride", type=int, default=None,
                    help="Voxels between decode window origins (default: config "
                         "generation.decode_stride, else 32)")
    ap.add_argument("--window-batch", type=int, default=WINDOW_BATCH,
                    help="Windows per UNet forward per timestep")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Resume into this existing run directory instead of creating a new "
                         "timestamped one (e.g. inference/<ldm_run>/<run_tag>). Combos whose "
                         "volume.tif + label.tif already exist there are skipped.")
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
    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator, theta_from_layup

    ldm, ldm_cfg = _load_ldm(args.checkpoint, device)
    latents_root = Path(args.latents_root)
    if not latents_root.is_absolute():
        latents_root = (repo / latents_root).resolve()
    vae, latent_mean, latent_std, store_meta = _load_latent_store_meta(latents_root, device)

    sched_cfg = ldm_cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(
        T=int(sched_cfg.get("T", 1000)),
        s=float(sched_cfg.get("s", 0.008)),
        device=device,
    )

    # Resolve guidance scales: CLI flags override resolved config, which overrides 1.0 default.
    guidance_cfg = ldm_cfg.get("guidance", {})
    s_por = float(args.s_por if args.s_por is not None else guidance_cfg.get("s_por", 1.0))
    s_nb  = float(args.s_nb  if args.s_nb  is not None else guidance_cfg.get("s_nb",  1.0))

    # Sampler geometry: CLI overrides the config's generation block, which overrides
    # the defaults.  The training run logged its own values here, so a generation run
    # that says nothing reproduces the geometry the run was babysat with.
    generation_cfg = ldm_cfg.get("generation", {}) or {}
    chunk_tiles = tuple(
        args.chunk_tiles if args.chunk_tiles is not None
        else generation_cfg.get("chunk_tiles", DEFAULT_CHUNK_TILES)
    )
    window_stride = int(
        args.window_stride if args.window_stride is not None
        else generation_cfg.get("window_stride", DEFAULT_WINDOW_STRIDE)
    )
    decode_stride = int(
        args.decode_stride if args.decode_stride is not None
        else generation_cfg.get("decode_stride", DEFAULT_DECODE_STRIDE)
    )

    sampler = DDIMSampler(ldm, schedule, device, n_steps=args.ddim_steps,
                          s_por=s_por, s_nb=s_nb)
    logger.info("DDIM sampler (%d steps)  guided=%s  s_por=%.2f  s_nb=%.2f",
                args.ddim_steps, sampler.guided, s_por, s_nb)

    # Derive voxel dimensions (snapped to nearest patch_size multiple, downward)
    vol_shape = tuple(
        (round(d / VOXEL_SIZE_MM) // PATCH_SIZE) * PATCH_SIZE for d in VOLUME_SIZE_MM
    )

    cond_meta = store_meta.get("conditioning") or {}
    _st = cond_meta.get("por_standardisation")
    if _st is None:
        raise SystemExit(
            f"{latents_root}/metadata.json has no conditioning.por_standardisation — "
            "run scripts/build_conditioning.py before generating."
        )
    por_log_stats = (float(_st["mean"]), float(_st["std"]))
    theta_deg = theta_from_layup(vol_shape[0], LAYUP_ANGLES_DEG, PLY_THICKNESS_VOX)

    generator = VolumeGenerator(
        sampler=sampler,
        vae=vae,
        device=device,
        patch_size=PATCH_SIZE,
        latent_size=PATCH_SIZE // 4,
        latent_mean=latent_mean,
        latent_std=latent_std,
        voxel_size_mm=VOXEL_SIZE_MM,
        por_log_stats=por_log_stats,
        theta_deg=theta_deg,
        chunk_tiles=chunk_tiles,
        window_stride=window_stride,
        decode_stride=decode_stride,
    )
    # Tile grid — the grid the requested porosity field is defined on.
    gz, gy, gx = (v // TILE_SIZE for v in vol_shape)
    n_chunks = 1
    for n, c in zip((gz, gy, gx), chunk_tiles):
        n_chunks *= -(-n // c)

    # Build a human-readable run folder under inference/.
    import datetime
    ldm_run_name = Path(args.checkpoint).parent.parent.name   # e.g. ldm06-run-0001-...
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = (
        f"{ts}"
        f"-chunk{'x'.join(str(c) for c in chunk_tiles)}"
        f"-ddim{args.ddim_steps}"
        f"-spor{s_por:.2f}"
        f"-snb{s_nb:.2f}"
    )
    out_root = Path(args.out_dir) if args.out_dir else Path("inference") / ldm_run_name / run_tag
    combinations = list(product(POROSITY_LEVELS, DISTRIBUTIONS))

    logger.info(
        "Generating %d volumes  shape=%s  tiles=%dx%dx%d  chunks=%d  chunk_tiles=%s"
        "  window_stride=%d  decode_stride=%d  device=%s  ddim_steps=%d",
        len(combinations), vol_shape, gz, gy, gx, n_chunks, chunk_tiles,
        window_stride, decode_stride, device, args.ddim_steps,
    )

    vol_pbar  = tqdm(total=len(combinations), unit="vol", position=0)
    step_pbar = tqdm(unit="step", position=1, leave=False)

    with vol_pbar, step_pbar:
        for por_level, dist in combinations:
            vol_pbar.set_description(f"por={por_level:.3f} {dist}")

            out_dir = out_root / f"por_{por_level}" / dist
            if (out_dir / "volume.tif").exists() and (out_dir / "label.tif").exists():
                logger.info("Skipping por=%.3f %s — already generated at %s", por_level, dist, out_dir)
                vol_pbar.update(1)
                continue

            local_por_map = _build_local_por_map(gz, gy, gx, por_level, dist)

            # One progress tick per DDIM step, over every chunk.
            step_pbar.reset(total=n_chunks * args.ddim_steps)
            step_pbar.set_description("ddim steps")

            with torch.no_grad():
                xct_u8, label_u8, gen_stats = generator.generate(
                    volume_size_mm=VOLUME_SIZE_MM,
                    target_porosity=por_level,
                    autocast_dtype=autocast_dtype,
                    local_por_map=local_por_map,
                    progress=step_pbar,
                    window_batch=args.window_batch,
                    decode_batch_size=DECODE_BATCH_SIZE,
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            # Both arrays go out on their native scale — see the module docstring.
            tifffile.imwrite(str(out_dir / "volume.tif"), xct_u8)
            tifffile.imwrite(str(out_dir / "label.tif"), label_u8)
            gen_stats["distribution"] = dist
            (out_dir / "generation_stats.json").write_text(json.dumps(gen_stats, indent=2))

            vol_pbar.update(1)

    logger.info("Done. Outputs in %s/", out_root)


if __name__ == "__main__":
    main()

"""Shared infrastructure for the eval v2 campaign (phase 1).

Three generation arms with corrected conditioning semantics
(src/poregen/diffusion/sampler.py, VolumeGenerator.conditioning_semantics):

    seq          mode=sequential, semantics=specimen (default), s_por 1.0
    joint_legacy mode=joint,      semantics=legacy,             s_por 1.5
    joint_oob    mode=joint,      semantics=specimen,           s_por 1.5

The per-arm s_por is each mode's best known operating point (dose-response +
cfg-sweep v1).  Output goes under ``$POREGEN_EVAL_ROOT`` (default
``runs/campaigns/03-eval-v2-buggy-decode``); the eval-v3 campaign points it at ``runs/campaigns/05-eval-v3-fixed-decode`` so the
buggy-decode record in ``runs/campaigns/03-eval-v2-buggy-decode`` is never touched.  Every generated
volume is saved (volume.tif uint8 grey level on the raw-scan scale, mask.tif
uint8 0/255) with a per-volume stats.json, so a crashed run resumes by
skipping cells whose stats.json already exists.

Void-corrected porosity uses the calibrated absolute grayscale threshold from
runs/campaigns/02-porosity-control-v1/void_mask_audit/results.json (u8 threshold 185, pooled dice
0.799 on real volumes): corrected = mean((mask > 0) | (xct_u8 < T)).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.conditioning import resolve_group_order  # noqa: E402
from poregen.diffusion.noise_schedule import DDPMSchedule  # noqa: E402
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS, build_porosity_field,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import (  # noqa: E402
    DDIMSampler, VolumeGenerator, theta_from_layup,
)
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402

CKPT = REPO / ("runs/ldm/ldm05-run-0001-20260827-114902-z4-c128-bs256-lr1e-04/"
               "checkpoints/ldm_step00130000.ckpt")
LATENTS_ROOT = REPO / "data/split_v2/latents_r07z4"
# Campaign output root.  One campaign = one root; POREGEN_EVAL_ROOT selects it
# so a re-run never writes over the record of a previous one.
EVAL_ROOT = Path(os.environ.get("POREGEN_EVAL_ROOT", str(REPO / "runs/campaigns/03-eval-v2-buggy-decode")))
VOID_AUDIT_RESULTS = REPO / "runs/campaigns/02-porosity-control-v1/void_mask_audit/results.json"

# arm name -> (mode, conditioning_semantics, s_por operating point)
ARMS = {
    "seq":          ("sequential", "specimen", 1.0),
    "joint_legacy": ("joint",      "legacy",   1.5),
    "joint_oob":    ("joint",      "specimen", 1.5),
}
SEEDS = [101, 202, 303]

VOLUME_MM = (4.8, 4.8, 4.8)              # 192 vox per axis at 0.025 mm/vox
VOLUME_VOX = 192
GRID = (3, 3, 3)                          # 3x3x3 patch grid, stride 64
LAYUP = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_VOX = 19.6
DDIM_STEPS = 50
GEN_BATCH = 27
DECODE_BATCH = 27
JOINT_WINDOW_STRIDE = 32
GATE = 0.005                              # D39: |delivered - requested| < 0.005
DEGEN_THR = 1e-4                          # 64^3 subblock porosity below this

ARM_COLORS = {"seq": "#1b6ca8", "joint_legacy": "#c2571a", "joint_oob": "#2e7d32"}


def load_void_detector() -> dict:
    """Calibrated dark-voxel detector from the void-mask audit.

    Since the spurious ``expit`` was removed from the decode path
    (2026-09-01) generated volumes share the real u8 scale, so the absolute
    calibrated threshold transfers directly.  The material-referenced form is
    kept for volumes produced before that fix.
    """
    cal = json.loads(VOID_AUDIT_RESULTS.read_text())
    det = cal["detector"]
    return {"k": float(det["k_material_referenced"]),
            "absolute_threshold_u8": int(det["absolute_threshold"]),
            "pooled_dice": float(det["pooled_dice"])}


def build_generator(device: torch.device):
    """Build model + VAE + VolumeGenerator once; arms swap sampler/semantics."""
    run_dir = CKPT.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    model = UNet3DDenoiser(UNet3DConfig.from_cfg(cfg)).to(device)
    step, _ = load_checkpoint(str(CKPT), model=model, map_location=device,
                              restore_rng=False)   # RAW weights, not EMA
    model.eval()
    print(f"Loaded RAW (non-EMA) weights at step {step}", flush=True)

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
    theta = theta_from_layup(VOLUME_VOX, LAYUP, PLY_VOX)

    gen = VolumeGenerator(
        sampler=sampler, vae=vae, device=device,
        patch_size=64, generation_stride=64, neighbour_offset=64,
        latent_size=16, latent_mean=latent_mean, latent_std=latent_std,
        voxel_size_mm=0.025,
        por_log_stats=(float(_st["mean"]), float(_st["std"])),
        theta_deg=theta,
        group_order=resolve_group_order(meta),
    )
    return gen, model, schedule, step


def configure_arm(gen, model, schedule, device, arm: str,
                  s_por: float | None = None):
    """Point the generator at one arm: mode + semantics + guidance scale.

    Returns the generation mode.  ``s_por=None`` uses the arm's operating
    point; a value overrides it (CFG sweep).
    """
    mode, semantics, s_por_default = ARMS[arm]
    sp = s_por_default if s_por is None else s_por
    if gen.sampler.s_por != sp:
        gen.sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS,
                                  s_por=sp, s_nb=1.0)
    gen.conditioning_semantics = semantics
    return mode, sp


def coherent_por_map(target: float, seed: int, te_sampler, corr_lengths):
    field = build_porosity_field(
        grid_shape=GRID, target=target, sampler=te_sampler,
        corr_lengths_voxels=corr_lengths, stride_voxels=64, seed=seed,
    )
    por_map = {
        (iz, iy, ix): float(field[iz, iy, ix])
        for iz in range(GRID[0]) for iy in range(GRID[1]) for ix in range(GRID[2])
    }
    return por_map, field


def run_one(gen, mode: str, por_map: dict, joint_window_batch: int):
    """Generate one volume; on CUDA OOM halve joint_window_batch and retry.

    Returns (xct_u8, mask_u8, stats, wall_s, jwb_used, oom_events).
    """
    oom_events = []
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
                    joint_window_batch=joint_window_batch,
                )
            wall = time.perf_counter() - t0
            return xct, mask, stats, wall, joint_window_batch, oom_events
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if joint_window_batch <= 1:
                raise
            oom_events.append(
                f"OOM at joint_window_batch={joint_window_batch}: {exc}")
            joint_window_batch //= 2
            print(f"  OOM — retrying with joint_window_batch={joint_window_batch}",
                  flush=True)


# ---------------------------------------------------------------------------
# Per-volume metrics
# ---------------------------------------------------------------------------

def corrected_porosity(xct_u8: np.ndarray, mask_u8: np.ndarray,
                       detector: dict) -> tuple[float, float]:
    """Union(mask, dark-voxel detector) porosity — catches voids the mask
    head missed but the grayscale channel rendered dark.

    Material-referenced detector on the raw generated u8 scale (the audit's
    recommendation for generated volumes): threshold
    ``material_mode − k · spread`` from the volume's own material histogram
    (scripts/analysis/void_mask_audit.py, material_stats), k calibrated on
    real volumes.  Returns (union_porosity, threshold_used).
    """
    from void_mask_audit import material_stats

    mode, spread = material_stats(xct_u8, None)
    thr = mode - detector["k"] * spread
    return float(((mask_u8 > 0) | (xct_u8 < thr)).mean()), float(thr)


def cell_porosities(mask_u8: np.ndarray) -> dict:
    """Delivered mask porosity per 64^3 tile cell, keyed '(iz,iy,ix)'."""
    out = {}
    for iz in range(GRID[0]):
        for iy in range(GRID[1]):
            for ix in range(GRID[2]):
                blk = mask_u8[iz * 64:(iz + 1) * 64,
                              iy * 64:(iy + 1) * 64,
                              ix * 64:(ix + 1) * 64]
                out[f"{iz},{iy},{ix}"] = float((blk > 0).mean())
    return out


def degenerate_fraction(mask_u8: np.ndarray) -> float:
    """Fraction of the 27 stride-64 64^3 subblocks with porosity < DEGEN_THR."""
    cells = cell_porosities(mask_u8)
    return float(np.mean([v < DEGEN_THR for v in cells.values()]))


# ---------------------------------------------------------------------------
# Volume persistence (generate_volumes.py convention) + resume
# ---------------------------------------------------------------------------

def save_volume(vol_dir: Path, xct_u8: np.ndarray, mask_u8: np.ndarray,
                record: dict) -> None:
    vol_dir.mkdir(parents=True, exist_ok=True)
    # Store the native uint8 grey level — the same scale and dtype as the real
    # scans.  Writing u8/255 as float32 quadrupled the file size and preserved
    # nothing, and it invited readers to guess at a transform.
    tifffile.imwrite(str(vol_dir / "volume.tif"), xct_u8)
    tifffile.imwrite(str(vol_dir / "mask.tif"), mask_u8)
    with open(vol_dir / "stats.json", "w") as fh:
        json.dump(record, fh, indent=2)


def load_existing(vol_dir: Path) -> dict | None:
    """Resume support: a completed cell has all three files."""
    if all((vol_dir / f).exists()
           for f in ("volume.tif", "mask.tif", "stats.json")):
        return json.loads((vol_dir / "stats.json").read_text())
    return None


def fit_ols(requested: np.ndarray, delivered: np.ndarray) -> dict:
    slope, intercept = np.polyfit(requested, delivered, 1)
    pred = slope * requested + intercept
    ss_res = float(np.sum((delivered - pred) ** 2))
    ss_tot = float(np.sum((delivered - delivered.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}

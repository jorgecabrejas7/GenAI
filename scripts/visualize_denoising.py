"""Visualise DDIM denoising progression for a single patch.

For each (porosity, n_steps) combination, runs an independent DDIM chain from a
fixed seed, decodes every intermediate latent through the 3-class VAE, and
writes an MP4 showing the middle z/y/x slices of the XCT, the pore probability
and the argmax label evolving from noise to final output.

The patch is conditioned as a mid-thickness interior one: depth 0.5, all six
face distances saturated (no surface within 64 voxels), an all-material map and
no neighbours (all UNKNOWN at nb_t 0 — the CFG neighbour null).

Usage
-----
python scripts/visualize_denoising.py \\
    --checkpoint runs/ldm/.../checkpoints/best.ckpt \\
    --vae-run    runs/vae/r08-run-... \\
    [--latents-root data/split_v3/latents_r08z4]

Output
------
<checkpoint_stem>_denoising/
  por_0.005/
    steps_0020_ddim.mp4  steps_0050_ddim.mp4  ...
  por_0.01/
    ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────────────
DDIM_STEP_COUNTS = [20, 50, 100, 200, 500, 1000]
POROSITY_LEVELS  = [0.005, 0.01, 0.02, 0.05]
SEED             = 42
FPS              = 15
PATCH_SIZE       = 64
LATENT_SIZE      = 16
# Nominal layup (D32 §4) used to write the orientation conditioning.
LAYUP_ANGLES_DEG = [45, -45, 90, 0, 45, -45, 0, 90, -45, 45]
PLY_THICKNESS_VOX = 19.6
# ─────────────────────────────────────────────────────────────────────────────


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
    return here.parent


def _load_ldm(checkpoint: str | Path, device: torch.device):
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
    else:
        load_checkpoint(str(checkpoint), model=model, map_location=device)

    model.eval()
    return model, cfg


def _load_vae(vae_run_dir: str | Path, device: torch.device):
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
        raise FileNotFoundError(f"No checkpoint found in {vae_run_dir}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    vae.load_state_dict(state)
    vae.to(device).eval().requires_grad_(False)
    return vae


def _load_latent_stats(latents_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel denormalisation stats from the latent store's metadata.json."""
    norm = json.loads((latents_root / "metadata.json").read_text())["normalization"]
    c = len(norm["per_channel_mean"])
    mean = torch.tensor(norm["per_channel_mean"], dtype=torch.float32).view(c, 1, 1, 1)
    std  = torch.tensor(norm["per_channel_std"],  dtype=torch.float32).view(c, 1, 1, 1)
    return mean, std


@torch.no_grad()
def run_chain(
    model: torch.nn.Module,
    schedule,
    vae: torch.nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    por: float,
    por_log_stats: tuple[float, float],
    n_steps: int,
    seed: int,
    step_pbar=None,
) -> list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]:
    """Run one DDIM chain and decode every intermediate latent.

    Returns a list of (xct_slices, pore_slices, label_slices) per denoising
    step.  Each *_slices is [z_slice, y_slice, x_slice] as float32 arrays.
    """
    from poregen.diffusion.conditioning import (
        NB_UNKNOWN, N_DIST6, N_NEIGHBOURS, porosity_to_cond,
    )
    from poregen.diffusion.orientation import orientation_tensor
    from poregen.diffusion.sampler import DDIMSampler, theta_from_layup
    from poregen.models.vae.base import decode_class_probs, decode_label, decode_xct

    sampler = DDIMSampler(model, schedule, device, n_steps=n_steps)
    C = model.cfg.z_channels
    L = LATENT_SIZE

    cond_por = torch.tensor([float(porosity_to_cond(por, por_log_stats))],
                            dtype=torch.float32, device=device)
    nb_latents = torch.zeros(1, N_NEIGHBOURS, C, L, L, L, device=device)
    nb_avail = torch.full((1, N_NEIGHBOURS), NB_UNKNOWN, dtype=torch.long, device=device)
    nb_t = torch.zeros((1, N_NEIGHBOURS), dtype=torch.long, device=device)
    # Mid-thickness interior patch: depth 0.5, every face more than 64 voxels
    # from a surface, all material, orientation straight from the ply sequence.
    cond_depth = torch.tensor([0.5], dtype=torch.float32, device=device)
    cond_dist6 = torch.ones(1, N_DIST6, dtype=torch.float32, device=device)
    cond_material = torch.ones(1, 1, L, L, L, dtype=torch.float32, device=device)
    cond_orient = torch.from_numpy(orientation_tensor(
        theta_from_layup(PATCH_SIZE, LAYUP_ANGLES_DEG, PLY_THICKNESS_VOX), L
    )).unsqueeze(0).to(device)

    torch.manual_seed(seed)
    _, intermediates = sampler.sample_batch(
        nb_latents, nb_avail, nb_t, cond_por, cond_depth, cond_dist6,
        cond_orient, cond_material,
        autocast_dtype=autocast_dtype, return_intermediates=True,
    )

    mid = PATCH_SIZE // 2
    frames = []
    for z_cpu in intermediates:
        z_batch = (z_cpu * latent_std + latent_mean).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            dec = vae.decoder(z_batch)
            xct_out = vae.xct_head(dec)
            class_logits = vae.class_head(dec)

        xct = decode_xct(xct_out.float()).squeeze().cpu().numpy()
        probs = decode_class_probs(class_logits.float())[0].cpu().numpy()
        pore = probs[1]
        label = decode_label(class_logits.float()).squeeze().float().cpu().numpy() / 2.0

        frames.append((
            [xct[mid], xct[:, mid], xct[:, :, mid]],
            [pore[mid], pore[:, mid], pore[:, :, mid]],
            [label[mid], label[:, mid], label[:, :, mid]],
        ))

        if step_pbar is not None:
            step_pbar.update(1)

    return frames


def write_video(
    frames: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]],
    out_path: Path,
    por: float,
    n_steps: int,
) -> None:
    """Render frames to an MP4 video with a 3×3 panel layout.

    Row 0 : XCT grey level        (gray)
    Row 1 : pore probability      (magma)
    Row 2 : argmax label / 2      (magma; 0 material, 0.5 pore, 1 air)
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=False)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.03,
                        wspace=0.06, hspace=0.10)

    col_titles = ["z = 32", "y = 32", "x = 32"]
    row_labels  = ["XCT", "p(pore)", "label\n/2"]
    cmaps       = [["gray"] * 3, ["magma"] * 3, ["magma"] * 3]

    im_handles = []
    blank = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for row in range(3):
        row_handles = []
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(blank, vmin=0, vmax=1, cmap=cmaps[row][col],
                           interpolation="nearest", aspect="equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=9, pad=3)
            if col == 0:
                ax.text(-0.14, 0.5, row_labels[row], color="white", fontsize=8,
                        transform=ax.transAxes, va="center", ha="right",
                        rotation=90, linespacing=1.4)
            row_handles.append(im)
        im_handles.append(row_handles)

    # Shared colorbar for both mask rows
    cbar_ax = fig.add_axes([0.90, 0.03, 0.022, 0.56])
    cbar = fig.colorbar(im_handles[1][0], cax=cbar_ax)
    cbar.set_label("pore probability / label", color="white", fontsize=8, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, quality=8, format="FFMPEG")

    try:
        for step_idx, (xct_slices, pore_slices, label_slices) in enumerate(frames):
            for col in range(3):
                im_handles[0][col].set_data(xct_slices[col])
                im_handles[1][col].set_data(pore_slices[col])
                im_handles[2][col].set_data(label_slices[col])
            fig.suptitle(
                f"por = {por:.3f}   DDIM   step {step_idx + 1} / {n_steps}",
                color="white", fontsize=11,
            )
            fig.canvas.draw()
            frame_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame_rgb)
    finally:
        writer.close()
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise DDIM denoising progression per step count.")
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--vae-run",     required=True)
    ap.add_argument("--latents-root", default="data/split_v3/latents_r08z4",
                    help="Latent store root (metadata.json supplies per-channel norm stats)")
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

    model, ldm_cfg = _load_ldm(args.checkpoint, device)
    vae = _load_vae(args.vae_run, device)
    latents_root = Path(args.latents_root)
    if not latents_root.is_absolute():
        latents_root = (repo / latents_root).resolve()
    latent_mean, latent_std = _load_latent_stats(latents_root)

    # Raw phi levels must be mapped through the store's porosity transform
    # before they can be used as cond_por.
    _store_meta = json.loads((latents_root / "metadata.json").read_text())
    _st = (_store_meta.get("conditioning") or {}).get("por_standardisation")
    if _st is None:
        raise RuntimeError(
            f"{latents_root}/metadata.json has no conditioning.por_standardisation — "
            "cannot build cond_por for the requested porosity levels."
        )
    por_log_stats = (float(_st["mean"]), float(_st["std"]))

    schedule = DDPMSchedule.from_cfg(ldm_cfg, device)

    out_root = Path(f"{Path(args.checkpoint).stem}_denoising")

    combinations = [(por, n) for por in POROSITY_LEVELS for n in DDIM_STEP_COUNTS]

    run_pbar  = tqdm(total=len(combinations), unit="run",  position=0)
    step_pbar = tqdm(unit="step", position=1, leave=False)

    with run_pbar, step_pbar:
        for por, n_steps in combinations:
            run_pbar.set_description(f"por={por:.3f}  DDIM  steps={n_steps:4d}")
            step_pbar.reset(total=n_steps)
            step_pbar.set_description("denoising")

            frames = run_chain(
                model, schedule, vae, device, autocast_dtype,
                latent_mean, latent_std, por, por_log_stats, n_steps, SEED,
                step_pbar=step_pbar,
            )

            out_path = out_root / f"por_{por}" / f"steps_{n_steps:04d}_ddim.mp4"
            write_video(frames, out_path, por, n_steps)

            run_pbar.update(1)

    print(f"Done. Videos in {out_root}/")


if __name__ == "__main__":
    main()

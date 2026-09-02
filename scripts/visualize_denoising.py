"""Visualise DDIM denoising progression for a single patch.

For each (porosity, n_steps) combination, runs an independent DDIM chain from a fixed
seed, decodes every intermediate latent through the VAE, and writes an MP4 showing the
middle z/y/x slices of XCT and mask evolving from noise to final output.

Usage
-----
python scripts/visualize_denoising.py \\
    --checkpoint runs/ldm/.../checkpoints/best.ckpt \\
    --vae-run    runs/vae/r05-run-... \\
    [--latent-std 0.4571]

Output
------
<checkpoint_stem>_denoising/
  por_0.005/
    steps_020.mp4  steps_050.mp4  steps_100.mp4  steps_200.mp4  steps_500.mp4
  por_0.01/
    ...
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
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
    sampler_type: str,
    seed: int,
    step_pbar=None,
) -> list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]:
    """Run one DDIM or DDPM chain and decode every intermediate latent.

    Returns a list of (xct_slices, mask_slices, mask_reenc_slices) per denoising step.
    Each *_slices is [z_slice, y_slice, x_slice] as float32 numpy arrays.
    mask_slices      : mask predicted directly from the LDM latent via vae.mask_head
    mask_reenc_slices: mask from re-encoding the generated XCT back through the full VAE
    """
    from poregen.models.vae.base import decode_xct
from poregen.diffusion.sampler import DDIMSampler, DDPMSampler

    if sampler_type == "ddpm":
        sampler = DDPMSampler(model, schedule, device)
    else:
        sampler = DDIMSampler(model, schedule, device, n_steps=n_steps)

    C = model.cfg.z_channels
    from poregen.diffusion.orientation import orientation_tensor
    from poregen.diffusion.sampler import porosity_to_cond, theta_from_layup

    cond_por = float(porosity_to_cond(por, por_log_stats))

    nb_latents = torch.zeros(6, C, LATENT_SIZE, LATENT_SIZE, LATENT_SIZE)
    nb_avail   = torch.zeros(6, dtype=torch.long)
    # Mid-thickness patch of the nominal layup (D32 §4): depth 0.5, far from
    # any outer surface, orientation written straight from the ply sequence.
    orient = torch.from_numpy(orientation_tensor(
        theta_from_layup(PATCH_SIZE, LAYUP_ANGLES_DEG, PLY_THICKNESS_VOX), LATENT_SIZE
    ))

    torch.manual_seed(seed)
    _, intermediates = sampler.sample_patch(
        nb_latents, nb_avail, cond_por, 0.5, 1.0, orient,
        autocast_dtype=autocast_dtype,
        return_intermediates=True,
    )

    mid = PATCH_SIZE // 2
    frames = []
    for z_cpu in intermediates:
        z_batch = (z_cpu.unsqueeze(0) * latent_std + latent_mean).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            dec        = vae.decoder(z_batch)
            xct_out = vae.xct_head(dec)
            msk_logits = vae.mask_head(dec)

        xct  = decode_xct(xct_out).squeeze().float().cpu().numpy()
        mask = torch.sigmoid(msk_logits).squeeze().float().cpu().numpy()

        # Re-encode the generated XCT through the full VAE to get a refined mask.
        # mask argument is unused by the encoder (XCT-only design), so zeros is fine.
        xct_t = torch.from_numpy(xct).unsqueeze(0).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            vae_out = vae(xct_t, torch.zeros_like(xct_t))
        mask_reenc = torch.sigmoid(vae_out.mask_logits).squeeze().float().cpu().numpy()

        xct_slices        = [xct[mid],        xct[:, mid],        xct[:, :, mid]]
        mask_slices       = [mask[mid],       mask[:, mid],       mask[:, :, mid]]
        mask_reenc_slices = [mask_reenc[mid], mask_reenc[:, mid], mask_reenc[:, :, mid]]
        frames.append((xct_slices, mask_slices, mask_reenc_slices))

        if step_pbar is not None:
            step_pbar.update(1)

    return frames


def write_video(
    frames: list[tuple[list[np.ndarray], list[np.ndarray]]],
    out_path: Path,
    por: float,
    n_steps: int,
    sampler_type: str = "ddim",
) -> None:
    """Render frames to an MP4 video with a 3×3 panel layout.

    Row 0 : XCT (gray)
    Row 1 : Mask from LDM latent           (magma)
    Row 2 : Mask re-encoded from XCT→VAE  (magma)
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=False)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.03,
                        wspace=0.06, hspace=0.10)

    col_titles = ["z = 32", "y = 32", "x = 32"]
    row_labels  = ["XCT", "Mask\n(LDM)", "Mask\n(reenc)"]
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
    cbar.set_label("pore probability", color="white", fontsize=8, labelpad=6)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, quality=8, format="FFMPEG")

    try:
        for step_idx, (xct_slices, mask_slices, mask_reenc_slices) in enumerate(frames):
            for col in range(3):
                im_handles[0][col].set_data(xct_slices[col])
                im_handles[1][col].set_data(mask_slices[col])
                im_handles[2][col].set_data(mask_reenc_slices[col])
            fig.suptitle(
                f"por = {por:.3f}   {sampler_type.upper()}   step {step_idx + 1} / {n_steps}",
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
    ap.add_argument("--latents-root", default="data/split_v2/latents_r07z4",
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
    # (D32 §1) before they can be used as cond_por.
    _store_meta = json.loads((latents_root / "metadata.json").read_text())
    _st = (_store_meta.get("conditioning") or {}).get("por_standardisation")
    if _st is None:
        raise RuntimeError(
            f"{latents_root}/metadata.json has no conditioning.por_standardisation — "
            "cannot build cond_por for the requested porosity levels."
        )
    por_log_stats = (float(_st["mean"]), float(_st["std"]))

    sched_cfg = ldm_cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(
        T=int(sched_cfg.get("T", 1000)),
        s=float(sched_cfg.get("s", 0.008)),
        device=device,
    )

    out_root = Path(f"{Path(args.checkpoint).stem}_denoising")

    # (por, n_steps, sampler_type)
    combinations = (
        [(por, n, "ddim") for por in POROSITY_LEVELS for n in DDIM_STEP_COUNTS]
        + [(por, schedule.T, "ddpm") for por in POROSITY_LEVELS]
    )

    run_pbar  = tqdm(total=len(combinations), unit="run",  position=0)
    step_pbar = tqdm(unit="step", position=1, leave=False)

    with run_pbar, step_pbar:
        for por, n_steps, sampler_type in combinations:
            run_pbar.set_description(
                f"por={por:.3f}  {sampler_type.upper()}  steps={n_steps:4d}"
            )
            step_pbar.reset(total=n_steps)
            step_pbar.set_description("denoising")

            frames = run_chain(
                model, schedule, vae, device, autocast_dtype,
                latent_mean, latent_std, por, por_log_stats, n_steps, sampler_type, SEED,
                step_pbar=step_pbar,
            )

            out_path = (
                out_root / f"por_{por}" / f"steps_{n_steps:04d}_{sampler_type}.mp4"
            )
            write_video(frames, out_path, por, n_steps, sampler_type)

            run_pbar.update(1)

    print(f"Done. Videos in {out_root}/")


if __name__ == "__main__":
    main()

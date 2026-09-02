"""Diagnostic: does the r07 decoder produce sane MASKS from off-manifold latents?

Encodes ~256 val patches, perturbs the posterior means with Gaussian noise at
several scales, decodes each latent set, and reports mask porosity statistics,
degenerate-mask fractions, and mushy-voxel fractions per set.

Outputs land in runs/diag/mask_sanity_r07z4/ (summary.json + one slice-grid
PNG per latent set).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from poregen.experiments.train_vae import build_model, resolve_data_root
from poregen.training.checkpoint import load_checkpoint
from poregen.training.data import build_patch_dataloaders

DEFAULT_CKPT = (
    "runs/vae/r07-run-0006-20260814-084936-archv2-conv_noattn_dualbranch"
    "-z4-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone/best.ckpt"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "runs/diag/mask_sanity_r07z4"
N_PATCHES = 256
N_SHOW = 8
DEGENERATE_LO = 1e-5
DEGENERATE_HI = 0.5
MUSHY_LO, MUSHY_HI = 0.2, 0.8


@torch.no_grad()
def encode_mu_std(model, xct, mask):
    """Posterior mean and std for a batch (dual-branch encoder path)."""
    enc_in = torch.cat([xct, mask], dim=1)
    h = model.fusion(torch.cat([model.encoder_a(enc_in), model.encoder_b(enc_in)], dim=1))
    mu = model.to_mu(h)
    std = torch.exp(0.5 * model.to_logvar(h))
    return mu, std


@torch.no_grad()
def decode(model, z):
    """Decode a latent batch to (xct, mask_prob)."""
    dec = model.decoder(z)
    return model.xct_head(dec), torch.sigmoid(model.mask_head(dec))


def mask_stats(mask_prob: torch.Tensor) -> dict:
    """Porosity / degeneracy / mushiness statistics for one decoded set."""
    binary = (mask_prob > 0.5).float()
    porosity = binary.mean(dim=(1, 2, 3, 4))  # (N,)
    degenerate = (porosity < DEGENERATE_LO) | (porosity > DEGENERATE_HI)
    mushy = ((mask_prob >= MUSHY_LO) & (mask_prob <= MUSHY_HI)).float().mean()
    return {
        "porosity_mean": porosity.mean().item(),
        "porosity_std": porosity.std().item(),
        "porosity_min": porosity.min().item(),
        "porosity_max": porosity.max().item(),
        "degenerate_fraction": degenerate.float().mean().item(),
        "mushy_voxel_fraction": mushy.item(),
    }


def save_grid(xct: torch.Tensor, mask_prob: torch.Tensor, title: str, path: Path):
    """Save a 2×N_SHOW grid of central z-slices (xct row, binarized mask row)."""
    zc = xct.shape[2] // 2
    n = min(N_SHOW, xct.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for i in range(n):
        axes[0, i].imshow(xct[i, 0, zc].clamp(0, 1).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow((mask_prob[i, 0, zc] > 0.5).float().cpu(), cmap="gray", vmin=0, vmax=1)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("xct")
    axes[1, 0].set_ylabel("mask")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = parser.parse_args()

    ckpt_path = (REPO_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    run_dir = ckpt_path.parent if (ckpt_path.parent / "resolved_config.yaml").exists() else ckpt_path.parent.parent
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = build_model(cfg, device)
    step, _ = load_checkpoint(ckpt_path, model, map_location=device, restore_rng=False)
    model.eval()
    print(f"Loaded {ckpt_path.name} at step {step}")

    # Data: val split, no workers needed for a one-shot diagnostic.
    cfg["data"]["num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["timeout"] = 0
    _, val_loader, _ = build_patch_dataloaders(cfg, resolve_data_root(cfg, REPO_ROOT))

    xcts, masks = [], []
    n = 0
    for batch in val_loader:
        xcts.append(batch["xct"])
        masks.append(batch["mask"])
        n += batch["xct"].shape[0]
        if n >= N_PATCHES:
            break
    xct = torch.cat(xcts)[:N_PATCHES].to(device)
    mask = torch.cat(masks)[:N_PATCHES].to(device)

    mu, post_std = encode_mu_std(model, xct, mask)
    # Per-channel posterior std, averaged over batch and space: (1, C, 1, 1, 1)
    ch_post_std = post_std.mean(dim=(0, 2, 3, 4), keepdim=True)
    # Empirical per-channel moments of the real latents (mu)
    ch_emp_std = mu.std(dim=(0, 2, 3, 4), keepdim=True)
    ch_emp_mean = mu.mean(dim=(0, 2, 3, 4), keepdim=True)

    gt_porosity = mask.mean(dim=(1, 2, 3, 4))

    latent_sets = {
        "real_mu": mu,
        "noise_0.5x_post_std": mu + 0.5 * ch_post_std * torch.randn_like(mu),
        "noise_1.0x_post_std": mu + 1.0 * ch_post_std * torch.randn_like(mu),
        "noise_2.0x_post_std": mu + 2.0 * ch_post_std * torch.randn_like(mu),
        "prior_N0_emp_std": ch_emp_std * torch.randn_like(mu),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "n_patches": int(xct.shape[0]),
        "per_channel_posterior_std": ch_post_std.flatten().tolist(),
        "per_channel_empirical_std_of_mu": ch_emp_std.flatten().tolist(),
        "per_channel_empirical_mean_of_mu": ch_emp_mean.flatten().tolist(),
        "ground_truth_val_porosity": {
            "mean": gt_porosity.mean().item(),
            "std": gt_porosity.std().item(),
            "min": gt_porosity.min().item(),
            "max": gt_porosity.max().item(),
        },
        "sets": {},
    }

    for name, z in latent_sets.items():
        dec_xct_chunks, dec_mask_chunks = [], []
        for i in range(0, z.shape[0], 64):
            dx, dm = decode(model, z[i : i + 64])
            dec_xct_chunks.append(dx)
            dec_mask_chunks.append(dm)
        dec_xct = torch.cat(dec_xct_chunks)
        dec_mask = torch.cat(dec_mask_chunks)

        stats = mask_stats(dec_mask)
        summary["sets"][name] = stats
        save_grid(dec_xct, dec_mask, name, OUT_DIR / f"grid_{name}.png")
        print(
            f"{name:24s} porosity {stats['porosity_mean']:.5f} ± {stats['porosity_std']:.5f}  "
            f"degenerate {stats['degenerate_fraction']:.3f}  mushy {stats['mushy_voxel_fraction']:.5f}"
        )

    save_grid(xct, mask, "ground_truth", OUT_DIR / "grid_ground_truth.png")
    print(
        f"{'ground_truth':24s} porosity {summary['ground_truth_val_porosity']['mean']:.5f} "
        f"± {summary['ground_truth_val_porosity']['std']:.5f}"
    )

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

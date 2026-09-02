"""Diagnostic: does the VAE decoder stay sane on off-manifold latents?

The LDM does not hand the decoder posterior means — it hands it samples that
drift away from the encoder's manifold. This script measures how far the
decoded segmentation moves when the latents are pushed off it: encode ~256 val
patches, perturb the posterior means with Gaussian noise at 0.5x / 1x / 2x the
per-channel posterior std (plus a draw from the empirical prior), decode each
set, and report how the segmentation statistics drift.

Works with either decoder head, because the checkpoint decides:

* **binary mask head** (r07 and earlier) — porosity, degenerate fraction,
  mushy-voxel fraction from ``sigmoid(mask_logits)``.
* **3-class head** (r08+) — the same, read off ``argmax(class_logits)``, plus
  the AIR-class fraction. Air is the class that should be near zero on interior
  patches, so air appearing under perturbation is the failure mode to watch.

Outputs land in ``runs/diagnostics/mask_sanity/<run name>/`` — summary.json and
one slice-grid PNG per latent set.

Usage:
    python scripts/diag_mask_sanity.py                       # the r07 default
    python scripts/diag_mask_sanity.py --checkpoint runs/vae/r08-.../best.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import yaml

from poregen.experiments.train_vae import build_model, resolve_data_root
from poregen.models.vae.base import (
    CLASS_AIR, CLASS_MATERIAL, CLASS_PORE, decode_class_probs, decode_label,
    decode_xct,
)
from poregen.training.checkpoint import load_checkpoint
from poregen.training.engine import encoder_input_keys
from poregen.training.data import build_patch_dataloaders

DEFAULT_CKPT = (
    "runs/vae/r07-run-0006-20260814-084936-archv2-conv_noattn_dualbranch"
    "-z4-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone/best.ckpt"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "runs/diagnostics/mask_sanity"
N_PATCHES = 256
N_SHOW = 8
DEGENERATE_LO = 1e-5
DEGENERATE_HI = 0.5
MUSHY_LO, MUSHY_HI = 0.2, 0.8

# material grey, pore orange, air blue — the same three-way palette everywhere.
LABEL_CMAP = mcolors.ListedColormap(["#3a3a3a", "#c2571a", "#1b6ca8"])


@torch.no_grad()
def decode_segmentation(model, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Decode a latent batch. Returns (xct grey in [0,1], segmentation dict).

    The segmentation dict carries ``label`` (int64 class index) and
    ``confidence`` (the winning class probability per voxel), whichever head
    the model has, so every statistic below is written once.
    """
    dec = model.decoder(z)
    xct = decode_xct(model.xct_head(dec))
    if getattr(model, "class_head", None) is not None:
        logits = model.class_head(dec)
        probs = decode_class_probs(logits)
        return xct, {"label": decode_label(logits),
                     "confidence": probs.max(dim=1).values,
                     "n_classes": 3}
    prob = torch.sigmoid(model.mask_head(dec)).squeeze(1)
    return xct, {"label": (prob > 0.5).long(),
                 "confidence": torch.maximum(prob, 1.0 - prob),
                 "n_classes": 2}


def seg_stats(seg: dict) -> dict:
    """Porosity / air / degeneracy / uncertainty for one decoded set."""
    label = seg["label"]
    por = (label == CLASS_PORE).flatten(1).float().mean(1)     # (N,)
    degenerate = (por < DEGENERATE_LO) | (por > DEGENERATE_HI)
    # "Mushy" = the winning class is barely winning. For a binary head this is
    # the old sigmoid-in-[0.2, 0.8] rule written in terms of confidence.
    mushy = (seg["confidence"] <= MUSHY_HI).float().mean()
    out = {
        "porosity_mean": por.mean().item(),
        "porosity_std": por.std().item(),
        "porosity_min": por.min().item(),
        "porosity_max": por.max().item(),
        "degenerate_fraction": degenerate.float().mean().item(),
        "mushy_voxel_fraction": mushy.item(),
    }
    if seg["n_classes"] == 3:
        air = (label == CLASS_AIR).flatten(1).float().mean(1)
        out.update({
            "air_mean": air.mean().item(),
            "air_std": air.std().item(),
            "air_max": air.max().item(),
            "material_mean": (label == CLASS_MATERIAL).flatten(1).float().mean(1).mean().item(),
        })
    return out


def save_grid(xct: torch.Tensor, label: torch.Tensor, n_classes: int,
              title: str, path: Path) -> None:
    """2 x N_SHOW grid of central z-slices: XCT row, label row."""
    zc = xct.shape[2] // 2
    n = min(N_SHOW, xct.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for i in range(n):
        axes[0, i].imshow(xct[i, 0, zc].clamp(0, 1).cpu(), cmap="gray",
                          vmin=0, vmax=1)
        axes[1, i].imshow(label[i, zc].cpu(), cmap=LABEL_CMAP,
                          vmin=0, vmax=2, interpolation="nearest")
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("xct")
    axes[1, 0].set_ylabel("label" if n_classes == 3 else "mask")
    fig.suptitle(title + ("   [grey material / orange pore / blue air]"
                          if n_classes == 3 else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    run_dir = (ckpt_path.parent
               if (ckpt_path.parent / "resolved_config.yaml").exists()
               else ckpt_path.parent.parent)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = build_model(cfg, device)
    step, _ = load_checkpoint(ckpt_path, model, map_location=device,
                             restore_rng=False)
    model.eval()
    keys = encoder_input_keys(model)
    print(f"Loaded {ckpt_path.name} at step {step}; encoder inputs {keys}")

    out_dir = OUT_ROOT / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data: val split, no workers needed for a one-shot diagnostic.
    cfg["data"]["num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["timeout"] = 0
    _, val_loader, _ = build_patch_dataloaders(cfg, resolve_data_root(cfg, REPO_ROOT))

    collected: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    gt_label_chunks: list[torch.Tensor] = []
    n = 0
    for batch in val_loader:
        for k in keys:
            collected[k].append(batch[k])
        gt_label_chunks.append(batch["label"])
        n += batch["xct"].shape[0]
        if n >= N_PATCHES:
            break
    inputs = tuple(torch.cat(collected[k])[:N_PATCHES].to(device) for k in keys)
    gt_label = torch.cat(gt_label_chunks)[:N_PATCHES].to(device)
    xct = inputs[0]

    with torch.no_grad():
        mu, logvar = model.encode_moments(*inputs)
    post_std = torch.exp(0.5 * logvar)

    # Per-channel posterior std, averaged over batch and space: (1, C, 1, 1, 1)
    ch_post_std = post_std.mean(dim=(0, 2, 3, 4), keepdim=True)
    ch_emp_std = mu.std(dim=(0, 2, 3, 4), keepdim=True)
    ch_emp_mean = mu.mean(dim=(0, 2, 3, 4), keepdim=True)

    gt_por = (gt_label == CLASS_PORE).flatten(1).float().mean(1)
    gt_air = (gt_label == CLASS_AIR).flatten(1).float().mean(1)

    latent_sets = {
        "real_mu": mu,
        "noise_0.5x_post_std": mu + 0.5 * ch_post_std * torch.randn_like(mu),
        "noise_1.0x_post_std": mu + 1.0 * ch_post_std * torch.randn_like(mu),
        "noise_2.0x_post_std": mu + 2.0 * ch_post_std * torch.randn_like(mu),
        "prior_N0_emp_std": ch_emp_std * torch.randn_like(mu),
    }

    summary = {
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "step": step,
        "model": cfg["model"]["name"],
        "encoder_inputs": list(keys),
        "n_patches": int(xct.shape[0]),
        "dataset_root": cfg["data"]["dataset_root"],
        "per_channel_posterior_std": ch_post_std.flatten().tolist(),
        "per_channel_empirical_std_of_mu": ch_emp_std.flatten().tolist(),
        "per_channel_empirical_mean_of_mu": ch_emp_mean.flatten().tolist(),
        "ground_truth_val": {
            "porosity_mean": gt_por.mean().item(),
            "porosity_std": gt_por.std().item(),
            "air_mean": gt_air.mean().item(),
            "air_std": gt_air.std().item(),
        },
        "sets": {},
    }

    for name, z in latent_sets.items():
        xct_chunks, label_chunks, stat_seg = [], [], None
        conf_chunks = []
        for i in range(0, z.shape[0], 64):
            dx, seg = decode_segmentation(model, z[i:i + 64])
            xct_chunks.append(dx)
            label_chunks.append(seg["label"])
            conf_chunks.append(seg["confidence"])
            stat_seg = seg
        merged = {"label": torch.cat(label_chunks),
                  "confidence": torch.cat(conf_chunks),
                  "n_classes": stat_seg["n_classes"]}
        stats = seg_stats(merged)
        summary["sets"][name] = stats
        save_grid(torch.cat(xct_chunks), merged["label"], merged["n_classes"],
                  name, out_dir / f"grid_{name}.png")
        air = (f"  air {stats['air_mean']:.5f}" if "air_mean" in stats else "")
        print(f"{name:24s} porosity {stats['porosity_mean']:.5f} "
              f"± {stats['porosity_std']:.5f}  "
              f"degenerate {stats['degenerate_fraction']:.3f}  "
              f"mushy {stats['mushy_voxel_fraction']:.5f}{air}")

    # Drift relative to the unperturbed decode — the number the gate reads.
    base = summary["sets"]["real_mu"]
    for name, s in summary["sets"].items():
        s["porosity_drift_vs_real_mu"] = (
            abs(s["porosity_mean"] - base["porosity_mean"])
            / max(base["porosity_mean"], 1e-12))
        if "air_mean" in s:
            s["air_drift_vs_real_mu"] = (
                abs(s["air_mean"] - base["air_mean"])
                / max(base["air_mean"], 1e-12))

    save_grid(xct, gt_label, 3, "ground_truth", out_dir / "grid_ground_truth.png")
    print(f"{'ground_truth':24s} porosity "
          f"{summary['ground_truth_val']['porosity_mean']:.5f} "
          f"± {summary['ground_truth_val']['porosity_std']:.5f}  "
          f"air {summary['ground_truth_val']['air_mean']:.5f}")
    print("porosity drift at 2x posterior std: "
          f"{summary['sets']['noise_2.0x_post_std']['porosity_drift_vs_real_mu']:.1%}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

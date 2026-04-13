"""Standalone diagnostic for reconstruction visualization issues.

Investigates 4 hypotheses:
  H1: Central D-axis slice is the wrong plane (orientation issue)
  H2: Poor model convergence (logits far from GT range)
  H3: Z-score stats computed on foreground only (background voxels → -4.0)
  H4: volume_stats.json has wrong/missing entries (silent /255 fallback)

Usage
-----
::

    # Without checkpoint (H1, H3, H4 only):
    python scripts/debug_reconstruction.py

    # With checkpoint (all hypotheses including H2):
    python scripts/debug_reconstruction.py --checkpoint runs/vae/.../last.ckpt \\
        --config src/poregen/configs/vae_default.yaml

Output: debug_reconstruction_report/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_ROOT = Path("data/split_v1")
OUT_DIR   = Path("debug_reconstruction_report")


def _banner(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _plot_patch_grid(patches: list[np.ndarray], title: str, out_path: Path) -> None:
    """Save a grid of N patch slices (D/H/W central slices) as a PNG."""
    n = len(patches)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing
    fig.suptitle(title, fontsize=9)
    for row, patch in enumerate(patches):
        D, H, W = patch.shape
        for col, (sl, ax_label) in enumerate([
            (patch[D // 2],       "D-axis (Z)"),
            (patch[:, H // 2, :], "H-axis (Y)"),
            (patch[:, :, W // 2], "W-axis (X)"),
        ]):
            axes[row, col].imshow(sl, cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(f"{ax_label}  shape={sl.shape}")
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def section1_patch_check(report_lines: list[str]) -> None:
    """H1: Sample real training patches from zarr+index and show all 3 orthogonal slices.

    This verifies:
    - The patch coordinates from the index map to sensible data
    - The axis orientation (which D/H/W slice looks meaningful)
    - The raw uint8 intensity range and distribution
    """
    report_lines.append(_banner("H1: Patch Check — raw zarr patches as seen during training"))

    zarr_path  = DATA_ROOT / "volumes.zarr"
    index_path = DATA_ROOT / "patch_index.parquet"

    if not zarr_path.exists():
        report_lines.append(f"  SKIP: {zarr_path} not found")
        return
    if not index_path.exists():
        report_lines.append(f"  SKIP: {index_path} not found")
        return

    root  = zarr.open_group(str(zarr_path), mode="r")
    df    = pd.read_parquet(str(index_path))
    n_total = len(df)
    report_lines.append(f"  Patch index: {n_total:,} patches total")

    # Sample 5 patches from each split for visual inspection
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        if split_df.empty:
            report_lines.append(f"\n  {split}: no patches")
            continue

        sample = split_df.sample(min(5, len(split_df)), random_state=0)
        report_lines.append(f"\n  {split}: {len(split_df):,} patches — sampling {len(sample)}")

        patches_raw: list[np.ndarray] = []
        titles: list[str] = []
        for _, row in sample.iterrows():
            vid = row["volume_id"]
            ps  = int(row["ps"])
            z0, y0, x0 = int(row["z0"]), int(row["y0"]), int(row["x0"])

            if vid not in root:
                report_lines.append(f"    MISSING in zarr: {vid}")
                continue

            patch = np.array(root[vid]["xct"][z0:z0+ps, y0:y0+ps, x0:x0+ps])
            patch_norm = patch.astype(np.float32) / 255.0

            patches_raw.append(patch)
            por = float(row.get("porosity", float("nan")))
            titles.append(f"{vid[:30]} z0={z0} y0={y0} x0={x0} por={por:.3f}")

            report_lines.append(
                f"    patch shape={patch.shape}  "
                f"min={patch.min():3d}  max={patch.max():3d}  mean={patch.mean():.1f}  "
                f"normalized=[{patch_norm.min():.3f}, {patch_norm.max():.3f}]  "
                f"porosity={por:.3f}"
            )

        if patches_raw:
            safe_split = split.replace("/", "_")
            out_path = OUT_DIR / f"h1_patches_{safe_split}.png"
            _plot_patch_grid(
                patches_raw,
                f"{split} split — raw uint8 patches (D/H/W central slices)",
                out_path,
            )
            report_lines.append(f"    Saved: {out_path}")

    report_lines.append("\n  ACTION: Open h1_patches_*.png. For each patch, check which")
    report_lines.append("  column (D/H/W) shows the most meaningful pore cross-section.")


def section2_stats_coverage(report_lines: list[str]) -> None:
    """H4: Check volume_stats.json for missing/anomalous entries."""
    report_lines.append(_banner("H4: Volume Stats Coverage"))

    stats_path = DATA_ROOT / "volume_stats.json"
    if not stats_path.exists():
        report_lines.append(f"  FAIL: {stats_path} not found — all patches will use /255 fallback")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    report_lines.append(f"  Stats file: {stats_path}")
    report_lines.append(f"  Number of entries: {len(stats)}")

    if not stats:
        report_lines.append("  FAIL: stats file is empty")
        return

    df = pd.DataFrame(stats).T[["mean", "std"]].astype(float)

    report_lines.append(f"\n  Summary:\n{df.describe().to_string()}")

    # Flag fallback values (mean=128, std=50 are the hardcoded defaults)
    fallbacks = df[(df["mean"].between(127, 129)) & (df["std"].between(49, 51))]
    if not fallbacks.empty:
        report_lines.append(f"\n  WARNING: {len(fallbacks)} volume(s) have fallback stats (mean≈128, std≈50):")
        for vid in fallbacks.index:
            report_lines.append(f"    {vid}  mean={df.loc[vid,'mean']:.2f}  std={df.loc[vid,'std']:.2f}")
    else:
        report_lines.append("\n  OK: No fallback stats detected")

    # Flag tiny std (would make z-score explode)
    tiny_std = df[df["std"] < 1.0]
    if not tiny_std.empty:
        report_lines.append(f"\n  WARNING: {len(tiny_std)} volume(s) with std < 1.0:")
        for vid in tiny_std.index:
            report_lines.append(f"    {vid}  std={df.loc[vid,'std']:.4f}")
    else:
        report_lines.append("  OK: All std values >= 1.0")

    # Check against patch index
    index_path = DATA_ROOT / "patch_index.parquet"
    if index_path.exists():
        idx_df = pd.read_parquet(str(index_path))
        patch_vids = set(idx_df["volume_id"].unique())
        stats_vids = set(stats.keys())
        missing = patch_vids - stats_vids
        if missing:
            report_lines.append(f"\n  FAIL: {len(missing)} volume(s) in patch index have NO stats entry:")
            for vid in sorted(missing):
                report_lines.append(f"    {vid}  ← will use /255 fallback silently!")
        else:
            report_lines.append(f"\n  OK: All {len(patch_vids)} patch-index volumes have stats entries")


def section3_dataloader_range(report_lines: list[str]) -> None:
    """Check DataLoader output range — expect [0, 1] with /255 normalisation."""
    report_lines.append(_banner("DataLoader Output Range"))

    index_path = DATA_ROOT / "patch_index.parquet"
    if not index_path.exists():
        report_lines.append(f"  SKIP: {index_path} not found")
        return

    try:
        from poregen.dataset.loader import PatchDataset
    except ImportError as e:
        report_lines.append(f"  SKIP: cannot import PatchDataset: {e}")
        return

    try:
        ds = PatchDataset(index_path, DATA_ROOT, split="val")
    except Exception as e:
        report_lines.append(f"  SKIP: failed to build PatchDataset: {e}")
        return

    report_lines.append(f"  Dataset size (val split): {len(ds)} patches")
    report_lines.append(f"  Sampling 20 patches (expect xct in [0, 1]):\n")

    mins, maxs, means = [], [], []

    for i in range(min(20, len(ds))):
        s = ds[i]
        xct = s["xct"]
        mn, mx, mu = xct.min().item(), xct.max().item(), xct.mean().item()
        mins.append(mn); maxs.append(mx); means.append(mu)
        flag = ""
        if mn < -0.01 or mx > 1.01:
            flag = " ← OUT OF [0,1] RANGE — normalisation issue!"
        report_lines.append(
            f"  [{i:02d}] {s['volume_id'][:35]:35s}  "
            f"min={mn:+.3f}  max={mx:+.3f}  mean={mu:+.3f}{flag}"
        )

    report_lines.append(f"\n  Aggregate over {len(mins)} patches:")
    report_lines.append(f"    min(min)={min(mins):+.3f}  max(max)={max(maxs):+.3f}  mean(mean)={np.mean(means):+.3f}")

    global_min = min(mins)
    global_max = max(maxs)
    if global_min >= 0.0 and global_max <= 1.01:
        report_lines.append("  STATUS: OK — /255 normalisation confirmed, all values in [0, 1]")
    else:
        report_lines.append(f"  STATUS: FAIL — unexpected range [{global_min:.3f}, {global_max:.3f}]")


def _build_model(checkpoint: str | None, config: str | None, device, report_lines: list[str]):
    """Return (model, label) — trained checkpoint if provided, else randomly initialised."""
    try:
        import torch
        from poregen.models.vae import build_vae
        from poregen.models.vae.base import VAEConfig
        from poregen.configs.config import load_config
    except ImportError as e:
        report_lines.append(f"  SKIP: import error: {e}")
        return None, None

    # Always build the model from config (or default config)
    cfg_path = config or "src/poregen/configs/vae_default.yaml"
    try:
        cfg = load_config(cfg_path)
        model_cfg = VAEConfig(
            z_channels=cfg["model"].get("z_channels", 8),
            base_channels=cfg["model"].get("base_channels", 32),
            n_blocks=cfg["model"].get("n_blocks", 2),
            patch_size=cfg["model"].get("patch_size", 64),
        )
        model = build_vae(cfg["model"].get("name", "conv"), **{
            k: v for k, v in vars(model_cfg).items()
        })
    except Exception as e:
        report_lines.append(f"  WARNING: could not load config ({e}), using default VAEConfig")
        model_cfg = VAEConfig()
        model = build_vae("conv")

    if checkpoint is not None:
        try:
            from poregen.training.checkpoint import load_checkpoint
            load_checkpoint(checkpoint, model=model, restore_rng=False, map_location=device)
            label = f"trained checkpoint: {checkpoint}"
        except Exception as e:
            report_lines.append(f"  WARNING: failed to load checkpoint ({e}) — falling back to random init")
            label = "randomly initialised (checkpoint load failed)"
    else:
        label = "randomly initialised (no checkpoint provided)"

    model = model.to(device).eval()
    return model, label


def section4_logit_range(report_lines: list[str], checkpoint: str | None, config: str | None) -> None:
    """H2: Run model (random init or checkpoint) and compare logit range vs GT range.

    A randomly initialised model lets us verify:
    - The data pipeline produces valid inputs (shape, dtype, range)
    - The model forward pass runs without errors
    - The output logit range is plausible (random init should give ~N(0,1) logits)

    A trained checkpoint additionally shows whether reconstruction is converging.
    """
    report_lines.append(_banner("H2: Model Logit Range vs GT Range"))

    try:
        import torch
        from torch.utils.data import DataLoader
        from poregen.dataset.loader import PatchDataset
    except ImportError as e:
        report_lines.append(f"  SKIP: import error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_lines.append(f"  Device: {device}")

    model, label = _build_model(checkpoint, config, device, report_lines)
    if model is None:
        return
    report_lines.append(f"  Model: {label}")

    index_path = DATA_ROOT / "patch_index.parquet"
    try:
        ds     = PatchDataset(index_path, DATA_ROOT, split="val")
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    except Exception as e:
        report_lines.append(f"  SKIP: DataLoader error: {e}")
        return

    gt_mins, gt_maxs, gt_means = [], [], []
    recon_mins, recon_maxs, recon_means = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= 5:
                break
            xct    = batch["xct"].to(device)
            mask   = batch["mask"].to(device)
            out    = model(xct, mask)
            logits = out.xct_logits

            gt_mins.append(xct.min().item());       gt_maxs.append(xct.max().item());       gt_means.append(xct.mean().item())
            recon_mins.append(logits.min().item()); recon_maxs.append(logits.max().item()); recon_means.append(logits.mean().item())

            report_lines.append(
                f"  Batch {step}: GT=[{xct.min():+.3f}, {xct.max():+.3f}] mean={xct.mean():+.3f} | "
                f"Logit=[{logits.min():+.3f}, {logits.max():+.3f}] mean={logits.mean():+.3f}"
            )

    report_lines.append(f"\n  GT    aggregate: min={min(gt_mins):+.3f}  max={max(gt_maxs):+.3f}  mean={np.mean(gt_means):+.3f}")
    report_lines.append(f"  Recon aggregate: min={min(recon_mins):+.3f}  max={max(recon_maxs):+.3f}  mean={np.mean(recon_means):+.3f}")

    # For random init: logits should be roughly zero-centred with moderate range
    # For trained model: logits should be in ~[0, 1] matching GT
    if checkpoint is None:
        report_lines.append(
            "  NOTE: random init — logits reflect untrained decoder weights. "
            "Expected roughly N(0, σ) depending on initialisation. "
            "This confirms the forward pass works and input shapes are correct."
        )
    else:
        gt_range    = max(gt_maxs)   - min(gt_mins)
        recon_range = max(recon_maxs) - min(recon_mins)
        if recon_range < gt_range * 0.1:
            report_lines.append("  WARNING: Recon logit range is very narrow vs GT — model may have collapsed")
        elif recon_range > gt_range * 5:
            report_lines.append("  WARNING: Recon logit range is much wider than GT — model may be diverging")
        else:
            report_lines.append("  OK: Recon logit range is plausible relative to GT range")

    _save_slice_comparison(model, loader, device, checkpoint, report_lines)


def _save_slice_comparison(model, loader, device, checkpoint: str | None, report_lines: list[str]) -> None:
    """Save D/H/W side-by-side GT vs reconstruction slices for one batch."""
    import torch

    try:
        batch = next(iter(loader))
    except Exception:
        return

    with torch.no_grad():
        xct    = batch["xct"].to(device)   # (B,1,D,H,W) in [0,1]
        mask   = batch["mask"].to(device)
        out    = model(xct, mask)
        logits = out.xct_logits.clamp(0.0, 1.0)  # (B,1,D,H,W) clamped to [0,1] for display

    label_suffix = "trained" if checkpoint else "random_init"

    B = xct.shape[0]
    for b in range(min(B, 2)):
        vid       = batch["volume_id"][b]
        gt_vol    = xct[b, 0].cpu().numpy()     # (D,H,W) in [0,1]
        recon_vol = logits[b, 0].cpu().numpy()  # (D,H,W) in [0,1]

        D, H, W = gt_vol.shape
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"GT (top) vs Reconstruction [{label_suffix}] (bottom)\n{vid[:60]}",
            fontsize=9,
        )

        for row, vol, label in [(0, gt_vol, "GT"), (1, recon_vol, f"Recon [{label_suffix}]")]:
            axes[row, 0].imshow(vol[D // 2],       cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{label} D-central slice")
            axes[row, 1].imshow(vol[:, H // 2, :], cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title(f"{label} H-central slice")
            axes[row, 2].imshow(vol[:, :, W // 2], cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title(f"{label} W-central slice")

        plt.tight_layout()
        safe_id  = vid[:40].replace("/", "_").replace(" ", "_")
        out_path = OUT_DIR / f"h2_{label_suffix}_patch{b}_{safe_id}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        report_lines.append(f"  Saved: {out_path}")


def section5_metric_sanity(report_lines: list[str]) -> None:
    """Bug A: Show how sigmoid(z_score) vs z_score causes bad MAE/PSNR."""
    report_lines.append(_banner("Bug A: Metric Sanity Check (sigmoid in z-score space)"))

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        report_lines.append("  SKIP: torch not available")
        return

    # Simulate: GT is z-scored in [-4, 4], pred is also z-scored (good reconstruction)
    torch.manual_seed(0)
    gt     = torch.randn(1000) * 1.5        # simulated z-scored GT, std≈1.5
    pred   = gt + torch.randn(1000) * 0.2   # good reconstruction (small noise)

    mae_correct = F.l1_loss(pred, gt).item()
    mae_buggy   = F.l1_loss(torch.sigmoid(pred), gt).item()

    report_lines.append(f"  Simulated z-scored GT (std=1.5) + good reconstruction (noise std=0.2):")
    report_lines.append(f"    Correct MAE (pred vs gt in z-score space): {mae_correct:.4f}  ← meaningful")
    report_lines.append(f"    Buggy   MAE (sigmoid(pred) vs gt):          {mae_buggy:.4f}  ← always ≈3-4, garbage")

    if mae_buggy > mae_correct * 5:
        report_lines.append("  CONFIRMED BUG: sigmoid-in-z-score-space inflates MAE by "
                            f"{mae_buggy/mae_correct:.1f}×, masking reconstruction quality")
    else:
        report_lines.append("  NOTE: MAE inflation smaller than expected in this simulation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug reconstruction visualization issues")
    parser.add_argument("--checkpoint", default=None, help="Path to .ckpt file (enables H2 check)")
    parser.add_argument("--config",     default=None, help="Path to YAML config (needed with --checkpoint)")
    parser.add_argument("--data_root",  default="data/split_v1", help="Path to dataset root")
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = [
        "RECONSTRUCTION DIAGNOSTIC REPORT",
        f"Data root: {DATA_ROOT}",
        f"Output dir: {OUT_DIR}",
        "",
    ]

    section1_patch_check(report_lines)
    section2_stats_coverage(report_lines)
    section3_dataloader_range(report_lines)
    section4_logit_range(report_lines, args.checkpoint, args.config)
    section5_metric_sanity(report_lines)

    report_lines.append(_banner("SUMMARY"))
    report_lines.append("  Check H1 PNGs: h1_patches_*.png — do patches look correct in each axis?")
    report_lines.append("  Check DataLoader range above: xct should be in [0, 1]")
    report_lines.append("  Check H2 (if checkpoint provided): h2_gt_vs_recon_*.png")
    report_lines.append("  Bug A is confirmed by the metric sanity section above.\n")

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "report.txt"
    report_path.write_text(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")
    print(f"Images saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()

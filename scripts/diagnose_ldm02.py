"""Four-check diagnostic for ldm02.

Checks
------
1. Confirm latent_std was recomputed on sampled z (μ+σε), not carried from ldm01.
2. Plot ε-MSE vs diffusion timestep t (requires --checkpoint).
3. Report σ = exp(0.5·logvar) distribution from the packed store.
4. Print a reminder that ldm02 success is downstream generation, not ε-MSE.

Usage
-----
# Checks 1, 3, 4 only (no model needed):
python scripts/diagnose_ldm02.py

# All four checks (checkpoint required for loss-vs-t plot):
python scripts/diagnose_ldm02.py \\
    --checkpoint runs/ldm/ldm02-run-0001-.../checkpoints/best.ckpt \\
    --n-batches 200 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
    return here.parent


REPO = _repo_root()
sys.path.insert(0, str(REPO / "src"))

LDMS64_STATS  = REPO / "data/split_v2/latents_s64/latent_scale_stats.json"
LDMS64S_STATS = REPO / "data/split_v2/latents_s64_sampled/latent_scale_stats.json"
LDMS64S_ROOT  = REPO / "data/split_v2/latents_s64_sampled"


# ── Check 1 ───────────────────────────────────────────────────────────────────

def check1_latent_std() -> None:
    print("\n" + "=" * 60)
    print("CHECK 1 — latent_std origin")
    print("=" * 60)

    ok1 = LDMS64_STATS.exists()
    ok2 = LDMS64S_STATS.exists()

    if ok1:
        with open(LDMS64_STATS) as f:
            s1 = json.load(f)
        print(f"  ldm01 (μ only)   std = {s1['std']:.4f}")
    else:
        print(f"  ldm01 stats file NOT FOUND at {LDMS64_STATS}")

    if ok2:
        with open(LDMS64S_STATS) as f:
            s2 = json.load(f)
        print(f"  ldm02 (μ+σε)     std = {s2['std']:.4f}  mean = {s2['mean']:.4f}")
    else:
        print(f"  ldm02 stats file NOT FOUND at {LDMS64S_STATS}")
        return

    if ok1 and ok2:
        diff = abs(s1["std"] - s2["std"])
        if diff < 0.02:
            print(
                f"\n  WARNING: stds differ by only {diff:.4f}. "
                "Likely ldm02 stats were NOT recomputed — they look identical to ldm01."
            )
        else:
            print(
                f"\n  PASS: std changed from {s1['std']:.4f} → {s2['std']:.4f}. "
                "Stats were recomputed on sampled z."
            )
            if s2["std"] > 0.9:
                print(
                    f"  INFO: std≈{s2['std']:.3f} ≈ 1.0 — expected for z=μ+σε when "
                    "KL has squeezed posterior toward N(0,1)."
                )


# ── Check 3 ───────────────────────────────────────────────────────────────────

def check3_sigma_sanity(n_samples: int = 5000) -> None:
    print("\n" + "=" * 60)
    print("CHECK 3 — σ = exp(0.5·logvar) distribution")
    print("=" * 60)

    meta_path = LDMS64S_ROOT / "latents_meta.json"
    bin_path  = LDMS64S_ROOT / "latents.bin"
    parq_path = LDMS64S_ROOT / "latents_index.parquet"

    if not (bin_path.exists() and meta_path.exists()):
        print(f"  SKIP: latents.bin / latents_meta.json not found in {LDMS64S_ROOT}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    n_total   = meta["N"]
    n_chan     = meta["n_channels"]
    spatial   = meta["spatial"]
    z_ch      = meta["z_channels"]

    mmap = np.memmap(
        str(bin_path), dtype="float16", mode="r",
        shape=(n_total, n_chan, *spatial),
    )

    import pandas as pd
    df = pd.read_parquet(str(parq_path))
    train_idxs = df.index[df["split"] == "train"].to_numpy()

    rng = np.random.default_rng(seed=0)
    sel = rng.choice(train_idxs, min(n_samples, len(train_idxs)), replace=False)

    chunk = 256
    all_logvar: list[np.ndarray] = []
    for start in range(0, len(sel), chunk):
        batch = np.asarray(mmap[sel[start:start + chunk]], dtype=np.float32)
        logvar_batch = batch[:, z_ch:]          # (B, C, D, H, W)
        all_logvar.append(logvar_batch.reshape(-1))

    logvar_flat = np.concatenate(all_logvar)
    sigma_flat  = np.exp(0.5 * logvar_flat)

    print(f"  logvar  — min={logvar_flat.min():.3f}  max={logvar_flat.max():.3f}  "
          f"mean={logvar_flat.mean():.3f}  std={logvar_flat.std():.3f}")
    print(f"  σ       — min={sigma_flat.min():.4f}  max={sigma_flat.max():.4f}  "
          f"mean={sigma_flat.mean():.4f}  std={sigma_flat.std():.4f}")
    print(f"  σ pct   — p5={np.percentile(sigma_flat, 5):.4f}  "
          f"p50={np.percentile(sigma_flat, 50):.4f}  "
          f"p95={np.percentile(sigma_flat, 95):.4f}  "
          f"p99={np.percentile(sigma_flat, 99):.4f}")

    mean_sigma = sigma_flat.mean()
    if mean_sigma > 2.0:
        print(
            f"\n  WARNING: mean σ={mean_sigma:.3f} >> 1 — posterior is much wider than N(0,1). "
            "Check KL weight / training length."
        )
    elif mean_sigma < 0.3:
        print(
            f"\n  WARNING: mean σ={mean_sigma:.4f} << 1 — posterior collapsed. "
            "Latents are near-deterministic; ldm02 ≈ ldm01 in this regime."
        )
    else:
        print(f"\n  PASS: mean σ={mean_sigma:.4f} is reasonable (expect ~0.3–1.2 for healthy VAE).")

    # Per-channel breakdown (mean logvar per channel)
    all_logvar_ch: list[np.ndarray] = []
    for start in range(0, len(sel), chunk):
        batch = np.asarray(mmap[sel[start:start + chunk]], dtype=np.float32)
        lv_b  = batch[:, z_ch:]
        all_logvar_ch.append(lv_b.reshape(lv_b.shape[0] * lv_b.shape[1], -1).mean(-1).reshape(-1, z_ch).mean(0))

    ch_mean_logvar = np.stack(all_logvar_ch).mean(0)
    ch_sigma       = np.exp(0.5 * ch_mean_logvar)
    print(f"\n  Per-channel mean σ (first 8 / {z_ch}):")
    print("  " + "  ".join(f"ch{i}:{ch_sigma[i]:.3f}" for i in range(min(8, z_ch))))
    if z_ch > 8:
        print("  " + "  ".join(f"ch{i}:{ch_sigma[i]:.3f}" for i in range(8, z_ch)))


# ── Check 2 ───────────────────────────────────────────────────────────────────

def check2_loss_vs_t(checkpoint: str, n_batches: int, device_str: str) -> None:
    print("\n" + "=" * 60)
    print("CHECK 2 — ε-MSE vs diffusion timestep t")
    print("=" * 60)

    from poregen.configuration import resolve_experiment
    from poregen.diffusion.noise_schedule import DDPMSchedule
    from poregen.diffusion.sampled_latent_dataset import SampledLatentPatchDataset
    from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser
    from poregen.training.checkpoint import load_checkpoint

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16

    # ── Load experiment config from checkpoint's sibling resolved_config.yaml ──
    ckpt_path = Path(checkpoint)
    run_dir   = ckpt_path.parent.parent
    cfg_path  = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found at {cfg_path}")

    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = UNet3DConfig.from_cfg(cfg)
    model     = UNet3DDenoiser(model_cfg).to(device)
    load_checkpoint(str(ckpt_path), model=model, map_location=device)
    model.eval()
    logger.info("Loaded model from %s", ckpt_path)

    # ── Build schedule ────────────────────────────────────────────────────────
    ns = cfg.get("noise_schedule", {})
    schedule = DDPMSchedule(T=int(ns.get("T", 1000)), s=float(ns.get("s", 0.008)), device=device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    latents_root = (REPO / cfg["data"]["latents_root"]).resolve()
    stride       = int(cfg.get("data", {}).get("patch_stride", 64))
    ds = SampledLatentPatchDataset(str(latents_root), "val", patch_stride=stride)
    latent_std = ds.latent_std

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)

    # ── Accumulate per-t loss ─────────────────────────────────────────────────
    T = schedule.T
    N_BINS = 10
    bin_edges = np.linspace(0, T, N_BINS + 1, dtype=int)
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(N_BINS)]
    bin_sum   = np.zeros(N_BINS, dtype=np.float64)
    bin_count = np.zeros(N_BINS, dtype=np.int64)

    from torch.nn import functional as F

    def _batch_to_device(b):
        return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()}

    logger.info("Running %d val batches …", n_batches)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            b = _batch_to_device(batch)
            z  = b["z"]
            nb = b["nb_latents"]
            na = b["nb_avail"]
            pf = b["pos_frac"]
            gp = b["global_por"].squeeze(1)
            lp = b["local_por"].squeeze(1)

            B  = z.shape[0]
            t  = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(z)
            z_t   = schedule.q_sample(z, t, noise)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                eps_pred = model(z_t, t, nb, na, pf, gp, lp)

            # Per-sample MSE
            per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=(1, 2, 3, 4))  # (B,)
            t_np  = t.cpu().numpy()
            ps_np = per_sample.float().cpu().numpy()

            for bi in range(N_BINS):
                mask = (t_np >= bin_edges[bi]) & (t_np < bin_edges[bi + 1])
                if mask.any():
                    bin_sum[bi]   += ps_np[mask].sum()
                    bin_count[bi] += mask.sum()

            if (i + 1) % 20 == 0:
                logger.info("  batch %d/%d", i + 1, n_batches)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n  latent_std used at training = {latent_std:.4f}")
    print(f"\n  {'t range':>14}  {'count':>8}  {'mean MSE':>10}")
    print("  " + "-" * 36)
    losses = []
    for bi in range(N_BINS):
        mean_mse = bin_sum[bi] / max(bin_count[bi], 1)
        losses.append(mean_mse)
        print(f"  {bin_labels[bi]:>14}  {bin_count[bi]:>8}  {mean_mse:>10.5f}")

    losses = np.array(losses)
    low_t_loss  = losses[:3].mean()   # t = 0-299
    high_t_loss = losses[7:].mean()   # t = 700-999

    print(f"\n  low-t  mean MSE (t=0–299)  : {low_t_loss:.5f}")
    print(f"  high-t mean MSE (t=700–999): {high_t_loss:.5f}")

    if high_t_loss > 0 and low_t_loss / high_t_loss > 2.5:
        print(
            "\n  DIAGNOSIS: Loss concentrates at LOW t (irreducible σ·ε floor).\n"
            "  The denoiser cannot recover the stochastic component — expected behaviour.\n"
            "  This is hypothesis 1 (irreducible). NOT a normalization bug."
        )
    elif losses.max() / max(losses.min(), 1e-9) < 1.5:
        print(
            "\n  DIAGNOSIS: Loss is roughly UNIFORM across t.\n"
            "  Consistent with hypothesis 2 (global scale/normalization issue).\n"
            "  Check whether latent_std is correct and that z is ~unit-variance before diffusion."
        )
    else:
        print("\n  DIAGNOSIS: Mixed pattern — investigate further.")

    # ── Optional matplotlib plot ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        mid_t = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(N_BINS)]
        ax.bar(mid_t, losses, width=(T / N_BINS) * 0.8, color="steelblue", alpha=0.8)
        ax.set_xlabel("Diffusion timestep t")
        ax.set_ylabel("ε-MSE")
        ax.set_title("ldm02 — ε-MSE vs timestep")
        out_path = run_dir / "loss_vs_t.png"
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved → {out_path}")
    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")


# ── Check 4 ───────────────────────────────────────────────────────────────────

def check4_success_criterion() -> None:
    print("\n" + "=" * 60)
    print("CHECK 4 — Correct success criterion for ldm02")
    print("=" * 60)
    print("""
  ldm01 final val ε-MSE ≈ 0.20  (50k steps, trains on μ only)
  ldm02 final val ε-MSE ≈ 0.48  (25k steps, trains on μ+σε)

  A HIGHER ε-MSE in ldm02 is expected and does NOT mean ldm02 is worse.
  Reason: ldm02 must denoise z=μ+σε, which has an irreducible σ·ε floor
  that the denoiser cannot predict.  The extra variance in the target
  inflates the loss ceiling independent of model quality.

  The correct comparison is DOWNSTREAM GENERATION quality:
    • Mask Dice against GT volumes
    • Porosity MAE against GT volumes
    • Two-point correlation S₂(r) match
    • Ripley's K match

  Run scripts/generate_volumes.py with each checkpoint and compare.
  Pass --latent-std 0.9985 for ldm02 (vs 0.4571 for ldm01).
    """)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic checks for ldm02.")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to ldm02 .ckpt for loss-vs-t plot (check 2). "
                         "If omitted, only checks 1, 3, 4 run.")
    ap.add_argument("--n-batches",  type=int, default=200,
                    help="Val batches to use for loss-vs-t (default 200).")
    ap.add_argument("--device",     default="cuda",
                    help="Device for model inference (default: cuda).")
    ap.add_argument("--n-sigma-samples", type=int, default=5000,
                    help="Train patches sampled for σ check (default 5000).")
    args = ap.parse_args()

    check1_latent_std()
    check3_sigma_sanity(n_samples=args.n_sigma_samples)

    if args.checkpoint is not None:
        check2_loss_vs_t(args.checkpoint, args.n_batches, args.device)
    else:
        print("\n" + "=" * 60)
        print("CHECK 2 — ε-MSE vs t  (SKIPPED — pass --checkpoint to enable)")
        print("=" * 60)

    check4_success_criterion()


if __name__ == "__main__":
    main()

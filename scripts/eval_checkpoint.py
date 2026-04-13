"""Checkpoint evaluation script — full-volume metrics on the test set.

Stitches non-overlapping 64³ patches into full volumes, then computes:
  - Porosity preservation error
  - 2-point correlation function S2(r)
  - Pore size distribution + Wasserstein-1 distance
  - Pore morphology (sphericity, equivalent diameter)
  - Memorization score (latent nearest-neighbour distance to train set)

Usage
-----
::

    python scripts/eval_checkpoint.py \\
        --checkpoint runs/vae/conv_baseline/last.ckpt \\
        --data_root  data/split_v1 \\
        --config     src/poregen/configs/vae_default.yaml \\
        --out_dir    runs/vae/conv_baseline/eval

    # Quick run — subsample train patches for memorization check
    python scripts/eval_checkpoint.py ... --n_train_sample 5000

"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy import ndimage
from scipy.stats import wasserstein_distance
from skimage.measure import label, regionprops

from poregen.configs.config import load_config
from poregen.dataset.loader import PatchDataset
from poregen.models.vae import build_vae, VAEConfig
from poregen.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

PATCH_SIZE = 64


# ── Volume reconstruction ─────────────────────────────────────────────────────

def reconstruct_volume(
    model: torch.nn.Module,
    zarr_root: zarr.Group,
    volume_id: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a full volume from non-overlapping 64³ patches.

    XCT is normalised as uint8 / 255 → [0, 1] consistently for both GT and
    reconstruction.  No per-volume z-score statistics are required.

    Returns
    -------
    xct_gt, mask_gt, xct_recon, mask_recon : float32 arrays in [0, 1], shape (D, H, W)
    """
    grp   = zarr_root[volume_id]
    xct_z = grp["xct"]
    msk_z = grp["mask"]
    D, H, W = xct_z.shape

    xct_recon  = np.zeros((D, H, W), dtype=np.float32)
    mask_recon = np.zeros((D, H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for z0 in range(0, D - PATCH_SIZE + 1, PATCH_SIZE):
            for y0 in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
                for x0 in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
                    xct_p  = xct_z [z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                    mask_p = msk_z[z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

                    xct_f  = xct_p.astype(np.float32) / 255.0
                    mask_f = mask_p.astype(np.float32)

                    xct_t  = torch.from_numpy(xct_f) .unsqueeze(0).unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device)

                    out = model(xct_t, mask_t)

                    xct_recon [z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = (
                        out.xct_logits.clamp(0.0, 1.0).squeeze().cpu().numpy()
                    )
                    mask_recon[z0:z0+PATCH_SIZE, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = (
                        torch.sigmoid(out.mask_logits).squeeze().cpu().numpy()
                    )

    xct_gt  = np.array(xct_z ).astype(np.float32) / 255.0
    mask_gt = np.array(msk_z).astype(np.float32)

    # Crop to tiled region (drops remainder voxels)
    dz = (D // PATCH_SIZE) * PATCH_SIZE
    dy = (H // PATCH_SIZE) * PATCH_SIZE
    dx = (W // PATCH_SIZE) * PATCH_SIZE

    return (
        xct_gt [:dz, :dy, :dx],
        mask_gt[:dz, :dy, :dx],
        xct_recon [:dz, :dy, :dx],
        mask_recon[:dz, :dy, :dx],
    )


# ── S2(r) — two-point correlation function ───────────────────────────────────

def s2_radial(binary: np.ndarray, r_max: int = 50, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic two-point correlation function via FFT autocorrelation.

    S2(r) = P(both points in pore phase | separation r).

    Returns
    -------
    r_vals : (n_bins,) — radial distances in voxels
    s2     : (n_bins,) — S2 values
    """
    vol = binary.astype(np.float64)
    fft = np.fft.fftn(vol)
    autocorr = np.real(np.fft.ifftn(fft * np.conj(fft))) / vol.size

    D, H, W = binary.shape
    dz = np.fft.fftfreq(D) * D
    dy = np.fft.fftfreq(H) * H
    dx = np.fft.fftfreq(W) * W
    ZZ, YY, XX = np.meshgrid(dz, dy, dx, indexing="ij")
    R = np.sqrt(ZZ**2 + YY**2 + XX**2)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    s2 = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if mask.any():
            s2[i] = autocorr[mask].mean()

    r_vals = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_vals, s2


def s2_wasserstein(s2_real: np.ndarray, s2_pred: np.ndarray) -> float:
    """Wasserstein-1 distance between two S2(r) curves (treated as distributions)."""
    # Normalise to unit sum before comparing as distributions
    eps = 1e-12
    a = np.clip(s2_real, 0, None)
    b = np.clip(s2_pred, 0, None)
    a = a / (a.sum() + eps)
    b = b / (b.sum() + eps)
    r = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(r, r, a, b))


# ── Pore size distribution ────────────────────────────────────────────────────

def pore_size_distribution(binary: np.ndarray) -> np.ndarray:
    """Connected-component pore volumes (in voxels)."""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.array([0.0])
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return np.array(sizes, dtype=np.float64)


# ── Pore morphology ───────────────────────────────────────────────────────────

def pore_morphology(binary: np.ndarray, max_pores: int = 500) -> dict[str, float]:
    """Mean sphericity and equivalent diameter via skimage regionprops.

    Sphericity = π^(1/3) × (6 V)^(2/3) / A  (sphere → 1.0).
    Limited to the *max_pores* largest pores for speed.
    """
    labeled = label(binary)
    props   = regionprops(labeled)

    if not props:
        return {"sphericity_mean": float("nan"), "eq_diameter_mean": float("nan")}

    # Sort by volume descending, take top N
    props = sorted(props, key=lambda p: p.area, reverse=True)[:max_pores]

    sphericities = []
    diameters    = []
    for p in props:
        vol  = p.area
        area = p.area_convex  # approximation — true surface area needs marching cubes
        if area > 0:
            sphericity = (np.pi ** (1 / 3)) * ((6 * vol) ** (2 / 3)) / area
            sphericities.append(float(sphericity))
        diameters.append(float(p.equivalent_diameter_area))

    return {
        "sphericity_mean": float(np.mean(sphericities)) if sphericities else float("nan"),
        "sphericity_std":  float(np.std(sphericities))  if sphericities else float("nan"),
        "eq_diameter_mean": float(np.mean(diameters)),
        "eq_diameter_std":  float(np.std(diameters)),
        "n_pores": len(props),
    }


# ── Memorization score ────────────────────────────────────────────────────────

@torch.no_grad()
def memorization_score(
    model: torch.nn.Module,
    test_ds: PatchDataset,
    train_ds: PatchDataset,
    device: torch.device,
    n_test: int = 1000,
    n_train: int = 5000,
    batch_size: int = 32,
) -> dict[str, float]:
    """Mean nearest-neighbour distance in latent space (test → train).

    A low score indicates the model is memorizing training patches rather
    than generalising.  Higher is better (more diverse representations).
    """
    from torch.utils.data import DataLoader, Subset
    import random

    def encode(ds: PatchDataset, n: int) -> torch.Tensor:
        idx = random.sample(range(len(ds)), min(n, len(ds)))
        sub = Subset(ds, idx)
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=4)
        mus = []
        model.eval()
        for batch in loader:
            xct  = batch["xct"].to(device)
            mask = batch["mask"].to(device)
            out  = model(xct, mask)
            # Flatten spatial dims → (B, C*d*h*w)
            mu_flat = out.mu.reshape(out.mu.shape[0], -1)
            mus.append(mu_flat.cpu())
        return torch.cat(mus, dim=0)  # (N, D_latent)

    logger.info("Encoding test patches for memorization check …")
    z_test  = encode(test_ds,  n_test)
    logger.info("Encoding train patches for memorization check …")
    z_train = encode(train_ds, n_train)

    # Compute pairwise L2: (n_test, n_train)
    # Done in chunks to avoid OOM
    chunk = 256
    min_dists = []
    for i in range(0, len(z_test), chunk):
        zt = z_test[i:i + chunk].unsqueeze(1)       # (chunk, 1, D)
        d  = (zt - z_train.unsqueeze(0)).pow(2).sum(-1).sqrt()  # (chunk, n_train)
        min_dists.append(d.min(dim=1).values)

    min_dists_all = torch.cat(min_dists)
    return {
        "memorization_nn_dist_mean": min_dists_all.mean().item(),
        "memorization_nn_dist_std":  min_dists_all.std().item(),
    }


# ── Per-volume metrics ────────────────────────────────────────────────────────

def eval_volume(
    volume_id: str,
    model: torch.nn.Module,
    zarr_root: zarr.Group,
    device: torch.device,
    r_max: int = 50,
) -> dict:
    """Compute all full-volume metrics for one test volume."""
    logger.info("Evaluating volume: %s", volume_id)
    t0 = time.perf_counter()

    xct_gt, mask_gt, xct_recon, mask_recon = reconstruct_volume(
        model, zarr_root, volume_id, device
    )

    bin_gt   = (mask_gt   >= 0.5).astype(bool)
    bin_pred = (mask_recon >= 0.5).astype(bool)

    # Porosity
    por_gt   = float(bin_gt.mean())
    por_pred = float(bin_pred.mean())
    por_err  = abs(por_pred - por_gt)

    # XCT reconstruction quality
    mae_val  = float(np.abs(xct_recon - xct_gt).mean())

    # S2(r)
    r_vals, s2_gt   = s2_radial(bin_gt,   r_max=r_max)
    _,      s2_pred = s2_radial(bin_pred, r_max=r_max)
    s2_w1 = s2_wasserstein(s2_gt, s2_pred)

    # Pore size distribution Wasserstein-1
    psd_gt   = pore_size_distribution(bin_gt)
    psd_pred = pore_size_distribution(bin_pred)
    psd_w1   = float(wasserstein_distance(psd_gt, psd_pred)) if (len(psd_gt) > 0 and len(psd_pred) > 0) else float("nan")

    # Morphology
    morph_gt   = pore_morphology(bin_gt)
    morph_pred = pore_morphology(bin_pred)

    elapsed = time.perf_counter() - t0
    logger.info("  done in %.1fs  por_err=%.4f  s2_w1=%.4f  psd_w1=%.4f",
                elapsed, por_err, s2_w1, psd_w1)

    return {
        "volume_id":      volume_id,
        "porosity_gt":    por_gt,
        "porosity_pred":  por_pred,
        "porosity_error": por_err,
        "xct_mae":        mae_val,
        "s2_wasserstein": s2_w1,
        "s2_gt":          s2_gt.tolist(),
        "s2_pred":        s2_pred.tolist(),
        "s2_r_vals":      r_vals.tolist(),
        "psd_wasserstein": psd_w1,
        "morphology_gt":  morph_gt,
        "morphology_pred": morph_pred,
        "elapsed_s":      elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-volume checkpoint evaluation for PoreGen VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument(
        "--data_root",
        default=None,
        help="Optional dataset root (defaults to data/<cfg.data.dataset_root>)",
    )
    p.add_argument("--config",     required=True, help="vae_default.yaml path")
    p.add_argument("--out_dir",    required=True, help="Directory to write eval results")
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--r_max",      type=int, default=50,   help="Max radius for S2(r) in voxels")
    p.add_argument("--n_test_sample",  type=int, default=1000, help="Test patches for memorization check")
    p.add_argument("--n_train_sample", type=int, default=5000, help="Train patches for memorization check")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = build_parser().parse_args(argv)
    t_global = time.perf_counter()

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ────────────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    dataset_root = cfg.get("data", {}).get("dataset_root", "split_v1")
    data_root = Path(args.data_root) if args.data_root else Path("data") / dataset_root
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Data root: %s", data_root)

    model = build_vae(
        cfg["model"]["name"],
        z_channels=cfg["model"]["z_channels"],
        base_channels=cfg["model"]["base_channels"],
        n_blocks=cfg["model"]["n_blocks"],
        patch_size=cfg["model"]["patch_size"],
    ).to(device)

    step, meta = load_checkpoint(args.checkpoint, model, restore_rng=False, map_location=device)
    logger.info("Loaded checkpoint at step %d", step)

    # ── Data ─────────────────────────────────────────────────────────────────
    index_path = data_root / "patch_index.parquet"
    zarr_path  = data_root / "volumes.zarr"

    zarr_root = zarr.open_group(str(zarr_path), mode="r")

    index_df = pd.read_parquet(str(index_path))
    test_vol_ids = index_df[index_df["split"] == "test"]["volume_id"].unique().tolist()
    logger.info("Test volumes: %d", len(test_vol_ids))

    # ── Per-volume evaluation ─────────────────────────────────────────────────
    volume_results = []
    for vol_id in test_vol_ids:
        try:
            res = eval_volume(
                vol_id, model, zarr_root, device, r_max=args.r_max
            )
            volume_results.append(res)
        except Exception as exc:
            logger.error("Failed on %s: %s", vol_id, exc)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scalar_keys = ["porosity_error", "xct_mae", "s2_wasserstein", "psd_wasserstein"]
    agg: dict = {"step": step, "n_volumes": len(volume_results)}
    for k in scalar_keys:
        vals = [r[k] for r in volume_results if not np.isnan(r.get(k, float("nan")))]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")

    # ── Memorization score ────────────────────────────────────────────────────
    logger.info("Computing memorization score …")
    try:
        train_ds = PatchDataset(index_path, data_root, split="train")
        test_ds  = PatchDataset(index_path, data_root, split="test")
        mem = memorization_score(
            model, test_ds, train_ds, device,
            n_test=args.n_test_sample,
            n_train=args.n_train_sample,
        )
        agg.update(mem)
        logger.info("Memorization NN dist: %.4f ± %.4f",
                    mem["memorization_nn_dist_mean"], mem["memorization_nn_dist_std"])
    except Exception as exc:
        logger.error("Memorization check failed: %s", exc)

    # ── Save ──────────────────────────────────────────────────────────────────
    summary_path = out_dir / f"eval_step{step:08d}.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": agg, "volumes": volume_results}, f, indent=2)
    logger.info("Saved → %s", summary_path)

    elapsed = time.perf_counter() - t_global
    logger.info("=" * 60)
    logger.info("EVAL COMPLETE  step=%d  volumes=%d  time=%.1fs", step, len(volume_results), elapsed)
    for k in scalar_keys:
        logger.info("  %-25s %.4f ± %.4f", k, agg.get(f"{k}_mean", float("nan")), agg.get(f"{k}_std", float("nan")))
    if "memorization_nn_dist_mean" in agg:
        logger.info("  %-25s %.4f", "memorization_nn_dist", agg["memorization_nn_dist_mean"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

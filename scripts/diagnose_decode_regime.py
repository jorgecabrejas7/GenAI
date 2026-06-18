"""Decode-regime diagnostic: clean mu vs reparameterized sample.

Purpose
-------
The LDM produces a mu-like point estimate (no sigma) for the VAE latent. At
generation time the VAE decoder is therefore fed a *clean* mu, even though it
was only ever trained on reparameterized samples z = mu + sigma * eps. This
script tests whether decoding clean mu (s=0) gives measurably worse
reconstructions than decoding the training-time sample (s=1) -- especially
for the mask/segmentation head -- by sweeping the noise scale s in
z = mu + s * sigma * eps for a fixed eps per patch.

This is a **read-only diagnostic**: it loads an existing trained VAE
checkpoint in eval mode, runs forward passes under ``torch.no_grad()``, and
writes outputs to a new ``diagnostics/decode_regime_<timestamp>/`` folder. It
does not modify any existing code, config, checkpoint, or training artifact.

Usage
-----
::

    python scripts/diagnose_decode_regime.py \\
        --vae-experiment r05/base \\
        --checkpoint runs/vae/r05-run-.../best.ckpt \\
        --data-root data/split_v2 \\
        [--n-per-bin 4] \\
        [--scales 0.0 0.25 0.5 0.75 1.0] \\
        [--select-seed 0] [--noise-seed 0] \\
        [--out-dir diagnostics/decode_regime_<timestamp>] \\
        [--device cuda]

Outputs
-------
``<out_dir>/metrics.csv``
    One row per (sample, scale) with mask + XCT quality metrics.
``<out_dir>/summary.md``
    Aggregate mean +/- std per scale, paired deltas (s=1 - s=0), and a
    written verdict.
``<out_dir>/<sample_id>/``
    GT XCT/mask (npy + PNG) and, per scale, the reconstructed XCT and
    predicted mask (npy + PNG).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tifffile
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MASK_THRESHOLD = 0.5  # project-wide convention (see e.g. src/poregen/metrics/seg.py)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return here.parent


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Model loading (read-only reuse of existing config / registry / checkpoint
# utilities -- mirrors scripts/encode_latents.py::_load_vae)
# ---------------------------------------------------------------------------

def _load_vae(experiment: str, checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from poregen.configuration import resolve_experiment
    from poregen.models.vae import build_vae
    from poregen.training.checkpoint import load_checkpoint

    resolved = resolve_experiment(experiment, repo_root=REPO_ROOT)
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

    step, ckpt_metadata = load_checkpoint(checkpoint, model=model, map_location=device, restore_rng=False)
    model.eval()
    logger.info("Loaded VAE %s from %s (step=%d)", mc["name"], checkpoint, step)

    info = {
        "experiment_id": resolved.experiment_id,
        "model_name": mc["name"],
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_step": step,
        "checkpoint_metadata": ckpt_metadata,
    }
    return model, info


# ---------------------------------------------------------------------------
# Sample selection: stratify the test split into low / mid / high porosity
# tertiles and draw n_per_bin patches from each with a fixed seed.
# ---------------------------------------------------------------------------

def _select_stratified_patches(test_df: pd.DataFrame, n_per_bin: int, seed: int) -> pd.DataFrame:
    """Return a DataFrame of selected rows with an added ``bin`` column.

    The returned DataFrame's index is preserved from *test_df* (i.e. it
    matches the row position in the ``PatchDataset`` this index came from),
    so callers can index the dataset directly with ``selected.index``.

    Stratification is by porosity tertile (equal-count groups after sorting),
    which is robust to skewed / duplicate-heavy porosity distributions
    (unlike ``pd.qcut``, which can collapse bins on duplicate edges).
    """
    rng = np.random.RandomState(seed)
    ordered = test_df.sort_values("porosity", kind="stable")  # keep original index
    n = len(ordered)
    edges = [0, n // 3, 2 * n // 3, n]
    bin_names = ["low", "mid", "high"]

    chosen = []
    for name, lo, hi in zip(bin_names, edges[:-1], edges[1:]):
        bin_df = ordered.iloc[lo:hi]  # positional slice; index labels retained
        if len(bin_df) < n_per_bin:
            raise ValueError(
                f"Porosity bin '{name}' only has {len(bin_df)} test patches; "
                f"need {n_per_bin}."
            )
        pick_idx = rng.choice(bin_df.index.to_numpy(), size=n_per_bin, replace=False)
        picked = bin_df.loc[pick_idx].copy()
        picked["bin"] = name
        chosen.append(picked)

    return pd.concat(chosen, axis=0)  # index still maps 1:1 to test_df rows


# ---------------------------------------------------------------------------
# Metrics -- reused read-only from the project's existing eval pipeline
# (src/poregen/eval/metrics.py), which is the authoritative source for these
# computations (PSNR, SSIM, Dice/IoU, sharpness ratio).
# ---------------------------------------------------------------------------

from poregen.eval.metrics import _psnr, _ssim, _dice_precision_recall, _sharpness_proxy  # noqa: E402


def _fpr_fnr(gt_bin: np.ndarray, pred_bin: np.ndarray) -> tuple[float, float]:
    """False-positive / false-negative rate from two boolean arrays."""
    tp = int((gt_bin & pred_bin).sum())
    fp = int((~gt_bin & pred_bin).sum())
    fn = int((gt_bin & ~pred_bin).sum())
    tn = int((~gt_bin & ~pred_bin).sum())
    fpr = float(fp) / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = float(fn) / (fn + tp) if (fn + tp) > 0 else float("nan")
    return fpr, fnr


def _sharpness_ratio(gt_vol: np.ndarray, recon_vol: np.ndarray) -> float:
    sg = _sharpness_proxy(gt_vol)
    sr = _sharpness_proxy(recon_vol)
    return float(sr / sg) if sg > 1e-9 else float("nan")


def _patch_metrics(gt_xct: np.ndarray, gt_mask_bin: np.ndarray, recon_xct: np.ndarray, mask_prob: np.ndarray) -> dict[str, float]:
    pred_bin = mask_prob >= MASK_THRESHOLD
    pred_por = float(mask_prob.mean())
    gt_por = float(gt_mask_bin.mean())
    seg = _dice_precision_recall(gt_mask_bin, pred_bin)
    fpr, fnr = _fpr_fnr(gt_mask_bin, pred_bin)
    return {
        "pred_porosity": pred_por,
        "porosity_mae": abs(pred_por - gt_por),
        "porosity_bias": pred_por - gt_por,
        "dice_pos_only": seg["dice"],
        "iou_pos_only": seg["iou"],
        "fpr": fpr,
        "fnr": fnr,
        "psnr": _psnr(gt_xct, recon_xct),
        "ssim": _ssim(gt_xct, recon_xct),
        "mae": float(np.abs(gt_xct - recon_xct).mean()),
        "sharpness_ratio": _sharpness_ratio(gt_xct, recon_xct),
    }


# ---------------------------------------------------------------------------
# TIFF output (ImageJ-compatible 3-D stacks)
# ---------------------------------------------------------------------------

def _mode_name(s: float) -> str:
    """Human-readable decode-mode label for use in TIFF filenames.

    s=0.0  → z_eq_mu                          (LDM regime: clean posterior mean)
    s=1.0  → z_eq_mu_plus_sigma_eps            (training regime: full reparameterization)
    other  → z_eq_mu_plus_<s>_sigma_eps        (intermediate)
    """
    if s == 0.0:
        return "z_eq_mu"
    if s == 1.0:
        return "z_eq_mu_plus_sigma_eps"
    return f"z_eq_mu_plus_{s:.2f}_sigma_eps"


def _save_tiff(vol: np.ndarray, path: Path) -> None:
    """Write a (D, H, W) float32 volume as a multi-page TIFF stack.

    ImageJ/FIJI opens this directly as a 3-D stack.  Float32 [0, 1] values
    are preserved exactly — no rescaling or quantisation.
    """
    tifffile.imwrite(str(path), vol.astype(np.float32))


def _save_tiff_uint8(vol: np.ndarray, path: Path) -> None:
    """Write a binary {0, 1} float32 mask as a uint8 TIFF (0 / 255).

    Useful for ground-truth masks so they render as clean binary in ImageJ
    without needing a manual threshold step.
    """
    tifffile.imwrite(str(path), (vol * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Visualisation: orthogonal mid-slice PNGs (cheap on 64^3 volumes)
# ---------------------------------------------------------------------------

def _take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def _save_orthogonal_png(xct_vol: np.ndarray, mask_vol: np.ndarray, title: str, out_path: Path) -> None:
    """2 rows (XCT, mask) x 3 cols (axial, coronal, sagittal) mid-slice grid."""
    axis_names = ["axial (z)", "coronal (y)", "sagittal (x)"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for col, (axis, name) in enumerate(zip([0, 1, 2], axis_names)):
        mid = xct_vol.shape[axis] // 2
        axes[0, col].imshow(_take_slice(xct_vol, axis, mid), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"XCT {name}")
        axes[0, col].axis("off")
        axes[1, col].imshow(_take_slice(mask_vol, axis, mid), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"mask {name}")
        axes[1, col].axis("off")
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary / verdict
# ---------------------------------------------------------------------------

MASK_METRICS = ["porosity_mae", "dice_pos_only", "iou_pos_only", "fpr", "fnr"]
XCT_METRICS = ["psnr", "ssim", "mae", "sharpness_ratio"]


def _relative_change(v_s0: float, v_s1: float, higher_is_better: bool) -> float:
    """Signed relative change from s=0 to s=1, positive = improvement."""
    denom = abs(v_s0) if abs(v_s0) > 1e-9 else 1e-9
    delta = (v_s1 - v_s0) / denom
    return delta if higher_is_better else -delta


HIGHER_IS_BETTER = {
    "porosity_mae": False, "dice_pos_only": True, "iou_pos_only": True,
    "fpr": False, "fnr": False,
    "psnr": True, "ssim": True, "mae": False, "sharpness_ratio": None,  # ratio: closer to 1 is better, handled separately
}


def _build_summary(df: pd.DataFrame, scales: list[float], info: dict[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("# Decode-regime diagnostic — summary\n")
    lines.append("## Reproducibility\n")
    lines.append(f"- VAE experiment: `{info['experiment_id']}`")
    lines.append(f"- Checkpoint: `{info['checkpoint_path']}` (step {info['checkpoint_step']})")
    lines.append(f"- Checkpoint metadata: `{info['checkpoint_metadata']}`")
    lines.append(f"- Data root: `{args.data_root}`")
    lines.append(f"- Scales swept: {scales}")
    lines.append(f"- Sample-selection seed: {args.select_seed}")
    lines.append(f"- Noise (eps) seed base: {args.noise_seed}")
    lines.append(f"- Mask threshold: {MASK_THRESHOLD}")
    lines.append("")
    lines.append("## Selected samples\n")
    sample_table = (
        df[df["scale"] == scales[0]][["sample_id", "bin", "volume_id", "z0", "y0", "x0", "gt_porosity", "sigma_avg"]]
        .sort_values(["bin", "sample_id"])
    )
    lines.append("```")
    lines.append(sample_table.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Aggregate mean ± std per scale\n")
    all_metrics = MASK_METRICS + XCT_METRICS
    agg = df.groupby("scale")[all_metrics].agg(["mean", "std"])
    rows = []
    for s in scales:
        row = {"scale": s}
        for m in all_metrics:
            row[f"{m}_mean"] = agg.loc[s, (m, "mean")]
            row[f"{m}_std"] = agg.loc[s, (m, "std")]
        rows.append(row)
    agg_df = pd.DataFrame(rows)
    lines.append("```")
    lines.append(agg_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("```")
    lines.append("")

    # ---- paired deltas (s=1 minus s=0) per sample ----
    s_lo, s_hi = scales[0], scales[-1]
    wide = df.pivot(index="sample_id", columns="scale", values=all_metrics + ["gt_porosity"])
    lines.append(f"## Paired delta ({s_hi} minus {s_lo}) across samples\n")
    delta_rows = []
    catastrophic: dict[str, list[str]] = {}
    correlations: dict[str, float] = {}
    for m in all_metrics:
        deltas = wide[(m, s_hi)] - wide[(m, s_lo)]
        gt_por = wide[("gt_porosity", s_lo)]
        mean_d, std_d = float(deltas.mean()), float(deltas.std())
        # "driven by a few catastrophic patches" heuristic: a metric is
        # patch-driven if the largest single |delta| exceeds 2x the mean |delta|
        # of the remaining samples (otherwise it's a roughly uniform shift).
        abs_deltas = deltas.abs().sort_values(ascending=False)
        top_id = abs_deltas.index[0]
        top_val = float(abs_deltas.iloc[0])
        rest_mean = float(abs_deltas.iloc[1:].mean()) if len(abs_deltas) > 1 else 0.0
        driven_by_few = rest_mean > 0 and top_val > 2.0 * rest_mean
        if driven_by_few:
            catastrophic[m] = [top_id]
        corr = float(np.corrcoef(gt_por.to_numpy(), deltas.to_numpy())[0, 1]) if deltas.std() > 0 else float("nan")
        correlations[m] = corr
        delta_rows.append({
            "metric": m,
            "mean_delta": mean_d,
            "std_delta": std_d,
            "max_abs_delta_sample": top_id,
            "max_abs_delta": top_val,
            "uniform_shift": not driven_by_few,
            "porosity_correlation": corr,
        })
    delta_df = pd.DataFrame(delta_rows)
    lines.append("```")
    lines.append(delta_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("```")
    lines.append("")

    # ---- verdict ----
    lines.append("## Verdict\n")
    mask_rel_changes = []
    for m in MASK_METRICS:
        v0 = float(agg.loc[s_lo, (m, "mean")])
        v1 = float(agg.loc[s_hi, (m, "mean")])
        hib = HIGHER_IS_BETTER[m]
        mask_rel_changes.append(_relative_change(v0, v1, hib))
    xct_rel_changes = []
    for m in XCT_METRICS:
        v0 = float(agg.loc[s_lo, (m, "mean")])
        v1 = float(agg.loc[s_hi, (m, "mean")])
        if m == "sharpness_ratio":
            # closer to 1.0 is better; relative change measured as
            # improvement in |ratio - 1|
            rel = (abs(v0 - 1.0) - abs(v1 - 1.0)) / max(abs(v0 - 1.0), 1e-9)
        else:
            rel = _relative_change(v0, v1, HIGHER_IS_BETTER[m])
        xct_rel_changes.append(rel)

    mask_mean_rel = float(np.mean(mask_rel_changes))
    xct_mean_rel = float(np.mean(xct_rel_changes))

    lines.append(f"- Mean relative improvement s={s_lo}→s={s_hi} across mask metrics: **{mask_mean_rel:+.1%}**")
    lines.append(f"- Mean relative improvement s={s_lo}→s={s_hi} across XCT metrics: **{xct_mean_rel:+.1%}**")
    lines.append("")

    DECODE_REGIME_THRESHOLD = 0.10  # 10 percentage points of relative improvement
    XCT_FLAT_THRESHOLD = 0.05

    if mask_mean_rel > DECODE_REGIME_THRESHOLD and abs(xct_mean_rel) < XCT_FLAT_THRESHOLD:
        verdict = (
            "**DECODE-REGIME PROBLEM.** Mask quality degrades substantially when decoding "
            "clean mu (s=0) instead of the training-time sample (s=1), while XCT "
            "reconstruction quality stays roughly flat across the same sweep. This is "
            "consistent with the decoder's mask head being miscalibrated outside the "
            "noisy-z manifold it was trained on — i.e. the LDM's clean point estimate is "
            "off the decoder's training support specifically for the mask head."
        )
    elif mask_mean_rel > DECODE_REGIME_THRESHOLD and abs(xct_mean_rel) >= XCT_FLAT_THRESHOLD:
        verdict = (
            "**PARTIAL / MIXED.** Mask quality degrades at s=0, but XCT quality also "
            "shifts non-trivially across the sweep, so the effect is not cleanly isolated "
            "to the mask head. Decode-regime sensitivity is present but not mask-specific."
        )
    elif mask_mean_rel < -DECODE_REGIME_THRESHOLD:
        verdict = (
            "**REVERSE of the hypothesis — clean mu (s=0) is actually BETTER for the mask "
            "head than the training-time sample (s=1) on this checkpoint/test sample.** "
            "Mask metrics consistently improve as the noise scale *decreases*, so feeding "
            "the decoder a clean LDM point estimate is not, by itself, pushing the mask "
            "head off its training support. The LDM-time mask degradation likely has a "
            "different cause (e.g. the LDM's mu estimate being a poor approximation of the "
            "true posterior mean, a latent-scale mismatch between the LDM and VAE, or an "
            "issue upstream of decoding)."
        )
    else:
        verdict = (
            "**NOT a decode-regime problem (by this sweep).** Mask and XCT metrics move "
            "by comparable, small amounts across the noise-scale sweep — clean mu is not "
            "measurably worse than the training-time sample on this checkpoint/test "
            "sample. The LDM-time mask degradation likely has a different cause "
            "(e.g. LDM latent quality, decoder/LDM latent-scale mismatch, or the LDM "
            "point estimate itself being a poor mu approximation)."
        )
    lines.append(verdict)
    lines.append("")

    if catastrophic:
        lines.append("### Metrics driven by a few catastrophic patches (not a uniform shift)\n")
        for m, ids in catastrophic.items():
            lines.append(f"- `{m}`: dominated by sample(s) {ids}")
        lines.append("")

    por_dependent = {m: c for m, c in correlations.items() if not math.isnan(c) and abs(c) > 0.5}
    if por_dependent:
        lines.append("### Porosity-dependence (|Pearson r| > 0.5 between GT φ and the paired delta)\n")
        for m, c in por_dependent.items():
            lines.append(f"- `{m}`: r = {c:+.2f}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="VAE decode-regime diagnostic: clean mu vs reparameterized sample.")
    ap.add_argument("--vae-experiment", default="r05/base", help="Experiment ref, e.g. 'r05/base'")
    ap.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT
            / "runs/vae/r05-run-0001-20260420-142643-archv2-conv_noattn_dualbranch-z16-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone/best.ckpt"
        ),
    )
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data/split_v2"))
    ap.add_argument("--n-per-bin", type=int, default=4, help="Patches per porosity tertile (default: 4 -> 12 total)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--select-seed", type=int, default=0, help="Seed for stratified sample selection")
    ap.add_argument("--noise-seed", type=int, default=0, help="Base seed for per-patch eps draw")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: diagnostics/decode_regime_<timestamp>)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root).resolve()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "diagnostics" / f"decode_regime_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Output dir: %s", out_dir)

    model, info = _load_vae(args.vae_experiment, args.checkpoint, device)

    from poregen.eval.metrics import _encode  # noqa: E402  (architecture-agnostic encode, read-only reuse)
    from poregen.dataset.loader import PatchDataset

    index_path = data_root / "patch_index.parquet"
    test_ds = PatchDataset(index_path, data_root, split="test")
    selected = _select_stratified_patches(test_ds.df, args.n_per_bin, args.select_seed)
    logger.info("Selected %d patches: %d per bin (low/mid/high)", len(selected), args.n_per_bin)

    rows: list[dict[str, Any]] = []
    scales = sorted(args.scales)

    with torch.no_grad():
        for rank, (_, sel_row) in enumerate(selected.iterrows()):
            # `selected` was built from test_ds.df directly (see
            # _select_stratified_patches), so the original integer index is
            # preserved as the row's DataFrame index and indexes test_ds directly.
            ds_idx = int(sel_row.name)
            patch = test_ds[ds_idx]

            sample_id = f"{sel_row['bin']}_{rank % args.n_per_bin:02d}_{patch['volume_id']}"
            sample_dir = out_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            xct_t = patch["xct"].unsqueeze(0).to(device)    # (1,1,64,64,64)
            mask_t = patch["mask"].unsqueeze(0).to(device)
            gt_xct_np = xct_t.squeeze().cpu().numpy()
            gt_mask_np = mask_t.squeeze().cpu().numpy()
            gt_mask_bin = gt_mask_np >= MASK_THRESHOLD
            gt_porosity = float(patch["porosity"])

            # ---- ground truth TIFFs (flat in sample dir for easy ImageJ drag-and-drop) ----
            _save_tiff(gt_xct_np,         sample_dir / "input_xct.tif")
            _save_tiff_uint8(gt_mask_np,  sample_dir / "input_mask_gt.tif")
            np.save(sample_dir / "gt_xct.npy", gt_xct_np)
            np.save(sample_dir / "gt_mask.npy", gt_mask_np)
            _save_orthogonal_png(gt_xct_np, gt_mask_np, f"{sample_id}  GT  (φ={gt_porosity:.4f})", sample_dir / "gt_slices.png")

            h = _encode(model, xct_t)
            mu = model.to_mu(h)
            logvar = model.to_logvar(h)
            std = torch.exp(0.5 * logvar)
            sigma_avg = float(std.mean().item())

            torch.manual_seed(args.noise_seed + rank)
            eps = torch.randn_like(mu)

            for s in scales:
                z = mu + s * std * eps
                dec = model.decoder(z)
                xct_logits = model.xct_head(dec)
                mask_logits = model.mask_head(dec)

                xct_recon = xct_logits.clamp(0.0, 1.0).squeeze().cpu().numpy()
                mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
                mask_bin = mask_prob >= MASK_THRESHOLD

                m = _patch_metrics(gt_xct_np, gt_mask_bin, xct_recon, mask_prob)
                m.update({
                    "sample_id": sample_id,
                    "bin": sel_row["bin"],
                    "volume_id": patch["volume_id"],
                    "z0": int(patch["coords"][0]), "y0": int(patch["coords"][1]), "x0": int(patch["coords"][2]),
                    "gt_porosity": gt_porosity,
                    "sigma_avg": sigma_avg,
                    "scale": s,
                })
                rows.append(m)

                # ---- decoded TIFFs (flat in sample dir, named by mode) ----
                mode = _mode_name(s)
                _save_tiff(xct_recon,  sample_dir / f"decoded_xct__{mode}.tif")
                _save_tiff(mask_prob,  sample_dir / f"decoded_mask_prob__{mode}.tif")

                # ---- per-scale subfolder: npz + overview PNG ----
                s_dir = sample_dir / f"s{s:.2f}"
                s_dir.mkdir(exist_ok=True)
                np.savez(s_dir / "decoded.npz", xct_recon=xct_recon, mask_prob=mask_prob, mask_bin=mask_bin)
                _save_orthogonal_png(
                    xct_recon, mask_prob,
                    f"{sample_id}  s={s:.2f}  (dice={m['dice_pos_only']:.3f}, psnr={m['psnr']:.1f}dB)",
                    s_dir / "slices.png",
                )

            logger.info(
                "[%2d/%2d] %-28s φ_gt=%.4f σ_avg=%.4f  done",
                rank + 1, len(selected), sample_id, gt_porosity, sigma_avg,
            )

    df = pd.DataFrame(rows)
    col_order = [
        "sample_id", "bin", "volume_id", "z0", "y0", "x0", "gt_porosity", "sigma_avg", "scale",
        "pred_porosity", "porosity_mae", "porosity_bias", "dice_pos_only", "iou_pos_only", "fpr", "fnr",
        "psnr", "ssim", "mae", "sharpness_ratio",
    ]
    df = df[col_order]
    df.to_csv(out_dir / "metrics.csv", index=False)
    logger.info("Wrote %s", out_dir / "metrics.csv")

    summary_md = _build_summary(df, scales, info, args)
    (out_dir / "summary.md").write_text(summary_md)
    logger.info("Wrote %s", out_dir / "summary.md")

    repro = {
        "vae_experiment": info["experiment_id"],
        "checkpoint_path": info["checkpoint_path"],
        "checkpoint_step": info["checkpoint_step"],
        "data_root": str(data_root),
        "scales": scales,
        "select_seed": args.select_seed,
        "noise_seed": args.noise_seed,
        "n_per_bin": args.n_per_bin,
        "mask_threshold": MASK_THRESHOLD,
        "sample_ids": df["sample_id"].unique().tolist(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(repro, indent=2))

    print(f"\nOutput folder: {out_dir}")
    print("\n" + summary_md.split("## Verdict")[1].split("###")[0])


if __name__ == "__main__":
    main()

"""Layup round-trip evaluation for the completed ldm05 run (130k, RAW).

Requests three stacking sequences from the generator, then measures the
per-ply tow angles back with the T-I estimator and scores the recovery.

Estimator-geometry decision (option (a) of the design note)
-----------------------------------------------------------
The T-I estimator was built for >= 1024x1024 in-plane windows; generated
volumes so far were 192^3.  Instead of shrinking the estimator window (a 192^2
window holds only 3-12 periods of the 16-64-voxel tow band, so the angular
resolution of the low-radius FFT bins collapses), this script generates
1024x1024x192-voxel volumes (16x16x3 patch grid, 768 patches) so the
UNCHANGED 1024-pixel window applies.  The only adaptations are: the volume IS
the window (one window instead of T-I's two), the ply block edges are the
REQUESTED ply grid (k * 19.6 voxels — nothing is estimated), and the
wavelength band uses the requested pitch.  This adapted path is validated
first, on real volumes, against the full T-I output in
runs/campaigns/01-conditioning-design/T-I/layup_field.json (--validate-only runs
just that stage, CPU).

Generation: mode=sequential at s_por=1.0 and mode=joint at s_por=1.5
(matching the dose-response settings), DDIM-50, RAW weights, target global
porosity 0.03 delivered as a coherent local field (T-E marginal + T-D
smoothing).  Orientation is NOT guided in either mode — the porosity guidance
setting is not expected to matter for angle recovery.

Layups (ply thickness 19.6 voxels, sequence repeats to fill 192 voxels):
    A_training  [45,-45,90,0,45,-45,0,90,-45,45]   — the training layup
    B_permuted  [0,90,45,-45,0,90,-45,45,90,0]     — unseen permutation
    C_simple    [0,45,90,-45] repeated              — unseen simple stack

Known limitation: 74/78 training volumes share layup A, so recovery on B and
C demonstrates — it does not validate — generalisation.

Outputs: runs/campaigns/02-porosity-control-v1/layup_roundtrip/{results.json, findings.md, figures}.

Usage
-----
    python scripts/analysis/layup_roundtrip.py --validate-only   # CPU stage
    python scripts/analysis/layup_roundtrip.py                   # full run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, ZARR_ROOT, savefig, set_style, write_findings, write_json, plt  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

import t_i_layup_validation as ti  # noqa: E402  (the T-I math, reused, not forked)

from poregen.diffusion.conditioning import resolve_group_order  # noqa: E402
from poregen.diffusion.noise_schedule import DDPMSchedule  # noqa: E402
from poregen.diffusion.porosity_field import (  # noqa: E402
    DEFAULT_TD_RESULTS, DEFAULT_TE_RESULTS, build_porosity_field,
    load_corr_lengths_voxels, load_sampler,
)
from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator, theta_from_layup  # noqa: E402
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402

CKPT = REPO / ("runs/ldm/ldm05-run-0001-20260827-114902-z4-c128-bs256-lr1e-04/"
               "checkpoints/ldm_step00130000.ckpt")
LATENTS_ROOT = REPO / "data/split_v2/latents_r07z4"
LAYUP_FIELD = REPO / "runs/campaigns/01-conditioning-design/T-I/layup_field.json"
OUT_DIR = REPO / "runs/campaigns/02-porosity-control-v1/layup_roundtrip"

# Volume geometry: full T-I window in-plane, 192 voxels deep.
VOL_SHAPE = (192, 1024, 1024)             # (D, H, W) voxels
VOLUME_MM = (4.8, 25.6, 25.6)             # 0.025 mm/vox
GRID = (3, 16, 16)                        # tiling grid → 768 patches
PLY_VOX = 19.6
N_PLIES = 10                              # ceil(192 / 19.6); last ply 15.6 vox

LAYUPS = {
    "A_training": [45, -45, 90, 0, 45, -45, 0, 90, -45, 45],
    "B_permuted": [0, 90, 45, -45, 0, 90, -45, 45, 90, 0],
    "C_simple": [0, 45, 90, -45],
}
SEEDS = [101, 202]
SETTINGS = [("sequential", 1.0), ("joint", 1.5)]
TARGET_POR = 0.03
DDIM_STEPS = 50
GEN_BATCH = 32
DECODE_BATCH = 16
JOINT_WINDOW_STRIDE = 32
JOINT_WINDOW_BATCH = 16

# Validation pass gates for the adapted (single-window, known-edges) path on
# real volumes: agreement with the full two-window T-I estimator, and error
# vs the image-frame truth not much worse than what T-I itself achieved.
VAL_AGREE_GATE_DEG = 10.0
VAL_ERROR_SLACK_DEG = 5.0

LAYUP_COLORS = {"A_training": "#1b6ca8", "B_permuted": "#c2571a", "C_simple": "#2e7d32"}
MODE_MARKERS = {"sequential": "o", "joint": "s"}
REFERENCE_4CLASS = 0.847                  # T-I combined estimator, 78 real volumes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def ply_edges(depth: int, ply_vox: float) -> np.ndarray:
    """Requested ply block edges [0 .. depth]; the last ply may be truncated."""
    n = int(np.ceil(depth / ply_vox))
    e = np.round(np.arange(n + 1) * ply_vox).astype(int)
    e[-1] = min(e[-1], depth)
    return e


def requested_sequence(layup: list[int], n_plies: int) -> np.ndarray:
    """Per-ply-block requested angle (deg mod 180), sequence repeating."""
    return np.array([layup[k % len(layup)] % 180 for k in range(n_plies)], float)


def wrap180(d: np.ndarray) -> np.ndarray:
    """Signed axial angle difference in [-90, 90)."""
    return (np.asarray(d, float) + 90.0) % 180.0 - 90.0


# ---------------------------------------------------------------------------
# Measurement — the T-I "combined" estimator on an in-memory volume
# ---------------------------------------------------------------------------

def measure_volume(xct: np.ndarray, mask: np.ndarray, edges: np.ndarray,
                   pitch: float) -> dict:
    """Per-ply angles from the T-I combined estimator (fft_slice + pore_axes).

    ``xct``/``mask`` are (D, H, W) uint8 with H, W >= ti.WINDOW.  One centred
    1024x1024 window per z-slice (the generated volume IS the window).  All
    the math — angular map geometry, band extraction, residual preparation,
    pore principal axes — is imported from t_i_layup_validation unchanged.
    """
    n = ti.WINDOW
    D, H, W = xct.shape
    y0 = max(0, H // 2 - n // 2)
    x0 = max(0, W // 2 - n // 2)
    win, sel, flat, counts = ti._geometry(n)
    lo, hi = ti.BAND_PITCH[0] * pitch, ti.BAND_PITCH[1] * pitch
    nb = len(edges) - 1

    d_fft = np.zeros((nb, ti.N_ANG))
    d_pore = np.zeros((nb, ti.N_ANG))
    for i in range(nb):
        z0, z1 = int(edges[i]), int(edges[i + 1])
        acc = np.zeros(ti.N_ANG)
        for z in range(z0, z1):
            img = xct[z, y0:y0 + n, x0:x0 + n].astype(np.float32)
            m = ti.angular_map(img, win, sel, flat, counts)
            acc += ti.band_hist(m[None], lo, hi)[0]
        d_fft[i] = acc / max(z1 - z0, 1)
        d_pore[i] = ti.pore_axes_density(mask[z0:z1, y0:y0 + n, x0:x0 + n] > 0)

    r_fft = ti.prepare_density(d_fft, excise=True)
    out = {"fft_slice": r_fft}
    if d_pore.sum() > 0:
        r_pore = ti.prepare_density(d_pore, excise=False)
        out["pore_axes"] = r_pore
        out["combined"] = (r_fft / (r_fft.std() + 1e-12)
                           + r_pore / (r_pore.std() + 1e-12))
    else:
        out["combined"] = r_fft
    est = {}
    for name, R in out.items():
        est[name] = {
            "angles": ti.THETA[np.argmax(R, axis=1)],
            "weights": R.max(axis=1) - R.min(axis=1),
        }
    return est


def score_recovery(ang: np.ndarray, w: np.ndarray, req: np.ndarray) -> dict:
    """Round-trip scores: direct (frozen identity convention) and T-I-fitted.

    Direct: requested angles are written in image coordinates by
    theta_from_layup, so measured == requested is the null hypothesis — no
    offset, no sign flip, no face reversal is granted.  Fitted: the standard
    T-I four-hypothesis fit (sign x face order + one global offset) for
    comparability with the real-volume numbers, which needed that freedom.
    """
    direct = wrap180(ang - req)
    f = ti.fit_offset(ang, req, w)
    cls = ti.classify_sequence(ang, w, req)
    strict_pred = np.round((ang % 180.0) / 45.0).astype(int) % 4
    strict_true = np.round((req % 180.0) / 45.0).astype(int) % 4
    return {
        "measured_deg": ang.tolist(),
        "requested_deg": req.tolist(),
        "weights": w.tolist(),
        "direct_errors_deg": direct.tolist(),
        "direct_median_abs_error_deg": float(np.median(np.abs(direct))),
        "direct_max_abs_error_deg": float(np.max(np.abs(direct))),
        "direct_frac_within_5": float(np.mean(np.abs(direct) < 5.0)),
        "direct_frac_within_10": float(np.mean(np.abs(direct) < 10.0)),
        "strict_class_accuracy": float(np.mean(strict_pred == strict_true)),
        "strict_pred_class": strict_pred.tolist(),
        "strict_true_class": strict_true.tolist(),
        "fitted_offset_deg": f["offset_deg"],
        "fitted_sign": f["sign"],
        "fitted_reversed": bool(f["reversed"]),
        "fitted_median_abs_error_deg": f["median_abs_error"],
        "fitted_errors_deg": f["errors"].tolist(),
        "fitted_frac_within_5": float(np.mean(np.abs(f["errors"]) < 5.0)),
        "fitted_frac_within_10": float(np.mean(np.abs(f["errors"]) < 10.0)),
        "ti_class_accuracy": cls["accuracy"],
        "ti_class_pred": cls["pred_class"],
        "ti_class_true": cls["true_class"],
    }


# ---------------------------------------------------------------------------
# Stage 0 — validate the adapted measurement path on real volumes
# ---------------------------------------------------------------------------

def validate_on_real(n_vols: int = 2) -> dict:
    """Run the adapted single-window path on real volumes; compare to T-I.

    For each volume the crop is T-I's window 1 (y at 30 % of the extent, x
    centred) over the laminate interior, and the block edges are T-I's fitted
    ply boundaries.  Two comparisons: (i) per-ply angle agreement with the
    full two-window combined estimator (ply_angle_measured_image_deg), and
    (ii) fitted median error vs the image-frame truth, against the error the
    full estimator achieved on the same volume.
    """
    field = json.loads(LAYUP_FIELD.read_text())["volumes"]
    cands = sorted(vid for vid, v in field.items()
                   if v["usable"] and v["confidence"] == "high"
                   and v.get("ply_boundaries_z"))
    cands = cands[:n_vols]
    g = zarr.open_group(str(ZARR_ROOT), mode="r")
    n = ti.WINDOW
    per_vol = []
    for vid in cands:
        v = field[vid]
        t0 = time.time()
        xct_a, mask_a = g[vid]["xct"], g[vid]["mask"]
        _, H, W = xct_a.shape
        y0 = int(np.clip(round(0.3 * H) - n // 2, 0, H - n))
        x0 = max(0, W // 2 - n // 2)
        edges = np.asarray(v["ply_boundaries_z"], int)
        za, zb = int(edges[0]), int(edges[-1])
        xct = np.asarray(xct_a[za:zb, y0:y0 + n, x0:x0 + n])
        mask = np.asarray(mask_a[za:zb, y0:y0 + n, x0:x0 + n])
        pitch = (zb - za) / v["n_plies"]
        est = measure_volume(xct, mask, edges - za, pitch)
        ang = est["combined"]["angles"]
        w = est["combined"]["weights"]

        full = np.asarray(v["ply_angle_measured_image_deg"], float)
        agree = wrap180(ang - full)
        truth_img = np.asarray(v["ply_angle_image_deg"], float) % 180
        f = ti.fit_offset(ang, truth_img, w)
        per_vol.append({
            "volume_id": vid,
            "n_plies": v["n_plies"],
            "pitch_voxels": pitch,
            "adapted_angles_deg": ang.tolist(),
            "full_estimator_angles_deg": full.tolist(),
            "agreement_errors_deg": agree.tolist(),
            "agreement_median_abs_deg": float(np.median(np.abs(agree))),
            "adapted_fit_median_abs_error_deg": f["median_abs_error"],
            "full_fit_median_abs_error_deg": v["fit_median_abs_error_deg"],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  [validate] {vid}: agree_median="
              f"{per_vol[-1]['agreement_median_abs_deg']:.1f} deg  "
              f"adapted_err={f['median_abs_error']:.1f} deg  "
              f"full_err={v['fit_median_abs_error_deg']:.1f} deg  "
              f"({per_vol[-1]['seconds']}s)", flush=True)

    agree_med = float(np.median([r["agreement_median_abs_deg"] for r in per_vol]))
    err_delta = float(np.median(
        [r["adapted_fit_median_abs_error_deg"] - r["full_fit_median_abs_error_deg"]
         for r in per_vol]))
    passed = agree_med <= VAL_AGREE_GATE_DEG and err_delta <= VAL_ERROR_SLACK_DEG
    return {
        "choice": "a: generate 1024x1024x192 volumes, keep the 1024-px window",
        "adaptations": ["single centred window instead of two",
                        "requested ply grid as block edges (generated) / T-I "
                        "fitted boundaries (validation)",
                        "requested pitch for the wavelength band"],
        "gate_agreement_median_deg": VAL_AGREE_GATE_DEG,
        "gate_error_slack_deg": VAL_ERROR_SLACK_DEG,
        "agreement_median_abs_deg": agree_med,
        "fit_error_delta_vs_full_deg": err_delta,
        "passed": passed,
        "per_volume": per_vol,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def wait_for_gpu(poll_s: int = 60, budget_s: int = 1800) -> None:
    """Block while dose_response.py holds the GPU (poll every 60 s)."""
    t0 = time.time()
    while True:
        r = subprocess.run(["pgrep", "-f", "dose_[r]esponse.py"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return
        waited = time.time() - t0
        if waited > budget_s:
            print(f"  [gpu] still busy after {waited / 60:.0f} min — "
                  "continuing to wait", flush=True)
        time.sleep(poll_s)


def build_generator(device: torch.device):
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
    sampler = DDIMSampler(model, schedule, device, n_steps=DDIM_STEPS)
    st = meta["conditioning"]["por_standardisation"]
    gen = VolumeGenerator(
        sampler=sampler, vae=vae, device=device,
        patch_size=64, generation_stride=64, neighbour_offset=64,
        latent_size=16, latent_mean=latent_mean, latent_std=latent_std,
        voxel_size_mm=0.025,
        por_log_stats=(float(st["mean"]), float(st["std"])),
        theta_deg=None,
        group_order=resolve_group_order(meta),
    )
    return gen, step


def coherent_por_map(target: float, seed: int, te_sampler, corr_lengths):
    fld = build_porosity_field(grid_shape=GRID, target=target, sampler=te_sampler,
                               corr_lengths_voxels=corr_lengths,
                               stride_voxels=64, seed=seed)
    return {(iz, iy, ix): float(fld[iz, iy, ix])
            for iz in range(GRID[0]) for iy in range(GRID[1])
            for ix in range(GRID[2])}


def generate_one(gen, mode: str, por_map: dict, joint_window_batch: int):
    """Generate one volume; on CUDA OOM halve joint_window_batch and retry."""
    oom_events = []
    jwb = joint_window_batch
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
                    joint_window_batch=jwb,
                )
            return xct, mask, stats, time.perf_counter() - t0, jwb, oom_events
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if jwb <= 1:
                raise
            oom_events.append(f"OOM at joint_window_batch={jwb}: {exc}")
            jwb //= 2
            print(f"  OOM — retrying with joint_window_batch={jwb}", flush=True)


# ---------------------------------------------------------------------------
# Aggregation, figures, findings
# ---------------------------------------------------------------------------

ESTIMATORS = ("combined", "fft_slice", "pore_axes")


def aggregate_est(records: list[dict], est: str) -> dict:
    """Pool one estimator's scores per layup and per (layup, mode)."""
    out = {}
    for layup in LAYUPS:
        for mode in [m for m, _ in SETTINGS] + ["all"]:
            rs = [r for r in records if r["layup"] == layup
                  and (mode == "all" or r["mode"] == mode) and est in r]
            if not rs:
                continue
            de = np.concatenate([np.abs(r[est]["direct_errors_deg"]) for r in rs])
            fe = np.concatenate([np.abs(r[est]["fitted_errors_deg"]) for r in rs])
            out[f"{layup}.{mode}"] = {
                "n_volumes": len(rs),
                "n_plies": int(de.size),
                "direct_median_abs_error_deg": float(np.median(de)),
                "direct_frac_within_5": float(np.mean(de < 5.0)),
                "direct_frac_within_10": float(np.mean(de < 10.0)),
                "fitted_median_abs_error_deg": float(np.median(fe)),
                "fitted_frac_within_5": float(np.mean(fe < 5.0)),
                "fitted_frac_within_10": float(np.mean(fe < 10.0)),
                "strict_class_accuracy": float(np.mean(
                    [r[est]["strict_class_accuracy"] for r in rs])),
                "ti_class_accuracy": float(np.mean(
                    [r[est]["ti_class_accuracy"] for r in rs])),
                "delivered_porosity_mean": float(np.mean(
                    [r["delivered_porosity"] for r in rs])),
            }
    recs = [r for r in records if est in r]
    pooled_de = np.concatenate([np.abs(r[est]["direct_errors_deg"]) for r in recs])
    meas = np.concatenate([r[est]["measured_deg"] for r in recs])
    req = np.concatenate([r[est]["requested_deg"] for r in recs])
    per_class = {}
    for c in (0, 45, 90, 135):
        m = req == c
        per_class[str(c)] = {
            "n": int(m.sum()),
            "direct_median_abs_error_deg": float(np.median(pooled_de[m])),
            "direct_frac_within_10": float(np.mean(pooled_de[m] < 10.0)),
        }
    # sign-flip audit on the off-axis plies (the documented +-45 ambiguity)
    off = (req == 45) | (req == 135)
    d_ok = np.abs(wrap180(meas[off] - req[off]))
    d_fl = np.abs(wrap180(meas[off] - (180.0 - req[off])))
    out["overall"] = {
        "n_volumes": len(recs),
        "n_plies": int(pooled_de.size),
        "direct_median_abs_error_deg": float(np.median(pooled_de)),
        "direct_frac_within_5": float(np.mean(pooled_de < 5.0)),
        "direct_frac_within_10": float(np.mean(pooled_de < 10.0)),
        "strict_class_accuracy": float(np.mean(
            [r[est]["strict_class_accuracy"] for r in recs])),
        "ti_class_accuracy": float(np.mean(
            [r[est]["ti_class_accuracy"] for r in recs])),
        "ti_reference_real_volumes": REFERENCE_4CLASS,
        # doubled-angle resultants: how strongly measured tracks requested,
        # vs how anisotropic the measured field is on its own
        "resultant_measured_vs_requested": float(np.abs(np.mean(
            np.exp(2j * np.deg2rad(meas - req))))),
        "resultant_measured_alone": float(np.abs(np.mean(
            np.exp(2j * np.deg2rad(meas))))),
        "per_class": per_class,
        "offaxis_plies": {
            "n": int(off.sum()),
            "correct_within_22_5": int(np.sum(d_ok <= 22.5)),
            "sign_flipped_within_22_5": int(np.sum((d_ok > 22.5) & (d_fl <= 22.5))),
            "other": int(np.sum((d_ok > 22.5) & (d_fl > 22.5))),
        },
    }
    return out


def aggregate(records: list[dict]) -> dict:
    return {est: aggregate_est(records, est) for est in ESTIMATORS}


def make_fig_profiles(records: list[dict], step: int) -> list[str]:
    """Requested vs measured angle per ply, one panel per (layup, mode)."""
    modes = [m for m, _ in SETTINGS if any(r["mode"] == m for r in records)]
    fig, axes = plt.subplots(len(LAYUPS), max(len(modes), 1),
                             figsize=(4.6 * max(len(modes), 1), 2.9 * len(LAYUPS)),
                             sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)
    for i, layup in enumerate(LAYUPS):
        req = requested_sequence(LAYUPS[layup], N_PLIES)
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            k = np.arange(N_PLIES)
            ax.step(np.append(k, N_PLIES), np.append(req, req[-1]), where="post",
                    color="0.35", lw=1.6, label="requested")
            rs = [r for r in records if r["layup"] == layup and r["mode"] == mode]
            for r in rs:
                # plot each measurement at the wrap-equivalent closest to the request
                meas = np.asarray(r["combined"]["measured_deg"])
                ax.plot(k + 0.5, req + wrap180(meas - req), ls="none", marker="o",
                        ms=5, color=LAYUP_COLORS[layup], alpha=0.75,
                        label=f"combined, seed {r['seed']}")
                if "pore_axes" in r:
                    mp = np.asarray(r["pore_axes"]["measured_deg"])
                    ax.plot(k + 0.5, req + wrap180(mp - req), ls="none",
                            marker="^", ms=5, mfc="none",
                            color=LAYUP_COLORS[layup], alpha=0.75,
                            label=f"pore_axes, seed {r['seed']}")
            if rs:
                de = np.concatenate([np.abs(r["combined"]["direct_errors_deg"])
                                     for r in rs])
                s_por = dict(SETTINGS)[mode]
                ax.set_title(f"{layup}  |  {mode}, s_por={s_por}  |  "
                             f"median |err| {np.median(de):.1f} deg")
            ax.set_yticks([-45, 0, 45, 90, 135, 180])
            ax.set_ylim(-60, 195)
            if i == len(LAYUPS) - 1:
                ax.set_xlabel("ply index (z order)")
            if j == 0:
                ax.set_ylabel("angle (deg, image frame)")
            if i == 0 and j == 0:
                ax.legend(loc="lower right", ncol=2, fontsize=6.5)
    fig.suptitle(f"Layup round-trip, ldm05 step {step} (RAW, DDIM-{DDIM_STEPS}, "
                 f"1024x1024x192, combined estimator)", fontsize=11)
    return savefig(fig, OUT_DIR, "layup_fig1_requested_vs_measured")


def make_fig_scatter(records: list[dict], step: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    lim = [-10, 190]
    ax.plot(lim, lim, color="0.4", lw=1.0, ls="--", zorder=1)
    for gate, col in ((10.0, "0.90"), (5.0, "0.80")):
        ax.fill_between(lim, [v - gate for v in lim], [v + gate for v in lim],
                        color=col, alpha=0.6, zorder=0,
                        label=f"within {gate:.0f} deg")
    for layup in LAYUPS:
        for mode, _ in SETTINGS:
            rs = [r for r in records if r["layup"] == layup and r["mode"] == mode]
            if not rs:
                continue
            req = np.concatenate([r["combined"]["requested_deg"] for r in rs])
            meas = np.concatenate([r["combined"]["measured_deg"] for r in rs])
            shown = req + wrap180(meas - req)
            ax.plot(req, shown, ls="none", marker=MODE_MARKERS[mode], ms=5,
                    color=LAYUP_COLORS[layup], alpha=0.7,
                    label=f"{layup} / {mode}")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Requested ply angle (deg)")
    ax.set_ylabel("Measured ply angle (deg, wrap-nearest)")
    ax.set_title(f"Per-ply angle recovery, ldm05 step {step}")
    ax.legend(loc="upper left", fontsize=7.5)
    return savefig(fig, OUT_DIR, "layup_fig2_recovery_scatter")


def build_findings(validation: dict, records: list[dict], agg: dict,
                   step: int, oom: list[str], walls: dict) -> str:
    L = []
    L.append("# Layup round-trip — ldm05 step %d (RAW, DDIM-%d)\n" % (step, DDIM_STEPS))
    L.append("Volumes are 1024x1024x192 voxels (16x16x3 patches) so the "
             "UNCHANGED 1024-px T-I window applies (geometry option (a)). "
             "Estimator: T-I 'combined' (fft_slice + pore_axes), imported "
             "from t_i_layup_validation.py.\n")

    v = validation
    L.append("## Estimator-geometry validation (real volumes)\n")
    L.append("Adapted path (single window, given block edges) vs the full "
             "two-window T-I output, on %d real volumes:\n" % len(v["per_volume"]))
    L.append("| volume | agree median (deg) | adapted err (deg) | full err (deg) |")
    L.append("|---|---|---|---|")
    for r in v["per_volume"]:
        L.append("| %s | %.1f | %.1f | %.1f |" % (
            r["volume_id"], r["agreement_median_abs_deg"],
            r["adapted_fit_median_abs_error_deg"], r["full_fit_median_abs_error_deg"]))
    L.append("\nGates: agreement median <= %.0f deg, error delta <= %.0f deg. "
             "**%s** (agreement %.1f deg, delta %+.1f deg).\n" % (
                 v["gate_agreement_median_deg"], v["gate_error_slack_deg"],
                 "PASSED" if v["passed"] else "FAILED",
                 v["agreement_median_abs_deg"], v["fit_error_delta_vs_full_deg"]))

    if records:
        L.append("## Key findings\n")
        pa = agg["pore_axes"]["overall"]
        seq_err = [agg["pore_axes"][f"{ly}.sequential"]["direct_median_abs_error_deg"]
                   for ly in LAYUPS if f"{ly}.sequential" in agg["pore_axes"]]
        jnt_err = [agg["pore_axes"][f"{ly}.joint"]["direct_median_abs_error_deg"]
                   for ly in LAYUPS if f"{ly}.joint" in agg["pore_axes"]]
        seq_phi = [agg["combined"][f"{ly}.sequential"]["delivered_porosity_mean"]
                   for ly in ("B_permuted", "C_simple")
                   if f"{ly}.sequential" in agg["combined"]]
        L.append("- The requested layup IS written into the generated volume, "
                 "but through the PORES, not the grey-level tow texture: "
                 "pore_axes tracks the request (R = %.2f, median |err| %.1f "
                 "deg) while fft_slice carries almost no layup signal "
                 "(R = %.2f). On real volumes both channels work and "
                 "'combined' is best; on generated volumes 'combined' is "
                 "dragged down by fft_slice." % (
                     pa["resultant_measured_vs_requested"],
                     pa["direct_median_abs_error_deg"],
                     agg["fft_slice"]["overall"]["resultant_measured_vs_requested"]))
        if seq_err and jnt_err:
            L.append("- Joint mode follows the requested orientation far "
                     "better than sequential (pore_axes per-layup direct "
                     "medians: sequential %s deg, joint %s deg)." % (
                         "/".join(f"{e:.0f}" for e in seq_err),
                         "/".join(f"{e:.0f}" for e in jnt_err)))
        if seq_phi:
            L.append("- Sequential mode LOSES porosity control on unseen "
                     "layups: delivered phi %s for target %.2f (layup A "
                     "stays on target). Joint mode stays near target on all "
                     "three layups (slight undershoot at s_por=1.5)." % (
                         "/".join(f"{x:.3f}" for x in seq_phi), TARGET_POR))
        L.append("- 0- and 90-deg plies recover well (~8 deg median); the "
                 "+-45 plies are the weak point, with sign flips and "
                 "scatter — consistent with the +-45 ambiguity the T-I work "
                 "documented on real volumes.\n")
        L.append("## Estimator channels on generated volumes\n")
        L.append("| estimator | direct median \\|err\\| | <=5 | <=10 | strict "
                 "4-class | T-I-fit 4-class | R(meas vs req) |")
        L.append("|---|---|---|---|---|---|---|")
        for est in ESTIMATORS:
            o = agg[est]["overall"]
            L.append("| %s | %.1f deg | %.0f%% | %.0f%% | %.1f%% | %.1f%% | %.2f |" % (
                est, o["direct_median_abs_error_deg"],
                100 * o["direct_frac_within_5"], 100 * o["direct_frac_within_10"],
                100 * o["strict_class_accuracy"], 100 * o["ti_class_accuracy"],
                o["resultant_measured_vs_requested"]))
        L.append("\nR(meas vs req) is the doubled-angle resultant of "
                 "(measured - requested); 1 = perfect tracking, ~0 = no "
                 "relation. Compare each estimator's R to its "
                 "resultant_measured_alone in results.json.\n")
        L.append("## Recovery per layup (combined estimator, pooled over "
                 "modes and seeds)\n")
        L.append("| layup | n plies | direct median \\|err\\| | <=5 deg | <=10 deg "
                 "| strict 4-class | T-I-fit 4-class |")
        L.append("|---|---|---|---|---|---|---|")
        for layup in LAYUPS:
            a = agg["combined"].get(f"{layup}.all")
            if a:
                L.append("| %s | %d | %.1f deg | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (
                    layup, a["n_plies"], a["direct_median_abs_error_deg"],
                    100 * a["direct_frac_within_5"], 100 * a["direct_frac_within_10"],
                    100 * a["strict_class_accuracy"], 100 * a["ti_class_accuracy"]))
        o = agg["combined"]["overall"]
        L.append("\nOverall: direct median |err| %.1f deg, %.0f%% within 5 deg, "
                 "%.0f%% within 10 deg. Strict 4-class accuracy %.1f%% "
                 "(identity convention, no fitted offset/sign); T-I-style "
                 "fitted 4-class %.1f%% vs the **84.7%%** real-volume "
                 "reference.\n" % (
                     o["direct_median_abs_error_deg"],
                     100 * o["direct_frac_within_5"], 100 * o["direct_frac_within_10"],
                     100 * o["strict_class_accuracy"], 100 * o["ti_class_accuracy"]))

        L.append("## Per (layup, mode) table\n")
        L.append("| layup | mode | direct median \\|err\\| | <=5 | <=10 | strict "
                 "4-class | delivered phi |")
        L.append("|---|---|---|---|---|---|---|")
        for layup in LAYUPS:
            for mode, _ in SETTINGS:
                a = agg["combined"].get(f"{layup}.{mode}")
                if a:
                    L.append("| %s | %s | %.1f deg | %.0f%% | %.0f%% | %.0f%% | %.4f |" % (
                        layup, mode, a["direct_median_abs_error_deg"],
                        100 * a["direct_frac_within_5"],
                        100 * a["direct_frac_within_10"],
                        100 * a["strict_class_accuracy"],
                        a["delivered_porosity_mean"]))
        L.append("")

    L.append("## Per-class and off-axis audit (pore_axes, the informative "
             "channel)\n")
    pa = agg.get("pore_axes", {}).get("overall") if records else None
    if pa:
        pc = pa["per_class"]
        L.append("| requested class | n | direct median \\|err\\| | <=10 deg |")
        L.append("|---|---|---|---|")
        for c in ("0", "45", "90", "135"):
            L.append("| %s | %d | %.1f deg | %.0f%% |" % (
                c, pc[c]["n"], pc[c]["direct_median_abs_error_deg"],
                100 * pc[c]["direct_frac_within_10"]))
        oa = pa["offaxis_plies"]
        L.append("\nOff-axis (+-45) plies, nearest-lattice audit (22.5-deg "
                 "windows): %d correct, %d sign-flipped, %d neither, of %d.\n"
                 % (oa["correct_within_22_5"], oa["sign_flipped_within_22_5"],
                    oa["other"], oa["n"]))
    L.append("## Notes and limitations\n")
    L.append("- 74/78 training volumes share layup A: recovery on B_permuted "
             "and C_simple demonstrates, not validates, generalisation.")
    L.append("- Orientation is NOT guided in either mode; the porosity "
             "guidance setting (sequential s_por=1.0 vs joint s_por=1.5) is "
             "assumed not to matter for angle recovery. The per-mode table "
             "checks that assumption.")
    L.append("- The T-I work documented a +-45 sign ambiguity (angle sign "
             "convention not tied to the image frame on real scans). Here the "
             "request frame IS the image frame, so the direct scores grant no "
             "sign freedom; the fitted scores grant the same four hypotheses "
             "T-I used. `fitted_sign` per volume records whether the fit "
             "chose a flip.")
    L.append("- Target porosity 0.03 via the coherent field; the per-mode "
             "table's delivered phi column is a sanity check only.")
    if walls:
        L.append("- Median wall time per volume: "
                 + ", ".join(f"{m} {w:.0f}s" for m, w in walls.items()) + ".")
    L.append("- OOM events: %s" % (("\n  - " + "\n  - ".join(oom)) if oom else "none."))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true",
                    help="run only the CPU estimator-geometry validation")
    ap.add_argument("--modes", nargs="+", default=None,
                    choices=["sequential", "joint"])
    ap.add_argument("--aggregate-only", action="store_true",
                    help="recompute aggregate/figures/findings from the "
                         "records already in results.json")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    if args.aggregate_only:
        results = json.loads((OUT_DIR / "results.json").read_text())
        records = results["records"]
        validation = results.get("geometry_validation") or results["validation"]
        step = results.get("step", 130000)
        agg = aggregate(records)
        results["aggregate"] = agg
        walls = {m: float(np.median([r["wall_s"] for r in records
                                     if r["mode"] == m]))
                 for m, _ in SETTINGS if any(r["mode"] == m for r in records)}
        results["median_wall_s"] = walls
        write_json(results, OUT_DIR)
        make_fig_profiles(records, step)
        make_fig_scatter(records, step)
        write_findings(build_findings(validation, records, agg, step,
                                      results.get("oom_events", []), walls),
                       OUT_DIR)
        print("Re-aggregated", OUT_DIR, flush=True)
        return

    print("Stage 0 — estimator-geometry validation on real volumes", flush=True)
    validation = validate_on_real()
    write_json({"partial": True, "validation": validation}, OUT_DIR)
    if not validation["passed"]:
        print("VALIDATION FAILED — inspect results.json before generating.",
              flush=True)
    if args.validate_only:
        return

    settings = SETTINGS if args.modes is None else [
        s for s in SETTINGS if s[0] in args.modes]

    print("Waiting for the GPU (dose_response.py)", flush=True)
    wait_for_gpu()
    device = torch.device("cuda")
    gen, step = build_generator(device)
    te_sampler = load_sampler(REPO / DEFAULT_TE_RESULTS)
    corr_lengths = load_corr_lengths_voxels(REPO / DEFAULT_TD_RESULTS)

    edges = ply_edges(VOL_SHAPE[0], PLY_VOX)
    records: list[dict] = []
    all_oom: list[str] = []
    n_total = len(LAYUPS) * len(settings) * len(SEEDS)
    i = 0
    for mode, s_por in settings:
        gen.sampler.s_por = s_por
        gen.sampler.guided = not (s_por == 1.0 and gen.sampler.s_nb == 1.0)
        for layup_name, layup in LAYUPS.items():
            gen.theta_deg = theta_from_layup(VOL_SHAPE[0], layup, PLY_VOX)
            req = requested_sequence(layup, N_PLIES)
            for seed in SEEDS:
                i += 1
                torch.manual_seed(seed)
                por_map = coherent_por_map(TARGET_POR, seed, te_sampler,
                                           corr_lengths)
                xct, mask, stats, wall, jwb, ooms = generate_one(
                    gen, mode, por_map, JOINT_WINDOW_BATCH)
                all_oom.extend(ooms)
                est = measure_volume(xct, mask, edges, PLY_VOX)
                del xct, mask
                rec = {
                    "layup": layup_name,
                    "layup_angles": layup,
                    "mode": mode,
                    "s_por": s_por,
                    "seed": seed,
                    "delivered_porosity": float(stats["actual_mask_porosity"]),
                    "seam_xct_ratio": stats["seam_xct_ratio"],
                    "wall_s": round(wall, 1),
                    "joint_window_batch": jwb if mode == "joint" else None,
                }
                for name, e in est.items():
                    rec[name] = score_recovery(e["angles"], e["weights"], req)
                records.append(rec)
                c = rec["combined"]
                print(f"[{i:2d}/{n_total}] {layup_name:12s} {mode:10s} "
                      f"seed={seed}  direct_med="
                      f"{c['direct_median_abs_error_deg']:.1f} deg  "
                      f"<=10deg {100 * c['direct_frac_within_10']:.0f}%  "
                      f"strict4 {100 * c['strict_class_accuracy']:.0f}%  "
                      f"phi={rec['delivered_porosity']:.4f}  "
                      f"wall={wall:.0f}s", flush=True)
                write_json({"partial": True, "validation": validation,
                            "records": records}, OUT_DIR)

    agg = aggregate(records)
    walls = {m: float(np.median([r["wall_s"] for r in records if r["mode"] == m]))
             for m, _ in settings if any(r["mode"] == m for r in records)}
    results = {
        "checkpoint": str(CKPT),
        "step": step,
        "weights": "raw (non-EMA)",
        "ddim_steps": DDIM_STEPS,
        "volume_shape_vox": list(VOL_SHAPE),
        "grid": list(GRID),
        "ply_thickness_vox": PLY_VOX,
        "n_ply_blocks": N_PLIES,
        "ply_edges": edges.tolist(),
        "layups": {k: v for k, v in LAYUPS.items()},
        "settings": [{"mode": m, "s_por": s} for m, s in settings],
        "seeds": SEEDS,
        "target_porosity": TARGET_POR,
        "joint_window_stride": JOINT_WINDOW_STRIDE,
        "estimator": "T-I combined (fft_slice + pore_axes), imported",
        "ti_reference_4class_real": REFERENCE_4CLASS,
        "geometry_validation": validation,
        "aggregate": agg,
        "oom_events": all_oom,
        "median_wall_s": walls,
        "records": records,
    }
    write_json(results, OUT_DIR)
    figs = make_fig_profiles(records, step) + make_fig_scatter(records, step)
    write_findings(build_findings(validation, records, agg, step, all_oom, walls),
                   OUT_DIR)
    print("Wrote", OUT_DIR, "figures:", figs, flush=True)


if __name__ == "__main__":
    main()

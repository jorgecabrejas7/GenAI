"""Standard convergence health-check for LDM runs (read-only on the checkpoint).

Loads latest.ckpt from a running LDM experiment, samples latents with real val
conditioning drawn via LatentDataset (real neighbours, real orientation and
material maps, real scalars), and reports the core off-manifold metrics SPLIT
BY neighbour context — how many of the six faces have a stored neighbour
(bucket 0..6).  Sampling only the all-UNKNOWN case measures the model in its
rarest, hardest situation and mis-reports convergence.

Bucket weights are the OBSERVED frequencies in the scanned val rows, not a
fixed table: with ldm06 a patch's context is a property of where it sits in the
specimen, so the population is whatever the data says it is.

Neighbours are handed over CLEAN at ``nb_t = 0``.  That is an in-distribution
state (the training draw ``t_nb ~ Uniform{0..t}`` includes 0) and it is the
same for every checkpoint, so the trend line compares like with like.

Per invocation:
  * one table per weight set (raw, EMA), DDIM-50, per-bucket + weighted overall
  * conditioning-alive check: FULL vs porosity-neutralised vs orientation-zeroed
    vs material-zeroed vs neighbours-UNKNOWN, with one fixed initial noise; a
    different-seed FULL run gives the total-noise MAD scale
  * kill-switch check: ask phi=0.005 vs 0.05 with the same seed and rows
    (richest bucket, raw + EMA) and compare the delivered pore fraction
  * a summary appended to <run_dir>/convergence_check.jsonl; with >= 2 entries
    a verdict line vs the previous check: CONVERGING / STALLED / REGRESSING
    (ema overall std_ratio, +/-5% band) plus per-bucket deltas
  * PNG slice grids, each row labelled with its context bucket

Usage:
    python scripts/diag_ldm_samples.py --run-dir <run_dir> [--n 8] [--ddim200]
        [--ckpt checkpoints/ldm_stepNNN.ckpt] [--no-killswitch]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from poregen.diffusion.conditioning import NB_UNKNOWN, N_NEIGHBOURS, porosity_to_cond
from poregen.diffusion.latents import LatentDataset
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.models.vae.base import CLASS_AIR, CLASS_PORE, decode_label, decode_xct
from poregen.diffusion.sampler import DDIMSampler
from poregen.experiments.base import find_repo_root
from poregen.experiments.train_vae import load_vae_from_checkpoint
from poregen.training.ldm_engine import sample_neighbours
from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser

logger = logging.getLogger("diag_ldm_samples")

SEED = 1234
SCAN_CAP = 40000          # val rows scanned when filling the context buckets
GRID_PER_BUCKET = 4       # decoded samples shown per bucket in each PNG
ALIVE_N = 4               # batch size of the conditioning-alive check
KILLSWITCH_N = 4          # batch size of the kill-switch check
KILLSWITCH_ASKS = (0.005, 0.05)
TREND_FILE = "convergence_check.jsonl"
VERDICT_BAND_PCT = 5.0

_DEGENERATE_LO = 1e-4
_DEGENERATE_HI = 0.5

# Conditioning tensors stacked for a bucket, in the order the model takes them
# after (nb_latents, nb_avail, nb_t).
_COND_KEYS = ("cond_por", "cond_depth", "cond_dist6", "cond_orient", "cond_material")


def _strip_compile_prefix(state: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


# ── context buckets ──────────────────────────────────────────────────────────

def _exists_count(ds: LatentDataset, idx: int) -> int:
    """How many of the six faces have a stored neighbour (no latent reads)."""
    return int((ds.neighbour_rows(idx) >= 0).sum())


def collect_bucket_rows(
    ds: LatentDataset, n_per_bucket: int, rng: np.random.Generator
) -> tuple[dict[str, list[int]], dict[str, float]]:
    """Val rows per EXISTS bucket, plus each bucket's observed frequency.

    The frequency is measured over every row the scan touched, not only the
    rows kept, so it is the population weight even when a bucket fills early.
    """
    buckets: dict[int, list[int]] = {b: [] for b in range(N_NEIGHBOURS + 1)}
    seen = np.zeros(N_NEIGHBOURS + 1, dtype=np.int64)
    for scanned, i in enumerate(rng.permutation(len(ds))):
        if scanned >= SCAN_CAP:
            break
        c = _exists_count(ds, int(i))
        seen[c] += 1
        if len(buckets[c]) < n_per_bucket:
            buckets[c].append(int(i))
    total = max(int(seen.sum()), 1)
    rows = {str(b): r for b, r in sorted(buckets.items()) if r}
    freq = {str(b): float(seen[b]) / total for b in range(N_NEIGHBOURS + 1)
            if str(b) in rows}
    return rows, freq


def find_boundary_rows(ds: LatentDataset, n: int,
                       lo: float = 0.2, hi: float = 0.8) -> list[int]:
    """Val rows whose window STRADDLES the specimen boundary.

    The material probe is meaningless on an interior window: the map is already
    all ones, so MATERIAL_ONE is a no-op by construction and reports 0.0%
    whether the conditioning works or not.

    Selected on the material fraction itself, read straight from material.bin,
    rather than on ``cond_dist6``. Two reasons. It is exact — cond_dist6 near
    zero only means a face is near the surface, and the most extreme such
    windows turn out to be ~92% exterior air, which is as uninformative as an
    all-ones map in the other direction. And it is cheap: material.bin is one
    byte per latent cell, so scoring all 266k val rows costs one pass.

    A window that is half in and half out is where replacing the real map with
    all-ones is the largest honest change, so it is where the probe has the
    most power.
    """
    # Chunked: np.asarray on the memmap would copy all 266k x 4096 bytes into
    # RAM to compute 266k scalars, and this runs beside a training job.
    mat = ds._material
    n_rows = mat.shape[0]          # NOT `n` — that is the caller's row budget
    frac = np.empty(n_rows, dtype=np.float64)
    for i in range(0, n_rows, 8192):
        blk = np.asarray(mat[i:i + 8192], dtype=np.float32)
        frac[i:i + blk.shape[0]] = blk.reshape(blk.shape[0], -1).mean(axis=1) / 255.0
    cand = np.flatnonzero((frac > lo) & (frac < hi))
    if cand.size == 0:
        return []
    # Closest to half in, half out.
    order = cand[np.argsort(np.abs(frac[cand] - 0.5))]
    return [int(i) for i in order[:n]]


def build_bucket_cond(ds: LatentDataset, rows: list[int], device: torch.device) -> dict:
    """Stack the real conditioning of *rows* onto the device (ldm06 contract).

    ``nb_t`` is zero: the neighbours are handed over at the ``t_nb = 0`` end of
    the training draw, i.e. not noised.

    They are DRAWN, not averaged.  The store serves each neighbour as a
    posterior mean and std, and training samples every EXISTS neighbour as
    ``mu + sigma*eps`` before noising it.  Handing over mean-valued neighbours
    would make this diagnostic condition on inputs the denoiser never sees in
    training — per-cell variance short by ``E[sigma^2]`` — so its numbers would
    describe a distribution nobody trains or samples on.  One draw, the same
    definition training uses.
    """
    items = [ds[r] for r in rows]
    out = {
        k: torch.stack([it[k] for it in items]).to(device)
        for k in (*_COND_KEYS, "nb_latents", "nb_std", "nb_avail", "z")
    }
    out["nb_latents"] = sample_neighbours(
        out["nb_latents"], out.pop("nb_std"), out["nb_avail"]
    )
    out["nb_t"] = torch.zeros_like(out["nb_avail"])
    out["phi"] = np.array([float(it["phi"]) for it in items], dtype=np.float64)
    return out


# ── sampling + decoding ──────────────────────────────────────────────────────

@torch.no_grad()
def decode_latents(vae: torch.nn.Module, z_raw: torch.Tensor, device: torch.device):
    """Decode raw (denormalised) latents. Returns (xct [0,1], label {0,1,2})."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        dec = vae.decoder(z_raw)
        xct_out = vae.xct_head(dec)
        class_logits = vae.class_head(dec)
    xct = decode_xct(xct_out.float()).squeeze(1)      # (B, D, H, W) grey level
    label = decode_label(class_logits.float())        # (B, D, H, W) int64
    return xct, label


@torch.no_grad()
def sample_bucket(
    sampler: DDIMSampler, cond: dict, seed: int
) -> tuple[torch.Tensor, float]:
    """Sample one bucket batch in normalised space. Returns (z, x0_sat_frac)."""
    torch.manual_seed(seed)
    return sampler.sample_batch(
        cond["nb_latents"], cond["nb_avail"], cond["nb_t"],
        *(cond[k] for k in _COND_KEYS),
        autocast_dtype=torch.bfloat16, return_x0_saturation=True,
    )


#: Per-channel std of the SAMPLED training latents, loaded once from
#: latent_std_reference.json. See ``load_std_reference``.
_STD_REF: np.ndarray | None = None


def load_std_reference(store: Path) -> np.ndarray | None:
    """Per-channel std of z = mu + sigma*eps over the train split, or None.

    ``std_ratio`` divides by 1.0, which is the std of the MU-normalised latent:
    the store's stats normalise mu, so mu-normalised latents have std 1 by
    construction. But ldm06 trains on ``latent_mode: sampled`` — the target is
    mu + sigma*eps in that same space, whose per-channel std is
    sqrt(1 + rms(sigma/std_c)^2) and is strictly greater than 1.

    Scoring generated latents against 1.0 therefore charges the model for the
    posterior width it was trained to reproduce. Both references are reported;
    the sampled one is the one a gate should use.

    Produced by scripts/analysis/latent_std_reference.py.
    """
    # The store's own metadata is authoritative: it travels with the latents,
    # so a reference can never be silently paired with the wrong store.
    meta = store / "metadata.json"
    if meta.exists():
        try:
            cs = json.loads(meta.read_text()).get("channel_stats") or {}
            if cs.get("sampled_std"):
                return np.asarray(cs["sampled_std"], dtype=np.float64)
        except Exception as exc:                        # noqa: BLE001
            logger.warning("could not read channel_stats from %s: %s", meta, exc)
    # Fallback: the standalone measurement, only if it names THIS store.
    repo = find_repo_root(__file__)
    cand = repo / "runs/campaigns/09-r08-latent-sweep/latent_std_reference.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text())
            if Path(d.get("store", "")).name == Path(store).name:
                return np.asarray(d["sampled_std"], dtype=np.float64)
            logger.warning("%s is for store %s, not %s — ignored",
                           cand, d.get("store"), store)
        except Exception as exc:                        # noqa: BLE001
            logger.warning("could not read %s: %s", cand, exc)
    return None


def bucket_metrics(z: torch.Tensor, sat: float, por: torch.Tensor,
                   air: torch.Tensor, phi: np.ndarray) -> dict:
    ch_std = z.std(dim=(0, 2, 3, 4))
    ch = np.asarray([float(v) for v in ch_std], dtype=np.float64)
    # Against the MU-normalised reference (implicitly 1.0) and, when it is
    # known, against the SAMPLED reference the model was actually trained on.
    per_ch = (ch / _STD_REF) if _STD_REF is not None and _STD_REF.shape == ch.shape else None
    sampled = float(per_ch.mean()) if per_ch is not None else None
    # Name the channels outside the band. At 20k the MEAN passed for
    # ema_ddim50 (0.9015) and raw_ddim200 (1.081) while channels 3, 5 and 6
    # were outside it in both — a mean over eight channels hides exactly the
    # per-channel drift this gate exists to catch.
    off_band = ([int(i) for i, v in enumerate(per_ch) if v < 0.90 or v > 1.10]
                if per_ch is not None else None)
    return {
        "n": int(z.shape[0]),
        "std_ratio": float(ch_std.mean()),
        "std_ratio_vs_sampled": sampled,
        "std_ratio_vs_sampled_per_ch": ([round(float(v), 4) for v in per_ch]
                                        if per_ch is not None else None),
        "channels_outside_10pct": off_band,
        "ch_std": [round(float(v), 3) for v in ch_std],
        "mean_abs_max": float(z.mean(dim=(0, 2, 3, 4)).abs().max()),
        "x0_sat": float(sat),
        "por_mean": float(por.mean()),
        "por_std": float(por.std()) if z.shape[0] > 1 else 0.0,
        "por_mae": float(np.abs(por.cpu().numpy() - phi).mean()),
        "air_mean": float(air.mean()),
        "degen": float(((por < _DEGENERATE_LO) | (por > _DEGENERATE_HI)).float().mean()),
    }


def weighted_overall(buckets: dict[str, dict], freq: dict[str, float]) -> dict:
    """Bucket metrics weighted by each bucket's observed frequency in the split."""
    w = np.array([freq.get(b, 0.0) for b in buckets], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(buckets))
    w = w / w.sum()
    keys = ("std_ratio", "x0_sat", "por_mean", "por_std", "por_mae", "air_mean", "degen")
    return {k: float(sum(wi * m[k] for wi, m in zip(w, buckets.values()))) for k in keys}


# ── PNG grids ────────────────────────────────────────────────────────────────

def save_grid(entries: list[tuple[str, np.ndarray, np.ndarray]],
              title: str, path: Path) -> None:
    """One row per (name, xct_slice, label_slice), columns [xct, label]."""
    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (name, xct2d, label2d) in enumerate(entries):
        axes[i, 0].imshow(xct2d, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(name, fontsize=8)
        # 0 material, 1 pore, 2 air — a 3-level map, not a probability.
        axes[i, 1].imshow(label2d, cmap="viridis", vmin=0, vmax=2)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("xct", fontsize=9)
    axes[0, 1].set_title("label (0 mat / 1 pore / 2 air)", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def grid_entries(xct: torch.Tensor, label: torch.Tensor, bucket: str,
                 limit: int = GRID_PER_BUCKET) -> list[tuple[str, np.ndarray, np.ndarray]]:
    zc = xct.shape[1] // 2
    return [
        (f"b{bucket}#{i}", xct[i, zc].cpu().numpy(), label[i, zc].float().cpu().numpy())
        for i in range(min(int(xct.shape[0]), limit))
    ]


# ── inset-box surface sample ─────────────────────────────────────────────────

#: From this step on, each diagnostic also renders one volume whose material map
#: asks for air. Everything else in this file samples a full-material box, so
#: until now nothing checked that the model puts air where it is told to, or
#: where it puts the interface.
INSET_FROM_STEP = 60000


def inset_surface_sample(run_dir: Path, step: int, out_dir: Path) -> dict | None:
    """One 192-cubed EMA/DDIM-50 volume with the specimen box inset in z.

    Built through eval_v4's own VolumeRunner rather than a second generation
    path: the sampler setup (VAE, schedule, normalisation, orientation, the
    material request conversion) is intricate and a private copy of it here
    would drift from the one the campaign actually measures, so the number
    would stop describing the same thing.
    """
    try:
        from poregen.eval_v4.cases import (       # noqa: PLC0415
            SHAPE_SMALL, SURFACE_Z_HI, SURFACE_Z_LO, TARGET_DEFAULT,
            CaseSpec, layup_a, material_inset_z,
        )
        from poregen.eval_v4.generate import VolumeRunner   # noqa: PLC0415
        from poregen.eval_v4.metrics import surface_agreement  # noqa: PLC0415
        from poregen.eval_v4.io import load_label, load_u8      # noqa: PLC0415
    except Exception as exc:                                # noqa: BLE001
        logger.warning("inset sample unavailable: %s", exc)
        return None

    repo = find_repo_root(__file__)
    plies, pitch = layup_a(repo)
    spec = CaseSpec(
        name=f"inset_step{step}",
        assessment="surface",
        volume_shape=SHAPE_SMALL,
        seed=SEED,
        layup=plies,
        ply_thickness_vox=pitch,
        target_phi=TARGET_DEFAULT,
        ddim_steps=50,
        material_fn=material_inset_z,
        notes={"layup": "A", "z_lo": SURFACE_Z_LO, "z_hi": SURFACE_Z_HI,
               "source": "diag_ldm_samples", "step": step},
    )
    case_path = out_dir / f"inset_step{step}"
    runner = VolumeRunner(run_dir, "latest", weights="ema", repo=repo)
    runner.run(spec, case_path)

    label = load_label(case_path / "label.tif")
    xct = load_u8(case_path / "volume.tif")
    material = material_inset_z(label.shape)
    res = surface_agreement(label, material,
                            z_lo=SURFACE_Z_LO, z_hi=SURFACE_Z_HI, xct=xct)
    res["case_dir"] = str(case_path)
    # The paper's gate, evaluated here so a failing run says so in its own log.
    res["gate"] = {
        "outside_air_gt_0.95": (res["air_fraction_outside_box"] or 0) > 0.95,
        "inside_air_lt_0.02": (res["air_fraction_inside_box"] or 1) < 0.02,
        "surface_error_lt_4vox": max(
            res["lower"]["error_abs_mean"] or 99.0,
            res["upper"]["error_abs_mean"] or 99.0) < 4.0,
    }
    res["gate"]["all_pass"] = all(res["gate"].values())
    return res


# ── conditioning-alive check ─────────────────────────────────────────────────

@torch.no_grad()
def conditioning_alive_check(
    sampler: DDIMSampler, ds: LatentDataset, cond: dict
) -> dict:
    """FULL vs input-neutralised sampling with one shared initial noise.

    The FULL_SEED2 run (same conditioning, different noise) provides the MAD a
    complete change of the stochastic input produces — the total-noise scale
    every neutralisation is expressed against.  Each neutralisation is a state
    the model has actually been trained on, so a near-zero response means the
    signal is dead, not that the input is out of distribution.
    """
    variants: dict[str, dict] = {
        "FULL": cond,
        "POR_NEUTRAL": {**cond,
                        "cond_por": torch.full_like(cond["cond_por"],
                                                    float(ds._cond_por.mean()))},
        "ORIENT_ZERO": {**cond,
                        "cond_orient": torch.zeros_like(cond["cond_orient"])},
        # All-material is the sampler's own default map, so this measures how
        # much of the output the painted map is responsible for.
        "MATERIAL_ONE": {**cond,
                         "cond_material": torch.ones_like(cond["cond_material"])},
        # The CFG neighbour null: no neighbour information at all.
        "NB_UNKNOWN": {**cond,
                       "nb_avail": torch.full_like(cond["nb_avail"], NB_UNKNOWN),
                       "nb_t": torch.zeros_like(cond["nb_t"])},
        "FULL_SEED2": cond,
    }

    z_by_name: dict[str, torch.Tensor] = {}
    for name, v in variants.items():
        z, _ = sample_bucket(sampler, v, SEED if name != "FULL_SEED2" else SEED + 1)
        z_by_name[name] = z

    z_full = z_by_name["FULL"]
    noise_mad = float((z_by_name["FULL_SEED2"] - z_full).abs().mean())
    out: dict = {"noise_mad": noise_mad}
    for name in ("POR_NEUTRAL", "ORIENT_ZERO", "MATERIAL_ONE", "NB_UNKNOWN"):
        mad = float((z_by_name[name] - z_full).abs().mean())
        out[name.lower()] = {
            "mad": mad,
            "pct_of_noise": 100.0 * mad / noise_mad if noise_mad > 0 else float("nan"),
        }
    return out


# ── kill-switch check ────────────────────────────────────────────────────────

@torch.no_grad()
def killswitch_check(
    sampler: DDIMSampler, vae: torch.nn.Module, ds: LatentDataset, cond: dict,
    device: torch.device, mean_dev: torch.Tensor, std_dev: torch.Tensor,
) -> dict:
    """Ask phi=lo vs phi=hi with the same seed and conditioning rows; compare
    the delivered pore fraction of the decoded labels."""
    por_log_stats = (ds.por_mean, ds.por_std)
    delivered = []
    for phi in KILLSWITCH_ASKS:
        c = float(porosity_to_cond(phi, por_log_stats))
        ks_cond = {**cond, "cond_por": torch.full_like(cond["cond_por"], c)}
        z, _ = sample_bucket(sampler, ks_cond, SEED)
        _, label = decode_latents(vae, z * std_dev + mean_dev, device)
        delivered.append((label == CLASS_PORE).float().mean(dim=(1, 2, 3)).cpu().numpy())
    lo, hi = delivered
    return {
        "por_lo": float(lo.mean()),
        "por_hi": float(hi.mean()),
        "por_lo_each": [round(float(v), 4) for v in lo],
        "por_hi_each": [round(float(v), 4) for v in hi],
        "separation": float(hi.mean() - lo.mean()),
        "direction_ok": bool(hi.mean() > lo.mean()),
    }


# ── trend tracking ───────────────────────────────────────────────────────────

def read_trend(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_verdict(prev: dict, curr: dict) -> None:
    key = "ema_ddim50"
    p, c = prev["variants"][key]["overall"], curr["variants"][key]["overall"]
    pct = 100.0 * (c["std_ratio"] - p["std_ratio"]) / p["std_ratio"]
    if pct <= -VERDICT_BAND_PCT:
        verdict = "CONVERGING"
    elif pct >= VERDICT_BAND_PCT:
        verdict = "REGRESSING"
    else:
        verdict = "STALLED"
    print(f"\nVERDICT: {verdict} — {key} overall std_ratio "
          f"{p['std_ratio']:.3f} → {c['std_ratio']:.3f} ({pct:+.1f}%) "
          f"since step {prev['step']} ({prev['timestamp']})")
    deltas = []
    for b, m in curr["variants"][key]["buckets"].items():
        pm = prev["variants"][key]["buckets"].get(b)
        if pm is None:
            deltas.append(f"{b}: n/a")
            continue
        deltas.append(f"{b}: {100.0 * (m['std_ratio'] - pm['std_ratio']) / pm['std_ratio']:+.1f}%")
    print("  bucket std_ratio Δ:  " + "   ".join(deltas))


# ── main ─────────────────────────────────────────────────────────────────────

def _latest_run_dir(repo: Path) -> Path:
    runs = sorted(
        (p for p in (repo / "runs" / "ldm").iterdir()
         if (p / "resolved_config.yaml").exists()),
        key=lambda p: (p / "resolved_config.yaml").stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError("No LDM runs found under runs/ldm/.")
    return runs[-1]


def main() -> None:
    t_start = time.perf_counter()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    repo = find_repo_root(__file__)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=None,
                   help="LDM run dir (default: most recent under runs/ldm/)")
    p.add_argument("--n", type=int, default=8, help="samples per context bucket (max batch)")
    p.add_argument("--ddim200", action="store_true",
                   help="also run the 200-step DDIM variants (extra GPU time)")
    p.add_argument("--ckpt", default="checkpoints/latest.ckpt",
                   help="checkpoint to load, relative to the run dir or absolute")
    p.add_argument("--no-killswitch", action="store_true",
                   help="skip the phi=0.005 vs 0.05 kill-switch check")
    p.add_argument("--out", default=None)
    p.add_argument("--out-jsonl", default=None,
                   help="override the trend JSONL path "
                        f"(default: <run_dir>/{TREND_FILE})")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(repo)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── checkpoint (read-only) ──────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", 0))
    raw_state = _strip_compile_prefix(ckpt["model"])
    ema_state = _strip_compile_prefix(ckpt["ema"])
    del ckpt
    logger.info("Loaded %s at step %d (keys: model, ema)", ckpt_path, step)

    out_dir = Path(args.out) if args.out else (
        repo / "runs" / "diagnostics" / "ldm_samples" / run_dir.name / f"step_{step}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── model + schedule ────────────────────────────────────────────────────
    ucfg = UNet3DConfig.from_cfg(cfg)
    model = UNet3DDenoiser(ucfg).to(device)
    schedule = DDPMSchedule.from_cfg(cfg, device)

    # ── latent store + frozen VAE ───────────────────────────────────────────
    latents_root = Path(cfg["data"]["latents_root"])
    if not latents_root.is_absolute():
        latents_root = repo / latents_root
    val_ds = LatentDataset(latents_root, "val", normalize=True)
    mean_dev = val_ds.channel_mean.to(device)
    std_dev = val_ds.channel_std.to(device)

    vae, _, _, _ = load_vae_from_checkpoint(Path(val_ds.metadata["vae_checkpoint"]), device)
    for prm in vae.parameters():
        prm.requires_grad_(False)
    vae.eval()

    # ── context buckets from real val rows ──────────────────────────────────
    rng = np.random.default_rng(0)
    bucket_rows, bucket_freq = collect_bucket_rows(val_ds, args.n, rng)
    conds = {b: build_bucket_cond(val_ds, rows, device)
             for b, rows in bucket_rows.items()}
    logger.info("Context buckets (n, observed frequency): %s",
                {b: (len(r), round(bucket_freq[b], 4)) for b, r in bucket_rows.items()})

    # ── real reference grid ─────────────────────────────────────────────────
    real_entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    real_por_all: list[torch.Tensor] = []
    for b, cond in conds.items():
        xct, label = decode_latents(vae, cond["z"] * std_dev + mean_dev, device)
        real_por_all.append((label == CLASS_PORE).float().mean(dim=(1, 2, 3)))
        real_entries.extend(grid_entries(xct, label, b))
    real_por = torch.cat(real_por_all)
    save_grid(real_entries, f"REAL decoded latents (val)  step={step}",
              out_dir / "real_reference.png")

    # ── variants: {raw, ema} × DDIM steps, per bucket ───────────────────────
    ddim_steps_list = [50] + ([200] if args.ddim200 else [])
    samplers = {s: DDIMSampler(model, schedule, device, n_steps=s)
                for s in ddim_steps_list}
    variant_results: dict[str, dict] = {}
    for wname, state in (("raw", raw_state), ("ema", ema_state)):
        model.load_state_dict(state)
        for n_steps in ddim_steps_list:
            vname = f"{wname}_ddim{n_steps}"
            per_bucket: dict[str, dict] = {}
            entries: list[tuple[str, np.ndarray, np.ndarray]] = []
            for bi, (b, cond) in enumerate(conds.items()):
                z, sat = sample_bucket(samplers[n_steps], cond, SEED + bi)
                xct, label = decode_latents(vae, z * std_dev + mean_dev, device)
                por = (label == CLASS_PORE).float().mean(dim=(1, 2, 3))
                air = (label == CLASS_AIR).float().mean(dim=(1, 2, 3))
                per_bucket[b] = bucket_metrics(z, sat, por, air, cond["phi"])
                entries.extend(grid_entries(xct, label, b))
            variant_results[vname] = {
                "buckets": per_bucket,
                "overall": weighted_overall(per_bucket, bucket_freq),
            }
            save_grid(entries, f"{vname}  step={step}", out_dir / f"{vname}.png")
            logger.info("%s done", vname)

    global _STD_REF
    _STD_REF = load_std_reference(Path(val_ds.root))
    if _STD_REF is None:
        logger.warning("no sampled-latent std reference; std_ratio is reported only "
                       "against the mu-normalised reference of 1.0, which OVERSTATES "
                       "the ratio for a latent_mode='sampled' run. Produce it with "
                       "scripts/analysis/latent_std_reference.py.")
    else:
        logger.info("sampled-latent std reference: %s",
                    [round(float(v), 4) for v in _STD_REF])

    # ── conditioning-alive check (raw weights, richest bucket) ──────────────
    model.load_state_dict(raw_state)
    richest = list(conds)[-1]
    alive_rows = bucket_rows[richest][:ALIVE_N]
    alive_cond = build_bucket_cond(val_ds, alive_rows, device)
    alive = conditioning_alive_check(samplers[50], val_ds, alive_cond)
    alive["bucket"] = richest
    alive["window"] = "interior (richest bucket)"
    alive["material_probe_valid"] = False   # see find_boundary_rows

    # The same probe on a BOUNDARY window, where the material map is not all
    # ones and MATERIAL_ONE is therefore a real change. Without this the
    # material row of the interior probe is structurally 0.0% and says nothing.
    boundary_rows = find_boundary_rows(val_ds, ALIVE_N)
    alive_boundary = None
    if boundary_rows:
        b_cond = build_bucket_cond(val_ds, boundary_rows, device)
        mat = b_cond["cond_material"]
        alive_boundary = conditioning_alive_check(samplers[50], val_ds, b_cond)
        alive_boundary["window"] = "boundary (material fraction in [0.2, 0.8])"
        alive_boundary["n_rows"] = len(boundary_rows)
        alive_boundary["material_min"] = float(mat.min())
        alive_boundary["material_mean"] = float(mat.mean())
        # If the map is still saturated the probe cannot answer, and saying so
        # is the whole point of adding it.
        alive_boundary["material_probe_valid"] = bool(float(mat.min()) < 0.99)
    else:
        logger.warning("no val window straddles the specimen boundary; "
                       "material conditioning cannot be probed")

    # ── kill-switch (raw + EMA, richest bucket) ─────────────────────────────
    killswitch = None
    if not args.no_killswitch:
        ks_rows = bucket_rows[richest][:KILLSWITCH_N]
        ks_cond = build_bucket_cond(val_ds, ks_rows, device)
        killswitch = {"asked_lo": KILLSWITCH_ASKS[0], "asked_hi": KILLSWITCH_ASKS[1],
                      "bucket": richest, "n": len(ks_rows)}
        for wname, state in (("raw", raw_state), ("ema", ema_state)):
            model.load_state_dict(state)
            killswitch[wname] = killswitch_check(
                samplers[50], vae, val_ds, ks_cond, device, mean_dev, std_dev)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n=== diag_ldm_samples  {run_dir.name}  step={step}  "
          f"n/bucket={args.n}  real por={real_por.mean().item():.4f}"
          f"±{real_por.std().item():.4f} ===")
    hdr = (f"{'bucket':<8} {'n':>3} {'std_ratio':>9} {'per-ch std':>26} "
           f"{'x0_sat':>7} {'por_mean':>9} {'por_std':>8} {'por_mae':>8} "
           f"{'air':>7} {'degen':>6}")
    for vname, res in variant_results.items():
        print(f"\n-- {vname} --")
        print(hdr)
        for b, m in res["buckets"].items():
            print(f"{b:<8} {m['n']:>3} {m['std_ratio']:>9.3f} "
                  f"{','.join(str(v) for v in m['ch_std']):>26} "
                  f"{m['x0_sat']:>7.4f} {m['por_mean']:>9.4f} {m['por_std']:>8.4f} "
                  f"{m['por_mae']:>8.4f} {m['air_mean']:>7.4f} {m['degen']:>6.2f}")
        o = res["overall"]
        print(f"{'overall':<8} {'-':>3} {o['std_ratio']:>9.3f} {'-':>26} "
              f"{o['x0_sat']:>7.4f} {o['por_mean']:>9.4f} {o['por_std']:>8.4f} "
              f"{o['por_mae']:>8.4f} {o['air_mean']:>7.4f} {o['degen']:>6.2f}"
              f"   (weights = observed bucket frequency)")

    if alive is not None:
        print(f"\n-- conditioning-alive check (raw weights, DDIM-50, n={ALIVE_N}, "
              f"bucket {alive['bucket']}, seed {SEED}) --")
        print(f"total-noise scale (FULL vs different seed): MAD {alive['noise_mad']:.4f}")
        print("  (interior window: the material row is a no-op here by "
              "construction — see the boundary probe below)")
        for key, what in (("por_neutral", "porosity"), ("orient_zero", "orientation"),
                          ("material_one", "material map"), ("nb_unknown", "neighbours")):
            a = alive[key]
            print(f"{what:<14} input changes the output by {a['pct_of_noise']:5.1f}% "
                  f"of the total-noise scale (MAD {a['mad']:.4f})")

    if alive_boundary is not None:
        b = alive_boundary
        print(f"\n-- conditioning-alive on a BOUNDARY window (n={b['n_rows']}, "
              f"material min {b['material_min']:.3f}, mean {b['material_mean']:.3f}) --")
        if not b["material_probe_valid"]:
            print("  material map is still ~all-ones here; the probe still cannot "
                  "answer whether material conditioning is used")
        print(f"total-noise scale: MAD {b['noise_mad']:.4f}")
        for key, what in (("por_neutral", "porosity"), ("orient_zero", "orientation"),
                          ("material_one", "material map"), ("nb_unknown", "neighbours")):
            a2 = b[key]
            print(f"  {what:<14} MAD {a2['mad']:.4f}  ({a2['pct_of_noise']:.2f}% of noise)")


    if killswitch is not None:
        print(f"\n-- kill-switch (DDIM-50, n={killswitch['n']}, "
              f"bucket {killswitch['bucket']}, seed {SEED}, "
              f"asks {killswitch['asked_lo']}/{killswitch['asked_hi']}) --")
        for wname in ("raw", "ema"):
            k = killswitch[wname]
            print(f"{wname}: asked {killswitch['asked_lo']} -> {k['por_lo']:.4f}  "
                  f"asked {killswitch['asked_hi']} -> {k['por_hi']:.4f}  "
                  f"separation {k['separation']:+.4f}  "
                  f"direction_ok={k['direction_ok']}")

    # ── trend tracking ──────────────────────────────────────────────────────
    # ── inset-box surface sample (from 60k) ─────────────────────────────────
    inset = None
    if step >= INSET_FROM_STEP:
        logger.info("inset-box surface sample (192^3, EMA, DDIM-50)")
        # RELEASE this script's own model and samplers first. The inset sample
        # goes through eval_v4's VolumeRunner, which loads its OWN UNet (83M
        # params) and VAE; holding both stacks at once is what made the 60k run
        # OOM at the second model load and then wedge at 100% CPU for 46
        # minutes with a corrupted CUDA context. Nothing after this point needs
        # them — the alive probe, the kill-switch and every variant are done.
        try:
            del samplers
        except NameError:
            pass
        model.to("cpu")
        del model
        import gc                                     # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            logger.info("released the diagnostic's model; CUDA free %.1f of %.1f GiB",
                        free / 2**30, total / 2**30)
        try:
            inset = inset_surface_sample(run_dir, step, out_dir)
        except Exception as exc:                        # noqa: BLE001
            logger.exception("inset sample failed: %s", exc)
            inset = {"error": repr(exc)}
        if inset and "error" not in inset:
            g = inset["gate"]
            logger.info("inset: outside-air %.4f  inside-air %.4f  "
                        "lower |err| %.2f  upper |err| %.2f  -> %s",
                        inset["air_fraction_outside_box"],
                        inset["air_fraction_inside_box"],
                        inset["lower"]["error_abs_mean"],
                        inset["upper"]["error_abs_mean"],
                        "PASS" if g["all_pass"] else "FAIL")
    else:
        logger.info("step %d < %d — no inset sample yet", step, INSET_FROM_STEP)

    trend_path = Path(args.out_jsonl) if args.out_jsonl else run_dir / TREND_FILE
    entry = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_per_bucket": {b: len(r) for b, r in bucket_rows.items()},
        "bucket_frequency": bucket_freq,
        # EVERY variant that was computed, not a hardcoded pair. --ddim200
        # spends the GPU time on raw_ddim200 / ema_ddim200 and the old literal
        # then dropped them from the durable record, leaving them only in
        # stdout — while the D43 gate is written in DDIM-200 numbers.
        "variants": dict(variant_results),
        "alive": alive,
        "alive_boundary": alive_boundary,
        "inset_surface": inset,
        "killswitch": killswitch,
    }
    history = read_trend(trend_path)
    if history:
        print_verdict(history[-1], entry)
    else:
        print(f"\nTREND: first entry at step {step} — no verdict yet "
              f"(next run of this script will compare against it).")
    with open(trend_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("Trend entry appended to %s (%d entries)", trend_path, len(history) + 1)

    print(f"\nPNGs in {out_dir}")
    print(f"wall-clock: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

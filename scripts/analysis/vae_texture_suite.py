#!/usr/bin/env python
"""Four texture proxies of VAE reconstructions, beside grey L1. (CPU.)

Why (author, 2026-10-19): L1 rewards the conditional mean and so penalises
texture; the discriminator phase makes texture look better and L1 worse. These
proxies only TRACK whether texture improves over the step checkpoints. They do
not stop training and do not pick checkpoints (author's instruction).

Patches: the harness's held-out val patches (vae_val_l1.load_model_and_patches,
20 batches of 32 on a fixed stride), interior only (no air voxel), so the
specimen boundary does not dominate the texture statistics.

  A  spectrum   per-band power ratio, coherence and relative error of the 3-D
                Hann-windowed FFT, recon vs its own input (vae_texture_fidelity.py,
                validated in campaign 32). Power < 1 = blur, > 1 = invented detail.
  B  SWD        sliced Wasserstein distance between sets of 7^3 neighbourhoods of a
                3-D Laplacian pyramid (levels 64, 32, 16), ProGAN (Karras et al.
                2018) in 3-D. Descriptors normalised per level by the real set's
                mean / sd; 4 x 128 random directions; x 1e3.
  C  FID / KID  Inception-v3 pool3 on the central slice of each patch in the
                three planes (eval_v4.microstructure: inception_features,
                frechet_distance); KID = unbiased MMD^2, cubic kernel, 50 subsets
                of 100 (Binkowski et al. 2018), x 1e3.
  D  pores      pores from the class head (argmax == PORE) vs the true label:
                S2(r) shape distance (curve_w1, voxels) of the pooled mean S2 of
                patches that hold pores, and the pore-diameter W1 (psd_w1, voxels).

B and C are set-to-set distances, so each is reported against a floor: the
interior patches are split into halves H1 / H2 by index parity, and
  value = mean of d(recon H1, real H2) and d(recon H2, real H1)
  floor = d(real H1, real H2)
i.e. the same set sizes, never a patch against itself. value ~ floor = as close
to the scan as another sample of the scan is.

Modes
  --validate        no model: replace the recon by known transforms of the input
                    (blur 0.5/1/2, noise 0.1/0.3 sd; pore masks dilated, eroded,
                    speckled) and check each metric moves the right way.
  --run RUN --ckpt C [--ckpt C2 ...]   score step checkpoints; one JSON line per
                    checkpoint appended to <out>/<run-id>.jsonl (skips done ones).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_val_l1 import REPO, load_model_and_patches  # noqa: E402
from vae_texture_fidelity import interior_patches, shell_sums, summarise  # noqa: E402
from poregen.dataset.loader import LABEL_PORE  # noqa: E402
from poregen.eval_v4.microstructure import (  # noqa: E402
    curve_w1, frechet_distance, inception_features, pore_diameters, psd_w1, s2_radial)

SWD_PATCH, SWD_PER_VOL, SWD_DIRS, SWD_REPEATS = 7, 64, 128, 4


# ── B: sliced Wasserstein on a 3-D Laplacian pyramid ────────────────────────

def laplacian_pyramid(v: np.ndarray, levels: int = 3) -> list[np.ndarray]:
    from scipy.ndimage import gaussian_filter, zoom
    out, g = [], v.astype(np.float32)
    for _ in range(levels):
        down = gaussian_filter(g, 1.0)[::2, ::2, ::2]
        out.append(g - zoom(down, 2, order=1))
        g = down
    return out


def swd_descriptors(vols: list[np.ndarray], seed: int) -> list[np.ndarray]:
    """Per pyramid level, (n_vols * SWD_PER_VOL, 343) random 7^3 neighbourhoods."""
    rng = np.random.default_rng(seed)
    per_level: list[list[np.ndarray]] = [[], [], []]
    for v in vols:
        for li, lap in enumerate(laplacian_pyramid(v)):
            s = lap.shape[0] - SWD_PATCH + 1
            c = rng.integers(0, s, size=(SWD_PER_VOL, 3))
            per_level[li].append(np.stack([lap[z:z + SWD_PATCH, y:y + SWD_PATCH, x:x + SWD_PATCH].ravel()
                                           for z, y, x in c]))
    return [np.concatenate(p) for p in per_level]


def swd(a: np.ndarray, b: np.ndarray, ref: np.ndarray, seed: int) -> float:
    """Sliced Wasserstein-1 (x 1e3) between descriptor sets, normalised by ``ref``."""
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    a, b = torch.from_numpy((a - mu) / sd), torch.from_numpy((b - mu) / sd)
    n = min(len(a), len(b))
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(SWD_REPEATS):
        d = torch.randn(a.shape[1], SWD_DIRS, generator=g, dtype=a.dtype)
        d /= d.norm(dim=0, keepdim=True)
        pa = (a[torch.randperm(len(a), generator=g)[:n]] @ d).sort(0).values
        pb = (b[torch.randperm(len(b), generator=g)[:n]] @ d).sort(0).values
        vals.append((pa - pb).abs().mean().item())
    return float(np.mean(vals) * 1e3)


# ── C: FID / KID on central slices in three planes ──────────────────────────

def central_slices(vols: list[np.ndarray]) -> np.ndarray:
    c = vols[0].shape[0] // 2
    return np.stack([s for v in vols for s in (v[c], v[:, c], v[:, :, c])]).astype(np.float32)


def kid(fa: np.ndarray, fb: np.ndarray, n_sub: int = 50, size: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    d = fa.shape[1]
    k = lambda x, y: (x @ y.T / d + 1.0) ** 3  # noqa: E731
    m = min(size, len(fa), len(fb))
    vals = []
    for _ in range(n_sub):
        x = fa[rng.choice(len(fa), m, replace=False)]
        y = fb[rng.choice(len(fb), m, replace=False)]
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals) * 1e3)


# ── D: pore morphology ──────────────────────────────────────────────────────

def pore_stats(rec_pore: list[np.ndarray], true_pore: list[np.ndarray]) -> dict:
    keep = [i for i, t in enumerate(true_pore) if t.any()]
    r, s2r = None, []
    s2t = []
    for i in keep:
        r, a = s2_radial(rec_pore[i]); _, b = s2_radial(true_pore[i])
        s2r.append(a); s2t.append(b)
    dr = np.concatenate([pore_diameters(rec_pore[i]) for i in keep] or [np.zeros(0)])
    dt = np.concatenate([pore_diameters(true_pore[i]) for i in keep] or [np.zeros(0)])
    return {"n_patches_with_pores": len(keep),
            "s2_shape_w1_vox": curve_w1(r, np.nanmean(s2r, 0), np.nanmean(s2t, 0)),
            "pore_diam_w1_vox": psd_w1(dr, dt),
            "n_pores_recon": int(dr.size), "n_pores_true": int(dt.size)}


# ── the suite on one set of (recon, input) pairs ────────────────────────────

def suite(rec: list[np.ndarray], gt: list[np.ndarray], rec_pore, true_pore, feats_gt=None) -> dict:
    h1, h2 = list(range(0, len(gt), 2)), list(range(1, len(gt), 2))
    pick = lambda xs, ix: [xs[i] for i in ix]  # noqa: E731
    # A
    a = summarise(np.stack([shell_sums(r, t) for r, t in zip(rec, gt)]))["bands"]
    # B
    dr, dg = swd_descriptors(rec, 1), swd_descriptors(gt, 1)
    n1 = len(h1) * SWD_PER_VOL
    half = lambda d, which: d.reshape(len(gt), SWD_PER_VOL, -1)[which].reshape(-1, d.shape[1])  # noqa: E731
    b = {}
    for li, name in enumerate(("lap64", "lap32", "lap16")):
        R1, R2, G1, G2 = half(dr[li], h1), half(dr[li], h2), half(dg[li], h1), half(dg[li], h2)
        b[name] = {"value": (swd(R1, G2, dg[li], li) + swd(R2, G1, dg[li], li)) / 2,
                   "floor": swd(G1, G2, dg[li], li)}
    del n1
    # C
    fr = inception_features(central_slices(rec), torch.device("cpu")).reshape(len(rec), 3, -1)
    fg = feats_gt if feats_gt is not None else inception_features(central_slices(gt), torch.device("cpu")).reshape(len(gt), 3, -1)
    f = lambda F, ix: F[ix].reshape(-1, F.shape[-1])  # noqa: E731
    c = {"fid": {"value": (frechet_distance(f(fr, h1), f(fg, h2)) + frechet_distance(f(fr, h2), f(fg, h1))) / 2,
                 "floor": frechet_distance(f(fg, h1), f(fg, h2))},
         "kid_x1e3": {"value": (kid(f(fr, h1), f(fg, h2)) + kid(f(fr, h2), f(fg, h1))) / 2,
                      "floor": kid(f(fg, h1), f(fg, h2))}}
    # D
    d = pore_stats(rec_pore, true_pore)
    l1 = float(np.mean([np.abs(r - t).mean() for r, t in zip(rec, gt)]))
    return {"n_interior_patches": len(gt), "grey_l1": l1, "A_spectrum": a, "B_swd": b, "C_slices": c, "D_pores": d}, fg


# ── data ────────────────────────────────────────────────────────────────────

def pairs_for(run: Path, ckpt: str, split: str, n_batches: int, batch_size: int):
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs
    model, ds, idx, _, _ = load_model_and_patches(run, split, n_batches, batch_size, None, ckpt)
    torch.manual_seed(0)
    keep = [i for i, _ in interior_patches(ds, idx)]
    rec, gt, rp, tp = [], [], [], []
    for s in range(0, len(keep), batch_size):
        rows = [ds[j] for j in keep[s:s + batch_size]]
        batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0] if torch.is_tensor(rows[0][k])}
        bdev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            out = model(*args)
        x = decode_xct(out.xct_out)[:, 0].numpy()
        cls = out.class_logits.argmax(1).numpy()
        for bi in range(len(rows)):
            rec.append(x[bi].astype(np.float64)); gt.append(bdev["xct"][bi, 0].numpy().astype(np.float64))
            rp.append(cls[bi] == LABEL_PORE); tp.append(rows[bi]["label"].numpy().reshape(x.shape[1:]) == LABEL_PORE)
    return rec, gt, rp, tp, keep


def validate(ref_run: Path, ckpt: str, split: str, nb: int, bs: int, out: Path) -> int:
    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
    _, gt, _, tp, _ = pairs_for(ref_run, ckpt, split, nb, bs)
    rng = np.random.default_rng(0)
    speck = lambda p: p | (rng.random(p.shape) < 0.002)  # noqa: E731
    cases = {
        "identity": (lambda t: t, lambda p: p),
        "blur_s0.5": (lambda t: gaussian_filter(t, 0.5), lambda p: p),
        "blur_s1": (lambda t: gaussian_filter(t, 1.0), lambda p: binary_erosion(p)),
        "blur_s2": (lambda t: gaussian_filter(t, 2.0), lambda p: binary_erosion(p, iterations=2)),
        "noise_0.1sd": (lambda t: t + 0.1 * t.std() * rng.standard_normal(t.shape), lambda p: binary_dilation(p)),
        "noise_0.3sd": (lambda t: t + 0.3 * t.std() * rng.standard_normal(t.shape), speck),
    }
    res, fg = {}, None
    for name, (fx, fp) in cases.items():
        res[name], fg = suite([fx(t) for t in gt], gt, [fp(p) for p in tp], tp, fg)
        print(name, json.dumps(flat(res[name])), flush=True)
    v = lambda n, *k: _get(res[n], k)  # noqa: E731
    checks = {}
    for metric in (("B_swd", "lap64", "value"), ("C_slices", "fid", "value"), ("C_slices", "kid_x1e3", "value")):
        lab = "/".join(metric[:2])
        # absolute, not a ratio: KID is unbiased and its floor sits near 0, where a
        # ratio is noise. Identity must be closer to the floor than 2 % of the
        # distance from the floor to the mildest blur.
        fl = _get(res["identity"], metric[:2] + ("floor",))
        checks[f"{lab}: identity at the floor (|id - floor| < 2 % of blur_s0.5 - floor)"] = (
            abs(v("identity", *metric) - fl) < 0.02 * (v("blur_s0.5", *metric) - fl))
        checks[f"{lab}: blur worsens monotonically"] = v("identity", *metric) < v("blur_s0.5", *metric) < v("blur_s1", *metric) < v("blur_s2", *metric)
        checks[f"{lab}: noise worsens monotonically"] = v("identity", *metric) < v("noise_0.1sd", *metric) < v("noise_0.3sd", *metric)
    fine = ("A_spectrum", "fine_4-8", "power_ratio")
    checks["A: blur power < 1, monotonic"] = 1 > v("blur_s0.5", *fine) > v("blur_s1", *fine) > v("blur_s2", *fine)
    checks["A: noise finest power > 1"] = v("noise_0.1sd", "A_spectrum", "finest_2-4", "power_ratio") > 1
    for m in ("s2_shape_w1_vox", "pore_diam_w1_vox"):
        checks[f"D {m}: identity 0"] = v("identity", "D_pores", m) < 1e-9
        checks[f"D {m}: erosion, dilation, speckle all > 0"] = all(
            v(n, "D_pores", m) > 0 for n in ("blur_s1", "blur_s2", "noise_0.1sd", "noise_0.3sd"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps({"ref_run": ref_run.name, "ckpt": ckpt, "results": res,
                                                     "checks": checks}, indent=2) + "\n")
    for c, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + c)
    ok = all(checks.values())
    print("VALIDATION", "PASS" if ok else "FAIL", out / "validation.json")
    return 0 if ok else 1


def _get(d, keys):
    for k in keys:
        d = d[k]
    return d


def flat(r: dict) -> dict:
    o = {"l1": r["grey_l1"]}
    for b, q in r["A_spectrum"].items():
        o[f"A_{b.split('_')[0]}_P"] = q["power_ratio"]
    for k, q in r["B_swd"].items():
        o[f"B_{k}"] = q["value"]
    o["C_fid"], o["C_kid"] = r["C_slices"]["fid"]["value"], r["C_slices"]["kid_x1e3"]["value"]
    o["D_s2"], o["D_diam"] = r["D_pores"]["s2_shape_w1_vox"], r["D_pores"]["pore_diam_w1_vox"]
    return {k: round(float(v), 5) for k, v in o.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--run", type=Path)
    ap.add_argument("--ref-run", type=Path)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=REPO / "runs/campaigns/33-advperc-ldm/texture")
    a = ap.parse_args()
    ab = lambda p: p if p.is_absolute() else REPO / p  # noqa: E731
    if a.validate:
        return validate(ab(a.ref_run), a.ckpt[0] if a.ckpt else "best.ckpt", a.split, a.n_batches, a.batch_size, a.out)
    run = ab(a.run)
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"{run.name[:12]}.jsonl"
    done = {json.loads(l)["ckpt"] for l in open(dest)} if dest.exists() else set()
    fg = None
    for ck in a.ckpt:
        if ck in done:
            print("skip", ck); continue
        rec, gt, rp, tp, _ = pairs_for(run, ck, a.split, a.n_batches, a.batch_size)
        r, fg = suite(rec, gt, rp, tp, fg)
        step = int(torch.load(run / ck, map_location="cpu", weights_only=False).get("step", -1))
        with open(dest, "a") as fh:
            fh.write(json.dumps({"run": run.name, "ckpt": ck, "step": step, **r}) + "\n")
        print(run.name[:12], step, json.dumps(flat(r)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

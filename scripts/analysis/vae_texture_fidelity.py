"""Per-frequency-band fidelity of VAE grey (XCT) reconstructions.

Why this exists. The harness's two texture numbers (``vae_val_l1.py``) do not
measure fine detail well:

* ``texture_corr`` is a Pearson correlation after removing the patch mean. It
  weights every spatial frequency by its variance, and the coarse structure
  (pores, tows) holds most of the variance, so the fine scales barely move it.
* ``sharpness`` is a ratio of Laplacian standard deviations. It counts the
  AMOUNT of high-frequency energy, not whether it is in the right place: added
  noise or invented texture of the right strength scores 1.0.

This script splits each held-out interior patch (no air voxel) into radial
frequency shells of a Hann-windowed 3-D FFT and reports, per shell and per
octave band, three pooled quantities between reconstruction r and input t:

* coherence  FSC(k) = Re sum F_r conj(F_t) / sqrt(sum |F_r|^2 sum |F_t|^2)
  (Fourier shell correlation, the resolution measure of tomography / cryo-EM):
  is the detail in the RIGHT PLACE. Blind to a pure linear blur.
* power ratio  P(k) = sum |F_r|^2 / sum |F_t|^2: is the detail THERE, and is it
  overdone. < 1 = lost (blur), > 1 = added (noise, invented texture).
* relative band error  E(k) = sum |F_r - F_t|^2 / sum |F_t|^2: one number that
  grows with blur, with added noise and with displaced texture alike.

Bands are in wavelength (voxels): coarse 16-64, mid 8-16, fine 4-8, finest 2-4.
"Resolution" is the shortest wavelength down to which FSC stays >= 0.5.

Caveat that matters for reading it: in a single XCT patch much of the finest
band is likely scan noise, which a VAE cannot and should not reproduce. The
input's own spectrum tail is reported so that floor can be seen; the finest
band is not expected to reach 1.

Modes
  --validate        no model: degrade real interior patches with known
                    transforms (blur, added noise, phase scramble) and check
                    the metric moves the way it must. Run this FIRST.
  --run RUN         a VAE run directory: reconstruct the harness's held-out
                    patches and score them. Writes <out>/<tag>.json and
                    <out>/<tag>_slices.npz (central slices for the montage).
  --compare J ...   read result JSONs, write a band table, the curves figure
                    and the slice montage.

Same patches as the harness: split's val set restricted with
--exclude-train-of (split_v2 by default), 20 batches of 32 on a fixed stride.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_val_l1 import REPO, finalised_ckpt, load_model_and_patches  # noqa: E402
from poregen.dataset.loader import LABEL_AIR  # noqa: E402

N = 64
BANDS = {  # name: (k_lo inclusive, k_hi exclusive), k in cycles per 64 voxels
    "coarse_16-64": (1, 4),
    "mid_8-16": (4, 8),
    "fine_4-8": (8, 16),
    "finest_2-4": (16, 33),
}
KMAX = 32


def _geometry():
    """Shell index and rfft half-space weight for every rfftn bin of a 64^3 cube."""
    f = np.fft.fftfreq(N) * N
    fr = np.fft.rfftfreq(N) * N
    kz, ky, kx = np.meshgrid(f, f, fr, indexing="ij")
    shell = np.rint(np.sqrt(kz**2 + ky**2 + kx**2)).astype(np.int64)
    # bins with 0 < kx < N/2 stand for two conjugate bins of the full FFT
    w = np.where((fr > 0) & (fr < N / 2), 2.0, 1.0)[None, None, :] * np.ones_like(kz)
    shell[shell > KMAX] = -1
    return torch.from_numpy(shell), torch.from_numpy(w)


SHELL, WEIGHT = _geometry()
_h = torch.hann_window(N, periodic=False, dtype=torch.float64)
HANN = _h[:, None, None] * _h[None, :, None] * _h[None, None, :]


def shell_sums(r: np.ndarray | torch.Tensor, t: np.ndarray | torch.Tensor) -> np.ndarray:
    """Per-shell sums [rt, rr, tt, err] for one patch pair, shape (4, KMAX+1)."""
    r = torch.as_tensor(r, dtype=torch.float64).reshape(N, N, N)
    t = torch.as_tensor(t, dtype=torch.float64).reshape(N, N, N)
    fr = torch.fft.rfftn((r - r.mean()) * HANN)
    ft = torch.fft.rfftn((t - t.mean()) * HANN)
    quants = [
        (fr * ft.conj()).real, fr.abs() ** 2, ft.abs() ** 2, (fr - ft).abs() ** 2,
    ]
    out = np.zeros((4, KMAX + 1))
    m = SHELL >= 0
    idx = SHELL[m]
    for i, q in enumerate(quants):
        out[i] = torch.zeros(KMAX + 1, dtype=torch.float64).index_add_(
            0, idx, (q * WEIGHT)[m]).numpy()
    return out


def summarise(sums: np.ndarray) -> dict:
    """Pooled metrics from per-patch shell sums, shape (n_patches, 4, KMAX+1)."""
    tot = sums.sum(axis=0)
    rt, rr, tt, err = tot
    ks = np.arange(1, KMAX + 1)
    fsc = rt[ks] / np.sqrt(rr[ks] * tt[ks])
    res_k = next((int(k) for k, v in zip(ks, fsc) if v < 0.5), None)
    bands = {}
    for name, (lo, hi) in BANDS.items():
        s = slice(lo, hi)
        bands[name] = {
            "coherence": float(rt[s].sum() / np.sqrt(rr[s].sum() * tt[s].sum())),
            "power_ratio": float(rr[s].sum() / tt[s].sum()),
            "rel_error": float(err[s].sum() / tt[s].sum()),
        }
    return {
        "shell_k": ks.tolist(),
        "fsc": fsc.tolist(),
        "power_ratio": (rr[ks] / tt[ks]).tolist(),
        "rel_error": (err[ks] / tt[ks]).tolist(),
        "input_power": (tt[ks] / len(sums)).tolist(),
        "resolution_voxels": (None if res_k is None else N / res_k),
        "resolution_note": ("FSC >= 0.5 down to the Nyquist shell" if res_k is None
                            else f"first shell with FSC < 0.5 is k={res_k}"),
        "bands": bands,
    }


def bootstrap(sums: np.ndarray, n: int = 200, seed: int = 0) -> dict:
    """95 % intervals of each band number, resampling patches."""
    rng = np.random.default_rng(seed)
    draws = {b: {q: [] for q in ("coherence", "power_ratio", "rel_error")} for b in BANDS}
    for _ in range(n):
        s = summarise(sums[rng.integers(0, len(sums), len(sums))])["bands"]
        for b in BANDS:
            for q in draws[b]:
                draws[b][q].append(s[b][q])
    return {b: {q: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for q, v in d.items()} for b, d in draws.items()}


# ── patches ─────────────────────────────────────────────────────────────────

def interior_patches(ds, idx):
    """(xct, label) of the indexed patches with no AIR voxel, in index order."""
    for i in idx:
        item = ds[i]
        lab = item.get("label")
        if lab is not None and bool((lab == LABEL_AIR).any()):
            continue
        yield i, item


def reconstruct(run: Path, split: str, exclude: str | None, ckpt: str | None,
                n_batches: int, batch_size: int):
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    ckpt = ckpt or finalised_ckpt(run)
    model, ds, idx, cfg, _ = load_model_and_patches(
        run, split, n_batches, batch_size, exclude, ckpt)
    torch.manual_seed(0)  # the posterior is sampled on every forward pass
    keep = [i for i, _ in interior_patches(ds, idx)]
    pairs = []
    for s in range(0, len(keep), batch_size):
        rows = [ds[j] for j in keep[s:s + batch_size]]
        batch = {k: torch.stack([r[k] for r in rows])
                 for k in rows[0] if torch.is_tensor(rows[0][k])}
        bdev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            rec = decode_xct(model(*args).xct_out)
        for b in range(rec.shape[0]):
            pairs.append((rec[b, 0].numpy(), bdev["xct"][b, 0].numpy()))
    return pairs, keep, ckpt, cfg["data"].get("dataset_root")


# ── validation on known transforms ──────────────────────────────────────────

def _phase_scramble(t: np.ndarray, rng) -> np.ndarray:
    ft = np.fft.rfftn(t - t.mean())
    ph = np.fft.rfftn(rng.standard_normal(t.shape))
    return np.fft.irfftn(np.abs(ft) * np.exp(1j * np.angle(ph)), s=t.shape) + t.mean()


def validate(split: str, exclude: str | None, n_batches: int, batch_size: int,
             out: Path, ref_run: Path) -> int:
    from scipy.ndimage import gaussian_filter

    # the patches come through a model run's own loader so they are EXACTLY the
    # harness's patches; the model itself is not used here
    _, ds, idx, _, _ = load_model_and_patches(ref_run, split, n_batches, batch_size,
                                              exclude, finalised_ckpt(ref_run))
    gts = [item["xct"][0].numpy().astype(np.float64)
           for _, item in interior_patches(ds, idx)]
    rng = np.random.default_rng(0)
    transforms = {
        "identity": lambda t: t,
        "blur_s0.5": lambda t: gaussian_filter(t, 0.5),
        "blur_s1": lambda t: gaussian_filter(t, 1.0),
        "blur_s2": lambda t: gaussian_filter(t, 2.0),
        "noise_0.1sd": lambda t: t + 0.1 * t.std() * rng.standard_normal(t.shape),
        "noise_0.3sd": lambda t: t + 0.3 * t.std() * rng.standard_normal(t.shape),
        "phase_scramble": lambda t: _phase_scramble(t, rng),
    }
    res = {}
    for name, fn in transforms.items():
        sums = np.stack([shell_sums(fn(t), t) for t in gts])
        res[name] = summarise(sums)["bands"]
    f, fi = "fine_4-8", "finest_2-4"
    b = lambda n, band, q: res[n][band][q]  # noqa: E731
    checks = {
        "identity is perfect (coherence 1, power 1, error 0)": all(
            abs(b("identity", k, "coherence") - 1) < 1e-9 and abs(b("identity", k, "power_ratio") - 1) < 1e-9
            and b("identity", k, "rel_error") < 1e-12 for k in BANDS),
        "blur lowers fine power, monotonic in sigma": (
            b("blur_s0.5", f, "power_ratio") > b("blur_s1", f, "power_ratio") > b("blur_s2", f, "power_ratio")
            and b("blur_s0.5", f, "power_ratio") < 1),
        "blur keeps coherence high (a linear filter moves nothing)": all(
            b(n, k, "coherence") > 0.9 for n in ("blur_s0.5", "blur_s1") for k in ("coarse_16-64", "mid_8-16", f)),
        "blur error grows with sigma": (
            b("blur_s0.5", f, "rel_error") < b("blur_s1", f, "rel_error") < b("blur_s2", f, "rel_error")),
        "noise raises finest power above 1, more noise more power": (
            1 < b("noise_0.1sd", fi, "power_ratio") < b("noise_0.3sd", fi, "power_ratio")),
        "noise lowers finest coherence, more noise lower": (
            b("noise_0.3sd", fi, "coherence") < b("noise_0.1sd", fi, "coherence") < 1),
        "noise barely touches coarse coherence": b("noise_0.3sd", "coarse_16-64", "coherence") > 0.95,
        "phase scramble keeps power (~1) and kills coherence (~0)": all(
            0.8 < b("phase_scramble", k, "power_ratio") < 1.25 and abs(b("phase_scramble", k, "coherence")) < 0.15
            for k in BANDS),
        "phase scramble error ~2 (two unrelated signals)": all(
            1.6 < b("phase_scramble", k, "rel_error") < 2.4 for k in BANDS),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(
        {"n_interior_patches": len(gts), "split": split, "exclude_train_of": exclude,
         "bands": res, "checks": checks}, indent=2) + "\n")
    print(f"validation on {len(gts)} interior patches of {split}")
    for name, r in res.items():
        print(f"  {name:15s} " + "  ".join(
            f"{k.split('_')[0]}: C {v['coherence']:+.3f} P {v['power_ratio']:.3f} E {v['rel_error']:.3f}"
            for k, v in r.items()))
    ok = all(checks.values())
    for c, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {c}")
    print("VALIDATION", "PASS" if ok else "FAIL", "->", out / "validation.json")
    return 0 if ok else 1


# ── scoring a run ───────────────────────────────────────────────────────────

def score(run: Path, split: str, exclude: str | None, ckpt: str | None,
          n_batches: int, batch_size: int, out: Path, tag: str, n_slices: int) -> int:
    pairs, keep, ckpt, trained_on = reconstruct(run, split, exclude, ckpt, n_batches, batch_size)
    sums = np.stack([shell_sums(r, t) for r, t in pairs])
    res = summarise(sums)
    res.update({
        "run": run.name, "ckpt": ckpt, "trained_on": trained_on, "evaluated_on": split,
        "exclude_train_of": exclude, "n_interior_patches": len(pairs),
        "patch_rows": keep, "bands_ci95": bootstrap(sums), "tag": tag,
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    # central z-slices of a fixed, evenly spaced subset for the montage
    pick = np.linspace(0, len(pairs) - 1, n_slices).round().astype(int)
    np.savez_compressed(out / f"{tag}_slices.npz",
                        rows=np.array([keep[i] for i in pick]),
                        recon=np.stack([pairs[i][0][N // 2] for i in pick]),
                        input=np.stack([pairs[i][1][N // 2] for i in pick]))
    print(f"{tag}: {run.name[:60]} ({ckpt}), {len(pairs)} interior patches, "
          f"resolution {res['resolution_voxels']} vox")
    for k, v in res["bands"].items():
        ci = res["bands_ci95"][k]
        print(f"  {k:13s} coherence {v['coherence']:.3f} [{ci['coherence'][0]:.3f}, {ci['coherence'][1]:.3f}]"
              f"  power {v['power_ratio']:.3f} [{ci['power_ratio'][0]:.3f}, {ci['power_ratio'][1]:.3f}]"
              f"  error {v['rel_error']:.3f} [{ci['rel_error'][0]:.3f}, {ci['rel_error'][1]:.3f}]")
    return 0


# ── comparison ──────────────────────────────────────────────────────────────

def compare(jsons: list[Path], out: Path) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    res = [json.loads(p.read_text()) for p in jsons]
    lines = ["| model | n | resolution (vox) | " + " | ".join(
        f"{b} C / P / E" for b in BANDS) + " |", "|---" * (3 + len(BANDS)) + "|"]
    for r in res:
        lines.append(f"| {r['tag']} | {r['n_interior_patches']} | {r['resolution_voxels'] or '<2'} | " + " | ".join(
            f"{r['bands'][b]['coherence']:.3f} / {r['bands'][b]['power_ratio']:.3f} / {r['bands'][b]['rel_error']:.3f}"
            for b in BANDS) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "band_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for r in res:
        lam = [N / k for k in r["shell_k"]]
        ax[0].plot(lam, r["fsc"], label=r["tag"])
        ax[1].plot(lam, r["power_ratio"], label=r["tag"])
        ax[2].plot(lam, r["rel_error"], label=r["tag"])
    for a, t in zip(ax, ("coherence (FSC)", "power ratio recon / input", "relative band error")):
        a.set_xscale("log", base=2)
        a.invert_xaxis()
        a.set_xlabel("wavelength (voxels)")
        a.set_title(t)
        a.grid(alpha=.3)
    ax[0].axhline(0.5, color="k", lw=.6, ls=":")
    ax[1].axhline(1.0, color="k", lw=.6, ls=":")
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "band_curves.png", dpi=130)

    sl = [np.load(p.with_name(p.stem + "_slices.npz")) for p in jsons]
    n = sl[0]["recon"].shape[0]
    for s, r in zip(sl, res):
        if not np.array_equal(s["rows"], sl[0]["rows"]):
            print(f"WARNING: {r['tag']} montage patches differ from {res[0]['tag']}'s")
    rows = 2 * (len(res) + 1)
    fig, axs = plt.subplots(rows, n, figsize=(2.1 * n, 2.1 * rows))
    hp = lambda a: a - gaussian_filter(a, 2.0)  # noqa: E731  detail at wavelengths < ~8 vox
    for j in range(n):
        g = sl[0]["input"][j]
        lo, hi = np.percentile(g, [1, 99])
        hlim = np.percentile(np.abs(hp(g)), 99)
        imgs = [("input", g)] + [(r["tag"], s["recon"][j]) for r, s in zip(res, sl)]
        for i, (name, img) in enumerate(imgs):
            axs[2 * i, j].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            axs[2 * i + 1, j].imshow(hp(img), cmap="gray", vmin=-hlim, vmax=hlim)
            if j == 0:
                axs[2 * i, 0].set_ylabel(name, fontsize=7)
                axs[2 * i + 1, 0].set_ylabel(name + "\nhigh-pass", fontsize=7)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "slices_montage.png", dpi=110)
    print("->", out / "band_table.md", out / "band_curves.png", out / "slices_montage.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--run", type=Path)
    g.add_argument("--compare", type=Path, nargs="+")
    ap.add_argument("--ref-run", type=Path, help="run whose loader supplies the patches for --validate")
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--exclude-train-of", default="split_v2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-slices", type=int, default=6)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "runs/campaigns/32-texture-fidelity")
    a = ap.parse_args()
    abspath = lambda p: p if p.is_absolute() else REPO / p  # noqa: E731
    if a.validate:
        return validate(a.split, a.exclude_train_of, a.n_batches, a.batch_size, a.out, abspath(a.ref_run))
    if a.run:
        return score(abspath(a.run), a.split, a.exclude_train_of, a.ckpt, a.n_batches,
                     a.batch_size, a.out, a.tag or abspath(a.run).name[:12], a.n_slices)
    return compare([abspath(p) for p in a.compare], a.out)


if __name__ == "__main__":
    raise SystemExit(main())

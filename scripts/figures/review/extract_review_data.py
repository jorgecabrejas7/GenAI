#!/usr/bin/env python
"""Build the inputs for the review-deck figures from the campaign data, then draw them.

Reads campaigns 18 (final eval), 19 (stress geometry) and 20 (OOD conditioning) and writes
<out>/data/ (CSV + slices + manifest in the formats the fig2/3/4 scripts expect) and
<out>/fig{2,3,4}_*.{png,pdf}.  CPU only; ~2 GB of TIFF reads.

  python scripts/figures/review/extract_review_data.py \
      --out runs/campaigns/18-eval-v4-final/figures_review
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage

HERE = Path(__file__).resolve().parent
C18 = Path("runs/campaigns/18-eval-v4-final")
C19 = Path("runs/campaigns/19-stress-geometry/stress_geometry/volumes")
C20 = Path("runs/campaigns/20-ood-conditioning/ood_conditioning/volumes")
UM = 25.0


def richest(label, axis):
    other = tuple(i for i in range(3) if i != axis)
    return int(np.argmax((label == 1).sum(axis=other)))


def take(vol, axis, idx):
    return np.take(vol, idx, axis=axis)


def crop_center(a, n):
    z, y, x = [(s - n) // 2 for s in a.shape[:3]]
    return a[z:z + n, y:y + n, x:x + n]


def seq_label(layup):
    return "[" + "/".join(str(a) for a in layup) + "]"


# ---------------------------------------------------------------- fig 2
def fig2(out):
    d = out / "fig2"; (d / "slices").mkdir(parents=True, exist_ok=True)
    r = json.load(open(C18 / "porosity_global/results.json"))
    rows = [(c["requested_global_phi"], c["phi_pore"]) for c in r["per_case"]
            if c["requested_global_phi"] <= 0.10]          # 0.15 is above the training range (clamped) — excluded from the fit
    with open(d / "porosity.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["requested_porosity", "measured_porosity"]); w.writerows(rows)
    levels = ["0.005", "0.01", "0.03", "0.05", "0.1"]
    for i, lv in enumerate(levels):
        cd = C18 / f"porosity_global/volumes/target{lv}_ddim200_seed101"
        lab = tifffile.imread(cd / "label.tif"); g = tifffile.imread(cd / "volume.tif")
        np.save(d / "slices" / f"{i}_phi{lv}.npy", take(g, 1, richest(lab, 1)))      # through-thickness (y) slice
    return ["--csv", str(d / "porosity.csv"), "--slices", str(d / "slices"),
            "--slice-porosity", ",".join(levels)]


# ---------------------------------------------------------------- fig 3
def fig3(out):
    d = out / "fig3"; (d / "panels").mkdir(parents=True, exist_ok=True)
    # top row: through-thickness (y) slice, 192 deep x 384 wide, of trained and never-seen sequences
    seqs = [(C18 / "layup/volumes/A_seed101", "A (trained)"),
            (C18 / "layup/volumes/C_seed101", "C (trained)"),
            (C20 / "seq_quasi_iso_seed101", "quasi-isotropic (unseen)"),
            (C20 / "seq_crossply_seed101", "cross-ply (unseen)"),
            (C20 / "seq_blocked_seed101", "blocked (unseen)")]
    rows = []
    for i, (cd, name) in enumerate(seqs):
        m = json.load(open(cd / "manifest.json"))
        g = tifffile.imread(cd / "volume.tif"); lab = tifffile.imread(cd / "label.tif")
        y = richest(lab, 1); sl = take(g, 1, y)[:, 320:704]          # 192 x 384
        fn = f"seq_{i}.npy"; np.save(d / "panels" / fn, sl)
        rows.append((fn, "sequence", f"{name}\n{seq_label(m['requested_layup'])}"))
    # bottom row: ply pitch (thickness) and off-manifold shapes
    shapes = [(C20 / "pitch8_seed101", 1, None, "thin plies, 8 vox (0.2 mm)", (slice(None), slice(320, 704))),
              (C20 / "pitch32_seed101", 1, None, "thick plies, 32 vox (0.8 mm)", (slice(None), slice(320, 704))),
              (C19 / "lbracket_ddim50", 2, None, "L-bracket", (slice(None), slice(None))),
              (C19 / "taper_ddim50", 1, None, "tapered coupon", (slice(None), slice(256, 768))),
              (C19 / "letters_ddim50", 0, None, "letters (air)", "air_bbox")]
    for i, (cd, axis, _, name, crop) in enumerate(shapes):
        g = tifffile.imread(cd / "volume.tif"); lab = tifffile.imread(cd / "label.tif")
        idx = richest(lab, axis) if axis == 1 and "pitch" in cd.name else g.shape[axis] // 2
        sl = take(g, axis, idx)
        if crop == "air_bbox":
            zz, yy = np.nonzero(take(lab, axis, idx) == 2); pad = 60
            crop = (slice(max(zz.min() - pad, 0), zz.max() + pad), slice(max(yy.min() - pad, 0), yy.max() + pad))
        sl = sl[crop]
        fn = f"shape_{i}.npy"; np.save(d / "panels" / fn, sl)
        rows.append((fn, "shape", name))
    with open(d / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "row", "label"]); w.writerows(rows)
    return ["--images", str(d / "panels"), "--manifest", str(d / "manifest.csv")]


# ---------------------------------------------------------------- fig 4
def pore_diameters(label):
    cc, n = ndimage.label(label == 1)
    if n == 0:
        return np.array([])
    v = np.bincount(cc.ravel())[1:]
    return (6 * v / np.pi) ** (1 / 3) * UM


def fig4(out):
    d = out / "fig4"; (d / "real").mkdir(parents=True, exist_ok=True); (d / "synth").mkdir(exist_ok=True)
    levels = ["0.01", "0.03", "0.06"]
    pr, ps = [], []
    for i, lv in enumerate(levels):
        # displayed pair: real Na_05 crop 'a' vs generated seed 101 (central 128^3), richest in-plane slice each
        rd = C18 / f"real_floor/volumes/micro_phi{lv}__Na_05__a"; gd = C18 / f"microstructure/volumes/phi{lv}_seed101"
        rg, rl = tifffile.imread(rd / "volume.tif"), tifffile.imread(rd / "label.tif")
        gg, gl = crop_center(tifffile.imread(gd / "volume.tif"), 128), crop_center(tifffile.imread(gd / "label.tif"), 128)
        np.save(d / "real" / f"{i}_phi{lv}.npy", take(rg, 1, richest(rl, 1))); np.save(d / "synth" / f"{i}_phi{lv}.npy", take(gg, 1, richest(gl, 1)))
    # distributions: all 18 real reference crops vs all 9 generated volumes (central 128^3), same three phi levels
    rng = np.random.default_rng(0); gr, gs = [], []
    for p in sorted((C18 / "real_floor/volumes").glob("micro_phi*")):
        lab = tifffile.imread(p / "label.tif"); pr.append(pore_diameters(lab))
        v = tifffile.imread(p / "volume.tif")[lab != 2]; gr.append(rng.choice(v, 200_000, replace=False))
    for p in sorted((C18 / "microstructure/volumes").glob("phi*_seed*")):
        lab = crop_center(tifffile.imread(p / "label.tif"), 128); ps.append(pore_diameters(lab))
        v = crop_center(tifffile.imread(p / "volume.tif"), 128)[lab != 2]; gs.append(rng.choice(v, 400_000, replace=False))
    np.save(d / "grey_real.npy", np.concatenate(gr)); np.save(d / "grey_synth.npy", np.concatenate(gs))
    for name, arr in (("pores_real.csv", np.concatenate(pr)), ("pores_synth.csv", np.concatenate(ps))):
        with open(d / name, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pore_size_um"]); w.writerows([[f"{x:.2f}"] for x in arr])
    print(f"pores: real n={sum(map(len, pr))}  synthetic n={sum(map(len, ps))}")
    return ["--real", str(d / "real"), "--synth", str(d / "synth"),
            "--pores-real", str(d / "pores_real.csv"), "--pores-synth", str(d / "pores_synth.csv"),
            "--grey-real", str(d / "grey_real.npy"), "--grey-synth", str(d / "grey_synth.npy")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(C18 / "figures_review"))
    a = ap.parse_args(); out = Path(a.out); data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    jobs = [("fig2_porosity_control", fig2(data)), ("fig3_sequence_shape", fig3(data)), ("fig4_real_vs_synthetic", fig4(data))]
    for script, args in jobs:
        subprocess.run([sys.executable, str(HERE / f"{script}.py"), *args, "--out", str(out / script)], check=True)
    subprocess.run([sys.executable, str(HERE / "fig4_real_vs_synthetic.py"), *jobs[2][1], "--blind", "--seed", "7",
                    "--out", str(out / "fig4_real_vs_synthetic_blind")], check=True)


if __name__ == "__main__":
    main()

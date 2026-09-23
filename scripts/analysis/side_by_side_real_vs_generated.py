"""Side-by-side figures: a real test crop vs a generated volume at matched porosity and layup.

Matched pairs come from campaign 18: the microstructure assessment generated 192^3 volumes at
phi 0.01 / 0.03 / 0.06 (layup A, DDIM-200) and the real floor cut 128^3 reference crops at the
same phi levels from the test panels (Na_05 is a layup-A panel).  The generated volume is
cropped to its central 128^3 so both sides are shown at the same physical size (25 um/voxel).
A second figure compares a real 128x1024x1024 test crop with a generated 1024x1024x192 volume
(porosity NOT matched: the real crop is what the panel contains, the generated one was asked
for 0.03).  Slices shown are the ones holding the most pore voxels along each axis (same rule for both sides); grey only (grey=material, red=pore,
blue=air).  Output: <campaign>/figures_side_by_side/*.png.  CPU only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

LABEL_CMAP = ListedColormap([(0.75, 0.75, 0.75), (0.85, 0.10, 0.10), (0.20, 0.40, 0.85)])

def load(d: Path):
    g = tifffile.imread(d / "volume.tif"); l = tifffile.imread(d / "label.tif")
    return g, l, json.load(open(d / "manifest.json"))

def crop_center(a, n):
    z, y, x = [(s - n) // 2 for s in a.shape[:3]]
    return a[z:z + n, y:y + n, x:x + n]

def richest_index(label, axis):
    """Index along `axis` of the slice holding the most pore voxels (a fair 'show the pores' choice for both sides)."""
    other = tuple(i for i in range(3) if i != axis)
    return int(np.argmax((label == 1).sum(axis=other)))

def mid_slices(a):
    z, y, x = a.shape
    return {"z": a[z // 2], "y": a[:, y // 2, :], "x": a[:, :, x // 2]}

def phi_of(l):
    m = (l == 0).sum(); p = (l == 1).sum()
    return p / max(m + p, 1)

def panel(ax, img, label=None, title=""):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="equal")
    if label is not None:
        ax.imshow(np.ma.masked_where(label == 0, label), cmap=LABEL_CMAP, vmin=0, vmax=2, alpha=0.55, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

def matched_figure(camp: Path, level: str, panel_id: str, seed: int, out: Path):
    real_d = camp / "real_floor" / "volumes" / f"micro_phi{level}__{panel_id}__a"
    gen_d = camp / "microstructure" / "volumes" / f"phi{level}_seed{seed}"
    rg, rl, _ = load(real_d); gg, gl, _ = load(gen_d)
    gg, gl = crop_center(gg, 128), crop_center(gl, 128)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
    for c, ax in enumerate(("z", "y", "x")):
        panel(axes[0, c], mid_slices(rg)[ax], None, f"real, {ax}" if c else f"real (phi {phi_of(rl):.3f}), {ax}")
        panel(axes[1, c], mid_slices(gg)[ax], None, f"generated, {ax}" if c else f"generated (phi {phi_of(gl):.3f}), {ax}")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def wide_figure(camp: Path, out: Path, real_name: str = "Na_09_5"):
    real_d = next(p for p in sorted((camp / "real_floor" / "volumes").iterdir()) if p.name.endswith("__large") and real_name in p.name)
    gen_d = camp / "sampler" / "volumes" / "1024_ddim200_seed101"
    rg, rl, _ = load(real_d); gg, gl, _ = load(gen_d)
    # mid-z (in-plane) view
    rz, gz = richest_index(rl, 0), richest_index(gl, 0)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.6))
    panel(axes[0], rg[rz], None, f"real, test panel {real_name}, in-plane slice z={rz} (phi {phi_of(rl):.3f})")
    panel(axes[1], gg[gz], None, f"generated, request phi 0.03, in-plane slice z={gz} (phi {phi_of(gl):.3f})")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    # through-thickness (side) view: y-slice, plies run left-right; both volumes 192 deep
    fig, axes = plt.subplots(2, 1, figsize=(16, 5.2))
    ry, gy = richest_index(rl, 1), richest_index(gl, 1)
    panel(axes[0], rg[:, ry, :], None, f"real, {real_name}, through-thickness slice y={ry} (phi {phi_of(rl):.3f})")
    panel(axes[1], gg[:, gy, :], None, f"generated, through-thickness slice y={gy} (phi {phi_of(gl):.3f})")
    fig.tight_layout(); fig.savefig(out.with_name(out.stem + "_through_thickness.png"), dpi=150); plt.close(fig)
    # second side view along the other in-plane axis
    fig, axes = plt.subplots(2, 1, figsize=(16, 5.2))
    rx, gx = richest_index(rl, 2), richest_index(gl, 2)
    panel(axes[0], rg[:, :, rx], None, f"real, {real_name}, through-thickness slice x={rx} (phi {phi_of(rl):.3f})")
    panel(axes[1], gg[:, :, gx], None, f"generated, through-thickness slice x={gx} (phi {phi_of(gl):.3f})")
    fig.tight_layout(); fig.savefig(out.with_name(out.stem + "_through_thickness_x.png"), dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--campaign", default="runs/campaigns/18-eval-v4-final"); ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args(); camp = Path(a.campaign); out = camp / "figures_side_by_side"; out.mkdir(exist_ok=True)
    for level in ("0.01", "0.03", "0.06"):
        matched_figure(camp, level, "Na_05", a.seed, out / f"matched_phi{level}_Na05_vs_generated.png")
    wide_figure(camp, out / "wide_real_Na09_5_vs_generated_1024.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))

if __name__ == "__main__":
    main()

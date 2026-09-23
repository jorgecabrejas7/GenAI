"""Side-by-side figures: a real test crop vs a generated volume at matched porosity and layup.

Matched pairs come from campaign 18: the microstructure assessment generated 192^3 volumes at
phi 0.01 / 0.03 / 0.06 (layup A, DDIM-200) and the real floor cut 128^3 reference crops at the
same phi levels from the test panels (Na_05 is a layup-A panel).  The generated volume is
cropped to its central 128^3 so both sides are shown at the same physical size (25 um/voxel).
A second figure compares a real 128x1024x1024 test crop with a generated 1024x1024x192 volume
(porosity NOT matched: the real crop is what the panel contains, the generated one was asked
for 0.03).  Three orthogonal mid-slices per volume, grey and label (grey=material, red=pore,
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
    rg, rl, rm = load(real_d); gg, gl, gm = load(gen_d)
    gg, gl = crop_center(gg, 128), crop_center(gl, 128)
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 14))
    rows = [("REAL grey\n(test panel %s, layup A)" % panel_id, rg, None), ("REAL label", rg, rl),
            ("GENERATED grey\n(request phi %s, layup A, DDIM-200)" % level, gg, None), ("GENERATED label", gg, gl)]
    for r, (name, g, l) in enumerate(rows):
        for c, ax in enumerate(("z", "y", "x")):
            panel(axes[r, c], mid_slices(g)[ax], None if l is None else mid_slices(l)[ax], f"{name}\n{ax}-slice" if c == 0 else f"{ax}-slice")
    fig.suptitle(f"Real vs generated at matched porosity — real crop phi {phi_of(rl):.4f}, generated phi {phi_of(gl):.4f}\n(central 128³ of the 192³ generated volume; 25 µm/voxel; red = pore, blue = air)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=200); plt.close(fig)

def wide_figure(camp: Path, out: Path):
    real_d = next(p for p in sorted((camp / "real_floor" / "volumes").iterdir()) if p.name.endswith("__large") and "Na_05" in p.name)
    gen_d = camp / "sampler" / "volumes" / "1024_ddim200_seed101"
    rg, rl, _ = load(real_d); gg, gl, _ = load(gen_d)
    fig, axes = plt.subplots(4, 1, figsize=(16, 13))
    rz, gz = rg[rg.shape[0] // 2], gg[gg.shape[0] // 2]
    panel(axes[0], rz, None, f"REAL  test panel Na_05, 128×1024×1024 crop, mid-z slice (phi {phi_of(rl):.4f})")
    panel(axes[1], rz, rl[rl.shape[0] // 2], "REAL  label")
    panel(axes[2], gz, None, f"GENERATED  1024×1024×192, request phi 0.03, layup A, DDIM-200, mid-z slice (phi {phi_of(gl):.4f})")
    panel(axes[3], gz, gl[gl.shape[0] // 2], "GENERATED  label")
    fig.suptitle("Full panel width, same voxel size (25 µm). Porosity is NOT matched here: the real crop holds what the panel contains.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=150); plt.close(fig)
    # through-thickness (y-slice) strip: plies visible
    fig, axes = plt.subplots(4, 1, figsize=(16, 7))
    ry, gy = rg[:, rg.shape[1] // 2, :], gg[:, gg.shape[1] // 2, :]
    panel(axes[0], ry, None, "REAL  through-thickness (y-slice): plies run left–right"); panel(axes[1], ry, rl[:, rl.shape[1] // 2, :], "REAL  label")
    panel(axes[2], gy, None, "GENERATED  through-thickness (y-slice)"); panel(axes[3], gy, gl[:, gl.shape[1] // 2, :], "GENERATED  label")
    fig.tight_layout(); fig.savefig(out.with_name(out.stem + "_through_thickness.png"), dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--campaign", default="runs/campaigns/18-eval-v4-final"); ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args(); camp = Path(a.campaign); out = camp / "figures_side_by_side"; out.mkdir(exist_ok=True)
    for level in ("0.01", "0.03", "0.06"):
        matched_figure(camp, level, "Na_05", a.seed, out / f"matched_phi{level}_Na05_vs_generated.png")
    wide_figure(camp, out / "wide_real_Na05_vs_generated_1024.png")
    print("wrote", sorted(p.name for p in out.glob("*.png")))

if __name__ == "__main__":
    main()

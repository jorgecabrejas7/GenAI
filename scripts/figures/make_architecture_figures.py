#!/usr/bin/env python
"""Paper-style architecture figures for the two trained networks.

Emits, for each of r08 (the VAE) and ldm06 (the latent-diffusion denoiser), an
``.svg``, a 300-dpi ``.png`` and a ``.pdf``, drawn in matplotlib so the figures
are reproducible from committed code rather than hand-edited in a vector tool.

Every number in the figures is read off the code and the resolved run configs,
except the receptive-field measurements, which this script MEASURES on a freshly
built r08 (``--measure``) rather than deriving them by hand:

    one latent cell  <- 22^3 input voxels     (encoder receptive field)
    one latent cell  -> 22^3 output voxels    (decoder footprint)
    one output voxel <- 6^3 latent cells      (= 1728 latent values)

The 4-voxel (100 um) figure people reach for is the latent grid PITCH, not a
cell extent: the cells overlap heavily and no channel owns a position.  Saying
"100 um cells" implies a block-average of the volume, which this is not.

Usage
-----
    python scripts/figures/make_architecture_figures.py
    python scripts/figures/make_architecture_figures.py --measure   # re-verify RF
    python scripts/figures/make_architecture_figures.py --out /tmp/figs
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "docs" / "figures"

# ── style ────────────────────────────────────────────────────────────────────
# Figures are laid out in SVG-like units with y running DOWNWARDS (the y axis is
# inverted below), so box coordinates read top-left origin.  Font sizes are tied
# to those units by PT_PER_UNIT, so the type scale survives a change of figsize.

FIG_WIDTH_IN = 13.0
CANVAS_W = 1140.0
PT_PER_UNIT = FIG_WIDTH_IN / CANVAS_W * 72.0

FS_TITLE = 13.0 * PT_PER_UNIT
FS_SUB = 11.5 * PT_PER_UNIT
FS_MONO = 11.0 * PT_PER_UNIT
FS_SMALL = 10.0 * PT_PER_UNIT
FS_EYEBROW = 9.5 * PT_PER_UNIT

SANS = ["Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"]
MONO = ["DejaVu Sans Mono", "Liberation Mono", "monospace"]

INK = "#12161b"
INK_2 = "#4d555e"
INK_3 = "#79838d"
EDGE = "#9aa3ac"

# Categorical hues, validated colourblind-safe as a three-slot set
# (OKLab dE: worst pair 9.2 CVD / 24.0 normal vision).
LATENT = "#2a78d6"
VOXEL = "#1baf7a"
COND = "#eb6834"

FILL = {
    "net": ("#f1f3f5", EDGE, 0.9),
    "inner": ("#e5e9ec", EDGE, 0.7),
    "latent": ("#e6effb", LATENT, 1.1),
    "voxel": ("#e2f4ec", VOXEL, 1.1),
    "cond": ("#fdece4", COND, 1.1),
    "chip": ("#ffffff", EDGE, 0.9),
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep text as text, editable in Inkscape
    })


# ── primitives ───────────────────────────────────────────────────────────────

def canvas(height: float):
    """A figure whose data coordinates are SVG-like: origin top-left, y down."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * height / CANVAS_W))
    # The axes must FILL the figure.  With the default subplot margins the data
    # box is only ~0.775 of the figure width, so PT_PER_UNIT understates the
    # rendered type by 1.29x and every label overruns its box.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, kind="net", dashed=False):
    face, edge, lw = FILL[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=3",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (3.5, 2.5)) if dashed else "solid",
        mutation_aspect=1, zorder=2,
    ))


def label(ax, x, y, s, size=FS_SUB, color=INK_2, weight="normal",
          mono=False, ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="baseline", zorder=4,
            fontfamily=MONO if mono else SANS)


def title(ax, x, y, s):
    label(ax, x, y, s, size=FS_TITLE, color=INK, weight="bold")


def mono(ax, x, y, s, size=FS_MONO, color=INK_2, ha="left"):
    label(ax, x, y, s, size=size, color=color, mono=True, ha=ha)


def eyebrow(ax, x, y, s, ha="left"):
    ax.text(x, y, s, fontsize=FS_EYEBROW, color=INK_3, ha=ha, va="baseline",
            fontfamily=MONO, zorder=4)


def line(ax, pts, dashed=False, color=None, lw=None):
    color = color or (INK_3 if dashed else INK_2)
    lw = lw or (0.9 if dashed else 1.15)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            ls=(0, (4, 3)) if dashed else "solid", solid_capstyle="round",
            zorder=3)


def arrow(ax, pts, dashed=False, color=None, lw=None):
    """Polyline with a solid arrowhead on the far end.

    The head is drawn as its own short annotation with ``mutation_scale=1`` so
    it is sized in points: the default scale is the ambient font size, which at
    this data scale draws a 50-unit head.  Drawing it separately also keeps the
    head solid on a dashed line.
    """
    color = color or (INK_3 if dashed else INK_2)
    lw = lw or (0.9 if dashed else 1.15)
    line(ax, pts, dashed, color, lw)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    tail = (x1 - (x1 - x0) / d * 0.1, y1 - (y1 - y0) / d * 0.1)
    ax.annotate("", xy=(x1, y1), xytext=tail, zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=8,head_width=3.6",
                                color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=1))


def legend(ax, x, y, items):
    """Inline swatch legend along one row."""
    for text_, kind in items:
        if kind == "sup":
            ax.plot([x, x + 22], [y - 4, y - 4], color=INK_3, lw=1.1,
                    ls=(0, (4, 3)), zorder=3)
        else:
            box(ax, x, y - 11, 22, 11, kind)
        label(ax, x + 29, y, text_, size=FS_SMALL, color=INK_2)
        x += 29 + len(text_) * FS_SMALL / PT_PER_UNIT * 0.50 + 26


# ── figure 1 — r08 VAE ───────────────────────────────────────────────────────

def figure_vae():
    fig, ax = canvas(500)

    # --- main row -----------------------------------------------------------
    ROW_Y, ROW_H = 116, 150
    cols = [(16, 140), (178, 200), (400, 140), (562, 172), (756, 156), (934, 192)]
    tensors = ["(3, 64³)", "2 × (64, 16³)", "μ, log σ²", "(8, 16³)", "(32, 64³)"]

    for (x0, w0), (x1, _), t in zip(cols, cols[1:], tensors):
        a, b = x0 + w0, x1
        arrow(ax, [(a, ROW_Y + 55), (b, ROW_Y + 55)])
        mono(ax, (a + b) / 2, ROW_Y - 10, t, size=FS_SMALL, color=INK_3, ha="center")

    # 1 · input
    x, w = cols[0]
    box(ax, x, ROW_Y, w, ROW_H, "voxel")
    title(ax, x + 14, ROW_Y + 26, "Patch input")
    label(ax, x + 14, ROW_Y + 50, "XCT grey 64³")
    label(ax, x + 14, ROW_Y + 68, "pore plane")
    label(ax, x + 14, ROW_Y + 86, "air plane")
    mono(ax, x + 14, ROW_Y + 112, "cat → 3 ch")
    mono(ax, x + 14, ROW_Y + 132, "1.6 mm cube", FS_SMALL, INK_3)

    # 2 · dual encoder
    x, w = cols[1]
    box(ax, x, ROW_Y, w, ROW_H, "net")
    title(ax, x + 12, ROW_Y + 26, "Dual encoder")
    for i, (name, note) in enumerate([
        ("Branch A", "3→32 @32³ · 32→64 @16³"),
        ("Branch B", "independent weights"),
    ]):
        yy = ROW_Y + 38 + i * 54
        box(ax, x + 12, yy, w - 24, 46, "inner")
        label(ax, x + 22, yy + 18, name)
        mono(ax, x + 22, yy + 36, note, FS_SMALL, INK_3)

    # 3 · bottleneck
    x, w = cols[2]
    box(ax, x, ROW_Y, w, ROW_H, "net")
    title(ax, x + 12, ROW_Y + 26, "Bottleneck")
    label(ax, x + 12, ROW_Y + 52, "concat → 128 ch")
    label(ax, x + 12, ROW_Y + 72, "fusion 1×1×1")
    label(ax, x + 12, ROW_Y + 92, "128 → 64")
    mono(ax, x + 12, ROW_Y + 116, "to_μ · to_logσ²")
    mono(ax, x + 12, ROW_Y + 136, "Conv 1×1×1 → 8", FS_SMALL, INK_3)

    # 4 · latent
    x, w = cols[3]
    box(ax, x, ROW_Y, w, ROW_H, "latent")
    title(ax, x + 12, ROW_Y + 26, "Latent z")
    mono(ax, x + 12, ROW_Y + 50, "z = μ + σ⊙ε")
    label(ax, x + 12, ROW_Y + 74, "8 × 16³ values")
    label(ax, x + 12, ROW_Y + 94, "grid pitch 4 vox")
    mono(ax, x + 12, ROW_Y + 116, "= 100 µm spacing", FS_SMALL, INK_3)
    mono(ax, x + 12, ROW_Y + 134, "RF 22³ vox (550 µm)", FS_SMALL, INK_3)

    # 5 · decoder
    x, w = cols[4]
    box(ax, x, ROW_Y, w, ROW_H, "net")
    title(ax, x + 12, ROW_Y + 26, "Decoder")
    label(ax, x + 12, ROW_Y + 52, "up 8→64 @ 32³")
    label(ax, x + 12, ROW_Y + 72, "up 64→32 @ 64³")
    mono(ax, x + 12, ROW_Y + 98, "trilinear ×2 up", FS_SMALL, INK_3)
    mono(ax, x + 12, ROW_Y + 116, "Conv 3³ · BN · GELU", FS_SMALL, INK_3)
    mono(ax, x + 12, ROW_Y + 134, "no skip connections", FS_SMALL, INK_3)

    # 6 · heads
    x, w = cols[5]
    box(ax, x, ROW_Y, w, ROW_H, "voxel")
    title(ax, x + 12, ROW_Y + 26, "Output heads")
    for i, (name, note) in enumerate([
        ("xct_head → (1, 64³)", "grey level"),
        ("class_head → (3, 64³)", "material · pore · air"),
    ]):
        yy = ROW_Y + 38 + i * 54
        box(ax, x + 12, yy, w - 24, 46, "inner")
        label(ax, x + 22, yy + 18, name)
        mono(ax, x + 22, yy + 36, note, FS_SMALL, INK_3)

    BOT = ROW_Y + ROW_H

    # --- supervision --------------------------------------------------------
    CH_Y, CH_H = 340, 76
    eyebrow(ax, 380, CH_Y - 14, "TRAINING OBJECTIVE")

    arrow(ax, [(470, BOT), (470, CH_Y - 4)], dashed=True)
    arrow(ax, [(1030, BOT), (1030, CH_Y - 4)], dashed=True)
    arrow(ax, [(1000, BOT), (1000, 296), (707, 296), (707, CH_Y - 4)], dashed=True)
    mono(ax, 855, 290, "decoded grey slices → “fake”", FS_SMALL, INK_3, ha="center")
    arrow(ax, [(86, BOT), (86, 452), (707, 452), (707, CH_Y + CH_H + 4)], dashed=True)
    mono(ax, 330, 444, "real 2-D slices → “real” · 3 orthogonal planes",
         FS_SMALL, INK_3)

    chips = [
        (380, 180, "KL", "free-bits 0.1", "β = 0.05, no warm-up"),
        (602, 210, "PatchGAN discriminator", "2-D, spectral norm, base 64",
         "LSGAN · w = 0.05 · fp32"),
        (886, 240, "Reconstruction", "Charbonnier on grey · w 1.0",
         "weighted CE + soft Dice(pore, air)"),
    ]
    for x, w, t, s1, s2 in chips:
        box(ax, x, CH_Y, w, CH_H, "chip", dashed=True)
        title(ax, x + 12, CH_Y + 24, t)
        label(ax, x + 12, CH_Y + 45, s1)
        mono(ax, x + 12, CH_Y + 65, s2)

    # --- figure note --------------------------------------------------------
    label(ax, 16, 488,
          "Note — the latent grid pitch is 4 voxels (100 µm), but a cell is not a 4³ block: "
          "one cell's receptive field is 22³ voxels (550 µm), it writes back over 22³ voxels, "
          "and 6³ = 216 cells reach every output voxel.",
          FS_SMALL, INK_3)

    # --- header -------------------------------------------------------------
    label(ax, 16, 30, "r08 — dual-branch 3-class VAE", FS_TITLE * 1.35, INK, "bold")
    mono(ax, 16, 52, "v2.conv_noattn_dualbranch_cls  ·  z=8, base=32  ·  769 k params"
                     "  ·  runs/vae/r08-run-0004", FS_MONO, INK_3)
    legend(ax, 16, 88, [("voxel space (64³)", "voxel"),
                        ("latent space (16³)", "latent"),
                        ("network stage", "net"),
                        ("training-time supervision", "sup")])
    return fig


# ── figure 2 — ldm06 denoiser ────────────────────────────────────────────────

def figure_ldm():
    fig, ax = canvas(748)
    TOP = 72   # vertical offset that clears the header block above the drawing

    def y(v):
        return v + TOP

    # --- spatial input rail -------------------------------------------------
    eyebrow(ax, 16, y(40), "SPATIAL INPUTS — CONCATENATED AT 16³")
    rows = [
        ("z_t — noisy latent", "8", "latent"),
        ("cond_orient — (cos 2θ, sin 2θ)", "2", "cond"),
        ("cond_material — envelope fraction", "1", "cond"),
        ("6 neighbour latents, each at own t", "48", "cond"),
        ("6 availability embeddings", "48", "cond"),
        ("6 neighbour-timestep embeddings", "48", "cond"),
    ]
    for i, (text_, n, kind) in enumerate(rows):
        yy = y(52 + i * 38)
        box(ax, 16, yy, 270, 32, kind)
        label(ax, 28, yy + 20, text_)
        label(ax, 274, yy + 20, n, FS_MONO, INK, "bold", mono=True, ha="right")
        line(ax, [(286, yy + 16), (322, y(88))])
    mono(ax, 16, y(292), "OOB · EXISTS · UNKNOWN, one row per face",
         FS_SMALL, INK_3)
    arrow(ax, [(316, y(88)), (330, y(88))])
    label(ax, 330, y(48), "155 ch", FS_MONO, INK, "bold", mono=True)

    # --- scalar conditioning rail ------------------------------------------
    eyebrow(ax, 16, y(318), "SCALAR CONDITIONING — ADAGN")
    srows = [
        ("t — diffusion step", "sin 256 → MLP", "net"),
        ("φ — log porosity, standardised", "+ null token", "cond"),
        ("depth — relative, in [0, 1]", "MLP", "cond"),
        ("dist6 — distance to each face", "one MLP", "cond"),
        ("pooled EXISTS neighbours", "6×8 → MLP", "cond"),
    ]
    for i, (text_, note, kind) in enumerate(srows):
        yy = y(330 + i * 33)
        box(ax, 16, yy, 270, 28, kind)
        label(ax, 28, yy + 19, text_)
        mono(ax, 274, yy + 19, note, FS_SMALL, INK_3, ha="right")
        line(ax, [(286, yy + 14), (299, y(410))])

    ax.add_patch(Circle((310, y(410)), 11, facecolor="white", edgecolor=INK_2,
                        lw=1.1, zorder=3))
    label(ax, 310, y(415), "+", FS_TITLE, INK, "bold", ha="center")

    # cond bus into the U
    line(ax, [(310, y(399)), (310, y(182))])
    arrow(ax, [(310, y(182)), (330, y(182))])
    arrow(ax, [(310, y(286)), (500, y(286))])
    arrow(ax, [(310, y(397)), (670, y(397))])
    mono(ax, 330, y(440), "cond (512) → AdaGN in every ResBlock")

    # --- the U --------------------------------------------------------------
    box(ax, 330, y(58), 150, 60, "net")
    title(ax, 342, y(82), "input_proj")
    mono(ax, 342, y(103), "Conv 3³ · 155 → 128")
    arrow(ax, [(405, y(118)), (405, y(150))])

    levels = [
        (330, 150, 150, 64, "2 × ResBlock", "128 ch @ 16³"),
        (500, 254, 150, 64, "2 × ResBlock", "256 ch @ 8³"),
        (880, 254, 150, 64, "2 × ResBlock", "768 → 256 @ 8³"),
        (960, 150, 150, 64, "2 × ResBlock", "384 → 128 @ 16³"),
    ]
    for x, yy, w, h, t, s in levels:
        box(ax, x, y(yy), w, h, "net")
        title(ax, x + 12, y(yy + 26), t)
        mono(ax, x + 12, y(yy + 48), s)

    box(ax, 670, y(358), 190, 78, "net")
    title(ax, 682, y(384), "Bottleneck")
    mono(ax, 682, y(405), "2 × ResBlock → 512")
    mono(ax, 682, y(424), "+ ResBlock @ 4³ · no attention", FS_SMALL, INK_3)

    arrow(ax, [(470, y(214)), (512, y(256))])
    mono(ax, 464, y(248), "Conv 4³ s2", FS_SMALL, INK_3, ha="right")
    arrow(ax, [(640, y(318)), (682, y(360))])
    mono(ax, 634, y(352), "Conv 4³ s2", FS_SMALL, INK_3, ha="right")
    arrow(ax, [(864, y(396)), (938, y(322))])
    mono(ax, 872, y(404), "up ×2 · cat skip", FS_SMALL, INK_3)
    arrow(ax, [(1000, y(254)), (1038, y(216))])
    mono(ax, 1126, y(242), "up ×2 · cat skip", FS_SMALL, INK_3, ha="right")
    arrow(ax, [(1035, y(150)), (1035, y(120))])

    box(ax, 960, y(58), 150, 60, "latent")
    title(ax, 972, y(82), "output_proj → v̂")
    mono(ax, 972, y(103), "GN·SiLU·Conv 1³")

    arrow(ax, [(480, y(166)), (958, y(166))], dashed=True)
    mono(ax, 720, y(158), "skip · 128 ch @ 16³", FS_SMALL, INK_3, ha="center")
    arrow(ax, [(650, y(270)), (878, y(270))], dashed=True)
    mono(ax, 764, y(262), "skip · 256 ch @ 8³", FS_SMALL, INK_3, ha="center")

    # --- generation band ----------------------------------------------------
    ax.plot([16, 1126], [y(516)] * 2, color="#dfe3e7", lw=0.9, zorder=1)
    eyebrow(ax, 16, y(542), "GENERATION PATH — WHAT THE DENOISER IS CALLED FROM")

    band = [
        (16, 262, "v̂ → x̂₀", "cosine schedule, T = 1000",
         "v-prediction · zero terminal SNR", "recovers x̂₀ with no division", "chip"),
        (300, 248, "DDIM · 50 steps", "nested classifier-free guidance",
         "s_por · s_nb", "combined on raw network output", "chip"),
        (570, 280, "Chunked joint denoising", "3×3×3 tiles = 192³ voxels per chunk",
         "32-voxel window stride", "faces read from the live canvas", "chip"),
        (872, 254, "Frozen r08 decoder", "blended decode, 32-voxel stride",
         "grey volume + 3-class label", "argmax after blending", "voxel"),
    ]
    for x, w, t, s1, s2, s3, kind in band:
        box(ax, x, y(556), w, 88, kind, dashed=(kind == "chip"))
        title(ax, x + 12, y(580), t)
        label(ax, x + 12, y(601), s1)
        mono(ax, x + 12, y(621), s2)
        mono(ax, x + 12, y(637), s3, FS_SMALL, INK_3)
    for x in (278, 548, 850):
        arrow(ax, [(x, y(600)), (x + 22, y(600))])

    # --- figure note --------------------------------------------------------
    label(ax, 16, 738,
          "Note — the input width is 155 channels at z=8. The 127 quoted in the "
          "UNet3DConfig docstring and the ldm06/base YAML is the figure at z=4; "
          "the code computes it, so only those comments are stale.",
          FS_SMALL, INK_3)

    # --- header -------------------------------------------------------------
    label(ax, 16, 30, "ldm06 — conditional 3-D U-Net denoiser",
          FS_TITLE * 1.35, INK, "bold")
    mono(ax, 16, 52, "UNet3DDenoiser  ·  155 ch in @ 16³  ·  128/256/512  ·  83.0 M "
                     "params  ·  v-prediction  ·  runs/ldm/ldm06-run-0001",
         FS_MONO, INK_3)
    legend(ax, 16, 88, [("latent space", "latent"),
                        ("conditioning & control", "cond"),
                        ("network stage", "net"),
                        ("voxel space", "voxel"),
                        ("skip / non-flow", "sup")])
    return fig


# ── receptive-field measurement ──────────────────────────────────────────────

def measure_receptive_field() -> None:
    """Measure, rather than derive, the numbers the latent callout states."""
    import torch

    from poregen.models.vae.registry import build_vae

    m = build_vae("v2.conv_noattn_dualbranch_cls", in_channels=3, z_channels=8,
                  base_channels=32, n_blocks=2, patch_size=64).eval()

    def extent(t):
        idx = (t.abs() > 0).nonzero()
        return [int(idx[:, d].max() - idx[:, d].min()) + 1 for d in (0, 1, 2)]

    x = torch.zeros(1, 1, 64, 64, 64, requires_grad=True)
    lab = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    mu, _ = m.encode_moments(x, lab)
    mu[0, :, 8, 8, 8].sum().backward()
    print("encoder receptive field of one latent cell :", extent(x.grad[0, 0]))

    with torch.no_grad():
        z0 = torch.zeros(1, 8, 16, 16, 16)
        base = m.decoder(z0)
        z1 = z0.clone()
        z1[0, :, 8, 8, 8] = 1.0
        print("decoder footprint of one latent cell     :",
              extent((m.decoder(z1) - base)[0].abs().sum(0)))
        hits = 0
        for i in range(16):
            zi = z0.clone()
            zi[0, :, i, 8, 8] = 1.0
            if (m.decoder(zi) - base)[0, :, 32, 32, 32].abs().sum() > 0:
                hits += 1
    print(f"latent cells reaching one output voxel   : {hits}^3 = {hits ** 3}"
          f" cells = {hits ** 3 * 8} values")


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT,
                    help=f"output directory (default: {OUT_DEFAULT})")
    ap.add_argument("--measure", action="store_true",
                    help="measure the r08 receptive fields and print them")
    args = ap.parse_args()

    if args.measure:
        measure_receptive_field()

    set_style()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, builder in [("vae_r08_architecture", figure_vae),
                          ("ldm06_architecture", figure_ldm)]:
        fig = builder()
        for ext in ("svg", "png", "pdf"):
            path = args.out / f"{name}.{ext}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(f"wrote {path.relative_to(REPO) if REPO in path.parents else path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

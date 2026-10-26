"""Render every image and video of the group-meeting deck into deck/media/.

Sources: one real scan from raw_data/ (with its reference segmentation) and the
final model's evaluation volumes (campaign folders under runs/campaigns/).
Colours follow the deck: CT greys, amber = pore, blue = air, on #0b0d10.
Videos carry no text; the deck captions them.

Usage: python talks/poregen-group/scripts/render_media.py [name ...]
       (no name renders everything; names are the keys of JOBS)
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "deck" / "media"
FINAL = ROOT / "runs/campaigns/18-eval-v4-final-v4-run0005"
A2S = ROOT / "runs/campaigns/35-a2s-seam-v4"
RAW = ROOT / "raw_data/MedidasDB"
SCAN = "Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_02_2_volume_eq_aligned"
SEG = RAW / "onlypores files"
SETTINGS = "r30_k0.125_min8"            # the reference segmentation the dataset uses

BG = np.array([11, 13, 16], np.uint8)
PORE = np.array([255, 176, 0], np.float32)
AIR = np.array([47, 107, 255], np.float32)
GAP = 24
FPS = 24


# === helpers ================================================================
def stretch(vol: np.ndarray) -> np.ndarray:
    """Map grey to [0, 1] with the 0.5/99.8 percentiles of non-air voxels."""
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v[v > 0] if (v > 0).any() else v, [0.5, 99.8])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)


def grey_rgb(g: np.ndarray) -> np.ndarray:
    """CT grey with a slight cold tint, 20..235 so the panel never hits pure black/white."""
    x = 20 + g * 215
    return np.stack([x, x, x + 4], -1).clip(0, 255)


def overlay(g: np.ndarray, lab: np.ndarray | None, alpha=0.85) -> np.ndarray:
    rgb = grey_rgb(g)
    if lab is not None:
        for cls, col in ((1, PORE), (2, AIR)):
            m = lab == cls
            rgb[m] = rgb[m] * (1 - alpha) + col * alpha
    return rgb.astype(np.uint8)


def row(panels: list[np.ndarray], height: int) -> np.ndarray:
    """Resize panels to one height and join them with GAP pixels of background."""
    out = []
    for p in panels:
        h, w = p.shape[:2]
        im = Image.fromarray(p).resize((round(w * height / h), height), Image.NEAREST if h < height else Image.LANCZOS)
        out.append(np.asarray(im))
        out.append(np.broadcast_to(BG, (height, GAP, 3)))
    frame = np.concatenate(out[:-1], 1)
    pad_w = (-frame.shape[1]) % 16                       # x264 wants even sizes; 16 keeps every player happy
    pad_h = (-frame.shape[0]) % 16
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=11)


def write_video(name: str, frames, pingpong=True):
    frames = list(frames)
    if pingpong:
        frames = frames + frames[-2:0:-1]
    w = imageio.get_writer(OUT / f"{name}.mp4", fps=FPS, codec="libx264", quality=6,
                           pixelformat="yuv420p", macro_block_size=16)
    for f in frames:
        w.append_data(f)
    w.close()
    imageio.imwrite(OUT / f"{name}_poster.jpg", frames[len(frames) // 4], quality=88)
    print(f"{name}.mp4  {len(frames)} frames  {frames[0].shape[1]}x{frames[0].shape[0]}")


def gen(assessment: str, case: str):
    d = FINAL / assessment / "volumes" / case
    return tifffile.imread(d / "volume.tif"), tifffile.imread(d / "label.tif")


def scan_pages(path: Path, zs) -> np.ndarray:
    t = tifffile.TiffFile(path)
    return np.stack([t.pages[z].asarray() for z in zs])


def real_label(zs, ys, xs) -> np.ndarray:
    """Reference segmentation of the scan: 1 = pore, 2 = air (outside the sample mask)."""
    p = scan_pages(SEG / f"{SCAN}_onlypores_{SETTINGS}.tif", zs)[:, ys, xs] > 0
    m = scan_pages(SEG / f"{SCAN}_samplemask_{SETTINGS}.tif", zs)[:, ys, xs] > 0
    lab = np.zeros(p.shape, np.uint8)
    lab[p] = 1
    lab[~m] = 2
    return lab


# === real scan ==============================================================
def title():
    """Title slice (z 110, crop around a drilled hole) and its segmentation as RGBA."""
    zs, ys, xs = [110], slice(400, 1424), slice(300, 1324)
    g = scan_pages(RAW / f"{SCAN}.tif", zs)[0, ys, xs]
    Image.fromarray((stretch(g) * 255).astype(np.uint8)).save(OUT / "title_slice.jpg", quality=86)
    lab = real_label(zs, ys, xs)[0]
    rgba = np.zeros((*lab.shape, 4), np.uint8)
    rgba[lab == 1] = [255, 176, 0, 255]
    rgba[lab == 2] = [47, 107, 255, 255]
    Image.fromarray(rgba).save(OUT / "title_seg.png")


def data_slice():
    """A second region of the same scan: grey and labelled, for the data slide wipe."""
    zs, ys, xs = [55], slice(1900, 2924), slice(500, 1524)
    g = stretch(scan_pages(RAW / f"{SCAN}.tif", zs)[0, ys, xs])
    lab = real_label(zs, ys, xs)[0]
    Image.fromarray(overlay(g, None)).save(OUT / "data_grey.jpg", quality=88)
    Image.fromarray(overlay(g, lab)).save(OUT / "data_seg.jpg", quality=88)


def data_xsec():
    """Through-thickness section (z x) of the scan: plies and inter-ply pores."""
    t = tifffile.TiffFile(RAW / f"{SCAN}.tif")
    y = 1500
    sec = np.stack([p.asarray()[y] for p in t.pages])                 # (z, x)
    Image.fromarray(overlay(stretch(sec), None)).save(OUT / "data_xsec.jpg", quality=88)


def pores3d():
    """Rotating pore projection: a real 192³ crop beside a generated volume at matched porosity."""
    zs = range(14, 206)
    lab_r = real_label(zs, slice(1900, 2092), slice(700, 892))
    phi_r = float((lab_r == 1).mean())
    v, lab_g = gen("porosity_global", "target0.1_ddim50_seed101")
    print(f"pores3d: real crop phi {phi_r:.4f}, generated phi {(lab_g == 1).mean():.4f}")
    write_video("pores3d", spin([lab_r == 1, lab_g == 1], 600), pingpong=False)


def spin(masks, height, n=120, scale=None):
    """Rotating projection of binary pore masks about z (vertical on screen).

    The brightness of a pixel is the pore path length along the view ray.
    One scale is shared by every panel, so equal brightness means equal pore
    length; it defaults to the 99.5th percentile of the densest panel at 0°.
    """
    masks = [m.astype(np.float32) for m in masks]
    if scale is None:
        scale = max(float(np.percentile(m.sum(2), 99.5)) for m in masks) or 1.0
    side = int(np.ceil(max(m.shape[1] for m in masks) * 1.42))
    frames = []
    for ang in np.linspace(0, 360, n, endpoint=False):
        panels = []
        for m in masks:
            r = ndimage.rotate(m, ang, axes=(1, 2), reshape=True, order=1)
            proj = np.clip(r.sum(2) / scale, 0, 1) ** 0.6              # (z, rotated y)
            rgb = BG.astype(np.float32) * (1 - proj[..., None]) + PORE * proj[..., None]
            canvas = np.broadcast_to(BG, (m.shape[0] + 40, side, 3)).copy()
            y0, x0 = 20, (side - rgb.shape[1]) // 2
            canvas[y0:y0 + rgb.shape[0], x0:x0 + rgb.shape[1]] = rgb.astype(np.uint8)
            panels.append(canvas)
        frames.append(row(panels, height))
    return frames


def sweep3d():
    """Rotating pore projection for three requested porosities, one shared brightness scale."""
    labs = [gen("porosity_global", f"target{t}_ddim50_seed101")[1] == 1 for t in ("0.005", "0.03", "0.1")]
    write_video("sweep3d", spin(labs, 420), pingpong=False)


# === generated volumes ======================================================
def local_field():
    """Requested field (3x3x3 tiles) beside the delivered pore fraction per tile, and a rotating projection."""
    labs = []
    for case in ("halves_ddim50_seed101", "checkerboard_ddim50_seed101"):
        d = FINAL / "porosity_local/volumes" / case
        req = np.load(d / "requested_field.npy")
        lab = tifffile.imread(d / "label.tif")
        dl = (lab == 1).reshape(3, 64, 3, 64, 3, 64).mean((1, 3, 5))
        np.save(OUT / f"{case.split('_')[0]}_tiles.npy", np.stack([req, dl]))
        labs.append(lab == 1)
    write_video("local_field", spin(labs, 480), pingpong=False)


def geometry():
    """Notch-and-hole request and the sphere request: slices through z."""
    v1, l1 = gen("geometry", "notch_hole_seed101")
    v2, l2 = gen("geometry", "sphere_256_ddim50_seed101")
    req = np.load(FINAL / "geometry/volumes/notch_hole_seed101/requested_material.npy")
    Image.fromarray((req[req.shape[0] // 2] * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST).save(OUT / "notch_request.png")
    g1, g2 = stretch(v1), stretch(v2)
    frames = []
    for i in range(192):
        z2 = round(i * 255 / 191)
        frames.append(row([overlay(g1[i], l1[i]), overlay(g2[z2], l2[z2])], 512))
    write_video("geometry", frames)


def layup():
    """Layups A and C at 1024² (shown at 512²), with a glyph of the requested ply angle."""
    import json
    nums = json.loads((OUT / "numbers.json").read_text())
    angA, angC = nums["layup_A_requested_deg"]["value"], nums["layup_C_requested_deg"]["value"]
    vA, lA = gen("layup", "A_seed101")
    vC, lC = gen("layup", "C_seed101")
    gA, gC = stretch(vA[:, ::2, ::2]), stretch(vC[:, ::2, ::2])
    lA, lC = lA[:, ::2, ::2], lC[:, ::2, ::2]
    t = 19.6                                                           # requested ply thickness, voxels (manifest)
    frames = []
    for z in range(192):
        panels = []
        for g, l, ang in ((gA, lA, angA), (gC, lC, angC)):
            im = Image.fromarray(overlay(g[z], l[z]))
            dr = ImageDraw.Draw(im)
            a = np.deg2rad(ang[min(int(z // t), len(ang) - 1)])
            cx, cy, r = 470, 42, 26                                    # requested-angle glyph, top-right corner
            dr.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6], fill=(11, 13, 16))
            dr.line([cx - r * np.cos(a), cy + r * np.sin(a), cx + r * np.cos(a), cy - r * np.sin(a)], fill=(255, 176, 0), width=5)
            panels.append(np.asarray(im))
        frames.append(row(panels, 512))
    write_video("layup", frames)


def real_vs_gen():
    """A real 128³ crop at about 3 % porosity beside a generated volume at 0.03."""
    d = FINAL / "real_floor/volumes/micro_phi0.03__Na_05__a"
    vr, lr = tifffile.imread(d / "volume.tif"), tifffile.imread(d / "label.tif")
    vg, lg = gen("porosity_global", "target0.03_ddim50_seed101")
    vg, lg = vg[32:160, 32:160, 32:160], lg[32:160, 32:160, 32:160]
    gr, gg = stretch(vr), stretch(vg)
    frames = [row([overlay(gr[z], None), overlay(gg[z], None), overlay(gr[z], lr[z]), overlay(gg[z], lg[z])], 384) for z in range(128)]
    write_video("real_vs_gen", frames)


def large_volume():
    """One xy slice of a 1024×1024×192 generated volume (mid-depth), grey only."""
    v, _ = gen("sampler", "1024_ddim50_seed101")
    g = stretch(v[96])
    Image.fromarray(overlay(g, None)).save(OUT / "large_grey.jpg", quality=86)


def seam_maps():
    """Pore fraction per y and per x position, production chunking against the overlapped scheme."""
    for name, root in (("seam_production", FINAL / "sampler"), ("seam_overlapped", A2S / "sampler")):
        lab = tifffile.imread(root / "volumes/1024_ddim50_seed101/label.tif")
        prof_y, prof_x = (lab == 1).mean((0, 2)), (lab == 1).mean((0, 1))
        np.save(OUT / f"{name}_profiles.npy", np.stack([prof_y, prof_x]))
        print(name, "mean phi", float((lab == 1).mean()))


JOBS = {f.__name__: f for f in (title, data_slice, data_xsec, pores3d, sweep3d, local_field,
                                geometry, layup, real_vs_gen, large_volume, seam_maps)}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()

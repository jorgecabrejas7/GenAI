"""Real coupon against its generated twin: air fix, porosity, comparison videos, seam images.

Input (read only): runs/campaigns/36-pore-morphology/compare/<coupon>/ — real_volume/label.tif and
twin_<sampler>_volume/label.tif, one shape per folder (Z, Y, X), 25 µm voxels, labels 0 material,
1 pore, 2 air. Every volume is read lazily through tifffile.memmap; the host also runs a GPU job.

Jobs:
  airfix    every voxel the twin label calls air gets ONE grey value, round(mean grey over air),
            per twin volume -> build/twins/<coupon>/twin_<sampler>_volume_airfix.tif (labels
            unchanged; real scans already have uniform air and are never touched)
  porosity  pore / (pore + material) of each real coupon and twin, from the label files
  videos    per coupon: real | twin (overlapped chunks), same slice, grey 0-255 linear;
            whole coupon (XY, YZ, XZ), then the label overlay, then a full-resolution 512² window
  seam      real | twin (non-overlapped) | twin (overlapped), pores summed through the thickness,
            on a region that crosses chunk joins (every 192 voxels in y and x)
All numbers go to build/twins/twins.json, which extract_numbers.py reads with this file as source.

Usage: python talks/poregen-group/scripts/render_twins.py [airfix porosity videos seam]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from render_media import AIR, BG, OUT, PORE, write_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "runs/campaigns/36-pore-morphology/compare"
BUILD = Path(__file__).resolve().parents[1] / "build" / "twins"
STATS = BUILD / "twins.json"
COUPONS = ("Na_05_2", "Na_09_1", "8", "Na_09_5", "Na_02_4")    # ascending porosity
SAMPLERS = ("a2s", "production")
CHUNK = 192                                                     # 3 tiles of 64 voxels, from the canvas origin
ZOOM = 512


def stats() -> dict:
    return json.loads(STATS.read_text()) if STATS.exists() else {}


def save_stats(d: dict):
    BUILD.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(d, indent=1))


def twin_volume(c: str, s: str) -> np.ndarray:
    return tifffile.memmap(BUILD / c / f"twin_{s}_volume_airfix.tif", mode="r")


# === air fix ================================================================
def airfix():
    d = stats()
    for c in COUPONS:
        for s in SAMPLERS:
            src, lab = SRC / c / f"twin_{s}_volume.tif", SRC / c / f"twin_{s}_label.tif"
            dst = BUILD / c / f"twin_{s}_volume_airfix.tif"
            if not src.exists() or dst.exists():
                continue
            v, l = tifffile.memmap(src, mode="r"), tifffile.memmap(lab, mode="r")
            tot = n = 0
            for z in range(v.shape[0]):
                a = l[z] == 2
                tot += int(v[z][a].sum(dtype=np.int64)); n += int(a.sum())
            val = int(round(tot / n))
            dst.parent.mkdir(parents=True, exist_ok=True)
            out = tifffile.memmap(dst, shape=v.shape, dtype=np.uint8, imagej=True)
            for z in range(v.shape[0]):
                page = np.array(v[z]); page[l[z] == 2] = val; out[z] = page
            out.flush(); del out
            d.setdefault(c, {})[f"air_{s}"] = {"value": val, "mean_exact": tot / n, "n_air_voxels": n}
            save_stats(d)
            print(c, s, "air ->", val, flush=True)


# === porosity ===============================================================
def porosity():
    d = stats()
    for c in COUPONS:
        for name in ["real"] + [f"twin_{s}" for s in SAMPLERS]:
            l = tifffile.memmap(SRC / c / f"{name}_label.tif", mode="r")
            pore = mat = 0
            per_z = []
            for z in range(l.shape[0]):
                cnt = np.bincount(l[z].ravel(), minlength=3)
                pore += int(cnt[1]); mat += int(cnt[0]); per_z.append(int(cnt[1]))
            d.setdefault(c, {})[f"phi_{name}"] = pore / (pore + mat)
            d[c][f"pores_per_z_{name}"] = per_z
            d[c]["shape"] = list(l.shape)
            print(c, name, f"phi {100 * pore / (pore + mat):.3f} %", flush=True)
        save_stats(d)


# === comparison videos ======================================================
# Fixed frame boxes (pixels), so the slide can put its labels in HTML at fixed places.
FW, FH, COL = 1680, 512, 864          # frame size; the twin column starts at x = COL
XY_BOX, XZ_X, YZ_Y = (744, 372), 756, 384
THIN = 60                             # depth strips are at most this thick
FPS = 24


def grey(page: np.ndarray, size=None) -> np.ndarray:
    """Grey 0-255 linear (the same scale for real and generated), optional BOX downsample."""
    im = Image.fromarray(np.ascontiguousarray(page))
    if size is not None:
        im = im.resize(size, Image.BOX)
    return np.asarray(im).astype(np.float32)


def labels_ds(lab: np.ndarray, size) -> tuple[np.ndarray, np.ndarray]:
    """Pore and air masks at display size. A pixel is pore if ANY voxel under it is pore
    (so small pores stay visible when downsampled, the same way in both columns),
    air if most voxels under it are air."""
    def frac(m):
        return np.asarray(Image.fromarray(m.astype(np.float32)).resize(size, Image.BOX))
    return frac(lab == 1) > 0, frac(lab == 2) > 0.5


def paint(g: np.ndarray, pore=None, air=None, alpha=0.9) -> np.ndarray:
    x = 20 + g / 255 * 215
    rgb = np.stack([x, x, x + 4], -1)
    for m, col in ((air, AIR), (pore, PORE)):
        if m is not None:
            rgb[m] = rgb[m] * (1 - alpha) + col * alpha
    return rgb.clip(0, 255).astype(np.uint8)


def put(frame, img, x, y):
    h, w = img.shape[:2]
    frame[y:y + h, x:x + w] = img


def zoom_window(c: str, n_z) -> tuple[int, int]:
    """The 512² window (y0, x0) with the most REAL pores, summed through the thickness,
    among windows on a 32-voxel grid that hold no air at mid-depth (no holes, no edge)."""
    l = tifffile.memmap(SRC / c / "real_label.tif", mode="r")
    proj = np.zeros(l.shape[1:], np.int32)
    for z in n_z:
        proj += l[z] == 1
    air = (l[(n_z[0] + n_z[-1]) // 2] == 2).astype(np.int32)
    def win_sum(a):
        s = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return s[ZOOM:, ZOOM:] - s[:-ZOOM, ZOOM:] - s[ZOOM:, :-ZOOM] + s[:-ZOOM, :-ZOOM]
    ps, ai = win_sum(proj)[::32, ::32], win_sum(air)[::32, ::32]
    ps[ai > 0] = -1
    iy, ix = np.unravel_index(int(ps.argmax()), ps.shape)
    return int(iy * 32), int(ix * 32)


def videos():
    d = stats()
    for c in COUPONS:
        names = {"real": (tifffile.memmap(SRC / c / "real_volume.tif", mode="r"),
                          tifffile.memmap(SRC / c / "real_label.tif", mode="r")),
                 "twin": (twin_volume(c, "a2s"), tifffile.memmap(SRC / c / "twin_a2s_label.tif", mode="r"))}
        Z, Y, X = names["real"][0].shape
        # the sweep covers the depths where the real coupon has material, plus 4 voxels
        mat = [int((names["real"][1][z] == 0).any()) for z in range(Z)]
        zs = [z for z in range(Z) if mat[z]]
        zs = list(range(max(zs[0] - 4, 0), min(zs[-1] + 5, Z)))
        s = min(XY_BOX[0] / Y, XY_BOX[1] / X)
        wy, wx = round(Y * s), round(X * s)
        y0, x0 = zoom_window(c, zs)
        cy, cx = y0 + ZOOM // 2, x0 + ZOOM // 2             # the depth cuts go through the window centre
        loc_s = 272 / Y
        ly, lx = round(Y * loc_s), round(X * loc_s)
        # static depth cuts, read once: YZ at x = cx (Z, Y), XZ at y = cy (Z, X)
        cuts = {}
        for k, (vol, lab) in names.items():
            vz = np.stack([np.asarray(vol[z][:, cx]) for z in range(Z)])
            lz = np.stack([np.asarray(lab[z][:, cx]) for z in range(Z)])
            vx = np.stack([np.asarray(vol[z][cy, :]) for z in range(Z)])
            lx_ = np.stack([np.asarray(lab[z][cy, :]) for z in range(Z)])
            hz = round(Z * s)
            cuts[k] = {"yz": (grey(vz, (wy, hz)), *labels_ds(lz, (wy, hz))),
                       "xz": (grey(vx.T, (hz, wx)), *labels_ds(lx_.T, (hz, wx)))}
        phases = {"whole": [], "labels": [], "zoom": []}
        for z in zs:
            fr = {p: np.broadcast_to(BG, (FH, FW, 3)).copy() for p in phases}
            for col, (k, (vol, lab)) in zip((0, COL), names.items()):
                page, lp = np.asarray(vol[z]), np.asarray(lab[z])
                g = grey(page.T, (wy, wx))                      # Y horizontal, X vertical
                pm, am = labels_ds(lp.T, (wy, wx))
                hz = cuts[k]["yz"][0].shape[0]
                zr = round((z + 0.5) * s)
                for p, lab_on in (("whole", False), ("labels", True)):
                    f = fr[p]
                    put(f, paint(g, pm if lab_on else None, am if lab_on else None), col, 0)
                    yz, xz = cuts[k]["yz"], cuts[k]["xz"]
                    put(f, paint(yz[0], *(yz[1:] if lab_on else (None, None))), col, YZ_Y)
                    put(f, paint(xz[0], *(xz[1:] if lab_on else (None, None))), col + XZ_X, 0)
                    # where the depth cuts lie (ticks just outside the slice), and the current depth
                    ty, tx = round(cy * s), round(cx * s)
                    f[wx:wx + 6, col + ty - 1:col + ty + 1] = 200
                    f[tx - 1:tx + 1, col + wy:col + wy + 6] = 200
                    f[YZ_Y + min(zr, hz - 1), col:col + wy] = (f[YZ_Y + min(zr, hz - 1), col:col + wy] * 0.4 + 140).astype(np.uint8)
                    f[0:wx, col + XZ_X + min(zr, hz - 1)] = (f[0:wx, col + XZ_X + min(zr, hz - 1)] * 0.4 + 140).astype(np.uint8)
                    # the zoom window, outlined
                    r0, r1, q0, q1 = round(y0 * s), round((y0 + ZOOM) * s), round(x0 * s), round((x0 + ZOOM) * s)
                    for a, b in ((slice(q0, q1), r0), (slice(q0, q1), r1 - 1)):
                        f[a, col + b] = 232
                    f[q0, col + r0:col + r1] = 232; f[q1 - 1, col + r0:col + r1] = 232
                f = fr["zoom"]
                win = page[y0:y0 + ZOOM, x0:x0 + ZOOM].T           # full resolution, y horizontal, x vertical
                put(f, paint(win.astype(np.float32)), col, 0)
                loc = paint(grey(page.T, (ly, lx)))
                put(f, loc, col + 544, 0)
                a0, a1, b0, b1 = round(y0 * loc_s), round((y0 + ZOOM) * loc_s), round(x0 * loc_s), round((x0 + ZOOM) * loc_s)
                f[b0:b1, col + 544 + a0] = 232; f[b0:b1, col + 544 + a1 - 1] = 232
                f[b0, col + 544 + a0:col + 544 + a1] = 232; f[b1 - 1, col + 544 + a0:col + 544 + a1] = 232
            for p in phases:
                phases[p].append(fr[p])
        # dwell: slow down where pores are (pores of real + twin at that depth)
        pz = np.array(d[c]["pores_per_z_real"]) + np.array(d[c]["pores_per_z_twin_a2s"])
        dwell = [2 + int(round(4 * pz[z] / max(pz.max(), 1))) for z in zs]
        frames, marks = [], {}
        for p in phases:
            marks[p] = len(frames) / FPS
            for f, n in zip(phases[p], dwell):
                frames += [f] * n
        name = f"twin_{c}"
        write_video(name, frames, pingpong=False)
        d[c]["video"] = {"name": name, "phase_start_s": marks, "length_s": len(frames) / FPS,
                         "zoom_y0_x0": [y0, x0], "cut_y": cy, "cut_x": cx, "scale": s,
                         "whole_px": [wy, wx], "depth_px": round(Z * s), "z_range": [zs[0], zs[-1]],
                         "loc_px": [ly, lx]}
        save_stats(d)
        del phases, frames


# === seam profiles ==========================================================
SEAM_COUPON = "Na_09_5"


def seam():
    """Pore fraction (pore / non-air) per y and per x position of one coupon, for the real scan
    and both twins. The non-overlapped twin has chunk joins every 192 voxels from the volume
    origin in y and x; the band is a few voxels wide, too thin to see in a picture, so the slide
    plots the profile. Also folds each profile onto the distance from the nearest join."""
    d = stats()
    c, res = SEAM_COUPON, {}
    for k, lf in (("real", "real_label.tif"), ("production", "twin_production_label.tif"),
                  ("a2s", "twin_a2s_label.tif")):
        l = tifffile.memmap(SRC / c / lf, mode="r")
        py, my = np.zeros(l.shape[1]), np.zeros(l.shape[1])
        px, mx = np.zeros(l.shape[2]), np.zeros(l.shape[2])
        for z in range(l.shape[0]):
            p, m = l[z] == 1, l[z] != 2
            py += p.sum(1); my += m.sum(1); px += p.sum(0); mx += m.sum(0)
        # 4-voxel moving average: the twins carry a pore pattern with a period of exactly 4 voxels
        # (one latent cell, f = 4) that would otherwise hide the band; a 4-wide box removes it exactly
        box = np.ones(4) / 4
        fy = np.convolve(py / np.maximum(my, 1), box, "same")
        fx = np.convolve(px / np.maximum(mx, 1), box, "same")
        dist = np.arange(-24, 25)
        fold = np.mean([f[j + dist] / f[j - 96:j + 96].mean()
                        for f in (fy, fx) for j in range(192, len(f) - 96, 192)], 0)
        res[k] = {"profile_y": fy.round(5).tolist(), "fold_dist": dist.tolist(), "fold": fold.round(4).tolist(),
                  "fold_min": float(fold.min()), "fold_at_join_pm2": float(fold[22:27].mean())}
        print(c, k, "at join (±2 voxels)", round(res[k]["fold_at_join_pm2"], 3), flush=True)
    d["seam"] = {"coupon": c, **res,
                 "how": "pore / non-air voxels per position, 4-voxel moving average; fold = mean over every interior 192-voxel "
                        "join in y and x of the profile divided by its mean over ±96 voxels"}
    save_stats(d)


JOBS = {"airfix": airfix, "porosity": porosity, "videos": videos, "seam": seam}

if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()

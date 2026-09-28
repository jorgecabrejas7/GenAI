#!/usr/bin/env python
"""The reverse process, as a film. (GPU.)

Two videos, for presentations:

A. ``--mode single`` — ONE 192-cubed chunk, 3x3x3 jointly-denoised windows, no
   neighbours. One frame per DDIM step: the model's x0 PREDICTION for the whole
   chunk, decoded, shown as three orthogonal mid-slices.

B. ``--mode multi`` — a 384-cubed volume as 2x2x2 chunks through the production
   hybrid sampler. Frames every ``--every`` steps and at each chunk's end; only
   the chunk that changed is decoded and pasted into a decoded canvas, and the
   chunk being worked on is outlined. Unfinished chunks show their current
   noise, because that is what the sampler is actually holding.

WHY x0 AND NOT x_t. The noisy latent is noise until very late and a video of it
says nothing. x0 is what the model believes the finished volume is at that step,
so the film shows a belief sharpening rather than a fog thinning. The sampler's
``on_step`` hook hands it over without perturbing the chain — see
``tests/test_volume_generator_hybrid.py::TestOnStepHook``.

Usage
-----
    python scripts/analysis/denoising_video.py --mode single \\
        --run runs/ldm/ldm06-run-0002-... --ckpt latest \\
        --vae-run runs/vae/r08-run-0004-... \\
        --out runs/campaigns/18-eval-v4-final/videos
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
FPS = 12
DPI = 150


def _frame(fig, axes, txt, slices, title):
    for ax, sl in zip(axes, slices):
        ax.images[0].set_data(sl)
    txt.set_text(title)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def _as_u8(dec) -> np.ndarray:
    """The decoded grey on the RAW-SCAN SCALE, 0-255 uint8.

    _decode_canvas returns [0, 1] — the grey level itself, since the XCT head is
    regressed against xct/255. Every figure in this project shows that as uint8
    on a FIXED 0-255 scale, so these frames do too. Scaling each frame to its
    own range would make the film flicker and would leave its last frame on a
    different scale from the paper's side-by-sides; early frames are supposed to
    look grey and flat, because early in the reverse process they are.
    """
    arr = dec.detach().float().cpu().numpy() if torch.is_tensor(dec) else np.asarray(dec)
    return (np.clip(np.squeeze(arr), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _mid_slices(vol: np.ndarray) -> list[np.ndarray]:
    """z in-plane, then y and x through the thickness."""
    d, h, w = vol.shape
    return [vol[d // 2], vol[:, h // 2], vol[:, :, w // 2]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("single", "multi"), default="single")
    ap.add_argument("--run", required=True, help="LDM run directory")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--phi", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--every", type=int, default=5, help="multi: frame stride")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator
    from poregen.eval_v4.cases import layup_a
    from poregen.eval_v4.generate import (
        LATENT_SIZE, PATCH_SIZE, VOXEL_SIZE_MM, VolumeRunner, theta_for_canvas,
    )

    out = a.out if a.out.is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = VolumeRunner(a.run, a.ckpt, weights=a.weights, device=dev)
    plies, pitch = layup_a(REPO)

    tiles = 3 if a.mode == "single" else 6          # 192 or 384 voxels per axis
    chunk_tiles = (3, 3, 3)                          # one chunk in single mode
    shape = (tiles * PATCH_SIZE,) * 3
    size_mm = tuple(s * VOXEL_SIZE_MM for s in shape)
    theta = theta_for_canvas(shape[0], plies, pitch, 0)

    sampler = DDIMSampler(runner.model, runner.schedule, dev, n_steps=a.steps,
                          s_por=1.0, s_nb=1.0, cfg_rescale=runner.cfg_rescale)
    gen = VolumeGenerator(
        sampler=sampler, vae=runner.vae, device=dev,
        patch_size=PATCH_SIZE, latent_size=LATENT_SIZE,
        latent_mean=runner.latent_mean, latent_std=runner.latent_std,
        voxel_size_mm=VOXEL_SIZE_MM, por_log_stats=runner.por_log_stats,
        theta_deg=theta, chunk_tiles=chunk_tiles,
        # No neighbours in single mode: the chunk is the whole volume and the
        # point is the chunk's own reverse process, not the assembly.
        neighbour_mode="unknown" if a.mode == "single" else "canvas",
    )

    keep: list[dict] = []
    every = 1 if a.mode == "single" else a.every

    def on_step(rec):
        last = rec["step"] == rec["n_steps"] - 1
        if rec["step"] % every == 0 or last:
            keep.append({"chunk": rec["chunk_index"], "step": rec["step"],
                         "t": rec["t"], "x0": rec["x0"],
                         "slice": rec["chunk_slice"], "n_chunks": rec["n_chunks"],
                         "n_steps": rec["n_steps"]})

    t0 = time.time()
    print(f"generating {shape} in {a.mode} mode, {a.steps} DDIM steps", flush=True)
    gen.generate(size_mm, target_porosity=a.phi, seed=a.seed, on_step=on_step)
    print(f"  sampling done in {time.time() - t0:.0f}s, {len(keep)} frames kept",
          flush=True)

    # ── decode the kept x0 predictions ──────────────────────────────────────
    # Decoding is the expensive half, so it happens once per KEPT frame and not
    # once per step.
    vpc = PATCH_SIZE // LATENT_SIZE
    vols: list[np.ndarray] = []
    canvas = None
    for i, rec in enumerate(keep):
        z = rec["x0"][0]                                   # (C, *chunk_cells)
        if a.mode == "single":
            dec, _ = gen._decode_canvas(z.to(dev), shape, runner.autocast_dtype, 64)
            vols.append(_as_u8(dec))
        else:
            # ONLY THE CHUNK THAT CHANGED is decoded and pasted. Decoding the
            # whole 384-cubed canvas every frame would be eight times the work
            # for seven eighths of a picture that did not move.
            if canvas is None:
                # A chunk that has not started has no latents to show, so its
                # region is mid-grey until its first step. Stated in the README.
                canvas = np.full(shape, 128, dtype=np.uint8)
            sl = rec["slice"]
            vox = tuple(slice(s.start * vpc, s.stop * vpc) for s in sl)
            chunk_shape = tuple(v.stop - v.start for v in vox)
            dec, _ = gen._decode_canvas(z.to(dev), chunk_shape,
                                        runner.autocast_dtype, 64)
            canvas[vox] = _as_u8(dec)
            vols.append(canvas.copy())
        if (i + 1) % 10 == 0:
            print(f"  decoded {i + 1}/{len(keep)}", flush=True)

    # ── draw ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=DPI)
    names = ("z mid-slice (in-plane)", "y mid-slice (through thickness)",
             "x mid-slice (through thickness)")
    for ax, nm, sl in zip(axes, names, _mid_slices(vols[0])):
        ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
        ax.set_title(nm, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    # IN THE CORNER, not across the top: a suptitle sat on the middle panel's
    # own title and the two overprinted each other.
    txt = fig.text(0.012, 0.975, "", fontsize=12, ha="left", va="top",
                   family="monospace")

    # THE CHUNK BEING WORKED ON, outlined on whichever panels its mid-slice
    # actually cuts through. A panel whose mid-slice misses the chunk gets no
    # box, which is itself information: that chunk is not in this view.
    from matplotlib.patches import Rectangle
    boxes = [Rectangle((0, 0), 0, 0, fill=False, lw=1.6, ec="#ff3b30")
             for _ in axes]
    for ax, b in zip(axes, boxes):
        ax.add_patch(b)

    def place(rec, shape):
        vpc = PATCH_SIZE // LATENT_SIZE
        z0, z1 = rec["slice"][0].start * vpc, rec["slice"][0].stop * vpc
        y0, y1 = rec["slice"][1].start * vpc, rec["slice"][1].stop * vpc
        x0, x1 = rec["slice"][2].start * vpc, rec["slice"][2].stop * vpc
        mid = [s // 2 for s in shape]
        spec = [(z0 <= mid[0] < z1, (x0, y0), x1 - x0, y1 - y0),
                (y0 <= mid[1] < y1, (x0, z0), x1 - x0, z1 - z0),
                (x0 <= mid[2] < x1, (y0, z0), y1 - y0, z1 - z0)]
        for b, (show, xy, w, h) in zip(boxes, spec):
            b.set_visible(bool(show))
            b.set_xy(xy); b.set_width(w); b.set_height(h)

    frames = []
    for rec, vol in zip(keep, vols):
        if a.mode == "multi":
            place(rec, shape)
        else:
            for b in boxes:
                b.set_visible(False)
        if a.mode == "single":
            title = (f"DDIM step {rec['step'] + 1}/{rec['n_steps']}   t = {rec['t']}"
                     f"   —  x0 prediction, {shape[0]}³ single chunk")
        else:
            title = (f"chunk {rec['chunk'] + 1}/{rec['n_chunks']}   "
                     f"step {rec['step'] + 1}/{rec['n_steps']}   t = {rec['t']}"
                     f"   —  {shape[0]}³, 2×2×2 chunks")
        frames.append(_frame(fig, axes, txt, _mid_slices(vol), title))

    stem = f"denoising_{a.mode}_{shape[0]}_ddim{a.steps}_phi{a.phi}_seed{a.seed}"
    mp4 = out / f"{stem}.mp4"
    with imageio.get_writer(str(mp4), fps=FPS, quality=8, format="FFMPEG") as w:
        for f in frames:
            w.append_data(f)
    png = out / f"{stem}_final.png"
    imageio.imwrite(str(png), frames[-1])
    plt.close(fig)
    print(f"wrote {mp4}  ({len(frames)} frames)")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

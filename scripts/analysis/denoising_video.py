#!/usr/bin/env python
"""The whole pipeline as a film, in three stages. (GPU.)

Every film shows the same three stages, each announced by a title card:

  STAGE 1  latent denoising            the model's current clean estimate,
                                       decoded, one frame per DDIM step
  STAGE 2  decoding with 32-voxel      the PRODUCTION overlapped decode,
           overlapped tiles            animated tile by tile as it blends
  STAGE 3  assembled volume            a fly-through of the finished volume

THE CLEAN ESTIMATE, NOT THE NOISY LATENT. ldm06 is trained on the v objective,
so the model emits v and the clean estimate is recovered as

    x0 = sqrt(alpha_bar) * x_t  -  sqrt(1 - alpha_bar) * v

which is what DDPMSchedule.predict_x0 computes and what the sampler's on_step
hook hands over. x_t itself is noise the eye cannot read until very late; the
clean estimate is what the model believes the finished volume is at that step,
so the film shows a belief sharpening rather than a fog thinning.

THE GREY SCALE IS FIXED AT 0-255 EVERYWHERE — the raw-scan scale, exactly as
decode_xct_u8 produces it and as every other figure in this project uses.
Nothing is normalised per frame or per film. Early frames look flat and grey,
which is correct: early in the reverse process they are.

SEAMS. Chunk-end frames, the final frame and the whole fly-through use the
production overlapped decode of the full canvas. The intermediate within-chunk
frames use a cheap per-chunk decode and say so on screen: that decode leaves
visible chunk borders and the VOLUME DOES NOT HAVE THEM — campaign 18 measures
the chunk-plane grey seam at 0.894-0.896 against real material's 0.906 at 1024
scale, and a direct comparison of the two decodes on one finished 384-cubed
canvas differs by +0.04 grey levels in the mean.

Usage
-----
    python scripts/analysis/denoising_video.py --film plain_single --run ... --out ...

Films: plain_single, plain_multi, shape_single, shape_multi.
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
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
BASE_FPS = 24
FPS_STAGE = {1: 15, 2: 12, 3: 24}
TITLE_SECONDS = 1.5
DPI = 150

#: film -> (mode, volume shape, ddim steps, frame stride, material request)
FILMS = {
    "plain_single": ("single", (192, 192, 192), 100, 1, None),
    "shape_single": ("single", (192, 192, 192), 100, 1, "notch_hole"),
    "plain_multi":  ("multi",  (384, 384, 384), 100, 2, None),
    "shape_multi":  ("multi",  (192, 1024, 1024), 100, 2, "letters"),
}
#: Stage 2 frame cap, and the fly-through strides. Every film is meant to be
#: about 40 seconds; a stage nobody watches to the end is a stage that should
#: have been shorter.
STAGE2_MAX = 60
FLY_Z_STEP, FLY_Y_STEP = 2, 8
#: Voxels of finished neighbour decoded together with a chunk, so the border
#: between them is blended in the picture exactly as it is in the volume.
HALO_VOX = 32


def _u8(arr) -> np.ndarray:
    """Decoded grey on the RAW-SCAN SCALE, 0-255, never per-frame normalised."""
    a = arr.detach().float().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
    a = np.squeeze(a)
    if a.dtype == np.uint8:
        return a
    return (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _slices(vol: np.ndarray, at: tuple[int, int, int]) -> list[np.ndarray]:
    """z in-plane, then y and x through the thickness."""
    d, h, w = vol.shape
    return [vol[min(at[0], d - 1)], vol[:, min(at[1], h - 1)], vol[:, :, min(at[2], w - 1)]]


def _repeat_plan(n: int, target_fps: int) -> list[int]:
    """How many times to emit each frame so *n* frames play at *target_fps*.

    The container runs at one rate, so a stage that wants a different one gets
    it by repetition. The average is exact even when the ratio is not an
    integer: 24/15 is 1.6, emitted as 2,2,1,2,1 and so on.
    """
    step = BASE_FPS / target_fps
    out, carry = [], 0.0
    for _ in range(n):
        carry += step
        k = int(round(carry))
        out.append(max(1, k))
        carry -= out[-1]
    return out


class Panel:
    """The three-panel figure every stage draws into."""

    def __init__(self, shape, at, down=1):
        self.shape, self.at, self.down = shape, at, down
        # THE THREE PANELS ARE NOT THE SAME SHAPE unless the volume is a cube.
        # At 192x1024x1024 the z slice is 1024x1024 and the two thickness
        # slices are 192x1024 — laid side by side at equal aspect they leave
        # the strips floating in white space and squeeze the in-plane view. So
        # a plate-shaped volume gets a STACKED layout, each panel at its own
        # natural aspect, and a cube keeps the row.
        hs = [s.shape[0] for s in _slices(np.zeros(shape, np.uint8), at)]
        ws = [s.shape[1] for s in _slices(np.zeros(shape, np.uint8), at)]
        self.stacked = max(hs) / min(hs) > 2.0
        if self.stacked:
            # The in-plane view on the left at its own square aspect, the two
            # thin thickness views stacked on the right. A single column would
            # be a 1650x2433 portrait film; this keeps it landscape.
            # Positions are set EXPLICITLY. tight_layout on a mixed gridspec
            # pushed the square in-plane axes out of its own cell and drew it
            # underneath the two strips.
            self.fig = plt.figure(figsize=(15.5, 8.0), dpi=DPI)
            self.axes = [self.fig.add_axes((0.025, 0.04, 0.44, 0.88)),
                         self.fig.add_axes((0.52, 0.56, 0.465, 0.30)),
                         self.fig.add_axes((0.52, 0.12, 0.465, 0.30))]
        else:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=DPI)
        names = ("z mid-slice (in-plane)", "y (through thickness)",
                 "x (through thickness)")
        blank = [np.zeros((1, 1), np.uint8)] * 3
        for ax, nm, b in zip(self.axes, names, blank):
            ax.imshow(b, cmap="gray", vmin=0, vmax=255)
            ax.set_title(nm, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        self.title = self.fig.text(0.012, 0.978, "", fontsize=12, ha="left",
                                   va="top", family="monospace")
        self.note = self.fig.text(0.988, 0.978, "", fontsize=8.5, ha="right",
                                  va="top", color="#b23")
        self.chunk = [Rectangle((0, 0), 0, 0, fill=False, lw=1.6, ec="#ff3b30")
                      for _ in self.axes]
        self.material = None      # (Z, Y, X) fraction, for the shape films
        self._mat_art: list = []
        self.marks = [ax.axhline(-5, color="#ffd400", lw=1.2, visible=False)
                      for ax in self.axes]
        for ax, c in zip(self.axes, self.chunk):
            ax.add_patch(c)
        if not self.stacked:
            self.fig.tight_layout(rect=(0, 0, 1, 0.93))

    def _material_outline(self):
        """The REQUESTED envelope, in blue, on whichever slice is showing.

        Drawn from the same voxel map that was handed to the sampler, so the
        film shows what was asked for against what came back rather than an
        artist's impression of it.
        """
        for art in self._mat_art:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._mat_art = []
        if self.material is None:
            return
        for ax, sl in zip(self.axes, _slices(self.material, self.at)):
            if self.down > 1:
                sl = sl[::self.down, ::self.down]
            if 0.0 < float(sl.min()) or float(sl.max()) <= 0.0:
                continue           # no boundary in this slice
            cs = ax.contour(sl, levels=[0.5], colors="#2f6fff", linewidths=1.2)
            self._mat_art.append(cs)

    def draw(self, vol, title, note="", marks=None):
        for ax, sl in zip(self.axes, _slices(vol, self.at)):
            if self.down > 1:
                sl = sl[::self.down, ::self.down]
            im = ax.images[0]
            im.set_data(sl)
            im.set_extent((-0.5, sl.shape[1] - 0.5, sl.shape[0] - 0.5, -0.5))
        self._material_outline()
        self.title.set_text(title)
        self.note.set_text(note)
        for i, m in enumerate(self.marks):
            if marks is not None and marks[i] is not None:
                m.set_ydata([marks[i] / self.down] * 2); m.set_visible(True)
            else:
                m.set_visible(False)
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[..., :3].copy()

    def card(self, big, small):
        for ax in self.axes:
            ax.set_visible(False)
        t = self.fig.text(0.5, 0.56, big, fontsize=26, ha="center", va="center")
        u = self.fig.text(0.5, 0.40, small, fontsize=13, ha="center", va="center",
                          color="#555")
        self.title.set_text(""); self.note.set_text("")
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[..., :3].copy()
        t.remove(); u.remove()
        for ax in self.axes:
            ax.set_visible(True)
        return frame

    def place_chunk(self, sl_vox):
        mid = self.at
        z0, z1 = sl_vox[0].start, sl_vox[0].stop
        y0, y1 = sl_vox[1].start, sl_vox[1].stop
        x0, x1 = sl_vox[2].start, sl_vox[2].stop
        spec = [(z0 <= mid[0] < z1, (x0, y0), x1 - x0, y1 - y0),
                (y0 <= mid[1] < y1, (x0, z0), x1 - x0, z1 - z0),
                (x0 <= mid[2] < x1, (y0, z0), y1 - y0, z1 - z0)]
        for b, (show, xy, w, h) in zip(self.chunk, spec):
            b.set_visible(bool(show))
            b.set_xy((xy[0] / self.down, xy[1] / self.down))
            b.set_width(w / self.down); b.set_height(h / self.down)

    def hide_chunk(self):
        for b in self.chunk:
            b.set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--film", choices=tuple(FILMS), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--phi", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--slice-index", type=int, default=None)
    ap.add_argument("--down", type=int, default=None,
                    help="display downsample; default 2 for the 1024-wide film")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator
    from poregen.eval_v4 import stress_geometry as SG
    from poregen.eval_v4.cases import layup_a, material_notch_and_hole
    from poregen.eval_v4.generate import (
        LATENT_SIZE, PATCH_SIZE, VOXEL_SIZE_MM, VolumeRunner,
        latent_material_map, theta_for_canvas,
    )

    mode, shape, steps, stride, material = FILMS[a.film]
    down = a.down if a.down is not None else (2 if max(shape) >= 1024 else 1)
    at = ((a.slice_index,) * 3 if a.slice_index is not None
          else (96, 96, 96) if mode == "multi" else tuple(s // 2 for s in shape))
    out = a.out if a.out.is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = VolumeRunner(a.run, a.ckpt, weights=a.weights, device=dev)
    plies, pitch = layup_a(REPO)
    theta = theta_for_canvas(shape[0], plies, pitch, 0)
    size_mm = tuple(s * VOXEL_SIZE_MM for s in shape)

    # The envelope is requested on the LATENT CELL grid; the voxel map is kept
    # only to draw the outline the film shows the result against.
    mat_vox = mat_cells = None
    if material == "notch_hole":
        mat_vox = np.asarray(material_notch_and_hole(shape), bool)
    elif material == "letters":
        mat_vox = np.asarray(SG.material_letters(shape), bool)
    if mat_vox is not None:
        mat_cells = latent_material_map(mat_vox)

    sampler = DDIMSampler(runner.model, runner.schedule, dev, n_steps=steps,
                          s_por=1.0, s_nb=1.0, cfg_rescale=runner.cfg_rescale)
    gen = VolumeGenerator(
        sampler=sampler, vae=runner.vae, device=dev,
        patch_size=PATCH_SIZE, latent_size=LATENT_SIZE,
        latent_mean=runner.latent_mean, latent_std=runner.latent_std,
        voxel_size_mm=VOXEL_SIZE_MM, por_log_stats=runner.por_log_stats,
        theta_deg=theta, chunk_tiles=(3, 3, 3),
        neighbour_mode="unknown" if mode == "single" else "canvas",
    )

    vpc = PATCH_SIZE // LATENT_SIZE
    keep: list[dict] = []

    def on_step(rec):
        if rec["step"] % stride == 0 or rec["step"] == rec["n_steps"] - 1:
            keep.append({k: rec[k] for k in
                         ("chunk_index", "step", "t", "x0", "chunk_slice",
                          "n_chunks", "n_steps")})

    t0 = time.time()
    print(f"[{a.film}] sampling {shape}, DDIM-{steps}", flush=True)
    xct_final, _label, _stats, z_final = gen.generate(
        size_mm, target_porosity=a.phi, seed=a.seed, on_step=on_step,
        return_latents=True,
        **({"material_map": mat_cells} if mat_cells is not None else {}))
    print(f"  sampled in {time.time() - t0:.0f}s, {len(keep)} step frames",
          flush=True)

    panel = Panel(shape, at, down)
    panel.material = mat_vox.astype(np.float32) if mat_vox is not None else None
    stage_frames: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    # ── STAGE 1 ─────────────────────────────────────────────────────────────
    canvas = np.full(shape, 128, np.uint8)
    # A RUNNING LATENT CANVAS, so a chunk can be decoded together with a halo of
    # its FINISHED neighbours. Decoding a chunk alone leaves a hard border where
    # it meets one, which is what the user saw and what the volume does not
    # have: the production decoder blends across that border with 32-voxel
    # overlapped tiles. Here the same decoder runs over chunk + halo, so the
    # border is blended in the picture exactly as it is in the volume. A border
    # with a not-yet-generated region stays hard, which is honest — there is
    # nothing there yet.
    halo_cells = HALO_VOX // vpc
    ccells = tuple(s // vpc for s in shape)
    zcanvas = torch.zeros((keep[0]["x0"].shape[1], *ccells)) if keep else None
    for i, rec in enumerate(keep):
        z = rec["x0"][0]
        cs = rec["chunk_slice"]
        zcanvas[(slice(None), *cs)] = z
        lo = [max(c.start - halo_cells, 0) for c in cs]
        hi = [min(c.stop + halo_cells, ccells[a]) for a, c in enumerate(cs)]
        reg = tuple(slice(lo[a], hi[a]) for a in range(3))
        sl = tuple(slice(reg[a].start * vpc, reg[a].stop * vpc) for a in range(3))
        cshape = tuple(s.stop - s.start for s in sl)
        dec, _ = gen._decode_canvas(zcanvas[(slice(None), *reg)].to(dev), cshape,
                                    runner.autocast_dtype, 64)
        canvas[sl] = _u8(dec)
        if mode == "multi":
            panel.place_chunk(sl)
            title = (f"chunk {rec['chunk_index'] + 1}/{rec['n_chunks']}  "
                     f"step {rec['step'] + 1}/{rec['n_steps']}  t={rec['t']}")
        else:
            panel.hide_chunk()
            title = f"step {rec['step'] + 1}/{rec['n_steps']}  t={rec['t']}"
        stage_frames[1].append(panel.draw(
            canvas, title + "   model's current clean estimate (from v)",
            note=""))
        if (i + 1) % 25 == 0:
            print(f"  stage 1 {i + 1}/{len(keep)}", flush=True)

    # ── STAGE 2 ─────────────────────────────────────────────────────────────
    # The REAL overlapped decode, filmed through the sampler's own hook.
    panel.hide_chunk()

    # WHAT HAS BEEN DECODED SO FAR. The accumulator is ZERO where no tile has
    # landed yet, and zero drawn at vmin 0 is black — so the first frames were a
    # black rectangle rather than a canvas filling in. Tracked here from the
    # tile slices rather than asked of the sampler: the hook already says which
    # voxels each tile wrote.
    covered = np.zeros(shape, bool)
    disp = np.full(shape, 128, np.uint8)
    stage2_means: list[float] = []

    def on_tile(rec):
        every = max(1, rec["n_tiles"] // STAGE2_MAX)
        covered[rec["slice"]] = True
        if rec["tile"] % every == 0 or rec["tile"] == rec["n_tiles"] - 1:
            # THROUGH _u8. The hook hands over the accumulator in [0, 1] and the
            # panels are drawn at vmin 0, vmax 255 — passing the float straight
            # in made every stage-2 frame solid black, and I shipped four films
            # that way because I only ever looked at their LAST frame, which is
            # stage 3.
            np.copyto(disp, _u8(rec["partial"]), where=covered)
            stage_frames[2].append(panel.draw(
                disp,
                f"overlapped decode  tile {rec['tile'] + 1}/{rec['n_tiles']}"
                f"   32-voxel stride, Tukey blend"))
            stage2_means.append(float(disp[covered].mean()))

    print("  stage 2: re-decoding the finished canvas through the hook",
          flush=True)
    gen._decode_canvas(torch.from_numpy(z_final).to(dev), shape,
                       runner.autocast_dtype, 64, on_tile=on_tile)
    print(f"  stage 2 frames: {len(stage_frames[2])}", flush=True)
    # FAIL LOUDLY RATHER THAN WRITE A BLACK FILM.
    # ON THE DECODED VOXELS, not on the rendered frame. A frame whose panels are
    # solid black still averages about 80 because of the white surround, which
    # is exactly the number the rejected films scored — an assertion on the
    # frame mean would have passed them.
    for i, m in enumerate(stage2_means):
        if m < 30.0:
            raise RuntimeError(
                f"stage-2 frame {i}: mean grey {m:.1f} over the DECODED voxels. "
                "The decode partial is not on the 0-255 scale the panels are "
                "drawn at; refusing to write a black film.")

    # ── STAGE 3 ─────────────────────────────────────────────────────────────
    final = _u8(xct_final)
    for zi in range(0, shape[0], FLY_Z_STEP):
        panel.at = (zi, at[1], at[2])
        stage_frames[3].append(panel.draw(final, f"fly-through  z = {zi}/{shape[0]}",
                                          marks=[None, zi, zi]))
    panel.at = at
    for yi in range(0, shape[1], FLY_Y_STEP):
        panel.at = (at[0], yi, at[2])
        stage_frames[3].append(panel.draw(final, f"fly-through  y = {yi}/{shape[1]}",
                                          marks=[yi, None, None]))
    panel.at = at

    # ── assemble ────────────────────────────────────────────────────────────
    names = {1: ("STAGE 1", "latent denoising"),
             2: ("STAGE 2", "decoding with 32-voxel overlapped tiles"),
             3: ("STAGE 3", "assembled volume")}
    out_frames: list[np.ndarray] = []
    for st in (1, 2, 3):
        fr = stage_frames[st]
        if not fr:
            continue
        card = panel.card(*names[st])
        out_frames.extend([card] * int(TITLE_SECONDS * BASE_FPS))
        for f, k in zip(fr, _repeat_plan(len(fr), FPS_STAGE[st])):
            out_frames.extend([f] * k)

    stem = f"{a.film}_{shape[0]}x{shape[1]}x{shape[2]}_ddim{steps}_seed{a.seed}"
    mp4 = out / f"{stem}.mp4"
    with imageio.get_writer(str(mp4), fps=BASE_FPS, quality=8, format="FFMPEG") as w:
        for f in out_frames:
            w.append_data(f)
    imageio.imwrite(str(out / f"{stem}_final.png"), stage_frames[3][-1])

    # A CONTACT SHEET OF EIGHT FRAMES SPREAD OVER EVERY STAGE. No film counts as
    # finished until this has been looked at: four films went out with a solid
    # black stage 2 because the only frame I ever checked was the last one.
    picks = []
    for st in (1, 2, 3):
        fr = stage_frames[st]
        if not fr:
            continue
        n = 3 if st != 2 else 2
        picks += [(st, j, fr[j]) for j in
                  np.linspace(0, len(fr) - 1, n, dtype=int)]
    sheet, sax = plt.subplots(2, 4, figsize=(18, 7.5), dpi=110)
    for ax, (st, j, fr) in zip(sax.ravel(), picks):
        ax.imshow(fr)
        ax.set_title(f"stage {st}, frame {j}   mean {fr.mean():.0f}", fontsize=9)
    for ax in sax.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    sheet.suptitle(f"{stem} — 8 frames across all stages", fontsize=12)
    sheet.tight_layout()
    contact = out / f"{stem}_contact.png"
    sheet.savefig(contact)
    plt.close(sheet)
    print(f"wrote {contact}")
    plt.close(panel.fig)
    print(f"wrote {mp4}  ({len(out_frames)} frames at {BASE_FPS} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

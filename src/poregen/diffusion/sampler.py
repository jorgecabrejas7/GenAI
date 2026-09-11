"""DDIM patch sampler and hybrid chunked volume generator for the PoreGen LDM.

There is exactly ONE generation path (ldm06): **hybrid chunked joint
denoising**.  It is the union of the two ldm05 modes, and it exists because
each of them was only half right.

* *Joint* (MultiDiffusion) denoising kept overlapping windows on one latent
  canvas coherent, but the canvas had to fit in memory all at once and every
  window ran with all-UNKNOWN neighbours, so the neighbour conditioning — and
  with it the ``s_nb`` guidance arm — was inert.
* *Sequential* generation walked a parity schedule with real neighbour
  conditioning, but each patch was denoised to completion on its own, so
  neighbouring patches only ever met at a plane and the seams showed.

The hybrid keeps both: the volume is cut into CHUNKS of ``chunk_tiles``
64-voxel tiles, chunks are generated in raster order, and inside a chunk
overlapping windows jointly denoise that chunk's own latent canvas.  A window's
six face neighbours are real again — they come from the current chunk's canvas
at the current timestep, or from an already finished chunk re-noised to that
same timestep, or they are OOB (the volume ends) or UNKNOWN (a chunk that does
not exist yet).  Every neighbour therefore arrives at a KNOWN noise level,
which is what ``nb_t`` carries into the denoiser.  ``chunk_tiles=(1,1,1)``
reduces the whole thing to patch-at-a-time sequential generation; a single
chunk covering the volume reduces it to pure joint denoising.

Where a window's neighbours COME FROM is :data:`NEIGHBOUR_MODES`.  Production is
``"canvas"`` — the chunk canvas and the finished chunks, described above.  The
other two exist so the production choice can be measured against its
alternatives and against a ceiling (``eval_v4``'s ``assembly_modes``), and
neither is ever the production path:

``"unknown"``
    every in-bounds face is the CFG null, so the neighbour conditioning is
    inert.  With one chunk over the whole volume this is exactly the ldm05
    joint sampler.  Faces where the canvas ENDS stay OOB: the arm removes
    neighbour content, not the fact that the volume has an edge.
``"reference"``
    every in-bounds face is read from ``reference_latents``, a canvas of REAL
    encoded material.  Teacher forcing: the upper bound the sampler would reach
    if the neighbours it assembles against were perfect.

Decoding is overlapped too: the finished latent canvas is decoded in windows at
``decode_stride`` voxels and the decoded grey levels and class logits are
blended with a tapered window (the fix validated on the VAE tile seams —
see ``poregen.eval.blended``).  Direct stride-64 tiling put a decoder-side
seam at every patch face; blending removes it.

Every random draw is taken in the frame of the REQUEST, not of the canvas: one
canvas-sized noise field per draw, rolled by ``request_offset`` before use
(:func:`region_noise_field`).  Translating a request inside a bigger canvas
therefore translates its noise with it, which is the only way an assembly-offset
comparison can hold the noise realisation fixed while the grid moves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poregen.diffusion.conditioning import (
    N_NEIGHBOURS,
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    POR_MAX,
    POR_MIN,
    dist6_from_box,
    porosity_to_cond,
)
from poregen.diffusion.noise_schedule import X0_CLAMP
from poregen.diffusion.orientation import orientation_tensor
from poregen.eval.blended import tukey_window_3d
from poregen.models.vae.base import decode_class_probs, decode_label

logger = logging.getLogger(__name__)

AXIS_NAMES = ("z", "y", "x")

#: Where a window's six face neighbours come from.  See the module docstring:
#: ``"canvas"`` is the production path and the other two are measurement arms.
NEIGHBOUR_MODES = ("canvas", "unknown", "reference")

# The decode blend window must never be exactly zero: a volume's own outer face
# is covered by a single decode window, and a zero weight there would leave the
# face undefined (0/0).  The floor lifts the Tukey taper off zero without
# changing the interior blend to three decimal places.
_DECODE_WINDOW_FLOOR = 1e-3

__all__ = [
    "AXIS_NAMES",
    "DDIMSampler",
    "NEIGHBOUR_MODES",
    "VolumeGenerator",
    "porosity_to_cond",
    "region_noise_field",
    "rescale_guidance",
    "theta_from_layup",
    "window_origins",
    "window_tile_mean",
    "window_weight",
    "seam_discontinuity",
]


def rescale_guidance(
    guided: torch.Tensor,
    conditional: torch.Tensor,
    phi: float,
) -> torch.Tensor:
    """Rescale a CFG-combined prediction back onto the conditional arm's scale.

    Lin et al. 2024 (arXiv:2305.08891) §3.4.  A guidance scale above 1 is an
    extrapolation, and extrapolation inflates the standard deviation of the
    combined prediction; the inflated prediction decodes to an over-exposed
    sample.  Correcting it is a two-step operation — rescale to the
    conditional arm's own per-item standard deviation, then interpolate back
    towards the raw guided value by ``phi`` so the correction can be dialled
    rather than being all-or-nothing::

        rescaled = guided · std(conditional) / std(guided)
        out      = phi·rescaled + (1 − phi)·guided

    The standard deviation is taken per item over every non-batch dimension,
    which is where the inflation lives.

    Parameters
    ----------
    guided      : (B, C, …) the CFG-combined model output
    conditional : (B, C, …) the full-conditional arm, the reference scale
    phi         : interpolation factor in [0, 1]; the paper uses 0.7
    """
    dims = tuple(range(1, guided.ndim))
    std_cond = conditional.float().std(dim=dims, keepdim=True)
    std_guided = guided.float().std(dim=dims, keepdim=True).clamp(min=1e-8)
    rescaled = guided * (std_cond / std_guided).to(guided.dtype)
    return phi * rescaled + (1.0 - phi) * guided


def theta_from_layup(
    depth_vox: int,
    ply_angles_deg,
    ply_thickness_vox: float,
) -> np.ndarray:
    """θ(z) in degrees for a requested layup — the generation-side field.

    At generation time θ(z) is written directly from the requested stacking
    sequence and ply thickness; nothing is estimated.  Each ply occupies
    ``ply_thickness_vox`` voxels and the sequence repeats if the volume is
    deeper than the stack.  The encoding into (cos2θ, sin2θ) is done by
    :func:`poregen.diffusion.orientation.orientation_tensor`, the single shared
    implementation used by training as well.

    Returns
    -------
    (depth_vox,) float32 — degrees.
    """
    angles = np.asarray(ply_angles_deg, dtype=np.float64)
    if angles.size == 0:
        raise ValueError("ply_angles_deg must contain at least one ply angle.")
    z = np.arange(depth_vox, dtype=np.float64)
    ply_idx = (np.floor(z / float(ply_thickness_vox)).astype(np.int64)) % angles.size
    return angles[ply_idx].astype(np.float32)


# ── overlapping-window helpers ────────────────────────────────────────────────

def window_origins(
    canvas_cells: tuple[int, int, int],
    win_cells: int,
    stride_cells: int,
) -> list[tuple[int, int, int]]:
    """Origins (in latent cells) of the overlapping windows over one canvas.

    Windows of side ``win_cells`` are placed at ``stride_cells`` spacing on
    each axis.  Every canvas cell must be covered, so each axis must hold a
    whole number of strides: ``(n - win) % stride == 0`` with the last window
    ending exactly at the canvas edge.
    """
    w, s = int(win_cells), int(stride_cells)
    if s <= 0 or s > w:
        raise ValueError(f"stride_cells={s} must be in [1, win_cells={w}].")
    axes: list[list[int]] = []
    for n in canvas_cells:
        n = int(n)
        if n < w or (n - w) % s != 0:
            raise ValueError(
                f"Canvas axis of {n} cells cannot be covered by windows of "
                f"{w} cells at stride {s}: the last window must end exactly at "
                f"the canvas edge ((n - window) % stride == 0)."
            )
        axes.append(list(range(0, n - w + 1, s)))
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


def window_tile_mean(
    origin: tuple[int, int, int],
    patch_size: int,
    tile_field: dict,
    default: float,
) -> float:
    """Mean of a TILE-grid field over one window's voxel footprint.

    ``origin`` is the window's voxel origin and the window spans
    ``patch_size`` voxels on each axis — exactly one tile's worth, but placed
    every ``window_stride`` voxels, so it generally straddles up to eight
    tiles.  Each tile is weighted by the VOLUME of the window it covers, and
    tiles the field does not name contribute ``default``.

    The average is on RAW values.  Averaging ``cond_por`` instead would be
    wrong: ``porosity_to_cond`` is a log, and the mean of the transform is not
    the transform of the mean.
    """
    P = int(patch_size)
    spans: list[list[tuple[int, int]]] = []
    for a in range(3):
        o = int(origin[a])
        axis: list[tuple[int, int]] = []
        v = o
        while v < o + P:
            end = min((v // P + 1) * P, o + P)
            axis.append((v // P, end - v))
            v = end
        spans.append(axis)
    total = 0.0
    for iz, wz in spans[0]:
        for iy, wy in spans[1]:
            for ix, wx in spans[2]:
                total += float(tile_field.get((iz, iy, ix), default)) * wz * wy * wx
    return total / float(P ** 3)


def region_noise_field(
    channels: int,
    canvas_cells: tuple[int, int, int],
    offset_cells: tuple[int, int, int],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(C, Z, Y, X) standard normal noise in the REGION-RELATIVE frame.

    Every random draw of the reverse process is anchored to the REQUEST, not to
    the canvas: the field is drawn once in the frame of the requested region and
    then rolled onto the canvas, so the value used at canvas cell ``p`` is the
    value the request sees at region cell ``p - offset``.  Translating the
    request therefore translates its noise with it.

    This is what makes the assembly-offset comparison mean anything.  With the
    draw anchored to the canvas, the same region generated at two offsets
    differs in the assembly geometry AND in the noise realisation, and no metric
    over the pair can say which of the two moved the answer.  With the draw
    anchored to the region, the noise is held and the geometry is the only
    thing left that differs.

    ``torch.roll`` is a permutation of one draw, so the field is still exactly
    iid standard normal, and the wrap only reaches canvas cells outside the
    requested region.
    """
    n = torch.randn(int(channels), *canvas_cells, device=device, generator=generator)
    shifts = tuple(int(o) for o in offset_cells)
    if any(shifts):
        n = torch.roll(n, shifts=shifts, dims=(1, 2, 3))
    return n


def window_weight(win_cells: int) -> torch.Tensor:
    """(L, L, L) separable cosine (Hann-type) fusion weight, strictly positive.

    ``w1d[i] = sin²(π·(i + 0.5)/L)`` — the half-cell offset keeps every entry
    positive, so the per-voxel weight normalisation is well defined even where
    a single window covers a canvas corner.  Cosine rather than uniform
    weighting: a uniform average makes the effective per-voxel weight field
    piecewise constant with jumps exactly at window borders (where the model
    has the least receptive-field context), re-introducing a grid of weak
    seams into the fused prediction field.  The cosine profile down-weights each
    window's border predictions and varies smoothly across the canvas.
    """
    L = int(win_cells)
    i = torch.arange(L, dtype=torch.float32) + 0.5
    w1 = torch.sin(np.pi * i / L) ** 2
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def seam_discontinuity(
    volume: np.ndarray,
    period,
    prefix: str = "seam",
    interior_exclude=None,
) -> dict[str, float]:
    """Discontinuity across the planes where two independent answers meet.

    This is the primary assembly-quality metric.  For each axis the mean
    absolute slice-to-slice difference is computed at the SEAM planes (boundary
    index a multiple of ``period``) and at every INTERIOR plane.  The interior
    value is the natural slice-to-slice variation of the material and is the
    baseline the seam is judged against::

        ratio = seam_mad / interior_mad

    ``ratio ≈ 1`` means the seam is indistinguishable from ordinary internal
    texture change — the assembly is continuous.  ``ratio >> 1`` means the two
    sides disagree where they meet and the seam is visible.

    Parameters
    ----------
    volume  : (D, H, W) decoded volume, any continuous scale
    period  : seam spacing in voxels — an int, or one value per axis
    prefix  : metric-name prefix (e.g. ``"seam_xct"``)
    interior_exclude : planes that are multiples of this are excluded from the
        interior baseline (int or per-axis).  Defaults to ``period``.  Set it
        to the window period when measuring a coarser chunk period, so both
        metrics are judged against the SAME baseline.

    Returns
    -------
    Flat metric dict::

        {prefix}_{z,y,x}_planes         number of seam planes on that axis
        {prefix}_{z,y,x}_mad            mean |Δ| across the seam planes
        {prefix}_{z,y,x}_interior_mad   mean |Δ| across all other planes
        {prefix}_{z,y,x}_ratio          seam_mad / interior_mad
        {prefix}_planes                 total seam planes
        {prefix}_mad, {prefix}_interior_mad, {prefix}_ratio   volume aggregate
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"seam_discontinuity expects a 3-D volume, got {vol.shape}.")
    periods = (int(period),) * 3 if np.isscalar(period) else tuple(int(p) for p in period)
    if interior_exclude is None:
        excludes = periods
    elif np.isscalar(interior_exclude):
        excludes = (int(interior_exclude),) * 3
    else:
        excludes = tuple(int(p) for p in interior_exclude)

    metrics: dict[str, float] = {}
    seam_sum = seam_w = int_sum = int_w = 0.0
    total_planes = 0

    for axis, name in enumerate(AXIS_NAMES):
        n = vol.shape[axis]
        if n < 2:
            continue
        diff = np.abs(np.diff(vol, axis=axis))
        others = tuple(i for i in range(3) if i != axis)
        per_plane = diff.mean(axis=others)               # (n-1,)
        plane_elems = float(diff.size) / float(n - 1)    # voxels behind each mean
        boundary = np.arange(1, n)                       # plane between k-1 and k
        is_seam = (boundary % periods[axis]) == 0
        is_int = (boundary % excludes[axis]) != 0

        n_seam = int(is_seam.sum())
        total_planes += n_seam
        metrics[f"{prefix}_{name}_planes"] = float(n_seam)

        seam_mad = float(per_plane[is_seam].mean()) if n_seam else float("nan")
        n_int = int(is_int.sum())
        int_mad = float(per_plane[is_int].mean()) if n_int else float("nan")
        metrics[f"{prefix}_{name}_mad"] = seam_mad
        metrics[f"{prefix}_{name}_interior_mad"] = int_mad
        metrics[f"{prefix}_{name}_ratio"] = (
            seam_mad / int_mad if (n_seam and n_int and int_mad > 1e-12) else float("nan")
        )

        if n_seam:
            seam_sum += seam_mad * n_seam * plane_elems
            seam_w   += n_seam * plane_elems
        if n_int:
            int_sum += int_mad * n_int * plane_elems
            int_w   += n_int * plane_elems

    metrics[f"{prefix}_planes"] = float(total_planes)
    agg_seam = seam_sum / seam_w if seam_w else float("nan")
    agg_int  = int_sum / int_w if int_w else float("nan")
    metrics[f"{prefix}_mad"] = agg_seam
    metrics[f"{prefix}_interior_mad"] = agg_int
    metrics[f"{prefix}_ratio"] = (
        agg_seam / agg_int if (seam_w and int_w and agg_int > 1e-12) else float("nan")
    )
    return metrics


class DDIMSampler:
    """DDIM patch sampler — deterministic inference in n_steps < T steps.

    Parameters
    ----------
    model    : UNet3DDenoiser
    schedule : DDPMSchedule
    device   : torch.device
    n_steps  : int — number of denoising steps (default 50)
    s_por    : float — porosity guidance scale; 1.0 = un-guided (default)
    s_nb     : float — neighbour guidance scale; 1.0 = un-guided (default)
    cfg_rescale : float — guidance rescale factor φ (Lin et al. 2024 §3.4);
                  0.0 = off (default), the paper's setting is 0.7

    When both scales are 1.0 the standard single-pass full-conditional denoiser
    is used.  Any other combination activates the 3-pass nested CFG
    decomposition, written on the model's RAW OUTPUT — ε under an ε-objective
    schedule, v under a v-objective one::

        out_uncond = model(z_t, t, nb, ALL_UNK, nb_t=0, por, …, drop_por=True)
        out_por    = model(z_t, t, nb, ALL_UNK, nb_t=0, por, …, drop_por=False)
        out_full   = model(z_t, t, nb, REAL,    nb_t,    por, …, drop_por=False)
        out = out_uncond + s_por*(out_por - out_uncond) + s_nb*(out_full - out_por)

    Combining in v-space is the same operation as combining in ε-space: at a
    fixed ``t`` the map v ↔ ε is affine with shared coefficients, so an affine
    combination of the three arms commutes with it.  The conversion to x̂₀/ε̂
    happens once, in :meth:`DDPMSchedule.ddim_step`.

    At s_por=s_nb=1 this telescopes to out_full — exact un-guided equality.
    The ALL_UNKNOWN arms use ``nb_t = 0`` because that is exactly the neighbour
    null the training step draws (``drop_nb``).  Position, orientation and
    material are always on in all three passes.

    ``cfg_rescale`` addresses the over-exposure Lin et al. describe: raising a
    guidance scale inflates the standard deviation of the combined prediction,
    which pushes the sample towards saturated extremes.  The fix rescales the
    guided output back to the conditional arm's own standard deviation and then
    interpolates by φ::

        rescaled = out · std(out_full) / std(out)
        out      = φ·rescaled + (1 − φ)·out

    φ = 0 leaves the guided output untouched, which is why it is the default:
    the correction only makes sense once a guidance scale is actually above 1.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        schedule: Any,
        device: torch.device,
        n_steps: int = 50,
        s_por: float = 1.0,
        s_nb: float = 1.0,
        cfg_rescale: float = 0.0,
    ) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.n_steps  = n_steps
        self.s_por    = s_por
        self.s_nb     = s_nb
        self.cfg_rescale = float(cfg_rescale)
        self.guided   = not (s_por == 1.0 and s_nb == 1.0)
        T = schedule.T
        ts = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long)
        # Store as Python ints for torch.compile compatibility (no dynamic shapes)
        self.timesteps: list[int] = ts.flip(0).tolist()   # [T-1, ..., 0]
        # The chain must start at the schedule's most-noisy step: under zero
        # terminal SNR that is the only index where ᾱ = 0, i.e. the pure-noise
        # state the initial randn actually is.  Starting one step in would hand
        # the model a latent it believes still carries signal.
        if self.timesteps[0] != T - 1 or self.timesteps[-1] != 0:
            raise ValueError(
                f"DDIM timestep grid must run from T-1={T - 1} down to 0, got "
                f"{self.timesteps[0]} … {self.timesteps[-1]}."
            )

    def predict_out(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        nb_t: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist6: torch.Tensor,
        cond_orient: torch.Tensor,
        cond_material: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """One denoiser output for a batch, honouring the CFG guidance scales.

        The return value is in the schedule's own objective space (ε or v) —
        never converted here.  This is the single choke point every sampling
        loop goes through: the chunked joint path calls it directly to get raw
        per-window predictions before fusing them on the chunk canvas, and a
        fusion of v predictions at one shared ``t`` is the same latent as a
        fusion of the matching ε predictions.
        """
        if not self.guided:
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                return self.model(x, t, nb_latents, nb_avail, nb_t, cond_por,
                                  cond_depth, cond_dist6, cond_orient, cond_material)

        B = x.shape[0]
        all_unk  = torch.full_like(nb_avail, NB_UNKNOWN)
        zero_t   = torch.zeros_like(nb_t)
        drop_all = torch.ones(B, dtype=torch.bool, device=x.device)

        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            out_uncond = self.model(x, t, nb_latents, all_unk, zero_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, drop_all)
            out_por    = self.model(x, t, nb_latents, all_unk, zero_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, None)
            out_full   = self.model(x, t, nb_latents, nb_avail, nb_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, None)

        guided = (
            out_uncond
            + self.s_por * (out_por  - out_uncond)
            + self.s_nb  * (out_full - out_por)
        )
        if self.cfg_rescale > 0.0:
            guided = rescale_guidance(guided, out_full, self.cfg_rescale)
        return guided

    @torch.no_grad()
    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        nb_t: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist6: torch.Tensor,
        cond_orient: torch.Tensor,
        cond_material: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_x0_saturation: bool = False,
        return_intermediates: bool = False,
        generator: torch.Generator | None = None,
    ) -> Any:
        """Run the DDIM reverse process for a batch of independent patches.

        Neighbour conditioning is held FIXED across the reverse process here —
        this is the patch-level sampler used by the in-training diagnostics,
        not the volume path.  ``nb_t`` therefore describes the noise level of
        the neighbours as handed in, and stays constant.

        ``generator`` seeds the one random draw this method makes — the initial
        noise the chain starts from.  Pass one to make the batch reproducible
        without touching the global torch generator; ``None`` draws from the
        global one, which is what an unseeded caller wants.

        Parameters
        ----------
        nb_latents  : (B, 6, C, D, D, D) on device
        nb_avail    : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN
        nb_t        : (B, 6) long — per-neighbour noise level
        cond_por    : (B,) float — standardised log porosity
        cond_depth  : (B,) float — relative depth
        cond_dist6  : (B, 6) float — per-face distance to the specimen box
        cond_orient : (B, 2, D, D, D) float
        cond_material : (B, 1, D, D, D) float
        return_x0_saturation : also return the fraction of x0-prediction
            elements hitting the ±10 clamp inside ddim_step, averaged over all
            denoising steps — an off-manifold diagnostic.
        return_intermediates : also return the latent after each step, on CPU.

        Returns
        -------
        (B, C, D, D, D) float32 on device, plus the requested extras in the
        order (samples, sat_frac, intermediates).
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        B = nb_latents.shape[0]
        C = nb_latents.shape[2]
        D = nb_latents.shape[3]

        x = torch.randn(B, C, D, D, D, device=self.device, generator=generator)
        sat_sum = torch.zeros((), device=self.device)
        n_steps = 0
        inter: list[torch.Tensor] = []
        for i, t_val in enumerate(self.timesteps[:-1]):
            t_prev_val = self.timesteps[i + 1]
            t      = torch.full((B,), t_val,      dtype=torch.long, device=self.device)
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=self.device)
            model_out = self.predict_out(
                x, t, nb_latents, nb_avail, nb_t, cond_por, cond_depth,
                cond_dist6, cond_orient, cond_material, autocast_dtype,
            )
            if return_x0_saturation:
                x0_pred = schedule.predict_x0(x, t, model_out)
                sat_sum += (x0_pred.abs() >= X0_CLAMP).float().mean()
                n_steps += 1
            x = schedule.ddim_step(x, t, t_prev, model_out)
            if return_intermediates:
                inter.append(x.float().cpu())

        out: list[Any] = [x.float()]
        if return_x0_saturation:
            out.append((sat_sum / max(n_steps, 1)).item())
        if return_intermediates:
            out.append(inter)
        return out[0] if len(out) == 1 else tuple(out)


class VolumeGenerator:
    """Generate a full synthetic volume with hybrid chunked joint denoising.

    Geometry, in three layers:

    ``tile``   64 voxels = ``patch_size`` = ``neighbour_offset``.  The unit the
               model was trained on, and the unit the seam metric measures.
    ``window`` one tile-sized denoising window, placed every ``window_stride``
               voxels inside a chunk (default 32 = 50 % overlap).
    ``chunk``  ``chunk_tiles`` tiles per axis (default 3×3×3 = 192³ voxels).
               One chunk is jointly denoised at a time; chunks run in raster
               order.  ``(1, 1, 1)`` is patch-at-a-time sequential generation.

    Neighbour conditioning inside a chunk, per window and per face:

    * the block lies in the current chunk → take it from the chunk canvas
      ``x_t``, EXISTS at ``nb_t = t``;
    * the block lies in an already finished chunk → take that chunk's clean
      latent and re-noise it to ``t`` with fresh noise (``q_sample``), EXISTS
      at ``nb_t = t``;
    * the block leaves the volume → OOB (the specimen ends there);
    * the block reaches into a chunk that has not been generated → UNKNOWN.

    Both EXISTS sources are at the same noise level ``t``, so a block that
    straddles the current chunk and a finished one is still coherent.

    Parameters
    ----------
    sampler       : DDIMSampler
    vae           : the frozen 3-class VAE (``decoder``, ``xct_head``, ``class_head``)
    device        : torch.device
    patch_size    : voxel side length of one tile (default 64)
    latent_size   : latent cells per tile (default 16)
    latent_mean   : float | (C,1,1,1) per-channel normalisation mean; generated
                    latents are denormalised ``z*std + mean`` before decoding
    latent_std    : float | (C,1,1,1) per-channel normalisation std
    voxel_size_mm : physical voxel size in millimetres (default 0.025 = 25 µm)
    por_log_stats : (mean, std) of ``log(phi + 1e-3)`` on the train split, from
                    the latent store metadata.  Required.
    theta_deg     : (vol_d,) requested θ(z) in degrees for the whole volume
                    (see :func:`theta_from_layup`), NaN where unknown.  Required.
    chunk_tiles   : tiles per chunk per axis (default (3, 3, 3))
    window_stride : voxels between window origins inside a chunk (default 32)
    decode_stride : voxels between decode window origins (default 32)
    neighbour_mode: one of :data:`NEIGHBOUR_MODES` (default ``"canvas"``, the
                    production path).  See the module docstring.
    reference_latents : (C, Z, Y, X) NORMALISED clean latents covering the whole
                    canvas, required by — and only by — ``neighbour_mode
                    ="reference"``.  Every in-bounds neighbour is read from it
                    at the window's own canvas position.
    """

    def __init__(
        self,
        sampler: DDIMSampler,
        vae: torch.nn.Module,
        device: torch.device,
        patch_size: int = 64,
        latent_size: int = 16,
        latent_mean: torch.Tensor | float = 0.0,
        latent_std: torch.Tensor | float = 1.0,
        voxel_size_mm: float = 0.025,
        por_log_stats: tuple[float, float] | None = None,
        theta_deg: np.ndarray | None = None,
        chunk_tiles: tuple[int, int, int] = (3, 3, 3),
        window_stride: int = 32,
        decode_stride: int = 32,
        neighbour_mode: str = "canvas",
        reference_latents: torch.Tensor | None = None,
    ) -> None:
        if patch_size % latent_size:
            raise ValueError(
                f"patch_size={patch_size} is not a multiple of latent_size={latent_size}."
            )
        self.sampler       = sampler
        self.vae           = vae
        self.device        = device
        self.patch_size    = int(patch_size)
        self.latent_size   = int(latent_size)
        self.latent_mean   = latent_mean
        self.latent_std    = latent_std
        self.voxel_size_mm = float(voxel_size_mm)
        self.por_log_stats = por_log_stats
        self.theta_deg     = None if theta_deg is None else np.asarray(theta_deg)
        self.chunk_tiles   = tuple(int(c) for c in chunk_tiles)
        self.window_stride = int(window_stride)
        self.decode_stride = int(decode_stride)
        self.downsample    = self.patch_size // self.latent_size
        self.z_channels    = sampler.model.cfg.z_channels
        self.neighbour_mode = str(neighbour_mode)
        self.reference_latents = reference_latents

        if self.neighbour_mode not in NEIGHBOUR_MODES:
            raise ValueError(
                f"neighbour_mode must be one of {NEIGHBOUR_MODES}, got "
                f"{self.neighbour_mode!r}."
            )
        if (self.reference_latents is not None) != (self.neighbour_mode == "reference"):
            raise ValueError(
                "reference_latents and neighbour_mode='reference' go together: "
                f"got neighbour_mode={self.neighbour_mode!r} with "
                f"reference_latents={'a tensor' if reference_latents is not None else None}. "
                "A reference canvas that no window reads would claim a teacher-forced "
                "run that did not happen."
            )
        if any(c < 1 for c in self.chunk_tiles):
            raise ValueError(f"chunk_tiles must all be >= 1, got {self.chunk_tiles}.")
        for name, stride in (("window_stride", self.window_stride),
                             ("decode_stride", self.decode_stride)):
            if stride <= 0 or self.patch_size % stride or stride % self.downsample:
                raise ValueError(
                    f"{name}={stride} must be a positive divisor of "
                    f"patch_size={self.patch_size} and a multiple of the VAE "
                    f"downsampling factor {self.downsample}."
                )

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _volume_shape(self, volume_size_mm: tuple[float, float, float]) -> tuple[int, int, int]:
        """Snap a physical size to a whole number of tiles, in voxels."""
        P = self.patch_size
        shape = tuple((round(d / self.voxel_size_mm) // P) * P for d in volume_size_mm)
        for d_mm, snapped in zip(volume_size_mm, shape):
            raw = round(d_mm / self.voxel_size_mm)
            if raw - snapped > 1:
                logger.warning(
                    "Volume axis %.3f mm: snapped %d vox → %d vox (dropped %d vox)",
                    d_mm, raw, snapped, raw - snapped,
                )
        if any(s < P for s in shape):
            raise ValueError(
                f"Volume {volume_size_mm} mm snaps to {shape} voxels — every axis "
                f"must hold at least one {P}-voxel tile."
            )
        return shape  # type: ignore[return-value]

    def _chunk_ranges(self, n_tiles: int, per_chunk: int) -> list[tuple[int, int]]:
        """[(tile_lo, tile_hi), …] partition of one axis into chunks."""
        return [
            (i, min(i + per_chunk, n_tiles))
            for i in range(0, n_tiles, per_chunk)
        ]

    # ── per-window conditioning ──────────────────────────────────────────────

    def _window_orient(self, z0: int) -> torch.Tensor:
        """(2, L, L, L) orientation profile for a window at depth origin z0."""
        if self.theta_deg is None:
            raise ValueError(
                "The denoiser conditions on ply orientation but no theta_deg was "
                "given. Build one with theta_from_layup(depth_vox, layup, "
                "ply_thickness_vox)."
            )
        P = self.patch_size
        if z0 + P > len(self.theta_deg):
            raise ValueError(
                f"theta_deg covers {len(self.theta_deg)} voxels — too few for a "
                f"window at z0={z0} (needs {z0}..{z0 + P - 1})."
            )
        return torch.from_numpy(
            orientation_tensor(self.theta_deg[z0 : z0 + P], self.latent_size)
        )

    def _window_position(
        self,
        origin: tuple[int, int, int],
        box_lo: tuple[int, int, int],
        box_hi: tuple[int, int, int],
    ) -> tuple[float, np.ndarray]:
        """(cond_depth, cond_dist6) for one window against the specimen box.

        ``cond_depth`` is the window centre's fractional depth in z inside the
        box; ``cond_dist6`` is the gap from each window face to the matching box
        face, capped at 64 voxels — the same definitions
        ``scripts/build_conditioning.py`` applies at training time (there the
        box comes from the volume's foreground extent).
        """
        centre_z = origin[0] + self.patch_size / 2.0
        span = max(box_hi[0] - box_lo[0], 1)
        depth = float(np.clip((centre_z - box_lo[0]) / span, 0.0, 1.0))
        return depth, dist6_from_box(origin, self.patch_size, box_lo, box_hi)

    # ── latent generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_latents(
        self,
        volume_shape: tuple[int, int, int],
        *,
        target_porosity: float | None,
        local_por_map: dict | None,
        material_map: np.ndarray | None,
        specimen_box: tuple[tuple[int, int, int], tuple[int, int, int]],
        autocast_dtype: torch.dtype,
        window_batch: int,
        offset_cells: tuple[int, int, int] = (0, 0, 0),
        progress=None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise the whole latent canvas chunk by chunk.  Returns (C, Z, Y, X).

        Two random draws happen here and nowhere else in generation: the noise
        the canvas starts from, and the fresh noise that re-noises the
        already-finished chunks to the current timestep.  Both take
        ``generator``, so a seeded call is reproducible without disturbing the
        global torch generator, and both are drawn in the region-relative frame
        ``offset_cells`` names (see :func:`region_noise_field`) — the initial
        field once for the whole canvas, the re-noising field afresh at every
        timestep of every chunk.
        """
        sampler  = self.sampler
        schedule = sampler.schedule.to(self.device)
        sampler.model.eval()

        ds, L, P = self.downsample, self.latent_size, self.patch_size
        C = self.z_channels
        canvas_cells = tuple(v // ds for v in volume_shape)
        n_tiles = tuple(v // P for v in volume_shape)
        s_cells = self.window_stride // ds
        box_lo, box_hi = specimen_box

        por_default = (
            float(np.clip(target_porosity, POR_MIN, POR_MAX))
            if target_porosity is not None else 0.05
        )
        if material_map is None:
            material_map = self._default_material_map(canvas_cells, box_lo, box_hi)
        material_map = np.asarray(material_map, dtype=np.float32)
        if material_map.shape != canvas_cells:
            raise ValueError(
                f"material_map has shape {material_map.shape}, expected the latent "
                f"canvas {canvas_cells} (one cell per {ds}³ voxels)."
            )

        z_clean = torch.zeros(C, *canvas_cells, device=self.device)
        available = np.zeros(canvas_cells, dtype=bool)   # cells of finished chunks

        reference = self.reference_latents
        if reference is not None:
            if tuple(reference.shape) != (C, *canvas_cells):
                raise ValueError(
                    f"reference_latents has shape {tuple(reference.shape)}, expected "
                    f"the latent canvas {(C, *canvas_cells)} of volume {volume_shape}."
                )
            reference = reference.to(self.device, dtype=torch.float32)

        chunk_grid = [
            self._chunk_ranges(n_tiles[a], self.chunk_tiles[a]) for a in range(3)
        ]
        chunks = [(cz, cy, cx) for cz in chunk_grid[0]
                  for cy in chunk_grid[1] for cx in chunk_grid[2]]
        timesteps = sampler.timesteps

        # ONE canvas-sized draw for the whole volume, in the request's frame:
        # every chunk starts from its own block of it.  Drawing per chunk would
        # be the same distribution but would tie the realisation to the chunk
        # grid, which is exactly what the assembly offset moves.
        init_noise = region_noise_field(
            C, canvas_cells, offset_cells, self.device, generator
        )

        logger.info(
            "VolumeGenerator: %s voxels = %s tiles, %d chunk(s) of %s tiles, "
            "windows every %d voxels, %d DDIM steps, neighbours from %s",
            volume_shape, n_tiles, len(chunks), self.chunk_tiles,
            self.window_stride, len(timesteps) - 1, self.neighbour_mode,
        )

        for chunk_idx, chunk in enumerate(chunks):
            lo = tuple(chunk[a][0] * L for a in range(3))           # cell lo
            hi = tuple(chunk[a][1] * L for a in range(3))           # cell hi
            chunk_cells = tuple(hi[a] - lo[a] for a in range(3))
            ctx_lo = tuple(max(lo[a] - L, 0) for a in range(3))
            ctx_hi = tuple(min(hi[a] + L, canvas_cells[a]) for a in range(3))
            ctx_sl = tuple(slice(ctx_lo[a], ctx_hi[a]) for a in range(3))
            cur_sl = tuple(slice(lo[a] - ctx_lo[a], hi[a] - ctx_lo[a]) for a in range(3))
            chunk_sl = tuple(slice(lo[a], hi[a]) for a in range(3))

            # Availability as the windows of THIS chunk see it.  The three
            # neighbour modes differ HERE and nowhere else in the plan: what a
            # window may treat as a real neighbour is a property of the canvas,
            # so _neighbour_plan needs no mode of its own.
            if self.neighbour_mode == "unknown":
                visible = np.zeros(canvas_cells, dtype=bool)   # every face is the CFG null
            elif self.neighbour_mode == "reference":
                visible = np.ones(canvas_cells, dtype=bool)    # every face is real material
            else:
                # finished chunks plus the chunk being generated
                visible = available.copy()
                visible[chunk_sl] = True
            done_mask = torch.from_numpy(available[ctx_sl].astype(np.float32)).to(self.device)

            origins = window_origins(chunk_cells, L, s_cells)
            g_origins = [tuple(lo[a] + o[a] for a in range(3)) for o in origins]
            n_win = len(g_origins)

            states, nb_slices = self._neighbour_plan(g_origins, visible, canvas_cells, ctx_lo)
            cond = self._window_conditioning(
                g_origins, ds, por_default, local_por_map, material_map,
                box_lo, box_hi,
            )
            avail_t = torch.from_numpy(states).to(self.device)              # (n_win, 6)

            win_sl = [
                (slice(o[0], o[0] + L), slice(o[1], o[1] + L), slice(o[2], o[2] + L))
                for o in origins
            ]
            weight = window_weight(L).to(self.device)
            weight_sum = torch.zeros((1, 1, *chunk_cells), device=self.device)
            for sl in win_sl:
                weight_sum[0, 0, sl[0], sl[1], sl[2]] += weight

            x = init_noise[(slice(None), *chunk_sl)].unsqueeze(0).clone()
            B_max = max(1, min(int(window_batch), n_win))

            for i, t_val in enumerate(timesteps[:-1]):
                t_prev_val = timesteps[i + 1]
                # Context canvas at this timestep: finished chunks re-noised to
                # t with FRESH noise, the current chunk at its live state.
                t_one = torch.full((1,), t_val, dtype=torch.long, device=self.device)
                # The re-noising draw runs in the same frame as the initial one:
                # a canvas-sized field, of which this chunk's context block is a
                # slice.  Drawing the context block on its own would anchor it
                # to the chunk again.  The field is drawn even when nothing is
                # finished yet (``done_mask`` is all zero there), so the draw
                # order does not depend on where the request sits.
                ctx = None
                if self.neighbour_mode == "reference":
                    # Teacher forcing: the context is REAL material at the same
                    # positions, re-noised to t like any other EXISTS neighbour.
                    # The current chunk's own live state does not enter it — a
                    # ceiling means every one of the six faces is perfect, not
                    # only the ones outside the chunk.
                    ctx_noise = region_noise_field(
                        C, canvas_cells, offset_cells, self.device, generator
                    )
                    ctx = schedule.q_sample(
                        reference[(slice(None), *ctx_sl)].unsqueeze(0),
                        t_one,
                        noise=ctx_noise[(slice(None), *ctx_sl)].unsqueeze(0),
                    )
                elif self.neighbour_mode == "canvas":
                    ctx_clean = z_clean[(slice(None), *ctx_sl)].unsqueeze(0)
                    ctx_noise = region_noise_field(
                        C, canvas_cells, offset_cells, self.device, generator
                    )
                    ctx = schedule.q_sample(
                        ctx_clean,
                        t_one,
                        noise=ctx_noise[(slice(None), *ctx_sl)].unsqueeze(0),
                    ) * done_mask
                    ctx[(0, slice(None), *cur_sl)] = x[0]
                # neighbour_mode == "unknown" builds no context at all: every
                # face is UNKNOWN, so nb_slices is all None and nothing reads it.

                out_sum = torch.zeros_like(x)
                for start in range(0, n_win, B_max):
                    idx = list(range(start, min(start + B_max, n_win)))
                    B = len(idx)
                    xw = torch.stack(
                        [x[0, :, win_sl[j][0], win_sl[j][1], win_sl[j][2]] for j in idx]
                    )
                    nb = torch.zeros(B, N_NEIGHBOURS, C, L, L, L, device=self.device)
                    for k, j in enumerate(idx):
                        for f in range(N_NEIGHBOURS):
                            sl = nb_slices[j][f]
                            if sl is not None:
                                nb[k, f] = ctx[(0, slice(None), *sl)]
                    av = avail_t[idx]
                    nb_t = torch.where(
                        av == NB_EXISTS,
                        torch.full_like(av, t_val),
                        torch.zeros_like(av),
                    )
                    t_b = torch.full((B,), t_val, dtype=torch.long, device=self.device)
                    out = sampler.predict_out(
                        xw, t_b, nb, av, nb_t,
                        cond["por"][idx], cond["depth"][idx], cond["dist6"][idx],
                        cond["orient"][idx], cond["material"][idx], autocast_dtype,
                    ).float()
                    for k, j in enumerate(idx):
                        sl = win_sl[j]
                        out_sum[0, :, sl[0], sl[1], sl[2]] += weight * out[k]

                t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
                t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
                x = schedule.ddim_step(x, t, t_prev, out_sum / weight_sum)
                if progress is not None:
                    progress.update(1)

            z_clean[(slice(None), *chunk_sl)] = x[0]
            available[chunk_sl] = True
            logger.info("Chunk %d/%d done (tiles %s)", chunk_idx + 1, len(chunks), chunk)

        return z_clean

    def _default_material_map(
        self,
        canvas_cells: tuple[int, int, int],
        box_lo: tuple[int, int, int],
        box_hi: tuple[int, int, int],
    ) -> np.ndarray:
        """The EXACT fraction of every latent cell the specimen box covers.

        ``cond_material`` is the envelope fraction per cell, so a cell the box
        crosses is neither 1 nor 0 — rounding the box to whole cells would tell
        the model the surface cells are solid specimen or pure air, which is
        the one place the map carries information.  The box is axis aligned, so
        the intersection volume factorises: the fraction is the product of the
        three per-axis overlaps, in closed form.
        """
        ds = float(self.downsample)
        axis: list[np.ndarray] = []
        for a in range(3):
            edge = np.arange(canvas_cells[a] + 1, dtype=np.float64) * ds
            lo = np.clip(edge[:-1], box_lo[a], box_hi[a])
            hi = np.clip(edge[1:], box_lo[a], box_hi[a])
            axis.append((hi - lo) / ds)
        m = (axis[0][:, None, None] * axis[1][None, :, None]
             * axis[2][None, None, :])
        return m.astype(np.float32)

    def _neighbour_plan(
        self,
        g_origins: list[tuple[int, int, int]],
        visible: np.ndarray,
        canvas_cells: tuple[int, int, int],
        ctx_lo: tuple[int, int, int],
    ) -> tuple[np.ndarray, list[list[tuple[slice, slice, slice] | None]]]:
        """Availability state and context slice of every window's six faces.

        The state does not change during a chunk's reverse process — only the
        content of the context canvas does — so it is computed once.
        """
        L = self.latent_size
        states = np.full((len(g_origins), N_NEIGHBOURS), NB_OOB, dtype=np.int64)
        slices: list[list[tuple[slice, slice, slice] | None]] = []
        for j, g in enumerate(g_origins):
            row: list[tuple[slice, slice, slice] | None] = []
            for f, d in enumerate(NEIGHBOUR_DIRS):
                a = tuple(g[k] + d[k] * L for k in range(3))
                if any(a[k] < 0 or a[k] + L > canvas_cells[k] for k in range(3)):
                    states[j, f] = NB_OOB           # the volume ends this way
                    row.append(None)
                    continue
                block = tuple(slice(a[k], a[k] + L) for k in range(3))
                if not visible[block].all():
                    states[j, f] = NB_UNKNOWN       # a chunk not generated yet
                    row.append(None)
                    continue
                states[j, f] = NB_EXISTS
                row.append(tuple(slice(a[k] - ctx_lo[k], a[k] - ctx_lo[k] + L)
                                 for k in range(3)))
            slices.append(row)
        return states, slices

    def _window_conditioning(
        self,
        g_origins: list[tuple[int, int, int]],
        ds: int,
        por_default: float,
        local_por_map: dict | None,
        material_map: np.ndarray,
        box_lo: tuple[int, int, int],
        box_hi: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        """Stack every window's scalar and spatial conditioning onto the device.

        ``por_default`` and ``local_por_map`` carry MATERIAL porosity — pore
        over specimen, the number eval_v4 measures and the number a user asks
        for.  ``cond_por`` is not that: the store conditions on
        ``phi = pore / patch_size**3``, the FULL patch with any air outside the
        specimen counted in the denominator.  The two agree only where a window
        is entirely inside the specimen, so each window's request is rescaled by
        its own material fraction before it is clipped and transformed.
        """
        L, P = self.latent_size, self.patch_size
        por, depth, dist6, orient, material = [], [], [], [], []
        for g in g_origins:
            ov = tuple(int(c) * ds for c in g)          # voxel origin
            block = material_map[g[0]:g[0] + L, g[1]:g[1] + L, g[2]:g[2] + L]
            phi = por_default
            if local_por_map is not None:
                # The requested porosity field is defined on the TILE grid but a
                # window steps by window_stride, so it straddles up to eight
                # tiles: its request is the field over its own footprint.
                phi = window_tile_mean(ov, P, local_por_map, por_default)
            # Material porosity -> full-patch phi.  Every latent cell covers the
            # same ds**3 voxels, so the mean envelope fraction over the window's
            # cells IS the material fraction of its voxel footprint.  This has
            # to happen before the clip (0.2 material porosity at half material
            # is a legal 0.1, not a clipped 0.107) and before porosity_to_cond,
            # which is a log — scaling after it would be an offset, not a scale.
            phi *= float(block.mean())
            phi = float(np.clip(phi, POR_MIN, POR_MAX))
            por.append(float(porosity_to_cond(phi, self.por_log_stats)))
            d, d6 = self._window_position(ov, box_lo, box_hi)
            depth.append(d)
            dist6.append(d6)
            orient.append(self._window_orient(ov[0]))
            material.append(torch.from_numpy(block.copy()).unsqueeze(0))
        return {
            "por":   torch.tensor(por,   dtype=torch.float32, device=self.device),
            "depth": torch.tensor(depth, dtype=torch.float32, device=self.device),
            "dist6": torch.from_numpy(np.stack(dist6)).to(self.device),
            "orient": torch.stack(orient).to(self.device),
            "material": torch.stack(material).to(self.device),
        }

    # ── overlapped decode ────────────────────────────────────────────────────

    @torch.no_grad()
    def _decode_canvas(
        self,
        z_clean: torch.Tensor,
        volume_shape: tuple[int, int, int],
        autocast_dtype: torch.dtype,
        decode_batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend-decode the latent canvas.  Returns (xct [0,1], class logits).

        Decode windows are one tile wide and step ``decode_stride`` voxels, and
        the decoded grey level and the raw 3-class logits are accumulated with
        a tapered window before any nonlinearity — the same construction
        ``poregen.eval.blended`` validated on real volumes.  Blending the
        LOGITS (not the argmax, not the probabilities) is what removes the
        class seam a stride-64 tiling leaves at every patch face.
        """
        for attr in ("decoder", "xct_head", "class_head"):
            if not hasattr(self.vae, attr):
                raise TypeError(
                    f"VolumeGenerator needs a 3-class VAE with .{attr}; got "
                    f"{type(self.vae).__name__}.  ldm06 decodes material/pore/air "
                    f"logits, not a binary mask."
                )
        ds, L, P = self.downsample, self.latent_size, self.patch_size
        canvas_cells = tuple(z_clean.shape[1:])
        s_cells = self.decode_stride // ds
        origins = window_origins(canvas_cells, L, s_cells)

        w3d = tukey_window_3d(P, floor=_DECODE_WINDOW_FLOOR)
        xct_acc   = np.zeros(volume_shape, dtype=np.float32)
        logit_acc = np.zeros((3, *volume_shape), dtype=np.float32)
        w_acc     = np.zeros(volume_shape, dtype=np.float32)

        mean = self.latent_mean
        std  = self.latent_std
        if isinstance(mean, torch.Tensor):
            mean = mean.to(self.device)
        if isinstance(std, torch.Tensor):
            std = std.to(self.device)

        self.vae.eval()
        for start in range(0, len(origins), decode_batch_size):
            batch = origins[start : start + decode_batch_size]
            z = torch.stack([
                z_clean[:, o[0]:o[0] + L, o[1]:o[1] + L, o[2]:o[2] + L] for o in batch
            ]) * std + mean
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                dec = self.vae.decoder(z)
                xct_out = self.vae.xct_head(dec)
                class_logits = self.vae.class_head(dec)
            xct_np = xct_out.float().squeeze(1).cpu().numpy()
            cls_np = class_logits.float().cpu().numpy()
            for i, o in enumerate(batch):
                ov = tuple(int(c) * ds for c in o)
                sl = np.s_[ov[0]:ov[0] + P, ov[1]:ov[1] + P, ov[2]:ov[2] + P]
                xct_acc[sl] += w3d * xct_np[i]
                logit_acc[(slice(None), *sl)] += w3d[None] * cls_np[i]
                w_acc[sl] += w3d

        xct_acc /= w_acc
        logit_acc /= w_acc[None]
        return xct_acc, logit_acc

    # ── public entry point ───────────────────────────────────────────────────

    def generate(
        self,
        volume_size_mm: tuple[float, float, float],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        material_map: np.ndarray | None = None,
        specimen_box: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
        request_offset: tuple[int, int, int] = (0, 0, 0),
        progress=None,
        window_batch: int = 32,
        decode_batch_size: int = 64,
        return_class_probs: bool = False,
        return_latents: bool = False,
        seed: int | None = None,
    ) -> tuple:
        """Generate one volume: chunked joint denoising, then blended decode.

        Parameters
        ----------
        volume_size_mm    : (D, H, W) physical size in millimetres; each axis is
                            snapped DOWN to a whole number of 64-voxel tiles
        target_porosity   : uniform per-tile VVF fallback (None → 0.05)
        local_por_map     : dict (iz, iy, ix) → requested φ, on the TILE grid
        material_map      : (Z, Y, X) specimen-envelope fraction per LATENT CELL
                            over the whole volume.  None → 1 inside the
                            specimen box, 0 outside.  Paint it to ask for
                            exterior air, drilled holes or a non-box specimen
                            shape.  It does NOT say where the pores go.
        specimen_box      : ((z, y, x) lo inclusive, (z, y, x) hi exclusive) in
                            voxels.  None → the whole generated volume, i.e.
                            "this volume IS the specimen".  Drives cond_depth
                            and cond_dist6.
        request_offset    : (z, y, x) voxels the REQUEST is translated by inside
                            the canvas.  It does not move any conditioning —
                            the caller has already placed the specimen box, the
                            material map, the porosity field and θ(z) where it
                            wants them — it names the frame every noise draw is
                            taken in, so translating a request translates its
                            noise with it (:func:`region_noise_field`).  Each
                            component must be a whole number of latent cells.
        progress          : optional tqdm; counts DDIM steps (chunks × steps)
        window_batch      : windows per UNet forward per timestep
        decode_batch_size : latent windows decoded in one VAE forward
        return_latents     : also return the finished latent canvas, the
                             (C, Z, Y, X) float32 array the decoder consumed.
                             Kept for the decoder fine-tune: comparing two
                             decoders is only meaningful on the SAME latents,
                             and regenerating them from a seed re-runs the
                             whole sampler to get an array the first run
                             already had.
        return_class_probs: also return the blended per-voxel class
                            probabilities, (3, D, H, W) float32
        seed              : makes the generation reproducible.  Every random
                            draw in the reverse process — the canvas the chunks
                            start from and the fresh noise that re-noises the
                            finished chunks at every timestep — is taken from a
                            LOCAL ``torch.Generator`` on this generator's own
                            device.  Two calls with the same seed and the same
                            request are therefore bit-identical, and the global
                            torch generator is left exactly as it was found, so
                            a caller's own RNG stream is not silently consumed
                            or reset by a generation.  ``None`` draws from the
                            global generator, and the volume is not reproducible.

        Returns
        -------
        ``(xct_u8, label_u8, stats)``, plus ``class_probs`` when requested.
        ``label_u8`` is the argmax of the blended class logits: 0 material,
        1 pore, 2 air.  ``stats`` carries the conditioning target, the
        assembled label's own porosity and air fraction, and the seam
        discontinuity at BOTH the window (64-voxel) and chunk periods.
        """
        volume_shape = self._volume_shape(volume_size_mm)
        P = self.patch_size
        if any(int(o) % self.downsample for o in request_offset):
            raise ValueError(
                f"request_offset={tuple(request_offset)} must be a whole number of "
                f"latent cells, i.e. a multiple of the VAE downsampling factor "
                f"{self.downsample}: the noise frame lives on the latent grid."
            )
        offset_cells = tuple(int(o) // self.downsample for o in request_offset)
        if specimen_box is None:
            specimen_box = ((0, 0, 0), volume_shape)
        box_lo, box_hi = tuple(specimen_box[0]), tuple(specimen_box[1])

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        z_clean = self._generate_latents(
            volume_shape,
            target_porosity=target_porosity,
            local_por_map=local_por_map,
            material_map=material_map,
            specimen_box=(box_lo, box_hi),
            autocast_dtype=autocast_dtype,
            window_batch=window_batch,
            offset_cells=offset_cells,
            progress=progress,
            generator=generator,
        )
        xct, class_logits = self._decode_canvas(
            z_clean, volume_shape, autocast_dtype, decode_batch_size
        )

        logits_t = torch.from_numpy(class_logits).unsqueeze(0)
        label = decode_label(logits_t)[0].numpy().astype(np.uint8)
        # numpy mirror of models.vae.base.decode_xct_u8: the XCT head is not a
        # logit, so the conversion is clamp-and-scale.  A sigmoid here would
        # squash every volume into [0.5, 0.731] and destroy its contrast.
        xct_u8 = np.round(np.clip(xct, 0.0, 1.0) * 255.0).astype(np.uint8)

        # ── seam diagnostics ─────────────────────────────────────────────────
        # Measured on the decoder's own continuous output: the grey level, and
        # the pore logit log p_pore - log(1 - p_pore) derived from the blended
        # 3-class logits (a per-class logit on its own is not comparable across
        # voxels; the pore-vs-rest log odds is).
        probs = decode_class_probs(logits_t)[0].numpy()
        p_pore = np.clip(probs[1], 1e-6, 1.0 - 1e-6)
        pore_logit = np.log(p_pore) - np.log1p(-p_pore)
        chunk_period = tuple(P * c for c in self.chunk_tiles)

        seam_stats = {
            **seam_discontinuity(np.clip(xct, 0.0, 1.0), P, prefix="seam_xct"),
            **seam_discontinuity(pore_logit, P, prefix="seam_pore"),
            **seam_discontinuity(np.clip(xct, 0.0, 1.0), chunk_period,
                                 prefix="seam_chunk_xct", interior_exclude=P),
            **seam_discontinuity(pore_logit, chunk_period,
                                 prefix="seam_chunk_pore", interior_exclude=P),
        }
        logger.info(
            "Seam ratio (1.0 = indistinguishable from interior): window "
            "xct=%.3f pore=%.3f | chunk xct=%.3f pore=%.3f",
            seam_stats.get("seam_xct_ratio", float("nan")),
            seam_stats.get("seam_pore_ratio", float("nan")),
            seam_stats.get("seam_chunk_xct_ratio", float("nan")),
            seam_stats.get("seam_chunk_pore_ratio", float("nan")),
        )

        actual_por = float((label == 1).mean())
        actual_air = float((label == 2).mean())
        clamped_por = (
            float(np.clip(target_porosity, POR_MIN, POR_MAX))
            if target_porosity is not None else 0.05
        )
        logger.info(
            "Assembled volume: porosity=%.4f  air=%.4f  target=%s  conditioned=%.4f",
            actual_por, actual_air,
            "None" if target_porosity is None else f"{target_porosity:.4f}",
            clamped_por,
        )
        stats: dict[str, Any] = {
            "volume_shape": list(volume_shape),
            "chunk_tiles": list(self.chunk_tiles),
            "neighbour_mode": self.neighbour_mode,
            "window_stride": self.window_stride,
            "decode_stride": self.decode_stride,
            "ddim_steps": len(self.sampler.timesteps) - 1,
            "s_por": float(self.sampler.s_por),
            "s_nb": float(self.sampler.s_nb),
            "cfg_rescale": float(self.sampler.cfg_rescale),
            "objective": str(self.sampler.schedule.objective),
            # None says plainly that this volume cannot be reproduced, which is
            # a property of the result and belongs beside it.
            "seed": None if seed is None else int(seed),
            "target_porosity": None if target_porosity is None else float(target_porosity),
            "conditioned_porosity": clamped_por,
            "actual_label_porosity": actual_por,
            "actual_label_air": actual_air,
            **seam_stats,
        }
        extra: tuple = ()
        if return_class_probs:
            extra = extra + (probs.astype(np.float32),)
        if return_latents:
            # .detach().cpu() first. z_clean is a CUDA tensor and np.asarray on
            # one raises TypeError — which is exactly what broke all nine
            # eval-v4 generate stages the first time --save-latents ran.
            z_np = z_clean.detach().cpu().numpy() if torch.is_tensor(z_clean) \
                else np.asarray(z_clean)
            extra = extra + (z_np.astype(np.float32),)
        return (xct_u8, label, stats) + extra

    @staticmethod
    def save_tiff(
        xct: np.ndarray,
        label: np.ndarray,
        path_xct: str | Path,
        path_label: str | Path,
    ) -> None:
        """Write the XCT and label volumes to TIFF files.

        Both go out on their NATIVE scale: ``xct`` is uint8 on the raw-scan
        grey scale, ``label`` is uint8 {0 material, 1 pore, 2 air}.  Anything
        that rescales them (a /255, an expit) makes generated and real volumes
        incomparable, which cost a whole evaluation campaign.
        """
        import tifffile
        path_xct  = Path(path_xct)
        path_label = Path(path_label)
        path_xct.parent.mkdir(parents=True, exist_ok=True)
        path_label.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path_xct),  xct)
        tifffile.imwrite(str(path_label), label)
        logger.info("Saved XCT   → %s", path_xct)
        logger.info("Saved label → %s", path_label)

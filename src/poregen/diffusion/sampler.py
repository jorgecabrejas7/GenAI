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

Decoding is overlapped too: the finished latent canvas is decoded in windows at
``decode_stride`` voxels and the decoded grey levels and class logits are
blended with a tapered window (the fix validated on the VAE tile seams —
see ``poregen.eval.blended``).  Direct stride-64 tiling put a decoder-side
seam at every patch face; blending removes it.
"""

from __future__ import annotations

import logging
import math
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
from poregen.diffusion.orientation import orientation_tensor
from poregen.eval.blended import tukey_window_3d
from poregen.models.vae.base import decode_class_probs, decode_label

logger = logging.getLogger(__name__)

_AXIS_NAMES = ("z", "y", "x")

# The decode blend window must never be exactly zero: a volume's own outer face
# is covered by a single decode window, and a zero weight there would leave the
# face undefined (0/0).  The floor lifts the Tukey taper off zero without
# changing the interior blend to three decimal places.
_DECODE_WINDOW_FLOOR = 1e-3

__all__ = [
    "DDIMSampler",
    "VolumeGenerator",
    "porosity_to_cond",
    "theta_from_layup",
    "window_origins",
    "window_weight",
    "seam_discontinuity",
]


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


def window_weight(win_cells: int) -> torch.Tensor:
    """(L, L, L) separable cosine (Hann-type) fusion weight, strictly positive.

    ``w1d[i] = sin²(π·(i + 0.5)/L)`` — the half-cell offset keeps every entry
    positive, so the per-voxel weight normalisation is well defined even where
    a single window covers a canvas corner.  Cosine rather than uniform
    weighting: a uniform average makes the effective per-voxel weight field
    piecewise constant with jumps exactly at window borders (where the model
    has the least receptive-field context), re-introducing a grid of weak
    seams into the fused ε field.  The cosine profile down-weights each
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

    for axis, name in enumerate(_AXIS_NAMES):
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

    When both scales are 1.0 the standard single-pass full-conditional denoiser
    is used.  Any other combination activates the 3-pass nested CFG
    decomposition::

        eps_uncond = model(z_t, t, nb, ALL_UNK, nb_t=0, por, …, drop_por=True)
        eps_por    = model(z_t, t, nb, ALL_UNK, nb_t=0, por, …, drop_por=False)
        eps_full   = model(z_t, t, nb, REAL,    nb_t,    por, …, drop_por=False)
        eps = eps_uncond + s_por*(eps_por - eps_uncond) + s_nb*(eps_full - eps_por)

    At s_por=s_nb=1 this telescopes to eps_full — exact un-guided equality.
    The ALL_UNKNOWN arms use ``nb_t = 0`` because that is exactly the neighbour
    null the training step draws (``drop_nb``).  Position, orientation and
    material are always on in all three passes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        schedule: Any,
        device: torch.device,
        n_steps: int = 50,
        s_por: float = 1.0,
        s_nb: float = 1.0,
    ) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device
        self.n_steps  = n_steps
        self.s_por    = s_por
        self.s_nb     = s_nb
        self.guided   = not (s_por == 1.0 and s_nb == 1.0)
        T = schedule.T
        ts = torch.linspace(0, T - 1, n_steps + 1, dtype=torch.long)
        # Store as Python ints for torch.compile compatibility (no dynamic shapes)
        self.timesteps: list[int] = ts.flip(0).tolist()   # [T-1, ..., 0]

    def predict_eps(
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
        """One ε prediction for a batch, honouring the CFG guidance scales.

        This is the single choke point every sampling loop goes through — the
        chunked joint path calls it directly to get raw per-window predictions
        before fusing them on the chunk canvas.
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
            eps_uncond = self.model(x, t, nb_latents, all_unk, zero_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, drop_all)
            eps_por    = self.model(x, t, nb_latents, all_unk, zero_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, None)
            eps_full   = self.model(x, t, nb_latents, nb_avail, nb_t, cond_por,
                                    cond_depth, cond_dist6, cond_orient,
                                    cond_material, None)

        return (
            eps_uncond
            + self.s_por * (eps_por  - eps_uncond)
            + self.s_nb  * (eps_full - eps_por)
        )

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
    ) -> Any:
        """Run the DDIM reverse process for a batch of independent patches.

        Neighbour conditioning is held FIXED across the reverse process here —
        this is the patch-level sampler used by the in-training diagnostics,
        not the volume path.  ``nb_t`` therefore describes the noise level of
        the neighbours as handed in, and stays constant.

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

        x = torch.randn(B, C, D, D, D, device=self.device)
        sat_sum = torch.zeros((), device=self.device)
        n_steps = 0
        inter: list[torch.Tensor] = []
        for i, t_val in enumerate(self.timesteps[:-1]):
            t_prev_val = self.timesteps[i + 1]
            t      = torch.full((B,), t_val,      dtype=torch.long, device=self.device)
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=self.device)
            eps_pred = self.predict_eps(
                x, t, nb_latents, nb_avail, nb_t, cond_por, cond_depth,
                cond_dist6, cond_orient, cond_material, autocast_dtype,
            )
            if return_x0_saturation:
                x0_pred = schedule.predict_x0(x, t, eps_pred)
                sat_sum += (x0_pred.abs() >= 10.0).float().mean()
                n_steps += 1
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
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
        progress=None,
    ) -> torch.Tensor:
        """Denoise the whole latent canvas chunk by chunk.  Returns (C, Z, Y, X)."""
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

        chunk_grid = [
            self._chunk_ranges(n_tiles[a], self.chunk_tiles[a]) for a in range(3)
        ]
        chunks = [(cz, cy, cx) for cz in chunk_grid[0]
                  for cy in chunk_grid[1] for cx in chunk_grid[2]]
        timesteps = sampler.timesteps

        logger.info(
            "VolumeGenerator: %s voxels = %s tiles, %d chunk(s) of %s tiles, "
            "windows every %d voxels, %d DDIM steps",
            volume_shape, n_tiles, len(chunks), self.chunk_tiles,
            self.window_stride, len(timesteps) - 1,
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

            # Availability as the windows of THIS chunk see it: finished chunks
            # plus the chunk being generated.
            visible = available.copy()
            visible[chunk_sl] = True
            done_mask = torch.from_numpy(available[ctx_sl].astype(np.float32)).to(self.device)

            origins = window_origins(chunk_cells, L, s_cells)
            g_origins = [tuple(lo[a] + o[a] for a in range(3)) for o in origins]
            n_win = len(g_origins)

            states, nb_slices = self._neighbour_plan(g_origins, visible, canvas_cells, ctx_lo)
            cond = self._window_conditioning(
                g_origins, ds, por_default, local_por_map, material_map,
                box_lo, box_hi, n_tiles,
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

            x = torch.randn(1, C, *chunk_cells, device=self.device)
            B_max = max(1, min(int(window_batch), n_win))

            for i, t_val in enumerate(timesteps[:-1]):
                t_prev_val = timesteps[i + 1]
                # Context canvas at this timestep: finished chunks re-noised to
                # t with FRESH noise, the current chunk at its live state.
                t_one = torch.full((1,), t_val, dtype=torch.long, device=self.device)
                ctx = schedule.q_sample(
                    z_clean[(slice(None), *ctx_sl)].unsqueeze(0), t_one
                ) * done_mask
                ctx[(0, slice(None), *cur_sl)] = x[0]

                eps_sum = torch.zeros_like(x)
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
                    eps = sampler.predict_eps(
                        xw, t_b, nb, av, nb_t,
                        cond["por"][idx], cond["depth"][idx], cond["dist6"][idx],
                        cond["orient"][idx], cond["material"][idx], autocast_dtype,
                    ).float()
                    for k, j in enumerate(idx):
                        sl = win_sl[j]
                        eps_sum[0, :, sl[0], sl[1], sl[2]] += weight * eps[k]

                t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
                t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
                x = schedule.ddim_step(x, t, t_prev, eps_sum / weight_sum)
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
        """Inside the specimen box the envelope is 1; outside it is 0 (air)."""
        ds = self.downsample
        m = np.zeros(canvas_cells, dtype=np.float32)
        sl = tuple(
            slice(int(math.ceil(box_lo[a] / ds)), int(box_hi[a] // ds))
            for a in range(3)
        )
        m[sl] = 1.0
        return m

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
        n_tiles: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        """Stack every window's scalar and spatial conditioning onto the device."""
        L, P = self.latent_size, self.patch_size
        por, depth, dist6, orient, material = [], [], [], [], []
        for g in g_origins:
            ov = tuple(int(c) * ds for c in g)          # voxel origin
            phi = por_default
            if local_por_map is not None:
                # The requested porosity field is defined on the TILE grid;
                # a window takes the tile that holds its centre.
                ti = tuple(min((ov[a] + P // 2) // P, n_tiles[a] - 1) for a in range(3))
                phi = local_por_map.get(ti, por_default)
            phi = float(np.clip(phi, POR_MIN, POR_MAX))
            por.append(float(porosity_to_cond(phi, self.por_log_stats)))
            d, d6 = self._window_position(ov, box_lo, box_hi)
            depth.append(d)
            dist6.append(d6)
            orient.append(self._window_orient(ov[0]))
            material.append(
                torch.from_numpy(
                    material_map[g[0]:g[0] + L, g[1]:g[1] + L, g[2]:g[2] + L].copy()
                ).unsqueeze(0)
            )
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
        progress=None,
        window_batch: int = 32,
        decode_batch_size: int = 64,
        return_class_probs: bool = False,
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
        progress          : optional tqdm; counts DDIM steps (chunks × steps)
        window_batch      : windows per UNet forward per timestep
        decode_batch_size : latent windows decoded in one VAE forward
        return_class_probs: also return the blended per-voxel class
                            probabilities, (3, D, H, W) float32

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
        if specimen_box is None:
            specimen_box = ((0, 0, 0), volume_shape)
        box_lo, box_hi = tuple(specimen_box[0]), tuple(specimen_box[1])

        z_clean = self._generate_latents(
            volume_shape,
            target_porosity=target_porosity,
            local_por_map=local_por_map,
            material_map=material_map,
            specimen_box=(box_lo, box_hi),
            autocast_dtype=autocast_dtype,
            window_batch=window_batch,
            progress=progress,
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
            "window_stride": self.window_stride,
            "decode_stride": self.decode_stride,
            "ddim_steps": len(self.sampler.timesteps) - 1,
            "s_por": float(self.sampler.s_por),
            "s_nb": float(self.sampler.s_nb),
            "target_porosity": None if target_porosity is None else float(target_porosity),
            "conditioned_porosity": clamped_por,
            "actual_label_porosity": actual_por,
            "actual_label_air": actual_air,
            **seam_stats,
        }
        if return_class_probs:
            return xct_u8, label, stats, probs.astype(np.float32)
        return xct_u8, label, stats

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

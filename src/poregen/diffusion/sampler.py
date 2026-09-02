"""DDPM patch sampler and full-volume generator for PoreGen LDM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    PARITY_GROUP_ORDER,
    latent_shift_cells,
    neighbour_states,
    parity_group,
    resolve_group_order,
    shift_into_target_frame,
    validate_group_order,
    validate_neighbour_geometry,
    validate_shift,
)
from poregen.diffusion.orientation import orientation_tensor

logger = logging.getLogger(__name__)

_NEIGHBOR_DIRS = list(NEIGHBOUR_DIRS)

# Availability states — single source of truth is conditioning.py
_NB_OOB     = NB_OOB
_NB_EXISTS  = NB_EXISTS
_NB_UNKNOWN = NB_UNKNOWN

# Porosity conditioning is clamped to the training distribution
# range (EDA ground truth: min 0.002, max 0.107) to avoid OOD extrapolation.
_POR_MIN = 0.002
_POR_MAX = 0.107

# Porosity transform (D32 §1): cond_por = (log(phi + eps) - mean) / std
# Both constants must match scripts/build_conditioning.py.
_POR_LOG_EPS = 1e-3
_DIST_CAP = 64.0


def porosity_to_cond(phi, por_log_stats: tuple[float, float] | None):
    """Map raw pore volume fraction to the model's ``cond_por`` scalar.

    ``por_log_stats`` is the (mean, std) of ``log(phi + 1e-3)`` over the train
    split, as recorded by the latent store.  Passing None is an error — a
    wrong standardisation silently mis-conditions every patch.
    """
    if por_log_stats is None:
        raise ValueError(
            "por_log_stats is required to build cond_por. Pass the (mean, std) of "
            "log(phi + 1e-3) recorded by the latent store's metadata."
        )
    mean, std = por_log_stats
    return (np.log(np.asarray(phi, dtype=np.float64) + _POR_LOG_EPS) - mean) / std


def theta_from_layup(
    depth_vox: int,
    ply_angles_deg,
    ply_thickness_vox: float,
) -> np.ndarray:
    """θ(z) in degrees for a requested layup — the generation-side field.

    D32 §4: at generation time θ(z) is written directly from the requested
    stacking sequence and ply thickness; nothing is estimated.  Each ply
    occupies ``ply_thickness_vox`` voxels and the sequence repeats if the
    volume is deeper than the stack.  The encoding into (cos2θ, sin2θ) is done
    by :func:`poregen.diffusion.orientation.orientation_tensor`, the single
    shared implementation used by training as well.

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


_AXIS_NAMES = ("z", "y", "x")


# ── joint (MultiDiffusion-style) window helpers ──────────────────────────────

def joint_window_origins(
    canvas_cells: tuple[int, int, int],
    window_cells: int,
    stride_cells: int,
) -> list[tuple[int, int, int]]:
    """Origins (in latent cells) of the overlapping joint-denoising windows.

    Windows of side ``window_cells`` are placed at ``stride_cells`` spacing on
    each axis.  Every canvas cell must be covered, so each axis must hold a
    whole number of strides: ``(n - window) % stride == 0`` with the last
    window ending exactly at the canvas edge.
    """
    w, s = int(window_cells), int(stride_cells)
    if s <= 0 or s > w:
        raise ValueError(f"stride_cells={s} must be in [1, window_cells={w}].")
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


def joint_window_weight(window_cells: int) -> torch.Tensor:
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
    L = int(window_cells)
    i = torch.arange(L, dtype=torch.float32) + 0.5
    w1 = torch.sin(np.pi * i / L) ** 2
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def seam_discontinuity(
    volume: np.ndarray,
    patch_size: int,
    prefix: str = "seam",
) -> dict[str, float]:
    """Discontinuity across the shared face of adjacent generated patches.

    This is the primary assembly-quality metric.  Patches tile at
    ``generation_stride == patch_size``, so there is no overlap left to
    compare: the only place two independently-denoised patches meet is the
    plane between them.  A visible seam shows up as an abnormally large jump
    from the last slice of one patch to the first slice of its neighbour.

    For each axis the mean absolute slice-to-slice difference is computed at
    the SEAM planes (boundary index a multiple of ``patch_size``) and at every
    other, INTERIOR plane.  The interior value is the natural slice-to-slice
    variation of the material and is the baseline the seam is judged against::

        ratio = seam_mad / interior_mad

    ``ratio ≈ 1`` means the seam is indistinguishable from ordinary internal
    texture change — the assembly is continuous.  ``ratio >> 1`` means the
    patches disagree where they meet and the seam is visible.

    Parameters
    ----------
    volume     : (D, H, W) decoded volume, any continuous scale
    patch_size : voxel side length of one patch — the seam spacing
    prefix     : metric-name prefix (e.g. ``"seam_xct"``)

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
    P = int(patch_size)

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
        is_seam = (boundary % P) == 0

        n_seam = int(is_seam.sum())
        total_planes += n_seam
        metrics[f"{prefix}_{name}_planes"] = float(n_seam)

        seam_mad = float(per_plane[is_seam].mean()) if n_seam else float("nan")
        n_int = int((~is_seam).sum())
        int_mad = float(per_plane[~is_seam].mean()) if n_int else float("nan")
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


class DDPMSampler:
    """Patch-level DDPM reverse diffusion sampler.

    Parameters
    ----------
    model : UNet3DDenoiser
    schedule : DDPMSchedule
    device : torch.device
    """

    def __init__(self, model: torch.nn.Module, schedule: Any, device: torch.device) -> None:
        self.model    = model
        self.schedule = schedule
        self.device   = device

    @torch.no_grad()
    def sample_patch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: float,
        cond_depth: float,
        cond_dist: float,
        cond_orient: torch.Tensor | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run the full T-step DDPM reverse process for one patch.

        Parameters
        ----------
        nb_latents          : (6, C, D, H, W) — neighbour latents, already shifted
                              into the target frame (zeros for unavailable)
        nb_avail            : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        cond_por            : float — standardised log porosity
        cond_depth          : float — relative depth in [0, 1]
        cond_dist           : float — min(d, 64)/64 in [0, 1]
        cond_orient         : (2, D, H, W) float or None
        return_intermediates: if True, return (final, list[Tensor]) where the list contains
                              the latent after each denoising step, on CPU float32

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent, or
        ((C, D, H, W), list[(C, D, H, W)]) when return_intermediates=True
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        C = nb_latents.shape[1]
        D = nb_latents.shape[2]

        # Add batch dim
        nb_l   = nb_latents.unsqueeze(0).to(self.device)     # (1,6,C,D,D,D)
        nb_a   = nb_avail.unsqueeze(0).to(self.device)       # (1,6)
        por    = torch.tensor([cond_por],   dtype=torch.float32, device=self.device)
        depth  = torch.tensor([cond_depth], dtype=torch.float32, device=self.device)
        dist   = torch.tensor([cond_dist],  dtype=torch.float32, device=self.device)
        orient = None if cond_orient is None else cond_orient.unsqueeze(0).to(self.device)

        x = torch.randn(1, C, D, D, D, device=self.device)
        intermediates: list[torch.Tensor] = [] if return_intermediates else None  # type: ignore[assignment]

        for t_idx in reversed(range(schedule.T)):
            t = torch.tensor([t_idx], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_l, nb_a, por, depth, dist, orient)
            x = schedule.p_sample(x, t, eps_pred)
            if return_intermediates:
                intermediates.append(x.squeeze(0).float().cpu())

        result = x.squeeze(0).float()
        if return_intermediates:
            return result, intermediates
        return result

    @torch.no_grad()
    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist: torch.Tensor,
        cond_orient: torch.Tensor | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Run the full T-step DDPM reverse process for a batch of patches.

        Parameters
        ----------
        nb_latents  : (B, 6, C, D, D, D) — shifted neighbour latents, on device
        nb_avail    : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN, on device
        cond_por    : (B,) float — standardised log porosity, on device
        cond_depth  : (B,) float — relative depth, on device
        cond_dist   : (B,) float — distance to surface, on device
        cond_orient : (B, 2, D, D, D) float or None — orientation profile

        Returns
        -------
        (B, C, D, D, D) float32 on device
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        B = nb_latents.shape[0]
        C = nb_latents.shape[2]
        D = nb_latents.shape[3]

        x = torch.randn(B, C, D, D, D, device=self.device)
        for t_idx in reversed(range(schedule.T)):
            t = torch.full((B,), t_idx, dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                eps_pred = self.model(x, t, nb_latents, nb_avail,
                                      cond_por, cond_depth, cond_dist, cond_orient)
            x = schedule.p_sample(x, t, eps_pred)
        return x.float()


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

    When both scales are 1.0 the standard single-pass full-conditional denoiser is
    used.  Any other combination activates the 3-pass nested CFG decomposition:

        eps_uncond = model(z_t, t, nb, ALL_UNK, por, ..., drop_por=True)
        eps_por    = model(z_t, t, nb, ALL_UNK, por, ..., drop_por=False)
        eps_full   = model(z_t, t, nb, REAL,    por, ..., drop_por=False)
        eps = eps_uncond + s_por*(eps_por - eps_uncond) + s_nb*(eps_full - eps_por)

    At s_por=s_nb=1 this telescopes to eps_full — exact un-guided equality.
    Position and orientation are always on (never dropped) in all three passes.
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
        self._timesteps: list[int] = ts.flip(0).tolist()   # [T-1, ..., 0]

    def _guided_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist: torch.Tensor,
        cond_orient: torch.Tensor | None,
        autocast_dtype: torch.dtype,
    ) -> torch.Tensor:
        """3-pass nested CFG decomposition.

        Position and orientation are on in all three calls.  The ALL_UNKNOWN
        passes rely on UNet3DDenoiser._build_nb_spatial masking latents by
        NB_EXISTS, so they are structurally independent of nb_latents values.
        """
        B = x.shape[0]
        all_unk  = torch.full_like(nb_avail, _NB_UNKNOWN)
        drop_all = torch.ones(B, dtype=torch.bool, device=self.device)

        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            eps_uncond = self.model(x, t, nb_latents, all_unk,  cond_por, cond_depth,
                                    cond_dist, cond_orient, drop_all)
            eps_por    = self.model(x, t, nb_latents, all_unk,  cond_por, cond_depth,
                                    cond_dist, cond_orient, None)
            eps_full   = self.model(x, t, nb_latents, nb_avail, cond_por, cond_depth,
                                    cond_dist, cond_orient, None)

        return (
            eps_uncond
            + self.s_por * (eps_por  - eps_uncond)
            + self.s_nb  * (eps_full - eps_por)
        )

    def predict_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist: torch.Tensor,
        cond_orient: torch.Tensor | None,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """One ε prediction for a batch, honouring the CFG guidance scales.

        This is the single choke point every DDIM sampling loop goes through —
        the joint (MultiDiffusion) volume path calls it directly to get raw
        per-window predictions before fusing them on the canvas.
        """
        if self.guided:
            return self._guided_eps(x, t, nb_latents, nb_avail, cond_por,
                                    cond_depth, cond_dist, cond_orient,
                                    autocast_dtype)
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            return self.model(x, t, nb_latents, nb_avail, cond_por,
                              cond_depth, cond_dist, cond_orient)

    @torch.no_grad()
    def sample_patch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: float,
        cond_depth: float,
        cond_dist: float,
        cond_orient: torch.Tensor | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run DDIM reverse process for one patch.

        Parameters
        ----------
        nb_latents          : (6, C, D, H, W) — shifted neighbour latents
        nb_avail            : (6,) long — 0=OOB, 1=EXISTS, 2=UNKNOWN per neighbor
        cond_por            : float — standardised log porosity
        cond_depth          : float — relative depth in [0, 1]
        cond_dist           : float — min(d, 64)/64 in [0, 1]
        cond_orient         : (2, D, H, W) float or None
        return_intermediates: if True, return (final, list[Tensor]) where the list contains
                              the latent after each denoising step, on CPU float32

        Returns
        -------
        (C, D, H, W) float32 — generated clean latent, or
        ((C, D, H, W), list[(C, D, H, W)]) when return_intermediates=True
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        C = nb_latents.shape[1]
        D = nb_latents.shape[2]

        nb_l   = nb_latents.unsqueeze(0).to(self.device)
        nb_a   = nb_avail.unsqueeze(0).to(self.device)
        por    = torch.tensor([cond_por],   dtype=torch.float32, device=self.device)
        depth  = torch.tensor([cond_depth], dtype=torch.float32, device=self.device)
        dist   = torch.tensor([cond_dist],  dtype=torch.float32, device=self.device)
        orient = None if cond_orient is None else cond_orient.unsqueeze(0).to(self.device)

        x = torch.randn(1, C, D, D, D, device=self.device)
        intermediates: list[torch.Tensor] = [] if return_intermediates else None  # type: ignore[assignment]

        for i, t_val in enumerate(self._timesteps[:-1]):
            t_prev_val = self._timesteps[i + 1]
            t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
            t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
            eps_pred = self.predict_eps(x, t, nb_l, nb_a, por, depth, dist, orient,
                                        autocast_dtype)
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
            if return_intermediates:
                intermediates.append(x.squeeze(0).float().cpu())

        result = x.squeeze(0).float()
        if return_intermediates:
            return result, intermediates
        return result

    @torch.no_grad()
    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist: torch.Tensor,
        cond_orient: torch.Tensor | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_x0_saturation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        """Run DDIM reverse process for a batch of patches.

        Parameters
        ----------
        nb_latents  : (B, 6, C, D, D, D) — shifted neighbour latents, on device
        nb_avail    : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN, on device
        cond_por    : (B,) float — standardised log porosity, on device
        cond_depth  : (B,) float — relative depth, on device
        cond_dist   : (B,) float — distance to surface, on device
        cond_orient : (B, 2, D, D, D) float or None — orientation profile
        return_x0_saturation : if True, also return the fraction of x0-prediction
            elements hitting the ±10 clamp inside ddim_step, averaged over all
            denoising steps — an off-manifold diagnostic.

        Returns
        -------
        (B, C, D, D, D) float32 on device — or (samples, sat_frac) when
        return_x0_saturation=True.
        """
        self.model.eval()
        schedule = self.schedule.to(self.device)
        B = nb_latents.shape[0]
        C = nb_latents.shape[2]
        D = nb_latents.shape[3]

        x = torch.randn(B, C, D, D, D, device=self.device)
        sat_sum = torch.zeros((), device=self.device)
        n_steps = 0
        for i, t_val in enumerate(self._timesteps[:-1]):
            t_prev_val = self._timesteps[i + 1]
            t      = torch.full((B,), t_val,      dtype=torch.long, device=self.device)
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=self.device)
            eps_pred = self.predict_eps(x, t, nb_latents, nb_avail, cond_por,
                                        cond_depth, cond_dist, cond_orient,
                                        autocast_dtype)
            if return_x0_saturation:
                x0_pred = schedule.predict_x0(x, t, eps_pred)
                sat_sum += (x0_pred.abs() >= 10.0).float().mean()
                n_steps += 1
            x = schedule.ddim_step(x, t, t_prev, eps_pred)
        if return_x0_saturation:
            return x.float(), (sat_sum / max(n_steps, 1)).item()
        return x.float()


class VolumeGenerator:
    """Generate a full synthetic 3D volume using the eight-group parity schedule.

    Two denoising modes share this class (``generate(mode=...)``): the default
    SEQUENTIAL parity schedule described below, and a MultiDiffusion-style
    JOINT mode (:meth:`_generate_latents_joint`) in which overlapping windows
    denoise one shared latent canvas together.  Decode and assembly are
    identical in both modes.

    Patches TILE: ``generation_stride == neighbour_offset == patch_size``, so
    adjacent patches touch and share no voxel.  Nothing overlaps, therefore
    nothing is blended and no neighbour can hand the denoiser a copy of the
    target (see ``poregen.diffusion.conditioning`` for the leak this replaced).

    Patches are grouped by spatial parity ``(iz mod 2, iy mod 2, ix mod 2)``
    and generated group by group in the fixed order
    :data:`poregen.diffusion.conditioning.PARITY_GROUP_ORDER` (D32 §3.3).  A
    face neighbour flips exactly one parity bit, so its availability is
    deterministic: EXISTS when its group precedes the target's group, UNKNOWN
    when it follows, OOB when it is outside the grid.  Availability comes from
    the shared :func:`~poregen.diffusion.conditioning.neighbour_states`, which
    the training dataset calls too.  The eight-group ordering is kept in
    preference to a two-colour checkerboard because it leaves only 1 patch in 8
    without neighbour context instead of 1 in 2.

    Every EXISTS neighbour is handed to the denoiser WHOLE and unshifted: it is
    face-adjacent context, not an overlapping view of the target.

    Assembly is a direct write of each decoded patch into its own block of the
    output.  The only place two independently-denoised patches meet is the
    plane between them, so :func:`seam_discontinuity` is the assembly-quality
    metric.

    Parameters
    ----------
    sampler          : DDPMSampler or DDIMSampler
    vae              : VAE model with .decoder, .xct_head, .mask_head attributes
    device           : torch.device
    patch_size       : int — voxel side length of each patch (default 64)
    generation_stride: int — stride between patch origins on the assembly grid
                       (default 64).  D32 §3.1: this is the ASSEMBLY grid, not
                       the dataset sampling density (``sample_stride``, a data
                       side parameter) and not the spatial relation between a
                       patch and its conditioning neighbours (``neighbour_offset``).
                       Must equal ``patch_size`` so patches tile exactly.
    neighbour_offset : int — voxel displacement of a face neighbour, i.e. the
                       relation the model was TRAINED on (default 64).  Must
                       equal ``generation_stride``, otherwise the grid's face
                       neighbours are not the neighbours the model expects.
    neighbour_shift  : bool — roll neighbours into the target frame.  Only
                       meaningful for overlapping neighbours; raises for
                       touching ones instead of feeding all-zero tensors.
    latent_size      : int — spatial side length of the latent (default 16)
    latent_mean      : float | (C,1,1,1) tensor — per-channel normalisation mean;
                       generated latents are denormalised ``z*std + mean`` before decoding
    latent_std       : float | (C,1,1,1) tensor — per-channel normalisation std
    voxel_size_mm    : float — physical voxel size in millimetres (default 0.025 = 25 µm)
    por_log_stats    : (mean, std) of ``log(phi + 1e-3)`` on the train split, from
                       the latent store metadata.  Required when the model uses
                       porosity conditioning.
    theta_deg        : (vol_d,) float array — the requested θ(z) in degrees for
                       the whole volume (see :func:`theta_from_layup`), NaN where
                       unknown.  Required when the model uses orientation
                       conditioning.
    group_order      : parity group ordering; defaults to the shared
                       ``PARITY_GROUP_ORDER``.  Pass the latent store's metadata
                       through :func:`resolve_group_order` to guarantee that
                       generation replays the training schedule.
    """

    def __init__(
        self,
        sampler: Any,
        vae: torch.nn.Module,
        device: torch.device,
        patch_size: int = 64,
        generation_stride: int = 64,
        neighbour_offset: int = 64,
        neighbour_shift: bool = False,
        latent_size: int = 16,
        latent_mean: torch.Tensor | float = 0.0,
        latent_std: torch.Tensor | float = 1.0,
        voxel_size_mm: float = 0.025,
        por_log_stats: tuple[float, float] | None = None,
        theta_deg: np.ndarray | None = None,
        group_order: tuple[tuple[int, int, int], ...] | None = None,
    ) -> None:
        if neighbour_offset != generation_stride:
            raise ValueError(
                f"neighbour_offset={neighbour_offset} != generation_stride="
                f"{generation_stride}. The face neighbours of the assembly grid sit "
                "exactly one generation_stride away, so a different neighbour_offset "
                "would feed the denoiser a spatial relation it never saw in training."
            )
        if generation_stride != patch_size:
            raise ValueError(
                f"generation_stride={generation_stride} != patch_size={patch_size}. "
                "Generated patches must tile exactly: a smaller stride makes them "
                "overlap, which both leaks target content into the neighbour "
                "conditioning and needs blending that averages two independent "
                "answers; a larger stride would leave gaps."
            )
        validate_neighbour_geometry(neighbour_offset, patch_size)
        self.sampler           = sampler
        self.vae               = vae
        self.device            = device
        self.patch_size        = patch_size
        self.generation_stride = generation_stride
        self.neighbour_offset  = neighbour_offset
        self.neighbour_shift   = bool(neighbour_shift)
        self.latent_size       = latent_size
        self.latent_mean       = latent_mean
        self.latent_std        = latent_std
        self.voxel_size_mm     = voxel_size_mm
        self.por_log_stats     = por_log_stats
        self.theta_deg         = None if theta_deg is None else np.asarray(theta_deg)
        # "specimen": the generated volume is the whole specimen — true
        #             cond_dist, OOB at the generation-grid boundary.
        # "interior": a window into an unbounded specimen — cond_dist 1.0
        #             everywhere, no OOB (edges become UNKNOWN).
        # "legacy":   the original joint behaviour — true cond_dist but
        #             all-UNKNOWN neighbours (contradictory at edges; kept
        #             as an evaluation arm). Sequential ignores "legacy".
        self.conditioning_semantics = "specimen"
        self.group_order       = (
            PARITY_GROUP_ORDER if group_order is None else validate_group_order(group_order)
        )
        self.z_channels        = sampler.model.cfg.z_channels
        self.downsample        = patch_size // latent_size
        # Touching neighbours are fed whole; a shift here would be all zeros
        # and validate_shift makes that a hard error rather than a silent one.
        self.latent_shift      = 0
        if self.neighbour_shift:
            self.latent_shift = latent_shift_cells(
                neighbour_offset, patch_size, latent_size
            )
            validate_shift(self.latent_shift, latent_size)

    def _tile_grid(
        self, volume_shape: tuple[int, int, int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Per-axis TILING-grid origins (stride 64), with the coverage guard.

        This is the assembly/decode grid for BOTH modes — the joint mode only
        changes how the latents are denoised, never how they are decoded.
        """
        stride = self.generation_stride
        P      = self.patch_size
        vol_d, vol_h, vol_w = volume_shape
        zs = list(range(0, vol_d - P + 1, stride))
        ys = list(range(0, vol_h - P + 1, stride))
        xs = list(range(0, vol_w - P + 1, stride))
        if not zs or not ys or not xs:
            raise ValueError(
                f"Volume {volume_shape} too small for patch_size={P}, "
                f"generation_stride={stride}."
            )
        covered = (len(zs) * stride, len(ys) * stride, len(xs) * stride)
        if covered != tuple(volume_shape):
            raise ValueError(
                f"Patch grid covers {covered} of a {tuple(volume_shape)} volume. "
                f"With generation_stride={stride} each axis must be a whole number "
                f"of patches, otherwise the tiling leaves a gap."
            )
        return zs, ys, xs

    # ── per-patch conditioning construction (D32 §4) ─────────────────────────

    def _patch_orient(self, z0: int) -> torch.Tensor:
        """(2, L, L, L) orientation profile for the patch at depth origin z0.

        Encoding is delegated to
        :func:`poregen.diffusion.orientation.orientation_tensor` — the same
        function the dataset uses, so training and generation cannot drift.
        """
        if self.theta_deg is None:
            raise ValueError(
                "The denoiser uses orientation conditioning but no theta_deg was "
                "given. Build one with theta_from_layup(depth_vox, layup, "
                "ply_thickness_vox)."
            )
        P = self.patch_size
        if z0 + P > len(self.theta_deg):
            raise ValueError(
                f"theta_deg covers {len(self.theta_deg)} voxels — too few for a "
                f"patch at z0={z0} (needs {z0}..{z0 + P - 1})."
            )
        return torch.from_numpy(
            orientation_tensor(self.theta_deg[z0 : z0 + P], self.latent_size)
        )

    def _patch_position(
        self,
        origin: tuple[int, int, int],
        volume_shape: tuple[int, int, int],
    ) -> tuple[float, float]:
        """(cond_depth, cond_dist) for a patch (D32 §1, signals 2 and 3).

        The generated volume *is* the specimen, so its own bounds are the outer
        surfaces.  ``cond_depth`` is the patch centre's fractional depth in z;
        ``cond_dist`` is the distance from that centre to the nearest outer
        surface over ALL THREE axes, capped at 64 voxels and divided by 64 —
        the same definition ``scripts/build_conditioning.py`` uses at training
        time (there the extents come from the foreground threshold).
        """
        centres = [o + self.patch_size / 2.0 for o in origin]
        depth = float(np.clip(centres[0] / max(volume_shape[0], 1), 0.0, 1.0))
        if self.conditioning_semantics == "interior":
            # The generated volume is a window into an unbounded specimen —
            # no patch is near a surface (T-C: near-surface training patches
            # legitimately contain unlabelled exterior air, so an honest
            # small cond_dist ASKS for air).
            return depth, 1.0
        d_sur = min(
            min(c, extent - c) for c, extent in zip(centres, volume_shape)
        )
        dist = float(np.clip(d_sur, 0.0, _DIST_CAP) / _DIST_CAP)
        return depth, dist

    def _generate_latents(
        self,
        volume_shape: tuple[int, int, int],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
        gen_batch_size: int = 32,
    ) -> tuple[dict[tuple[int, int, int], torch.Tensor], dict[tuple[int, int, int], tuple[int, int, int]]]:
        """Run the eight-group parity schedule and return per-patch latents.

        Parameters
        ----------
        volume_shape    : (D, H, W) in voxels
        target_porosity : uniform per-patch VVF fallback (None = 0.05).  Prefer
                          ``local_por_map`` — D32 §4 warns against painting a
                          single volume target into every patch.
        autocast_dtype  : AMP dtype for the denoiser
        local_por_map   : dict (iz, iy, ix) → per-patch raw porosity phi
        gen_batch_size  : number of patches to sample in parallel through the UNet

        Returns
        -------
        (generated, grid_origins) — ``generated`` maps grid index → generated
        latent tensor (on CPU), ``grid_origins`` maps grid index → voxel origin (z0, y0, x0).
        """
        stride = self.generation_stride
        model_cfg = self.sampler.model.cfg
        por    = float(np.clip(target_porosity, _POR_MIN, _POR_MAX)) if target_porosity is not None else 0.05

        zs, ys, xs = self._tile_grid(volume_shape)

        # Grid index → origin voxel
        grid_origins = {(iz, iy, ix): (zs[iz], ys[iy], xs[ix])
                        for iz in range(len(zs))
                        for iy in range(len(ys))
                        for ix in range(len(xs))}

        # Store generated latents keyed by grid index (iz, iy, ix), held on CPU
        generated: dict[tuple[int, int, int], torch.Tensor] = {}

        # Eight-group parity schedule (D32 §3.3).  Within a group no two grid
        # indices differ by less than 2 on any axis, so no two patches overlap
        # and their generation really is independent.  Sorting inside a group is
        # only for determinism.
        groups = [
            (g, sorted(gi for gi in grid_origins if parity_group(gi) == g))
            for g in self.group_order
        ]

        total = len(grid_origins)
        logger.info(
            "VolumeGenerator: %d patches, grid %d×%d×%d, 8-group schedule %s (sizes %s)",
            total, len(zs), len(ys), len(xs),
            "→".join("".join(str(v) for v in g) for g, _ in groups),
            [len(p) for _, p in groups],
        )

        zero_latent = torch.zeros(
            self.z_channels, self.latent_size, self.latent_size,
            self.latent_size, dtype=torch.float32,
        )

        n_done = 0
        log_interval = max(1, total // 20)

        for group_idx, (group, group_patches) in enumerate(groups):
            for chunk_start in range(0, len(group_patches), gen_batch_size):
                chunk_gis = group_patches[chunk_start : chunk_start + gen_batch_size]

                # Build inputs for this chunk
                chunk_nbl:   list[torch.Tensor] = []
                chunk_nba:   list[torch.Tensor] = []
                chunk_por:   list[float]        = []
                chunk_depth: list[float]        = []
                chunk_dist:  list[float]        = []
                chunk_orient: list[torch.Tensor] = []

                for gi in chunk_gis:
                    z0, y0, x0 = grid_origins[gi]

                    states = neighbour_states(gi, lambda g: g in grid_origins,
                                              self.group_order)
                    if self.conditioning_semantics == "interior":
                        # unbounded specimen: the sample never ends at the
                        # generation grid, it is merely not generated there
                        states = [_NB_UNKNOWN if st == _NB_OOB else st
                                  for st in states]
                    nb_latents_list: list[torch.Tensor] = []
                    for d, state in zip(NEIGHBOUR_DIRS, states):
                        if state != _NB_EXISTS:
                            nb_latents_list.append(zero_latent)
                            continue
                        ngi = (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
                        # Touching neighbour: the whole latent is face-adjacent
                        # context, so it goes in unshifted (see conditioning.py).
                        nb_latents_list.append(
                            shift_into_target_frame(generated[ngi], d, self.latent_shift)
                            if self.neighbour_shift else generated[ngi]
                        )

                    chunk_nbl.append(torch.stack(nb_latents_list, dim=0))       # (6,C,L,L,L)
                    chunk_nba.append(torch.tensor(states, dtype=torch.long))    # (6,)

                    local_por = (
                        local_por_map[gi]
                        if (local_por_map is not None and gi in local_por_map)
                        else por
                    )
                    # Clamp every per-patch phi to the training range —
                    # single choke point for all map builders.
                    phi = float(np.clip(local_por, _POR_MIN, _POR_MAX))
                    chunk_por.append(
                        float(porosity_to_cond(phi, self.por_log_stats))
                        if model_cfg.use_por_cond else 0.0
                    )
                    depth, dist = self._patch_position((z0, y0, x0), volume_shape)
                    chunk_depth.append(depth)
                    chunk_dist.append(dist)
                    if model_cfg.use_orient_cond:
                        chunk_orient.append(self._patch_orient(z0))

                # Move to device and sample
                nb_latents_t = torch.stack(chunk_nbl).to(self.device)                              # (B,6,C,L,L,L)
                nb_avail_t   = torch.stack(chunk_nba).to(self.device)                              # (B,6)
                por_t        = torch.tensor(chunk_por,   dtype=torch.float32, device=self.device)  # (B,)
                depth_t      = torch.tensor(chunk_depth, dtype=torch.float32, device=self.device)  # (B,)
                dist_t       = torch.tensor(chunk_dist,  dtype=torch.float32, device=self.device)  # (B,)
                orient_t     = (
                    torch.stack(chunk_orient).to(self.device) if chunk_orient else None
                )                                                                                  # (B,2,L,L,L)

                z_batch = self.sampler.sample_batch(
                    nb_latents_t, nb_avail_t, por_t, depth_t, dist_t, orient_t,
                    autocast_dtype=autocast_dtype,
                )  # (B, C, L, L, L) float32 on device

                prev_done = n_done
                for i, gi in enumerate(chunk_gis):
                    generated[gi] = z_batch[i].cpu()
                n_done += len(chunk_gis)

                if patch_pbar is not None:
                    patch_pbar.update(len(chunk_gis))
                elif n_done // log_interval > prev_done // log_interval:
                    logger.info(
                        "Generated %d / %d patches (group %d/8 = %s)",
                        n_done, total, group_idx + 1, group,
                    )

        return generated, grid_origins

    def _generate_latents_joint(
        self,
        volume_shape: tuple[int, int, int],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
        window_stride: int = 32,
        window_batch: int = 32,
    ) -> tuple[dict[tuple[int, int, int], torch.Tensor], dict[tuple[int, int, int], tuple[int, int, int]]]:
        """MultiDiffusion-style JOINT denoising of the whole latent canvas.

        Overlapping windows (one 64³-voxel patch position each, at
        ``window_stride`` voxels — default 32, 50 % overlap) all denoise the
        SAME latent canvas.  At every DDIM timestep each window's ε prediction
        is computed (in mini-batches of ``window_batch``), the per-voxel
        predictions are fused by a cosine-weighted average
        (:func:`joint_window_weight` — see there for why not uniform), and ONE
        DDIM step is taken on the canvas.  The averaging happens on the ε
        predictions INSIDE the reverse process, never on finished samples:
        averaging finished samples halves the variance in the overlap and puts
        it off-manifold, which is exactly why post-hoc blending was removed
        (D38).  ε and x̂₀ averaging are equivalent here because all windows
        share the same timestep, so the two are related by one affine map.

        Neighbour conditioning: every window runs with availability
        all-UNKNOWN and zero neighbour latents.  The joint process replaces
        the sequential neighbour mechanism — each window already sees its
        neighbours' current noisy state implicitly through the overlap, so
        feeding the explicit neighbour-latent channels as well would
        double-count the same context (and hand the denoiser overlapping views
        of its own target, the leak class of D38).  Consequence: the CFG
        neighbour arm degenerates (``eps_full == eps_por``), so ``s_nb`` has
        no effect in joint mode.

        Scalar/orientation conditioning per window uses the same construction
        as the sequential path (``_patch_position`` / ``_patch_orient``); the
        per-window porosity is looked up in ``local_por_map`` by the TILING
        cell that contains the window centre.

        The final canvas is returned sliced into the stride-64 tiling grid, in
        the same ``(generated, grid_origins)`` format as
        :meth:`_generate_latents`, so decode and assembly are shared.
        """
        sampler = self.sampler
        if not (hasattr(sampler, "predict_eps") and hasattr(sampler, "_timesteps")):
            raise TypeError(
                "Joint mode needs a DDIMSampler (predict_eps + a fixed timestep "
                f"ladder); got {type(sampler).__name__}. The stochastic DDPM "
                "ancestral step is not defined for a fused prediction."
            )
        P  = self.patch_size
        ds = self.downsample
        L  = self.latent_size
        if window_stride % ds != 0:
            raise ValueError(
                f"joint window_stride={window_stride} voxels is not a multiple of "
                f"the VAE downsampling factor {ds} — it has no latent-cell "
                "representation."
            )
        if window_stride <= 0 or P % window_stride != 0:
            raise ValueError(
                f"joint window_stride={window_stride} must be a positive divisor "
                f"of patch_size={P} so the windows cover every tiled volume "
                "exactly."
            )

        zs, ys, xs = self._tile_grid(volume_shape)
        model_cfg = sampler.model.cfg
        por = float(np.clip(target_porosity, _POR_MIN, _POR_MAX)) if target_porosity is not None else 0.05

        canvas_cells = tuple(v // ds for v in volume_shape)
        s_cells = window_stride // ds
        origins = joint_window_origins(canvas_cells, L, s_cells)
        n_win = len(origins)
        win_slices = [
            (slice(oc[0], oc[0] + L), slice(oc[1], oc[1] + L), slice(oc[2], oc[2] + L))
            for oc in origins
        ]

        logger.info(
            "VolumeGenerator[joint]: canvas %s cells, %d windows at %d-cell "
            "stride, %d DDIM steps, window_batch=%d",
            canvas_cells, n_win, s_cells, len(sampler._timesteps) - 1, window_batch,
        )

        # ── per-window conditioning (same construction as the sequential path)
        n_tiles = (len(zs), len(ys), len(xs))
        win_por:   list[float] = []
        win_avail: list[list[int]] = []
        win_depth: list[float] = []
        win_dist:  list[float] = []
        win_orient: list[torch.Tensor] = []
        for oc in origins:
            ov = tuple(int(c) * ds for c in oc)          # voxel origin
            phi = por
            if local_por_map is not None:
                ti = tuple(
                    min((ov[a] + P // 2) // P, n_tiles[a] - 1) for a in range(3)
                )
                phi = local_por_map.get(ti, por)
            phi = float(np.clip(phi, _POR_MIN, _POR_MAX))
            win_por.append(
                float(porosity_to_cond(phi, self.por_log_stats))
                if model_cfg.use_por_cond else 0.0
            )
            depth, dist = self._patch_position(ov, volume_shape)
            win_depth.append(depth)
            win_dist.append(dist)
            if self.conditioning_semantics in ("interior", "legacy"):
                win_avail.append([_NB_UNKNOWN] * len(_NEIGHBOR_DIRS))
            else:
                # honest edges: a face beyond which the volume ends is OOB,
                # exactly as training saw at real specimen boundaries;
                # in-volume faces stay UNKNOWN (not yet resolved).
                win_avail.append([
                    _NB_OOB if not all(
                        0 <= ov[a] + d[a] * self.patch_size
                        <= volume_shape[a] - self.patch_size
                        for a in range(3))
                    else _NB_UNKNOWN
                    for d in _NEIGHBOR_DIRS
                ])
            if model_cfg.use_orient_cond:
                win_orient.append(self._patch_orient(ov[0]))

        win_avail_t = torch.tensor(win_avail, dtype=torch.long,
                                   device=self.device)          # (n_win, 6)
        por_t   = torch.tensor(win_por,   dtype=torch.float32, device=self.device)
        depth_t = torch.tensor(win_depth, dtype=torch.float32, device=self.device)
        dist_t  = torch.tensor(win_dist,  dtype=torch.float32, device=self.device)
        orient_t = torch.stack(win_orient).to(self.device) if win_orient else None

        # ── fusion weights (cosine, strictly positive) ───────────────────────
        weight = joint_window_weight(L).to(self.device)              # (L,L,L)
        weight_sum = torch.zeros((1, 1, *canvas_cells), device=self.device)
        for sl in win_slices:
            weight_sum[0, 0, sl[0], sl[1], sl[2]] += weight

        # ── joint reverse process on the canvas ──────────────────────────────
        C = self.z_channels
        B_max = min(int(window_batch), n_win)
        nb_zero = torch.zeros(B_max, len(_NEIGHBOR_DIRS), C, L, L, L,
                              device=self.device)


        sampler.model.eval()
        schedule = sampler.schedule.to(self.device)
        x = torch.randn(1, C, *canvas_cells, device=self.device)
        timesteps = sampler._timesteps

        for i, t_val in enumerate(timesteps[:-1]):
            t_prev_val = timesteps[i + 1]
            eps_sum = torch.zeros_like(x)
            for start in range(0, n_win, B_max):
                idx = list(range(start, min(start + B_max, n_win)))
                B = len(idx)
                xw = torch.stack(
                    [x[0, :, win_slices[j][0], win_slices[j][1], win_slices[j][2]]
                     for j in idx]
                )                                                    # (B,C,L,L,L)
                t_b = torch.full((B,), t_val, dtype=torch.long, device=self.device)
                eps = sampler.predict_eps(
                    xw, t_b, nb_zero[:B], win_avail_t[idx],
                    por_t[idx], depth_t[idx], dist_t[idx],
                    None if orient_t is None else orient_t[idx],
                    autocast_dtype,
                ).float()
                for k, j in enumerate(idx):
                    sl = win_slices[j]
                    eps_sum[0, :, sl[0], sl[1], sl[2]] += weight * eps[k]

            t      = torch.tensor([t_val],      dtype=torch.long, device=self.device)
            t_prev = torch.tensor([t_prev_val], dtype=torch.long, device=self.device)
            x = schedule.ddim_step(x, t, t_prev, eps_sum / weight_sum)
            if patch_pbar is not None:
                patch_pbar.update(1)

        # ── slice the coherent canvas into the stride-64 tiling grid ─────────
        Lc = P // ds
        generated:    dict[tuple[int, int, int], torch.Tensor] = {}
        grid_origins: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for iz, z0 in enumerate(zs):
            for iy, y0 in enumerate(ys):
                for ix, x0 in enumerate(xs):
                    cz, cy, cx = z0 // ds, y0 // ds, x0 // ds
                    generated[(iz, iy, ix)] = (
                        x[0, :, cz:cz + Lc, cy:cy + Lc, cx:cx + Lc].float().cpu()
                    )
                    grid_origins[(iz, iy, ix)] = (z0, y0, x0)
        return generated, grid_origins

    def generate(
        self,
        volume_size_mm: tuple[float, float, float],
        target_porosity: float | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
        local_por_map: dict | None = None,
        patch_pbar=None,
        gen_batch_size: int = 32,
        decode_batch_size: int = 64,
        mode: str = "sequential",
        joint_window_stride: int = 32,
        joint_window_batch: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
        """Generate a full volume — sequential parity schedule or joint denoising.

        ``mode="sequential"`` (the ablation baseline) runs the eight-group
        parity schedule with explicit neighbour conditioning; each patch gets
        one full DDIM pass.  ``mode="joint"`` runs the MultiDiffusion-style
        joint reverse process on one latent canvas
        (:meth:`_generate_latents_joint`).  Both modes decode the SAME way:
        once, patch-by-patch on the non-overlapping stride-64 tiling grid.

        Parameters
        ----------
        volume_size_mm    : (D, H, W) physical size in millimetres; each dimension
                            is snapped down to the nearest multiple of patch_size
        target_porosity   : uniform per-patch VVF fallback (None = 0.05)
        autocast_dtype    : AMP dtype for the denoiser and VAE
        local_por_map     : dict (iz, iy, ix) → per-patch raw porosity phi,
                            keyed by the TILING grid in both modes
        gen_batch_size    : sequential mode — patches sampled in parallel
        decode_batch_size : latents decoded in one VAE forward pass
        mode              : "sequential" | "joint"
        joint_window_stride : joint mode — voxels between window origins
                            (default 32 = 50 % overlap); must divide patch_size
        joint_window_batch  : joint mode — windows per UNet forward per timestep

        Note: ``patch_pbar`` counts patches in sequential mode but DDIM
        timesteps in joint mode (all windows advance together).

        Returns
        -------
        (xct_uint8, mask_uint8, stats) — uint8 ndarrays of shape volume_shape,
        plus a stats dict with the conditioning target, the assembled mask's
        actual porosity (self-audit of every generation run) and the seam
        discontinuity across every patch-to-patch face (D32 §3.4).
        """
        P   = self.patch_size
        vsz = self.voxel_size_mm

        # Snap each physical dimension to the nearest patch_size multiple (in voxels)
        volume_shape = tuple(
            (round(d / vsz) // P) * P for d in volume_size_mm
        )
        for d_mm, snapped_vox in zip(volume_size_mm, volume_shape):
            raw_vox = round(d_mm / vsz)
            if raw_vox - snapped_vox > 1:
                logger.warning(
                    "Volume axis %.3f mm: snapped %d vox → %d vox (dropped %d vox)",
                    d_mm, raw_vox, snapped_vox, raw_vox - snapped_vox,
                )

        vol_d, vol_h, vol_w = volume_shape
        stride = self.generation_stride

        nz = len(range(0, vol_d - P + 1, stride))
        ny = len(range(0, vol_h - P + 1, stride))
        nx = len(range(0, vol_w - P + 1, stride))

        patch_size_mm = P * vsz
        logger.info(
            "Volume %.1f×%.1f×%.1f mm  grid %d×%d×%d  patch %.2f mm",
            *volume_size_mm, nz, ny, nx, patch_size_mm,
        )

        if mode == "sequential":
            generated, grid_origins = self._generate_latents(
                volume_shape=volume_shape,
                target_porosity=target_porosity,
                autocast_dtype=autocast_dtype,
                local_por_map=local_por_map,
                patch_pbar=patch_pbar,
                gen_batch_size=gen_batch_size,
            )
        elif mode == "joint":
            generated, grid_origins = self._generate_latents_joint(
                volume_shape=volume_shape,
                target_porosity=target_porosity,
                autocast_dtype=autocast_dtype,
                local_por_map=local_por_map,
                patch_pbar=patch_pbar,
                window_stride=joint_window_stride,
                window_batch=joint_window_batch,
            )
        else:
            raise ValueError(f"Unknown generation mode {mode!r} — use 'sequential' or 'joint'.")

        # ── Direct tiling assembly ────────────────────────────────────────────
        # generation_stride == patch_size, so each decoded patch owns its own
        # block of the volume: no window, no weight buffer, no averaging of two
        # independent answers.  Logits go straight in; the sigmoid is applied
        # once at the end.
        xct_grey_vol  = np.zeros(volume_shape, dtype=np.float32)
        mask_logit_vol = np.zeros(volume_shape, dtype=np.float32)

        all_items = list(generated.items())

        self.vae.eval()
        with torch.no_grad():
            for chunk_start in range(0, len(all_items), decode_batch_size):
                chunk = all_items[chunk_start : chunk_start + decode_batch_size]

                z_stacked = torch.stack(
                    [(z_gen * self.latent_std + self.latent_mean) for _, z_gen in chunk], dim=0
                ).to(self.device)   # (B, C, ls, ls, ls) — denormalised latents

                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    dec         = self.vae.decoder(z_stacked)
                    xct_out  = self.vae.xct_head(dec)
                    mask_logits = self.vae.mask_head(dec)

                # squeeze(1): drop channel dim (size 1) while keeping batch dim.
                # Store raw logits — the sigmoid is applied once, after assembly.
                xct_patches  = xct_out.squeeze(1).float().cpu().numpy()   # (B,P,P,P) logits
                mask_patches = mask_logits.squeeze(1).float().cpu().numpy()  # (B,P,P,P) logits

                for i, (gi, _) in enumerate(chunk):
                    z0, y0, x0 = grid_origins[gi]
                    sl = (slice(z0, z0 + P), slice(y0, y0 + P), slice(x0, x0 + P))

                    xct_grey_vol[sl]  = xct_patches[i]
                    mask_logit_vol[sl] = mask_patches[i]

        # Threshold the mask in logit space (>0 == >0.5 in probability space),
        # then turn the XCT head output into grey levels IN PLACE — a production
        # volume is 500 M voxels, so every extra float32 buffer costs 2 GB.
        # The XCT head is NOT a logit: it regresses xct/255 directly, so the
        # conversion is clamp-and-scale (numpy mirror of models.vae.base.
        # decode_xct_u8).  A sigmoid here would squash everything into
        # [0.5, 0.731] and destroy the contrast of every generated volume.
        mask_out = (mask_logit_vol > 0.0).astype(np.uint8) * 255
        np.clip(xct_grey_vol, 0.0, 1.0, out=xct_grey_vol)
        xct_grey = np.multiply(xct_grey_vol, 255.0, out=xct_grey_vol)
        xct_out  = np.round(xct_grey).astype(np.uint8)

        # ── Seam diagnostic (D32 §3.4, replaces the overlap disagreement) ─────
        # Patches tile, so the only patch-to-patch interface left is the plane
        # between two blocks.  Measured on the decoder's own output — grey level
        # for the XCT head, logits for the mask head — against the natural
        # slice-to-slice variation inside a patch.
        seam_stats = {
            **seam_discontinuity(xct_grey, P, prefix="seam_xct"),
            **seam_discontinuity(mask_logit_vol, P, prefix="seam_mask"),
        }
        logger.info(
            "Seam discontinuity (ratio, 1.0 = indistinguishable from interior): "
            "xct=%.3f (z=%.3f y=%.3f x=%.3f, %d planes)  mask=%.3f",
            seam_stats.get("seam_xct_ratio", float("nan")),
            seam_stats.get("seam_xct_z_ratio", float("nan")),
            seam_stats.get("seam_xct_y_ratio", float("nan")),
            seam_stats.get("seam_xct_x_ratio", float("nan")),
            int(seam_stats.get("seam_xct_planes", 0)),
            seam_stats.get("seam_mask_ratio", float("nan")),
        )

        # Post-generation self-audit: actual porosity of the assembled mask
        # vs the conditioning target.
        actual_por = float((mask_out > 0).mean())
        clamped_por = (
            float(np.clip(target_porosity, _POR_MIN, _POR_MAX))
            if target_porosity is not None else 0.05
        )
        logger.info(
            "Assembled volume porosity: actual=%.4f  target=%s  conditioned=%.4f",
            actual_por,
            "None" if target_porosity is None else f"{target_porosity:.4f}",
            clamped_por,
        )
        stats = {
            "generation_mode": mode,
            "target_porosity": None if target_porosity is None else float(target_porosity),
            "conditioned_porosity": clamped_por,
            "actual_mask_porosity": actual_por,
            **seam_stats,
        }
        if mode == "joint":
            stats["joint_window_stride"] = int(joint_window_stride)

        return xct_out, mask_out, stats

    @staticmethod
    def save_tiff(
        xct:  np.ndarray,
        mask: np.ndarray,
        path_xct:  str | Path,
        path_mask: str | Path,
    ) -> None:
        """Write XCT and mask volumes to TIFF files."""
        import tifffile
        path_xct  = Path(path_xct)
        path_mask = Path(path_mask)
        path_xct.parent.mkdir(parents=True, exist_ok=True)
        path_mask.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path_xct),  xct)
        tifffile.imwrite(str(path_mask), mask)
        logger.info("Saved XCT  → %s", path_xct)
        logger.info("Saved mask → %s", path_mask)

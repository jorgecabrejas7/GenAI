"""Shared conditioning + assembly conventions for the LDM (D32 / ldm05).

This module is the SINGLE source of truth for the conventions that the data
side (latent store construction, ``LatentDataset``) and the model/sampler side
(``UNet3DDenoiser``, ``VolumeGenerator``) must agree on exactly:

* the neighbour direction order (``NEIGHBOUR_DIRS``),
* the availability state codes (``NB_OOB`` / ``NB_EXISTS`` / ``NB_UNKNOWN``),
* the grid index of a patch (:func:`grid_index`),
* the eight-group spatial parity schedule and its fixed ordering
  (``PARITY_GROUP_ORDER``, :func:`neighbour_states`),
* the geometry guard that keeps neighbours from leaking target content
  (:func:`validate_neighbour_geometry`).

Anything that computes neighbour conditioning MUST import from here rather
than re-deriving it, otherwise training and generation silently disagree.

Neighbour availability states
-----------------------------
0 — OOB (out of bounds): the neighbour position is outside the volume.
1 — EXISTS: the neighbour latent is known and has been provided.
2 — UNKNOWN: the neighbour has not been generated yet, because its parity
    group comes later in the fixed group ordering.

Neighbours TOUCH, they do not overlap
-------------------------------------
``neighbour_offset`` must be at least ``patch_size``.  At the original
``neighbour_offset = 32`` with ``patch_size = 64`` a face neighbour shared
HALF its voxels with the target, and the six neighbours between them tiled the
target completely: the ``+z``/``-z`` pair alone covers latent z-cells 0-7 and
8-15 over the full y/x extent.  Because opposite faces flip the SAME parity
bit they are always both EXISTS or both UNKNOWN, so 7 of the 8 parity groups
received the target's entire content as "conditioning".  Measured leak: the
``+z`` neighbour reproduced the target's overlap region with MAE 0.067 against
0.744 for a random patch (ratio 0.09) — a verbatim copy.  The model learned to
copy rather than to generate.

At ``neighbour_offset = 64`` the neighbour is face-adjacent: zero shared
voxels, so nothing of the target can be read off it.  The correct network
input is then the FULL, UNSHIFTED neighbour latent.  Rolling it into the
target frame (:func:`shift_into_target_frame`) only ever made sense for an
overlapping neighbour; at offset 64 the displacement is 64/4 = 16 latent cells
on a 16-cell axis, so the "overlap" is empty and a shifted tensor would be all
zeros.  :func:`validate_shift` turns that silent-zero case into a hard error.

Eight-group parity schedule (D32 §3.3)
--------------------------------------
The generation grid has stride 64, so patches tile without overlapping and the
schedule's original premise (same-group patches are independent) holds by
construction.  The grid index is ``origin // neighbour_offset``, which makes a
face neighbour exactly ±1 grid step on one axis for ANY patch origin — the
stored patches are sampled every 32 voxels, i.e. eight interleaved copies of
the stride-64 grid, and every patch's six neighbours live in its own copy.

Groups are generated in the fixed order :data:`PARITY_GROUP_ORDER` (rank
``4·pz + 2·py + px``).  A face neighbour flips exactly one parity bit, so its
group is deterministic: EXISTS when that group precedes the target's group,
UNKNOWN when it follows, OOB when the grid position does not exist.
Equivalently, both neighbours on axis *a* are EXISTS iff the target's parity
bit *a* is 1.  Context per group is therefore graded 0 / 2 / 2 / 2 / 4 / 4 /
4 / 6 EXISTS neighbours: only 1 patch in 8 is generated blind.  A plain
two-colour checkerboard would also be collision-free at stride 64, but it
would generate HALF the volume with no neighbour context at all, so the
eight-group ordering is kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NB_OOB     = 0
NB_EXISTS  = 1
NB_UNKNOWN = 2

# Neighbour directions as (dz, dy, dx) grid-index offsets.  Index i of any
# ``nb_latents`` / ``nb_avail`` tensor refers to NEIGHBOUR_DIRS[i].
NEIGHBOUR_DIRS: tuple[tuple[int, int, int], ...] = (
    ( 1, 0, 0), (-1, 0, 0),
    ( 0, 1, 0), ( 0,-1, 0),
    ( 0, 0, 1), ( 0, 0,-1),
)

_N_NEIGHBORS = len(NEIGHBOUR_DIRS)

# Fixed ordering of the eight (iz%2, iy%2, ix%2) parity groups.  Plain
# lexicographic order on the parity triple; group rank == 4·pz + 2·py + px.
PARITY_GROUP_ORDER: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
)


# ── geometry guard: neighbours must not share voxels with the target ──────────

_LEAK_FLAG = "allow_neighbour_overlap"


def neighbour_shared_voxels(neighbour_offset: int, patch_size: int) -> int:
    """Voxels a face neighbour shares with the target patch.

    A face neighbour is displaced by ``neighbour_offset`` voxels along one
    axis, so the two cubes intersect in ``max(patch_size - neighbour_offset, 0)
    × patch_size × patch_size`` voxels.  Zero means the patches only touch.
    """
    overlap_1d = max(int(patch_size) - int(neighbour_offset), 0)
    return overlap_1d * int(patch_size) * int(patch_size)


def validate_neighbour_geometry(
    neighbour_offset: int,
    patch_size: int,
    *,
    allow_neighbour_overlap: bool = False,
) -> None:
    """Raise unless a face neighbour shares NO voxel with the target patch.

    ``neighbour_offset >= patch_size`` is the invariant that makes neighbour
    conditioning honest.  Below it the neighbour carries a verbatim copy of
    part of the answer and the denoiser learns to copy — the ldm05 failure this
    guard exists to prevent (see the module docstring).

    ``allow_neighbour_overlap=True`` disables the check.  It ALLOWS CONTENT
    LEAKAGE and exists only to reproduce the leak for an ablation; it is never
    a valid training setting.
    """
    shared = neighbour_shared_voxels(neighbour_offset, patch_size)
    if shared == 0:
        return
    if allow_neighbour_overlap:
        return
    raise ValueError(
        f"neighbour_offset={neighbour_offset} < patch_size={patch_size}: each face "
        f"neighbour would share {shared} voxels with the target "
        f"({patch_size - neighbour_offset} of {patch_size} planes), handing the "
        f"denoiser part of the answer.  Opposite faces flip the same parity bit, so "
        f"they arrive together and can tile the target completely.  Use "
        f"neighbour_offset >= {patch_size} (touching neighbours), or set "
        f"data.{_LEAK_FLAG}=true — which ALLOWS CONTENT LEAKAGE and is for "
        f"ablation only."
    )


def grid_index(origin, neighbour_offset: int) -> tuple[int, int, int]:
    """Assembly-grid index of a patch origin ``(z0, y0, x0)`` in voxels.

    Floor division by ``neighbour_offset`` — not by the dataset sampling
    stride — so that a ±``neighbour_offset`` displacement is exactly ±1 grid
    step on that axis whatever the origin is.  Patches sampled at a finer
    stride form several interleaved copies of the same grid; each copy is
    self-contained because neighbour relations never cross between copies.
    """
    g = int(neighbour_offset)
    return (int(origin[0]) // g, int(origin[1]) // g, int(origin[2]) // g)


# ── eight-group parity schedule ───────────────────────────────────────────────

def parity_group(gi: tuple[int, int, int]) -> tuple[int, int, int]:
    """Parity group ``(iz%2, iy%2, ix%2)`` of a grid index."""
    return (gi[0] % 2, gi[1] % 2, gi[2] % 2)


def validate_group_order(order) -> tuple[tuple[int, int, int], ...]:
    """Coerce and validate a group ordering (a permutation of the 8 groups)."""
    coerced = tuple(tuple(int(v) for v in g) for g in order)
    if sorted(coerced) != sorted(PARITY_GROUP_ORDER):
        raise ValueError(
            f"Invalid parity group ordering {coerced!r} — it must be a "
            f"permutation of the eight (iz%2, iy%2, ix%2) triples."
        )
    return coerced


def resolve_group_order(metadata: dict | None = None) -> tuple[tuple[int, int, int], ...]:
    """Group ordering recorded by the latent store, else the shared default.

    The data side writes ``metadata["assembly"]["parity_group_order"]`` when
    building the store; reading it here keeps exactly one copy of the ordering
    in play.  When the key is absent, :data:`PARITY_GROUP_ORDER` is used — the
    same constant the data side imports.
    """
    if metadata:
        raw = (metadata.get("assembly") or {}).get("parity_group_order")
        if raw is not None:
            return validate_group_order(raw)
    return PARITY_GROUP_ORDER


def group_rank(
    gi: tuple[int, int, int],
    order: tuple[tuple[int, int, int], ...] = PARITY_GROUP_ORDER,
) -> int:
    """Position of a grid index's parity group in the generation ordering."""
    return order.index(parity_group(gi))


def neighbour_states(
    gi: tuple[int, int, int],
    in_grid,
    order: tuple[tuple[int, int, int], ...] = PARITY_GROUP_ORDER,
) -> list[int]:
    """Availability state of each of the 6 face neighbours of *gi*.

    Parameters
    ----------
    gi      : (iz, iy, ix) grid index of the target patch
    in_grid : callable (iz, iy, ix) -> bool — is that grid position inside the
              generation grid?
    order   : parity group ordering in use

    Returns
    -------
    list of 6 ints, aligned with :data:`NEIGHBOUR_DIRS`.
    """
    rank = group_rank(gi, order)
    states: list[int] = []
    for d in NEIGHBOUR_DIRS:
        ngi = (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
        if not in_grid(ngi):
            states.append(NB_OOB)
        elif group_rank(ngi, order) < rank:
            states.append(NB_EXISTS)
        else:
            states.append(NB_UNKNOWN)
    return states


# ── neighbour → target frame shift (overlapping neighbours only) ──────────────

def latent_shift_cells(neighbour_offset: int, patch_size: int, latent_size: int) -> int:
    """Neighbour displacement expressed in latent cells.

    ``neighbour_offset`` is in voxels (64 for touching neighbours); the VAE
    downsamples by ``patch_size // latent_size`` (4), so the neighbour is
    displaced by 16 latent cells — the full latent extent.
    """
    if patch_size % latent_size != 0:
        raise ValueError(
            f"patch_size={patch_size} is not a multiple of latent_size={latent_size}."
        )
    ds = patch_size // latent_size
    if neighbour_offset % ds != 0:
        raise ValueError(
            f"neighbour_offset={neighbour_offset} voxels is not a multiple of the "
            f"VAE downsampling factor {ds} — it cannot be expressed in latent cells."
        )
    return neighbour_offset // ds


def validate_shift(shift: int, latent_size: int) -> None:
    """Raise unless a neighbour shift actually moves content into the frame.

    Shifting is only meaningful for an OVERLAPPING neighbour.  When
    ``|shift| >= latent_size`` the shared region is empty, so the shifted
    tensor would be all zeros — a silent no-signal input that looks like a
    working conditioning path.  That is exactly the class of bug this guard
    exists to make loud.
    """
    if int(shift) < 0:
        raise ValueError(f"Neighbour shift must be non-negative, got {shift}.")
    if int(shift) >= int(latent_size):
        raise ValueError(
            f"Neighbour shift of {shift} latent cells is >= the latent size "
            f"{latent_size}: the neighbour and the target share no cells, so the "
            f"shifted tensor would be all zeros.  Touching neighbours must be fed "
            f"UNSHIFTED (set neighbour_shift=false); shifting is only valid when "
            f"neighbour_offset < patch_size."
        )


def _axis_slices(d: int, length: int, shift: int) -> tuple[slice, slice] | None:
    """(target-frame slice, neighbour-frame slice) of the overlap on one axis."""
    if d == 0:
        return slice(0, length), slice(0, length)
    if shift >= length:
        return None                      # patches do not overlap on this axis
    if d > 0:
        return slice(shift, length), slice(0, length - shift)
    return slice(0, length - shift), slice(shift, length)


def overlap_slices(
    direction: tuple[int, int, int],
    length: int,
    shift: int,
) -> tuple[tuple[slice, slice, slice], tuple[tuple[slice, slice, slice]]] | None:
    """Slices of the region shared by a patch and the patch at *direction*.

    Returns ``(dst, src)``, each a 3-tuple of slices: ``dst`` indexes the
    target patch's own array, ``src`` indexes the neighbour's array, and the
    two select the SAME physical region.  Returns ``None`` when the two
    patches do not overlap at all.
    """
    dst: list[slice] = []
    src: list[slice] = []
    for d in direction:
        pair = _axis_slices(d, length, shift)
        if pair is None:
            return None
        dst.append(pair[0])
        src.append(pair[1])
    return (dst[0], dst[1], dst[2]), (src[0], src[1], src[2])


def shift_into_target_frame(
    nb: torch.Tensor,
    direction: tuple[int, int, int],
    shift: int,
) -> torch.Tensor:
    """Roll a neighbour latent into the target patch's coordinate frame.

    ONLY valid for an overlapping neighbour (``neighbour_offset < patch_size``).
    The neighbour at grid offset *direction* has its origin displaced by
    ``direction * shift`` latent cells from the target's origin, so its cell
    ``j`` describes the physical location that the target calls
    ``j + direction*shift``.  This moves the overlapping part to that index and
    zero-fills the rest, so a convolution sees both patches in one frame
    (D32 §3.2).

    For touching neighbours the shared region is empty; the function raises
    rather than returning an all-zero tensor (see :func:`validate_shift`).

    Parameters
    ----------
    nb        : (..., L, L, L) neighbour latent
    direction : (dz, dy, dx) grid offset of the neighbour
    shift     : displacement in latent cells (``latent_shift_cells``)

    Returns
    -------
    Tensor of the same shape, zero outside the shared region.
    """
    length = nb.shape[-1]
    if nb.shape[-3] != length or nb.shape[-2] != length:
        raise ValueError(f"shift_into_target_frame expects a cubic latent, got {tuple(nb.shape)}.")
    validate_shift(shift, length)
    out = torch.zeros_like(nb)
    sl = overlap_slices(direction, length, shift)
    if sl is None:
        return out
    dst, src = sl
    out[..., dst[0], dst[1], dst[2]] = nb[..., src[0], src[1], src[2]]
    return out


class NeighborAvailabilityEmbedding(nn.Module):
    """Learned embedding for 6-connected neighbor availability states.

    Wraps :class:`torch.nn.Embedding` with per-neighbor position offsets so
    that the embedding for "EXISTS at neighbor 0" is distinct from "EXISTS at
    neighbor 3", giving the model spatial awareness of which direction a known
    neighbor came from.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding per neighbor.
    """

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # 6 neighbors × 3 states = 18 entries; each neighbor gets its own offset
        self.embedding = nn.Embedding(_N_NEIGHBORS * 3, embed_dim)

    def forward(self, nb_avail: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        nb_avail : (B, 6) long — availability states {0, 1, 2}

        Returns
        -------
        (B, 6 * embed_dim) — flattened per-neighbor embeddings
        """
        B = nb_avail.shape[0]
        offsets   = torch.arange(_N_NEIGHBORS, device=nb_avail.device).unsqueeze(0) * 3
        avail_idx = (nb_avail + offsets).long()           # (B, 6)
        emb       = self.embedding(avail_idx)             # (B, 6, embed_dim)
        return emb.view(B, _N_NEIGHBORS * self.embed_dim) # (B, 6*embed_dim)

"""The seven assessments, as data.

An assessment is a list of :class:`CaseSpec` - a case says WHAT to ask the
model for, never how to ask it.  Everything about the sampler lives in
:mod:`poregen.eval_v4.generate`, so adding a case is a change to this file
alone, and a case can be read without knowing the sampler API.

Geometry is fixed across the suite: 64-voxel tiles, 3x3x3-tile chunks, windows
every 32 voxels, decode windows every 32 voxels.  Where an assessment needs a
different value it says so in its own builder and the manifest records it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from poregen.eval_v4.io import LATENT_DOWNSAMPLE, TILE, repo_root

SEEDS = (101, 202, 303)

#: (z, y, x) in voxels.  The small volume is three tiles a side - exactly one
#: chunk, so it has NO chunk plane and isolates the window seam.  The large one
#: is the T-I reader's 1024-pixel in-plane window over a 192-voxel laminate.
SHAPE_SMALL = (192, 192, 192)
SHAPE_LARGE = (192, 1024, 1024)
#: Wide enough in-plane for a 200-voxel hole and a 64-voxel notch to both fit
#: with clearance, and small enough to generate seven of.
SHAPE_GEOMETRY = (192, 512, 512)
#: Four tiles a side, so a chunk plane falls at voxel 192 and the 192-cubed
#: region can be placed on either side of it.
SHAPE_ASSEMBLY = (256, 256, 256)

CHUNK_TILES = (3, 3, 3)
WINDOW_STRIDE = 32
DECODE_STRIDE = 32
DDIM_DEFAULT = 200
TARGET_DEFAULT = 0.03

#: Fixed permutation of layup A with the SAME ply population - three -45, two
#: 0, three 45, two 90 - so any difference in recovery is the stacking ORDER
#: and not the mix of angles.  Chosen once and written down; it is deliberately
#: not the reverse of A, because reversing a stack is the face-order freedom
#: the direct scoring already refuses to grant.
LAYUP_C_PERMUTATION = (90, 45, -45, 45, 0, 90, -45, 0, 45, -45)

#: Ply pitch of the training layup, in voxels.  The nominal 0.508 mm cured-ply
#: thickness is 20.3 voxels at 25 um; 19.6 is the pitch the T-I estimator fitted
#: on the real scans and the one every earlier campaign generated with.
PLY_VOX_A = 19.6


@dataclass(frozen=True)
class CaseSpec:
    """One volume to generate.  Purely a request; nothing here is a setting of
    the sampler that the manifest does not also record."""

    name: str
    assessment: str
    volume_shape: tuple[int, int, int]
    seed: int
    layup: tuple[int, ...]
    ply_thickness_vox: float
    target_phi: float | None = None
    ddim_steps: int = DDIM_DEFAULT
    chunk_tiles: tuple[int, int, int] = CHUNK_TILES
    window_stride: int = WINDOW_STRIDE
    decode_stride: int = DECODE_STRIDE
    s_por: float = 1.0
    s_nb: float = 1.0
    #: builds the requested phi per 64-voxel TILE, given the tile grid and seed
    field_fn: Callable[[tuple[int, int, int], int], np.ndarray] | None = None
    #: builds the requested specimen envelope at VOXEL resolution
    material_fn: Callable[[tuple[int, int, int]], np.ndarray] | None = None
    #: ``(lo, hi)`` in voxels; drives cond_depth and cond_dist6.  ``None`` means
    #: "this whole volume is the specimen", which is the normal case.
    specimen_box: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    #: the sub-block of the canvas the case is about
    region_offset: tuple[int, int, int] | None = None
    region_shape: tuple[int, int, int] | None = None
    #: request translation inside the canvas - see :func:`assembly_cases`
    request_offset: tuple[int, int, int] = (0, 0, 0)
    notes: dict = field(default_factory=dict)

    @property
    def tile_grid(self) -> tuple[int, int, int]:
        return tuple(s // TILE for s in self.volume_shape)  # type: ignore[return-value]

    @property
    def latent_grid(self) -> tuple[int, int, int]:
        return tuple(s // LATENT_DOWNSAMPLE for s in self.volume_shape)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Requested porosity fields, on the tile grid
# ---------------------------------------------------------------------------

def field_two_halves(grid, seed, lo: float = 0.01, hi: float = 0.05) -> np.ndarray:
    """``lo`` in the first half of y, ``hi`` in the second.

    The bluntest local request there is: one straight boundary, far from every
    volume face, with a 5x step across it.  A model that cannot deliver this
    cannot deliver anything finer.
    """
    f = np.full(grid, float(lo), np.float32)
    f[:, grid[1] // 2:, :] = float(hi)
    return f


def field_checkerboard(grid, seed, lo: float = 0.01, hi: float = 0.05) -> np.ndarray:
    """Alternating ``lo``/``hi`` tiles - the request at the finest scale the
    conditioning can express, one tile."""
    iz, iy, ix = np.indices(grid)
    return np.where((iz + iy + ix) % 2 == 0, float(lo), float(hi)).astype(np.float32)


def field_coherent(grid, seed, target: float = TARGET_DEFAULT) -> np.ndarray:
    """The coherent field the production generator uses.

    A T-E marginal draw per tile smoothed with the T-D correlation lengths and
    rescaled to ``target``: the only one of the three requests that looks like
    real porosity, so it is the one whose failure would matter in use.
    """
    from poregen.diffusion.porosity_field import (  # noqa: PLC0415
        DEFAULT_TD_RESULTS,
        DEFAULT_TE_RESULTS,
        build_porosity_field,
        load_corr_lengths_voxels,
        load_sampler,
    )

    root = repo_root()
    return build_porosity_field(
        grid_shape=tuple(grid),
        target=float(target),
        sampler=load_sampler(root / DEFAULT_TE_RESULTS),
        corr_lengths_voxels=load_corr_lengths_voxels(root / DEFAULT_TD_RESULTS),
        stride_voxels=TILE,
        seed=int(seed),
    ).astype(np.float32)


FIELDS = {
    "halves": field_two_halves,
    "checkerboard": field_checkerboard,
    "coherent": field_coherent,
}


# ---------------------------------------------------------------------------
# Requested material maps, at voxel resolution
# ---------------------------------------------------------------------------

NOTCH_VOX = 64
HOLE_DIAMETER_VOX = 200


def material_notch_and_hole(shape: tuple[int, int, int]) -> np.ndarray:
    """A 64-voxel notch and a 200-voxel cylindrical hole through z.

    Both shapes are taken from the real coupons: the drilled registration holes
    are ~200 voxels across and are the only interior source of large air in the
    dataset, and a notch is the machining feature the model has never been
    asked for.  They are placed so they do not touch: the notch is cut into the
    ``y = 0`` face at mid-x, the hole is centred in y at three quarters of x.
    """
    d, h, w = shape
    m = np.ones(shape, dtype=bool)

    cx = w // 2
    m[:, :NOTCH_VOX, cx - NOTCH_VOX // 2: cx + NOTCH_VOX // 2] = False

    r = HOLE_DIAMETER_VOX / 2.0
    hy, hx = h / 2.0, 0.75 * w
    yy = np.arange(h)[:, None] - hy + 0.5
    xx = np.arange(w)[None, :] - hx + 0.5
    m[:, (yy ** 2 + xx ** 2) < r ** 2] = False

    if m.all():
        raise ValueError(f"the notch and hole do not fit in a {shape} volume.")
    return m


# ---------------------------------------------------------------------------
# Layups
# ---------------------------------------------------------------------------

def load_layups(repo: str | Path | None = None) -> dict[str, dict]:
    """The three requested stacking sequences.

    A and B16 are read from ``data/layup_ground_truth.json`` - they are the
    expert's record of what the panels actually are, and copying them into
    source would let the two drift.  C is the fixed permutation of A defined
    above.
    """
    root = Path(repo) if repo else repo_root()
    path = root / "data" / "layup_ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; the layup requests come from it.")
    seq = json.loads(path.read_text())["sequences"]

    a = tuple(int(x) for x in seq["A"]["plies"])
    b = tuple(int(x) for x in seq["B"]["plies"])
    b_pitch = float(seq["B"]["ply_thickness_mm"]) / 0.025
    if sorted(LAYUP_C_PERMUTATION) != sorted(a):
        raise ValueError(
            f"layup C {LAYUP_C_PERMUTATION} is not a permutation of A {a}; the "
            "assessment compares stacking ORDER, so the ply population must match."
        )
    return {
        "A": {"plies": a, "ply_vox": PLY_VOX_A,
              "note": "the training layup - 74 of 78 training volumes carry it"},
        "C": {"plies": LAYUP_C_PERMUTATION, "ply_vox": PLY_VOX_A,
              "note": "a permutation of A with the same ply population"},
        "B16": {"plies": b, "ply_vox": b_pitch,
                "note": f"the 16-ply {seq['B']['notation']} sequence, "
                        f"{seq['B']['ply_thickness_mm']} mm plies"},
    }


def layup_a(repo: str | Path | None = None) -> tuple[tuple[int, ...], float]:
    spec = load_layups(repo)["A"]
    return spec["plies"], spec["ply_vox"]


# ---------------------------------------------------------------------------
# The assessments
# ---------------------------------------------------------------------------

def sampler_cases(repo=None) -> list[CaseSpec]:
    """1 - what the DDIM step count and the volume scale cost and buy.

    Three step counts at two scales.  The small volume is one chunk, so it
    reports the window seam alone; the large one has chunk planes and is where
    a step count that looks sufficient at 192 has to prove it again.
    """
    plies, pitch = layup_a(repo)
    out = []
    for shape, tag in ((SHAPE_SMALL, "192"), (SHAPE_LARGE, "1024")):
        for steps in (50, 100, 200):
            for seed in SEEDS:
                out.append(CaseSpec(
                    name=f"{tag}_ddim{steps}_seed{seed}",
                    assessment="sampler",
                    volume_shape=shape,
                    seed=seed,
                    layup=plies,
                    ply_thickness_vox=pitch,
                    target_phi=TARGET_DEFAULT,
                    ddim_steps=steps,
                    notes={"scale": tag, "layup": "A"},
                ))
    return out


POROSITY_TARGETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
#: Above the training range (phi clamps at 0.107), so it is reported as a
#: failure mode and never enters the dose-response fit.
OFF_MANIFOLD_TARGET = 0.15


def porosity_global_cases(repo=None) -> list[CaseSpec]:
    """2 - the dose response, and one request the model was never trained for."""
    plies, pitch = layup_a(repo)
    out = []
    for target in (*POROSITY_TARGETS, OFF_MANIFOLD_TARGET):
        for seed in SEEDS:
            out.append(CaseSpec(
                name=f"target{target:g}_seed{seed}",
                assessment="porosity_global",
                volume_shape=SHAPE_SMALL,
                seed=seed,
                layup=plies,
                ply_thickness_vox=pitch,
                target_phi=target,
                notes={
                    "layup": "A",
                    "off_manifold": target == OFF_MANIFOLD_TARGET,
                },
            ))
    return out


def porosity_local_cases(repo=None) -> list[CaseSpec]:
    """3 - three painted fields on the 3x3x3 tile grid of a 192-cubed volume."""
    plies, pitch = layup_a(repo)
    out = []
    for fname, fn in FIELDS.items():
        for seed in SEEDS:
            out.append(CaseSpec(
                name=f"{fname}_seed{seed}",
                assessment="porosity_local",
                volume_shape=SHAPE_SMALL,
                seed=seed,
                layup=plies,
                ply_thickness_vox=pitch,
                target_phi=TARGET_DEFAULT,
                field_fn=fn,
                notes={"field": fname, "layup": "A"},
            ))
    return out


CFG_SPOR = (1.0, 1.5, 2.0)
CFG_TARGETS = (0.02, 0.05)
#: The porosity guidance the neighbour arm is measured at.  It matches the
#: campaign-05 joint_oob arm, so the s_nb rows sit beside numbers that exist.
CFG_SNB_SPOR = 1.5


def cfg_cases(repo=None) -> list[CaseSpec]:
    """4 - what the two guidance scales do.

    The porosity arm sweeps ``s_por`` at two targets.  The neighbour arm turns
    ``s_nb`` off and on at one setting, which is the only test that says whether
    neighbour conditioning acts at all: at ``s_nb = 0`` the denoiser is told
    nothing about its neighbours, so if the volumes agree across the chunk
    plane, the neighbour arm is inert.
    """
    plies, pitch = layup_a(repo)
    out = []
    for s_por in CFG_SPOR:
        for target in CFG_TARGETS:
            for seed in SEEDS:
                out.append(CaseSpec(
                    name=f"spor{s_por:g}_target{target:g}_seed{seed}",
                    assessment="cfg",
                    volume_shape=SHAPE_SMALL,
                    seed=seed,
                    layup=plies,
                    ply_thickness_vox=pitch,
                    target_phi=target,
                    s_por=s_por,
                    notes={"arm": "s_por", "layup": "A"},
                ))
    for s_nb in (0.0, 1.0):
        for seed in SEEDS:
            out.append(CaseSpec(
                # The chunk plane is what the neighbour arm acts on, so this arm
                # runs on the four-tile volume that has one.
                name=f"snb{s_nb:g}_seed{seed}",
                assessment="cfg",
                volume_shape=SHAPE_ASSEMBLY,
                seed=seed,
                layup=plies,
                ply_thickness_vox=pitch,
                target_phi=TARGET_DEFAULT,
                s_por=CFG_SNB_SPOR,
                s_nb=s_nb,
                notes={"arm": "s_nb", "layup": "A"},
            ))
    return out


def layup_cases(repo=None) -> list[CaseSpec]:
    """5 - three stacking sequences read back by two independent readers."""
    layups = load_layups(repo)
    out = []
    for name, spec in layups.items():
        for seed in SEEDS:
            out.append(CaseSpec(
                name=f"{name}_seed{seed}",
                assessment="layup",
                volume_shape=SHAPE_LARGE,
                seed=seed,
                layup=spec["plies"],
                ply_thickness_vox=spec["ply_vox"],
                target_phi=TARGET_DEFAULT,
                notes={"layup": name, "layup_note": spec["note"]},
            ))
    return out


#: The window grid is anchored at the chunk origin, so the only way to move it
#: relative to the requested content is to translate the request inside a bigger
#: canvas.  Offset 0 is the reference every other offset is read against.
#: 32 voxels is a WHOLE window stride and 16 is half of one, which is what
#: separates the two things an offset can move - see :func:`assembly_cases`.
ASSEMBLY_OFFSETS = (0, 16, 32)
ASSEMBLY_REGION = (192, 192, 192)


def assembly_cases(repo=None) -> list[CaseSpec]:
    """6 - does the answer depend on where the assembly grid happens to fall?

    The same 192-cubed request, generated three times with the same seed, at
    three positions in a 256-cubed canvas.  An offset moves TWO independent
    things and one offset cannot tell them apart, so there are two non-zero
    offsets:

    ``0``   the region is exactly chunk zero: no chunk plane crosses it, and
            the window grid starts on the region origin.  The reference.
    ``32``  one whole window stride, so the window grid keeps the same phase
            relative to the requested content - every window origin is still a
            multiple of 32 in region coordinates.  What changes is the CHUNK
            alignment: the chunk plane at canvas voxel 192 now runs through the
            region at region coordinate 160.
    ``16``  half a window stride, so the window grid falls in a different PHASE
            relative to the content (region-relative origins are 16, 48, 80 …),
            with a chunk plane at region coordinate 176 as well.

    Read 0 against 32 for chunk alignment and 32 against 16 for window phase.

    Everything the region is asked for is translated with it: the specimen box
    is the region itself, so ``cond_depth`` and ``cond_dist6`` at a given
    region-relative position are identical in every run; the porosity request
    is uniform, so it is translation-invariant; the orientation profile is
    shifted by the same offset; and ``request_offset`` puts the sampler's noise
    draws in the region's frame, so the noise realisation is held too.  What is
    left to differ is the grid the volume is assembled on.
    """
    plies, pitch = layup_a(repo)
    out = []
    for off in ASSEMBLY_OFFSETS:
        box = ((off, off, off), tuple(off + s for s in ASSEMBLY_REGION))
        for seed in SEEDS:
            out.append(CaseSpec(
                name=f"offset{off}_seed{seed}",
                assessment="assembly",
                volume_shape=SHAPE_ASSEMBLY,
                seed=seed,
                layup=plies,
                ply_thickness_vox=pitch,
                target_phi=TARGET_DEFAULT,
                specimen_box=box,
                material_fn=_box_material(box),
                region_offset=(off, off, off),
                region_shape=ASSEMBLY_REGION,
                request_offset=(off, off, off),
                notes={"offset": off, "layup": "A"},
            ))
    return out


def _box_material(box) -> Callable[[tuple[int, int, int]], np.ndarray]:
    """Envelope builder: specimen inside ``box``, exterior air outside it."""
    lo, hi = box

    def build(shape: tuple[int, int, int]) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        return m

    return build


def geometry_cases(repo=None) -> list[CaseSpec]:
    """7 - a material map the model must carve air into."""
    plies, pitch = layup_a(repo)
    return [
        CaseSpec(
            name=f"notch_hole_seed{seed}",
            assessment="geometry",
            volume_shape=SHAPE_GEOMETRY,
            seed=seed,
            layup=plies,
            ply_thickness_vox=pitch,
            target_phi=TARGET_DEFAULT,
            material_fn=material_notch_and_hole,
            notes={"layup": "A", "features": "64-voxel notch + 200-voxel hole"},
        )
        for seed in SEEDS
    ]


#: Porosity levels the microstructure statistics are compared at.  Three, not
#: seven: every level needs its own matched real reference, and the test panels
#: only reach so far up the porosity range.  They span it — 0.01 is where most
#: real material sits, 0.06 is near the top of what a test panel offers.
MICRO_TARGETS = (0.01, 0.03, 0.06)


def microstructure_cases(repo=None) -> list[CaseSpec]:
    """8 - does the microstructure have the right STATISTICS?

    Three porosity levels, because every distribution statistic here is
    confounded by porosity: a set with twice the pore fraction has a different
    S2 amplitude, a different pore count and different slice texture whatever
    the model does.  Comparing a generated set with a real set at a DIFFERENT
    porosity would measure the porosity gap and call it a texture gap, so each
    level is scored against real crops matched to it.

    DDIM-200 and layup A throughout: this assessment asks about the
    microstructure, so everything else is held at the setting the rest of the
    suite treats as standard.
    """
    plies, pitch = layup_a(repo)
    return [
        CaseSpec(
            name=f"phi{target:g}_seed{seed}",
            assessment="microstructure",
            volume_shape=SHAPE_SMALL,
            seed=seed,
            layup=plies,
            ply_thickness_vox=pitch,
            target_phi=target,
            ddim_steps=DDIM_DEFAULT,
            notes={"layup": "A", "micro_level": target},
        )
        for target in MICRO_TARGETS
        for seed in SEEDS
    ]


ASSESSMENTS: dict[str, Callable[..., list[CaseSpec]]] = {
    "sampler": sampler_cases,
    "porosity_global": porosity_global_cases,
    "porosity_local": porosity_local_cases,
    "cfg": cfg_cases,
    "layup": layup_cases,
    "assembly": assembly_cases,
    "geometry": geometry_cases,
    "microstructure": microstructure_cases,
}

#: Assessments whose measure step also reads another assessment's volumes.
#: ``microstructure`` reads the matched real crops the ``real-floor`` stage
#: writes: without them it has a number and no floor to read it against, which
#: for a distribution distance is no measurement at all.
BORROWS = {"assembly": ("sampler",), "microstructure": ("real_floor",)}


def build_cases(assessment: str, repo=None) -> list[CaseSpec]:
    if assessment not in ASSESSMENTS:
        raise KeyError(
            f"unknown assessment {assessment!r}; choose from {sorted(ASSESSMENTS)}"
        )
    return ASSESSMENTS[assessment](repo)

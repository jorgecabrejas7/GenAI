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

from functools import partial

import numpy as np

from poregen.eval_v4 import stress_geometry as SG
from poregen.diffusion.conditioning import POR_MAX, POR_MIN
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
    #: Chunk overlap in voxels, and how the shared strip is written.  0 is the
    #: production partition.  Non-zero exists for the chunk-band trial: the
    #: strip is pinned RePaint-style while the successor denoises, then written
    #: as a cosine blend of both chunks' outputs — or restored to the
    #: predecessor's value when ``chunk_overlap_write`` is "pin", which does
    #: NOT remove the band and is kept in order to show that.
    #: ``chunk_overlap_pinned`` (None = the whole overlap) is how much of the
    #: strip is held fixed while the successor denoises; a shorter pin frees
    #: the predecessor's own depleted rim for the successor to redraw.
    chunk_overlap: int = 0
    chunk_overlap_pinned: int | None = None
    chunk_overlap_write: str = "blend"
    #: Arm (f): for windows whose neighbour set is MIXED (some faces solved,
    #: the face toward the next chunk not), predict with the neighbour arm
    #: dropped. One model call, so it costs production time.
    drop_neighbours_when_mixed: bool = False
    window_stride: int = WINDOW_STRIDE
    decode_stride: int = DECODE_STRIDE
    s_por: float = 1.0
    s_nb: float = 1.0
    #: Where each window's six face neighbours come from - one of
    #: :data:`poregen.diffusion.sampler.NEIGHBOUR_MODES`.  ``"canvas"`` is the
    #: production path; only :func:`assembly_modes_cases` asks for another.
    neighbour_mode: str = "canvas"
    #: Hold the porosity request inside the training range before conditioning
    #: on it.  ON everywhere but assessment 13, whose whole point is to ask for
    #: a porosity the training set does not contain.  The clamp is not
    #: symmetric: at the bottom 0.000, 0.001 and 0.002 all collapse to ONE
    #: request with it on, and at the top 0.150 is only +0.26 sd of cond_por
    #: beyond POR_MAX.
    clamp_porosity: bool = True
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


#: The inset specimen box: material for z in [32, 160), air above and below.
#: 32 voxels of air is half a tile, so each face sits INSIDE a tile rather than
#: on a tile boundary — a surface that landed on a 64-plane could be produced by
#: the assembly rather than by the model obeying the map.
SURFACE_Z_LO = 32
SURFACE_Z_HI = 160


def material_inset_z(shape: tuple[int, int, int]) -> np.ndarray:
    """Material only for ``z in [SURFACE_Z_LO, SURFACE_Z_HI)``; air above and below.

    Every gate before this one was measured on a full-material box, so none of
    them could tell whether the model RENDERS air where the material map asks
    for it, or where it puts the interface. This is the simplest geometry that
    asks both questions and has an exact answer: two flat surfaces at known z.
    """
    d, h, w = shape
    if not 0 < SURFACE_Z_LO < SURFACE_Z_HI < d:
        raise ValueError(
            f"the inset box [{SURFACE_Z_LO}, {SURFACE_Z_HI}) does not fit in depth {d}"
        )
    m = np.zeros(shape, dtype=bool)
    m[SURFACE_Z_LO:SURFACE_Z_HI] = True
    return m


#: Fallback roughness for the rough surface request, in voxels, used only when
#: the real floor has not been measured yet. The floor is region-dependent —
#: detrended Sa over crops of one test volume ranged 0.17 to 2.23 voxels and the
#: lateral correlation length 2 to 58 — so these are the middle of the observed
#: range, not a precise claim, and every case records which source it used.
ROUGH_SA_FALLBACK_VOX = 1.5
ROUGH_CORR_LEN_FALLBACK_VOX = 12.0


def real_surface_target(repo=None) -> dict:
    """Sa and correlation length for the rough request, from the real floor.

    Read from the floor file when it exists so the request is matched to
    measured material rather than to a guess; falls back to the constants above
    otherwise, and says which it used. A rough request built on invented
    numbers would make the roughness RATIO gates meaningless — they compare the
    generated surface against this target.
    """
    from poregen.eval_v4.io import repo_root            # noqa: PLC0415
    from poregen.eval_v4.real_floor import SURFACE_FLOOR_FILE  # noqa: PLC0415

    root = Path(repo) if repo else repo_root()
    for cand in (root / "runs" / "campaigns" / "12-eval-v4" / "real_floor" / SURFACE_FLOOR_FILE,):
        if not cand.exists():
            continue
        try:
            d = json.loads(cand.read_text())
        except Exception:                               # noqa: BLE001
            continue
        sa, cl = [], []
        for st in (d.get("per_volume") or {}).values():
            for f in ("lower", "upper"):
                det = (st.get(f) or {}).get("detrended") or {}
                if det.get("sa") is not None:
                    sa.append(det["sa"])
                m = (det.get("correlation_length_vox") or {}).get("mean")
                if m is not None:
                    cl.append(m)
        if sa:
            return {"sa_vox": float(np.median(sa)),
                    "correlation_length_vox": float(np.median(cl)) if cl
                    else ROUGH_CORR_LEN_FALLBACK_VOX,
                    "source": str(cand), "n_faces": len(sa)}
    return {"sa_vox": ROUGH_SA_FALLBACK_VOX,
            "correlation_length_vox": ROUGH_CORR_LEN_FALLBACK_VOX,
            "source": "fallback constants (real floor not measured yet)",
            "n_faces": 0}


def gaussian_height_field(shape2d, sa: float, corr_len: float, seed: int) -> np.ndarray:
    """A Gaussian random height field with the requested Sa and correlation length.

    White noise smoothed by a Gaussian kernel, then rescaled so the mean
    absolute deviation is exactly ``sa``. For a Gaussian-smoothed field the
    autocovariance goes as exp(-r^2 / 4s^2), so the 1/e correlation length is
    2s and the kernel is ``corr_len / 2``.

    Amplitude and correlation length are both matched because a surface is not
    described by amplitude alone: the same Sa with the wrong lateral scale is a
    different surface, and the model would be asked for something real material
    never looks like.
    """
    from scipy import ndimage                           # noqa: PLC0415

    rng = np.random.default_rng(seed)
    h = rng.standard_normal(shape2d)
    sigma = max(corr_len / 2.0, 1e-3)
    h = ndimage.gaussian_filter(h, sigma=sigma, mode="wrap")
    h -= h.mean()
    cur = np.abs(h).mean()
    return h * (sa / cur) if cur > 0 else h


def material_rough_z(shape, *, seed: int, sa: float, corr_len: float) -> np.ndarray:
    """Specimen between two ROUGH faces, so the request has fractional rim cells.

    The flat request asks for a plane exactly on the latent grid: every pooled
    cell is 0 or 1, and the model can satisfy it with a razor-flat surface. Real
    training data never looks like that — a rough real surface cuts cells and
    produces FRACTIONAL boundary cells. This builds that request, so the flat
    result can be read as controllability rather than as realism.
    """
    d, h, w = shape
    lo = gaussian_height_field((h, w), sa, corr_len, seed)
    hi = gaussian_height_field((h, w), sa, corr_len, seed + 5000)
    z_lo = np.clip(SURFACE_Z_LO + lo, 1, d - 2)
    z_hi = np.clip(SURFACE_Z_HI + hi, 2, d - 1)
    zz = np.arange(d)[:, None, None]
    return (zz >= z_lo[None, :, :]) & (zz < z_hi[None, :, :])


def surface_cases(repo=None) -> list[CaseSpec]:
    """9 - does the model put air, and the interface, where the map asks?

    TWO request types, and the pair is the point.

    ``flat``  the specimen box with planar faces on the latent grid. Every
              pooled cell is 0 or 1. This measures CONTROLLABILITY: is the
              interface where it was asked for.
    ``rough`` faces displaced by a Gaussian height field matched to the real
              floor's detrended Sa and correlation length, so the pooled map
              carries FRACTIONAL rim cells. This measures REALISM: can the model
              produce a surface with the texture real material has.

    The flat case at step 74000 came out Sa 0.03 voxels against a real floor of
    order 1 voxel, i.e. far flatter than any real surface — while passing every
    position gate. That is not necessarily a defect: a 0/1 map REQUESTS a plane,
    and the model obeyed. The rough request is what separates "obeyed an
    unrealistic request" from "cannot make a rough surface".
    """
    plies, pitch = layup_a(repo)
    tgt = real_surface_target(repo)
    out = []
    for shape, tag, seeds in ((SHAPE_SMALL, "192", SEEDS),
                              (SHAPE_LARGE, "1024", SEEDS[:1])):
        for steps in (50, 200):
            for seed in seeds:
                out.append(CaseSpec(
                    name=f"flat_{tag}_ddim{steps}_seed{seed}",
                    assessment="surface",
                    volume_shape=shape, seed=seed,
                    layup=plies, ply_thickness_vox=pitch,
                    target_phi=TARGET_DEFAULT, ddim_steps=steps,
                    material_fn=material_inset_z,
                    notes={"layup": "A", "scale": tag, "ddim_steps": steps,
                           "request": "flat", "z_lo": SURFACE_Z_LO, "z_hi": SURFACE_Z_HI},
                ))
    # Rough: 192-cubed x 3 seeds and 1024-wide x 1 seed, DDIM-50 only — the
    # flat rows already show the step count does not change the surface, and a
    # 1024-wide case is expensive.
    for shape, tag, seeds in ((SHAPE_SMALL, "192", SEEDS),
                              (SHAPE_LARGE, "1024", SEEDS[:1])):
        for seed in seeds:
            out.append(CaseSpec(
                name=f"rough_{tag}_ddim50_seed{seed}",
                assessment="surface",
                volume_shape=shape, seed=seed,
                layup=plies, ply_thickness_vox=pitch,
                target_phi=TARGET_DEFAULT, ddim_steps=50,
                material_fn=partial(material_rough_z, seed=seed,
                                    sa=tgt["sa_vox"], corr_len=tgt["correlation_length_vox"]),
                notes={"layup": "A", "scale": tag, "ddim_steps": 50,
                       "request": "rough", "z_lo": SURFACE_Z_LO, "z_hi": SURFACE_Z_HI,
                       "requested_sa_vox": tgt["sa_vox"],
                       "requested_correlation_length_vox": tgt["correlation_length_vox"],
                       "roughness_source": tgt["source"]},
            ))
    return out


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


#: Porosity assessments run at BOTH step counts, and the reason is measured,
#: not precautionary. The ldm06 40k diagnostic shows porosity conditioning is
#: 5-7x worse at 200 steps than at 50: overall por_mae 0.0011-0.0014 at 50
#: against 0.0039-0.0043 at 200, and in the fully interior neighbour bucket
#: 0.0019-0.0021 against 0.0074-0.0075 — over the 0.005 gate. Reporting the
#: dose response at 200 alone would measure a conditioning error that the
#: 50-step operating point does not have, and attribute it to the model.
POROSITY_DDIM_STEPS = (50, 200)


def porosity_global_cases(repo=None) -> list[CaseSpec]:
    """2 - the dose response, and one request the model was never trained for."""
    plies, pitch = layup_a(repo)
    out = []
    for target in (*POROSITY_TARGETS, OFF_MANIFOLD_TARGET):
        for steps in POROSITY_DDIM_STEPS:
            for seed in SEEDS:
                out.append(CaseSpec(
                    name=f"target{target:g}_ddim{steps}_seed{seed}",
                    assessment="porosity_global",
                    volume_shape=SHAPE_SMALL,
                    seed=seed,
                    layup=plies,
                    ply_thickness_vox=pitch,
                    target_phi=target,
                    ddim_steps=steps,
                    notes={
                        "layup": "A",
                        "off_manifold": target == OFF_MANIFOLD_TARGET,
                        "ddim_steps": steps,
                    },
                ))
    return out


def porosity_local_cases(repo=None) -> list[CaseSpec]:
    """3 - three painted fields on the 3x3x3 tile grid of a 192-cubed volume."""
    plies, pitch = layup_a(repo)
    out = []
    for fname, fn in FIELDS.items():
        for seed in SEEDS:
            for steps in POROSITY_DDIM_STEPS:
                out.append(CaseSpec(
                    name=f"{fname}_ddim{steps}_seed{seed}",
                    assessment="porosity_local",
                    volume_shape=SHAPE_SMALL,
                    seed=seed,
                    layup=plies,
                    ply_thickness_vox=pitch,
                    target_phi=TARGET_DEFAULT,
                    field_fn=fn,
                    ddim_steps=steps,
                    notes={"field": fname, "layup": "A", "ddim_steps": steps},
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


#: Sphere radius in voxels. 80 leaves a 16-voxel margin in a 192-cubed canvas
#: and 48 in a 256-cubed one, so the curved surface is fully interior in both
#: and never coincides with the canvas edge.
SPHERE_RADIUS_VOX = 80


def material_sphere(shape, radius: float = SPHERE_RADIUS_VOX) -> np.ndarray:
    """A centred solid sphere: material inside, air outside.

    Every requested geometry so far has been axis-aligned — boxes, a notch, a
    cylindrical hole through z — so every requested surface has been flat or
    normal to an axis. A sphere is the first CURVED request, and after pooling
    to the 4-voxel cells its rim is fractional everywhere rather than only in
    the two z bands of the inset box.

    It is deliberately off-manifold: the training coupons are plates, so the
    model has never seen a specimen shaped like this.
    """
    d, h, w = shape
    zz = (np.arange(d) - (d - 1) / 2.0)[:, None, None]
    yy = (np.arange(h) - (h - 1) / 2.0)[None, :, None]
    xx = (np.arange(w) - (w - 1) / 2.0)[None, None, :]
    if radius >= min(d, h, w) / 2.0:
        raise ValueError(
            f"a radius-{radius} sphere does not fit inside {shape} with a margin"
        )
    return (zz ** 2 + yy ** 2 + xx ** 2) <= radius ** 2


def real_global_phi(repo=None, seed: int = 0) -> float:
    """A global porosity target drawn from the REAL distribution.

    The T-E marginal is the measured distribution of local porosity in the test
    material, so a draw from it is a request the model could plausibly be given
    in use, rather than a round number chosen for the table. Falls back to the
    default target if the T-E artefact is missing, and the case records which.
    """
    try:
        from poregen.diffusion.porosity_field import (  # noqa: PLC0415
            DEFAULT_TE_RESULTS, load_sampler)
        sampler = load_sampler((Path(repo) if repo else repo_root()) / DEFAULT_TE_RESULTS)
        # bin_centres_global_phi ARE the observed global-porosity bins of the
        # real material, so drawing one is drawing a request the model could
        # actually be given, rather than a round number chosen for the table.
        bins = np.asarray(sampler["bin_centres_global_phi"], dtype=np.float64)
        bins = bins[(bins > 0.004) & (bins < 0.12)]
        if bins.size == 0:
            return float(TARGET_DEFAULT)
        return float(np.random.default_rng(seed).choice(bins))
    except Exception:                                   # noqa: BLE001
        return float(TARGET_DEFAULT)


def sphere_cases(repo=None) -> list[CaseSpec]:
    """A curved, off-manifold specimen. EXPLORATORY — no gate.

    The user asked to see what the model does with a shape it has never been
    shown. There is no pass/fail here on purpose: there is no real spherical
    coupon to compare against, so any threshold would be invented.

    One property of the request is worth stating because it is part of what is
    being probed: ``cond_dist6`` is computed from the specimen BOUNDING BOX, as
    it is for every case, so the six face distances describe a cube around the
    sphere and not the curved surface. The model is therefore given geometry
    information that is correct for a box and misleading for a sphere, and how
    much that matters is one of the things this shows.
    """
    plies, pitch = layup_a(repo)
    out = []
    for shape, tag in (((192, 192, 192), "192"), ((256, 256, 256), "256")):
        for steps, seeds in ((50, SEEDS), (200, SEEDS[:1])):
            if tag == "256" and steps == 200:
                continue                    # one 256 case is enough to see scale
            for seed in seeds:
                if tag == "256" and seed != SEEDS[0]:
                    continue
                target = real_global_phi(repo, seed=seed)
                out.append(CaseSpec(
                    name=f"sphere_{tag}_ddim{steps}_seed{seed}",
                    assessment="geometry",
                    volume_shape=shape, seed=seed,
                    layup=plies, ply_thickness_vox=pitch,
                    target_phi=target, ddim_steps=steps,
                    material_fn=material_sphere,
                    field_fn=partial(field_coherent, target=target),
                    notes={"layup": "A", "scale": tag, "ddim_steps": steps,
                           "request": "sphere", "radius_vox": SPHERE_RADIUS_VOX,
                           "exploratory": True,
                           "global_phi_source": "T-E marginal median (real distribution)",
                           "cond_dist6_note": (
                               "computed from the sphere's BOUNDING BOX, as for any "
                               "case; the six face distances describe a cube, not the "
                               "curved surface. Part of what this case probes.")},
                ))
    return out


#: Cubic canvas for the multi-chunk set. 384 = 6 tiles a side, so at the
#: PRODUCTION chunk size of 3 tiles the canvas is 2 chunks on EVERY axis, with
#: the planes at 192 — the same chunk size the sampler and the trainer's own
#: [6,3,3] sample use. The 1024x1024x192 cases cross chunk planes only in x and
#: y, because 192 is a single chunk deep.
MULTICHUNK_SHAPE = (384, 384, 384)
MULTICHUNK_SLAB = (192, 384, 384)
MULTICHUNK_CHUNK_TILES = (3, 3, 3)
#: Radius for the multi-chunk sphere: 160 voxels puts the curved surface across
#: the chunk planes rather than inside one chunk, with a 32-voxel margin.
MULTICHUNK_SPHERE_RADIUS = 160


def multichunk_cases(repo=None) -> list[CaseSpec]:
    """10 - assembly when the volume does not fit in one chunk, on ALL axes.

    Every existing large case is 1024x1024x192, which is a single chunk deep:
    the z axis never crosses a chunk plane. These do, on all three.

    A NOTE ON PHYSICS, not assembly. No specimen in the dataset is thicker than
    about 330 voxels, so a 384-voxel-deep volume is asking for material that
    does not exist. That is deliberate and it bounds what these cases can show:
    they test whether the ASSEMBLY holds across chunk planes in z, not whether
    the result is a physically plausible laminate. Nothing here should be read
    as evidence about thick-specimen microstructure.
    """
    plies, pitch = layup_a(repo)
    common = dict(assessment="multichunk", layup=plies, ply_thickness_vox=pitch,
                  chunk_tiles=MULTICHUNK_CHUNK_TILES)
    out = []
    for steps, seeds in ((50, SEEDS[:2]), (200, SEEDS[:1])):
        for seed in seeds:
            out.append(CaseSpec(
                name=f"box384_ddim{steps}_seed{seed}",
                volume_shape=MULTICHUNK_SHAPE, seed=seed,
                target_phi=TARGET_DEFAULT, ddim_steps=steps,
                field_fn=field_coherent,
                notes={"layup": "A", "request": "box", "scale": "384",
                       "ddim_steps": steps}, **common))
    out.append(CaseSpec(
        name="sphere384_ddim50_seed101",
        volume_shape=MULTICHUNK_SHAPE, seed=SEEDS[0],
        target_phi=TARGET_DEFAULT, ddim_steps=50,
        field_fn=field_coherent,
        material_fn=partial(material_sphere, radius=MULTICHUNK_SPHERE_RADIUS),
        notes={"layup": "A", "request": "sphere", "scale": "384", "ddim_steps": 50,
               "radius_vox": MULTICHUNK_SPHERE_RADIUS, "exploratory": True,
               "cond_dist6_note": "bounding box, not the curved surface"}, **common))
    tgt = real_surface_target(repo)
    out.append(CaseSpec(
        name="rough384_ddim50_seed101",
        volume_shape=MULTICHUNK_SLAB, seed=SEEDS[0],
        target_phi=TARGET_DEFAULT, ddim_steps=50,
        field_fn=field_coherent,
        material_fn=partial(material_rough_z, seed=SEEDS[0],
                            sa=tgt["sa_vox"], corr_len=tgt["correlation_length_vox"]),
        notes={"layup": "A", "request": "rough", "scale": "384x384x192",
               "ddim_steps": 50, "requested_sa_vox": tgt["sa_vox"],
               "requested_correlation_length_vox": tgt["correlation_length_vox"],
               "roughness_source": tgt["source"]}, **common))
    return out


#: Scales the four assembly arms are compared at, and the tag each carries.
#: The 384 cube crosses a chunk plane on every axis; the slab is the T-I
#: reader's production shape and crosses them in x and y only.
ASSEMBLY_MODES_SHAPES = ((MULTICHUNK_SHAPE, "384"), (SHAPE_LARGE, "1024"))
#: Held at 50 for every arm.  This assessment is about the ASSEMBLY, and a step
#: count that moved with the arm would put a sampler difference and a step-count
#: difference in the same column.
ASSEMBLY_MODES_DDIM = 50
#: Seeds per scale.  Three at 384 and two at 1024x1024x192, which is 3.5x the
#: voxels: the same cost trade the multi-chunk set already makes.  Every arm at
#: one scale runs the SAME seeds, which is what makes the arms comparable.
ASSEMBLY_MODES_SEEDS = {"384": SEEDS, "1024": SEEDS[:2]}
#: The chunk grid every arm is MEASURED on, whatever grid it was GENERATED on.
#: The arms have different chunk geometries by construction - that is the point
#: of the assessment - so each arm's own chunk period would put four different
#: sets of planes in one table. The production period (3 tiles = 192 voxels) is
#: the reference: for the hybrid arm it is also the generation grid, so the
#: comparison is "what the other arms do at the planes the production sampler
#: would have had to assemble across".
ASSEMBLY_MODES_REFERENCE_TILES = CHUNK_TILES
#: Deepest canvas the teacher-forced arm can be built for, in voxels.  Test
#: patches in the r08 store reach z0 = 128, so the deepest real block on the
#: 64-voxel tile grid is 3 tiles.  See :mod:`poregen.eval_v4.teacher`.
TEACHER_MAX_DEPTH_VOX = 192


def assembly_modes_cases(repo=None) -> list[CaseSpec]:
    """11 - how much of the quality is the hybrid sampler, and what is the ceiling?

    Four ways to assemble the SAME request from the SAME seeds.  Everything
    except the assembly is held: one uniform porosity target (so any porosity
    drift from chunk to chunk is a defect and not the request), layup A, 50 DDIM
    steps, the production window stride and decode stride.

    ``joint``
        one chunk over the whole volume and every neighbour UNKNOWN - the ldm05
        MultiDiffusion sampler.  No chunk planes exist, so what it scores at the
        reference planes is what the MEASUREMENT reads on a volume that was
        never assembled across them.
    ``autoregressive``
        ``chunk_tiles = (1, 1, 1)``: every tile is its own chunk, so each patch
        is denoised to completion against finished material.  The ldm05
        sequential sampler, and the arm with the most planes to get wrong.
    ``hybrid``
        ``chunk_tiles = (3, 3, 3)``, the production setting.
    ``teacher_forced``
        the production chunking with every neighbour replaced by the encoding of
        a REAL test volume at the same canvas position.  It is a control, not a
        sampler: it is the quality the hybrid would reach if the material it
        assembles against were perfect, so the gap between it and ``hybrid`` is
        what better neighbour handling could still buy.

    The teacher-forced arm runs at the slab only.  No specimen in the dataset is
    384 voxels thick, so there is no real material to teach with at that depth,
    and filling the missing depth by repeating a block would put a fake join
    exactly on a chunk plane - the one place this assessment measures.  The
    ceiling is therefore reported at 1024x1024x192 and is absent at 384, which
    the results and the findings both say rather than leaving a blank cell.
    """
    plies, pitch = layup_a(repo)
    out = []
    for shape, tag in ASSEMBLY_MODES_SHAPES:
        whole = tuple(s // TILE for s in shape)
        arms = [
            ("joint", whole, "unknown"),
            ("autoregressive", (1, 1, 1), "canvas"),
            ("hybrid", CHUNK_TILES, "canvas"),
        ]
        if shape[0] <= TEACHER_MAX_DEPTH_VOX:
            arms.append(("teacher_forced", CHUNK_TILES, "reference"))
        for arm, chunk_tiles, mode in arms:
            for seed in ASSEMBLY_MODES_SEEDS[tag]:
                out.append(CaseSpec(
                    name=f"{arm}_{tag}_seed{seed}",
                    assessment="assembly_modes",
                    volume_shape=shape,
                    seed=seed,
                    layup=plies,
                    ply_thickness_vox=pitch,
                    target_phi=TARGET_DEFAULT,
                    ddim_steps=ASSEMBLY_MODES_DDIM,
                    chunk_tiles=chunk_tiles,
                    neighbour_mode=mode,
                    notes={"layup": "A", "arm": arm, "scale": tag,
                           "neighbour_mode": mode,
                           "ddim_steps": ASSEMBLY_MODES_DDIM,
                           "reference_chunk_tiles": list(ASSEMBLY_MODES_REFERENCE_TILES)},
                ))
    return out


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
            notes={"layup": "A", "request": "notch_hole",
                   "features": "64-voxel notch + 200-voxel hole"},
        )
        for seed in SEEDS
    ] + sphere_cases(repo)


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


# ---------------------------------------------------------------------------
# 12 - stress geometry (EXPLORATORY, off the gates)
# ---------------------------------------------------------------------------

#: Both step counts for every geometry.  A shape the model has never seen is
#: exactly where the step count might decide whether it holds at all, so
#: running one and inferring the other would be guessing about the case that
#: matters most.
#: One seed: this is a boundary probe, not a distribution.
STRESS_SEED = 101

STRESS_DDIM = (50, 200)

#: (name, shape, material_fn, field_fn, note).  Shapes are (z, y, x) and every
#: axis is a whole number of 64-voxel tiles.
STRESS_GEOMETRIES: tuple = (
    ("tube", (448, 448, 1024), SG.material_tube, None,
     "hollow cylinder, axis along x; outer r 200, wall 120 vox (5 mm / 3 mm)"),
    ("lbracket", (640, 640, 192), SG.material_l_bracket, None,
     "two 192-thick legs, 512 long, inner fillet r 96, outer round r 288"),
    # x is 1024 and not 384 so the T-I readers fit: the ply drop-off is the one
    # stress request where a ply angle is physically meaningful, and the readers
    # take a centred 1024x1024 in-plane window that a 384 axis cannot hold.
    # 2.67x the voxels of the 384 canvas, for that one reading.
    ("taper", (192, 1024, 1024), SG.material_taper, None,
     "thickness 192 -> 80 vox linearly along y; x widened to 1024 so the T-I "
     "layup readers fit"),
    ("gradient_spots", (192, 1024, 1024), None, SG.field_ramp_and_spots,
     "flat plate; requested phi ramps 0.005 -> 0.08 along y plus two "
     "sigma-96 spots peaking at 0.10"),
    ("cube1024", (1024, 1024, 1024), None, None,
     "full material at phi 0.03; production chunking on all three axes"),
    ("gyroid", (512, 512, 512), SG.material_gyroid, None,
     "triply periodic, 80-vox struts at period 512 (48 % material); no flat "
     "face and no outer surface"),
    ("hollow_sphere", (512, 512, 512), SG.material_hollow_sphere, None,
     "outer r 224, shell 80 vox; the interior is AIR, which no coupon has"),
    ("two_coupons", (192, 1024, 1024), SG.material_two_coupons, None,
     "two 448-wide plates with a 128-vox air gap along x: one request, two "
     "disconnected bodies"),
    ("letters", (192, 1024, 1024), SG.material_letters, None,
     "'PoreGen' cut through the plate as air, font size 256 (cap height "
     "~194 vox); pure controllability"),
)


def stress_geometry_cases(repo=None) -> list[CaseSpec]:
    """12 - shapes the training material never contained.  EXPLORATORY.

    Off the gates by construction: a model asked for a tube when every training
    coupon was a plate may fail in ways that say nothing about the material it
    was trained to make.  What the assessment establishes is the boundary of
    the specimen-envelope conditioning — which requests it honours at all.
    """
    plies, pitch = layup_a(repo)
    out = []
    for name, shape, mat_fn, field_fn, note in STRESS_GEOMETRIES:
        for ddim in STRESS_DDIM:
            out.append(CaseSpec(
                name=f"{name}_ddim{ddim}",
                assessment="stress_geometry",
                volume_shape=shape,
                seed=STRESS_SEED,
                layup=plies,
                ply_thickness_vox=pitch,
                target_phi=TARGET_DEFAULT,
                ddim_steps=ddim,
                material_fn=mat_fn,
                field_fn=field_fn,
                notes={"layup": "A", "request": name, "geometry": note,
                       "exploratory": True,
                       "off_gates_because":
                           "the request is a shape no training coupon "
                           "resembles; a failure here is not a defect of the "
                           "kind the gated assessments report"},
            ))
    return out


# ---------------------------------------------------------------------------
# 13 - conditioning out of distribution (EXPLORATORY, off the gates)
# ---------------------------------------------------------------------------

#: A coherent field at an ISOTROPIC correlation length, which is itself off the
#: manifold: the measured lengths are (79, 414, 901) voxels in (z, y, x),
#: because a laminate is not isotropic. The request here is the same length on
#: all three axes, so the case asks two things at once and the reporter says so.
def field_coherent_iso(grid, seed, length: float, target: float = TARGET_DEFAULT):
    """The production coherent field with every correlation length replaced."""
    from poregen.diffusion.porosity_field import (  # noqa: PLC0415
        DEFAULT_TE_RESULTS,
        build_porosity_field,
        load_sampler,
    )

    root = repo_root()
    return build_porosity_field(
        grid_shape=tuple(grid),
        target=float(target),
        sampler=load_sampler(root / DEFAULT_TE_RESULTS),
        corr_lengths_voxels=(float(length),) * 3,
        stride_voxels=TILE,
        seed=int(seed),
    ).astype(np.float32)


#: Four stacking sequences built ONLY from the trained angle set
#: {0, 45, -45, 90}. No new angle appears anywhere: what is out of distribution
#: is the ORDER, not the orientations, so a failure cannot be blamed on an
#: angle the model never saw.
OOD_SEQUENCES = {
    "seq_interleaved": (0, 90, 45, -45, 0, 90, 45, -45, 0, 90),
    "seq_blocked":     (45, -45, 45, -45, 0, 0, 90, 90, 45, -45),
    #: quasi-isotropic [0/45/90/-45]s
    "seq_quasi_iso":   (0, 45, 90, -45, -45, 90, 45, 0),
    #: cross-ply [0/90]s, repeated cyclically to fill the depth
    "seq_crossply":    (0, 90, 90, 0),
}

#: Ply pitches, in voxels. Trained: 19.6 (sequence A, 74 of 78 training
#: volumes) and 10.0 (sequence B, 0.25 mm). 8 is just BELOW the thinner trained
#: pitch and 32 is well above the thicker one, so the pair brackets the
#: training range rather than sitting on one side of it.
#: At 192 deep these give 24 and 6 ply blocks - which is the same request as
#: "6 thick plies and 24 thin plies filling the depth", so that is one set of
#: two cases and not two sets.
OOD_PITCH_VOX = (8.0, 32.0)

#: (requested phi, is the clamp lifted).  The clamp holds a request inside
#: [POR_MIN, POR_MAX] = [0.002, 0.107] before conditioning.
#:
#: IT IS LIFTED FOR ALL FOUR EXTREMES, NOT ONLY THE HIGH PAIR.  cond_por is
#: log(phi + 1e-3) standardised, and with the clamp ON the three low requests
#: are not three requests: 0.000, 0.001 and 0.002 all condition at -0.948 sd,
#: so the row would be one case run nine times. Lifted, they separate cleanly
#: to -1.814 / -1.268 / -0.948 sd.
#:
#: 0.002 keeps the clamp because there the clamp is a no-op, which makes it the
#: in-distribution anchor the other four are read against.
#:
#: The clamp is far milder at the top: 0.150 conditions at +0.264 sd beyond
#: POR_MAX and 0.200 at +0.489. So the high pair probes a shorter distance off
#: the manifold than the low pair does, and the two must not be read as
#: symmetric extremes.
OOD_POROSITY = ((0.000, False), (0.001, False), (0.002, True),
                (0.150, False), (0.200, False))

#: Isotropic correlation lengths, voxels. Production is (79, 414, 901).
OOD_CORR_VOX = (16.0, 64.0, 512.0)

#: 192 cubed: one production chunk, so the porosity extremes measure the
#: conditioning and not the assembly. A failure at 0.20 that turned out to be a
#: chunk-frontier artefact would say nothing about the request.
SHAPE_CHUNK = (192, 192, 192)

OOD_DDIM = 50

_OOD_OFF_GATE = (
    "the request is outside the conditioning distribution the model was "
    "trained on; a failure here is the boundary of the conditioning and not a "
    "defect of the kind the gated assessments report"
)


def ood_conditioning_cases(repo=None) -> list[CaseSpec]:
    """13 - conditioning asked for what the training set does not contain.

    EXPLORATORY. Four groups, and each asks about ONE axis of the conditioning
    with the others held at their trained values:

      1. stacking ORDER, with every angle inside the trained set
      2. ply PITCH, above and below the two trained pitches
      3. requested POROSITY, below and above the clamped training range
      4. the field's CORRELATION LENGTH

    Pore size and shape are NOT measured here; the author analyses those with
    an external program.
    """
    plies_a, pitch_a = layup_a(repo)
    out: list[CaseSpec] = []

    def add(name, group, **kw):
        notes = {"group": group, "exploratory": True,
                 "off_gates_because": _OOD_OFF_GATE, **kw.pop("notes", {})}
        out.append(CaseSpec(name=name, assessment="ood_conditioning",
                            ddim_steps=OOD_DDIM, notes=notes, **kw))

    # -- 1. stacking sequences the training set does not contain -------------
    for seq_name, plies in OOD_SEQUENCES.items():
        for seed in SEEDS:
            add(f"{seq_name}_seed{seed}", "sequence",
                volume_shape=SHAPE_LARGE, seed=seed, layup=plies,
                ply_thickness_vox=pitch_a, target_phi=TARGET_DEFAULT,
                notes={"sequence": seq_name, "n_plies_in_request": len(plies),
                       "angles": sorted(set(plies))})

    # -- 2. ply pitch, either side of the two trained ones -------------------
    for pitch in OOD_PITCH_VOX:
        for seed in SEEDS:
            add(f"pitch{pitch:g}_seed{seed}", "pitch",
                volume_shape=SHAPE_LARGE, seed=seed, layup=plies_a,
                ply_thickness_vox=pitch, target_phi=TARGET_DEFAULT,
                notes={"pitch_vox": pitch,
                       "n_ply_blocks": int(np.ceil(SHAPE_LARGE[0] / pitch)),
                       "trained_pitches_vox": [10.0, pitch_a]})

    # -- 3. requested porosity, below and above the clamped range ------------
    for phi, clamped in OOD_POROSITY:
        for seed in SEEDS:
            add(f"phi{phi:g}_seed{seed}", "porosity",
                volume_shape=SHAPE_CHUNK, seed=seed, layup=plies_a,
                ply_thickness_vox=pitch_a, target_phi=phi,
                clamp_porosity=clamped,
                notes={"requested_phi": phi, "clamp_lifted": not clamped,
                       "in_training_range": POR_MIN <= phi <= POR_MAX})

    # -- 4. the field's correlation length -----------------------------------
    for length in OOD_CORR_VOX:
        add(f"corr{length:g}", "correlation",
            volume_shape=SHAPE_LARGE, seed=SEEDS[0], layup=plies_a,
            ply_thickness_vox=pitch_a, target_phi=TARGET_DEFAULT,
            field_fn=partial(field_coherent_iso, length=length),
            notes={"corr_length_vox": length, "isotropic": True,
                   "production_corr_vox": [79.4, 413.6, 900.9]})
    return out


ASSESSMENTS: dict[str, Callable[..., list[CaseSpec]]] = {
    "sampler": sampler_cases,
    "porosity_global": porosity_global_cases,
    "porosity_local": porosity_local_cases,
    "cfg": cfg_cases,
    "layup": layup_cases,
    "assembly": assembly_cases,
    "geometry": geometry_cases,
    "surface": surface_cases,
    "multichunk": multichunk_cases,
    "microstructure": microstructure_cases,
    "assembly_modes": assembly_modes_cases,
    "stress_geometry": stress_geometry_cases,
    "ood_conditioning": ood_conditioning_cases,
}

#: Assessments whose measure step also reads another assessment's volumes.
#: ``microstructure`` reads the matched real crops the ``real-floor`` stage
#: writes: without them it has a number and no floor to read it against, which
#: for a distribution distance is no measurement at all.
BORROWS = {
    "assembly": ("sampler",),
    "microstructure": ("real_floor",),
    # Every per-chunk number the arms produce is a ratio or a fraction whose
    # value on real material is not 0 or 1; without the floor a seam ratio of
    # 1.2 cannot be called large or small.
    "assembly_modes": ("real_floor",),
}

#: Assessments that generate NOTHING.  ``field_stats`` re-measures the coherent
#: field out of volumes ``porosity_local`` and ``multichunk`` already wrote and
#: out of the real crops, so it has a measure step and a report but no cases and
#: no ``generate`` subcommand.  Asking the GPU for the same volumes a second
#: time would not make the answer better.
MEASURE_ONLY = ("field_stats",)


def build_cases(assessment: str, repo=None) -> list[CaseSpec]:
    if assessment not in ASSESSMENTS:
        raise KeyError(
            f"unknown assessment {assessment!r}; choose from {sorted(ASSESSMENTS)}"
        )
    return ASSESSMENTS[assessment](repo)

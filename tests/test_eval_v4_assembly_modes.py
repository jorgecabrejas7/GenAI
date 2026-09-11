"""The ``assembly_modes`` assessment, on volumes whose answer is put there by hand.

Nothing here loads a model or a GPU.  Every test builds an array with a KNOWN
discontinuity at a KNOWN plane, so a failure names the part of the measurement
that is wrong:

* is a step at a chunk plane attributed to the right CHUNK INDEX;
* is it attributed to the right PLANE FAMILY - a chunk plane is not a tile
  plane, and a tile plane is not a chunk plane;
* does the per-chunk porosity read the block it says it reads;
* does the neighbour mode reach the denoiser as the availability code the
  arm claims (a spy model records what it was handed).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from poregen.diffusion.conditioning import NB_EXISTS, NB_OOB, NB_UNKNOWN
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator
from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import (
    ASSEMBLY_MODES_REFERENCE_TILES,
    TEACHER_MAX_DEPTH_VOX,
    build_cases,
)
from poregen.eval_v4.io import LABEL_MATERIAL, LABEL_PORE, TILE
from poregen.eval_v4.manifest import Manifest

COMMIT = "0" * 40
PERIOD = tuple(TILE * c for c in ASSEMBLY_MODES_REFERENCE_TILES)   # (192, 192, 192)


def manifest_for(shape) -> Manifest:
    return Manifest(
        assessment="assembly_modes", case="c0", volume_shape=shape,
        git_commit=COMMIT, sampler="hybrid_chunked", model_run="runs/ldm/x",
        checkpoint_step=1, weights="ema", ddim_steps=50, chunk_tiles=(3, 3, 3),
        window_stride=32, decode="overlapped", decode_overlap=32,
        s_por=1.0, s_nb=1.0, seed=101, objective="v", cfg_rescale=0.0,
        requested_global_phi=0.03, requested_material="full",
    )


def textured(shape, seed=0, level=120.0, amp=6.0) -> np.ndarray:
    """A volume with ordinary slice-to-slice variation and no seam anywhere.

    The baseline every ratio is divided by has to be non-zero, or a ratio is
    0/0; this is the "ordinary internal texture change" the metric compares a
    seam against.
    """
    rng = np.random.default_rng(seed)
    return np.clip(level + amp * rng.standard_normal(shape), 0, 255).astype(np.uint8)


def add_step(vol: np.ndarray, axis: int, plane: int, size: float) -> np.ndarray:
    """Raise everything at and beyond ``plane`` on ``axis`` by ``size``.

    A pure step: the only plane whose slice-to-slice difference changes is the
    one between ``plane - 1`` and ``plane``.
    """
    out = vol.astype(np.float32)
    sl = [slice(None)] * 3
    sl[axis] = slice(plane, None)
    out[tuple(sl)] += size
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The reference chunk grid
# ---------------------------------------------------------------------------

def test_chunk_blocks_are_raster_order_and_own_their_lower_planes():
    blocks = M.chunk_blocks((384, 384, 384), PERIOD)
    assert len(blocks) == 8
    assert [b["origin"] for b in blocks] == [
        [0, 0, 0], [0, 0, 192], [0, 192, 0], [0, 192, 192],
        [192, 0, 0], [192, 0, 192], [192, 192, 0], [192, 192, 192],
    ]
    # Chunk 0 met nothing: it owns no plane.  Every other plane has exactly
    # one owner - the block above it.
    assert blocks[0]["lower_plane_axes"] == []
    assert blocks[1]["lower_plane_axes"] == [2]
    assert blocks[3]["lower_plane_axes"] == [1, 2]
    assert blocks[7]["lower_plane_axes"] == [0, 1, 2]
    owned = sum(len(b["lower_plane_axes"]) for b in blocks)
    assert owned == 12          # 3 axes x 4 blocks on the far side of each plane


def test_chunk_blocks_cut_a_short_last_block_like_the_sampler():
    # 1024 voxels is 16 tiles; chunks of 3 tiles leave a 1-tile remainder.
    blocks = M.chunk_blocks((192, 1024, 1024), PERIOD)
    xs = sorted({b["origin"][2] for b in blocks})
    assert xs == [0, 192, 384, 576, 768, 960]
    last = [b for b in blocks if b["origin"][2] == 960][0]
    assert last["shape"][2] == 64
    assert {b["origin"][0] for b in blocks} == {0}      # one chunk deep


# ---------------------------------------------------------------------------
# Attribution: which chunk, and which plane family
# ---------------------------------------------------------------------------

#: Size of the planted step, in u8 grey levels, and the texture it sits in.
#: A step of 40 against a texture whose slice-to-slice MAD is about 6.8 gives a
#: seam-to-interior ratio near 6 on the axis it is on: big enough to be
#: unmistakable, small enough that the clip at 0/255 never fires.
STEP_U8 = 40.0


@pytest.mark.parametrize(
    "axis, plane, expect_index, name",
    [
        (2, 192, 1, "x"),      # the x plane belongs to the block at x = 192
        (1, 192, 2, "y"),      # the y plane, to the block at y = 192
        (0, 192, 4, "z"),      # the z plane, to the block at z = 192
    ],
)
def test_a_step_at_a_chunk_plane_lands_on_the_right_chunk(axis, plane, expect_index, name):
    shape = (384, 384, 384)
    vol = add_step(textured(shape), axis, plane, STEP_U8)
    grey = vol.astype(np.float32) / 255.0
    blocks = M.chunk_blocks(shape, PERIOD)
    profiles = [M.chunk_seam_profile(grey, b) for b in blocks]

    hit = profiles[expect_index]
    base = hit["interior_mad"]
    # Right chunk, right axis, right FAMILY: the chunk plane moved and the
    # tile planes of the same axis did not.
    assert hit["per_axis"][name]["chunk_mad"] > 4.0 * base
    assert hit["per_axis"][name]["tile_mad"] == pytest.approx(base, rel=0.15)
    assert hit["chunk_plane_ratio"] > 4.0

    for i, (b, p) in enumerate(zip(blocks, profiles)):
        owns = axis in b["lower_plane_axes"] and b["origin"][axis] == plane
        mad = p["per_axis"][name]["chunk_mad"]
        if owns:
            assert mad > 4.0 * p["interior_mad"], i
        elif mad is not None:
            # a lower plane on that axis somewhere else - there is none in a
            # 2-chunk volume, but the claim is the general one
            assert mad == pytest.approx(p["interior_mad"], rel=0.15), i


def test_a_step_at_a_tile_plane_lands_on_the_tile_family_of_its_own_chunk():
    shape = (384, 384, 384)
    # 256 is a multiple of 64 and not of 192: a window plane strictly inside
    # the chunk that starts at 192.
    vol = add_step(textured(shape), 2, 256, STEP_U8)
    grey = vol.astype(np.float32) / 255.0
    blocks = M.chunk_blocks(shape, PERIOD)
    profiles = [M.chunk_seam_profile(grey, b) for b in blocks]

    for b, p in zip(blocks, profiles):
        base = p["interior_mad"]
        x = p["per_axis"]["x"]
        if b["origin"][2] == 192:
            # one of the two x tile planes of this block carries the step
            assert x["tile_mad"] > 2.0 * base
            # and it is NOT charged to the chunk family, whose plane is x = 192
            assert x["chunk_mad"] == pytest.approx(base, rel=0.15)
        else:
            assert x["tile_mad"] == pytest.approx(base, rel=0.15)


def test_the_interior_baseline_excludes_tile_planes():
    """A volume whose only defect is AT the tile planes must not hide it in its
    own baseline: if tile planes entered the interior mean they would inflate
    the divisor by exactly the amount the numerator went up."""
    shape = (192, 192, 192)
    vol = textured(shape).astype(np.float32)
    for plane in (64, 128):
        vol[:, :, plane:] += 20.0
    grey = np.clip(vol, 0, 255).astype(np.uint8).astype(np.float32) / 255.0
    block = M.chunk_blocks(shape, PERIOD)[0]
    p = M.chunk_seam_profile(grey, block)

    assert p["n_chunk_planes"] == 0 and p["chunk_plane_ratio"] is None
    assert p["per_axis"]["x"]["tile_mad"] > 2.0 * p["interior_mad"]
    # 191 planes per axis, of which 2 per axis are tile planes.
    assert p["n_tile_planes"] == 6
    assert p["n_interior_planes"] == 3 * (191 - 2)
    # The baseline is the y and z texture, untouched by the x steps.
    assert p["per_axis"]["y"]["interior_mad"] == pytest.approx(
        p["per_axis"]["z"]["interior_mad"], rel=0.05)


def test_the_seam_size_is_recovered_as_a_mad():
    """The chunk-plane MAD of a pure step IS the step, to the noise the texture
    adds - so the metric is calibrated and not merely monotone."""
    shape = (384, 384, 384)
    vol = add_step(textured(shape, amp=0.0), 2, 192, STEP_U8)
    grey = vol.astype(np.float32) / 255.0
    block = [b for b in M.chunk_blocks(shape, PERIOD) if b["origin"] == [0, 0, 192]][0]
    p = M.chunk_seam_profile(grey, block)
    assert p["per_axis"]["x"]["chunk_mad"] == pytest.approx(STEP_U8 / 255.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Porosity per chunk
# ---------------------------------------------------------------------------

def test_per_chunk_porosity_reads_the_block_it_names():
    shape = (384, 384, 192)
    label = np.full(shape, LABEL_MATERIAL, np.uint8)
    # A known porosity in exactly one chunk: the block at z=192, y=192.
    label[192:384, 192:384, :][:10] = LABEL_PORE
    material = np.ones(shape, bool)
    blocks = M.chunk_blocks(shape, PERIOD)
    expect = 10.0 / 192.0
    for b in blocks:
        got = M.chunk_porosity(label, material, b)["phi_pore"]
        if b["origin"][:2] == [192, 192]:
            assert got == pytest.approx(expect, rel=1e-6)
        else:
            assert got == 0.0


def test_chunk_profile_runs_end_to_end_and_keeps_generation_order():
    shape = (384, 384, 192)
    xct = add_step(textured(shape), 2, 0, 0.0)     # no step; just texture
    label = np.where(
        np.random.default_rng(1).random(shape) < 0.02, LABEL_PORE, LABEL_MATERIAL
    ).astype(np.uint8)
    material = np.ones(shape, bool)
    rows = M.chunk_profile(
        xct, label, material, manifest=manifest_for(shape), period=PERIOD,
        pore_logit=None, with_s2=False,
    )
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))
    assert rows[0]["origin"] == [0, 0, 0]
    assert "pore" not in rows[0]          # no pore logit was given, none invented
    assert all(r["porosity"]["phi_pore"] is not None for r in rows)


def test_s2_across_a_destroyed_join_is_further_from_the_inside_curve():
    """Teacher forcing is meant to show up here: a join where the structure
    stops dead must read a LARGER across-vs-inside S2 distance than a join
    where the same structure continues."""
    shape = (192, 192, 384)
    rng = np.random.default_rng(3)
    # Correlated pore structure: blobs, not salt and pepper, so S2 has a shape.
    field = rng.random(shape)
    pore = field < 0.03
    label = np.where(pore, LABEL_PORE, LABEL_MATERIAL).astype(np.uint8)
    material = np.ones(shape, bool)

    blocks = M.chunk_blocks(shape, PERIOD)
    joined = [b for b in blocks if b["origin"][2] == 192][0]
    good = M.chunk_s2(label, material, joined, window=128, min_material=0.99)

    # Now empty a slab on one side of the plane: the structure stops at it.
    broken = label.copy()
    broken[:, :, 160:192] = LABEL_MATERIAL
    bad = M.chunk_s2(broken, material, joined, window=128, min_material=0.99)

    assert good["s2_relative_distance"] is not None
    assert bad["s2_relative_distance"] > good["s2_relative_distance"]


# ---------------------------------------------------------------------------
# The four arms, as cases
# ---------------------------------------------------------------------------

def test_the_four_arms_are_built_with_the_same_request_and_the_same_seeds():
    specs = build_cases("assembly_modes")
    by_scale: dict[str, dict[str, list]] = {}
    for s in specs:
        by_scale.setdefault(s.notes["scale"], {}).setdefault(s.notes["arm"], []).append(s)

    assert set(by_scale) == {"384", "1024"}
    assert set(by_scale["384"]) == {"joint", "autoregressive", "hybrid"}
    assert set(by_scale["1024"]) == {
        "joint", "autoregressive", "hybrid", "teacher_forced"}

    for scale, arms in by_scale.items():
        seeds = {arm: sorted(s.seed for s in v) for arm, v in arms.items()}
        assert len({tuple(v) for v in seeds.values()}) == 1, seeds
        shapes = {s.volume_shape for v in arms.values() for s in v}
        assert len(shapes) == 1
        # The request itself is identical across arms: same target, same steps,
        # no painted field, same layup.
        for v in arms.values():
            for s in v:
                assert s.field_fn is None and s.material_fn is None
                assert s.target_phi == specs[0].target_phi
                assert s.ddim_steps == specs[0].ddim_steps

    modes = {s.notes["arm"]: s.neighbour_mode for s in specs}
    assert modes == {"joint": "unknown", "autoregressive": "canvas",
                     "hybrid": "canvas", "teacher_forced": "reference"}
    chunks = {s.notes["arm"]: s.chunk_tiles for s in specs if s.notes["scale"] == "384"}
    assert chunks["joint"] == (6, 6, 6)          # one chunk over the whole volume
    assert chunks["autoregressive"] == (1, 1, 1)
    assert chunks["hybrid"] == ASSEMBLY_MODES_REFERENCE_TILES


def test_no_teacher_forced_case_is_deeper_than_the_real_material():
    for s in build_cases("assembly_modes"):
        if s.neighbour_mode == "reference":
            assert s.volume_shape[0] <= TEACHER_MAX_DEPTH_VOX


# ---------------------------------------------------------------------------
# The neighbour modes, through the sampler, on a spy denoiser
# ---------------------------------------------------------------------------

P, LAT, DS, C = 64, 16, 4, 2


class _Cfg:
    z_channels = C


class _SpyModel(nn.Module):
    """Records the availability codes and neighbour content it was handed."""

    cfg = _Cfg()

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, z_t, t, nb_latents, nb_avail, nb_t, cond_por, cond_depth,
                cond_dist6, cond_orient, cond_material, drop_por=None):
        self.calls.append({
            "nb_avail": nb_avail.detach().clone(),
            "nb_latents": nb_latents.detach().clone(),
            "nb_t": nb_t.detach().clone(),
        })
        return torch.zeros_like(z_t)


class _ConstVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Identity()

    def decoder(self, z):
        return z

    def xct_head(self, h):
        return torch.full((h.shape[0], 1, P, P, P), 0.4, device=h.device)

    def class_head(self, h):
        out = torch.zeros((h.shape[0], 3, P, P, P), device=h.device)
        out[:, 0] = 1.0
        return out


def _generator(neighbour_mode, chunk_tiles, reference=None, n_steps=2):
    model = _SpyModel()
    schedule = DDPMSchedule(T=20, objective="eps", device=torch.device("cpu"))
    sampler = DDIMSampler(model, schedule, torch.device("cpu"), n_steps=n_steps)
    gen = VolumeGenerator(
        sampler=sampler, vae=_ConstVAE(), device=torch.device("cpu"),
        patch_size=P, latent_size=LAT,
        por_log_stats=(-4.6, 1.27),
        theta_deg=np.zeros(512, np.float32),
        chunk_tiles=chunk_tiles, window_stride=32, decode_stride=64,
        neighbour_mode=neighbour_mode, reference_latents=reference,
    )
    return gen, model


def _run(gen, shape=(128, 128, 128)):
    mm = tuple(s * gen.voxel_size_mm for s in shape)
    with torch.no_grad():
        gen.generate(volume_size_mm=mm, target_porosity=0.03, window_batch=8,
                     autocast_dtype=torch.bfloat16, seed=7)


def test_unknown_mode_never_hands_the_model_an_existing_neighbour():
    gen, model = _generator("unknown", (2, 2, 2))
    _run(gen)
    codes = torch.cat([c["nb_avail"].reshape(-1) for c in model.calls])
    assert set(codes.tolist()) <= {NB_OOB, NB_UNKNOWN}
    assert (codes == NB_UNKNOWN).any()
    # The CFG null carries no content and no noise level.
    for c in model.calls:
        assert float(c["nb_latents"].abs().max()) == 0.0
        assert int(c["nb_t"].abs().max()) == 0


def test_canvas_mode_does_hand_the_model_existing_neighbours():
    gen, model = _generator("canvas", (2, 2, 2))
    _run(gen)
    codes = torch.cat([c["nb_avail"].reshape(-1) for c in model.calls])
    assert (codes == NB_EXISTS).any()


def test_reference_mode_makes_every_in_bounds_neighbour_exist():
    cells = (32, 32, 32)
    ref = torch.zeros(C, *cells)
    gen, model = _generator("reference", (2, 2, 2), reference=ref)
    _run(gen)
    codes = torch.cat([c["nb_avail"].reshape(-1) for c in model.calls])
    # The only non-EXISTS code left is the canvas edge.
    assert set(codes.tolist()) <= {NB_OOB, NB_EXISTS}
    assert (codes == NB_EXISTS).any()


def test_a_reference_canvas_of_the_wrong_shape_is_refused():
    gen, _ = _generator("reference", (2, 2, 2), reference=torch.zeros(C, 8, 8, 8))
    with pytest.raises(ValueError, match="reference_latents has shape"):
        _run(gen)


def test_a_reference_canvas_without_the_mode_is_refused():
    with pytest.raises(ValueError, match="go together"):
        _generator("canvas", (2, 2, 2), reference=torch.zeros(C, 32, 32, 32))


def test_an_unknown_neighbour_mode_is_refused():
    with pytest.raises(ValueError, match="neighbour_mode must be one of"):
        _generator("teacher", (2, 2, 2))


# ---------------------------------------------------------------------------
# The teacher-forced reference block, on an index written by hand
# ---------------------------------------------------------------------------

def _fake_index(origins, volume_id="v0"):
    """The four arrays :mod:`teacher` reads out of an index.parquet."""
    a = np.asarray(origins, np.int64)
    return (np.array([volume_id] * len(a), dtype=object), a[:, 0], a[:, 1], a[:, 2])


def _lattice(shape_vox, start=(0, 0, 0), stride=32):
    return [(start[0] + z, start[1] + y, start[2] + x)
            for z in range(0, shape_vox[0], stride)
            for y in range(0, shape_vox[1], stride)
            for x in range(0, shape_vox[2], stride)]


def test_all_true_block_finds_the_lowest_origin():
    from poregen.eval_v4.teacher import _all_true_block

    occ = np.zeros((4, 4, 4), bool)
    occ[1:3, 1:3, 1:3] = True
    assert _all_true_block(occ, (2, 2, 2)) == (1, 1, 1)
    assert _all_true_block(occ, (3, 3, 3)) is None
    assert _all_true_block(np.zeros((1, 1, 1), bool), (2, 2, 2)) is None


def test_the_reference_block_is_found_on_the_stride_64_sublattice():
    from poregen.eval_v4.teacher import find_reference_block

    index = _fake_index(_lattice((192, 192, 192)))
    got = find_reference_block(index, (128, 128, 128), split="test", store_root="x")
    assert got["origin_zyx"] == [0, 0, 0]
    assert got["tiles"] == [2, 2, 2]
    assert got["volume_id"] == "v0"


def test_the_reference_block_search_covers_the_offset_lattice_phase():
    """A volume whose 64-grid starts at 32 holds blocks the 0-phase misses.

    Only the odd half of the 32-voxel lattice is populated here, so a search
    that tried the 0 phase alone would report that nothing fits.
    """
    from poregen.eval_v4.teacher import find_reference_block

    index = _fake_index(_lattice((160, 160, 160), start=(32, 32, 32), stride=64))
    got = find_reference_block(index, (128, 128, 128), split="test", store_root="x")
    assert got["origin_zyx"] == [32, 32, 32]


def test_a_shape_no_real_volume_holds_is_refused_not_faked():
    from poregen.eval_v4.teacher import find_reference_block

    # 192 voxels of real depth: a 384-deep canvas has nothing to teach with.
    index = _fake_index(_lattice((192, 512, 512)))
    with pytest.raises(ValueError, match="fake join on a chunk plane"):
        find_reference_block(index, (384, 384, 384), split="test", store_root="x")


def test_a_reference_shape_off_the_tile_grid_is_refused():
    from poregen.eval_v4.teacher import find_reference_block

    with pytest.raises(ValueError, match="whole number of tiles"):
        find_reference_block(_fake_index(_lattice((192, 192, 192))), (100, 128, 128),
                    split="test", store_root="x")


# ---------------------------------------------------------------------------
# measure -> results.json -> findings.md, on volumes written by hand
# ---------------------------------------------------------------------------

E2E_SHAPE = (128, 256, 256)      # 1 x 2 x 2 reference chunks, a short last one


def _write_case(root, assessment, case, *, arm, scale, seed, chunk_tiles,
                sampler="hybrid_chunked", step_axis=None, step_plane=None):
    from poregen.eval_v4.io import case_dir as case_path, save_case

    rng = np.random.default_rng(seed)
    xct = textured(E2E_SHAPE, seed=seed)
    if step_axis is not None:
        xct = add_step(xct, step_axis, step_plane, STEP_U8)
    label = np.where(rng.random(E2E_SHAPE) < 0.03, LABEL_PORE, LABEL_MATERIAL).astype(np.uint8)
    common = dict(
        assessment=assessment, case=case, volume_shape=E2E_SHAPE, git_commit=COMMIT,
        sampler=sampler, chunk_tiles=chunk_tiles, window_stride=32,
        decode="overlapped", decode_overlap=32, requested_material="full",
    )
    if sampler == "hybrid_chunked":
        common.update(
            model_run="runs/ldm/x", checkpoint_step=1, weights="ema", ddim_steps=50,
            s_por=1.0, s_nb=1.0, seed=seed, objective="v", cfg_rescale=0.0,
            requested_global_phi=0.03, wall_time_s=1.0,
            notes={"arm": arm, "scale": scale, "neighbour_mode": "canvas"},
        )
    else:
        common.update(notes={"shape_tag": scale})
    save_case(case_path(root, assessment, case), Manifest(**common), xct, label)


def test_measure_and_report_run_end_to_end(tmp_path):
    """An assessment with no reporter is silently dropped from the findings, so
    the reporter is exercised here and not only registered."""
    from poregen.eval_v4.measure import measure
    from poregen.eval_v4.report import report_one

    for seed in (101, 202):
        # The hybrid arm is given a real step at its own chunk plane; the joint
        # arm is not.  The findings must be able to tell them apart.
        _write_case(tmp_path, "assembly_modes", f"hybrid_128_seed{seed}",
                    arm="hybrid", scale="128", seed=seed, chunk_tiles=(3, 3, 3),
                    step_axis=2, step_plane=192)
        _write_case(tmp_path, "assembly_modes", f"joint_128_seed{seed}",
                    arm="joint", scale="128", seed=seed, chunk_tiles=(2, 4, 4))
    _write_case(tmp_path, "real_floor", "vol0__small", arm=None, scale="small",
                seed=7, chunk_tiles=(3, 3, 3), sampler="real")

    res = measure(tmp_path, "assembly_modes")
    assert set(res["cells"]) == {"hybrid@128", "joint@128"}
    hybrid = res["cells"]["hybrid@128"]
    joint = res["cells"]["joint@128"]
    assert hybrid["n_chunks"] == joint["n_chunks"] == 4
    assert hybrid["n_seeds"] == 2 and hybrid["seeds"] == [101, 202]

    # Chunk 0 owns no plane in either arm; the block at x = 192 owns one, and
    # only the arm that was given a step reads it.
    def at(cell, index):
        return cell["by_chunk_index"]["chunk_plane_seam_xct"][index]["mean"]

    assert at(hybrid, 0) is None and at(joint, 0) is None
    assert at(hybrid, 1) > 4.0
    assert at(joint, 1) == pytest.approx(1.0, abs=0.2)

    assert res["real_floor"]["small"]["n_volumes"] == 1
    assert res["real_floor"]["small"]["chunk_plane_seam_xct"]["mean"] is not None

    path = report_one(tmp_path, "assembly_modes")
    text = path.read_text()
    assert "hybrid@128" in text and "joint@128" in text
    assert "chunk-plane seam (grey)" in text
    assert "real small" in text


def test_every_arm_starts_from_the_same_noise_field():
    """Same seed, different arm: the canvas the reverse process starts from must
    be identical, or the arms differ in the noise realisation as well as in the
    assembly and no metric over the pair can say which moved the answer."""
    from poregen.diffusion.sampler import region_noise_field

    fields = []
    for _ in range(2):
        g = torch.Generator(device="cpu")
        g.manual_seed(7)
        fields.append(region_noise_field(C, (32, 32, 32), (0, 0, 0),
                                         torch.device("cpu"), g))
    assert torch.equal(fields[0], fields[1])

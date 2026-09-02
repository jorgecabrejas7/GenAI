"""Eight-group parity schedule, touching neighbours and the seam diagnostic.

The first ldm05 attempt ran with ``neighbour_offset = 32``: a face neighbour
overlapped the target by half, and because opposite faces flip the SAME parity
bit the ``+z``/``-z`` pair arrived together and tiled the target completely.
Seven of the eight parity groups were handed the whole answer.  These tests pin
the replacement geometry:

1. a face neighbour shares NO voxel with the target (and the offset-32 witness
   showing that it used to),
2. no two patches inside one parity group overlap — trivially true now that
   patches tile, and still the property the schedule relies on,
3. availability is exactly "EXISTS iff the neighbour's group precedes",
4. an EXISTS neighbour reaches the denoiser WHOLE and unshifted,
5. asking for a shift at this offset raises instead of silently zeroing,
6. the seam diagnostic separates a continuous volume from independent blocks.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    PARITY_GROUP_ORDER,
    grid_index,
    group_rank,
    latent_shift_cells,
    neighbour_shared_voxels,
    parity_group,
    shift_into_target_frame,
    validate_neighbour_geometry,
    validate_shift,
)
from poregen.diffusion.sampler import VolumeGenerator, seam_discontinuity

PATCH = 64
STRIDE = 64        # patches tile: generation_stride == neighbour_offset == patch
LATENT = 16
Z_CH = 2


class _FakeModelCfg:
    z_channels = Z_CH
    use_por_cond = False
    use_orient_cond = False


class _FakeModel:
    cfg = _FakeModelCfg()


class _RecordingSampler:
    """Returns a distinct random latent per patch and records its inputs."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.nb_avail_calls: list[torch.Tensor] = []
        self.nb_latent_calls: list[torch.Tensor] = []

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
        B = nb_latents.shape[0]
        for i in range(B):
            self.nb_avail_calls.append(nb_avail[i].clone())
            self.nb_latent_calls.append(nb_latents[i].clone())
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        return torch.randn(B, C, D, D, D)


def _run(grid_n: int = 3) -> tuple[_RecordingSampler, dict, dict, list]:
    sampler = _RecordingSampler()
    generator = VolumeGenerator(
        sampler=sampler,
        vae=None,   # not exercised — _generate_latents does not decode
        device=torch.device("cpu"),
        patch_size=PATCH,
        generation_stride=STRIDE,
        neighbour_offset=STRIDE,
        latent_size=LATENT,
    )
    vol = grid_n * STRIDE
    torch.manual_seed(0)
    generated, grid_origins = generator._generate_latents(
        volume_shape=(vol, vol, vol), target_porosity=None,
    )
    # Patches are visited group by group, sorted inside each group.
    ordered = [
        gi
        for g in PARITY_GROUP_ORDER
        for gi in sorted(x for x in grid_origins if parity_group(x) == g)
    ]
    return sampler, generated, grid_origins, ordered


# ── 1. neighbours touch, they do not overlap ─────────────────────────────────

def test_touching_neighbours_share_no_voxel() -> None:
    """The fix: at offset 64 a face neighbour and the target are disjoint."""
    assert neighbour_shared_voxels(64, PATCH) == 0
    origin = np.array([128, 128, 128])
    tgt = [(o, o + PATCH) for o in origin]
    for d in NEIGHBOUR_DIRS:
        nb_origin = origin + np.array(d) * 64
        nb = [(o, o + PATCH) for o in nb_origin]
        shared = 1
        for (a0, a1), (b0, b1) in zip(tgt, nb):
            shared *= max(0, min(a1, b1) - max(a0, b0))
        assert shared == 0, f"neighbour {d} shares {shared} voxels with the target"


def test_offset_32_would_share_voxels() -> None:
    """Leak witness: why ldm05's first attempt had to be thrown away.

    At offset 32 each face neighbour hands over half the target — 32x64x64
    voxels.  Opposite faces flip the same parity bit, so they are always both
    EXISTS or both UNKNOWN; one such pair tiles the target completely
    (-z supplies latent z-cells 0-7, +z supplies 8-15 over the full y/x
    extent).  Measured leak: neighbour-vs-target MAE 0.067 against 0.744 for a
    random patch, a ratio of 0.09 — effectively a verbatim copy.
    """
    assert neighbour_shared_voxels(32, PATCH) == 32 * PATCH * PATCH
    origin = np.array([128, 128, 128])
    for d in NEIGHBOUR_DIRS:
        nb_origin = origin + np.array(d) * 32
        shared = 1
        for o_t, o_n in zip(origin, nb_origin):
            shared *= max(0, min(o_t + PATCH, o_n + PATCH) - max(o_t, o_n))
        assert shared == 32 * PATCH * PATCH

    # The opposite pair covers the target with nothing left over.
    plus, minus = (1, 0, 0), (-1, 0, 0)
    assert parity_group(plus) == parity_group(minus)      # always arrive together
    assert 32 + 32 == PATCH                                # and together they tile it


def test_geometry_guard_rejects_overlapping_neighbours() -> None:
    validate_neighbour_geometry(64, PATCH)                 # touching: fine
    validate_neighbour_geometry(128, PATCH)                # gapped: also fine
    with pytest.raises(ValueError, match="allow_neighbour_overlap"):
        validate_neighbour_geometry(32, PATCH)
    # ...and the documented ablation escape hatch
    validate_neighbour_geometry(32, PATCH, allow_neighbour_overlap=True)


def test_grid_index_makes_a_neighbour_one_step() -> None:
    """A ±neighbour_offset move is ±1 grid step for ANY origin.

    The store samples patches every 32 voxels, so origins are not all multiples
    of 64: the store is eight interleaved copies of the stride-64 grid and each
    copy must index consistently.
    """
    for base in (0, 32, 96, 160):
        gi = grid_index((base, base, base), 64)
        for d in NEIGHBOUR_DIRS:
            n_origin = tuple(base + k * 64 for k in d)
            ngi = grid_index(n_origin, 64)
            assert tuple(ngi[k] - gi[k] for k in range(3)) == d


# ── 2. the schedule itself ───────────────────────────────────────────────────

def test_no_two_patches_in_a_group_overlap() -> None:
    """Within one parity group every pair of patches is disjoint."""
    grid_n = 4
    grid = list(itertools.product(range(grid_n), repeat=3))
    for group in PARITY_GROUP_ORDER:
        members = [gi for gi in grid if parity_group(gi) == group]
        for a, b in itertools.combinations(members, 2):
            gap = max(abs(a[k] - b[k]) for k in range(3))
            assert gap >= 2, f"{a} and {b} are in group {group} but only {gap} apart"
            assert gap * STRIDE >= PATCH


def test_availability_matches_group_order() -> None:
    sampler, _, grid_origins, ordered = _run()
    assert len(sampler.nb_avail_calls) == len(grid_origins)

    for gi, avail in zip(ordered, sampler.nb_avail_calls):
        rank = group_rank(gi)
        for k, d in enumerate(NEIGHBOUR_DIRS):
            ngi = (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
            state = int(avail[k])
            if ngi not in grid_origins:
                assert state == NB_OOB
            elif group_rank(ngi) < rank:
                assert state == NB_EXISTS
            else:
                assert state == NB_UNKNOWN

    # First group has nothing before it, last group has everything before it.
    first = [gi for gi in ordered if parity_group(gi) == PARITY_GROUP_ORDER[0]]
    last  = [gi for gi in ordered if parity_group(gi) == PARITY_GROUP_ORDER[-1]]
    by_gi = dict(zip(ordered, sampler.nb_avail_calls))
    for gi in first:
        assert NB_EXISTS not in set(by_gi[gi].tolist())
    for gi in last:
        assert NB_UNKNOWN not in set(by_gi[gi].tolist())


def test_axis_availability_follows_the_parity_bit() -> None:
    """Both neighbours on axis *a* are EXISTS iff the target's parity bit a is 1.

    Rank is 4*pz + 2*py + px, and a face neighbour flips exactly one bit, so
    flipping a set bit always lowers the rank.  This is why context is graded
    0 / 2 / 4 / 6 known neighbours across the eight groups instead of the
    0 / 6 split a two-colour checkerboard would give.
    """
    counts = []
    for group in PARITY_GROUP_ORDER:
        rank = group_rank(group)
        n_exists = 0
        for axis in range(3):
            for sign in (+1, -1):
                d = tuple(sign if k == axis else 0 for k in range(3))
                ngi = tuple(group[k] + d[k] for k in range(3))
                exists = group_rank(ngi) < rank
                assert exists == bool(group[axis]), (
                    f"group {group} axis {axis}: expected EXISTS={bool(group[axis])}")
                n_exists += int(exists)
        counts.append(n_exists)
    assert sorted(counts) == [0, 2, 2, 2, 4, 4, 4, 6]


# ── 3. neighbours are fed whole and unshifted ────────────────────────────────

def test_shift_at_a_touching_offset_raises() -> None:
    """The silent-zero guard: a 16-cell shift on a 16-cell axis is empty."""
    assert latent_shift_cells(STRIDE, PATCH, LATENT) == LATENT
    nb = torch.randn(Z_CH, LATENT, LATENT, LATENT)
    with pytest.raises(ValueError, match="all zeros"):
        shift_into_target_frame(nb, (1, 0, 0), LATENT)
    with pytest.raises(ValueError, match="all zeros"):
        validate_shift(LATENT, LATENT)
    validate_shift(LATENT - 1, LATENT)          # an overlapping shift is fine


def test_generator_rejects_a_shift_it_cannot_apply() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        VolumeGenerator(
            sampler=_RecordingSampler(),
            vae=None,
            device=torch.device("cpu"),
            patch_size=PATCH,
            generation_stride=STRIDE,
            neighbour_offset=STRIDE,
            neighbour_shift=True,
            latent_size=LATENT,
        )


def test_generator_feeds_whole_unshifted_neighbours() -> None:
    """The tensor handed to the denoiser is the neighbour latent, untouched."""
    sampler, generated, grid_origins, ordered = _run()
    checked = 0
    for gi, avail, nb_in in zip(ordered, sampler.nb_avail_calls,
                                sampler.nb_latent_calls):
        for k, d in enumerate(NEIGHBOUR_DIRS):
            if int(avail[k]) != NB_EXISTS:
                assert torch.count_nonzero(nb_in[k]) == 0, "non-EXISTS slot must be zero"
                continue
            ngi = (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
            assert torch.equal(nb_in[k], generated[ngi])
            assert torch.count_nonzero(nb_in[k]) > 0, "an EXISTS neighbour must carry signal"
            checked += 1
    assert checked > 0


# ── 4. the seam diagnostic ───────────────────────────────────────────────────

def _tiled(rng, blocks: int = 2, offset: float = 0.0) -> np.ndarray:
    """A volume of `blocks`³ patches; each block shifted by `offset` * index."""
    n = blocks * PATCH
    vol = rng.standard_normal((n, n, n)).astype(np.float32)
    if offset:
        for i, b in enumerate(itertools.product(range(blocks), repeat=3)):
            sl = tuple(slice(k * PATCH, (k + 1) * PATCH) for k in b)
            vol[sl] += offset * i
    return vol


def test_seam_ratio_is_one_for_a_coherent_volume() -> None:
    """One continuous field: the seam plane is no different from any other."""
    rng = np.random.default_rng(0)
    stats = seam_discontinuity(_tiled(rng, blocks=3), PATCH, prefix="seam")
    assert stats["seam_planes"] == 6                 # 2 seams per axis
    for a in ("z", "y", "x"):
        assert stats[f"seam_{a}_planes"] == 2
        assert stats[f"seam_{a}_ratio"] == pytest.approx(1.0, abs=0.05)
    assert stats["seam_ratio"] == pytest.approx(1.0, abs=0.03)
    assert stats["seam_mad"] > 0.0
    assert stats["seam_interior_mad"] > 0.0


def test_seam_ratio_flags_independent_blocks() -> None:
    """Blocks that do not agree at their shared face give a ratio well above 1."""
    rng = np.random.default_rng(1)
    stats = seam_discontinuity(_tiled(rng, blocks=2, offset=5.0), PATCH, prefix="seam")
    assert stats["seam_ratio"] > 3.0
    for a in ("z", "y", "x"):
        assert stats[f"seam_{a}_ratio"] > 3.0


def test_seam_ratio_is_one_for_a_linear_ramp() -> None:
    """A perfectly smooth volume: every plane has the same step, seams included."""
    n = 2 * PATCH
    ramp = np.broadcast_to(
        np.arange(n, dtype=np.float32)[:, None, None], (n, n, n)
    ).copy()
    stats = seam_discontinuity(ramp, PATCH, prefix="seam")
    assert stats["seam_z_ratio"] == pytest.approx(1.0, abs=1e-6)
    # y and x are constant, so there is nothing to compare: NaN, not a lie.
    assert np.isnan(stats["seam_y_ratio"])


def test_seam_diagnostic_needs_a_3d_volume() -> None:
    with pytest.raises(ValueError, match="3-D"):
        seam_discontinuity(np.zeros((4, 4)), PATCH)


# ── 5. generator geometry guards ─────────────────────────────────────────────

def test_generation_stride_must_equal_neighbour_offset() -> None:
    with pytest.raises(ValueError, match="neighbour_offset"):
        VolumeGenerator(
            sampler=_RecordingSampler(),
            vae=None,
            device=torch.device("cpu"),
            patch_size=PATCH,
            generation_stride=64,
            neighbour_offset=128,
            latent_size=LATENT,
        )


def test_generation_stride_must_equal_patch_size() -> None:
    """Overlapping generation is gone: it leaked, and it needed blending."""
    with pytest.raises(ValueError, match="patch_size"):
        VolumeGenerator(
            sampler=_RecordingSampler(),
            vae=None,
            device=torch.device("cpu"),
            patch_size=PATCH,
            generation_stride=32,
            neighbour_offset=32,
            latent_size=LATENT,
        )


def test_grid_must_tile_the_volume_exactly() -> None:
    sampler = _RecordingSampler()
    gen = VolumeGenerator(
        sampler=sampler,
        vae=None,
        device=torch.device("cpu"),
        patch_size=PATCH,
        generation_stride=STRIDE,
        neighbour_offset=STRIDE,
        latent_size=LATENT,
    )
    with pytest.raises(ValueError, match="leaves a gap"):
        gen._generate_latents(volume_shape=(96, 64, 64), target_porosity=None)

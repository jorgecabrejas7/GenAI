"""The teacher-forced ceiling: a canvas of REAL encoded material.

The ``assembly_modes`` assessment asks how far the hybrid sampler's assembly is
from an upper bound.  The bound is a run in which every window's six face
neighbours are not the sampler's own output but the encodings of a real test
volume at the same canvas positions: whatever the sampler cannot reach with
PERFECT neighbours, better neighbour handling cannot buy.

The neighbours come from the run's own latent store — the same
``data/split_v3/latents_r08z8`` the model was trained on, resolved from the run
config by :func:`poregen.eval_v4.generate.resolve_latent_store`.  Nothing here
encodes anything: an encoder other than the one that built the store would
produce latents the denoiser has never seen, which would measure the encoder
and call it a ceiling.

Geometry.  Store rows sit on a 32-voxel lattice (``sample_stride``) and each
row is a 64-voxel patch, so the rows whose origins are 64 apart tile a block
with no overlap and no blending.  A canvas is therefore assembled by pasting
one row per 64-voxel tile.  Every window origin and every neighbour block in
the sampler is a multiple of 32 voxels, so a neighbour read out of this canvas
is a real, contiguous block of real material — it may straddle up to eight
stored patches, exactly as a generated neighbour straddles the canvas.

What is served is a posterior DRAW, ``mu + sigma * eps``, per-channel
normalised.  Not the mean: the store holds a posterior and the training step
draws from it, so a mean-valued canvas is a distribution neither the model nor
the sampler ever sees.  The draw is seeded by the case, so the arm is
reproducible from its manifest.

The hard limit this module reports rather than works around: **test patches
reach z0 = 128**, so the deepest real block on the 64-voxel grid is 192 voxels.
A canvas deeper than that cannot be teacher forced from real material, and
repeating a block to fill it would put a fake join on a chunk plane — the one
place the assessment measures.  :func:`reference_latent_canvas` raises instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Spacing of the rows used to tile a canvas.  It is the store's
#: ``generation_stride`` and equals the patch size, so the tiles touch and never
#: overlap.
TILE_VOX = 64
#: Spacing of the rows the store actually holds.
SAMPLE_STRIDE_VOX = 32


def read_index(store_root: str | Path, split: str):
    """``(volume_id, z0, y0, x0)`` of every row of one split, as numpy arrays."""
    import pandas as pd  # noqa: PLC0415

    path = Path(store_root) / split / "index.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist: the teacher-forced arm reads real latents "
            f"from the run's own store, and this store has no {split!r} split."
        )
    df = pd.read_parquet(str(path), columns=["volume_id", "z0", "y0", "x0"])
    return (
        df["volume_id"].to_numpy(),
        df["z0"].to_numpy(np.int64),
        df["y0"].to_numpy(np.int64),
        df["x0"].to_numpy(np.int64),
    )


def _all_true_block(occ: np.ndarray, extent: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """First (raster) origin of an all-True ``extent`` box in ``occ``, or None.

    A 3-D summed-area table, so the search costs one pass over ``occ`` however
    many origins are tried.
    """
    ez, ey, ex = extent
    if any(occ.shape[a] < extent[a] for a in range(3)):
        return None
    c = np.pad(occ.astype(np.int64).cumsum(0).cumsum(1).cumsum(2),
               ((1, 0), (1, 0), (1, 0)))
    box = (
        c[ez:, ey:, ex:]
        - c[:-ez, ey:, ex:] - c[ez:, :-ey, ex:] - c[ez:, ey:, :-ex]
        + c[:-ez, :-ey, ex:] + c[:-ez, ey:, :-ex] + c[ez:, :-ey, :-ex]
        - c[:-ez, :-ey, :-ex]
    )
    hits = np.argwhere(box == ez * ey * ex)
    if not len(hits):
        return None
    return tuple(int(v) for v in hits[0])  # type: ignore[return-value]


def find_reference_block(index, shape, *, split, store_root) -> dict:
    """A real block of ``shape`` voxels every 64-voxel tile of which is stored.

    ``index`` is what :func:`read_index` returns, handed in rather than read
    here so the caller that also needs the row lookup reads the parquet once.

    Volumes are tried in sorted order and, within a volume, the lowest origin
    wins, so the answer is a property of the store and not of when it was asked.
    The lattice PHASE is searched too: a volume's rows may start at 0 or at 32,
    and refusing the second phase would reject blocks that exist.

    Returns ``{"volume_id", "origin_zyx", "tiles", "split"}``.  Raises when no
    volume in the split holds the shape — see the module docstring on depth.
    """
    shape = tuple(int(s) for s in shape)
    if any(s % TILE_VOX for s in shape):
        raise ValueError(
            f"a reference canvas is tiled from {TILE_VOX}-voxel patches, so "
            f"{shape} must be a whole number of tiles on every axis."
        )
    tiles = tuple(s // TILE_VOX for s in shape)
    vol, z0, y0, x0 = index

    step = TILE_VOX // SAMPLE_STRIDE_VOX          # lattice steps per tile
    for vid in sorted(set(vol.tolist())):
        sel = vol == vid
        lz, ly, lx = z0[sel] // SAMPLE_STRIDE_VOX, y0[sel] // SAMPLE_STRIDE_VOX, x0[sel] // SAMPLE_STRIDE_VOX
        occ = np.zeros((lz.max() + 1, ly.max() + 1, lx.max() + 1), dtype=bool)
        occ[lz, ly, lx] = True
        for pz in range(step):
            for py in range(step):
                for px in range(step):
                    hit = _all_true_block(occ[pz::step, py::step, px::step], tiles)
                    if hit is None:
                        continue
                    origin = (
                        (hit[0] * step + pz) * SAMPLE_STRIDE_VOX,
                        (hit[1] * step + py) * SAMPLE_STRIDE_VOX,
                        (hit[2] * step + px) * SAMPLE_STRIDE_VOX,
                    )
                    return {
                        "volume_id": str(vid),
                        "origin_zyx": list(origin),
                        "tiles": list(tiles),
                        "split": split,
                    }
    raise ValueError(
        f"no {split} volume in {store_root} holds a {shape}-voxel block covered by "
        f"{TILE_VOX}-voxel patches on every tile. The teacher-forced arm needs REAL "
        f"material at every canvas position; the deepest real block this store "
        f"carries is bounded by the specimen thickness, and filling the rest by "
        f"repeating a block would put a fake join on a chunk plane."
    )


def reference_latent_canvas(
    store_root: str | Path,
    shape: tuple[int, int, int],
    *,
    seed: int,
    split: str = "test",
) -> tuple[np.ndarray, dict]:
    """``(canvas, provenance)`` — normalised real latents over the whole canvas.

    ``canvas`` is ``(C, Z/4, Y/4, X/4)`` float32 in the sampler's normalised
    latent space, assembled by pasting one stored patch per 64-voxel tile.
    ``provenance`` names the volume, the origin and the draw, and belongs in the
    manifest: without it the arm is an unattributable number.
    """
    store_root = Path(store_root)
    meta = json.loads((store_root / "metadata.json").read_text())
    if meta["storage"]["pack_scheme"] != "mu_then_std":
        raise ValueError(
            f"{store_root}/metadata.json packs latents as "
            f"{meta['storage']['pack_scheme']!r}; this reader knows 'mu_then_std'."
        )
    c, lz, ly, lx = (int(v) for v in meta["latent_shape"])
    if not (lz == ly == lx):
        raise ValueError(f"expected a cubic latent patch, got {meta['latent_shape']}.")
    if int(meta.get("patch_size", TILE_VOX)) != TILE_VOX:
        raise ValueError(
            f"the store's patch_size is {meta.get('patch_size')}, not {TILE_VOX}; "
            "the canvas tiling assumes one stored patch per tile."
        )
    norm = meta["normalization"]
    mean = np.asarray(norm["per_channel_mean"], np.float32).reshape(c, 1, 1, 1)
    std = np.asarray(norm["per_channel_std"], np.float32).reshape(c, 1, 1, 1)

    index = read_index(store_root, split)
    block = find_reference_block(index, shape, split=split, store_root=store_root)
    origin = tuple(block["origin_zyx"])
    tiles = tuple(block["tiles"])

    vol, z0, y0, x0 = index
    sel = np.flatnonzero(vol == block["volume_id"])
    # One lookup table per volume, keyed by the tile index inside the block.
    key = {}
    for row in sel:
        key[(int(z0[row]), int(y0[row]), int(x0[row]))] = int(row)

    dtype = np.dtype(meta["storage"]["dtype"])
    item = (2 * c, lz, ly, lx)
    n_rows = len(vol)
    latents = np.memmap(str(store_root / split / "latents.bin"), dtype=dtype,
                        mode="r", shape=(n_rows, *item))

    rng = np.random.default_rng(int(seed))
    canvas = np.zeros((c, tiles[0] * lz, tiles[1] * ly, tiles[2] * lx), np.float32)
    for iz in range(tiles[0]):
        for iy in range(tiles[1]):
            for ix in range(tiles[2]):
                pos = (origin[0] + iz * TILE_VOX,
                       origin[1] + iy * TILE_VOX,
                       origin[2] + ix * TILE_VOX)
                packed = np.asarray(latents[key[pos]], np.float32)
                mu, sd = packed[:c], packed[c:]
                z = mu + sd * rng.standard_normal(mu.shape).astype(np.float32)
                canvas[:, iz * lz:(iz + 1) * lz,
                       iy * ly:(iy + 1) * ly,
                       ix * lx:(ix + 1) * lx] = (z - mean) / std

    provenance = {
        **block,
        "store": str(store_root),
        "n_patches": int(np.prod(tiles)),
        "draw": "mu + sigma*eps, per-channel normalised",
        "draw_seed": int(seed),
    }
    logger.info(
        "teacher-forced reference: %s at %s, %d patches of %s",
        block["volume_id"], origin, provenance["n_patches"], meta["latent_shape"],
    )
    return canvas, provenance

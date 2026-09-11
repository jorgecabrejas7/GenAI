"""The memorisation check: is a generated volume a copy of training material?

Every other statistic in the suite asks whether the generated material has the
right *distribution*.  This one asks the question a reviewer asks first: did
the model reproduce a patch it was trained on?

What makes the answer hard is that a nearest-neighbour distance has no natural
scale.  Two things fix that here, and neither is optional.

**The search is exhaustive.**  The earlier implementation scored each generated
patch against a random 10 000 of the ~1.6 M training latents.  That is an
*upper bound* on the distance to the nearest training patch, not the distance
to the nearest training patch — and an upper bound is exactly the wrong side of
the question, because it can only ever make a copy look further away than it
is.  Here every query is scored against the FULL train split, restricted to the
``stride``-64 rows.  The restriction is not a sample: the store is built on a
32-voxel sampling stride, so eight interleaved copies of the 64-voxel grid sit
in it and any patch's "second-nearest neighbour" would otherwise be the SAME
material shifted by 32 voxels.  On the stride-64 grid no two bank rows share a
voxel, so the second-nearest neighbour is genuinely different material.

**The statistic is a ratio, not a distance.**  Following Favero, the query's
distance to its nearest bank row is divided by its distance to the
second-nearest::

    ratio = ||x - x'|| / ||x - x''||

A copy sits on top of one training patch and a normal distance from every
other, so its ratio collapses toward 0.  A fresh sample sits a typical distance
from both, so its ratio approaches 1.  :data:`RATIO_THRESHOLD` (1/3) is the
memorisation verdict.  The ratio is dimensionless, which is what lets the same
number be read in two spaces that have no common unit.

**Both spaces are searched.**  Latent space is where the LDM works and where a
copied latent is sharpest; decoded grey space is where a copy would actually be
visible.  A model can only be cleared by both.

**The floor is real held-out material, and in grey space there are two of
them.**  Real validation patches are not memorised by construction — they come
from panels the VAE and the LDM never saw — yet they still have near neighbours
in the train set, because the panels are made of the same material.  What the
val patches score is therefore the value the ratio takes when nothing has been
copied, and the generated number is only readable beside it.  Without that
floor a generated ratio of 0.6 means nothing at all.

In grey space one more thing has to be matched.  A generated patch reaches grey
through the VAE decoder and carries the decoder's own reconstruction error; a
raw scan patch does not.  So the primary grey floor is the val patch ROUND
TRIPPED through the same frozen VAE — its stored posterior mean, decoded — and
it therefore carries the same error the query does.  The raw val patch is kept
as a second row because it bounds the answer from the other side: it is what
the ratio would be with no decoder error at all.  With the round-tripped row
present the grey comparison is like-for-like and the grey ratio is no longer
merely an upper bound.  (The absolute 1/3 verdict on a grey ratio stays
conservative — a volume that copied a training latent exactly still sits one
reconstruction error from the real patch — but the generated-against-floor
reading is now fair, which is the reading that decides anything.)

**Both neighbour buckets are filled.**  A 192-cubed volume is exactly one
chunk, so no window face is ever UNKNOWN and a search over those volumes alone
has one bucket and no contrast.  The ``multichunk`` and ``assembly_modes``
volumes are big enough to hold a chunk plane, and inside ONE of them some
windows look into chunks that are already solved and others into chunks that
are not.  :func:`window_states` labels each query patch by the state of the
window that was denoised at its position, using the sampler's own chunk order —
which makes this an ordering fact, not a geometric one.

Failure modes, stated plainly
-----------------------------
* **The query is re-encoded, not taken from ``latents.npy``.**  ldm06 trains on
  a posterior DRAW (``data.latent_mode: sampled``), so a generated latent lives
  in sample space while the store holds posterior MEANS.  Comparing the two
  directly would add ``sigma*eps`` to every distance.  The generated volume is
  therefore pushed back through the VAE encoder, so both sides of the latent
  distance are means of the same encoder.
* **The bank is every stride-64 train row**, air-heavy rows included.  They are
  far from any whole-material query and cannot become its nearest neighbour;
  filtering them would only cost a pass over ``material.bin``.
* **Volumes above :data:`MAX_TILES_PER_VOLUME` tiles are left out**, which is a
  cost decision and is named in the result rather than silently applied.  The
  192x1024x1024 cases are 768 query patches each; nothing about them is
  different in kind.
* **A patch position that was never a window origin cannot be bucketed per
  window.**  It does not happen at any shape the search currently takes — a
  test asserts that — but if it ever does, that volume falls back to the
  volume-level bucket and is listed in ``cases_bucketed_at_volume_level``.  A
  label that is subtly wrong is worse than one that is openly coarse.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poregen.eval_v4.io import LATENT_DOWNSAMPLE, TILE, load_cases

logger = logging.getLogger(__name__)

#: The side of the patch the VAE was trained on, and the tiling the store is
#: built on.  Nothing here works on any other size.
PATCH = TILE

#: Only store rows whose origin sits on this grid enter the bank.  See the
#: module docstring: it is what makes the SECOND-nearest neighbour meaningful.
BANK_STRIDE = 64

#: Favero's memorisation criterion.  ``nearest / second-nearest`` below this is
#: a copy.
RATIO_THRESHOLD = 1.0 / 3.0

#: Bank rows per GPU chunk.  The store is 272 GB and cannot be held; these are
#: sized so one chunk of the distance computation stays under about half a
#: gigabyte at each side's dimensionality.
LATENT_BANK_CHUNK = 4096
GREY_BANK_CHUNK = 512

#: Real validation patches drawn for the floor.  They are the whole statistic's
#: reference, so the count buys precision on the floor and nothing else.
FLOOR_PATCHES = 512

#: The assessments whose generated volumes are searched.  ``sampler`` and
#: ``porosity_global`` are the production operating point; ``multichunk`` and
#: ``assembly_modes`` are the only volumes big enough to hold a chunk plane,
#: and therefore the only ones that can fill the UNKNOWN neighbour bucket.  An
#: assessment that a campaign has not generated is skipped, not an error.
ASSESSMENTS = ("sampler", "porosity_global", "multichunk", "assembly_modes")

#: Cost ceiling, in 64-voxel tiles, on the volumes that enter the search.  The
#: grey pass is linear in the number of queries and the whole query block stays
#: resident on the device, so a volume is taken whole or not at all rather than
#: sampled.  192 cubed is 27 tiles, 384 cubed is 216 and 192x384x384 is 108,
#: all of which are taken; the 192x1024x1024 cases are 768 each and are not.
#: Stated as a tile count rather than a shape list so an assessment written
#: after this module still lands on one side of the line by itself.
MAX_TILES_PER_VOLUME = 256

#: The two buckets a case falls in, derived from its manifest geometry.
NB_PRESENT = "neighbours_present"
NB_UNKNOWN_BUCKET = "neighbours_unknown"


# ---------------------------------------------------------------------------
# The pure statistic
# ---------------------------------------------------------------------------

class Top2:
    """Running nearest and second-nearest bank row for every query.

    The bank is streamed in chunks, so no accumulator may need more than the
    two best seen so far.  Merging is done by sorting the running pair against
    the chunk's candidates, which is correct however many candidates a chunk
    offers and however they tie.
    """

    def __init__(self, n_query: int):
        self.d1 = np.full(int(n_query), np.inf, np.float64)
        self.d2 = np.full(int(n_query), np.inf, np.float64)
        self.i1 = np.full(int(n_query), -1, np.int64)
        self.i2 = np.full(int(n_query), -1, np.int64)

    def update_candidates(self, dist: np.ndarray, index: np.ndarray) -> None:
        """Merge ``(n_query, k)`` candidate distances and their bank row ids."""
        dist = np.asarray(dist, np.float64)
        index = np.asarray(index, np.int64)
        if dist.shape != index.shape:
            raise ValueError(
                f"candidate distances {dist.shape} and ids {index.shape} differ."
            )
        if dist.shape[0] != len(self.d1):
            raise ValueError(
                f"candidates are for {dist.shape[0]} queries, this accumulator "
                f"holds {len(self.d1)}."
            )
        dd = np.concatenate([np.stack([self.d1, self.d2], 1), dist], 1)
        ii = np.concatenate([np.stack([self.i1, self.i2], 1), index], 1)
        order = np.argsort(dd, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(dd))[:, None]
        self.d1, self.d2 = dd[rows, order].T.copy()
        self.i1, self.i2 = ii[rows, order].T.copy()

    def update(self, dist: np.ndarray, index: np.ndarray) -> None:
        """Merge a whole ``(n_query, n_bank_chunk)`` distance block."""
        dist = np.asarray(dist, np.float64)
        index = np.asarray(index, np.int64)
        if dist.shape[1] != len(index):
            raise ValueError(
                f"distance block has {dist.shape[1]} bank columns but "
                f"{len(index)} bank ids were given."
            )
        k = min(2, dist.shape[1])
        take = np.argsort(dist, axis=1, kind="stable")[:, :k]
        rows = np.arange(len(dist))[:, None]
        self.update_candidates(dist[rows, take], index[take])

    def ratio(self) -> np.ndarray:
        """Favero's ``||x-x'|| / ||x-x''||`` for every query."""
        return favero_ratio(self.d1, self.d2)

    def verdict(self) -> np.ndarray:
        """Boolean: is this query memorised by the 1/3 criterion?"""
        return self.ratio() < RATIO_THRESHOLD


def favero_ratio(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """``d1 / d2``, with the two degenerate cases decided rather than divided.

    ``d2 == 0`` means the query lands exactly on two distinct bank rows at
    once, which is as memorised as a query can be; the ratio is 0, not
    ``0 / 0``.  An infinite ``d2`` means the bank held fewer than two rows,
    which the search refuses long before this point.
    """
    d1 = np.asarray(d1, np.float64)
    d2 = np.asarray(d2, np.float64)
    out = np.full(d1.shape, np.nan)
    ok = (d2 > 0.0) & np.isfinite(d2)
    out[ok] = d1[ok] / d2[ok]
    out[np.isfinite(d2) & (d2 <= 0.0)] = 0.0
    return out


def squared_distances(query, bank, query_sq=None):
    """``(Q, B)`` squared L2 distances by expansion, clamped at zero.

    The expansion ``|q|^2 + |b|^2 - 2 q.b`` is used rather than an explicit
    difference because a 262 144-dimensional explicit difference would need a
    ``(Q, B, D)`` intermediate that no device holds.

    It is a SELECTION kernel and nothing more.  Two large float32 numbers are
    subtracted here, so it cancels worst exactly where this check looks — a
    near-duplicate — and its rounding even depends on the chunk width, because
    a different bank block size makes the matmul reduce in a different order.
    The ranking survives that; the value does not, which is why
    :func:`search` recomputes the two chosen distances from an explicit
    float64 difference before anyone reads them.

    ``query_sq`` is ``|q|^2`` when the caller already has it.  The query block
    is two gigabytes in grey space and is the same for every bank chunk, so
    recomputing its norms 430 times would allocate that much again per chunk.
    """
    import torch  # noqa: PLC0415

    q = query.to(torch.float32)
    b = bank.to(torch.float32)
    qn = (q * q).sum(1, keepdim=True) if query_sq is None else query_sq
    d2 = qn + (b * b).sum(1)[None, :] - 2.0 * (q @ b.T)
    return torch.clamp_min_(d2, 0.0)


# ---------------------------------------------------------------------------
# What the generated volume contributes
# ---------------------------------------------------------------------------

def patch_origins(material: np.ndarray, patch: int = PATCH) -> list[tuple[int, int, int]]:
    """Origins of the non-overlapping whole-material ``patch``-cubed tiling.

    That tiling is the grid the store was built on, so a generated patch and a
    training row are the same kind of object and a distance between them means
    something.  A patch that reaches outside the requested specimen is dropped:
    no training row contains air placed by a request, so it has no honest
    nearest neighbour.
    """
    d, h, w = material.shape
    return [
        (z, y, x)
        for z in range(0, d - patch + 1, patch)
        for y in range(0, h - patch + 1, patch)
        for x in range(0, w - patch + 1, patch)
        if material[z:z + patch, y:y + patch, x:x + patch].all()
    ]


def grey_patches(
    xct_u8: np.ndarray,
    origins: list[tuple[int, int, int]],
    patch: int = PATCH,
) -> np.ndarray:
    """``(m, patch**3)`` float32 grey in [0, 1] for the given origins."""
    if not origins:
        return np.zeros((0, patch ** 3), np.float32)
    out = np.stack([
        xct_u8[z:z + patch, y:y + patch, x:x + patch] for z, y, x in origins
    ])
    return (out.reshape(len(origins), -1).astype(np.float32) / 255.0)


def encode_patches(
    vae,
    xct_u8: np.ndarray,
    label_u8: np.ndarray,
    origins: list[tuple[int, int, int]],
    device=None,
    patch: int = PATCH,
    batch: int = 4,
) -> np.ndarray:
    """``(m, C*d*h*w)`` RAW posterior means of the given patches of a volume.

    The encoder inputs come from the model's own ``encoder_inputs`` declaration
    through :func:`poregen.training.engine.encoder_input_keys`, and the moments
    from ``encode_moments`` — the same two entry points
    ``scripts/build_latent_dataset.py`` used to write the store.  The r08 VAE
    encodes the grey volume AND the 3-class label; feeding it the grey alone
    would produce latents from a different function to the ones in the store.
    """
    import torch  # noqa: PLC0415

    from poregen.training.engine import encoder_input_keys  # noqa: PLC0415

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = encoder_input_keys(vae)
    unknown = set(keys) - {"xct", "label"}
    if unknown:
        raise ValueError(
            f"the VAE declares encoder inputs {keys}; this check can supply only "
            f"'xct' and 'label' from a generated case, not {sorted(unknown)}."
        )
    if not origins:
        return np.zeros((0, 0), np.float32)

    out = []
    with torch.no_grad():
        for s in range(0, len(origins), batch):
            sl = [np.s_[z:z + patch, y:y + patch, x:x + patch]
                  for z, y, x in origins[s:s + batch]]
            arrays = {
                "xct": torch.from_numpy(
                    np.stack([xct_u8[i] for i in sl]).astype(np.float32) / 255.0
                ).unsqueeze(1),
                "label": torch.from_numpy(
                    np.stack([label_u8[i] for i in sl]).astype(np.int64)
                ),
            }
            mu, _ = vae.encode_moments(*(arrays[k].to(device) for k in keys))
            out.append(mu.flatten(1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def decode_grey(vae, mu: np.ndarray, device=None, batch: int = 16) -> np.ndarray:
    """``(n, patch**3)`` grey in [0, 1], decoded from RAW posterior means.

    This is the second half of the round trip that gives the grey floor the
    same decoder signature the generated queries have.  It is the sampler's
    own decode path — ``xct_head(decoder(z))`` then the clamp-and-scale of
    :func:`poregen.models.vae.base.decode_xct` — and never a sigmoid, which
    would squash the volume into [0.5, 0.731].

    The store's ``mu`` is what its own encoder produced for that patch, so
    decoding it IS "encode the real patch, then decode it"; re-running the
    encoder would only reproduce a number the store already holds.
    """
    import torch  # noqa: PLC0415

    from poregen.models.vae.base import decode_xct  # noqa: PLC0415

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = np.asarray(mu, np.float32)
    if mu.ndim != 5:
        raise ValueError(
            f"decode_grey needs (n, C, d, h, w) raw means, got shape {mu.shape}."
        )
    out = []
    with torch.no_grad():
        for s in range(0, len(mu), batch):
            z = torch.from_numpy(mu[s:s + batch]).to(device)
            grey = decode_xct(vae.xct_head(vae.decoder(z)))
            out.append(grey.flatten(1).float().cpu().numpy())
    if not out:
        return np.zeros((0, PATCH ** 3), np.float32)
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Neighbour availability, rebuilt per window from the manifest alone
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowStates:
    """Which of a window's six faces saw a neighbour WHEN IT WAS DENOISED.

    Not a geometric property.  A face is UNKNOWN only when the chunk it
    reaches into had not been solved yet, so the answer depends on the order
    the sampler walks the chunk grid in.  ``states`` is keyed by the window's
    global origin in LATENT CELLS, and holds the six availability codes in
    :data:`poregen.diffusion.conditioning.NEIGHBOUR_DIRS` order.
    """

    cells: tuple[int, int, int]
    win: int
    stride: int
    n_chunks: int
    states: dict[tuple[int, int, int], tuple[int, ...]]

    def census(self) -> dict:
        """Face totals over the whole volume, and the volume-level bucket."""
        from poregen.diffusion.conditioning import (  # noqa: PLC0415
            NB_EXISTS, NB_OOB, NB_UNKNOWN,
        )

        flat = [v for row in self.states.values() for v in row]
        unknown = flat.count(NB_UNKNOWN)
        return {
            "n_chunks": self.n_chunks,
            "n_windows": len(self.states),
            "exists": flat.count(NB_EXISTS),
            "oob": flat.count(NB_OOB),
            "unknown": unknown,
            "bucket": NB_UNKNOWN_BUCKET if unknown else NB_PRESENT,
        }

    def at_voxel(self, origin_vox: tuple[int, int, int]) -> dict | None:
        """The window that starts at this VOXEL origin, or ``None``.

        ``None`` is not a detail to paper over: it means this patch position
        was never a window origin, so no honest per-window bucket exists for
        it and the caller must fall back to the volume-level one.
        """
        from poregen.diffusion.conditioning import (  # noqa: PLC0415
            NB_EXISTS, NB_OOB, NB_UNKNOWN,
        )

        if any(o % LATENT_DOWNSAMPLE for o in origin_vox):
            return None
        key = tuple(int(o) // LATENT_DOWNSAMPLE for o in origin_vox)
        row = self.states.get(key)
        if row is None:
            return None
        unknown = row.count(NB_UNKNOWN)
        return {
            "exists": row.count(NB_EXISTS),
            "oob": row.count(NB_OOB),
            "unknown": unknown,
            "bucket": NB_UNKNOWN_BUCKET if unknown else NB_PRESENT,
        }


def window_states(manifest) -> WindowStates:
    """Rebuild every window's neighbour availability from the manifest.

    The sampler decides this per window and records none of it, so it is
    rebuilt here from the three manifest fields that determine it —
    ``volume_shape``, ``chunk_tiles`` and ``window_stride`` — using the
    sampler's own window grid, chunk partition, chunk ORDER and neighbour
    directions.  The rule is
    :meth:`poregen.diffusion.sampler.VolumeGenerator._neighbour_plan`'s: a
    face is OOB when the neighbour block leaves the canvas, EXISTS when it
    lies inside a finished chunk or inside the chunk being generated, and
    UNKNOWN when it reaches into a chunk that has not been generated yet.

    Two facts make the reconstruction exact rather than a guess.  The canvas
    is ``volume_shape`` and nothing else — ``request_offset`` names the frame
    the noise is drawn in, not a larger canvas, so a translated request has
    the same chunk grid.  And the chunk order is the fixed z-major nesting of
    ``_chunk_ranges`` over the three axes, with no dependence on the content,
    the seed or the machine.  Reproduce both and every window's state follows.
    """
    from poregen.diffusion.conditioning import (  # noqa: PLC0415
        NB_EXISTS, NB_OOB, NB_UNKNOWN, NEIGHBOUR_DIRS,
    )
    from poregen.diffusion.sampler import window_origins  # noqa: PLC0415

    if manifest.chunk_tiles is None or manifest.window_stride is None:
        raise ValueError(
            f"{manifest.case}: the manifest carries no chunk_tiles/window_stride, "
            "so the neighbour states cannot be rebuilt. Only a generated volume "
            "has them."
        )

    cells = tuple(int(s) // LATENT_DOWNSAMPLE for s in manifest.volume_shape)
    win = PATCH // LATENT_DOWNSAMPLE
    stride = int(manifest.window_stride) // LATENT_DOWNSAMPLE
    n_tiles = tuple(int(s) // PATCH for s in manifest.volume_shape)
    chunk_tiles = tuple(int(c) for c in manifest.chunk_tiles)

    # _chunk_ranges, and the same z-major nesting `generate` walks them in.
    chunk_grid = [
        [(i, min(i + chunk_tiles[a], n_tiles[a]))
         for i in range(0, n_tiles[a], chunk_tiles[a])]
        for a in range(3)
    ]
    chunks = [(cz, cy, cx) for cz in chunk_grid[0]
              for cy in chunk_grid[1] for cx in chunk_grid[2]]

    available = np.zeros(cells, bool)
    states: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for chunk in chunks:
        lo = tuple(chunk[a][0] * win for a in range(3))
        hi = tuple(chunk[a][1] * win for a in range(3))
        chunk_sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        visible = available.copy()
        visible[chunk_sl] = True
        chunk_cells = tuple(hi[a] - lo[a] for a in range(3))
        for o in window_origins(chunk_cells, win, stride):
            g = tuple(lo[a] + o[a] for a in range(3))
            row = []
            for d in NEIGHBOUR_DIRS:
                a_lo = tuple(g[k] + d[k] * win for k in range(3))
                if any(a_lo[k] < 0 or a_lo[k] + win > cells[k] for k in range(3)):
                    row.append(NB_OOB)
                    continue
                block = tuple(slice(a_lo[k], a_lo[k] + win) for k in range(3))
                row.append(NB_EXISTS if visible[block].all() else NB_UNKNOWN)
            states[g] = tuple(row)
        available[chunk_sl] = True

    return WindowStates(cells=cells, win=win, stride=stride,
                        n_chunks=len(chunks), states=states)


def neighbour_census(manifest) -> dict:
    """Face totals and the volume-level bucket.  See :func:`window_states`."""
    return window_states(manifest).census()


# ---------------------------------------------------------------------------
# The store, read in chunks
# ---------------------------------------------------------------------------

@dataclass
class PatchStore:
    """The stride-64 rows of one split, in both spaces, read a chunk at a time.

    Nothing here holds more than one chunk.  ``latents.bin`` is 209 GB for the
    train split alone and the raw source patches are another 564 GB, so both
    are memory-mapped and gathered row by row.

    Grey comes from the SOURCE patches the store was built from
    (``patches_xct.bin`` beside the ``source_patch_index`` the metadata names),
    addressed by each row's ``source_row``.  That is the actual training image,
    which is what a memorisation claim has to be made against — not the VAE's
    rendering of it.
    """

    root: Path
    split: str
    rows: np.ndarray            # row ids into this split's latents.bin
    source_rows: np.ndarray     # row ids into patches_xct.bin
    latent_channels: int
    latent_spatial: tuple[int, ...]
    latent_dtype: np.dtype
    n_split_rows: int
    patch: int
    patch_file: Path
    n_patch_rows: int
    norm_mean: np.ndarray
    norm_std: np.ndarray

    @classmethod
    def open(
        cls,
        latents_root: str | Path,
        split: str,
        *,
        stride: int = BANK_STRIDE,
    ) -> "PatchStore":
        import pandas as pd  # noqa: PLC0415

        root = Path(latents_root)
        meta = json.loads((root / "metadata.json").read_text())
        storage = meta["storage"]
        if storage["pack_scheme"] != "mu_then_std":
            raise ValueError(
                f"{root}: pack_scheme is {storage['pack_scheme']!r}; this reader "
                "knows 'mu_then_std' — channels 0..C-1 are the posterior mean."
            )
        if int(meta["patch_size"]) != PATCH:
            raise ValueError(
                f"{root}: the store is built on {meta['patch_size']}-voxel patches "
                f"and this check is defined on {PATCH}-voxel ones."
            )
        c, *spatial = (int(v) for v in meta["latent_shape"])

        idx = pd.read_parquet(
            str(root / split / "index.parquet"),
            columns=["source_row", "z0", "y0", "x0"],
        )
        on_grid = (
            (idx["z0"].to_numpy() % stride == 0)
            & (idx["y0"].to_numpy() % stride == 0)
            & (idx["x0"].to_numpy() % stride == 0)
        )
        rows = np.flatnonzero(on_grid).astype(np.int64)
        if rows.size < 2:
            raise ValueError(
                f"{root}/{split} holds {rows.size} rows on the stride-{stride} "
                "grid; the second-nearest neighbour is undefined below two."
            )

        patch_index = Path(meta["source_patch_index"])
        patch_file = patch_index.parent / "patches_xct.bin"
        if not patch_file.exists():
            raise FileNotFoundError(
                f"{patch_file} is not there. The store names "
                f"{patch_index} as its source patch index, and the grey half of "
                "this check reads the raw patches beside it."
            )
        n_patch_rows = patch_file.stat().st_size // (PATCH ** 3)

        norm = meta["normalization"]
        return cls(
            root=root, split=split, rows=rows,
            source_rows=idx["source_row"].to_numpy()[rows].astype(np.int64),
            latent_channels=c, latent_spatial=tuple(spatial),
            latent_dtype=np.dtype(storage["dtype"]), n_split_rows=len(idx),
            patch=PATCH, patch_file=patch_file, n_patch_rows=n_patch_rows,
            norm_mean=np.asarray(norm["per_channel_mean"], np.float32),
            norm_std=np.asarray(norm["per_channel_std"], np.float32),
        )

    # -- geometry ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def latent_dim(self) -> int:
        return self.latent_channels * int(np.prod(self.latent_spatial))

    @property
    def grey_dim(self) -> int:
        return self.patch ** 3

    # -- chunk readers -----------------------------------------------------

    def latent_memmap(self):
        return np.memmap(
            str(self.root / self.split / "latents.bin"), dtype=self.latent_dtype,
            mode="r",
            shape=(self.n_split_rows, 2 * self.latent_channels, *self.latent_spatial),
        )

    def patch_memmap(self):
        return np.memmap(
            str(self.patch_file), dtype=np.uint8, mode="r",
            shape=(self.n_patch_rows, self.patch, self.patch, self.patch),
        )

    def raw_latent_at(self, positions: np.ndarray, memmap=None) -> np.ndarray:
        """``(n, C, d, h, w)`` posterior means in the units the VAE emits.

        This is what the decoder eats.  Everything that MEASURES a latent uses
        :meth:`latent_at` instead, which normalises.
        """
        store = self.latent_memmap() if memmap is None else memmap
        pos = np.asarray(positions, np.int64)
        return np.asarray(store[self.rows[pos], :self.latent_channels], np.float32)

    def latent_at(self, positions: np.ndarray, memmap=None) -> np.ndarray:
        """``(n, latent_dim)`` per-channel-normalised posterior means.

        ``positions`` index :attr:`rows`, not the split.  Normalised with the
        store's own train-split statistics, so every latent channel contributes
        to the L2 on the same scale: raw channel standard deviations on this
        store span 0.41 to 0.78, and a raw L2 would be a distance in whichever
        channel happens to be widest.
        """
        mu = self.raw_latent_at(positions, memmap)
        mu -= self.norm_mean[None, :, None, None, None]
        mu /= self.norm_std[None, :, None, None, None]
        return mu.reshape(len(mu), -1)

    def grey_at(self, positions: np.ndarray, memmap=None) -> np.ndarray:
        """``(n, grey_dim)`` raw source grey in [0, 1] for those rows."""
        store = self.patch_memmap() if memmap is None else memmap
        pos = np.asarray(positions, np.int64)
        block = np.asarray(store[self.source_rows[pos]], np.float32) / 255.0
        return block.reshape(len(pos), -1)

    def latent_chunk(self, lo: int, hi: int, memmap=None) -> np.ndarray:
        return self.latent_at(np.arange(lo, hi), memmap)

    def grey_chunk(self, lo: int, hi: int, memmap=None) -> np.ndarray:
        return self.grey_at(np.arange(lo, hi), memmap)

    # -- dropping pages behind the read ------------------------------------
    #
    # THIS MACHINE HAS 121 GB OF UNIFIED MEMORY SHARED BY CPU AND GPU.  Page
    # cache is reclaimable in principle, but a CUDA allocation does not wait
    # for the kernel to reclaim it: it fails.  A streaming pass over a 195 GiB
    # store will fill the cache in minutes, and "available" stays high the
    # whole time while CUDA jobs die.  So the pass drops its own pages as it
    # goes, and the resident file-backed set stays bounded by the chunk size
    # instead of growing to the size of the store.

    def _release_span(self, path: Path, file_rows: np.ndarray, row_bytes: int,
                      memmap=None) -> None:
        """Drop the page cache for the byte span these rows occupy.

        The bank rows are the stride-64 subset, so a chunk of positions covers
        one contiguous SPAN of the file with unwanted rows interleaved.  The
        whole span is released: the interleaved rows are not in the bank and
        are not wanted either.

        `madvise` first, to drop the pages this process has mapped, then
        `posix_fadvise` on the file, which is what actually evicts them —
        fadvise cannot free a page another mapping still holds.  Both are
        advisory and both are allowed to fail; the search is still correct if
        nothing is released, only hungrier.
        """
        if file_rows.size == 0:
            return
        lo = int(file_rows.min()) * row_bytes
        hi = (int(file_rows.max()) + 1) * row_bytes
        handle = getattr(memmap, "_mmap", None)
        if handle is not None:
            page = mmap.PAGESIZE
            off = (lo // page) * page
            try:
                handle.madvise(mmap.MADV_DONTNEED, off, hi - off)
            except (OSError, ValueError, AttributeError):
                pass
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, lo, hi - lo, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    def release_latent_chunk(self, lo: int, hi: int, memmap=None) -> None:
        row_bytes = (2 * self.latent_channels * int(np.prod(self.latent_spatial))
                     * self.latent_dtype.itemsize)
        self._release_span(self.root / self.split / "latents.bin",
                           self.rows[lo:hi], row_bytes, memmap)

    def release_grey_chunk(self, lo: int, hi: int, memmap=None) -> None:
        self._release_span(self.patch_file, self.source_rows[lo:hi],
                           self.patch ** 3, memmap)

    def normalise_latents(self, mu: np.ndarray) -> np.ndarray:
        """Put raw encoder means on the store's normalised scale."""
        if mu.shape[1] != self.latent_dim:
            raise ValueError(
                f"the VAE produced {mu.shape[1]}-dimensional latents and the "
                f"store holds {self.latent_dim}-dimensional ones. The store was "
                "built by a different VAE, so a distance between them would be "
                "meaningless."
            )
        n = len(mu)
        z = mu.reshape(n, self.latent_channels, *self.latent_spatial).astype(np.float32)
        z = (z - self.norm_mean[None, :, None, None, None]) \
            / self.norm_std[None, :, None, None, None]
        return z.reshape(n, -1)


# ---------------------------------------------------------------------------
# The streaming search
# ---------------------------------------------------------------------------

def refine(acc: Top2, queries: np.ndarray, store: PatchStore, space: str,
           block: int = 64) -> None:
    """Recompute the two chosen distances exactly, in place.

    :func:`squared_distances` ranks well and rounds badly (see its docstring),
    and the whole statistic is a ratio of two distances that can both be tiny.
    Only ``2 * n_query`` bank rows are involved once the search has chosen
    them, so an explicit float64 difference is affordable here and nowhere
    else.
    """
    reader = store.latent_at if space == "latent" else store.grey_at
    memmap = (store.latent_memmap if space == "latent" else store.patch_memmap)()
    for lo in range(0, len(queries), block):
        hi = min(lo + block, len(queries))
        q = np.asarray(queries[lo:hi], np.float64)
        for dist, idx in ((acc.d1, acc.i1), (acc.d2, acc.i2)):
            rows = np.asarray(reader(idx[lo:hi], memmap), np.float64)
            dist[lo:hi] = np.linalg.norm(q - rows, axis=1)


def search(
    queries: np.ndarray,
    store: PatchStore,
    space: str,
    *,
    device=None,
    chunk: int | None = None,
) -> Top2:
    """Nearest and second-nearest bank row of every query, over the WHOLE bank.

    One pass over the store; the bank never exists in full.  ``space`` is
    ``"latent"`` or ``"grey"`` and picks the chunk reader and the default chunk
    size — the two spaces differ by a factor of eight in width, so they cannot
    share one.  The two distances that come back are :func:`refine`'d, so they
    do not depend on the chunk size the machine happened to allow.
    """
    import torch  # noqa: PLC0415

    readers = {
        "latent": (store.latent_chunk, store.latent_memmap,
                   store.release_latent_chunk, LATENT_BANK_CHUNK),
        "grey": (store.grey_chunk, store.patch_memmap,
                 store.release_grey_chunk, GREY_BANK_CHUNK),
    }
    if space not in readers:
        raise KeyError(f"space must be one of {sorted(readers)}, got {space!r}")
    reader, mapper, release, default_chunk = readers[space]
    chunk = int(chunk or default_chunk)

    expected = store.latent_dim if space == "latent" else store.grey_dim
    if queries.shape[1] != expected:
        raise ValueError(
            f"{space} queries are {queries.shape[1]}-dimensional but the store's "
            f"rows are {expected}-dimensional. The store was built by a different "
            "VAE or on a different patch size, so a distance between them would "
            "be meaningless."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(np.ascontiguousarray(queries, np.float32)).to(device)
    q_sq = (q * q).sum(1, keepdim=True)
    acc = Top2(len(queries))
    memmap = mapper()
    n = len(store)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        bank = torch.from_numpy(reader(lo, hi, memmap)).to(device)
        d2 = squared_distances(q, bank, q_sq)
        k = min(2, hi - lo)
        vals, idx = torch.topk(d2, k, dim=1, largest=False)
        acc.update_candidates(
            vals.sqrt().double().cpu().numpy(),
            (idx.cpu().numpy().astype(np.int64) + lo),
        )
        del bank, d2
        release(lo, hi, memmap)
        if lo // chunk % 50 == 0:
            logger.info("memorisation %s search: %d/%d bank rows", space, hi, n)
    refine(acc, queries, store, space)
    return acc


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(acc: Top2, select: np.ndarray | None = None) -> dict:
    """The readable form of one accumulator, over all queries or a subset."""
    sel = slice(None) if select is None else np.asarray(select, bool)
    d1, d2 = acc.d1[sel], acc.d2[sel]
    if len(d1) == 0:
        return {"n": 0}
    ratio = favero_ratio(d1, d2)
    memorised = ratio < RATIO_THRESHOLD
    return {
        "n": int(len(d1)),
        "nn_distance_mean": float(d1.mean()),
        "nn_distance_min": float(d1.min()),
        "second_nn_distance_mean": float(d2.mean()),
        "ratio_mean": float(ratio.mean()),
        "ratio_median": float(np.median(ratio)),
        "ratio_min": float(ratio.min()),
        "ratio_p5": float(np.percentile(ratio, 5)),
        "n_memorised": int(memorised.sum()),
        "frac_memorised": float(memorised.mean()),
    }


def _spaces(accs: dict[str, Top2], select: np.ndarray | None = None) -> dict:
    return {space: summarise(acc, select) for space, acc in accs.items()}


# ---------------------------------------------------------------------------
# The entry point the measure stage calls
# ---------------------------------------------------------------------------

def gpu_jobs_other_than(pid: int) -> list[tuple[int, str]]:
    """CUDA processes on the card that are not `pid` or its children.

    THE STORE MAY NOT BE STREAMED WHILE THE CARD IS GENERATING.  GB10 has
    121 GB of unified memory shared by CPU and GPU: a streaming pass fills the
    page cache, and a CUDA allocation does not wait for the kernel to reclaim
    it — it fails.  This has already cost generation runs on this machine
    (ldm06 run note, incident 3), and `free` reports tens of GB "available"
    throughout, so the symptom never points at the cause.

    The check is here rather than in the queue script because the rule has to
    hold for a hand-run too: that is exactly how it was broken.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    mine = {pid}
    jobs = []
    for row in out.splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        other = int(parts[0])
        if other in mine:
            continue
        try:
            if int(Path(f"/proc/{other}/stat").read_text().split()[3]) == pid:
                continue                      # our own child
        except (OSError, IndexError, ValueError):
            pass
        jobs.append((other, parts[1]))
    return jobs


def memorisation(
    root: str | Path,
    *,
    repo: str | Path | None = None,
    device=None,
    assessments: tuple[str, ...] = ASSESSMENTS,
    max_tiles: int = MAX_TILES_PER_VOLUME,
    max_cases_per_assessment: int | None = None,
    n_floor: int = FLOOR_PATCHES,
    seed: int = 0,
    allow_busy_gpu: bool = False,
) -> dict:
    """Full-store memorisation of every generated volume small enough to search.

    The latent store is taken from the manifests of the volumes themselves and
    from nowhere else: the check compares against the TRAINING latents, so a
    guessed store would answer a different question and still print a number.

    A missing store or a missing VAE checkpoint is reported with the path it
    looked for.  A silent zero here would read as "no memorisation", which is
    the one wrong answer this function must never give.

    ``max_cases_per_assessment`` keeps the first N volumes of each assessment
    and records the cut in ``max_cases_per_assessment`` on the result, so a
    short run cannot be mistaken for the full one.  It exists for the smoke
    test that proves the hardware path — the VAE load, the encode, the decode
    and both store passes — before the full search is given the GPU for hours.
    """
    import torch  # noqa: PLC0415

    # This pass streams a 195 GiB store.  On unified memory the page cache it
    # fills makes a CUDA allocation FAIL rather than wait for reclaim, so a
    # generation running beside it dies — and `free` reports tens of GB
    # available throughout, so the symptom never points here.  Reported as
    # unavailable rather than raised: the other four statistics of assessment 8
    # are unaffected and should still be measured, and a reason that names the
    # blocking process is recoverable by re-running when the card is idle.
    busy = [] if allow_busy_gpu else gpu_jobs_other_than(os.getpid())
    if busy:
        names = ", ".join(f"{p} ({n})" for p, n in busy)
        logger.warning(
            "memorisation SKIPPED: the card is busy with %s. Re-run "
            "`eval_v4 measure microstructure` when it is idle.", names)
        return {
            "available": False,
            "blocked_by": [{"pid": p, "process": n} for p, n in busy],
            "reason": (
                f"the card is busy with {names}. This search streams a 195 GiB "
                "store and the page cache it fills makes CUDA allocations fail "
                "on unified memory, so it must not run beside a generating "
                "job. Re-run `eval_v4 measure microstructure` when the card is "
                "idle, or pass --allow-busy-gpu where host and device memory "
                "are separate pools."
            ),
        }

    cases, skipped, present = [], [], []
    for assessment in assessments:
        found = load_cases(root, assessment)
        if found:
            present.append(assessment)
        if max_cases_per_assessment is not None:
            found = found[:max_cases_per_assessment]
        for case in found:
            m = case.manifest
            if m.sampler == "real":
                continue
            tiles = int(np.prod([s // PATCH for s in m.volume_shape]))
            if tiles > max_tiles:
                skipped.append({"case": m.case, "assessment": assessment,
                                "volume_shape": list(m.volume_shape),
                                "tiles": tiles})
                continue
            cases.append(case)
    if not cases:
        return {"available": False,
                "reason": f"no generated volume of at most {max_tiles} tiles under "
                          f"{root} in {list(assessments)}"}

    store_path = (cases[0].manifest.notes or {}).get("latents_root")
    if not store_path:
        raise KeyError(
            f"{cases[0].manifest.case}: manifest notes carry no latents_root, so "
            "the store these volumes came from is unknown and the memorisation "
            "check cannot be run. Regenerate the assessment."
        )
    latents_root = Path(store_path)
    if not (latents_root / "metadata.json").exists():
        return {"available": False, "latents_root": str(latents_root),
                "reason": f"no latent store at {latents_root} \u2014 the memorisation "
                          "check needs the training latents the LDM was trained on"}

    meta = json.loads((latents_root / "metadata.json").read_text())
    ckpt = Path(meta["vae_checkpoint"])
    if not ckpt.is_absolute():
        ckpt = Path(repo or ".") / ckpt
    if not ckpt.exists():
        return {"available": False, "latents_root": str(latents_root),
                "reason": f"the store names VAE checkpoint {ckpt}, which is not there"}

    train = PatchStore.open(latents_root, "train")
    val = PatchStore.open(latents_root, "val")

    from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: PLC0415

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, _, _, _ = load_vae_from_checkpoint(ckpt, device)
    vae.requires_grad_(False)
    vae.eval()

    # -- the generated queries ---------------------------------------------
    q_latent, q_grey = [], []
    case_of: list[str] = []
    phi_of: list[float] = []
    bucket_of: list[str] = []
    per_case: dict[str, dict] = {}
    coarse_bucketed: list[str] = []
    for case in cases:
        m = case.manifest
        m.require(("requested_global_phi",), "memorisation")
        origins = patch_origins(case.material_voxels())
        win = window_states(m)
        census = win.census()
        mu = encode_patches(vae, case.xct, case.label, origins, device=device)
        if mu.size == 0:
            per_case[m.case] = {"n_patches": 0, "neighbours": census}
            continue
        per_patch = [win.at_voxel(o) for o in origins]
        # A patch position that was never a window origin has no honest
        # per-window state.  The volume-level bucket is coarser but true, and
        # the fallback is recorded rather than hidden: a bucket label that is
        # subtly wrong is worse than one that is openly coarse.
        fell_back = any(w is None for w in per_patch)
        if fell_back:
            coarse_bucketed.append(m.case)
        buckets = [census["bucket"] if w is None else w["bucket"] for w in per_patch]

        q_latent.append(train.normalise_latents(mu))
        q_grey.append(grey_patches(case.xct, origins))
        case_of += [m.case] * len(origins)
        phi_of += [float(m.requested_global_phi)] * len(origins)
        bucket_of += buckets
        per_case[m.case] = {
            "n_patches": len(origins),
            "requested_phi": float(m.requested_global_phi),
            "assessment": m.assessment,
            "volume_shape": list(m.volume_shape),
            "ddim_steps": m.ddim_steps,
            "seed": m.seed,
            "neighbours": census,
            "n_patches_with_unknown_face": sum(
                b == NB_UNKNOWN_BUCKET for b in buckets),
            "per_patch_state": "volume-level fallback" if fell_back else "per-window",
        }
        # 57 volumes at 192 cubed plus five at 384 is gigabytes of cached
        # arrays that nothing reads again, on a machine that is also generating.
        case.__dict__.pop("xct", None)
        case.__dict__.pop("label", None)
    if not q_latent:
        return {"available": False, "latents_root": str(latents_root),
                "reason": "no whole-material patch in any generated volume"}

    gen_latent = np.concatenate(q_latent)
    gen_grey = np.concatenate(q_grey)
    case_arr = np.asarray(case_of)
    phi_arr = np.asarray(phi_of, np.float64)
    bucket_arr = np.asarray(bucket_of)

    # -- the real-val floor, drawn from the store's own val split ----------
    # THREE query sets, not two.  In latent space the round trip is the
    # identity, so there is one floor.  In grey space there are two, and which
    # is which matters: `roundtrip` decodes the val patch's own stored mu, so
    # it carries the same decoder signature the generated query does and is
    # the like-for-like floor; `raw` is the untouched scan patch and is what
    # the ratio would be with no decoder error at all.  Together they bound
    # the answer from both sides.
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(val), size=min(int(n_floor), len(val)),
                              replace=False))
    val_latent = val.latent_at(pick)
    val_grey_raw = val.grey_at(pick)
    val_grey_rt = decode_grey(vae, val.raw_latent_at(pick), device=device)

    # -- one pass per space, over the whole train bank ---------------------
    n_gen = len(gen_latent)
    gen_acc: dict[str, Top2] = {}
    floor_rt: dict[str, Top2] = {}
    floor_raw: dict[str, Top2] = {}
    plan = (
        ("latent", gen_latent, val_latent, None),
        ("grey", gen_grey, val_grey_rt, val_grey_raw),
    )
    for space, q_gen, q_rt, q_raw in plan:
        blocks = [q_gen, q_rt] + ([] if q_raw is None else [q_raw])
        acc = search(np.concatenate(blocks), train, space, device=device)
        gen_acc[space] = _slice(acc, np.arange(n_gen))
        floor_rt[space] = _slice(acc, np.arange(n_gen, n_gen + len(q_rt)))
        if q_raw is not None:
            base = n_gen + len(q_rt)
            floor_raw[space] = _slice(acc, np.arange(base, base + len(q_raw)))

    for space, acc in gen_acc.items():
        for name in sorted(set(case_arr)):
            per_case.setdefault(name, {}).setdefault("spaces", {})[space] = \
                summarise(acc, case_arr == name)

    floor = {"n_patches": int(len(val_latent)), **_spaces(floor_rt)}
    floor["grey_raw"] = summarise(floor_raw["grey"])
    floor["note"] = (
        "`latent` and `grey` are the LIKE-FOR-LIKE floor: the val patch's own "
        "stored posterior mean, decoded by the same frozen VAE the generated "
        "volume was decoded by, so the floor carries the same reconstruction "
        "error the query does. `grey_raw` is the same val patches untouched, "
        "which is what the ratio would be with no decoder error at all. Read "
        "the generated grey row against `grey`; `grey_raw` bounds it from the "
        "other side."
    )

    return {
        "available": True,
        "latents_root": str(latents_root),
        "vae_checkpoint": str(ckpt),
        "criterion": {
            "statistic": "||x - x'|| / ||x - x''||  (nearest over second-nearest)",
            "threshold": RATIO_THRESHOLD,
            "reading": (
                "Below the threshold the patch is a copy: it sits on one training "
                "patch and a normal distance from every other. Near 1 it is a "
                "fresh sample. The number is only readable against the real-val "
                "floor in the same space, because real held-out material also has "
                "near neighbours in the train set."
            ),
        },
        "bank": {
            "split": "train",
            "stride": BANK_STRIDE,
            "n_rows": len(train),
            "n_rows_in_split": train.n_split_rows,
            "latent_dim": train.latent_dim,
            "grey_dim": train.grey_dim,
            "latent_scale": "per-channel normalised with the store's train stats",
            "grey_source": str(train.patch_file),
        },
        "assessments_requested": list(assessments),
        "assessments_found": present,
        "max_tiles_per_volume": int(max_tiles),
        "max_cases_per_assessment": max_cases_per_assessment,
        "n_cases": len(per_case),
        "n_patches": int(n_gen),
        "skipped_too_large": skipped,
        "neighbour_state_source": (
            "rebuilt per window from volume_shape + chunk_tiles + window_stride "
            "with the sampler's own chunk order and _neighbour_plan rule"
        ),
        "cases_bucketed_at_volume_level": sorted(coarse_bucketed),
        "generated": _spaces(gen_acc),
        "real_val_floor": floor,
        "by_requested_phi": {
            f"{phi:g}": {"n_patches": int((phi_arr == phi).sum()),
                         **_spaces(gen_acc, phi_arr == phi)}
            for phi in sorted(set(phi_arr.tolist()))
        },
        "by_neighbours": {
            bucket: {"n_patches": int((bucket_arr == bucket).sum()),
                     **_spaces(gen_acc, bucket_arr == bucket)}
            for bucket in sorted(set(bucket_arr.tolist()))
        },
        "per_case": per_case,
    }


def _slice(acc: Top2, take: np.ndarray) -> Top2:
    """The sub-accumulator for a contiguous block of the stacked query set."""
    out = Top2(len(take))
    out.d1, out.d2 = acc.d1[take].copy(), acc.d2[take].copy()
    out.i1, out.i2 = acc.i1[take].copy(), acc.i2[take].copy()
    return out

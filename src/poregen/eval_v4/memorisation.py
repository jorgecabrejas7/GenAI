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

**The floor is real held-out material.**  Real validation patches are not
memorised by construction — they come from panels the VAE and the LDM never
saw — yet they still have near neighbours in the train set, because the panels
are made of the same material.  What the val patches score is therefore the
value the ratio takes when nothing has been copied, and the generated number is
only readable beside it.  Without that floor a generated ratio of 0.6 means
nothing at all.

Failure modes, stated plainly
-----------------------------
* **The grey bank is raw scan data; the generated query is decoded.**  A
  generated patch reaches grey space through the VAE decoder and therefore
  carries the decoder's own reconstruction error, which a raw validation patch
  does not.  A generated volume that reproduced a training latent EXACTLY would
  still sit one reconstruction error away from the real patch, so the grey
  ratio is an upper bound on how memorised that volume is.  The latent ratio is
  the sharp one; grey is the corroborating pixel-level check, and the two are
  reported side by side for that reason.
* **The query is re-encoded, not taken from ``latents.npy``.**  ldm06 trains on
  a posterior DRAW (``data.latent_mode: sampled``), so a generated latent lives
  in sample space while the store holds posterior MEANS.  Comparing the two
  directly would add ``sigma*eps`` to every distance.  The generated volume is
  therefore pushed back through the VAE encoder, so both sides of the latent
  distance are means of the same encoder.
* **The bank is every stride-64 train row**, air-heavy rows included.  They are
  far from any whole-material query and cannot become its nearest neighbour;
  filtering them would only cost a pass over ``material.bin``.
"""

from __future__ import annotations

import json
import logging
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

#: The assessments whose generated volumes are searched, and the only shape
#: they are searched at.  1024-wide volumes are excluded on cost, not on
#: principle: a 192-cubed volume is 27 query patches, a 1024-wide one is 768.
ASSESSMENTS = ("sampler", "porosity_global")
QUERY_SHAPE = (192, 192, 192)

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


# ---------------------------------------------------------------------------
# Neighbour availability, from the manifest alone
# ---------------------------------------------------------------------------

def neighbour_census(manifest) -> dict:
    """How many window faces saw a neighbour, and how many saw nothing.

    The sampler decides this per window and does not record it, so it is
    rebuilt here from the three manifest fields that determine it —
    ``volume_shape``, ``chunk_tiles`` and ``window_stride`` — using the
    sampler's own window grid and neighbour directions.  The rule is the
    sampler's: a face is OOB when the neighbour block leaves the canvas,
    EXISTS when it lies inside a finished chunk or inside the chunk being
    generated, and UNKNOWN when it reaches into a chunk that has not been
    generated yet.

    A volume with no UNKNOWN face never had to invent neighbour context; one
    with UNKNOWN faces did.  That is the split the memorisation result is
    broken down by, because a model handed real neighbour context has more to
    copy from than one working blind.
    """
    from poregen.diffusion.conditioning import (  # noqa: PLC0415
        NB_EXISTS, NB_OOB, NB_UNKNOWN, NEIGHBOUR_DIRS,
    )
    from poregen.diffusion.sampler import window_origins  # noqa: PLC0415

    if manifest.chunk_tiles is None or manifest.window_stride is None:
        raise ValueError(
            f"{manifest.case}: the manifest carries no chunk_tiles/window_stride, "
            "so the neighbour census cannot be rebuilt. Only a generated volume "
            "has one."
        )
    if manifest.region_offset is not None:
        raise ValueError(
            f"{manifest.case}: the request was translated inside a larger canvas "
            "(region_offset is set), so volume_shape is not the canvas the "
            "sampler ran on and the census would be wrong."
        )

    cells = tuple(int(s) // LATENT_DOWNSAMPLE for s in manifest.volume_shape)
    win = PATCH // LATENT_DOWNSAMPLE
    stride = int(manifest.window_stride) // LATENT_DOWNSAMPLE
    n_tiles = tuple(int(s) // PATCH for s in manifest.volume_shape)
    chunk_tiles = tuple(int(c) for c in manifest.chunk_tiles)

    chunk_grid = [
        [(i, min(i + chunk_tiles[a], n_tiles[a]))
         for i in range(0, n_tiles[a], chunk_tiles[a])]
        for a in range(3)
    ]
    chunks = [(cz, cy, cx) for cz in chunk_grid[0]
              for cy in chunk_grid[1] for cx in chunk_grid[2]]

    available = np.zeros(cells, bool)
    counts = {NB_OOB: 0, NB_EXISTS: 0, NB_UNKNOWN: 0}
    for chunk in chunks:
        lo = tuple(chunk[a][0] * win for a in range(3))
        hi = tuple(chunk[a][1] * win for a in range(3))
        chunk_sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        visible = available.copy()
        visible[chunk_sl] = True
        chunk_cells = tuple(hi[a] - lo[a] for a in range(3))
        for o in window_origins(chunk_cells, win, stride):
            g = tuple(lo[a] + o[a] for a in range(3))
            for d in NEIGHBOUR_DIRS:
                a_lo = tuple(g[k] + d[k] * win for k in range(3))
                if any(a_lo[k] < 0 or a_lo[k] + win > cells[k] for k in range(3)):
                    counts[NB_OOB] += 1
                    continue
                block = tuple(slice(a_lo[k], a_lo[k] + win) for k in range(3))
                counts[NB_EXISTS if visible[block].all() else NB_UNKNOWN] += 1
        available[chunk_sl] = True

    return {
        "n_chunks": len(chunks),
        "exists": counts[NB_EXISTS],
        "oob": counts[NB_OOB],
        "unknown": counts[NB_UNKNOWN],
        "bucket": NB_UNKNOWN_BUCKET if counts[NB_UNKNOWN] else NB_PRESENT,
    }


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

    def latent_at(self, positions: np.ndarray, memmap=None) -> np.ndarray:
        """``(n, latent_dim)`` per-channel-normalised posterior means.

        ``positions`` index :attr:`rows`, not the split.  Normalised with the
        store's own train-split statistics, so every latent channel contributes
        to the L2 on the same scale: raw channel standard deviations on this
        store span 0.41 to 0.78, and a raw L2 would be a distance in whichever
        channel happens to be widest.
        """
        store = self.latent_memmap() if memmap is None else memmap
        pos = np.asarray(positions, np.int64)
        mu = np.asarray(store[self.rows[pos], :self.latent_channels], np.float32)
        mu -= self.norm_mean[None, :, None, None, None]
        mu /= self.norm_std[None, :, None, None, None]
        return mu.reshape(len(pos), -1)

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

    readers = {"latent": (store.latent_chunk, store.latent_memmap, LATENT_BANK_CHUNK),
               "grey": (store.grey_chunk, store.patch_memmap, GREY_BANK_CHUNK)}
    if space not in readers:
        raise KeyError(f"space must be one of {sorted(readers)}, got {space!r}")
    reader, mapper, default_chunk = readers[space]
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

def memorisation(
    root: str | Path,
    *,
    repo: str | Path | None = None,
    device=None,
    assessments: tuple[str, ...] = ASSESSMENTS,
    query_shape: tuple[int, int, int] = QUERY_SHAPE,
    n_floor: int = FLOOR_PATCHES,
    seed: int = 0,
) -> dict:
    """Full-store memorisation of every generated ``query_shape`` volume.

    The latent store is taken from the manifests of the volumes themselves and
    from nowhere else: the check compares against the TRAINING latents, so a
    guessed store would answer a different question and still print a number.

    A missing store or a missing VAE checkpoint is reported with the path it
    looked for.  A silent zero here would read as "no memorisation", which is
    the one wrong answer this function must never give.
    """
    import torch  # noqa: PLC0415

    cases, skipped = [], []
    for assessment in assessments:
        for case in load_cases(root, assessment):
            if case.manifest.sampler == "real":
                continue
            if tuple(case.manifest.volume_shape) != tuple(query_shape):
                skipped.append(case.manifest.case)
                continue
            cases.append(case)
    if not cases:
        return {"available": False,
                "reason": f"no {'x'.join(str(s) for s in query_shape)} generated "
                          f"volume under {root} in {list(assessments)}"}

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
                "reason": f"no latent store at {latents_root} — the memorisation "
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
    for case in cases:
        m = case.manifest
        m.require(("requested_global_phi",), "memorisation")
        origins = patch_origins(case.material_voxels())
        census = neighbour_census(m)
        mu = encode_patches(vae, case.xct, case.label, origins, device=device)
        if mu.size == 0:
            per_case[m.case] = {"n_patches": 0, "neighbours": census}
            continue
        q_latent.append(train.normalise_latents(mu))
        q_grey.append(grey_patches(case.xct, origins))
        case_of += [m.case] * len(origins)
        phi_of += [float(m.requested_global_phi)] * len(origins)
        bucket_of += [census["bucket"]] * len(origins)
        per_case[m.case] = {
            "n_patches": len(origins),
            "requested_phi": float(m.requested_global_phi),
            "assessment": m.assessment,
            "ddim_steps": m.ddim_steps,
            "seed": m.seed,
            "neighbours": census,
        }
        # 57 volumes at 192 cubed is 800 MB of cached arrays that nothing
        # reads again, on a machine that is also generating.
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
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(val), size=min(int(n_floor), len(val)),
                              replace=False))
    val_latent = val.latent_at(pick)
    val_grey = val.grey_at(pick)

    # -- one pass per space, over the whole train bank ---------------------
    n_gen = len(gen_latent)
    gen_acc, floor_acc = {}, {}
    for space, q_gen, q_val in (("latent", gen_latent, val_latent),
                                ("grey", gen_grey, val_grey)):
        acc = search(np.concatenate([q_gen, q_val]), train, space, device=device)
        gen_acc[space] = _slice(acc, np.arange(n_gen))
        floor_acc[space] = _slice(acc, np.arange(n_gen, n_gen + len(q_val)))

    for space, acc in gen_acc.items():
        for name in sorted(set(case_arr)):
            per_case.setdefault(name, {}).setdefault("spaces", {})[space] = \
                summarise(acc, case_arr == name)

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
        "assessments": list(assessments),
        "query_shape": list(query_shape),
        "n_cases": len(per_case),
        "n_patches": int(n_gen),
        "skipped_other_shape": sorted(skipped),
        "generated": _spaces(gen_acc),
        "real_val_floor": {"n_patches": int(len(val_latent)),
                           **_spaces(floor_acc)},
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

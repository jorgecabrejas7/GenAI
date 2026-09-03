"""Load side of the latent dataset (ldm06 store + conditioning).

Reads the layout produced by ``scripts/build_latent_dataset.py`` plus the
conditioning sidecar produced by ``scripts/build_conditioning.py``::

    <root>/                      e.g. data/split_v3/latents_r08z4/
    ├── metadata.json            — latent shape, storage record, per-channel
    │                              train-split normalisation stats, VAE
    │                              provenance, `conditioning` block
    └── <split>/
        ├── latents.bin          — float16 C-contiguous memmap,
        │                          shape (N, 2C, d, h, w); channels 0..C-1 = mu,
        │                          C..2C-1 = std ("mu_then_std" packing —
        │                          ONE contiguous read per item)
        ├── index.parquet        — source_row, volume_id, z0, y0, x0, phi, …
        ├── cond.parquet         — cond_depth, cond_dist6_*, cond_por_raw,
        │                          row-aligned with index.parquet
        ├── material.bin         — uint8 (N, d, h, w); value/255 = the specimen
        │                          ENVELOPE fraction (label != 2) of each cell
        └── air.bin              — float32 (N,); air fraction (label == 2),
                                   equal to 1 - material.mean()

Row i of ``index.parquet``, ``cond.parquet``, ``material.bin``, ``air.bin``
and ``latents.bin`` all align.

With ``with_label=True`` the dataset also serves the source patch's 3-class
voxel label.  That array is NOT part of the store: it is
``patches_label.bin`` in the data root recorded as
``metadata["source_patch_index"]``, addressed by ``source_row`` and
row-aligned with ``patch_index.parquet`` by construction.  Only the decoded
auxiliary loss needs it, and it is 64x the size of a latent, so it stays
opt-in.

Latents are stored raw; :class:`LatentDataset` optionally applies per-channel
normalisation ``(mu - mean_c) / std_c`` at load time.  The posterior std is
scaled by ``1 / std_c`` under the same affine map.

Three distinct strides (conflating them was a documented bug):

``sample_stride``
    Spacing of the patches actually stored in this dataset (32).  It is purely
    a training-data multiplier: at ``neighbour_offset = 64`` the store holds
    eight interleaved copies of the stride-64 grid, and each patch's six face
    neighbours are two sampling steps away, so they already exist without a
    rebuild.
``generation_stride``
    Spacing of the tiling grid the sampler decodes on (64 = ``patch_size``).
``neighbour_offset``
    Spatial displacement, in voxels, of a face neighbour from the target (64).
    Must equal ``generation_stride`` and must be at least ``patch_size``:
    below that a neighbour shares voxels with the target and leaks the answer
    (see ``poregen.diffusion.conditioning``).

Neighbours are fed WHOLE and UNSHIFTED — touching patches share no cell, so
the entire neighbour latent is honest face-adjacent context.  The latents this
dataset serves are the CLEAN posterior means of the stored neighbours; the
training step is what noises them to their own timestep, and the sampler is
what replaces them with canvas state.  Availability here is only ever EXISTS
(the store holds a patch there) or OOB (it does not — the specimen ends).
UNKNOWN is produced by the training step's neighbour dropout and by the
sampler, never by the store.

Worker safety: the memmaps are opened read-only once in ``__init__``; after
``fork()`` each DataLoader worker inherits the mapping, which is safe for
read-only access.  The neighbour lookup is a pair of flat numpy arrays for the
same reason — a python dict of 1.8 M keys would be copied page by page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .conditioning import (
    DIST6_NAMES,
    N_NEIGHBOURS,
    NB_EXISTS,
    NB_OOB,
    NEIGHBOUR_DIRS,
    grid_index,
    validate_neighbour_geometry,
)
from .orientation import OrientationField

logger = logging.getLogger(__name__)

_KEY_BASE = 8192           # per-axis key radix; every coordinate must be < this

_DIST6_COLUMNS = tuple(f"cond_dist6_{n}" for n in DIST6_NAMES)


class LatentDataset(Dataset):
    """Random-access dataset of pre-computed VAE posterior latents.

    Returns the fixed ldm06 batch contract; see :meth:`__getitem__`.

    Parameters
    ----------
    root : str | Path
        Latent dataset root (contains ``metadata.json`` and split dirs).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    normalize : bool
        If True, return per-channel normalised latents using the train-split
        stats stored in ``metadata.json``.  Neighbour latents use the same map.
    orientation_field : str | Path | None
        Path to ``orientation_field.json``.  Defaults to the path recorded in
        the store metadata, resolved relative to the repo root.
    sample_stride, generation_stride, neighbour_offset : int
        See the module docstring.  Kept separate on purpose.
    with_label : bool
        Also serve the source patch's 3-class voxel label, read from
        ``patches_label.bin`` in the data root the store was built from.  Off
        by default: the label is 256 KB per item and only the decoded
        auxiliary loss (``loss.decoded.enabled``) consumes it, so paying for
        it on every ldm06 run would be a pure loader tax.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        normalize: bool = False,
        orientation_field: str | Path | None = None,
        sample_stride: int = 32,
        generation_stride: int = 64,
        neighbour_offset: int = 64,
        with_label: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.sample_stride = int(sample_stride)
        self.generation_stride = int(generation_stride)
        self.neighbour_offset = int(neighbour_offset)
        self.with_label = bool(with_label)

        with open(self.root / "metadata.json") as fh:
            self.metadata: dict[str, Any] = json.load(fh)

        norm = self.metadata["normalization"]
        c = len(norm["per_channel_mean"])
        self.channel_mean = torch.tensor(
            norm["per_channel_mean"], dtype=torch.float32
        ).view(c, 1, 1, 1)
        self.channel_std = torch.tensor(
            norm["per_channel_std"], dtype=torch.float32
        ).view(c, 1, 1, 1)

        storage = self.metadata["storage"]
        if storage["pack_scheme"] != "mu_then_std":
            raise ValueError(
                f"LatentDataset [{split}]: unsupported pack_scheme "
                f"{storage['pack_scheme']!r} — expected 'mu_then_std'."
            )
        self.latent_shape: tuple[int, ...] = tuple(self.metadata["latent_shape"])
        self.patch_size = int(self.metadata.get("patch_size", 64))
        self.latent_size = int(self.latent_shape[-1])
        self.voxel_size_um = float(self.metadata["voxel_size_um"])
        dtype = np.dtype(storage["dtype"])
        item_shape = (2 * self.latent_shape[0], *self.latent_shape[1:])

        split_dir = self.root / split
        self.df = pd.read_parquet(str(split_dir / "index.parquet"))
        n = len(self.df)

        bin_path = split_dir / "latents.bin"
        expected_bytes = n * int(np.prod(item_shape)) * dtype.itemsize
        actual_bytes = bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"LatentDataset [{split}]: index.parquet has {n} rows "
                f"({expected_bytes} bytes expected) but latents.bin is "
                f"{actual_bytes} bytes — rebuild the store."
            )
        self._latents = np.memmap(
            str(bin_path), dtype=dtype, mode="r", shape=(n, *item_shape)
        )

        # ── flat per-row arrays (df.iloc per item is far too slow) ────────────
        self._z0 = self.df["z0"].to_numpy(np.int64)
        self._y0 = self.df["y0"].to_numpy(np.int64)
        self._x0 = self.df["x0"].to_numpy(np.int64)
        self._phi = self.df["phi"].to_numpy(np.float32)
        self._source_row = self.df["source_row"].to_numpy(np.int64)
        self._volume_id = self.df["volume_id"].to_numpy()

        for name, arr in (("z0", self._z0), ("y0", self._y0), ("x0", self._x0)):
            if arr.max(initial=0) >= _KEY_BASE:
                raise ValueError(
                    f"LatentDataset [{split}]: {name} reaches {arr.max()}, which "
                    f"overflows the neighbour-lookup radix {_KEY_BASE}."
                )
        for name, arr in (("z0", self._z0), ("y0", self._y0), ("x0", self._x0)):
            bad = int(np.count_nonzero(arr % self.sample_stride))
            if bad:
                raise ValueError(
                    f"LatentDataset [{split}]: {bad} rows have {name} not a "
                    f"multiple of sample_stride={self.sample_stride}."
                )
        # ── assembly geometry (see poregen.diffusion.conditioning) ───────────
        validate_neighbour_geometry(self.neighbour_offset, self.patch_size)
        if self.neighbour_offset != self.generation_stride:
            raise ValueError(
                f"LatentDataset [{split}]: neighbour_offset={self.neighbour_offset} != "
                f"generation_stride={self.generation_stride}.  The face neighbours of "
                f"the assembly grid sit exactly one generation_stride away, so any "
                f"other value would train a relation the sampler never reproduces."
            )
        if self.neighbour_offset % self.sample_stride != 0:
            raise ValueError(
                f"LatentDataset [{split}]: neighbour_offset={self.neighbour_offset} is "
                f"not a multiple of sample_stride={self.sample_stride}, so a face "
                f"neighbour does not land on a stored patch and no patch would ever "
                f"see a neighbour."
            )

        # ── conditioning sidecar ─────────────────────────────────────────────
        cond_path = split_dir / "cond.parquet"
        if not cond_path.exists():
            raise FileNotFoundError(
                f"LatentDataset [{split}]: missing {cond_path} — run "
                f"scripts/build_conditioning.py."
            )
        cond = pd.read_parquet(str(cond_path))
        if len(cond) != n:
            raise RuntimeError(
                f"LatentDataset [{split}]: cond.parquet has {len(cond)} rows but "
                f"index.parquet has {n} — rebuild the conditioning sidecar."
            )
        if not np.array_equal(cond["source_row"].to_numpy(np.int64), self._source_row):
            raise RuntimeError(
                f"LatentDataset [{split}]: cond.parquet is not row-aligned with "
                f"index.parquet — rebuild the conditioning sidecar."
            )
        missing_cols = [c for c in _DIST6_COLUMNS if c not in cond.columns]
        if missing_cols:
            raise RuntimeError(
                f"LatentDataset [{split}]: cond.parquet is missing {missing_cols}. "
                f"ldm06 conditions on six per-face distances, not one scalar "
                f"cond_dist — rebuild the sidecar with scripts/build_conditioning.py."
            )
        cond_meta = self.metadata["conditioning"]
        self._check_store_geometry(cond_meta)
        st = cond_meta["por_standardisation"]
        self.por_mean, self.por_std = float(st["mean"]), float(st["std"])
        self._cond_depth = cond["cond_depth"].to_numpy(np.float32)
        self._cond_dist6 = np.stack(
            [cond[c].to_numpy(np.float32) for c in _DIST6_COLUMNS], axis=1
        )                                                       # (N, 6)
        self._cond_por = (
            (cond["cond_por_raw"].to_numpy(np.float32) - self.por_mean) / self.por_std
        ).astype(np.float32)

        # ── orientation field ────────────────────────────────────────────────
        if orientation_field is None:
            rel = cond_meta["orientation_field"]
            orientation_field = Path(rel)
            if not orientation_field.is_absolute():
                orientation_field = _repo_root() / rel
        self.orientation = OrientationField(orientation_field)
        missing = {v for v in np.unique(self._volume_id) if v not in self.orientation}
        if missing:
            raise RuntimeError(
                f"LatentDataset [{split}]: orientation field has no record for "
                f"{sorted(missing)[:3]} ({len(missing)} volumes)."
            )

        # ── material map + air fraction sidecars ─────────────────────────────
        L = self.latent_size
        mat_path = split_dir / "material.bin"
        air_path = split_dir / "air.bin"
        for path, itemsize, per_row in ((mat_path, 1, L ** 3), (air_path, 4, 1)):
            expected = n * per_row * itemsize
            if not path.exists() or path.stat().st_size != expected:
                raise RuntimeError(
                    f"LatentDataset [{split}]: {path.name} missing or has "
                    f"{path.stat().st_size if path.exists() else 0} bytes "
                    f"(expected {expected}) — rebuild the store with "
                    f"scripts/build_latent_dataset.py."
                )
        self._material = np.memmap(str(mat_path), dtype=np.uint8, mode="r",
                                   shape=(n, L, L, L))
        self._air = np.memmap(str(air_path), dtype=np.float32, mode="r", shape=(n,))

        # ── voxel-label memmap (decoded auxiliary loss only) ─────────────────
        self._label = self._open_label_memmap() if self.with_label else None

        # ── neighbour lookup ─────────────────────────────────────────────────
        self._nb_dir_offsets = np.array(
            [[d[0], d[1], d[2]] for d in NEIGHBOUR_DIRS], dtype=np.int64
        ) * self.neighbour_offset
        codes, vol_codes = pd.factorize(self.df["volume_id"], sort=True)
        self._vol_index = codes.astype(np.int64)
        self._n_volumes = len(vol_codes)
        keys = self._encode_keys(self._vol_index, self._z0, self._y0, self._x0)
        order = np.argsort(keys, kind="stable")
        self._key_sorted = keys[order]
        self._key_rows = order.astype(np.int64)
        if np.any(np.diff(self._key_sorted) == 0):
            raise RuntimeError(
                f"LatentDataset [{split}]: duplicate (volume, z0, y0, x0) rows in "
                f"index.parquet — the neighbour lookup would be ambiguous."
            )

        logger.info(
            "LatentDataset [%s]: %d latents %s  normalize=%s  sample_stride=%d "
            "generation_stride=%d neighbour_offset=%d (touching, unshifted)",
            split, n, self.latent_shape, normalize, self.sample_stride,
            self.generation_stride, self.neighbour_offset,
        )

    # -- helpers -----------------------------------------------------------

    def _open_label_memmap(self) -> np.memmap:
        """Map ``patches_label.bin`` of the data root this store was built from.

        The store does not copy the voxel label — it is 256 KB per patch and
        the LDM does not normally need it.  What the store DOES record is
        ``metadata["source_patch_index"]``, the ``patch_index.parquet`` its
        rows point into through ``source_row``; ``patches_label.bin`` lives
        beside that file and is row-aligned with it by construction
        (``scripts/extract_patches_memmap.py``).  Deriving the path from the
        recorded provenance means the label can never come from a different
        dataset than the latents.

        The memmap covers the WHOLE patch index, not this split, so a row is
        addressed by ``source_row`` directly.
        """
        index_path = Path(self.metadata["source_patch_index"])
        data_root = index_path.parent
        label_path = data_root / "patches_label.bin"
        meta_path = data_root / "patches_meta.json"
        if not label_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"LatentDataset [{self.split}]: loss.decoded needs the voxel "
                f"label, but {label_path} or {meta_path} is missing.  The store "
                f"records its source index as {index_path}; run "
                f"scripts/extract_patches_memmap.py on that data root."
            )
        with open(meta_path) as fh:
            patch_meta = json.load(fh)
        n_all = int(patch_meta["N"])
        ps = int(patch_meta["patch_size"])
        if ps != self.patch_size:
            raise RuntimeError(
                f"LatentDataset [{self.split}]: {meta_path} says patch_size={ps} "
                f"but the latent store was built at {self.patch_size} — the two "
                f"were produced from different datasets."
            )
        expected = n_all * ps ** 3
        actual = label_path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"LatentDataset [{self.split}]: {label_path} is {actual} bytes, "
                f"expected {expected} for {n_all} patches of {ps}^3 uint8 — "
                f"re-extract it."
            )
        max_row = int(self._source_row.max(initial=0))
        if max_row >= n_all:
            raise RuntimeError(
                f"LatentDataset [{self.split}]: source_row reaches {max_row} but "
                f"{label_path.name} holds {n_all} patches — the store and the "
                f"patch index are out of step."
            )
        logger.info(
            "LatentDataset [%s]: serving voxel labels from %s (%d patches)",
            self.split, label_path, n_all,
        )
        return np.memmap(str(label_path), dtype=np.uint8, mode="r",
                         shape=(n_all, ps, ps, ps))

    def _check_store_geometry(self, cond_meta: dict[str, Any]) -> None:
        """Fail when the store was built for a different assembly geometry.

        ``scripts/build_conditioning.py`` records the strides it assumed in
        ``metadata["conditioning"]["geometry"]``.  Training against a store
        built for another neighbour relation would silently mis-condition
        every patch.
        """
        geom = cond_meta.get("geometry") or {}
        for key, ours in (
            ("sample_stride", self.sample_stride),
            ("generation_stride", self.generation_stride),
            ("neighbour_offset", self.neighbour_offset),
            ("patch_size", self.patch_size),
        ):
            theirs = geom.get(key)
            if theirs is not None and int(theirs) != int(ours):
                raise ValueError(
                    f"LatentDataset [{self.split}]: the store records "
                    f"conditioning.geometry.{key}={theirs} but this run asks for "
                    f"{ours}.  Re-run scripts/build_conditioning.py so the store "
                    f"metadata describes the geometry actually being trained."
                )

    @staticmethod
    def _encode_keys(vol: np.ndarray, z: np.ndarray, y: np.ndarray,
                     x: np.ndarray) -> np.ndarray:
        b = _KEY_BASE
        return (((vol.astype(np.int64) * b + z) * b + y) * b + x)

    def _lookup(self, vol: int, coords: np.ndarray) -> np.ndarray:
        """Row index of each (vol, z, y, x) in *coords*, or −1 when absent."""
        keys = self._encode_keys(
            np.full(len(coords), vol, np.int64),
            coords[:, 0], coords[:, 1], coords[:, 2],
        )
        pos = np.searchsorted(self._key_sorted, keys)
        ok = (pos < len(self._key_sorted))
        pos_c = np.where(ok, pos, 0)
        ok &= self._key_sorted[pos_c] == keys
        ok &= (coords >= 0).all(axis=1)
        return np.where(ok, self._key_rows[pos_c], -1)

    def grid_index(self, idx: int) -> tuple[int, int, int]:
        """Assembly-grid index ``(iz, iy, ix)`` of a row (shared convention)."""
        return grid_index(
            (self._z0[idx], self._y0[idx], self._x0[idx]), self.neighbour_offset
        )

    # -- normalisation helpers -------------------------------------------

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Per-channel normalise a raw latent (C, d, h, w) or (B, C, d, h, w)."""
        return (z - self.channel_mean) / self.channel_std

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize_latent`."""
        return z * self.channel_std + self.channel_mean

    # -- Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def neighbour_rows(self, idx: int) -> np.ndarray:
        """Store-row index of each of the six face neighbours (−1 when absent)."""
        origin = np.array([self._z0[idx], self._y0[idx], self._x0[idx]], np.int64)
        coords = origin[None, :] + self._nb_dir_offsets
        return self._lookup(int(self._vol_index[idx]), coords)

    def _neighbours(self, idx: int, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """``(nb_latents, nb_avail)`` for one row.

        A neighbour is EXISTS when the store holds a patch at that position
        and OOB when it does not — the specimen ends there, which is exactly
        what the sampler means by OOB at the edge of the requested volume.
        The latent handed over is the CLEAN posterior mean of the whole
        neighbour: at ``neighbour_offset == patch_size`` the two patches only
        touch, so there is nothing to shift and no voxel of the target inside
        it.  Noising it to its own timestep is the training step's job.
        """
        c = self.latent_shape[0]
        L = self.latent_size
        nb = torch.zeros((N_NEIGHBOURS, c, L, L, L), dtype=torch.float32)
        avail = torch.full((N_NEIGHBOURS,), NB_OOB, dtype=torch.int64)
        for i in range(N_NEIGHBOURS):
            if rows[i] < 0:
                continue
            avail[i] = NB_EXISTS
            raw = torch.from_numpy(
                np.asarray(self._latents[int(rows[i]), :c], dtype=np.float32)
            )
            nb[i] = self.normalize_latent(raw) if self.normalize else raw
        return nb, avail

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """The fixed ldm06 batch contract.

        =============  ==========================  =========  ==================
        key            shape                       dtype      meaning
        =============  ==========================  =========  ==================
        z              (C, 16, 16, 16)             float32    posterior mean
        std            (C, 16, 16, 16)             float32    posterior std
        cond_por       ()                          float32    standardised log φ
        cond_depth     ()                          float32    relative depth
        cond_dist6     (6,)                        float32    per-face distance
        cond_orient    (2, 16, 16, 16)             float32    (cos2θ, sin2θ)
        cond_material  (1, 16, 16, 16)             float32    envelope fraction
        nb_latents     (6, C, 16, 16, 16)          float32    CLEAN neighbours
        nb_avail       (6,)                        int64      EXISTS / OOB
        air_fraction   ()                          float32    air voxels / patch
        phi            (1,)                        float32    raw porosity
        label          (64, 64, 64)                uint8      voxel label*
        volume_id      —                           str        provenance
        coords         (3,)                        int32      z0, y0, x0
        grid_index     (3,)                        int64      assembly grid index
        source_row     —                           int        provenance
        =============  ==========================  =========  ==================

        ``cond_dist6`` is ordered by
        :data:`~poregen.diffusion.conditioning.DIST6_DIRS` — (z-, z+, y-, y+,
        x-, x+) — and ``cond_material`` is the specimen ENVELOPE fraction per
        latent cell (``label != 2``, so pores count as specimen).  It is 1
        throughout the interior and drops only at the outer surface and the
        drilled holes; it deliberately says nothing about where the pores are,
        which the model has to generate.
        ``nb_latents`` are the neighbours' clean posterior means; nothing in
        this dataset noises them.

        ``label`` (the starred row) is present ONLY when the dataset was built with
        ``with_label=True`` (the decoded auxiliary loss).  It is the source
        patch's 3-class voxel label — 0 material, 1 pore, 2 air — read from
        ``patches_label.bin`` at ``source_row``, and it is kept uint8 so a
        batch of 256 costs 64 MB rather than 512 MB; the loss casts what it
        needs.
        """
        packed = torch.from_numpy(
            np.asarray(self._latents[idx], dtype=np.float32)  # one contiguous read
        )
        c = self.latent_shape[0]
        z, std = packed[:c], packed[c:]

        if self.normalize:
            z = self.normalize_latent(z)
            std = std / self.channel_std

        vid = str(self._volume_id[idx])
        orient = torch.from_numpy(
            self.orientation.patch_tensor(
                vid, int(self._z0[idx]), self.patch_size, self.latent_size
            )
        )
        nb_latents, nb_avail = self._neighbours(idx, self.neighbour_rows(idx))

        item: dict[str, Any] = {
            "z": z,                                                      # (C, d, h, w)
            "std": std,                                                  # (C, d, h, w)
            "cond_por": torch.tensor(self._cond_por[idx], dtype=torch.float32),
            "cond_depth": torch.tensor(self._cond_depth[idx], dtype=torch.float32),
            "cond_dist6": torch.from_numpy(self._cond_dist6[idx].copy()),  # (6,)
            "cond_orient": orient,                                       # (2, d, h, w)
            "cond_material": torch.from_numpy(
                np.asarray(self._material[idx], dtype=np.float32) / 255.0
            ).unsqueeze(0),                                              # (1, d, h, w)
            "nb_latents": nb_latents,                                    # (6, C, d, h, w)
            "nb_avail": nb_avail,                                        # (6,)
            "air_fraction": torch.tensor(float(self._air[idx]), dtype=torch.float32),
            "phi": torch.tensor([float(self._phi[idx])], dtype=torch.float32),
            "volume_id": vid,
            "coords": torch.tensor(
                [int(self._z0[idx]), int(self._y0[idx]), int(self._x0[idx])],
                dtype=torch.int32,
            ),
            "grid_index": torch.tensor(self.grid_index(idx), dtype=torch.int64),
            "source_row": int(self._source_row[idx]),
        }
        if self._label is not None:
            # np.array copies: a memmap slice is read-only, and torch refuses
            # to share memory it cannot write to.
            item["label"] = torch.from_numpy(
                np.array(self._label[int(self._source_row[idx])])
            )                                                        # (P, P, P) uint8
        return item


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_latent_dataloaders(
    cfg: dict[str, Any],
    latents_root: str | Path,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders over :class:`LatentDataset`.

    The LDM always trains in normalised latent space (``normalize=True``);
    denormalisation happens only immediately before VAE decoding.

    Reads ``data.sample_stride`` (32) / ``data.generation_stride`` (64) /
    ``data.neighbour_offset`` (64) — three separate knobs, never one.

    ``loss.decoded.enabled`` turns on the voxel label in the batch.  The flag
    is read here, in the one place that builds the loaders, so a run cannot
    ask for the decoded auxiliary loss and be served batches without the
    label it scores against.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg    = cfg.get("data", {})
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    decoded_cfg = (cfg.get("loss") or {}).get("decoded") or {}
    with_label  = bool(decoded_cfg.get("enabled", False))

    ds_kwargs: dict[str, Any] = dict(
        normalize=True,
        sample_stride=int(data_cfg.get("sample_stride", 32)),
        generation_stride=int(data_cfg.get("generation_stride", 64)),
        neighbour_offset=int(data_cfg.get("neighbour_offset", 64)),
        orientation_field=data_cfg.get("orientation_field"),
        with_label=with_label,
    )
    train_ds = LatentDataset(latents_root, "train", **ds_kwargs)
    val_ds   = LatentDataset(latents_root, "val",   **ds_kwargs)

    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"]    = int(data_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader

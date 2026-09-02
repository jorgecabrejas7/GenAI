"""Load side of the latent dataset (ldm04 store, ldm05 conditioning).

Reads the layout produced by ``scripts/build_latent_dataset.py``, plus the
conditioning sidecar produced by ``scripts/build_conditioning.py``::

    <root>/                      e.g. data/split_v2/latents_r07z4/
    ├── metadata.json            — latent shape, storage record, per-channel
    │                              train-split normalisation stats, VAE
    │                              provenance, `conditioning` + `assembly` blocks
    └── <split>/
        ├── latents.bin          — float16 C-contiguous memmap,
        │                          shape (N, 2C, d, h, w); channels 0..C-1 = mu,
        │                          C..2C-1 = std ("mu_then_std" packing —
        │                          ONE contiguous read per item)
        ├── index.parquet        — source_row, volume_id, z0, y0, x0, phi, …
        ├── cond.parquet         — cond_depth, cond_dist, cond_por_raw,
        │                          row-aligned with index.parquet
        ├── material.bin         — OPTIONAL (ldm06, scripts/build_material_maps.py):
        │                          uint8 (N, d, h, w); value/255 = material
        │                          fraction of each latent cell
        └── air.bin              — OPTIONAL (ldm06): float32 (N,);
                                   1 - sample_mask.mean() over the patch

Row i of ``index.parquet``, ``cond.parquet``, ``material.bin``, ``air.bin``
and ``latents.bin`` all align.

Latents are stored raw; :class:`LatentDataset` optionally applies per-channel
normalisation ``(mu - mean_c) / std_c`` at load time.  The posterior std is
scaled by ``1 / std_c`` under the same affine map.

Three distinct strides (D32 §3.1 — conflating them was a documented bug):

``sample_stride``
    Spacing of the patches actually stored in this dataset (32).  It is purely
    a training-data multiplier: at ``neighbour_offset = 64`` the store holds
    eight interleaved copies of the stride-64 grid, and each patch's six face
    neighbours are two sampling steps away, so they already exist without a
    rebuild.
``generation_stride``
    Spacing of the assembly grid the sampler will walk (64 = ``patch_size``,
    so patches tile without overlapping and no blending is needed).
``neighbour_offset``
    Spatial displacement, in voxels, of a face neighbour from the target (64).
    Must equal ``generation_stride`` and must be at least ``patch_size``:
    below that a neighbour shares voxels with the target and leaks the answer
    (see ``poregen.diffusion.conditioning``).  It also defines the grid index
    ``(iz, iy, ix) = (z0, y0, x0) // neighbour_offset``, hence the eight-group
    parity schedule.

Neighbours are fed UNSHIFTED.  Touching patches share no cells, so rolling a
neighbour into the target frame would produce an all-zero tensor; the shift is
kept only as a leak-ablation control and raises when it cannot do anything.

Worker safety: the memmap is opened read-only once in ``__init__``; after
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
    NB_EXISTS,
    NB_OOB,
    NEIGHBOUR_DIRS,
    grid_index,
    latent_shift_cells,
    neighbour_states,
    resolve_group_order,
    shift_into_target_frame,
    validate_neighbour_geometry,
    validate_shift,
)
from .orientation import OrientationField

logger = logging.getLogger(__name__)

_KEY_BASE = 8192           # per-axis key radix; every coordinate must be < this
_N_NEIGHBOURS = len(NEIGHBOUR_DIRS)


class LatentDataset(Dataset):
    """Random-access dataset of pre-computed VAE posterior latents.

    Returns the fixed ldm05 batch contract (D32 §5); see :meth:`__getitem__`.

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
    neighbours : bool
        Emit ``nb_latents`` / ``nb_avail``.  Disabling it skips the neighbour
        reads entirely and returns zero tensors with all-OOB availability.
    neighbour_shift : bool
        Roll neighbours into the target frame.  Only meaningful for
        OVERLAPPING neighbours; with touching neighbours it raises rather than
        emitting all-zero tensors.  Default False.
    allow_neighbour_overlap : bool
        ALLOWS CONTENT LEAKAGE — ablation only.  Lets ``neighbour_offset`` drop
        below ``patch_size``, so each neighbour hands the denoiser a verbatim
        copy of part of the target.  Never a valid training setting.
    material : bool
        ldm06 (D40 §1): also serve ``cond_material`` (material fraction per
        latent cell), ``air_fraction`` and ``nb_air_fraction`` from the
        ``material.bin`` / ``air.bin`` sidecars built by
        ``scripts/build_material_maps.py``.  Default off — the ldm05 batch
        contract is untouched.
    air_patch_cap : float | None
        When set (requires ``material=True``), rows with
        ``air_fraction >= air_patch_threshold`` are deterministically
        subsampled so they make up at most this fraction of the served
        dataset.  ``None`` (default) serves every stored row, as today.
        Dropped rows stay in the store and still act as neighbours.
    air_patch_threshold : float
        Air fraction at and above which a patch counts as "all air" for the
        cap.  Default 0.999.
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
        neighbours: bool = True,
        neighbour_shift: bool = False,
        allow_neighbour_overlap: bool = False,
        material: bool = False,
        air_patch_cap: float | None = None,
        air_patch_threshold: float = 0.999,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.sample_stride = int(sample_stride)
        self.generation_stride = int(generation_stride)
        self.neighbour_offset = int(neighbour_offset)
        self.use_neighbours = bool(neighbours)
        self.neighbour_shift = bool(neighbour_shift)
        self.allow_neighbour_overlap = bool(allow_neighbour_overlap)
        self.use_material = bool(material)
        self.air_patch_cap = air_patch_cap
        self.air_patch_threshold = float(air_patch_threshold)

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
        for name, arr, stride in (("z0", self._z0, self.sample_stride),
                                  ("y0", self._y0, self.sample_stride),
                                  ("x0", self._x0, self.sample_stride)):
            bad = int(np.count_nonzero(arr % stride))
            if bad:
                raise ValueError(
                    f"LatentDataset [{split}]: {bad} rows have {name} not a "
                    f"multiple of sample_stride={stride}."
                )
        # ── assembly geometry (see poregen.diffusion.conditioning) ───────────
        validate_neighbour_geometry(
            self.neighbour_offset,
            self.patch_size,
            allow_neighbour_overlap=self.allow_neighbour_overlap,
        )
        if self.allow_neighbour_overlap:
            logger.warning(
                "LatentDataset [%s]: allow_neighbour_overlap=True — neighbours share "
                "voxels with the target and LEAK ITS CONTENT.  Ablation only.",
                split,
            )
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
        self._check_store_assembly_metadata()

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
        cond_meta = self.metadata["conditioning"]
        st = cond_meta["por_standardisation"]
        self.por_mean, self.por_std = float(st["mean"]), float(st["std"])
        self._cond_depth = cond["cond_depth"].to_numpy(np.float32)
        self._cond_dist = cond["cond_dist"].to_numpy(np.float32)
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

        # ── material sidecars (ldm06, optional) ──────────────────────────────
        self._material = None
        self._air = None
        if self.use_material:
            mat_meta = self.metadata.get("material")
            if mat_meta is None:
                raise FileNotFoundError(
                    f"LatentDataset [{split}]: material=True but the store has no "
                    f"`material` metadata block — run scripts/build_material_maps.py."
                )
            L = self.latent_size
            mat_path = split_dir / mat_meta["files"]["material"]
            air_path = split_dir / mat_meta["files"]["air"]
            for path, itemsize, per_row in ((mat_path, 1, L ** 3), (air_path, 4, 1)):
                expected = n * per_row * itemsize
                if not path.exists() or path.stat().st_size != expected:
                    raise RuntimeError(
                        f"LatentDataset [{split}]: {path.name} missing or has "
                        f"{path.stat().st_size if path.exists() else 0} bytes "
                        f"(expected {expected}) — re-run scripts/build_material_maps.py."
                    )
            self._material = np.memmap(str(mat_path), dtype=np.uint8, mode="r",
                                       shape=(n, L, L, L))
            self._air = np.memmap(str(air_path), dtype=np.float32, mode="r", shape=(n,))

        # ── all-air patch cap (ldm06, optional) ──────────────────────────────
        # A remap of the SAMPLING surface only: dropped rows stay in the
        # store, so they still serve as neighbours of kept rows.
        self._sel: np.ndarray | None = None
        if air_patch_cap is not None:
            if not self.use_material:
                raise ValueError(
                    f"LatentDataset [{split}]: air_patch_cap requires material=True "
                    f"(the air fractions come from air.bin)."
                )
            cap = float(air_patch_cap)
            if not (0.0 <= cap < 1.0):
                raise ValueError(
                    f"LatentDataset [{split}]: air_patch_cap must be in [0, 1), got {cap}."
                )
            air_rows = np.flatnonzero(self._air >= self.air_patch_threshold)
            other_rows = np.flatnonzero(self._air < self.air_patch_threshold)
            n_keep = min(len(air_rows), int(cap * len(other_rows) / max(1.0 - cap, 1e-9)))
            rng = np.random.default_rng(0xA16)  # deterministic across runs/workers
            keep_air = (rng.choice(air_rows, size=n_keep, replace=False)
                        if n_keep < len(air_rows) else air_rows)
            self._sel = np.sort(np.concatenate([other_rows, keep_air]))
            logger.info(
                "LatentDataset [%s]: air_patch_cap=%.3f — serving %d/%d all-air rows "
                "(threshold %.3f); %d rows total.",
                split, cap, n_keep, len(air_rows), self.air_patch_threshold,
                len(self._sel),
            )

        # ── neighbour schedule ───────────────────────────────────────────────
        self.group_order = resolve_group_order(self.metadata)
        # Touching neighbours are fed unshifted; shift_cells stays 0.  Asking
        # for a shift at this offset raises instead of yielding zero tensors.
        self.shift_cells = 0
        if self.neighbour_shift:
            self.shift_cells = latent_shift_cells(
                self.neighbour_offset, self.patch_size, self.latent_size
            )
            validate_shift(self.shift_cells, self.latent_size)
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
            "generation_stride=%d neighbour_offset=%d (touching, shift=%s) "
            "neighbours=%s",
            split, n, self.latent_shape, normalize, self.sample_stride,
            self.generation_stride, self.neighbour_offset,
            f"{self.shift_cells} cells" if self.neighbour_shift else "off",
            self.use_neighbours,
        )

    # -- helpers -----------------------------------------------------------

    def _check_store_assembly_metadata(self) -> None:
        """Fail when the store was built for a different assembly geometry.

        ``scripts/build_conditioning.py`` records the strides it assumed in
        ``metadata["assembly"]``.  Training against a store built for another
        neighbour relation would silently mis-condition every patch.
        """
        asm = self.metadata.get("assembly") or {}
        for key, ours in (
            ("sample_stride", self.sample_stride),
            ("generation_stride", self.generation_stride),
            ("neighbour_offset", self.neighbour_offset),
        ):
            theirs = asm.get(key)
            if theirs is not None and int(theirs) != int(ours):
                raise ValueError(
                    f"LatentDataset [{self.split}]: the store records "
                    f"assembly.{key}={theirs} but this run asks for {ours}.  Re-run "
                    f"scripts/build_conditioning.py so the store metadata describes "
                    f"the geometry actually being trained."
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
        return len(self._sel) if self._sel is not None else len(self.df)

    def _neighbour_rows(self, idx: int) -> np.ndarray:
        """Store-row index of each of the six face neighbours (−1 when absent)."""
        origin = np.array([self._z0[idx], self._y0[idx], self._x0[idx]], np.int64)
        coords = origin[None, :] + self._nb_dir_offsets
        return self._lookup(int(self._vol_index[idx]), coords)

    def _neighbours(self, idx: int, rows: np.ndarray | None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(nb_latents, nb_avail)`` for one row.

        Availability comes from the SHARED
        :func:`~poregen.diffusion.conditioning.neighbour_states`, the same
        function :class:`~poregen.diffusion.sampler.VolumeGenerator` calls, so
        training reproduces the generation schedule by construction.  A grid
        position counts as in-grid here when the store actually holds a patch
        there; the sampler's predicate is grid membership.  Only EXISTS
        neighbours carry content — an UNKNOWN neighbour has not been generated
        yet at inference, so training must not show it either.

        The latent handed over is the FULL neighbour: at ``neighbour_offset ==
        patch_size`` the two patches only touch, so there is nothing to shift
        and no voxel of the target inside it.
        """
        c = self.latent_shape[0]
        L = self.latent_size
        nb = torch.zeros((_N_NEIGHBOURS, c, L, L, L), dtype=torch.float32)
        avail = torch.full((_N_NEIGHBOURS,), NB_OOB, dtype=torch.int64)
        if not self.use_neighbours or rows is None:
            return nb, avail

        gi = self.grid_index(idx)

        # A face neighbour is exactly ±1 grid step (grid_index divides by
        # neighbour_offset), so direction i maps to gi + NEIGHBOUR_DIRS[i].
        stored = {
            (gi[0] + d[0], gi[1] + d[1], gi[2] + d[2])
            for i, d in enumerate(NEIGHBOUR_DIRS) if rows[i] >= 0
        }
        states = neighbour_states(gi, lambda g: g in stored, self.group_order)

        for i, d in enumerate(NEIGHBOUR_DIRS):
            avail[i] = states[i]
            if states[i] != NB_EXISTS:
                continue
            raw = torch.from_numpy(
                np.asarray(self._latents[int(rows[i]), :c], dtype=np.float32)
            )
            if self.normalize:
                raw = self.normalize_latent(raw)
            nb[i] = (
                shift_into_target_frame(raw, d, self.shift_cells)
                if self.neighbour_shift else raw
            )
        return nb, avail

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """The fixed ldm05 batch contract (D32 §5).

        =============  ==========================  =========  ==================
        key            shape                       dtype      meaning
        =============  ==========================  =========  ==================
        z              (C, 16, 16, 16)             float32    posterior mean
        std            (C, 16, 16, 16)             float32    posterior std
        cond_por       ()                          float32    standardised log φ
        cond_depth     ()                          float32    relative depth
        cond_dist      ()                          float32    distance to surface
        cond_orient    (2, 16, 16, 16)             float32    (cos2θ, sin2θ)
        nb_latents     (6, C, 16, 16, 16)          float32    touching neighbours
        nb_avail       (6,)                        int64      OOB/EXISTS/UNKNOWN
        phi            (1,)                        float32    raw porosity
        volume_id      —                           str        provenance
        coords         (3,)                        int32      z0, y0, x0
        grid_index     (3,)                        int64      assembly grid index
        source_row     —                           int        provenance
        =============  ==========================  =========  ==================

        With ``material=True`` (ldm06, D40 §1) three keys are added:

        ===============  =================  =========  ========================
        cond_material    (1, 16, 16, 16)    float32    material fraction / cell
        air_fraction     ()                 float32    1 − material mean, patch
        nb_air_fraction  (6,)               float32    per-neighbour air frac
        ===============  =================  =========  ========================

        ``nb_air_fraction[i]`` is the stored air fraction of neighbour *i*
        whenever the store holds a patch there (EXISTS or UNKNOWN — the
        material map is paintable before generation, so this is conditioning,
        not a content leak), and 1.0 where no patch is stored (OOB — treated
        as all air, matching the sampler's semantics).
        """
        if self._sel is not None:
            idx = int(self._sel[idx])
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
        nb_rows = (self._neighbour_rows(idx)
                   if (self.use_neighbours or self.use_material) else None)
        nb_latents, nb_avail = self._neighbours(idx, nb_rows)

        item = {
            "z": z,                                                      # (C, d, h, w)
            "std": std,                                                  # (C, d, h, w)
            "cond_por": torch.tensor(self._cond_por[idx], dtype=torch.float32),
            "cond_depth": torch.tensor(self._cond_depth[idx], dtype=torch.float32),
            "cond_dist": torch.tensor(self._cond_dist[idx], dtype=torch.float32),
            "cond_orient": orient,                                       # (2, d, h, w)
            "nb_latents": nb_latents,                                    # (6, C, d, h, w)
            "nb_avail": nb_avail,                                        # (6,)
            "phi": torch.tensor([float(self._phi[idx])], dtype=torch.float32),
            "volume_id": vid,
            "coords": torch.tensor(
                [int(self._z0[idx]), int(self._y0[idx]), int(self._x0[idx])],
                dtype=torch.int32,
            ),
            "grid_index": torch.tensor(self.grid_index(idx), dtype=torch.int64),
            "source_row": int(self._source_row[idx]),
        }
        if self.use_material:
            item["cond_material"] = torch.from_numpy(
                np.asarray(self._material[idx], dtype=np.float32) / 255.0
            ).unsqueeze(0)
            item["air_fraction"] = torch.tensor(
                float(self._air[idx]), dtype=torch.float32
            )
            nb_air = np.ones(_N_NEIGHBOURS, dtype=np.float32)  # absent → all air
            present = nb_rows >= 0
            nb_air[present] = self._air[nb_rows[present]]
            item["nb_air_fraction"] = torch.from_numpy(nb_air)
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
    ``data.neighbour_offset`` (64) — three separate knobs, never one
    (D32 §3.1) — plus ``data.neighbour_shift`` (false) and the ablation-only
    ``data.allow_neighbour_overlap`` (false).  ldm06 additions, all default
    off: ``data.material`` (serve cond_material / air fractions),
    ``data.air_patch_cap`` (train-split all-air subsampling) and
    ``data.air_patch_threshold``.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg    = cfg.get("data", {})
    batch_size  = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    ds_kwargs: dict[str, Any] = dict(
        normalize=True,
        sample_stride=int(data_cfg.get("sample_stride", 32)),
        generation_stride=int(data_cfg.get("generation_stride", 64)),
        neighbour_offset=int(data_cfg.get("neighbour_offset", 64)),
        neighbours=bool(data_cfg.get("use_neighbours", True)),
        neighbour_shift=bool(data_cfg.get("neighbour_shift", False)),
        allow_neighbour_overlap=bool(data_cfg.get("allow_neighbour_overlap", False)),
        orientation_field=data_cfg.get("orientation_field"),
        material=bool(data_cfg.get("material", False)),
        air_patch_threshold=float(data_cfg.get("air_patch_threshold", 0.999)),
    )
    # The all-air cap shapes the TRAINING distribution only; val stays intact.
    cap = data_cfg.get("air_patch_cap")
    train_ds = LatentDataset(
        latents_root, "train",
        air_patch_cap=(float(cap) if cap is not None else None), **ds_kwargs,
    )
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

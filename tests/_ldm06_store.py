"""Builder for a miniature ldm06 latent store, shared by several test modules.

Scaled-down copy of the production geometry: C=1, latent 4³, patch 16 voxels
(downsample 4), ``sample_stride`` 8 (half the patch, so the store holds eight
interleaved copies of the tiling grid) and ``neighbour_offset`` 16 == the patch
size, so neighbours TOUCH.  Every patch is a crop of ONE global field, so a
neighbour must reproduce the field block immediately next to the target — and
must share none of the target's own cells.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SYN = {
    "C": 1,
    "L": 4,             # latent cells per patch
    "PATCH": 16,        # voxels per patch
    "SAMPLE_STRIDE": 8,
    "OFFSET": 16,       # == PATCH: touching neighbours
    "VOL": 80,          # voxels per axis of the synthetic volume
}

DIST6_COLUMNS = ("cond_dist6_zm", "cond_dist6_zp", "cond_dist6_ym",
                 "cond_dist6_yp", "cond_dist6_xm", "cond_dist6_xp")


def row_std(row: int) -> float:
    """Posterior std stored for store row *row* (see ``build_store``)."""
    return 0.05 * (1 + row % 5)


def build_store(root: Path) -> tuple[Path, np.ndarray, int, int]:
    """Write a complete store under *root*.

    Returns ``(root, field, ds_factor, sample_stride)`` where ``field`` is the
    ``(C, gz, gy, gx)`` latent field every patch was cropped from.
    """
    root = Path(root)
    L, C = SYN["L"], SYN["C"]
    patch, stride, vol = SYN["PATCH"], SYN["SAMPLE_STRIDE"], SYN["VOL"]
    ds_factor = patch // L                                   # 4
    vol_shape = (vol, vol, vol)
    gsize = tuple(s // ds_factor for s in vol_shape)          # 20³ latent cells
    rng = np.random.default_rng(7)
    field = rng.standard_normal((C, *gsize)).astype(np.float32)
    # Specimen-envelope field on the same latent grid, so a patch's material
    # map is also a crop of one coherent volume: 1 in the interior, a taper at
    # the low-z surface, 0 outside it — the shape a real envelope has.
    mat_field = np.ones(gsize, dtype=np.float32)
    mat_field[0] = 0.0
    mat_field[1] = 0.25
    mat_field[2] = 0.75

    coords = list(itertools.product(range(0, vol - patch + 1, stride), repeat=3))
    rows = []
    data = np.zeros((len(coords), 2 * C, L, L, L), np.float16)
    material = np.zeros((len(coords), L, L, L), np.uint8)
    air = np.zeros(len(coords), np.float32)
    for i, (z0, y0, x0) in enumerate(coords):
        cz, cy, cx = z0 // ds_factor, y0 // ds_factor, x0 // ds_factor
        data[i, :C] = field[:, cz:cz + L, cy:cy + L, cx:cx + L]
        # Posterior std varies with the ROW, so a test can tell whether a
        # neighbour's std was read from the neighbour's row or the target's.
        # The five values are exact in float16, and the six face neighbours of
        # any patch all land on a different residue than the patch itself.
        data[i, C:] = row_std(i)
        cells = mat_field[cz:cz + L, cy:cy + L, cx:cx + L]
        material[i] = np.rint(cells * 255.0).astype(np.uint8)
        # The store's own invariant: air is 1 - the envelope mean, exactly.
        air[i] = np.float32(1.0 - cells.mean(dtype=np.float64))
        rows.append({"source_row": i, "volume_id": "vol_a", "z0": z0, "y0": y0,
                     "x0": x0, "phi": 0.01 + 0.0001 * i})
    df = pd.DataFrame(rows)

    split_dir = root / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(split_dir / "latents.bin")
    material.tofile(split_dir / "material.bin")
    air.tofile(split_dir / "air.bin")
    df.to_parquet(split_dir / "index.parquet", index=False)

    por_raw = np.log(df["phi"].to_numpy() + 1e-3).astype(np.float32)
    cond = {
        "source_row": df["source_row"].to_numpy(np.int64),
        "cond_depth": np.linspace(0, 1, len(df), dtype=np.float32),
        "cond_por_raw": por_raw,
    }
    for k, name in enumerate(DIST6_COLUMNS):
        cond[name] = np.linspace(0, 1, len(df), dtype=np.float32) * (0.1 * (k + 1))
    pd.DataFrame(cond).to_parquet(split_dir / "cond.parquet", index=False)

    of = root / "orientation_field.json"
    theta = [0.0] * (vol // 2) + [90.0] * (vol - vol // 2)
    of.write_text(json.dumps({
        "voxel_size_um": 25.0,
        "volumes": {"vol_a": {"shape": list(vol_shape), "orientation_usable": True,
                              "confidence": "high", "theta_deg": theta}},
    }))

    (root / "metadata.json").write_text(json.dumps({
        "latent_shape": [C, L, L, L],
        "patch_size": patch,
        "voxel_size_um": 25.0,
        "storage": {"format": "memmap", "file": "latents.bin",
                    "dtype": "float16", "pack_scheme": "mu_then_std"},
        "normalization": {"computed_over": "train",
                          "per_channel_mean": [0.0] * C,
                          "per_channel_std": [1.0] * C},
        "material": {"files": {"material": "material.bin", "air": "air.bin"}},
        "conditioning": {
            "orientation_field": str(of),
            "por_standardisation": {"mean": float(por_raw.mean()),
                                    "std": float(por_raw.std())},
            "geometry": {
                "sample_stride": stride,
                "generation_stride": SYN["OFFSET"],
                "neighbour_offset": SYN["OFFSET"],
                "patch_size": patch,
            },
        },
    }))
    return root, field, ds_factor, stride


def dataset_kwargs(**overrides):
    """The store's own geometry as ``LatentDataset`` keyword arguments."""
    kw = dict(sample_stride=SYN["SAMPLE_STRIDE"],
              generation_stride=SYN["OFFSET"],
              neighbour_offset=SYN["OFFSET"])
    kw.update(overrides)
    return kw

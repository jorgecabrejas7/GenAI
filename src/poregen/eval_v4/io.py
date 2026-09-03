"""Campaign layout, and the reader that refuses to guess.

One campaign directory holds one run of the suite::

    runs/campaigns/<NN>-<name>/
        <assessment>/
            volumes/<case>/
                volume.tif            uint8, raw-scan grey
                label.tif             uint8, 0 material / 1 pore / 2 air
                probs.npz             optional: pore log-odds (+ class probs)
                requested_field.npy   optional: requested phi per 64-voxel tile
                requested_material.npy optional: requested envelope per latent cell
                manifest.json
            results.json
            findings.md
            figures/*.pdf, *.png
        real_floor/                   same shape; sampler = "real"
        README.md

``volume.tif`` and ``label.tif`` go in and come out on their NATIVE scale.
:func:`load_u8` refuses anything that is not already ``uint8``: a generated
volume and a real one must be the same kind of number, and eval v2 lost a whole
campaign to a silent rescale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import tifffile

from poregen.eval_v4.manifest import Manifest, ManifestError

VOLUME_TIF = "volume.tif"
LABEL_TIF = "label.tif"
PROBS_NPZ = "probs.npz"
FIELD_NPY = "requested_field.npy"
MATERIAL_NPY = "requested_material.npy"

LABEL_MATERIAL, LABEL_PORE, LABEL_AIR = 0, 1, 2

#: The geometry every assessment is defined on.  A tile is what the model was
#: trained on and what the window seam metric measures; the latent cell is what
#: the material request is expressed on.
TILE = 64
LATENT_DOWNSAMPLE = 4

#: Above this voxel count the full 3-class probability volume is not stored -
#: only the pore log-odds, which is all any metric reads.
FULL_PROBS_MAX_VOXELS = 16_000_000


def repo_root(start: str | Path | None = None) -> Path:
    """Nearest ancestor holding ``pyproject.toml``."""
    here = Path(start or __file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(f"no pyproject.toml above {here}")


# ---------------------------------------------------------------------------
# Volume IO
# ---------------------------------------------------------------------------

def load_u8(path: str | Path) -> np.ndarray:
    """Read a uint8 TIFF, and refuse anything else.

    There is no rescaling branch on purpose.  A float volume reaching this
    function means it was written on the wrong scale, and guessing the
    transform is exactly the failure that invalidated eval v2.
    """
    path = Path(path)
    arr = tifffile.imread(str(path))
    if arr.dtype != np.uint8:
        raise TypeError(
            f"{path} is {arr.dtype}, not uint8.  Generated and real volumes are "
            "written on the raw-scan uint8 scale; this suite will not guess a "
            "conversion."
        )
    if arr.ndim != 3:
        raise ValueError(f"{path} has shape {arr.shape}, expected a 3-D volume.")
    return arr


def load_label(path: str | Path) -> np.ndarray:
    """Read the 3-class label TIFF and check it only holds 0, 1 and 2."""
    arr = load_u8(path)
    bad = np.setdiff1d(np.unique(arr), np.array([0, 1, 2], np.uint8))
    if bad.size:
        raise ValueError(
            f"{path} carries label values {bad.tolist()}; the contract is "
            "0 material, 1 pore, 2 air."
        )
    return arr


def save_case(
    case_dir: str | Path,
    manifest: Manifest,
    xct_u8: np.ndarray,
    label_u8: np.ndarray,
    *,
    pore_logit: np.ndarray | None = None,
    class_probs: np.ndarray | None = None,
    requested_field: np.ndarray | None = None,
    requested_material: np.ndarray | None = None,
) -> Path:
    """Write one case directory.  The manifest is written last, on purpose.

    A half-written case therefore has no manifest, and every reader refuses it
    rather than measuring a truncated volume.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest.check_array(xct_u8, "save_case", "volume")
    manifest.check_array(label_u8, "save_case", "label")

    tifffile.imwrite(str(case_dir / VOLUME_TIF), np.ascontiguousarray(xct_u8, np.uint8))
    tifffile.imwrite(str(case_dir / LABEL_TIF), np.ascontiguousarray(label_u8, np.uint8))

    if pore_logit is not None:
        payload = {"pore_logit": np.asarray(pore_logit, np.float16)}
        if class_probs is not None and xct_u8.size <= FULL_PROBS_MAX_VOXELS:
            payload["class_probs"] = np.asarray(class_probs, np.float16)
        np.savez_compressed(case_dir / PROBS_NPZ, **payload)

    if requested_field is not None:
        np.save(case_dir / FIELD_NPY, np.asarray(requested_field, np.float32))
    if requested_material is not None:
        np.save(case_dir / MATERIAL_NPY, np.asarray(requested_material, np.float32))

    manifest.write(case_dir)
    return case_dir


@dataclass
class Case:
    """One measured volume: its directory, its manifest and its arrays.

    Arrays are read on first use and then held, so a metric chain over one case
    touches the disk once per array.
    """

    path: Path
    manifest: Manifest

    @classmethod
    def load(cls, case_dir: str | Path) -> "Case":
        case_dir = Path(case_dir)
        return cls(path=case_dir, manifest=Manifest.read(case_dir))

    @cached_property
    def xct(self) -> np.ndarray:
        return load_u8(self.path / VOLUME_TIF)

    @cached_property
    def label(self) -> np.ndarray:
        return load_label(self.path / LABEL_TIF)

    @cached_property
    def pore_logit(self) -> np.ndarray | None:
        """Blended pore-vs-rest log odds, or ``None`` when it was not stored."""
        f = self.path / PROBS_NPZ
        if not f.exists():
            return None
        with np.load(f) as z:
            if "pore_logit" not in z:
                return None
            return z["pore_logit"].astype(np.float32)

    @cached_property
    def requested_field(self) -> np.ndarray | None:
        """Requested phi per 64-voxel tile, ``(gz, gy, gx)``."""
        name = self.manifest.requested_field
        if name is None:
            return None
        return np.load(self.path / name).astype(np.float32)

    @cached_property
    def requested_material(self) -> np.ndarray | None:
        """Requested specimen-envelope fraction per LATENT cell, or ``None``
        when the whole volume was requested as specimen (``"full"``)."""
        name = self.manifest.requested_material
        if name in (None, "full"):
            return None
        return np.load(self.path / name).astype(np.float32)

    # -- derived requests --------------------------------------------------

    def material_voxels(self) -> np.ndarray:
        """Boolean requested-specimen mask at VOXEL resolution.

        The request the model actually saw is the per-latent-cell envelope
        fraction, so the voxel-level request is that map upsampled by the VAE
        downsampling factor and thresholded at half a cell.  Deriving it from
        the stored map rather than re-painting the shape keeps the metric
        comparing against what was asked, not against what was meant.
        """
        mat = self.requested_material
        shape = tuple(self.manifest.volume_shape)
        if mat is None:
            return np.ones(shape, dtype=bool)
        cells = tuple(s // LATENT_DOWNSAMPLE for s in shape)
        if mat.shape != cells:
            raise ManifestError(
                f"{self.path.name}: requested_material has shape {mat.shape}, "
                f"expected the latent canvas {cells} of volume_shape {shape}."
            )
        return np.repeat(
            np.repeat(np.repeat(mat >= 0.5, LATENT_DOWNSAMPLE, 0), LATENT_DOWNSAMPLE, 1),
            LATENT_DOWNSAMPLE, 2,
        )

    def requested_phi_per_tile(self) -> np.ndarray:
        """Requested phi on the 64-voxel tile grid, uniform when only a global
        target was asked for."""
        field = self.requested_field
        grid = tuple(s // TILE for s in self.manifest.volume_shape)
        if field is not None:
            if field.shape != grid:
                raise ManifestError(
                    f"{self.path.name}: requested_field has shape {field.shape}, "
                    f"expected the tile grid {grid}."
                )
            return field
        self.manifest.require(("requested_global_phi",), "requested_phi_per_tile")
        return np.full(grid, float(self.manifest.requested_global_phi), np.float32)


# ---------------------------------------------------------------------------
# Campaign layout
# ---------------------------------------------------------------------------

def assessment_dir(root: str | Path, assessment: str) -> Path:
    return Path(root) / assessment


def volumes_dir(root: str | Path, assessment: str) -> Path:
    return assessment_dir(root, assessment) / "volumes"


def case_dir(root: str | Path, assessment: str, case: str) -> Path:
    return volumes_dir(root, assessment) / case


def iter_case_dirs(root: str | Path, assessment: str):
    """Every directory under ``<root>/<assessment>/volumes`` that has a manifest."""
    base = volumes_dir(root, assessment)
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            yield d


def load_cases(root: str | Path, assessment: str) -> list[Case]:
    cases = [Case.load(d) for d in iter_case_dirs(root, assessment)]
    for c in cases:
        if c.manifest.assessment != assessment:
            raise ManifestError(
                f"{c.path} sits under assessment {assessment!r} but its manifest "
                f"says {c.manifest.assessment!r}."
            )
    return cases


def write_results(root: str | Path, assessment: str, results: dict) -> Path:
    out = assessment_dir(root, assessment)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(jsonable(results), indent=2) + "\n")
    return path


def read_results(root: str | Path, assessment: str) -> dict:
    path = assessment_dir(root, assessment) / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - run `eval_v4 measure {assessment} "
            f"--root {root}` first."
        )
    return json.loads(path.read_text())


def write_findings(root: str | Path, assessment: str, text: str) -> Path:
    out = assessment_dir(root, assessment)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "findings.md"
    path.write_text(text.rstrip() + "\n")
    return path


def figures_dir(root: str | Path, assessment: str) -> Path:
    d = assessment_dir(root, assessment) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def jsonable(obj):
    """Make numpy scalars, arrays, paths and NaNs survive ``json.dumps``."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

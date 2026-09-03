"""The manifest: what a generated volume claims about itself, and its schema.

Every volume the suite writes carries a ``manifest.json`` beside it.  The
manifest is the only channel through which a metric learns what was ASKED for,
so a metric that needs a request declares the fields it reads and refuses to
run when they are absent or contradict the array it was handed.  That refusal
is the point of the file: eval v3 recovered a volume's requested porosity from
its directory name, so renaming a directory silently changed the answer.

Two kinds of manifest exist and both go through the same schema:

``sampler = "hybrid_chunked"``
    a volume from :class:`poregen.diffusion.sampler.VolumeGenerator`.  Every
    generation field must be filled in.
``sampler = "real"``
    a crop of a real scan, written by ``eval_v4 real-floor``.  It has no model,
    no seed and no request, so those fields are ``None`` and any metric that
    needs a request refuses it — which is why the real floor is reported for
    the request-free metrics only.
"""

from __future__ import annotations

import functools
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_VERSION = "v4"
MANIFEST_NAME = "manifest.json"

SAMPLERS = ("hybrid_chunked", "real")
WEIGHTS = ("raw", "ema")
DECODES = ("tiled", "overlapped")

#: Fields a generated (non-real) volume must carry with a non-``None`` value.
GENERATED_REQUIRED = (
    "model_run", "checkpoint_step", "weights", "ddim_steps", "chunk_tiles",
    "window_stride", "decode", "decode_overlap", "s_por", "s_nb", "objective",
    "cfg_rescale", "seed", "requested_material",
)

#: What the denoiser is trained to predict.  The sampler converts through the
#: schedule either way, so the objective never changes how a metric is computed
#: - but it does change which numbers may be compared with which.
OBJECTIVES = ("eps", "v")


class ManifestError(ValueError):
    """The manifest is malformed, or does not match the volume beside it."""


@dataclass(frozen=True)
class Manifest:
    """What one volume is, what was requested of it, and what it cost.

    Paths in ``requested_field`` and ``requested_material`` are relative to the
    case directory, so a campaign tree can be moved without invalidating them.
    ``requested_material`` is the string ``"full"`` when the whole volume was
    requested as specimen.
    """

    # -- identity ----------------------------------------------------------
    assessment: str
    case: str
    volume_shape: tuple[int, int, int]
    git_commit: str
    sampler: str = "hybrid_chunked"
    manifest_version: str = MANIFEST_VERSION

    # -- model and sampler settings ----------------------------------------
    model_run: str | None = None
    checkpoint_step: int | None = None
    weights: str | None = None
    ddim_steps: int | None = None
    chunk_tiles: tuple[int, int, int] | None = None
    window_stride: int | None = None
    decode: str | None = None
    decode_overlap: int | None = None
    s_por: float | None = None
    s_nb: float | None = None
    #: What the denoiser predicts - ``eps`` or ``v``.  Two checkpoints trained
    #: on different objectives are different models, and a seam or a porosity
    #: number from one says nothing about the other, so the objective belongs
    #: in the manifest beside the step count.
    objective: str | None = None
    #: Guidance rescaling factor; 0 is plain classifier-free guidance.
    cfg_rescale: float | None = None
    seed: int | None = None

    # -- the request -------------------------------------------------------
    requested_global_phi: float | None = None
    requested_field: str | None = None
    requested_layup: tuple[int, ...] | None = None
    requested_ply_thickness_vox: float | None = None
    requested_material: str | None = None

    # The sub-block of the generated canvas the case is ABOUT.  Assessment 6
    # asks for the same region assembled on two different window grids, which
    # the sampler can only deliver by translating the request inside a larger
    # canvas; the region says which part to measure.  ``None`` means the whole
    # volume, which is the normal case.
    region_offset: tuple[int, int, int] | None = None
    region_shape: tuple[int, int, int] | None = None

    # -- cost --------------------------------------------------------------
    wall_time_s: float | None = None
    peak_gpu_memory_bytes: int | None = None

    # -- free-form provenance (latent store, layup name, ...) --------------
    notes: dict[str, Any] | None = None

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "volume_shape", _triple(self.volume_shape, "volume_shape"))
        if self.chunk_tiles is not None:
            set_(self, "chunk_tiles", _triple(self.chunk_tiles, "chunk_tiles"))
        if self.region_offset is not None:
            set_(self, "region_offset", _triple(self.region_offset, "region_offset"))
        if self.region_shape is not None:
            set_(self, "region_shape", _triple(self.region_shape, "region_shape"))
        if self.requested_layup is not None:
            set_(self, "requested_layup", tuple(int(a) for a in self.requested_layup))
        self.validate()

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ManifestError(f"manifest carries unknown fields {sorted(unknown)}")
        missing = {"assessment", "case", "volume_shape", "git_commit"} - set(d)
        if missing:
            raise ManifestError(f"manifest is missing required fields {sorted(missing)}")
        return cls(**d)

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        if not path.exists():
            raise ManifestError(f"no manifest at {path} - the volume cannot be measured")
        return cls.from_dict(json.loads(path.read_text()))

    def write(self, case_dir: str | Path) -> Path:
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        out = case_dir / MANIFEST_NAME
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ManifestError` unless the manifest is self-consistent."""
        if self.manifest_version != MANIFEST_VERSION:
            raise ManifestError(
                f"manifest version {self.manifest_version!r} - this suite writes "
                f"and reads {MANIFEST_VERSION!r}."
            )
        if self.sampler not in SAMPLERS:
            raise ManifestError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        if not self.git_commit:
            raise ManifestError("git_commit is empty - the volume is not traceable to code")
        if any(s <= 0 for s in self.volume_shape):
            raise ManifestError(f"volume_shape {self.volume_shape} has a non-positive axis")
        if self.weights is not None and self.weights not in WEIGHTS:
            raise ManifestError(f"weights must be one of {WEIGHTS}, got {self.weights!r}")
        if self.decode is not None and self.decode not in DECODES:
            raise ManifestError(f"decode must be one of {DECODES}, got {self.decode!r}")
        if self.objective is not None and self.objective not in OBJECTIVES:
            raise ManifestError(
                f"objective must be one of {OBJECTIVES}, got {self.objective!r}"
            )
        if self.cfg_rescale is not None and self.cfg_rescale < 0:
            raise ManifestError(f"cfg_rescale must not be negative, got {self.cfg_rescale}")
        if self.decode is not None:
            overlap = self.decode_overlap
            if overlap is None:
                raise ManifestError("decode is set but decode_overlap is not")
            if (self.decode == "tiled") != (overlap == 0):
                raise ManifestError(
                    f"decode={self.decode!r} contradicts decode_overlap={overlap}: "
                    "a tiled decode has zero overlap and an overlapped one does not."
                )
        if self.requested_material is not None and self.requested_material != "full":
            if not str(self.requested_material).endswith(".npy"):
                raise ManifestError(
                    "requested_material must be 'full' or the name of an .npy file, "
                    f"got {self.requested_material!r}"
                )
        if self.requested_field is not None and not str(self.requested_field).endswith(".npy"):
            raise ManifestError(
                f"requested_field must name an .npy file, got {self.requested_field!r}"
            )
        if (self.region_offset is None) != (self.region_shape is None):
            raise ManifestError("region_offset and region_shape must be given together")
        if self.region_offset is not None:
            for a in range(3):
                end = self.region_offset[a] + self.region_shape[a]
                if self.region_offset[a] < 0 or end > self.volume_shape[a]:
                    raise ManifestError(
                        f"region {self.region_offset}+{self.region_shape} leaves the "
                        f"volume {self.volume_shape} on axis {a}"
                    )
        if self.sampler == "hybrid_chunked":
            absent = [f for f in GENERATED_REQUIRED if getattr(self, f) is None]
            if absent:
                raise ManifestError(
                    f"a hybrid_chunked volume must declare {absent} - without them the "
                    "measurement cannot say what it is measuring."
                )
            if self.requested_global_phi is None and self.requested_field is None:
                raise ManifestError(
                    "a hybrid_chunked volume must declare a porosity request: "
                    "requested_global_phi, requested_field, or both."
                )

    # -- what a metric asks of it ------------------------------------------

    @property
    def is_real(self) -> bool:
        return self.sampler == "real"

    @property
    def region(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """``(offset, shape)`` of the sub-block the case is about."""
        if self.region_offset is None:
            return (0, 0, 0), tuple(self.volume_shape)
        return tuple(self.region_offset), tuple(self.region_shape)

    def require(self, needed: tuple[str, ...], metric: str) -> None:
        """Refuse the volume unless every field ``metric`` reads is present."""
        absent = [f for f in needed if getattr(self, f, None) is None]
        if absent:
            raise ManifestError(
                f"{metric} needs manifest fields {absent}, which "
                f"{self.assessment}/{self.case} (sampler={self.sampler}) does not carry."
            )

    def check_array(self, arr: np.ndarray, metric: str, name: str = "volume") -> None:
        """Refuse an array whose spatial shape is not the one declared."""
        shape = tuple(int(s) for s in np.shape(arr)[-3:])
        if shape != tuple(self.volume_shape):
            raise ManifestError(
                f"{metric}: {name} has shape {shape} but "
                f"{self.assessment}/{self.case} declares volume_shape "
                f"{tuple(self.volume_shape)}."
            )


def requires(*needed: str):
    """Declare the manifest fields a metric reads.

    The wrapper checks them before the metric body runs, and cross-checks the
    spatial shape of every POSITIONAL array argument against
    ``manifest.volume_shape``.  A metric therefore cannot quietly measure a
    volume that is not the one the manifest describes.

    The positional-only rule is the calling convention every metric follows:
    volume-shaped arrays go in positionally, and an array on a different grid -
    a tile-grid field, a second volume - goes in by keyword and is checked by
    the metric itself against the grid it belongs to.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, manifest: "Manifest", **kwargs):
            if not isinstance(manifest, Manifest):
                raise ManifestError(
                    f"{fn.__name__} needs a Manifest, got {type(manifest).__name__}."
                )
            manifest.require(needed, fn.__name__)
            for arg in args:
                if isinstance(arg, np.ndarray) and arg.ndim in (3, 4):
                    manifest.check_array(arg, fn.__name__)
            return fn(*args, manifest=manifest, **kwargs)

        wrapper.requires = tuple(needed)
        return wrapper

    return decorate


def head_commit(repo: str | Path) -> str:
    """HEAD of ``repo``, suffixed ``-dirty`` when the working tree is modified."""
    repo = str(repo)
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - env
        raise ManifestError(f"cannot read the source commit of {repo}: {exc}") from exc
    return f"{sha}-dirty" if dirty else sha


def _triple(v, name: str) -> tuple[int, int, int]:
    t = tuple(int(x) for x in v)
    if len(t) != 3:
        raise ManifestError(f"{name} must have three entries, got {v!r}")
    return t  # type: ignore[return-value]

#!/usr/bin/env python
"""Is the synthetic data USEFUL, or only realistic?

Every other measurement in this project asks whether a generated volume
RESEMBLES a real one.  This one asks the question a reader actually cares
about: does the generated data help a downstream task?  The task is the one
the dataset already defines — 3-class voxel segmentation (material / pore /
air) of an XCT patch — and the answer is a segmentation score on the REAL
held-out test panels.

Three arms, one budget
----------------------
Every arm trains the same 3-D U-Net, for the same number of steps, at the same
batch size, with the same optimiser, the same augmentation and the same TOTAL
patch count.  The ONLY thing that differs is where the patches come from::

    (a) real                 patch_count real train-panel patches
    (b) synthetic            patch_count ldm06 patches
    (c) real_plus_synthetic  half of each

Holding the patch count fixed in every arm is what makes the three numbers
comparable: arm (c) is a REPLACEMENT of half the real data, not an addition on
top of it.  An arm that saw twice the data would be measuring dataset size and
calling it data quality.  The three arms therefore read as one mixing curve at
a constant data budget — 100 % real, 50/50, 100 % synthetic — with the
real-only arm as the reference row.

Matching the request distribution
---------------------------------
A synthetic arm drawn from whatever volumes happen to exist would differ from
the real arm in POROSITY as well as in provenance, and the porosity gap would
be reported as a quality gap.  So the synthetic patches are drawn to reproduce
the real arm's own joint histogram over (porosity bin x has-air), stratum by
stratum: the conditions the model was asked for are, patch for patch, the
conditions the real material presents.  The porosity bins are the ones the
per-bin error is reported in, so the match is exact in the units of the answer.

Where the synthetic data comes from
-----------------------------------
The eval v4 campaign, assessments ``sampler``, ``porosity_global``,
``microstructure`` and ``surface`` — every case generated at the operating
point with a request a real panel could plausibly present.  Excluded on
purpose: the off-manifold phi 0.15 dose point, the painted checkerboard and
two-halves fields, the non-operating CFG scales, the layups the panels do not
have, the notch/hole/sphere geometries, and the sampler-geometry probes.  None
of those is a distribution the real material draws from.

``surface`` is not optional decoration: it is the only source of AIR in the
synthetic set, because every other case requests a full-material box.  Without
it the synthetic arm would never see a specimen boundary and its air Dice would
be a measurement of the case list rather than of the model.

The script REFUSES to run the synthetic or mixed arm until every required case
exists.  Training on a partial campaign would silently answer a different
question, and the answer would look exactly like a real one.

Usage
-----
    python scripts/analysis/downstream_utility.py --dry-run     # no GPU: plans only
    python scripts/analysis/downstream_utility.py               # ~1 GPU-day
    python scripts/analysis/downstream_utility.py --arms real synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_findings, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.dataset.loader import (  # noqa: E402
    LABEL_AIR,
    LABEL_PORE,
    MemmapPatchDataset,
)
from poregen.dataset.patch_index import (  # noqa: E402
    generate_patch_coords,
    patch_fractions,
)
from poregen.eval_v4.cases import build_cases  # noqa: E402
from poregen.losses.mask import combined_class_loss  # noqa: E402
from poregen.metrics.seg import multiclass_metrics, porosity_binned_mae  # noqa: E402
from poregen.models.vae.v2.unet import DownBlockV2, UpBlockV2  # noqa: E402
from poregen.training.device import get_autocast_dtype, select_device  # noqa: E402
from poregen.training.seed import seed_everything  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed geometry and paths
# ---------------------------------------------------------------------------

#: Patch size and stride of the real dataset (``data/split_v3/patches_meta.json``).
#: The synthetic volumes are cut the same way, so a synthetic patch and a real
#: patch are the same kind of object.
PATCH = 64
STRIDE = 32

DATA_ROOT = REPO / "data" / "split_v3"
PATCH_INDEX = DATA_ROOT / "patch_index.parquet"
CLASS_WEIGHTS_JSON = DATA_ROOT / "class_weights.json"
CAMPAIGN_ROOT = REPO / "runs" / "campaigns" / "12-eval-v4"
OUT_ROOT = REPO / "runs" / "campaigns" / "14-downstream-utility"

#: Porosity bin edges.  These ARE the bins the per-bin error is reported in
#: (``poregen.metrics.seg.porosity_binned_mae`` defaults), so matching the
#: synthetic draw on them makes the match exact in the units of the answer.
POROSITY_BINS: tuple[float, ...] = (0.0, 0.01, 0.03, 0.06, float("inf"))

#: A patch either touches the specimen boundary or it does not.  Two bins, not
#: four: the air FRACTION of a boundary patch is set by where the box falls,
#: and splitting it further empties strata the synthetic set cannot fill.
AIR_PRESENT_THRESHOLD = 1e-9

#: eval v4 assessments whose cases are legitimate training material - see the
#: module docstring for what is excluded and why.
SYNTHETIC_ASSESSMENTS: tuple[str, ...] = (
    "sampler",
    "porosity_global",
    "microstructure",
    "surface",
)

#: The dose-response point outside the training porosity range.  It is a
#: deliberate stress test of the model, not a request the real material makes,
#: so it is not training data.
OFF_MANIFOLD_PHI = 0.15

#: The test-set subset every arm and every seed is scored on.  Fixed and
#: independent of the arm seed on purpose: a different eval set per arm would
#: put eval noise into the comparison the campaign exists to make.
EVAL_SEED = 20260911


# ---------------------------------------------------------------------------
# The budget and the arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Budget:
    """Everything the three arms hold IDENTICAL.

    Nothing here is per-arm.  That is the whole point: if a setting can differ
    between arms the comparison is worthless, so the settings live in one
    frozen object that the arm never touches.
    """

    steps: int = 8000
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    base_channels: int = 32
    #: Total training patches per arm.  Sized so every stratum of the real
    #: histogram can be filled from the synthetic pool with modest reuse.
    patch_count: int = 16000
    #: Real TEST patches every arm is scored on.
    test_patches: int = 20000
    seeds: tuple[int, ...] = (101, 202, 303)


@dataclass(frozen=True)
class Arm:
    """One data mix.  ``real_fraction`` is the ONLY per-arm degree of freedom."""

    name: str
    real_fraction: float


ARMS: tuple[Arm, ...] = (
    Arm("real", 1.0),
    Arm("synthetic", 0.0),
    Arm("real_plus_synthetic", 0.5),
)

REFERENCE_ARM = "real"


def arm_by_name(name: str) -> Arm:
    for a in ARMS:
        if a.name == name:
            return a
    raise KeyError(f"unknown arm {name!r}; choose from {[a.name for a in ARMS]}")


def arm_patch_counts(arm: Arm, budget: Budget) -> tuple[int, int]:
    """``(n_real, n_synthetic)``, which ALWAYS sum to ``budget.patch_count``.

    The synthetic count is the remainder rather than a second rounding, so no
    arm can end up one patch short of another through a rounding difference.
    """
    n_real = int(round(budget.patch_count * arm.real_fraction))
    n_real = max(0, min(budget.patch_count, n_real))
    return n_real, budget.patch_count - n_real


# ---------------------------------------------------------------------------
# Stratification: the real request distribution, as a histogram
# ---------------------------------------------------------------------------

N_POROSITY_BINS = len(POROSITY_BINS) - 1
N_AIR_BINS = 2
N_STRATA = N_POROSITY_BINS * N_AIR_BINS


def stratum_keys(porosity: np.ndarray, air: np.ndarray) -> np.ndarray:
    """Joint ``(porosity bin, has air)`` stratum index per patch, in ``[0, N_STRATA)``."""
    por_bin = np.digitize(np.asarray(porosity, np.float64), np.asarray(POROSITY_BINS[1:-1]))
    air_bin = (np.asarray(air, np.float64) >= AIR_PRESENT_THRESHOLD).astype(np.int64)
    return (por_bin * N_AIR_BINS + air_bin).astype(np.int64)


def stratum_label(key: int) -> str:
    por_bin, air_bin = divmod(int(key), N_AIR_BINS)
    lo, hi = POROSITY_BINS[por_bin], POROSITY_BINS[por_bin + 1]
    hi_s = "inf" if not np.isfinite(hi) else f"{hi:g}"
    return f"phi[{lo:g},{hi_s})/{'air' if air_bin else 'interior'}"


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Split *total* across the POSITIVE *weights* as integers summing to it.

    Zero-weight entries stay at zero.  Letting one of them collect a rounding
    unit is how a redistribution quietly puts patches back into the stratum it
    was redistributing away from.
    """
    w = np.asarray(weights, np.float64)
    out = np.zeros(w.size, np.int64)
    if total <= 0:
        return out
    pos = np.flatnonzero(w > 0)
    if pos.size == 0:
        raise ValueError("cannot distribute a remainder over zero total weight")
    exact = w[pos] * (total / w[pos].sum())
    base = np.floor(exact).astype(np.int64)
    for i in np.argsort(-(exact - base))[: total - int(base.sum())]:
        base[i] += 1
    out[pos] = base
    return out


def match_strata(
    target_keys: np.ndarray,
    pool_keys: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Draw ``len(target_keys)`` pool indices reproducing the target histogram.

    A stratum the pool cannot supply AT ALL is not silently ignored: its quota
    is redistributed over the strata that exist, and the report names it.  A
    stratum the pool can supply but not in the quantity asked for is drawn with
    replacement, and the report says by how much.  Both facts belong in the
    campaign README, because both bound what the synthetic arm could have
    learnt.

    Returns the selected pool indices (shuffled) and a report dict.
    """
    target_keys = np.asarray(target_keys, np.int64)
    pool_keys = np.asarray(pool_keys, np.int64)
    n = int(target_keys.size)

    asked = np.bincount(target_keys, minlength=N_STRATA).astype(np.int64)
    by_stratum = [np.flatnonzero(pool_keys == k) for k in range(N_STRATA)]

    unfilled = [int(k) for k in range(N_STRATA) if asked[k] > 0 and by_stratum[k].size == 0]
    want = asked.copy()
    deficit = int(want[unfilled].sum()) if unfilled else 0
    want[unfilled] = 0

    if deficit:
        if not want.any():
            raise ValueError(
                "the synthetic pool has no patch in any stratum the real "
                "distribution asks for; there is nothing to train on."
            )
        want += _largest_remainder(want.astype(np.float64), deficit)

    picks: list[np.ndarray] = []
    detail: dict[str, dict] = {}
    max_reuse = 1.0
    for k in range(N_STRATA):
        n_k, pool_k = int(want[k]), by_stratum[k]
        if asked[k] == 0 and n_k == 0:
            continue
        reuse = (n_k / pool_k.size) if pool_k.size else float("inf")
        detail[stratum_label(k)] = {
            "asked": int(asked[k]),
            "drawn": n_k,
            "pool": int(pool_k.size),
            "reuse": round(float(reuse), 3) if np.isfinite(reuse) else None,
        }
        if n_k == 0:
            continue
        picks.append(rng.choice(pool_k, size=n_k, replace=n_k > pool_k.size))
        max_reuse = max(max_reuse, reuse)

    out = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    rng.shuffle(out)
    if out.size != n:
        raise AssertionError(f"matched {out.size} patches, asked for {n}")

    report = {
        "n": n,
        "strata": detail,
        "unfilled_strata": [stratum_label(k) for k in unfilled],
        "redistributed_patches": deficit,
        "max_stratum_reuse": round(float(max_reuse), 3),
    }
    return out.astype(np.int64), report


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_flips(
    xct: torch.Tensor, label: torch.Tensor, flips: tuple[bool, bool, bool]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip the LAST three (spatial) axes of both tensors by the same pattern.

    ``xct`` carries a leading channel axis and ``label`` does not, so the axes
    are addressed from the end.  Addressing them from the front would flip the
    channel axis of one and a spatial axis of the other, and a segmentation
    trained on a label that does not match its image still converges - it just
    learns nothing.
    """
    dims = [i for i, f in enumerate(flips) if f]
    if not dims:
        return xct, label
    return (
        torch.flip(xct, dims=[xct.ndim - 3 + d for d in dims]),
        torch.flip(label, dims=[label.ndim - 3 + d for d in dims]),
    )


def sample_flips() -> tuple[bool, bool, bool]:
    """One of the eight axis-flip combinations, from the ambient torch RNG.

    The ambient generator is what ``DataLoader`` re-seeds per worker per epoch,
    so the same patch is augmented differently on each pass without any
    per-index state of our own.
    """
    r = torch.rand(3)
    return bool(r[0] < 0.5), bool(r[1] < 0.5), bool(r[2] < 0.5)


class FlipAugmented(Dataset):
    """The same augmentation for every arm.  There is no per-arm switch."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, i: int):
        xct, label = self.base[i]
        return apply_flips(xct, label, sample_flips())


# ---------------------------------------------------------------------------
# Real patches
# ---------------------------------------------------------------------------

class RealPatchSubset(Dataset):
    """Selected rows of a :class:`MemmapPatchDataset`, as ``(xct, label)``."""

    def __init__(self, base: MemmapPatchDataset, rows: np.ndarray) -> None:
        self.base = base
        self.rows = np.asarray(rows, np.int64)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, i: int):
        item = self.base[int(self.rows[i])]
        return item["xct"], item["label"]


# ---------------------------------------------------------------------------
# Synthetic patches
# ---------------------------------------------------------------------------

class MissingSyntheticVolumes(RuntimeError):
    """Raised before any training when the eval v4 campaign is incomplete."""


def required_synthetic_cases(repo: Path | None = None) -> list[tuple[str, str]]:
    """``(assessment, case name)`` for every case the synthetic arm needs."""
    out: list[tuple[str, str]] = []
    for assessment in SYNTHETIC_ASSESSMENTS:
        for case in build_cases(assessment, repo):
            if case.target_phi is not None and float(case.target_phi) == OFF_MANIFOLD_PHI:
                continue
            out.append((assessment, case.name))
    return out


def check_synthetic_volumes(
    campaign_root: Path, repo: Path | None = None
) -> list[Path]:
    """Every required case directory, or raise naming what is missing.

    Called BEFORE the first optimiser step of any arm that uses synthetic data.
    A partial campaign trains perfectly well and answers a different question,
    and nothing downstream would reveal which one.
    """
    campaign_root = Path(campaign_root)
    found: list[Path] = []
    missing: list[str] = []
    for assessment, name in required_synthetic_cases(repo):
        d = campaign_root / assessment / "volumes" / name
        if all((d / f).exists() for f in ("manifest.json", "volume.tif", "label.tif")):
            found.append(d)
        else:
            missing.append(f"{assessment}/{name}")

    if missing:
        shown = "\n  ".join(missing[:12])
        more = f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise MissingSyntheticVolumes(
            f"{len(missing)} of {len(missing) + len(found)} required eval v4 cases "
            f"are missing from {campaign_root}.\n"
            f"  {shown}{more}\n"
            "A case counts as present only with manifest.json, volume.tif and "
            "label.tif - the manifest is written last, so a case without one is "
            "half-generated.\n"
            "This campaign will not train on a partial synthetic set: the arms "
            "would no longer be comparable and nothing in the result would say so.\n"
            "Generate them first, one command per assessment:\n"
            + "".join(
                f"    eval_v4 generate {a} --model <ldm06 run> --ckpt <step> "
                f"--out {campaign_root}\n"
                for a in SYNTHETIC_ASSESSMENTS
            )
        )
    return found


@dataclass
class SyntheticPool:
    """Every 64-cubed patch of every required case, described but not read.

    Porosity and air fraction come from the integral-volume patch summary, so
    describing a 1024-wide case costs one pass over its label and not one read
    per patch.
    """

    case_dirs: list[Path]
    #: ``(N, 4)`` int32 - case index, then ``(z0, y0, x0)``.
    coords: np.ndarray
    porosity: np.ndarray
    air: np.ndarray

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    @property
    def keys(self) -> np.ndarray:
        return stratum_keys(self.porosity, self.air)


def scan_synthetic_pool(case_dirs: list[Path], verbose: bool = True) -> SyntheticPool:
    import tifffile  # noqa: PLC0415  - only needed on the synthetic path

    coords_all, por_all, air_all = [], [], []
    for ci, d in enumerate(case_dirs):
        label = np.asarray(tifffile.memmap(str(d / "label.tif")))
        coords = np.asarray(generate_patch_coords(label.shape, PATCH, STRIDE), np.int64)
        if coords.size == 0:
            continue
        por = patch_fractions(label == LABEL_PORE, coords, PATCH)
        air = patch_fractions(label == LABEL_AIR, coords, PATCH)
        block = np.empty((coords.shape[0], 4), np.int32)
        block[:, 0] = ci
        block[:, 1:] = coords
        coords_all.append(block)
        por_all.append(por)
        air_all.append(air)
        del label
        if verbose:
            print(f"  scanned {d.parent.parent.name}/{d.name}: {coords.shape[0]} patches",
                  flush=True)

    if not coords_all:
        raise RuntimeError("no synthetic patches found - every case was too small")
    return SyntheticPool(
        case_dirs=list(case_dirs),
        coords=np.concatenate(coords_all),
        porosity=np.concatenate(por_all).astype(np.float32),
        air=np.concatenate(air_all).astype(np.float32),
    )


class SyntheticPatchSubset(Dataset):
    """Selected pool patches, read from the case TIFFs on demand.

    The TIFFs are memory-mapped rather than loaded: the selected patches of a
    1024-wide case are a few per cent of its 200 MB, and a memmap is the same
    access pattern the real arm has through ``patches_xct.bin``.  Handles open
    lazily, so each DataLoader worker gets its own after fork.
    """

    def __init__(self, pool: SyntheticPool, rows: np.ndarray) -> None:
        self.case_dirs = list(pool.case_dirs)
        self.coords = pool.coords[np.asarray(rows, np.int64)]
        self._handles: dict[int, tuple] = {}

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def _case(self, ci: int) -> tuple:
        if ci not in self._handles:
            import tifffile  # noqa: PLC0415
            d = self.case_dirs[ci]
            self._handles[ci] = (
                tifffile.memmap(str(d / "volume.tif")),
                tifffile.memmap(str(d / "label.tif")),
            )
        return self._handles[ci]

    def __getitem__(self, i: int):
        ci, z0, y0, x0 = (int(v) for v in self.coords[i])
        vol, lab = self._case(ci)
        sl = np.s_[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH]
        xct = torch.from_numpy(np.asarray(vol[sl], np.float32)).mul_(1.0 / 255.0).unsqueeze(0)
        label = torch.from_numpy(np.asarray(lab[sl]).astype(np.int64))
        return xct, label


# ---------------------------------------------------------------------------
# The model - one architecture, built from the budget alone
# ---------------------------------------------------------------------------

class SegUNet3D(nn.Module):
    """A plain 3-D U-Net: grey patch in, three class logits out.

    Built from the repo's own v2 blocks (``DownBlockV2`` / ``UpBlockV2``), so
    the normalisation, activation and upsampling are the ones every other
    PoreGen model uses and none of them is a new variable in this comparison.
    Three stride-2 levels take a 64-cubed patch to 8 cubed; the decoder takes
    the input itself as its finest skip, so the head sees full resolution.
    """

    def __init__(self, base_channels: int = 32, in_channels: int = 1, n_classes: int = 3):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.down1 = DownBlockV2(in_channels, c1)      # 64 -> 32
        self.down2 = DownBlockV2(c1, c2)               # 32 -> 16
        self.down3 = DownBlockV2(c2, c3)               # 16 ->  8
        self.up1 = UpBlockV2(c3, c2, c2)               #  8 -> 16
        self.up2 = UpBlockV2(c2, c1, c1)               # 16 -> 32
        self.up3 = UpBlockV2(c1, in_channels, c1)      # 32 -> 64
        self.head = nn.Conv3d(c1, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        h = self.up1(d3, d2)
        h = self.up2(h, d1)
        h = self.up3(h, x)
        return self.head(h)


def build_model(budget: Budget) -> SegUNet3D:
    """The model depends on the BUDGET and on nothing else.

    There is deliberately no ``arm`` argument.  An architecture that could
    differ between arms is the single easiest way to invalidate the whole
    campaign, so the call site has no way to express one.
    """
    return SegUNet3D(base_channels=budget.base_channels)


def load_class_weights() -> list[float]:
    """The 3-class loss weights of the REAL train split.

    The same weights in every arm, including the synthetic one: they encode the
    real class balance the task is scored against, not the balance of whatever
    data an arm happens to hold.
    """
    return list(json.loads(CLASS_WEIGHTS_JSON.read_text())["class_weights"])


# ---------------------------------------------------------------------------
# The plan - what each arm trains on
# ---------------------------------------------------------------------------

@dataclass
class ArmPlan:
    arm: str
    seed: int
    real_rows: np.ndarray
    synthetic_rows: np.ndarray
    match_report: dict

    @property
    def total(self) -> int:
        return int(self.real_rows.size + self.synthetic_rows.size)


def seed_permutation(n_real_pool: int, budget: Budget, seed: int) -> np.ndarray:
    """The ONE real-patch draw a seed makes, shared by all three arms.

    Arm (a) trains on all of it.  Arm (c) trains on its first half and asks for
    synthetic patches matching the histogram of its second half.  Arm (b) asks
    for synthetic patches matching the histogram of the whole thing.  So every
    arm at a given seed targets the same real distribution, patch for patch,
    and arm (c)'s real half is literally a subset of arm (a)'s patches.
    """
    rng = np.random.default_rng([seed, 0xD0])
    return rng.permutation(n_real_pool)[: budget.patch_count].astype(np.int64)


def plan_arm(
    arm: Arm,
    budget: Budget,
    seed: int,
    perm: np.ndarray,
    real_keys: np.ndarray,
    pool_keys: np.ndarray | None,
) -> ArmPlan:
    n_real, n_synth = arm_patch_counts(arm, budget)
    real_rows = perm[:n_real]
    if n_synth == 0:
        return ArmPlan(arm.name, seed, real_rows, np.zeros(0, np.int64), {})
    if pool_keys is None:
        raise MissingSyntheticVolumes(
            f"arm {arm.name!r} needs {n_synth} synthetic patches but no pool was scanned."
        )
    target_keys = real_keys[perm[n_real:]]
    rng = np.random.default_rng([seed, 0x5E])
    synth_rows, report = match_strata(target_keys, pool_keys, rng)
    return ArmPlan(arm.name, seed, real_rows, synth_rows, report)


def arm_run_spec(arm: Arm, budget: Budget, seed: int, plan: ArmPlan) -> dict:
    """Everything that describes a run.  Everything but the data is shared."""
    return {
        "arm": arm.name,
        "seed": seed,
        "real_fraction": arm.real_fraction,
        "n_real_patches": int(plan.real_rows.size),
        "n_synthetic_patches": int(plan.synthetic_rows.size),
        "total_patches": plan.total,
        **asdict(budget),
    }


#: The keys of :func:`arm_run_spec` that are allowed to differ between arms.
#: Anything else differing means the comparison is broken.
ARM_VARYING_KEYS = frozenset(
    {"arm", "real_fraction", "n_real_patches", "n_synthetic_patches"}
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class SegmentationScore:
    """Pore/air Dice and per-bin porosity error over a whole test set.

    Dice is the repo's own :func:`multiclass_metrics`, accumulated as a
    patch-count-weighted mean over batches, so the number is the mean per-PATCH
    Dice of the full set and not a mean of batch means.  The per-bin porosity
    error is the repo's :func:`porosity_binned_mae` over every patch at once.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._n = 0
        self._pred_por: list[torch.Tensor] = []
        self._gt_por: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, label: torch.Tensor) -> None:
        n = int(label.shape[0])
        for k, v in multiclass_metrics(logits, label).items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
        self._n += n
        pred = logits.argmax(dim=1)
        self._pred_por.append((pred == LABEL_PORE).flatten(1).float().mean(1).cpu())
        self._gt_por.append((label == LABEL_PORE).flatten(1).float().mean(1).cpu())

    def result(self) -> dict[str, float]:
        if self._n == 0:
            raise RuntimeError("SegmentationScore.result() before any update()")
        out = {k: v / self._n for k, v in self._sums.items()}
        out.update(
            porosity_binned_mae(
                torch.cat(self._pred_por), torch.cat(self._gt_por), bins=POROSITY_BINS
            )
        )
        out["n_patches"] = float(self._n)
        return out


# ---------------------------------------------------------------------------
# Train and evaluate one arm
# ---------------------------------------------------------------------------

def build_train_dataset(
    plan: ArmPlan, real_base: MemmapPatchDataset, pool: SyntheticPool | None
) -> Dataset:
    parts: list[Dataset] = []
    if plan.real_rows.size:
        parts.append(RealPatchSubset(real_base, plan.real_rows))
    if plan.synthetic_rows.size:
        assert pool is not None
        parts.append(SyntheticPatchSubset(pool, plan.synthetic_rows))
    base = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return FlipAugmented(base)


def _endless(loader: DataLoader):
    """Cycle a loader for a fixed STEP budget, however small the dataset is."""
    if len(loader) == 0:
        raise ValueError(
            f"a batch of {loader.batch_size} does not fit in "
            f"{len(loader.dataset)} patches with drop_last - the arm would "  # type: ignore[arg-type]
            "never take a step"
        )
    while True:
        yield from loader


def train_and_score(
    arm: Arm,
    budget: Budget,
    seed: int,
    plan: ArmPlan,
    real_base: MemmapPatchDataset,
    pool: SyntheticPool | None,
    test_loader: DataLoader,
    device: torch.device,
    num_workers: int,
) -> dict:
    seed_everything(seed, deterministic=False)
    model = build_model(budget).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=budget.lr,
                            weight_decay=budget.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=budget.steps)
    amp_dtype = get_autocast_dtype(device)
    class_weights = torch.tensor(load_class_weights(), dtype=torch.float32, device=device)

    train_ds = build_train_dataset(plan, real_base, pool)
    loader = DataLoader(
        train_ds,
        batch_size=budget.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    t0 = time.time()
    model.train()
    running, n_running = 0.0, 0
    batches = _endless(loader)
    for step in range(1, budget.steps + 1):
        xct, label = next(batches)
        xct = xct.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype):
            logits = model(xct)
        loss = combined_class_loss(
            logits.float(), label, class_weights=class_weights
        )["class_total"]
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += float(loss.detach())
        n_running += 1
        if step % 500 == 0 or step == budget.steps:
            print(f"    {arm.name} seed {seed}  step {step}/{budget.steps}  "
                  f"loss {running / n_running:.4f}", flush=True)
            running, n_running = 0.0, 0
    train_s = time.time() - t0

    model.eval()
    score = SegmentationScore()
    with torch.no_grad():
        for xct, label in test_loader:
            xct = xct.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype):
                logits = model(xct)
            score.update(logits.float(), label)

    out = arm_run_spec(arm, budget, seed, plan)
    out["metrics"] = score.result()
    out["train_seconds"] = round(train_s, 1)
    out["match_report"] = plan.match_report
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HEADLINE_KEYS = ("dice_pore", "dice_air", "porosity_mae", "air_mae")


def aggregate(runs: list[dict]) -> dict:
    """Mean and sd over seeds, per arm, for every metric."""
    out: dict[str, dict] = {}
    for arm in {r["arm"] for r in runs}:
        rows = [r for r in runs if r["arm"] == arm]
        keys = sorted(rows[0]["metrics"])
        agg = {}
        for k in keys:
            vals = np.array([r["metrics"][k] for r in rows], np.float64)
            finite = vals[np.isfinite(vals)]
            agg[k] = {
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                "n_seeds": int(finite.size),
            }
        out[arm] = agg
    return out


def _fmt(v: float, places: int = 4) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{places}f}"


def render_findings(results: dict) -> str:
    agg = results["aggregate"]
    order = [a for a in (REFERENCE_ARM, "real_plus_synthetic", "synthetic") if a in agg]
    ref = agg.get(REFERENCE_ARM)
    bin_keys = sorted(
        k for k in next(iter(agg.values())) if k.startswith("porosity_mae_bin_")
    )

    lines = [
        "# Downstream utility of ldm06 volumes",
        "",
        "Is the synthetic data useful, or only realistic?  Three arms at one "
        "identical training budget, scored on the REAL held-out test panels.",
        "",
        f"Budget: {results['budget']['steps']} steps x batch "
        f"{results['budget']['batch_size']}, {results['budget']['patch_count']} training "
        f"patches per arm, {results['budget']['test_patches']} real test patches, "
        f"seeds {list(results['budget']['seeds'])}.",
        "",
        "## Headline (mean +/- sd over seeds)",
        "",
        "| arm | pore Dice | air Dice | porosity MAE | air MAE |",
        "|---|---|---|---|---|",
    ]
    for arm in order:
        a = agg[arm]
        cells = [f"{_fmt(a[k]['mean'])} +/- {_fmt(a[k]['sd'])}" for k in HEADLINE_KEYS]
        tag = f"**{arm}** (reference)" if arm == REFERENCE_ARM else arm
        lines.append("| " + " | ".join([tag, *cells]) + " |")

    if ref is not None:
        lines += ["", "Delta against the real-only arm:", "",
                  "| arm | d pore Dice | d air Dice | d porosity MAE | d air MAE |",
                  "|---|---|---|---|---|"]
        for arm in order:
            if arm == REFERENCE_ARM:
                continue
            cells = [
                _fmt(agg[arm][k]["mean"] - ref[k]["mean"]) for k in HEADLINE_KEYS
            ]
            lines.append("| " + " | ".join([arm, *cells]) + " |")

    lines += ["", "## Porosity error per GT-porosity bin", "",
              "| arm | " + " | ".join(
                  f"phi[{POROSITY_BINS[i]:g},"
                  f"{'inf' if not np.isfinite(POROSITY_BINS[i + 1]) else f'{POROSITY_BINS[i + 1]:g}'})"
                  for i in range(len(bin_keys))
              ) + " |",
              "|---|" + "---|" * len(bin_keys)]
    for arm in order:
        cells = [_fmt(agg[arm][k]["mean"]) for k in bin_keys]
        lines.append("| " + " | ".join([arm, *cells]) + " |")
    counts = [int(agg[order[0]][k.replace("mae", "n")]["mean"]) for k in bin_keys]
    lines += ["", f"Test patches per bin: {counts}.  A bin with few patches carries "
                  "a noisy MAE - the counts are part of the reading, not decoration."]

    caveats = results.get("caveats", [])
    if caveats:
        lines += ["", "## Caveats", ""] + [f"- {c}" for c in caveats]
    return "\n".join(lines) + "\n"


def render_readme(results: dict, command: str) -> str:
    agg = results["aggregate"]
    b = results["budget"]
    head = []
    for arm in (REFERENCE_ARM, "real_plus_synthetic", "synthetic"):
        if arm in agg:
            head.append(
                f"- `{arm}`: pore Dice {_fmt(agg[arm]['dice_pore']['mean'])} "
                f"+/- {_fmt(agg[arm]['dice_pore']['sd'])}, "
                f"air Dice {_fmt(agg[arm]['dice_air']['mean'])}, "
                f"porosity MAE {_fmt(agg[arm]['porosity_mae']['mean'])}"
            )
    lines = [
        "# 13 - Downstream utility",
        "",
        "## Question",
        "",
        "Is the ldm06 synthetic data USEFUL, not merely realistic?  Does a "
        "segmentation model trained on generated volumes work on real ones?",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Checkpoint and settings",
        "",
        f"- Synthetic volumes: `{results['campaign_root']}`, assessments "
        f"{', '.join(SYNTHETIC_ASSESSMENTS)} (off-manifold phi "
        f"{OFF_MANIFOLD_PHI} excluded), {results['n_synthetic_cases']} cases, "
        f"{results['synthetic_pool_size']} candidate patches.",
        f"- Model: plain 3-D U-Net, base {b['base_channels']} channels, three "
        f"stride-2 levels, 3-class head.  Identical in every arm.",
        f"- Budget: {b['steps']} steps x batch {b['batch_size']}, AdamW lr "
        f"{b['lr']} cosine to zero, {b['patch_count']} patches per arm, "
        f"axis-flip augmentation, seeds {list(b['seeds'])}.",
        f"- Test: {b['test_patches']} patches of the split_v3 TEST panels "
        "(fixed subset, identical for every arm and seed).",
        "",
        "## Headline numbers",
        "",
        *head,
        "",
        "Full tables in `findings.md`; every run in `results.json`.",
        "",
        "## Caveats and limitations",
        "",
        *[f"- {c}" for c in results.get("caveats", [])],
        "",
        "## Vault note",
        "",
        "note pending",
    ]
    return "\n".join(lines) + "\n"


def build_caveats(results: dict) -> list[str]:
    out = [
        "The synthetic labels are the ldm06 decoder's own 3-class output, so the "
        "synthetic arm learns the decoder's grey-to-label mapping.  The real arm "
        "learns the `onlypores` segmentation the dataset was built with.  Part of "
        "any gap is that difference, not sample quality.",
        "All three arms hold the TOTAL patch count fixed, so arm "
        "`real_plus_synthetic` REPLACES half the real data rather than adding to "
        "it.  This campaign does not measure augmentation on top of the full real "
        "training set.",
        "The synthetic pool is finite.  Where a stratum of the real histogram is "
        "larger than the synthetic patches available in it, patches repeat; "
        "`match_report.max_stratum_reuse` in `results.json` is the worst case.",
        "Air in the synthetic set comes only from the `surface` assessment, whose "
        "specimen faces are flat or Gaussian-rough planes.  Real air also comes "
        "from lateral specimen edges, which no case requests.",
    ]
    unfilled = sorted({
        s for r in results["runs"] for s in r.get("match_report", {}).get("unfilled_strata", [])
    })
    if unfilled:
        out.append(
            "The synthetic pool has no patch at all in these strata of the real "
            f"distribution: {', '.join(unfilled)}.  Their share of the draw was "
            "redistributed over the strata that exist, so the synthetic arm never "
            "saw that material."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_test_loader(
    real_test: MemmapPatchDataset, budget: Budget, num_workers: int, device: torch.device
) -> DataLoader:
    rng = np.random.default_rng(EVAL_SEED)
    n = min(budget.test_patches, len(real_test))
    rows = np.sort(rng.permutation(len(real_test))[:n])
    return DataLoader(
        RealPatchSubset(real_test, rows),
        batch_size=budget.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT,
                   help="eval v4 campaign holding the synthetic volumes")
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--arms", nargs="+", default=[a.name for a in ARMS],
                   choices=[a.name for a in ARMS])
    p.add_argument("--steps", type=int, default=Budget.steps)
    p.add_argument("--batch-size", type=int, default=Budget.batch_size)
    p.add_argument("--patch-count", type=int, default=Budget.patch_count)
    p.add_argument("--base-channels", type=int, default=Budget.base_channels,
                   help="U-Net width - a BUDGET knob, shared by every arm")
    p.add_argument("--test-patches", type=int, default=Budget.test_patches)
    p.add_argument("--seeds", type=int, nargs="+", default=list(Budget.seeds))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="check the campaign, scan the pool, build and print every "
                        "arm plan - no model, no GPU")
    args = p.parse_args(argv)

    budget = Budget(
        steps=args.steps,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        patch_count=args.patch_count,
        test_patches=args.test_patches,
        seeds=tuple(args.seeds),
    )
    arms = [arm_by_name(n) for n in args.arms]
    needs_synthetic = any(a.real_fraction < 1.0 for a in arms)

    pool: SyntheticPool | None = None
    case_dirs: list[Path] = []
    if needs_synthetic:
        print(f"Checking eval v4 campaign {args.campaign_root} ...", flush=True)
        try:
            case_dirs = check_synthetic_volumes(args.campaign_root, REPO)
        except MissingSyntheticVolumes as exc:
            # A stack trace here would bury the one thing the operator needs.
            print(f"\nREFUSING TO RUN\n\n{exc}", file=sys.stderr)
            return 2
        print(f"  {len(case_dirs)} cases present.  Scanning patches ...", flush=True)
        pool = scan_synthetic_pool(case_dirs)
        print(f"  synthetic pool: {len(pool)} patches", flush=True)

    print("Loading the real patch index ...", flush=True)
    real_train = MemmapPatchDataset(PATCH_INDEX, DATA_ROOT, split="train")
    real_keys = stratum_keys(real_train.df["porosity"].to_numpy(),
                             real_train.df["air_fraction"].to_numpy())

    plans: dict[tuple[str, int], ArmPlan] = {}
    for seed in budget.seeds:
        perm = seed_permutation(len(real_train), budget, seed)
        for arm in arms:
            plans[(arm.name, seed)] = plan_arm(
                arm, budget, seed, perm, real_keys,
                pool.keys if pool is not None else None,
            )

    for (name, seed), plan in plans.items():
        print(f"  plan {name:<20} seed {seed}: {plan.real_rows.size} real + "
              f"{plan.synthetic_rows.size} synthetic = {plan.total}", flush=True)
        if plan.match_report:
            print(f"      max stratum reuse "
                  f"{plan.match_report['max_stratum_reuse']}, unfilled "
                  f"{plan.match_report['unfilled_strata'] or 'none'}", flush=True)

    if args.dry_run:
        print("\n--dry-run: plans built, nothing trained.")
        return 0

    device = select_device()
    real_test = MemmapPatchDataset(PATCH_INDEX, DATA_ROOT, split="test")
    test_loader = build_test_loader(real_test, budget, args.num_workers, device)
    print(f"Device {device}; scoring on {len(test_loader.dataset)} real test patches.",
          flush=True)

    runs = []
    for arm in arms:
        for seed in budget.seeds:
            print(f"\n=== {arm.name}  seed {seed} ===", flush=True)
            runs.append(train_and_score(
                arm, budget, seed, plans[(arm.name, seed)], real_train, pool,
                test_loader, device, args.num_workers,
            ))

    results = {
        "question": "Is the ldm06 synthetic data useful, not merely realistic?",
        "campaign_root": str(args.campaign_root),
        "n_synthetic_cases": len(case_dirs),
        "synthetic_pool_size": len(pool) if pool is not None else 0,
        "budget": asdict(budget),
        "reference_arm": REFERENCE_ARM,
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    results["caveats"] = build_caveats(results)

    out_dir = Path(args.out)
    write_json(results, out_dir)
    write_findings(render_findings(results), out_dir)
    command = "python scripts/analysis/downstream_utility.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    (out_dir / "README.md").write_text(render_readme(results, command.strip()))
    print(f"\nWrote {out_dir}/results.json, findings.md, README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

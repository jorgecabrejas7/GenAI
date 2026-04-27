"""Evaluation output-saving routines for the PoreGen VAE eval pipeline.

This module is the single place where all eval artefacts are written to disk.
Every public function documents precisely what it saves and how to load it,
so that the output directory is self-describing without the README.

Output directory layout
-----------------------
::

    <run_dir>/eval/<timestamp>-<tier>/
        README.md                     ← human-readable manifest + load guide
        eval_metrics.json             ← all scalar metrics + eval config
        patch_arrays.npz              ← per-patch arrays for scatter plots
        latent_audit.json             ← latent-space audit scalars
        prior_samples.png             ← z~N(0,I) decode grid
        porosity_scatter.png          ← φ_gt vs φ_pred scatter
        latent_audit.png              ← per-channel KL / σ figure
        volumes/                      ← one sub-directory per reconstructed volume
            <vol_name>/
                <vol_name>_arrays.npz           ← all 3 reconstruction modes + GT
                <vol_name>_xct_stoch_mean.tiff  ← float32 [0,1]
                <vol_name>_mask_stoch_mean.tiff ← uint8 {0,255}
                <vol_name>_xct_stoch_single.tiff
                <vol_name>_mask_stoch_single.tiff
                <vol_name>_xct_mu.tiff
                <vol_name>_mask_mu.tiff
                <vol_name>_xct_std.tiff         ← float32 voxel-wise std
                <vol_name>_mask_std.tiff
                <vol_name>_metrics.json
                <vol_name>_slices_axial_stoch_mean.png
                <vol_name>_slices_coronal_stoch_mean.png
                <vol_name>_slices_sagittal_stoch_mean.png
                <vol_name>_slices_axial_stoch_single.png
                <vol_name>_slices_coronal_stoch_single.png
                <vol_name>_slices_sagittal_stoch_single.png
                <vol_name>_slices_axial_mu.png
                <vol_name>_slices_coronal_mu.png
                <vol_name>_slices_sagittal_mu.png
                <vol_name>_std_axial.png
                <vol_name>_std_coronal.png
                <vol_name>_std_sagittal.png
                <vol_name>_s2r.png              ← only if run_s2r
                <vol_name>_psd.png              ← only if run_psd
                <vol_name>_pore_small.gif       ← only if run_pore_gifs
                <vol_name>_pore_medium.gif
                <vol_name>_pore_large.gif
"""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile

if TYPE_CHECKING:
    from poregen.eval.config import EvalConfig
    from poregen.eval.metrics import LatentAudit, PatchMetrics, VolumeMetrics
    from poregen.eval.stochastic import VolumeReconstruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory creation helper
# ---------------------------------------------------------------------------

def make_eval_output_dir(
    run_dir: Path,
    eval_cfg: "EvalConfig",
    *,
    timestamp: str | None = None,
) -> Path:
    """Create and return a timestamped eval output directory under *run_dir*.

    The directory is named ``<timestamp>-<tier>`` so that multiple eval runs
    on the same checkpoint are never clobbered.

    Parameters
    ----------
    run_dir : Path
        Training run directory (the one containing ``best.ckpt`` etc.).
    eval_cfg : EvalConfig
        Used to embed the tier label in the directory name.
    timestamp : str, optional
        Override the timestamp string (``YYYYMMDD-HHMMSS``).  Useful for
        deterministic test fixtures.

    Returns
    -------
    Path
        The newly created directory.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = run_dir / "eval" / f"{ts}-{eval_cfg.tier}"
    (out_dir / "volumes").mkdir(parents=True, exist_ok=True)
    return out_dir


def volume_out_dir(out_dir: Path, vol_name: str) -> Path:
    """Return (and create) the per-volume subdirectory inside *out_dir*."""
    d = out_dir / "volumes" / vol_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Volume arrays and TIFFs
# ---------------------------------------------------------------------------

def save_volume_npz(
    vol: "VolumeReconstruction",
    vol_name: str,
    out_dir: Path,
) -> Path:
    """Save all reconstruction modes and GT arrays for one volume as a single NPZ.

    The file is ``<out_dir>/volumes/<vol_name>/<vol_name>_arrays.npz``.

    Arrays stored
    -------------
    ``xct_gt``           float32, shape (Dz, Dy, Dx), [0, 1]
    ``mask_gt``          float32, shape (Dz, Dy, Dx), {0, 1}
    ``xct_stoch_mean``   float32, [0, 1] — mean of N=50 stochastic passes
    ``mask_stoch_mean``  float32, [0, 1]
    ``xct_stoch_std``    float32, ≥0  — voxel-wise std across N passes
    ``mask_stoch_std``   float32, ≥0
    ``xct_stoch_single`` float32, [0, 1] — first of the N stochastic passes
    ``mask_stoch_single`` float32, [0, 1]
    ``xct_mu``           float32, [0, 1] — deterministic z = μ (no sampling)
    ``mask_mu``          float32, [0, 1]

    Loading example
    ---------------
    >>> data = np.load("MyCylinder_arrays.npz")
    >>> xct_gt = data["xct_gt"]        # shape (D, H, W), float32
    >>> xct_mean = data["xct_stoch_mean"]

    Parameters
    ----------
    vol : VolumeReconstruction
    vol_name : str
        Short volume name used in the filename.
    out_dir : Path
        Root eval output directory.  The file is written to
        ``out_dir/volumes/<vol_name>/<vol_name>_arrays.npz``.

    Returns
    -------
    Path
        Path to the saved NPZ file.
    """
    vdir = volume_out_dir(out_dir, vol_name)
    path = vdir / f"{vol_name}_arrays.npz"

    metadata_str = json.dumps({
        "volume_id": vol.volume_id,
        "shape_original": list(vol.shape_original),
        "shape_tiled": list(vol.shape_tiled),
        "arrays": {
            "xct_gt": "float32 [0,1] — original XCT normalised to [0,1]",
            "mask_gt": "float32 {0,1} — ground-truth pore mask",
            "xct_stoch_mean": "float32 [0,1] — mean of N stochastic reconstructions",
            "mask_stoch_mean": "float32 [0,1] — mean of N stochastic mask probabilities",
            "xct_stoch_std": "float32 ≥0 — voxel-wise std across N stochastic passes",
            "mask_stoch_std": "float32 ≥0 — voxel-wise mask std across N passes",
            "xct_stoch_single": "float32 [0,1] — single stochastic reconstruction (pass 1)",
            "mask_stoch_single": "float32 [0,1] — single stochastic mask (pass 1)",
            "xct_mu": "float32 [0,1] — deterministic reconstruction via z = mu",
            "mask_mu": "float32 [0,1] — deterministic mask via z = mu",
        },
    })

    np.savez_compressed(
        path,
        xct_gt=vol.xct_gt,
        mask_gt=vol.mask_gt,
        xct_stoch_mean=vol.xct_stoch_mean,
        mask_stoch_mean=vol.mask_stoch_mean,
        xct_stoch_std=vol.xct_stoch_std,
        mask_stoch_std=vol.mask_stoch_std,
        xct_stoch_single=vol.xct_stoch_single,
        mask_stoch_single=vol.mask_stoch_single,
        xct_mu=vol.xct_mu,
        mask_mu=vol.mask_mu,
        _metadata=np.array(metadata_str),
    )
    logger.info("  Saved %s", path)
    return path


def save_volume_tiffs(
    vol: "VolumeReconstruction",
    vol_name: str,
    out_dir: Path,
) -> None:
    """Save six reconstruction TIFFs and two std TIFFs for one volume.

    Files written
    -------------
    ``<vol_name>_xct_stoch_mean.tiff``   float32 [0,1]
    ``<vol_name>_mask_stoch_mean.tiff``  uint8 {0, 255}  (thresholded at 0.5)
    ``<vol_name>_xct_stoch_single.tiff`` float32 [0,1]
    ``<vol_name>_mask_stoch_single.tiff`` uint8 {0, 255}
    ``<vol_name>_xct_mu.tiff``           float32 [0,1]
    ``<vol_name>_mask_mu.tiff``          uint8 {0, 255}
    ``<vol_name>_xct_std.tiff``          float32 ≥0  (voxel-wise std)
    ``<vol_name>_mask_std.tiff``         float32 ≥0

    All TIFFs are saved with shape (D, H, W); no channel dimension.
    Float TIFFs can be opened directly in ImageJ/Fiji.  Binary mask TIFFs
    use 8-bit encoding (0 = matrix, 255 = pore) to match the convention used
    in the original dataset.

    Parameters
    ----------
    vol : VolumeReconstruction
    vol_name : str
    out_dir : Path
    """
    vdir = volume_out_dir(out_dir, vol_name)

    def _save_f32(arr: np.ndarray, name: str) -> None:
        p = vdir / name
        tifffile.imwrite(str(p), arr.astype(np.float32))
        logger.info("  Saved %s", p)

    def _save_mask(arr: np.ndarray, name: str) -> None:
        p = vdir / name
        tifffile.imwrite(str(p), (arr >= 0.5).astype(np.uint8) * 255)
        logger.info("  Saved %s", p)

    _save_f32(vol.xct_stoch_mean,   f"{vol_name}_xct_stoch_mean.tiff")
    _save_mask(vol.mask_stoch_mean, f"{vol_name}_mask_stoch_mean.tiff")
    _save_f32(vol.xct_stoch_single,   f"{vol_name}_xct_stoch_single.tiff")
    _save_mask(vol.mask_stoch_single, f"{vol_name}_mask_stoch_single.tiff")
    _save_f32(vol.xct_mu,   f"{vol_name}_xct_mu.tiff")
    _save_mask(vol.mask_mu, f"{vol_name}_mask_mu.tiff")
    _save_f32(vol.xct_stoch_std,  f"{vol_name}_xct_std.tiff")
    _save_f32(vol.mask_stoch_std, f"{vol_name}_mask_std.tiff")


# ---------------------------------------------------------------------------
# Patch-level arrays
# ---------------------------------------------------------------------------

def save_patch_npz(
    patch_metrics: "PatchMetrics | dict",
    out_dir: Path,
) -> Path:
    """Save per-patch arrays for scatter plots and detailed offline analysis.

    The file is ``<out_dir>/patch_arrays.npz``.

    Arrays stored
    -------------
    ``porosity_gt``    float32, (N,) — GT porosity per patch
    ``porosity_pred``  float32, (N,) — predicted porosity per patch
    ``volume_ids``     object array of strings — volume id for each patch

    These are the same values used by :func:`visualise.save_porosity_scatter`.
    Storing them allows the notebook to regenerate or customise the scatter
    plot without re-running eval.

    Loading example
    ---------------
    >>> data = np.load("patch_arrays.npz", allow_pickle=True)
    >>> gt   = data["porosity_gt"]
    >>> pred = data["porosity_pred"]
    >>> vids = data["volume_ids"]

    Parameters
    ----------
    patch_metrics : PatchMetrics
    out_dir : Path

    Returns
    -------
    Path
    """
    path = out_dir / "patch_arrays.npz"
    # Accept either a single PatchMetrics or a dict (mode → PatchMetrics);
    # in the latter case use "deterministic" as the canonical scatter source.
    if isinstance(patch_metrics, dict):
        patch_metrics = patch_metrics.get("deterministic", next(iter(patch_metrics.values())))
    sc = patch_metrics._porosity_scatter

    np.savez_compressed(
        path,
        porosity_gt=np.array(sc.get("gt", []), dtype=np.float32),
        porosity_pred=np.array(sc.get("pred", []), dtype=np.float32),
        volume_ids=np.array(sc.get("volume_id", []), dtype=object),
    )
    logger.info("  Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# JSON metrics
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v  # NaN → null
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_volume_metrics_json(
    vm: "VolumeMetrics",
    vol_name: str,
    out_dir: Path,
) -> Path:
    """Save per-volume scalar metrics to ``<out_dir>/volumes/<vol_name>/<vol_name>_metrics.json``.

    Large array fields (``psd_gt``, ``psd_pred``, ``s2_gt``, ``s2_pred``,
    ``s2_r_vals``) are stripped from the JSON to keep file sizes small — they
    are preserved in the NPZ.

    Parameters
    ----------
    vm : VolumeMetrics
    vol_name : str
    out_dir : Path

    Returns
    -------
    Path
    """
    from dataclasses import asdict
    vdir = volume_out_dir(out_dir, vol_name)
    path = vdir / f"{vol_name}_metrics.json"

    # strip large list fields
    _large = {"psd_gt", "psd_pred", "s2_gt", "s2_pred", "s2_r_vals"}
    data = {k: v for k, v in asdict(vm).items() if k not in _large}

    with open(path, "w") as fh:
        json.dump(_serialise(data), fh, indent=2)
    logger.info("  Saved %s", path)
    return path


def save_eval_metrics_json(
    vol_metrics: list["VolumeMetrics"],
    patch_metrics_val: "dict[str, PatchMetrics]",
    patch_metrics_test: "dict[str, PatchMetrics]",
    latent_audit: "LatentAudit",
    eval_cfg: "EvalConfig",
    step: int,
    run_name: str,
    checkpoint_path: str,
    out_dir: Path,
) -> Path:
    """Save the consolidated scalar metrics summary to ``<out_dir>/eval_metrics.json``.

    This file is the primary machine-readable output of an eval run.

    The ``patch_metrics_val`` and ``patch_metrics_test`` parameters are dicts
    keyed by decoding mode (``"deterministic"``, ``"stoch_single"``,
    ``"stoch_mean"``).  In the JSON they appear as
    ``patch_metrics_val_<mode>`` and ``patch_metrics_test_<mode>``.

    Parameters
    ----------
    vol_metrics : list of VolumeMetrics
    patch_metrics_val : dict[str, PatchMetrics]
    patch_metrics_test : dict[str, PatchMetrics]
    latent_audit : LatentAudit
    eval_cfg : EvalConfig
    step : int
    run_name : str
    checkpoint_path : str
    out_dir : Path

    Returns
    -------
    Path
    """
    from dataclasses import asdict

    _large = {"psd_gt", "psd_pred", "s2_gt", "s2_pred", "s2_r_vals",
               "_porosity_scatter"}

    def _clean_vm(vm: "VolumeMetrics") -> dict:
        return {k: v for k, v in asdict(vm).items() if k not in _large}

    def _clean_pm(pm: "PatchMetrics") -> dict:
        return {k: v for k, v in asdict(pm).items() if k not in _large}

    def _clean_pm_dict(pm_dict: "dict[str, PatchMetrics]") -> dict:
        return {mode: _clean_pm(pm) for mode, pm in pm_dict.items()}

    doc = {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "step": step,
        "eval_config": eval_cfg.to_dict(),
        "patch_metrics_val":  _clean_pm_dict(patch_metrics_val),
        "patch_metrics_test": _clean_pm_dict(patch_metrics_test),
        "latent_audit": asdict(latent_audit),
        "volume_metrics": [_clean_vm(vm) for vm in vol_metrics],
    }

    path = out_dir / "eval_metrics.json"
    with open(path, "w") as fh:
        json.dump(_serialise(doc), fh, indent=2)
    logger.info("  Saved %s", path)
    return path


def save_latent_audit_json(latent_audit: "LatentAudit", out_dir: Path) -> Path:
    """Save latent audit scalars to ``<out_dir>/latent_audit.json``."""
    from dataclasses import asdict
    path = out_dir / "latent_audit.json"
    with open(path, "w") as fh:
        json.dump(_serialise(asdict(latent_audit)), fh, indent=2)
    logger.info("  Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# README manifest
# ---------------------------------------------------------------------------

def write_readme(
    out_dir: Path,
    *,
    run_name: str,
    checkpoint_path: str,
    eval_cfg: "EvalConfig",
    step: int,
    elapsed_s: float,
    vol_names: list[str],
    repo_root: Path,
) -> Path:
    """Write ``<out_dir>/README.md`` — a comprehensive self-describing manifest.

    The README contains:

    * What was evaluated (run, checkpoint, step, date, elapsed time)
    * How to reproduce the eval run (exact CLI command)
    * Eval configuration summary
    * **File manifest** — every file in the directory with description,
      shape/dtype, and a Python loading snippet
    * Metric glossary — every scalar defined with expected ranges
    * LDM readiness interpretation guide

    Parameters
    ----------
    out_dir : Path
        Eval output directory.
    run_name : str
    checkpoint_path : str
    eval_cfg : EvalConfig
    step : int
        Training step of the checkpoint.
    elapsed_s : float
        Wall-clock evaluation time in seconds.
    vol_names : list of str
        Names of reconstructed volumes (for the manifest section).
    repo_root : Path

    Returns
    -------
    Path
        ``<out_dir>/README.md``
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_min = elapsed_s / 60.0

    # Relative checkpoint path for the reproduction command
    try:
        rel_ckpt = Path(checkpoint_path).relative_to(repo_root)
    except ValueError:
        rel_ckpt = Path(checkpoint_path)

    vol_manifest_lines: list[str] = []
    for vn in vol_names:
        vol_manifest_lines.append(f"\n### `volumes/{vn}/`\n")
        vol_manifest_lines.append(
            f"| File | Format | Description |\n"
            f"|------|--------|-------------|\n"
            f"| `{vn}_arrays.npz` | NPZ (compressed) | All 10 arrays for this volume — see Loading Examples below |\n"
            f"| `{vn}_xct_stoch_mean.tiff` | float32 TIFF | XCT reconstruction — mean of {eval_cfg.n_stochastic_samples} stochastic passes |\n"
            f"| `{vn}_mask_stoch_mean.tiff` | uint8 TIFF | Pore mask — mean of {eval_cfg.n_stochastic_samples} passes, thresholded at 0.5 |\n"
            f"| `{vn}_xct_stoch_single.tiff` | float32 TIFF | XCT reconstruction — single stochastic pass (pass 1 of {eval_cfg.n_stochastic_samples}) |\n"
            f"| `{vn}_mask_stoch_single.tiff` | uint8 TIFF | Pore mask — single stochastic pass |\n"
            f"| `{vn}_xct_mu.tiff` | float32 TIFF | XCT reconstruction — deterministic z = μ (no sampling) |\n"
            f"| `{vn}_mask_mu.tiff` | uint8 TIFF | Pore mask — deterministic z = μ |\n"
            f"| `{vn}_xct_std.tiff` | float32 TIFF | Voxel-wise XCT std across {eval_cfg.n_stochastic_samples} stochastic passes |\n"
            f"| `{vn}_mask_std.tiff` | float32 TIFF | Voxel-wise mask std across {eval_cfg.n_stochastic_samples} passes |\n"
            f"| `{vn}_metrics.json` | JSON | All scalar metrics for this volume |\n"
            f"| `{vn}_slices_*.png` | PNG | Slice comparison grids (3 modes × 3 axes) |\n"
            f"| `{vn}_std_*.png` | PNG | Uncertainty maps (voxel-wise std, plasma colourmap) |\n"
        )
        if eval_cfg.run_s2r:
            vol_manifest_lines.append(f"| `{vn}_s2r.png` | PNG | S₂(r) curves: GT vs stochastic-mean reconstruction |\n")
        if eval_cfg.run_psd:
            vol_manifest_lines.append(f"| `{vn}_psd.png` | PNG | Pore size distribution (equiv. diameter histogram) |\n")
        if eval_cfg.run_pore_gifs:
            vol_manifest_lines.append(
                f"| `{vn}_pore_small.gif` / `_medium.gif` / `_large.gif` | GIF | "
                f"Rotating 3-D views of representative pores |\n"
            )

    vol_manifest = "".join(vol_manifest_lines)

    # Loading snippets
    npz_snippet = textwrap.dedent("""\
        ```python
        import numpy as np
        import tifffile

        # Load all arrays for a volume
        data = np.load("volumes/<vol_name>/<vol_name>_arrays.npz", allow_pickle=True)
        xct_gt           = data["xct_gt"]            # float32, (D, H, W), [0, 1]
        mask_gt          = data["mask_gt"]            # float32, (D, H, W), {0, 1}
        xct_stoch_mean   = data["xct_stoch_mean"]    # float32, [0, 1]
        mask_stoch_mean  = data["mask_stoch_mean"]   # float32, [0, 1]
        xct_stoch_std    = data["xct_stoch_std"]     # float32, stddev map
        mask_stoch_std   = data["mask_stoch_std"]    # float32, stddev map
        xct_stoch_single = data["xct_stoch_single"]  # float32, single stochastic pass
        mask_stoch_single= data["mask_stoch_single"] # float32
        xct_mu           = data["xct_mu"]            # float32, deterministic z=μ
        mask_mu          = data["mask_mu"]           # float32

        # Load TIFF reconstructions
        xct_mean_vol = tifffile.imread("volumes/<vol_name>/<vol_name>_xct_stoch_mean.tiff")

        # Load scalar metrics
        import json
        with open("volumes/<vol_name>/<vol_name>_metrics.json") as f:
            metrics = json.load(f)
        print(metrics["xct_psnr_stoch"], metrics["dice"])
        ```
        """)

    patch_snippet = textwrap.dedent("""\
        ```python
        import numpy as np
        import matplotlib.pyplot as plt

        data = np.load("patch_arrays.npz", allow_pickle=True)
        gt   = data["porosity_gt"]      # float32, (N_patches,)
        pred = data["porosity_pred"]    # float32, (N_patches,)
        vids = data["volume_ids"]       # object array of strings

        fig, ax = plt.subplots()
        ax.scatter(gt, pred, s=2, alpha=0.3)
        ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlabel("φ GT"); ax.set_ylabel("φ Predicted")
        plt.show()
        ```
        """)

    metrics_snippet = textwrap.dedent("""\
        ```python
        import json

        with open("eval_metrics.json") as f:
            d = json.load(f)

        # Patch-level summary — test set, deterministic mode
        pm = d["patch_metrics_test"]["deterministic"]
        print(f"PSNR:  {pm['psnr']['mean']:.2f} ± {pm['psnr']['std']:.2f}")
        print(f"Dice:  {pm['dice']['mean']:.4f} ± {pm['dice']['std']:.4f}")
        print(f"IoU:   {pm['iou']['mean']:.4f}")
        print(f"Por. MAE: {pm['porosity_mae']['mean']:.4f}")
        print(f"Sharpness ratio: {pm['sharpness_ratio']['mean']:.3f}")

        # Per-porosity-bin breakdown
        for bin_name, bin_data in pm['per_bin'].items():
            print(bin_name, "n=", bin_data['n'], "dice=", bin_data['dice']['mean'])

        # All three modes (test set)
        for mode in ("deterministic", "stoch_single", "stoch_mean"):
            pm_m = d["patch_metrics_test"][mode]
            print(f"{mode}: PSNR={pm_m['psnr']['mean']:.2f}  dice={pm_m['dice']['mean']:.4f}")

        # Latent audit
        la = d["latent_audit"]
        print(f"LDM ready: {la['ldm_ready']}")
        print(f"Flagged high: channels {la['channels_flagged_high']}")

        # Volume-level
        for vm in d["volume_metrics"]:
            print(vm["volume_id"], "PSNR stoch:", vm["xct_psnr_stoch"])
        ```
        """)

    ldm_guide = textwrap.dedent("""\
        ## LDM Readiness Guide

        The VAE must be LDM-ready before training a latent diffusion model on top of it.
        Check the following:

        | Check | Target | Where to find it |
        |-------|--------|-----------------|
        | `ldm_ready` | `true` | `eval_metrics.json` → `latent_audit.ldm_ready` |
        | Per-channel σ | all in [0.3, 0.7] | `latent_audit.json` → `sigma_avg` |
        | `recon_std_mean` | < 0.01 | `eval_metrics.json` → `patch_metrics.recon_std_mean` |
        | Active channels | ≥ 80% of total | `latent_audit.n_active_channels / n_total_channels` |
        | PSNR stoch mean | ≥ baseline | compare across experiments |

        **Why σ ∈ [0.3, 0.7]?**  The LDM cosine schedule needs to bridge
        the gap between the VAE posterior (σ_posterior) and N(0,I).
        If σ is too large (> 0.7), the posterior is nearly N(0,I) already and
        the model is under-regularised.  If σ is too small (< 0.3), the posterior
        is very tight and the LDM needs many noisy steps to cover it, degrading
        sample quality.

        **Why recon_std_mean < 0.01?**  A low std across N=50 stochastic
        reconstructions means the decoder is confident — the latent code
        determines the output up to a small noise level.  A high std means
        the posterior is wide and the LDM will struggle to learn a sharp score
        function.
        """)

    content = textwrap.dedent(f"""\
        # PoreGen VAE Evaluation Results

        **Generated:** {ts}
        **Run:** `{run_name}`
        **Checkpoint:** `{rel_ckpt}`  (step {step:,})
        **Eval tier:** `{eval_cfg.tier}`
        **Wall-clock time:** {elapsed_min:.1f} min

        ## How to Reproduce

        ```bash
        train_vae eval {run_name} \\
            --checkpoint {rel_ckpt} \\
            --eval-config eval/{eval_cfg.tier.replace('_', '')}
        ```

        Or from the TUI: `train_vae` → Runs → select `{run_name}` → *Run evaluation*.

        ---

        ## Eval Configuration

        | Field | Value |
        |-------|-------|
        | n_stochastic_samples | {eval_cfg.n_stochastic_samples} |
        | n_volumes | {eval_cfg.n_volumes} |
        | run_tiff_reconstruction | {eval_cfg.run_tiff_reconstruction} |
        | run_s2r | {eval_cfg.run_s2r} |
        | run_psd | {eval_cfg.run_psd} |
        | run_fid | {eval_cfg.run_fid} |
        | run_ripley | {eval_cfg.run_ripley} |
        | run_pore_gifs | {eval_cfg.run_pore_gifs} |
        | run_memorization | {eval_cfg.run_memorization} |
        | stochastic_seed | {eval_cfg.stochastic_seed} |
        | ldm_sigma_low / high | {eval_cfg.ldm_sigma_low} / {eval_cfg.ldm_sigma_high} |

        ---

        ## File Manifest

        ### Root directory

        | File | Description |
        |------|-------------|
        | `README.md` | This file |
        | `eval_metrics.json` | All scalar metrics + eval config (machine-readable) |
        | `latent_audit.json` | Latent-space audit scalars (redundant with eval_metrics.json) |
        | `patch_arrays.npz` | Per-patch porosity arrays for scatter plots |
        | `porosity_scatter.png` | φ_gt vs φ_pred scatter plot, coloured by volume |
        | `latent_audit.png` | Per-channel KL, μ histogram, σ bar chart |
        | `prior_samples.png` | z~N(0,I) → XCT + mask mid-slice grid (8 samples) |

        {vol_manifest}

        ---

        ## Loading Examples

        ### Volume arrays (NPZ)

        {npz_snippet}

        ### Patch porosity arrays (NPZ)

        {patch_snippet}

        ### Scalar metrics (JSON)

        {metrics_snippet}

        ---

        ## Metric Glossary

        ### Patch-level metrics (val and test splits × three decoding modes)

        JSON keys: `patch_metrics_val` / `patch_metrics_test`, each containing
        `deterministic`, `stoch_single`, `stoch_mean` sub-dicts.

        | Metric | Description | Expected range |
        |--------|-------------|----------------|
        | `psnr` | Peak signal-to-noise ratio of XCT reconstruction | > 25 dB = good |
        | `ssim` | Structural similarity (XCT) | > 0.85 = good |
        | `mae` | Mean absolute error (XCT) | < 0.03 = good |
        | `dice` | Dice coefficient (pore mask) | > 0.7 = good |
        | `iou` | Intersection-over-union (pore mask) | > 0.55 = good |
        | `precision` | Mask precision (TP / (TP+FP)) | — |
        | `recall` | Mask recall (TP / (TP+FN)) | — |
        | `f1` | F1 score (= Dice for binary) | > 0.7 = good |
        | `porosity_mae` | Mean absolute porosity error per patch | < 0.005 = good |
        | `porosity_bias` | Signed mean porosity error (pred − gt) | near 0 = good |
        | `sharpness_ratio` | Recon/GT sharpness ratio (porous patches only) | near 1.0 = good |
        | `recon_std_mean` | Mean per-patch std across N stochastic passes | < 0.01 for LDM |
        | `per_bin` | Per-porosity-bin breakdown (4 bins): psnr, dice, porosity_mae | — |

        ### Volume-level metrics (per reconstructed TIFF volume)

        | Metric | Description |
        |--------|-------------|
        | `xct_psnr_stoch` / `_single` / `_mu` | PSNR for each reconstruction mode |
        | `xct_ssim_stoch` / `_single` / `_mu` | SSIM for each mode |
        | `xct_mae_stoch` / `_single` / `_mu` | MAE for each mode |
        | `xct_boundary_mae` | Mean absolute voxel discontinuity at patch seams |
        | `dice` | Dice on full stitched volume (stoch-mean mode) |
        | `porosity_mae` | |φ_gt − φ_pred| on full volume |
        | `pore_count_gt` / `pore_count_pred` | Connected-component pore counts |
        | `s2_wasserstein` | W₁ distance between GT and pred S₂(r) curves |
        | `psd_wasserstein` | W₁ distance between GT and pred pore size distributions |
        | `ripley_w1` | W₁ distance between GT and pred Ripley's K(r) curves |
        | `xct_recon_std_mean` | Mean voxel-wise std across N stochastic passes |

        {ldm_guide}
        """)

    path = out_dir / "README.md"
    path.write_text(content)
    logger.info("  Saved %s", path)
    return path

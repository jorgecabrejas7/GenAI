"""Evaluation configuration dataclass for the PoreGen VAE eval pipeline.

An :class:`EvalConfig` is constructed either programmatically or by reading
the ``eval:`` block from a resolved experiment config.  The CLI and TUI both
go through :func:`load_eval_config`, which resolves a config id like
``eval/r03_base`` into the corresponding YAML file, merges it with any
CLI overrides, and returns a validated ``EvalConfig``.

Metric tiers
------------
Three named tiers group metrics by expected wall-clock cost and use-case:

``required``
    Fast patch-level and latent-space metrics that should be run after every
    training run to check convergence.  No full-volume TIFF reconstruction.

``best_checkpoint``
    Everything in ``required`` plus moderately-expensive structural metrics
    (S2(r), PSD, FID) and full-volume TIFF reconstruction for 2–3 test volumes.
    Intended for the best checkpoint of each experiment.

``paper``
    Everything above plus slow metrics: Ripley's K, 3-D pore GIFs, and
    memorization NN distance.  Run once before writing up final results.

The ``tier`` field is *informational* — it does not gate logic by itself.
Each ``run_*`` flag explicitly controls whether the corresponding computation
runs, so you can mix-and-match tiers freely by overriding individual flags.

LDM readiness gate
------------------
The latent audit reports a per-channel mean σ = exp(0.5 × logvar), averaged
over ``latent_patches`` random test patches.  Channels are flagged if their σ
falls outside ``[ldm_sigma_low, ldm_sigma_high]``.  All channels must be in
range for :attr:`LatentAudit.ldm_ready` to be ``True``.  The target range
[0.3, 0.7] was chosen to match the prior used by the LDM scheduler: σ close
to 1 means the posterior is already near N(0,I), so a simple cosine schedule
can bridge the gap without pathological diffusion steps.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    """Full specification for one evaluation run.

    All fields have sensible defaults so that ``EvalConfig()`` produces a
    "best_checkpoint" tier run with N=50 stochastic samples and 3 test volumes.

    Parameters
    ----------
    tier : str
        Informational label: ``"required"``, ``"best_checkpoint"``, or
        ``"paper"``.  Does not gate logic by itself — use the ``run_*`` flags.
    n_stochastic_samples : int
        Number of forward passes (z = μ + σε) used to compute the stochastic
        mean and std reconstruction per patch / volume.  Default 50.
    n_volumes : int
        Number of test volumes for full-volume TIFF reconstruction.  Volumes
        are chosen to span low / mid / high porosity using the test-split index.
        Set to ``-1`` or ``0`` to evaluate all available test volumes.
    run_tiff_reconstruction : bool
        If True, load raw TIFFs, tile into 64³ patches, stitch back, and run
        all volume-level metrics.  Prerequisite for S2(r), PSD, FID, Ripley,
        pore GIFs, and slice PNGs.
    run_s2r : bool
        Compute S2(r) two-point correlation function and Wasserstein-1 distance
        between GT and reconstructed binary pore masks.
    run_psd : bool
        Compute pore size distribution (equivalent diameter histogram) and
        Wasserstein-1 distance.
    run_fid : bool
        Compute Fréchet Inception Distance on 2-D axial / coronal / sagittal
        mid-slices extracted from GT and reconstructed volumes.
    run_ripley : bool
        Compute Ripley's K(r) on pore centroid point clouds.  Slow (O(N²) in
        the number of pores).  Off by default.
    run_pore_gifs : bool
        Generate rotating 3-D GIFs for representative small / medium / large
        pores.  Slow.  Off by default.
    run_memorization : bool
        Compute latent nearest-neighbour distance (test → train) as a
        memorization proxy.  Slow.  Off by default.
    r_max : int
        Maximum radius in voxels for the S2(r) computation.  Default 50.
    patch_batches : int or None
        Limit patch-metric evaluation to this many DataLoader batches.
        ``None`` (default) evaluates the full test set.
    latent_patches : int
        Number of random test patches used for the latent audit and the
        posterior-tightness ``recon_std_mean`` computation.  Default 512.
    stochastic_seed : int
        Value passed to ``torch.manual_seed`` before each volume's
        reconstruction loop.  Guarantees patch-to-patch reproducibility across
        evaluation runs.
    ldm_sigma_low : float
        Lower bound of the per-channel σ readiness range.  Default 0.3.
    ldm_sigma_high : float
        Upper bound of the per-channel σ readiness range.  Default 0.7.
    batch_size : int
        DataLoader batch size for patch-level evaluation.  Default 32.
    patch_n_stochastic_samples : int
        Number of stochastic passes used for the N-pass patch metric mode.
        Decoupled from ``n_stochastic_samples`` so volume reconstruction can
        use N=50 while patch evaluation (over 200k+ patches) uses a smaller N.
        Default 5.
    """

    tier: str = "best_checkpoint"
    n_stochastic_samples: int = 50
    n_volumes: int = 3
    run_tiff_reconstruction: bool = True
    run_s2r: bool = True
    run_psd: bool = True
    run_fid: bool = True
    run_ripley: bool = False
    run_pore_gifs: bool = False
    run_memorization: bool = False
    r_max: int = 50
    patch_batches: int | None = None
    latent_patches: int = 512
    stochastic_seed: int = 42
    ldm_sigma_low: float = 0.3
    ldm_sigma_high: float = 0.7
    batch_size: int = 32
    patch_n_stochastic_samples: int = 5

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalConfig":
        """Construct from a plain dict, ignoring unknown keys.

        Parameters
        ----------
        d : dict
            Dictionary of field values.  Unrecognised keys are silently ignored
            so that YAML files can carry extra comments or future fields without
            breaking older code.

        Returns
        -------
        EvalConfig
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON / YAML output."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def any_volume_metric(self) -> bool:
        """True if at least one volume-level metric is enabled."""
        return any([
            self.run_tiff_reconstruction,
            self.run_s2r,
            self.run_psd,
            self.run_fid,
            self.run_ripley,
            self.run_pore_gifs,
        ])


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eval_config(
    eval_ref: str | Path | None,
    *,
    repo_root: Path,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    """Resolve an eval config reference and return a validated :class:`EvalConfig`.

    Resolution order
    ----------------
    1. If *eval_ref* is an absolute path, load it directly.
    2. If *eval_ref* looks like ``eval/r03_base`` (no extension), resolve
       relative to ``<repo_root>/configs/``.
    3. If *eval_ref* is ``None``, fall back to
       ``<repo_root>/configs/eval/full.yaml``.
    4. After loading the YAML, apply *overrides* (shallow merge on the ``eval``
       sub-dict) to allow CLI arguments to override individual fields.

    Parameters
    ----------
    eval_ref : str, Path, or None
        Config id (e.g. ``"eval/r03_base"``), explicit YAML path, or ``None``
        to use the built-in default.
    repo_root : Path
        Repository root used for relative path resolution.
    overrides : dict, optional
        Extra key-value pairs that override fields from the YAML after loading.

    Returns
    -------
    EvalConfig

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist.
    """
    configs_root = repo_root / "configs"

    if eval_ref is None:
        yaml_path = configs_root / "eval" / "full.yaml"
    else:
        candidate = Path(eval_ref)
        if candidate.is_absolute():
            yaml_path = candidate
        else:
            # Try plain path, then add .yaml, then look under configs/
            search = [
                repo_root / candidate,
                (repo_root / candidate).with_suffix(".yaml"),
                configs_root / candidate,
                (configs_root / candidate).with_suffix(".yaml"),
            ]
            yaml_path = next((p for p in search if p.exists()), None)
            if yaml_path is None:
                raise FileNotFoundError(
                    f"Could not resolve eval config reference '{eval_ref}'. "
                    f"Searched: {[str(p) for p in search]}"
                )

    if not yaml_path.exists():
        raise FileNotFoundError(f"Eval config not found: {yaml_path}")

    with yaml_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    # The YAML may be written with or without an outer "eval:" key.
    if "eval" in raw:
        data: dict[str, Any] = dict(raw["eval"])
    else:
        data = dict(raw)

    if overrides:
        data.update(overrides)

    return EvalConfig.from_dict(data)


def list_eval_configs(repo_root: Path) -> list[dict[str, Any]]:
    """Return lightweight metadata for every YAML in ``configs/eval/``.

    Each entry contains the keys ``id``, ``path``, and ``tier``.  Used by the
    TUI to populate the eval-config picker list.

    Parameters
    ----------
    repo_root : Path

    Returns
    -------
    list of dict
    """
    eval_dir = repo_root / "configs" / "eval"
    if not eval_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for yaml_path in sorted(eval_dir.rglob("*.yaml")):
        try:
            with yaml_path.open() as fh:
                raw = yaml.safe_load(fh) or {}
            data = raw.get("eval", raw)
            tier = data.get("tier", "unknown")
        except Exception:
            tier = "unknown"

        rel = yaml_path.relative_to(repo_root / "configs")
        config_id = str(rel.with_suffix("")).replace("\\", "/")
        results.append({"id": config_id, "path": str(yaml_path), "tier": tier})

    return results

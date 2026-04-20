"""PoreGen VAE evaluation package.

Public interface
----------------
:class:`~poregen.eval.config.EvalConfig`
    Configuration dataclass for a full evaluation run.

:func:`~poregen.eval.config.load_eval_config`
    Resolve an eval-config reference (YAML path or ``"eval/r03_base"`` style
    id) and return a validated :class:`EvalConfig`.

:func:`~poregen.eval.runner.run_eval`
    Orchestrate a complete evaluation run: load checkpoint, reconstruct
    volumes, compute all metrics, save outputs, return the output directory.

:class:`~poregen.eval.stochastic.VolumeReconstruction`
    All three reconstruction modes (stoch_mean, stoch_single, mu) plus GT
    arrays for one full volume.

:func:`~poregen.eval.stochastic.reconstruct_volume_three_modes`
    Tile a full volume into 64³ patches, run encode_patch_three_modes on
    each, and stitch the results back.

:class:`~poregen.eval.metrics.PatchMetrics`
:class:`~poregen.eval.metrics.VolumeMetrics`
:class:`~poregen.eval.metrics.LatentAudit`
    Metric result dataclasses.
"""

from poregen.eval.config import EvalConfig, load_eval_config, list_eval_configs
from poregen.eval.stochastic import VolumeReconstruction, reconstruct_volume_three_modes
from poregen.eval.metrics import (
    PatchMetrics,
    VolumeMetrics,
    LatentAudit,
    eval_patches,
    eval_volume_metrics,
    latent_audit,
    compute_fid_slices,
    memorization_score,
    select_test_volumes,
)

__all__ = [
    # config
    "EvalConfig",
    "load_eval_config",
    "list_eval_configs",
    # stochastic
    "VolumeReconstruction",
    "reconstruct_volume_three_modes",
    # metrics
    "PatchMetrics",
    "VolumeMetrics",
    "LatentAudit",
    "eval_patches",
    "eval_volume_metrics",
    "latent_audit",
    "compute_fid_slices",
    "memorization_score",
    "select_test_volumes",
]

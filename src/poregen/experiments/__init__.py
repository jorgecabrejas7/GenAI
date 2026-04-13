"""Experiment runtimes for PoreGen VAE experiments.

Each experiment subclasses :class:`~poregen.experiments.base.ExperimentRuntime`
and exposes a single import surface for its analysis notebooks.

Current experiments
-------------------
- ``poregen.experiments.r03`` — R03 auxiliary-decoder + latent-space analysis
"""

from poregen.experiments.base import ExperimentRuntime, build_patch_loader, find_repo_root

__all__ = ["ExperimentRuntime", "build_patch_loader", "find_repo_root"]

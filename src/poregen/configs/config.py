"""Configuration loading utilities for PoreGen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Load a YAML config file and apply optional dot-notation overrides.

    Parameters
    ----------
    path : str | Path
        Path to a YAML config file (e.g. ``vae_default.yaml``).
    **overrides : Any
        Dot-notation key=value pairs to override after loading.
        Example::

            cfg = load_config("vae_default.yaml",
                              **{"loss.kl_max_beta": 0.01,
                                 "training.total_steps": 50000})

    Returns
    -------
    dict
        Nested config dict. Top-level keys are the YAML sections
        (``model``, ``loss``, ``training``, ``data``).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If an override key refers to a section or field that does not
        exist in the loaded config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        if len(parts) != 2:
            raise KeyError(
                f"Override key '{dotkey}' must be in 'section.field' format."
            )
        section, field = parts
        if section not in cfg:
            raise KeyError(f"Config section '{section}' not found.")
        if field not in cfg[section]:
            raise KeyError(
                f"Field '{field}' not found in config section '{section}'."
            )
        cfg[section][field] = value

    return cfg

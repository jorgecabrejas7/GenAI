"""Simple name → class registry for VAE architectures."""

from __future__ import annotations

from typing import Type

import torch.nn as nn

from poregen.models.vae.base import VAEConfig

_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_vae(name: str):
    """Decorator that registers a VAE class under *name*."""

    def _wrap(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"VAE '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build_vae(name: str, **overrides) -> nn.Module:
    """Instantiate a registered VAE.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"conv"``, ``"unet"``).
    **overrides
        Keyword arguments forwarded to :class:`VAEConfig`.

    Returns
    -------
    nn.Module
        The constructed model.

    Raises
    ------
    KeyError
        If *name* is not registered.
    TypeError
        If any override key is invalid for :class:`VAEConfig`.
    """
    if name not in _REGISTRY:
        avail = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown VAE '{name}'. Available: {avail}")

    # Validate overrides against VAEConfig fields.
    valid_fields = {f.name for f in VAEConfig.__dataclass_fields__.values()}
    bad = set(overrides) - valid_fields
    if bad:
        raise TypeError(
            f"Invalid VAEConfig fields: {bad}. Valid: {sorted(valid_fields)}"
        )

    cfg = VAEConfig(**overrides)
    return _REGISTRY[name](cfg)


def list_vaes() -> list[str]:
    """Return names of all registered VAE architectures."""
    return sorted(_REGISTRY)

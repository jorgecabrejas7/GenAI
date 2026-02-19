"""Backward-compatible re-export — use ``poregen.models.vae`` directly.

This module exists so that ``from poregen.vae import ...`` keeps working.
"""

from poregen.models.vae import VAEConfig, VAEOutput, build_vae, register_vae

__all__ = ["VAEConfig", "VAEOutput", "build_vae", "register_vae"]

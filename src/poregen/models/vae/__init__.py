"""VAE framework for PoreGen — shared types, registry, and all architecture generations.

Architecture families
---------------------
* ``poregen.models.vae.v1`` — first-gen (GroupNorm, SiLU, ConvTranspose3d).
  Registered as ``"conv"``, ``"conv_noattn"``, ``"unet"``.
* ``poregen.models.vae.v2`` — second-gen (BatchNorm3d, GELU, Upsample+Conv).
  Registered as ``"v2.conv"``, ``"v2.conv_noattn"``,
  ``"v2.conv_noattn_dualbranch"``, ``"v2.unet"``.

Usage
-----
>>> from poregen.models.vae import build_vae, VAEConfig, VAEOutput
>>> model = build_vae("conv_noattn", z_channels=8, base_channels=32)
>>> model_v2 = build_vae("v2.conv_noattn", z_channels=8, base_channels=32)
"""

from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import build_vae, list_vaes, register_vae

# Trigger registration of all built-in architectures.
import poregen.models.vae.v1  # noqa: F401  registers "conv", "conv_noattn", "unet"
import poregen.models.vae.v2  # noqa: F401  registers "v2.conv", "v2.conv_noattn", "v2.unet"

__all__ = ["VAEConfig", "VAEOutput", "build_vae", "list_vaes", "register_vae"]

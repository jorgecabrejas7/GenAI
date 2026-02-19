"""VAE model architectures for 3-D XCT + mask generation."""

from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import build_vae, register_vae

# Trigger registration of built-in architectures.
import poregen.models.vae.conv_vae as _conv  # noqa: F401
import poregen.models.vae.unet_vae as _unet  # noqa: F401

__all__ = ["VAEConfig", "VAEOutput", "build_vae", "register_vae"]

"""Model architectures for PoreGen (VAE, future diffusion)."""

from poregen.models.vae.registry import build_vae

__all__ = ["build_vae"]

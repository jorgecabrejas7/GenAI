"""The porosity-only latent diffusion baseline (Naiff et al. 2026).

A published recipe — scalar-porosity-conditioned latent diffusion for porous
media — reimplemented on OUR data, OUR compressor and OUR store, so that the
difference between it and ldm06 is the CONDITIONING and not the dataset, the
autoencoder or the schedule.
"""

from poregen.baselines.ldm_phi_only.networks import PhiOnlyDenoiser

__all__ = ["PhiOnlyDenoiser"]

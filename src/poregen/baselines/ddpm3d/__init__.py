"""A 3-D pixel-space DDPM baseline: no latent space, no conditioning.

The second comparator. Where SliceGAN asks "can 2-D sections teach a 3-D
structure", this one asks the question the whole latent design is an answer to:
**what does diffusion directly in voxel space cost, and what does it buy?**

ldm06 diffuses in a VAE latent at f=4, which is what makes a 1024-cubed volume
affordable. A pixel-space DDPM at the same patch size has 64 times the elements
per step and no compression, so it is trained at 64^3 and sampled at 64^3 —
and that limit is itself the result. It cannot produce the 1024-wide volumes
every scale-dependent table in this project is built on.

UNCONDITIONAL, like SliceGAN and for the same reason: adding a conditioning
path would make it a different method. It is scored only on the request-free
metrics, at the one shape it can make.

WHAT IS SHARED WITH ldm06, SO THE COMPARISON IS ABOUT THE SPACE AND NOT THE
SCHEDULE: the cosine schedule, the v objective and the zero-terminal-SNR
correction all come from `poregen.diffusion.noise_schedule`, the same module
ldm06 trains against. Only the data the denoiser sees differs — voxels here,
latents there.
"""

from poregen.baselines.ddpm3d.networks import UNet3DPixel

__all__ = ["UNet3DPixel"]

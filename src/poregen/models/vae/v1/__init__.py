"""First-generation VAE architectures (GroupNorm, SiLU, ConvTranspose3d decoder).

Importing this package registers all v1 architectures in the global VAE registry:

* ``"conv"``        — :class:`~poregen.models.vae.v1.conv.ConvVAE3D`
* ``"conv_noattn"`` — :class:`~poregen.models.vae.v1.conv_noattn.ConvVAE3DNoAttn`
* ``"unet"``        — :class:`~poregen.models.vae.v1.unet.UNetVAE3D`

Checkpoints trained with these architectures remain fully compatible after
the refactor — all ``nn.Module`` attribute names and parameter tensor names
are preserved.
"""

import poregen.models.vae.v1.conv          # noqa: F401  registers "conv"
import poregen.models.vae.v1.conv_noattn   # noqa: F401  registers "conv_noattn"
import poregen.models.vae.v1.unet          # noqa: F401  registers "unet"

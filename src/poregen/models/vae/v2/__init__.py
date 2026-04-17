"""Second-generation VAE architectures (BatchNorm3d, GELU, Upsample+Conv decoder).

Importing this package registers all v2 architectures in the global VAE registry
under the ``v2.*`` namespace:

* ``"v2.conv"``                  — :class:`~poregen.models.vae.v2.conv.ConvVAE3DV2`
* ``"v2.conv_noattn"``           — :class:`~poregen.models.vae.v2.conv_noattn.ConvVAE3DNoAttnV2`
* ``"v2.conv_noattn_dualbranch"``— :class:`~poregen.models.vae.v2.conv_noattn_dualbranch.ConvVAE3DNoAttnDualBranchV2`
* ``"v2.unet"``                  — :class:`~poregen.models.vae.v2.unet.UNetVAE3DV2`
"""

import poregen.models.vae.v2.conv                    # noqa: F401  registers "v2.conv"
import poregen.models.vae.v2.conv_noattn             # noqa: F401  registers "v2.conv_noattn"
import poregen.models.vae.v2.conv_noattn_dualbranch  # noqa: F401  registers "v2.conv_noattn_dualbranch"
import poregen.models.vae.v2.unet                    # noqa: F401  registers "v2.unet"

"""Second-generation VAE architectures (BatchNorm3d, GELU, Upsample+Conv decoder).

Importing this package registers all v2 architectures in the global VAE registry
under the ``v2.*`` namespace:

* ``"v2.conv"``                  — :class:`~poregen.models.vae.v2.conv.ConvVAE3DV2`
* ``"v2.conv_noattn"``           — :class:`~poregen.models.vae.v2.conv_noattn.ConvVAE3DNoAttnV2`
* ``"v2.conv_noattn_dualbranch"``— :class:`~poregen.models.vae.v2.conv_noattn_dualbranch.ConvVAE3DNoAttnDualBranchV2`
* ``"v2.conv_noattn_dualbranch_cls"`` — :class:`~poregen.models.vae.v2.conv_noattn_dualbranch_cls.ConvVAE3DNoAttnDualBranchClsV2`
  (r08: 3-channel encoder input, 3-class decoder head, no binary mask head)
* ``"v2.unet"``                  — :class:`~poregen.models.vae.v2.unet.UNetVAE3DV2`
* ``"v2.vrrae"``                 — :class:`~poregen.models.vae.v2.vrrae.ConvVAE3DVRRAEV2`
* ``"v2.vrrae_linear"``          — :class:`~poregen.models.vae.v2.vrrae_linear.ConvVAE3DVRRAELinearV2`
"""

import poregen.models.vae.v2.conv                    # noqa: F401  registers "v2.conv"
import poregen.models.vae.v2.conv_noattn             # noqa: F401  registers "v2.conv_noattn"
import poregen.models.vae.v2.conv_noattn_dualbranch  # noqa: F401  registers "v2.conv_noattn_dualbranch"
import poregen.models.vae.v2.conv_noattn_dualbranch_cls  # noqa: F401  registers "v2.conv_noattn_dualbranch_cls"
import poregen.models.vae.v2.unet                    # noqa: F401  registers "v2.unet"
import poregen.models.vae.v2.vrrae                   # noqa: F401  registers "v2.vrrae"
import poregen.models.vae.v2.vrrae_linear            # noqa: F401  registers "v2.vrrae_linear"

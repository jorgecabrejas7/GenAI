"""The redecode must reproduce the volume production actually wrote.

THE BUG THIS EXISTS FOR. `latents.npy` holds the canvas in the LDM's NORMALISED
space; the decoder expects native latents. `decoder_ft_redecode.py` decoded the
saved array directly, feeding the decoder latents about three times too wide,
and every generated-arm number of the D43 gate was computed that way — sharpness
ratios, S2, PSD, and the agreement between decoders. The val arm was unaffected
because it decodes `mu` straight from the frozen encoder, which is native scale.

Nothing failed and nothing warned. The numbers were plausible: a sharpness ratio
of 1.38, pore agreement of 0.95. They were plausible and wrong, and they were
reported before anyone compared them with the volumes sitting next to the
latents on disk.

So the test is that comparison: decode a saved latent canvas through the
redecode's own path and check it against the stored `volume.tif`. The tolerance
is set by tiled-versus-overlapped decode, not by scale — a wrong scale misses by
an order of magnitude more.

Skipped when campaign 18 is not on this machine, because it is a test about real
artefacts and a synthetic stand-in would not have caught the bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

CASE = (REPO / "runs" / "campaigns" / "18-eval-v4-final"
        / "sampler" / "volumes" / "192_ddim50_seed101")
BASELINE = next(
    (REPO / "runs" / "vae").glob("r08-run-0004-*/best.ckpt"), None)

#: Mean absolute u8 difference between one tiled window and the production
#: overlapped Tukey blend. Measured at 2.06; 6 leaves room for a different
#: window position without admitting a scale error, which measured 16.92.
MAE_TOLERANCE = 6.0
LATENT_WIN = 16
VOX_PER_CELL = 4

pytestmark = pytest.mark.skipif(
    not (CASE / "latents.npy").exists() or BASELINE is None,
    reason="campaign 18 latents or the r08 baseline checkpoint are not on this machine",
)


def _decode_one_window(z_native: np.ndarray, vae, origin: int) -> np.ndarray:
    import torch

    w = z_native[:, origin:origin + LATENT_WIN,
                 origin:origin + LATENT_WIN,
                 origin:origin + LATENT_WIN]
    with torch.no_grad():
        dec = vae.decoder(torch.from_numpy(w).unsqueeze(0))
        xct = vae.xct_head(dec).float().clamp(0.0, 1.0)
    return (xct.squeeze().numpy() * 255.0).astype(np.uint8)


@pytest.fixture(scope="module")
def decoded():
    import tifffile
    import torch

    from decoder_ft_redecode import latent_normalisation
    from poregen.experiments.train_vae import load_vae_from_checkpoint

    vae, _, _, _ = load_vae_from_checkpoint(BASELINE, torch.device("cpu"))
    vae.eval()
    z = np.load(CASE / "latents.npy").astype(np.float32)
    mean, std = latent_normalisation(CASE)
    z_native = z * std[:, None, None, None] + mean[:, None, None, None]
    origin = z.shape[1] // 2 - LATENT_WIN // 2
    stored = tifffile.imread(CASE / "volume.tif")
    v0 = origin * VOX_PER_CELL
    n = LATENT_WIN * VOX_PER_CELL
    ref = stored[v0:v0 + n, v0:v0 + n, v0:v0 + n]
    return {
        "native": _decode_one_window(z_native, vae, origin),
        "as_saved": _decode_one_window(z, vae, origin),
        "stored": ref,
    }


def _mae(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def test_the_native_scale_decode_matches_what_production_wrote(decoded):
    mae = _mae(decoded["native"], decoded["stored"])
    assert mae < MAE_TOLERANCE, (
        f"decoding the saved latents at native scale differs from the stored "
        f"volume by MAE {mae:.2f}, over the {MAE_TOLERANCE} allowed for "
        f"tiled-versus-overlapped decode. Either the normalisation is wrong "
        f"again or the decode path has drifted from production.")


def test_decoding_the_saved_array_directly_is_clearly_wrong(decoded):
    """The guard on the guard.

    If this ever passes, the tolerance above has stopped discriminating and the
    first test would no longer catch the bug it was written for.
    """
    wrong = _mae(decoded["as_saved"], decoded["stored"])
    right = _mae(decoded["native"], decoded["stored"])
    assert wrong > MAE_TOLERANCE, (
        f"decoding normalised latents directly scored MAE {wrong:.2f}, inside "
        f"the tolerance. The test can no longer tell a scale error from a "
        f"decode-geometry difference.")
    assert wrong > 3 * right, (
        f"the wrong scale ({wrong:.2f}) is not clearly worse than the right one "
        f"({right:.2f}); this test has lost its discriminating power.")


def test_the_grey_statistics_match_too(decoded):
    """MAE alone can be small while the distribution is displaced."""
    native, stored = decoded["native"], decoded["stored"]
    assert abs(float(native.mean()) - float(stored.mean())) < 3.0
    assert abs(float(native.std()) - float(stored.std())) < 3.0

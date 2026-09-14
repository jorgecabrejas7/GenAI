"""Decode campaign-18 latents with BOTH decoders and write the volumes. (CPU.)

The D43 gate stage saved only `results.json`, so the two decoders could be
compared numerically and not looked at. This writes the volumes, in the eval_v4
case layout, so a viewer can put them side by side.

    <out>/volumes/<assessment>__<case>/
        baseline/{volume.tif,label.tif,manifest.json}
        finetuned/{volume.tif,label.tif,manifest.json}
    <out>/volumes/INDEX.json

`volume.tif` is uint8 and `label.tif` is the 3-class map, the same conventions
every eval_v4 case uses. Each `manifest.json` is the SOURCE case's manifest plus
`decoder` and `decoder_checkpoint`, so a volume here can always be traced to the
case it came from and the weights that made it.

THE DECODE IS THE PRODUCTION ONE: overlapped windows at `decode_stride`, a
Tukey taper, and the 3-class LOGITS blended before the argmax — not the tiled
decode the gate script used. Tiling leaves a class seam at every patch face,
which is exactly what someone comparing two decoders by eye would notice and
misattribute.

AND THE LATENTS ARE RETURNED TO NATIVE SCALE FIRST. `latents.npy` is saved in
the LDM's normalised space. Decoding it directly is a wrong-scale decode — MAE
16.92 against the stored volume rather than 2.06 — and it is the bug that
invalidated the gate's generated arm. `tests/test_redecode_scale.py` guards it.

CPU by design: the GPU belongs to the queue. A 192-cubed case is a few minutes.

Usage:
    python scripts/analysis/decoder_ft_write_volumes.py            # the 13 inspection cases
    python scripts/analysis/decoder_ft_write_volumes.py --max-voxels 2e8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))

logger = logging.getLogger("write_volumes")

C18 = REPO / "runs" / "campaigns" / "18-eval-v4-final"
OUT = REPO / "runs" / "campaigns" / "11-decoder-ft"
BASE_GLOB = "r08-run-0004-*/best.ckpt"
FT_GLOB = "r08-run-0007-*/best.ckpt"
LATENT_WIN = 16
VOX_PER_CELL = 4
DECODE_STRIDE_VOX = 32          # production
DECODE_BATCH = 8                # small: this is a CPU decode


def _checkpoint(glob: str) -> Path:
    hits = sorted((REPO / "runs" / "vae").glob(glob))
    if not hits:
        raise SystemExit(f"no checkpoint matches {glob}")
    return hits[-1]


def tukey_3d(n: int, floor: float = 0.1) -> np.ndarray:
    from poregen.diffusion.sampler import tukey_window_3d
    return np.asarray(tukey_window_3d(n, floor=floor), np.float32)


def decode_canvas(vae, z_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Production overlapped decode: Tukey taper, LOGITS blended, then argmax."""
    C, Z, Y, X = z_native.shape
    shape = (Z * VOX_PER_CELL, Y * VOX_PER_CELL, X * VOX_PER_CELL)
    s_cells = DECODE_STRIDE_VOX // VOX_PER_CELL
    patch = LATENT_WIN * VOX_PER_CELL

    def starts(n: int) -> list[int]:
        if n <= LATENT_WIN:
            return [0]
        out = list(range(0, n - LATENT_WIN + 1, s_cells))
        if out[-1] != n - LATENT_WIN:
            out.append(n - LATENT_WIN)
        return out

    origins = [(z, y, x) for z in starts(Z) for y in starts(Y) for x in starts(X)]
    w3 = tukey_3d(patch)
    xct_acc = np.zeros(shape, np.float32)
    log_acc = np.zeros((3, *shape), np.float32)
    w_acc = np.zeros(shape, np.float32)

    for i in range(0, len(origins), DECODE_BATCH):
        chunk = origins[i:i + DECODE_BATCH]
        zb = torch.from_numpy(np.stack([
            z_native[:, z:z + LATENT_WIN, y:y + LATENT_WIN, x:x + LATENT_WIN]
            for z, y, x in chunk]))
        with torch.no_grad():
            dec = vae.decoder(zb)
            xct = vae.xct_head(dec).float().clamp(0.0, 1.0).squeeze(1).numpy()
            logits = vae.class_head(dec).float().numpy()
        for j, (z, y, x) in enumerate(chunk):
            z0, y0, x0 = z * VOX_PER_CELL, y * VOX_PER_CELL, x * VOX_PER_CELL
            sl = np.s_[z0:z0 + patch, y0:y0 + patch, x0:x0 + patch]
            xct_acc[sl] += xct[j] * w3
            log_acc[(slice(None), *sl)] += logits[j] * w3
            w_acc[sl] += w3
    w = np.maximum(w_acc, 1e-8)
    grey = np.clip(xct_acc / w, 0.0, 1.0)
    label = np.argmax(log_acc / w, axis=0).astype(np.uint8)
    return (grey * 255.0).round().astype(np.uint8), label


def inspection_cases() -> list[tuple[str, str]]:
    mf = C18 / "inspection" / "inspection_manifest.json"
    d = json.loads(mf.read_text())
    rows = d.get("cases", d) if isinstance(d, dict) else d
    return [(r["assessment"], r["case"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, default=C18)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-voxels", type=float, default=6e7,
                    help="skip cases larger than this, naming them in INDEX.json. "
                         "A CPU decode of a 192-cubed case is minutes; a "
                         "1024-wide one is hours, and the queue needs the card.")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    import tifffile

    from decoder_ft_redecode import latent_normalisation
    from poregen.experiments.train_vae import load_vae_from_checkpoint

    dev = torch.device("cpu")
    base_ck, ft_ck = _checkpoint(BASE_GLOB), _checkpoint(FT_GLOB)
    logger.info("baseline  %s", base_ck)
    logger.info("finetuned %s", ft_ck)
    base, _, _, _ = load_vae_from_checkpoint(base_ck, dev); base.eval()
    ft, _, _, _ = load_vae_from_checkpoint(ft_ck, dev); ft.eval()

    vol_root = args.out / "volumes"
    vol_root.mkdir(parents=True, exist_ok=True)
    index, skipped = [], []
    for assessment, case in inspection_cases():
        if args.only and case not in args.only:
            continue
        src = args.campaign / assessment / "volumes" / case
        lat = src / "latents.npy"
        if not lat.exists():
            skipped.append({"case": f"{assessment}/{case}", "why": "no latents.npy"})
            continue
        z = np.load(lat).astype(np.float32)
        n_vox = int(np.prod(z.shape[1:])) * VOX_PER_CELL ** 3
        if n_vox > args.max_voxels:
            skipped.append({"case": f"{assessment}/{case}",
                            "why": f"{n_vox/1e6:.0f} Mvox over the "
                                   f"{args.max_voxels/1e6:.0f} Mvox CPU limit"})
            logger.info("skip %s/%s (%.0f Mvox)", assessment, case, n_vox / 1e6)
            continue
        mean, std = latent_normalisation(src)
        z_native = z * std[:, None, None, None] + mean[:, None, None, None]
        src_manifest = json.loads((src / "manifest.json").read_text())
        entry = {"assessment": assessment, "case": case,
                 "source": str(src), "volume_shape": None, "paths": {}}
        for name, model, ck in (("baseline", base, base_ck), ("finetuned", ft, ft_ck)):
            t = time.time()
            grey, label = decode_canvas(model, z_native)
            d = vol_root / f"{assessment}__{case}" / name
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / "volume.tif"), grey)
            tifffile.imwrite(str(d / "label.tif"), label)
            (d / "manifest.json").write_text(json.dumps(
                {**src_manifest, "decoder": name,
                 "decoder_checkpoint": str(ck),
                 "decoded_by": "scripts/analysis/decoder_ft_write_volumes.py",
                 "decode": "overlapped", "decode_stride_vox": DECODE_STRIDE_VOX},
                indent=2) + "\n")
            entry["volume_shape"] = list(grey.shape)
            entry["paths"][name] = str(d)
            logger.info("%s/%s %s: %s in %.1f s  (phi %.4f)",
                        assessment, case, name, grey.shape, time.time() - t,
                        float((label == 1).mean()))
        index.append(entry)

    (vol_root / "INDEX.json").write_text(json.dumps({
        "campaign": str(args.campaign),
        "baseline_checkpoint": str(base_ck),
        "finetuned_checkpoint": str(ft_ck),
        "decode": "production overlapped Tukey, logits blended before argmax",
        "latents": "returned to native scale with the store's per-channel "
                   "mean/std before decoding — see tests/test_redecode_scale.py",
        "layout": "<assessment>__<case>/{baseline,finetuned}/{volume.tif,label.tif,manifest.json}",
        "n_cases": len(index), "cases": index, "skipped": skipped,
    }, indent=2) + "\n")
    logger.info("wrote %d case pairs -> %s", len(index), vol_root)
    if skipped:
        logger.info("%d skipped (named in INDEX.json)", len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

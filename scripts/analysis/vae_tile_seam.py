"""0a — do the mask seams come from the VAE decoder, or from the LDM?

Generated volumes carry a strong mask seam (``seam_mask_ratio`` ~1.8-2.1 for
the joint_oob arm in ``runs/campaigns/05-eval-v3-fixed-decode``).  Two things
could produce it: the tiled ASSEMBLY (each 64^3 block is decoded independently
and written in with no blending) or the LDM itself (neighbouring latents that
do not agree).  This script removes the LDM from the picture entirely: it
encodes REAL volume regions with the production r07 z=4 VAE and re-assembles
them the two ways.

    A  tiled      encode at stride 64, decode, write each block in place.
                  Exactly what ``VolumeGenerator`` does after sampling.
    B  overlapped encode at stride 32, decode, blend the XCT output and the
                  mask LOGITS with the Tukey window of ``poregen.eval.blended``
                  (logits blended before the sigmoid, as that module does).
    C  control    the real volume itself — the floor of the metric.

Both A and B use the posterior mean ``mu`` (no sampling), so the ONLY thing
that differs between them is the assembly.  Whatever seam A shows and B does
not is the VAE's tiling, not the LDM.

Region choice: a 192 x 512 x 512 box, 64-aligned in y/x, entirely inside
``sample_mask`` — which also excludes the three drilled registration holes,
since they are False there.  See ``scripts/analysis/_real_windows.py``.
Assembly B reads a 64-voxel margin of REAL data around the box, so the Tukey
taper is fed real neighbours instead of a reflection, and the box is cropped
out afterwards.

Outputs -> ``runs/campaigns/08-pre-ldm06-diagnostics/vae_tile_seam/``:
results.json, findings.md, per-assembly volume.tif / mask.tif.

Usage:
    python scripts/analysis/vae_tile_seam.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, ZARR_ROOT, write_findings, write_json  # noqa: E402
from _real_windows import cell_ok_by_slice, find_region  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.diffusion.sampler import seam_discontinuity  # noqa: E402
from poregen.eval.blended import _tukey_window_3d  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402

OUT_DIR = REPO / "runs/campaigns/08-pre-ldm06-diagnostics/vae_tile_seam"
LATENTS_ROOT = REPO / "data/split_v2/latents_r07z4"
ORIENT_FIELD = REPO / "data/split_v2/orientation_field.json"

VOLUMES = {
    "Pegaso_1_26": "MedidasDB__Airbus_Panel_Pegaso_probetas_1_26_volumen_eq_aligned",
    "Na_02_2": "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_02_2_volume_eq_aligned",
    "Na_08_4": "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_08_4_volume_eq_aligned",
}

PATCH = 64
REGION_YX = 512                     # in-plane box side — 8 x 8 tile cells
# Box depth, tried in order.  The Nacho laminates are only ~185 voxels of
# continuous material, so no 192-deep all-material box exists in them; those
# volumes fall back to 128.  The seam metric is a ratio of mean |slice
# difference| at seam planes to the same at interior planes, so it does not
# depend on how many planes there are.
DEPTHS = (192, 128)
MARGIN = 64                         # real-data halo for the overlapped assembly
STRIDE_A = 64
STRIDE_B = 32
BATCH = 32
MASK_LOGIT_CLIP = 10.0              # control-arm mask logit scale
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _encode_mu(model, xct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Posterior mean of the dual-branch encoder.

    r07 was trained with ``in_channels=2``: the encoder sees ``cat([xct, mask])``,
    exactly as ``scripts/build_latent_dataset.py`` feeds it when it builds the
    latents the LDM is trained on.  ``poregen.eval.blended._encode`` assumes the
    one-channel R03 design, so it cannot be reused here.
    """
    enc_in = xct if model.cfg.in_channels == 1 else torch.cat([xct, mask], dim=1)
    h_a = model.encoder_a(enc_in)
    h_b = model.encoder_b(enc_in)
    return model.to_mu(model.fusion(torch.cat([h_a, h_b], dim=1)))


@torch.no_grad()
def decode_patches(model, xct_pad: np.ndarray, mask_pad: np.ndarray,
                   coords: list[tuple[int, int, int]],
                   device: torch.device, batch: int = BATCH):
    """Encode -> mu -> decode each patch. Yields (coords, xct_out, mask_logit).

    ``xct_out`` is the raw XCT-head output (grey level in [0, 1] before the
    clamp) and ``mask_logit`` the raw mask logit — both pre-activation, which
    is what assembly B has to blend.  bfloat16 autocast matches the sampler's
    decode path, so the numbers are comparable with the generated volumes.
    """
    for b0 in range(0, len(coords), batch):
        bc = coords[b0:b0 + batch]
        xa = np.stack([xct_pad[z:z + PATCH, y:y + PATCH, x:x + PATCH] for z, y, x in bc])
        ma = np.stack([mask_pad[z:z + PATCH, y:y + PATCH, x:x + PATCH] for z, y, x in bc])
        xt = torch.from_numpy(xa).unsqueeze(1).to(device)
        mt = torch.from_numpy(ma).unsqueeze(1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            dec = model.decoder(_encode_mu(model, xt, mt))
            xct_out = model.xct_head(dec)
            mask_logit = model.mask_head(dec)
        yield (bc,
               xct_out.squeeze(1).float().cpu().numpy(),
               mask_logit.squeeze(1).float().cpu().numpy())


def assemble_tiled(model, core: np.ndarray, core_mask: np.ndarray,
                   device) -> tuple[np.ndarray, np.ndarray, int]:
    """Assembly A — stride-64 tiling, each decoded block written in place."""
    D, H, W = core.shape
    coords = [(z, y, x)
              for z in range(0, D - PATCH + 1, STRIDE_A)
              for y in range(0, H - PATCH + 1, STRIDE_A)
              for x in range(0, W - PATCH + 1, STRIDE_A)]
    xct = np.zeros(core.shape, np.float32)
    mlog = np.zeros(core.shape, np.float32)
    for bc, xo, mo in decode_patches(model, core, core_mask, coords, device):
        for i, (z, y, x) in enumerate(bc):
            sl = np.s_[z:z + PATCH, y:y + PATCH, x:x + PATCH]
            xct[sl] = xo[i]
            mlog[sl] = mo[i]
    return xct, mlog, len(coords)


def assemble_blended(model, padded: np.ndarray, padded_mask: np.ndarray,
                     device, core_shape, off: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Assembly B — stride-32 patches, Tukey-blended, then cropped to the core.

    The window and the blend-before-activation rule are those of
    ``poregen.eval.blended``; the padding is real neighbouring data rather than
    a reflection, so the crop has full weight everywhere.
    """
    D, H, W = padded.shape
    coords = [(z, y, x)
              for z in range(0, D - PATCH + 1, STRIDE_B)
              for y in range(0, H - PATCH + 1, STRIDE_B)
              for x in range(0, W - PATCH + 1, STRIDE_B)]
    w3 = _tukey_window_3d(PATCH)
    xct_w = np.zeros(padded.shape, np.float32)
    mlog_w = np.zeros(padded.shape, np.float32)
    acc = np.zeros(padded.shape, np.float32)
    for z, y, x in coords:
        acc[z:z + PATCH, y:y + PATCH, x:x + PATCH] += w3
    for bc, xo, mo in decode_patches(model, padded, padded_mask, coords, device):
        for i, (z, y, x) in enumerate(bc):
            sl = np.s_[z:z + PATCH, y:y + PATCH, x:x + PATCH]
            xct_w[sl] += w3 * xo[i]
            mlog_w[sl] += w3 * mo[i]
    safe = np.where(acc > 0, acc, 1.0)
    D0, H0, W0 = core_shape
    crop = np.s_[off:off + D0, off:off + H0, off:off + W0]
    return (xct_w / safe)[crop].copy(), (mlog_w / safe)[crop].copy(), len(coords)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def grey_u8_scale(xct_out: np.ndarray) -> np.ndarray:
    """XCT-head output -> grey level on the raw-scan u8 scale (float).

    Numpy mirror of ``poregen.models.vae.base.decode_xct`` * 255, i.e. what
    the sampler measures its own seam on.  No sigmoid.
    """
    return np.clip(xct_out, 0.0, 1.0) * 255.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / denom if denom > 0 else float("nan")


def score(name: str, grey: np.ndarray, mlog: np.ndarray,
          gt_mask: np.ndarray | None) -> dict:
    s = {"assembly": name}
    s.update({k: v for k, v in seam_discontinuity(grey, PATCH, "seam_xct").items()
              if k.endswith("ratio") or k == "seam_xct_mad"})
    s.update({k: v for k, v in seam_discontinuity(mlog, PATCH, "seam_mask").items()
              if k.endswith("ratio") or k == "seam_mask_mad"})
    pred = mlog > 0.0
    s["porosity"] = float(pred.mean())
    s["dice_vs_stored_mask"] = (float("nan") if gt_mask is None
                                else dice(pred, gt_mask))
    return s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_volume(model, device, short: str, vid: str, g, field: dict,
               save: bool) -> dict:
    arr_x, arr_m, arr_s = g[vid]["xct"], g[vid]["mask"], g[vid]["sample_mask"]
    Dv, Hv, Wv = arr_x.shape
    ext = field[vid]["extent_foreground"]["z"]
    H = W = REGION_YX

    log(f"{short}: searching region in {arr_x.shape}, laminate z extent {ext}")
    # sample_mask, not the foreground extent, is the criterion: it is False
    # outside the specimen AND inside the three drilled holes.  Searching the
    # whole z range lets a box sit wherever the material actually is.
    ok_z = cell_ok_by_slice(arr_s, PATCH)
    reg = D = None
    for depth in DEPTHS:
        reg = find_region(None, depth, H // PATCH, W // PATCH, ok_z=ok_z)
        if reg is not None:
            D = depth
            break
    if reg is None:
        raise SystemExit(f"{short}: no {DEPTHS[-1]}x{H}x{W} box lies fully "
                         "inside sample_mask")
    reg["depth"] = D
    z0, y0, x0 = reg["z0"], reg["y0"], reg["x0"]
    log(f"{short}: region z={z0}:{z0 + D} y={y0}:{y0 + H} x={x0}:{x0 + W} "
        f"(z 64-aligned={reg['z_aligned']}, usable cells={reg['usable_cell_fraction']:.2f})")

    # Core + a halo for the overlapped assembly.  The halo is REAL neighbouring
    # data wherever the array reaches that far; the laminate is only ~200
    # voxels deep, so in z it usually does not, and the deficit is reflected
    # (what poregen.eval.blended does on every side).
    lo = np.array([z0, y0, x0]) - MARGIN
    hi = np.array([z0 + D, y0 + H, x0 + W]) + MARGIN
    shape = np.array([Dv, Hv, Wv])
    clo = np.clip(lo, 0, shape)
    chi = np.clip(hi, 0, shape)
    pad = [(int(clo[i] - lo[i]), int(hi[i] - chi[i])) for i in range(3)]
    halo_real = {ax: 1.0 - (pad[i][0] + pad[i][1]) / (2.0 * MARGIN)
                 for i, ax in enumerate("zyx")}
    log(f"{short}: halo real fraction per axis {halo_real}")

    sl = np.s_[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]]
    padded = np.pad(np.asarray(arr_x[sl]).astype(np.float32) / 255.0,
                    pad, mode="reflect")
    padded_mask = np.pad((np.asarray(arr_m[sl]) > 0).astype(np.float32),
                         pad, mode="reflect")
    crop_core = np.s_[MARGIN:MARGIN + D, MARGIN:MARGIN + H, MARGIN:MARGIN + W]
    core = padded[crop_core].copy()
    core_mask = padded_mask[crop_core].copy()
    gt_mask = core_mask > 0

    region_shape = (D, H, W)
    rows = []
    vols = {}

    t = time.time()
    xa, ma, na = assemble_tiled(model, core, core_mask, device)
    log(f"{short}: assembly A done, {na} patches, {time.time() - t:.0f}s")
    rows.append(score("A_tiled", grey_u8_scale(xa), ma, gt_mask))
    rows[-1]["n_patches"] = na
    vols["A_tiled"] = (grey_u8_scale(xa), ma)
    del xa

    t = time.time()
    xb, mb, nb = assemble_blended(model, padded, padded_mask, device, region_shape, MARGIN)
    log(f"{short}: assembly B done, {nb} patches, {time.time() - t:.0f}s")
    rows.append(score("B_overlapped", grey_u8_scale(xb), mb, gt_mask))
    rows[-1]["n_patches"] = nb
    vols["B_overlapped"] = (grey_u8_scale(xb), mb)
    del xb

    # Control: the real data.  The stored mask is binary, so its "logit" is a
    # two-valued surrogate — the seam_mask number for C measures how the binary
    # mask itself changes across a 64-plane, not a decoder disagreement.
    ctrl_logit = np.where(gt_mask, MASK_LOGIT_CLIP, -MASK_LOGIT_CLIP).astype(np.float32)
    rows.append(score("C_real", core * 255.0, ctrl_logit, gt_mask))
    rows[-1]["n_patches"] = 0

    if save:
        for name, (grey, mlog) in vols.items():
            d = OUT_DIR / "volumes" / short / name
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / "volume.tif"),
                             np.round(grey).astype(np.uint8))
            tifffile.imwrite(str(d / "mask.tif"),
                             ((mlog > 0).astype(np.uint8) * 255))
        d = OUT_DIR / "volumes" / short / "C_real"
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / "volume.tif"),
                         np.round(core * 255.0).astype(np.uint8))
        tifffile.imwrite(str(d / "mask.tif"), (gt_mask.astype(np.uint8) * 255))

    return {"volume": short, "volume_id": vid, "region": reg,
            "region_shape": list(region_shape),
            "halo_real_fraction": halo_real,
            "stored_mask_porosity": float(gt_mask.mean()),
            "assemblies": rows}


def build_findings(records: list[dict], meta: dict) -> str:
    L = ["# 0a — VAE tile seam: assembly, not the LDM", "",
         f"VAE `{meta['vae_checkpoint']}` (production r07 z=4, the one that "
         f"built `data/split_v2/latents_r07z4`). {len(records)} real val "
         f"volumes, one box each ("
         + ", ".join("%s %dx%dx%d" % (r["volume"], *r["region_shape"])
                     for r in records)
         + "), entirely inside `sample_mask` (which is False inside the three "
         "drilled holes, so they are excluded by construction). Posterior "
         "mean `mu` in both assemblies — the only difference between A and B "
         "is how the decoded patches are put together.", "",
         "| volume | assembly | seam_xct_ratio | seam_mask_ratio | pore Dice "
         "| porosity |", "|---|---|---|---|---|---|"]
    for r in records:
        for a in r["assemblies"]:
            d = a["dice_vs_stored_mask"]
            L.append(f"| {r['volume']} | {a['assembly']} "
                     f"| {a['seam_xct_ratio']:.3f} | {a['seam_mask_ratio']:.3f} "
                     f"| {'—' if not np.isfinite(d) else f'{d:.3f}'} "
                     f"| {a['porosity']:.4f} |")
    L += ["", "## Mean over volumes", "",
          "| assembly | seam_xct_ratio | seam_mask_ratio | pore Dice | porosity |",
          "|---|---|---|---|---|"]
    for name in ("A_tiled", "B_overlapped", "C_real"):
        sel = [a for r in records for a in r["assemblies"] if a["assembly"] == name]
        if not sel:
            continue
        L.append(f"| {name} "
                 f"| {np.mean([a['seam_xct_ratio'] for a in sel]):.3f} "
                 f"| {np.mean([a['seam_mask_ratio'] for a in sel]):.3f} "
                 f"| {np.nanmean([a['dice_vs_stored_mask'] for a in sel]):.3f} "
                 f"| {np.mean([a['porosity'] for a in sel]):.4f} |")
    L += ["",
          "Reference: the joint_oob arm of "
          "`runs/campaigns/05-eval-v3-fixed-decode` reports "
          "`seam_mask_ratio` 2.25-2.53 and `seam_xct_ratio` 4.25-4.69 on "
          "generated 1024x1024x192 volumes (DDIM-50, RAW weights).", "",
          "## Caveats", "",
          "- Assembly C's `seam_mask_ratio` is computed on a two-valued "
          f"surrogate logit (+/-{MASK_LOGIT_CLIP:.0f} from the stored binary "
          "mask), so it measures how the stored mask itself changes across a "
          "64-plane. It is a floor for the metric, not a decoder measurement.",
          "- Both assemblies decode under bfloat16 autocast, the same dtype "
          "the sampler uses, so the ratios are comparable with the generated "
          "volumes.",
          "- Assembly B reads a 64-voxel halo of real data around the box "
          "wherever the array reaches that far, so the Tukey taper rarely "
          "falls back on a reflection; `halo_real_fraction` records how much "
          "was real per axis. A sampler cannot do that at the outer face of a "
          "generated volume.",
          "- The Nacho laminates hold only ~185 voxels of continuous "
          "material, so a 192-deep all-material box does not exist in them "
          "and those volumes use a 128-deep box. The seam metric is a ratio "
          "of seam to interior slice-difference, so it does not depend on the "
          "number of planes.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save-volumes", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    meta = json.loads((LATENTS_ROOT / "metadata.json").read_text())
    ckpt = Path(meta["vae_checkpoint"])
    vae, cfg, _, _ = load_vae_from_checkpoint(ckpt, device)
    vae.requires_grad_(False)
    log(f"VAE loaded: {cfg['model']['name']} from {ckpt}")

    g = zarr.open_group(str(ZARR_ROOT), mode="r")
    field = json.loads(ORIENT_FIELD.read_text())["volumes"]

    records = [run_volume(vae, device, short, vid, g, field,
                          save=not args.no_save_volumes)
               for short, vid in VOLUMES.items()]

    info = {"vae_checkpoint": str(ckpt), "vae_model": cfg["model"]["name"]}
    results = {
        "campaign": "08 — 0a VAE tile seam (assembly vs LDM)",
        "question": "Do the mask seams in generated volumes come from the VAE "
                    "tiled assembly or from the LDM?",
        **info,
        "region_shape_yx": REGION_YX,
        "region_depths_tried": list(DEPTHS),
        "halo_vox": MARGIN,
        "patch_size": PATCH,
        "stride_tiled": STRIDE_A,
        "stride_overlapped": STRIDE_B,
        "blend_window": "Tukey alpha=0.5 (poregen.eval.blended._tukey_window_3d)",
        "latent_sampling": "posterior mean mu (no stochastic pass)",
        "decode_dtype": "bfloat16 autocast (matches VolumeGenerator)",
        "reference_generated": {
            "source": "runs/campaigns/05-eval-v3-fixed-decode/volumes/layup/joint_oob",
            "seam_mask_ratio": [2.254, 2.380, 2.526],
            "seam_xct_ratio": [4.252, 4.689, 4.650],
        },
        "records": records,
    }
    write_json(results, OUT_DIR)
    write_findings(build_findings(records, info), OUT_DIR)
    log(f"wrote {OUT_DIR}/results.json and findings.md")


if __name__ == "__main__":
    main()

"""Compare two decoders on the SAME latents — the D43 decoder fine-tune gate.

A decoder fine-tune is only judgeable against the decoder it replaces, on
identical input. Regenerating volumes from a seed does not give that: it re-runs
the sampler and produces a different latent canvas whenever anything about the
sampling path changes. So this script reads the canvas the generation actually
used (`latents.npy`, written by ``generate_volumes.py --save-latents``) and puts
it through both decoders.

Two arms, because the D43 gates ask two different questions:

**val** — real val patches encoded by the FROZEN encoder, decoded both ways.
Answers "did the fine-tune improve reconstruction sharpness without moving
segmentation", and has ground truth, so pore Dice / porosity MAE / air Dice are
real numbers here.

**generated** — ldm06 latents decoded both ways. Answers "does the improvement
survive on the distribution the decoder actually meets in production". There is
no ground truth, so sharpness is scored as a RATIO against real volumes, and
segmentation is compared BETWEEN the two decoders rather than against truth.
That between-decoders comparison is the guard the fallback needs: a refiner that
buys sharpness by inventing pore-scale texture moves S2 at small r and the pore
size distribution while leaving the sharpness ratio looking excellent.

Usage
-----
    python scripts/analysis/decoder_ft_redecode.py \
        --baseline runs/vae/r08-run-0004-.../best.ckpt \
        --finetuned runs/vae/r08-decoder-ft-run-0001-.../best.ckpt \
        --latents 'inference/ldm06-.../por_*/coherent/latents.npy' \
        --out runs/campaigns/11-decoder-ft/redecode
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poregen.dataset.loader import LABEL_AIR, LABEL_PORE  # noqa: E402
from poregen.eval_v4 import microstructure as MS  # noqa: E402
from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: E402
from poregen.metrics.recon import sharpness_proxy  # noqa: E402
from poregen.runtime.preflight import prepare_patch_dataloaders  # noqa: E402

logger = logging.getLogger("decoder_ft_redecode")

#: Latent windows decoded per forward pass.
DECODE_BATCH = 32
#: Val patches scored. 2048 at batch 32 is 64 forwards per decoder — enough for
#: a Dice difference of 0.005 to clear the between-batch noise, and small
#: enough to run in the gap between two training jobs.
N_VAL_PATCHES = 2048
#: Latent cells per decoded window, and voxels per latent cell. The VAE maps a
#: 16^3 latent window to a 64^3 patch at f=4.
LATENT_WIN = 16
VOX_PER_CELL = 4
#: Voxels per decoded block: a 16-cell window becomes a 64^3 patch.
PATCH_VOX = LATENT_WIN * VOX_PER_CELL
#: "Small r" for the texture guard, in voxels. Pore-scale invention shows up
#: here first: the mean pore is ~1.8 voxels across, so r <= 4 is the band a
#: decoder would have to corrupt to fake sharpness.
S2_SMALL_R = 4


def _manifest_for(path: Path):
    """The case's own Manifest, or None.

    s2_profile type-checks its manifest, and a Manifest must declare a real
    sampler plus the fourteen fields that say what produced the volume. An
    eval-v4 case has all of it in manifest.json beside latents.npy. When that
    file is absent the honest answer is to skip S2 and say so — fabricating a
    manifest to satisfy the validator would put a provenance claim on a number
    that has none, which is the opposite of what the validator is for.
    """
    from poregen.eval_v4.manifest import Manifest

    mf_path = path.parent / "manifest.json"
    if not mf_path.exists():
        return None
    try:
        return Manifest.read(mf_path)
    except Exception as exc:                        # noqa: BLE001
        logger.warning("%s unreadable (%s); S2 skipped for this canvas", mf_path, exc)
        return None


def decode_latents(model, z: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(B, C, 16, 16, 16) latents -> (grey in [0,1], class label) as numpy."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        dec = model.decoder(z.to(device))
        xct = model.xct_head(dec).float()
        logits = model.class_head(dec).float()
    return (
        xct.squeeze(1).clamp(0.0, 1.0).cpu().numpy(),
        logits.argmax(dim=1).cpu().numpy().astype(np.uint8),
    )


def class_agreement(a: np.ndarray, b: np.ndarray, cls: int) -> float:
    """Dice between two label arrays for one class — no ground truth needed."""
    pa, pb = (a == cls), (b == cls)
    inter = float(np.logical_and(pa, pb).sum())
    card = float(pa.sum() + pb.sum())
    return 1.0 if card == 0.0 else 2.0 * inter / card


def dice_vs_truth(pred: np.ndarray, truth: np.ndarray, cls: int) -> float:
    return class_agreement(pred, truth, cls)


def val_arm(base, ft, device, data_root, cfg) -> dict:
    """Real val patches through the frozen encoder, decoded by both."""
    cfg, (_, val_loader, _) = prepare_patch_dataloaders(cfg, data_root)
    out: dict[str, list] = {k: [] for k in (
        "sharp_base", "sharp_ft", "sharp_gt",
        "dice_pore_base", "dice_pore_ft", "dice_air_base", "dice_air_ft",
        "por_base", "por_ft", "por_gt",
    )}
    seen = 0
    for batch in val_loader:
        xct = batch["xct"].to(device)
        label = batch["label"].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # The encoder is frozen and SHARED: mu is identical for both
            # decoders by construction, so encoding once is not a shortcut, it
            # is the guarantee that the two arms differ only in the decoder.
            enc_in = torch.cat(
                [xct, _label_planes(label).to(xct.dtype)], dim=1
            )
            h = base.fusion(torch.cat([base.encoder_a(enc_in), base.encoder_b(enc_in)], dim=1))
            mu = base.to_mu(h).float()
        g_b, l_b = decode_latents(base, mu, device)
        g_f, l_f = decode_latents(ft, mu, device)
        gt = label.cpu().numpy()
        out["sharp_gt"].append(float(sharpness_proxy(xct.cpu())))
        out["sharp_base"].append(float(sharpness_proxy(torch.from_numpy(g_b).unsqueeze(1))))
        out["sharp_ft"].append(float(sharpness_proxy(torch.from_numpy(g_f).unsqueeze(1))))
        for tag, lab in (("base", l_b), ("ft", l_f)):
            out[f"dice_pore_{tag}"].append(dice_vs_truth(lab, gt, LABEL_PORE))
            out[f"dice_air_{tag}"].append(dice_vs_truth(lab, gt, LABEL_AIR))
            out[f"por_{tag}"].append(float((lab == LABEL_PORE).mean()))
        out["por_gt"].append(float((gt == LABEL_PORE).mean()))
        seen += xct.shape[0]
        if seen >= N_VAL_PATCHES:
            break
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["n_patches"] = seen
    res["sharpness_ratio_base"] = res["sharp_base"] / res["sharp_gt"]
    res["sharpness_ratio_ft"] = res["sharp_ft"] / res["sharp_gt"]
    res["porosity_mae_base"] = abs(res["por_base"] - res["por_gt"])
    res["porosity_mae_ft"] = abs(res["por_ft"] - res["por_gt"])
    return res


def _label_planes(label: torch.Tensor) -> torch.Tensor:
    from poregen.models.vae.v2.conv_noattn_dualbranch_cls import label_to_channels
    return label_to_channels(label)


def _ddim_steps_of(path: Path) -> int | None:
    """The step count the canvas beside this file was generated at.

    Read from the case's own manifest, not from its directory name: the name is
    a convenience and the manifest is the record.
    """
    mf = path.parent / "manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text()).get("ddim_steps")
    except Exception:                                   # noqa: BLE001
        return None


def _group_by_steps(rows: list[dict]) -> dict:
    """Summarise the generated arm separately for each DDIM step count.

    Both are reported because the two are not interchangeable on this model: the
    ldm06 40k diagnostic has porosity conditioning 5-7x worse at 200 steps than
    at 50 and the latent std ~8% wide on the high-sigma channels, so judging a
    decoder only on 200-step volumes would charge it for the sampler's weaker
    operating point. The verdict is taken at whichever step count the sampler
    assessment identifies as the operating point; the other stands beside it.
    """
    out: dict = {}
    for steps in sorted({r.get("ddim_steps") for r in rows if r.get("ddim_steps")}):
        sel = [r for r in rows if r.get("ddim_steps") == steps]

        def mean(key):
            v = [r[key] for r in sel if r.get(key) is not None]
            return float(np.mean(v)) if v else None

        out[f"ddim{steps}"] = {
            "n_volumes": len(sel),
            "sharpness_ratio_base": mean("sharpness_ratio_base"),
            "sharpness_ratio_ft": mean("sharpness_ratio_ft"),
            "agree_pore": mean("agree_pore"),
            "agree_air": mean("agree_air"),
            "porosity_base": mean("porosity_base"),
            "porosity_ft": mean("porosity_ft"),
            "s2_small_r_max_abs_diff": mean("s2_small_r_max_abs_diff"),
            "s2_w1_base_vs_ft": mean("s2_w1_base_vs_ft"),
            "psd_w1_base_vs_ft": mean("psd_w1_base_vs_ft"),
            # Gate 2 as such, evaluated per step count rather than pooled.
            "gate2_sharpness_ratio_ft_ge_0.95": (
                (mean("sharpness_ratio_ft") or 0.0) >= 0.95),
        }
    ungrouped = [r for r in rows if not r.get("ddim_steps")]
    if ungrouped:
        out["unknown_step_count"] = {
            "n_volumes": len(ungrouped),
            "note": ("these canvases carry no manifest, so their step count is "
                     "unknown and they are excluded from the per-step verdict"),
        }
    return out


def generated_arm(base, ft, device, latent_files: list[Path], real_sharp: float) -> dict:
    """ldm06 latents decoded both ways; no ground truth, so compare to each other."""
    rows = []
    for f in latent_files:
        z_all = np.load(f).astype(np.float32)          # (C, Z, Y, X) latent cells
        C, Z, Y, X = z_all.shape
        if min(Z, Y, X) < LATENT_WIN:
            logger.warning("%s: canvas %s smaller than one %d-cell window, skipped",
                           f.name, (Z, Y, X), LATENT_WIN)
            continue
        # Decode into FULL volumes, not a stack of patches. S2 needs
        # S2_WINDOW-cubed windows of contiguous material, which do not exist in
        # a pile of loose 64^3 blocks — measuring on the pile would silently
        # report a different quantity from the eval_v4 tables this has to match.
        vol_shape = (Z * VOX_PER_CELL, Y * VOX_PER_CELL, X * VOX_PER_CELL)
        g_b = np.empty(vol_shape, dtype=np.float32)
        g_f = np.empty(vol_shape, dtype=np.float32)
        l_b = np.empty(vol_shape, dtype=np.uint8)
        l_f = np.empty(vol_shape, dtype=np.uint8)

        origins = [(z, y, x)
                   for z in range(0, Z - LATENT_WIN + 1, LATENT_WIN)
                   for y in range(0, Y - LATENT_WIN + 1, LATENT_WIN)
                   for x in range(0, X - LATENT_WIN + 1, LATENT_WIN)]
        for i in range(0, len(origins), DECODE_BATCH):
            chunk = origins[i:i + DECODE_BATCH]
            zb = torch.from_numpy(np.stack([
                z_all[:, z:z + LATENT_WIN, y:y + LATENT_WIN, x:x + LATENT_WIN]
                for z, y, x in chunk
            ]))
            a_g, a_l = decode_latents(base, zb, device)
            b_g, b_l = decode_latents(ft, zb, device)
            for j, (z, y, x) in enumerate(chunk):
                # The origin steps by LATENT_WIN cells, so the decoded block
                # spans LATENT_WIN * VOX_PER_CELL voxels — not one cell's worth.
                z0, y0, x0 = z * VOX_PER_CELL, y * VOX_PER_CELL, x * VOX_PER_CELL
                sl = np.s_[z0:z0 + PATCH_VOX, y0:y0 + PATCH_VOX, x0:x0 + PATCH_VOX]
                g_b[sl], l_b[sl] = a_g[j], a_l[j]
                g_f[sl], l_f[sl] = b_g[j], b_l[j]

        row = {
            "latents": str(f),
            "ddim_steps": _ddim_steps_of(f),
            "case": f.parent.name,
            "volume_shape": list(vol_shape),
            "n_windows": len(origins),
            "sharp_base": float(sharpness_proxy(torch.from_numpy(g_b)[None, None])),
            "sharp_ft": float(sharpness_proxy(torch.from_numpy(g_f)[None, None])),
            # Between decoders — the guard. Ground truth does not exist here.
            "agree_pore": class_agreement(l_b, l_f, LABEL_PORE),
            "agree_air": class_agreement(l_b, l_f, LABEL_AIR),
            "porosity_base": float((l_b == LABEL_PORE).mean()),
            "porosity_ft": float((l_f == LABEL_PORE).mean()),
        }
        row["sharpness_ratio_base"] = row["sharp_base"] / real_sharp
        row["sharpness_ratio_ft"] = row["sharp_ft"] / real_sharp

        # S2 at small r and the pore size distribution: a decoder that buys
        # sharpness by inventing pore-scale texture moves these while leaving
        # the sharpness ratio looking healthy.
        #
        # BOTH decoders are scored inside the BASELINE's material mask. Using
        # each decoder's own air class would select different analysis windows
        # for each, and two S2 curves measured over different windows are not
        # comparable — the difference would partly be which windows were picked.
        material = (l_b != LABEL_AIR)
        mf = _manifest_for(f)
        try:
            if mf is None:
                raise FileNotFoundError(
                    f"no manifest.json beside {f.name}; S2 needs the case manifest "
                    "and this canvas carries no provenance to build one from")
            s2_b = MS.s2_profile(l_b, material, manifest=mf)
            s2_f = MS.s2_profile(l_f, material, manifest=mf)
            row["s2_zero_lag_base"] = s2_b["s2_zero_lag"]
            row["s2_zero_lag_ft"] = s2_f["s2_zero_lag"]
            row["s2_n_windows"] = s2_b["n_windows"]
            r = np.asarray(s2_b["r"])
            row["s2_w1_base_vs_ft"] = float(
                MS.curve_w1(r, np.asarray(s2_b["s2"]), np.asarray(s2_f["s2"])))
            small = r <= S2_SMALL_R
            row["s2_small_r_max_abs_diff"] = float(
                np.abs(np.asarray(s2_b["s2"])[small] - np.asarray(s2_f["s2"])[small]).max())
        except Exception as exc:                    # noqa: BLE001
            row["s2_error"] = repr(exc)
        try:
            d_b = MS.pore_diameters((l_b == LABEL_PORE) & material)
            d_f = MS.pore_diameters((l_f == LABEL_PORE) & material)
            row["psd_w1_base_vs_ft"] = float(MS.psd_w1(d_b, d_f))
            row["psd_median_base"] = float(np.median(d_b)) if d_b.size else None
            row["psd_median_ft"] = float(np.median(d_f)) if d_f.size else None
        except Exception as exc:                    # noqa: BLE001
            row["psd_error"] = repr(exc)

        rows.append(row)
        logger.info("%s  sharp %.4f -> %.4f  pore agree %.4f  s2 small-r max diff %s",
                    f.name, row["sharp_base"], row["sharp_ft"], row["agree_pore"],
                    row.get("s2_small_r_max_abs_diff", "n/a"))
    return {"per_volume": rows, "by_ddim_steps": _group_by_steps(rows)}


def _at_ddim(files: list[Path], steps: int) -> list[Path]:
    """Keep the canvases whose CASE MANIFEST records ``steps``.

    The directory name is not the selector and must not be used as one. Case
    names carry the step count only when the assessment varies it: in campaign
    12, 55 of the 72 DDIM-50 cases say "ddim50" in their name but only 45 of the
    99 DDIM-200 cases say "ddim200". Globbing on the name would have built the
    DDIM-200 gate row out of less than half its volumes and reported it as the
    whole thing.

    A canvas whose manifest is missing or unreadable is DROPPED and named, not
    silently kept: the gate compares two decoders on identical input, and a
    canvas whose sampling settings cannot be established is not identical to
    anything.
    """
    keep = []
    for f in files:
        mf = f.parent / "manifest.json"
        if not mf.exists():
            logger.warning("no manifest beside %s — dropped", f)
            continue
        try:
            got = json.loads(mf.read_text()).get("ddim_steps")
        except Exception as exc:                          # noqa: BLE001
            logger.warning("unreadable manifest %s (%s) — dropped", mf, exc)
            continue
        if got == steps:
            keep.append(f)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="the decoder being replaced")
    ap.add_argument("--finetuned", required=True, help="the fine-tuned decoder")
    ap.add_argument("--latents", default=None,
                    help="glob for latents.npy files from generate_volumes.py --save-latents. "
                         "Omit to run the val arm only.")
    ap.add_argument("--ddim-steps", type=int, default=None,
                    help="keep only canvases whose case manifest records this "
                         "ddim_steps. Selection comes from the MANIFEST and never "
                         "from the directory name: in campaign 12 only 45 of the 99 "
                         "DDIM-200 cases carry 'ddim200' in their name, so a name "
                         "glob silently drops more than half the row.")
    ap.add_argument("--real-sharpness", type=float, default=None,
                    help="sharpness_proxy of real volumes, the denominator of the generated "
                         "arm's ratio. Default: measured on the val arm's ground truth.")
    ap.add_argument("--out", default="runs/campaigns/11-decoder-ft/redecode")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    base, cfg, _, base_run = load_vae_from_checkpoint(Path(args.baseline), device)
    ft, ft_cfg, _, ft_run = load_vae_from_checkpoint(Path(args.finetuned), device)

    # The comparison is only meaningful if the two share an encoder. If the
    # fine-tune moved it, every latent in the store is stale and this script is
    # measuring the wrong thing — so check rather than assume.
    with torch.no_grad():
        drift = max(
            float((pa - pb).abs().max())
            for (na, pa), (nb, pb) in zip(base.named_parameters(), ft.named_parameters())
            if na.startswith(("encoder_a", "encoder_b", "fusion", "to_mu", "to_logvar"))
        )
    if drift > 0.0:
        raise SystemExit(
            f"ENCODER MOVED (max |delta| = {drift:.3e}). The fine-tune was supposed to "
            "freeze it. data/split_v3/latents_r08z8 is stale and so is any LDM trained "
            "on it; re-run the fine-tune with training.freeze_modules set."
        )
    logger.info("encoder identical in both checkpoints (max |delta| = 0)")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    data_root = repo / "data" / cfg["data"].get("dataset_root", "split_v3")
    results = {
        "baseline": str(args.baseline), "baseline_run": str(base_run),
        "finetuned": str(args.finetuned), "finetuned_run": str(ft_run),
        "encoder_identical": True,
    }

    logger.info("val arm: %d patches", N_VAL_PATCHES)
    results["val"] = val_arm(base, ft, device, data_root, cfg)
    real_sharp = args.real_sharpness or results["val"]["sharp_gt"]
    results["real_sharpness"] = real_sharp

    if args.latents:
        files = [Path(p) for p in sorted(glob.glob(args.latents))]
        if not files:
            raise SystemExit(f"--latents matched nothing: {args.latents}")
        if args.ddim_steps is not None:
            files = _at_ddim(files, args.ddim_steps)
            if not files:
                raise SystemExit(
                    f"--latents matched canvases but none at ddim_steps="
                    f"{args.ddim_steps}: {args.latents}")
        logger.info("generated arm: %d latent canvases%s", len(files),
                    f" at DDIM-{args.ddim_steps}" if args.ddim_steps else "")
        results["generated"] = generated_arm(base, ft, device, files, real_sharp)
    else:
        results["generated"] = {"skipped": "no --latents given"}

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    v = results["val"]
    logger.info("VAL  sharpness ratio %.4f -> %.4f   pore Dice %.4f -> %.4f   air Dice %.4f -> %.4f",
                v["sharpness_ratio_base"], v["sharpness_ratio_ft"],
                v["dice_pore_base"], v["dice_pore_ft"],
                v["dice_air_base"], v["dice_air_ft"])
    g = results.get("generated", {}).get("by_ddim_steps") or {}
    for key, b in g.items():
        if "n_volumes" not in b or b.get("sharpness_ratio_ft") is None:
            continue
        logger.info("GEN %-8s (%d vols)  sharpness %.4f -> %.4f   gate2 %s   "
                    "pore agree %.4f   S2 small-r max diff %s",
                    key, b["n_volumes"], b["sharpness_ratio_base"],
                    b["sharpness_ratio_ft"],
                    "PASS" if b["gate2_sharpness_ratio_ft_ge_0.95"] else "FAIL",
                    b["agree_pore"],
                    (f"{b['s2_small_r_max_abs_diff']:.4f}"
                     if b.get("s2_small_r_max_abs_diff") is not None else "n/a"))
    if len(g) > 1:
        logger.info("Gate 2 verdict is taken at the step count the sampler "
                    "assessment identifies as the operating point; the other is "
                    "reported beside it.")
    logger.info("wrote %s", out_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

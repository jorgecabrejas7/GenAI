"""The adapter between an assessment case and the ldm06 hybrid sampler.

This is the only module in the suite that knows the sampler API.  An assessment
describes a request; :class:`VolumeRunner` turns it into a
:class:`poregen.diffusion.sampler.VolumeGenerator` call and writes the case
directory with its manifest.

Three places where the sampler's interface shapes what the suite can do, all
recorded here rather than worked around silently:

* **Seeding.** ``VolumeGenerator.generate`` takes no seed; the initial noise
  comes from the global torch generator.  The runner therefore seeds
  ``torch.manual_seed`` immediately before each call, and the manifest records
  the seed it set.
* **Size in millimetres.** ``generate`` takes a physical size and snaps it DOWN
  to whole tiles.  The runner converts a voxel shape to millimetres and then
  checks that the generator snapped back to exactly the shape that was asked
  for, so a rounding error cannot quietly change the volume.
* **Window phase.** Window origins are anchored at the chunk origin and there
  is no phase parameter.  Assessment 6 gets its 32-voxel shift by translating
  the REQUEST inside a larger canvas (``CaseSpec.request_offset``), which moves
  the assembly grid relative to the content without touching the sampler.

The runner reaches the model through two entry points only -
``DDPMSchedule.from_cfg`` and ``VolumeGenerator.generate`` - and never through
``DDIMSampler.predict_out``.  Whether a run predicts epsilon or v is the
schedule's business; the suite records which it was and otherwise leaves the
conversion alone.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from poregen.diffusion.conditioning import POR_MAX, POR_MIN
from poregen.eval_v4.cases import CaseSpec
from poregen.eval_v4.io import LATENT_DOWNSAMPLE, repo_root, save_case
from poregen.eval_v4.manifest import Manifest, head_commit

logger = logging.getLogger(__name__)

VOXEL_SIZE_MM = 0.025
PATCH_SIZE = 64
LATENT_SIZE = PATCH_SIZE // LATENT_DOWNSAMPLE
WINDOW_BATCH = 32
DECODE_BATCH = 64
DEFAULT_LATENTS_ROOT = "data/split_v3/latents_r08z4"


def theta_for_canvas(
    depth: int,
    layup,
    ply_thickness_vox: float,
    request_offset: int = 0,
) -> np.ndarray:
    """theta(z) for a canvas whose REQUEST starts at ``request_offset``.

    At ``request_offset = 0`` this is exactly
    :func:`poregen.diffusion.sampler.theta_from_layup`.  The offset shifts the
    ply grid with the request, so the region a case is about sees the same ply
    sequence at the same phase however it is placed in the canvas - which is
    what makes the assembly comparison a comparison of grids and not of content.
    """
    angles = np.asarray(layup, dtype=np.float64)
    if angles.size == 0:
        raise ValueError("layup must contain at least one ply angle.")
    z = np.arange(int(depth), dtype=np.float64) - float(request_offset)
    idx = np.floor(z / float(ply_thickness_vox)).astype(np.int64) % angles.size
    return angles[idx].astype(np.float32)


def latent_material_map(voxel_material: np.ndarray) -> np.ndarray:
    """Specimen-envelope FRACTION per latent cell - what ``cond_material`` is.

    The average over each 4-cubed block, not a sample of it: a cell that a
    notch clips is half specimen, and telling the model it is fully one or the
    other loses the edge it was asked to render.
    """
    d = LATENT_DOWNSAMPLE
    z, y, x = (s // d for s in voxel_material.shape)
    return (
        voxel_material[: z * d, : y * d, : x * d]
        .reshape(z, d, y, d, x, d)
        .mean(axis=(1, 3, 5), dtype=np.float64)
        .astype(np.float32)
    )


def _ckpt_path(run_dir: Path, ckpt: str) -> tuple[Path, int]:
    """Resolve ``--ckpt`` to a file and the training step it holds."""
    ckpts = run_dir / "checkpoints"
    if ckpt in ("best", "latest"):
        path = ckpts / f"{ckpt}.ckpt"
    else:
        try:
            step = int(ckpt)
        except ValueError as exc:
            raise ValueError(
                f"--ckpt must be a step number, 'best' or 'latest'; got {ckpt!r}"
            ) from exc
        path = ckpts / f"ldm_step{step:08d}.ckpt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Available: "
            f"{sorted(p.name for p in ckpts.glob('*.ckpt'))[:8]} ..."
        )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    step = int(raw.get("step", raw.get("global_step", -1)))
    return path, step


class VolumeRunner:
    """Loads the model once, then generates any number of cases.

    The LDM, the VAE and the latent-store statistics are loaded on construction
    because every case shares them.  The sampler and the generator are rebuilt
    per case - both are thin objects - so a case can change the step count, the
    guidance scales, the chunk geometry and the layup without any state
    surviving from the previous one.
    """

    def __init__(
        self,
        run_dir: str | Path,
        ckpt: str,
        *,
        weights: str = "ema",
        latents_root: str | Path | None = None,
        device: torch.device | None = None,
        repo: str | Path | None = None,
    ) -> None:
        import yaml  # noqa: PLC0415

        from poregen.diffusion.noise_schedule import DDPMSchedule  # noqa: PLC0415
        from poregen.experiments.train_vae import load_vae_from_checkpoint  # noqa: PLC0415
        from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser  # noqa: PLC0415
        from poregen.training.checkpoint import load_checkpoint  # noqa: PLC0415

        if weights not in ("raw", "ema"):
            raise ValueError(f"weights must be 'raw' or 'ema', got {weights!r}")

        self.repo = Path(repo) if repo else repo_root()
        self.run_dir = Path(run_dir).resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.git_commit = head_commit(self.repo)

        ckpt_path, step = _ckpt_path(self.run_dir, ckpt)
        self.checkpoint_path = ckpt_path
        self.checkpoint_step = step

        cfg = yaml.safe_load((self.run_dir / "resolved_config.yaml").read_text())
        self.model_cfg = cfg
        model = UNet3DDenoiser(UNet3DConfig.from_cfg(cfg)).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if weights == "ema":
            if "ema" not in state:
                raise KeyError(
                    f"{ckpt_path} carries no EMA weights; pass --weights raw to "
                    "measure the training weights instead of silently using them."
                )
            ema = state["ema"]
            ema = {k.removeprefix("_orig_mod."): v for k, v in ema.items()}
            model.load_state_dict({k: v.to(self.device) for k, v in ema.items()})
        else:
            load_checkpoint(str(ckpt_path), model=model, map_location=self.device)
        model.eval()
        self.model = model
        self.weights = weights

        root = Path(latents_root or (self.repo / DEFAULT_LATENTS_ROOT))
        if not root.is_absolute():
            root = (self.repo / root).resolve()
        meta = json.loads((root / "metadata.json").read_text())
        norm = meta["normalization"]
        c = len(norm["per_channel_mean"])
        self.latent_mean = torch.tensor(norm["per_channel_mean"], dtype=torch.float32).view(c, 1, 1, 1)
        self.latent_std = torch.tensor(norm["per_channel_std"], dtype=torch.float32).view(c, 1, 1, 1)
        stand = (meta.get("conditioning") or {}).get("por_standardisation")
        if stand is None:
            raise KeyError(
                f"{root}/metadata.json has no conditioning.por_standardisation; run "
                "scripts/build_conditioning.py before generating."
            )
        self.por_log_stats = (float(stand["mean"]), float(stand["std"]))
        self.latents_root = root

        vae, _, _, _ = load_vae_from_checkpoint(Path(meta["vae_checkpoint"]), self.device)
        vae.requires_grad_(False)
        self.vae = vae
        self.vae_checkpoint = str(meta["vae_checkpoint"])

        # DDPMSchedule.from_cfg is the single schedule builder: it is the only
        # thing that knows which config keys describe the schedule, the
        # prediction objective among them.  Reading those keys here would be a
        # second definition, and it would silently build an epsilon schedule for
        # a run trained on v - a plausible volume that is wrong.
        self.schedule = DDPMSchedule.from_cfg(cfg, self.device)
        self.objective = str(self.schedule.objective)
        self.cfg_rescale = float((cfg.get("guidance", {}) or {}).get("cfg_rescale", 0.0))
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.autocast_dtype = torch.bfloat16

    # -- one case ----------------------------------------------------------

    def run(self, spec: CaseSpec, case_dir: str | Path, *, progress=None) -> Manifest:
        """Generate one case and write its directory.  Returns the manifest."""
        from poregen.diffusion.sampler import DDIMSampler, VolumeGenerator  # noqa: PLC0415

        shape = tuple(int(s) for s in spec.volume_shape)
        size_mm = tuple(s * VOXEL_SIZE_MM for s in shape)

        theta = theta_for_canvas(
            shape[0], spec.layup, spec.ply_thickness_vox, spec.request_offset[0]
        )
        sampler = DDIMSampler(
            self.model, self.schedule, self.device,
            n_steps=spec.ddim_steps, s_por=spec.s_por, s_nb=spec.s_nb,
            cfg_rescale=self.cfg_rescale,
        )
        generator = VolumeGenerator(
            sampler=sampler,
            vae=self.vae,
            device=self.device,
            patch_size=PATCH_SIZE,
            latent_size=LATENT_SIZE,
            latent_mean=self.latent_mean,
            latent_std=self.latent_std,
            voxel_size_mm=VOXEL_SIZE_MM,
            por_log_stats=self.por_log_stats,
            theta_deg=theta,
            chunk_tiles=spec.chunk_tiles,
            window_stride=spec.window_stride,
            decode_stride=spec.decode_stride,
        )
        snapped = generator._volume_shape(size_mm)
        if tuple(snapped) != shape:
            raise ValueError(
                f"{spec.name}: {shape} voxels became {size_mm} mm and snapped back to "
                f"{snapped}. Every axis must be a whole number of {PATCH_SIZE}-voxel tiles."
            )

        tile_field, por_map = self._porosity_request(spec)
        voxel_material, material_map = self._material_request(spec)

        torch.manual_seed(spec.seed)
        np.random.seed(spec.seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            xct_u8, label_u8, stats, probs = generator.generate(
                volume_size_mm=size_mm,
                target_porosity=spec.target_phi,
                autocast_dtype=self.autocast_dtype,
                local_por_map=por_map,
                material_map=material_map,
                specimen_box=spec.specimen_box,
                progress=progress,
                window_batch=WINDOW_BATCH,
                decode_batch_size=DECODE_BATCH,
                return_class_probs=True,
            )
        wall = time.perf_counter() - t0
        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda" else None
        )

        p_pore = np.clip(probs[1], 1e-6, 1.0 - 1e-6)
        pore_logit = (np.log(p_pore) - np.log1p(-p_pore)).astype(np.float32)

        clamped = (
            float(np.clip(spec.target_phi, POR_MIN, POR_MAX))
            if spec.target_phi is not None else None
        )
        manifest = Manifest(
            assessment=spec.assessment,
            case=spec.name,
            volume_shape=shape,
            git_commit=self.git_commit,
            sampler="hybrid_chunked",
            model_run=str(self.run_dir),
            checkpoint_step=self.checkpoint_step,
            weights=self.weights,
            ddim_steps=spec.ddim_steps,
            chunk_tiles=spec.chunk_tiles,
            window_stride=spec.window_stride,
            decode="tiled" if spec.decode_stride == PATCH_SIZE else "overlapped",
            decode_overlap=PATCH_SIZE - spec.decode_stride,
            s_por=spec.s_por,
            s_nb=spec.s_nb,
            objective=self.objective,
            cfg_rescale=self.cfg_rescale,
            seed=spec.seed,
            requested_global_phi=spec.target_phi,
            requested_field="requested_field.npy" if tile_field is not None else None,
            requested_layup=tuple(spec.layup),
            requested_ply_thickness_vox=spec.ply_thickness_vox,
            requested_material=(
                "requested_material.npy" if voxel_material is not None else "full"
            ),
            region_offset=spec.region_offset,
            region_shape=spec.region_shape,
            wall_time_s=wall,
            peak_gpu_memory_bytes=peak,
            notes={
                **spec.notes,
                "checkpoint": str(self.checkpoint_path),
                "vae_checkpoint": self.vae_checkpoint,
                "latents_root": str(self.latents_root),
                "device": str(self.device),
                "autocast_dtype": str(self.autocast_dtype),
                "conditioned_phi_after_clamp": clamped,
                "request_offset": list(spec.request_offset),
                "specimen_box": (
                    [list(spec.specimen_box[0]), list(spec.specimen_box[1])]
                    if spec.specimen_box is not None else None
                ),
                "generation_stats": {
                    k: v for k, v in stats.items() if not isinstance(v, (list, dict))
                },
            },
        )
        save_case(
            case_dir,
            manifest,
            xct_u8,
            label_u8,
            pore_logit=pore_logit,
            class_probs=probs,
            requested_field=tile_field,
            requested_material=material_map if voxel_material is not None else None,
        )
        logger.info(
            "%s/%s  phi=%.4f air=%.4f  %.1f min",
            spec.assessment, spec.name,
            float((label_u8 == 1).mean()), float((label_u8 == 2).mean()), wall / 60.0,
        )
        return manifest

    # -- request construction ----------------------------------------------

    def _porosity_request(self, spec: CaseSpec):
        """(tile field, per-tile dict) for the sampler.

        The dict is always fully populated - one entry per tile - so the
        sampler never falls back to its own default for a tile the case forgot.
        """
        grid = spec.tile_grid
        if spec.field_fn is None:
            if spec.target_phi is None:
                raise ValueError(f"{spec.name}: neither a target nor a field was requested.")
            value = float(spec.target_phi)
            field = None
        else:
            field = np.asarray(spec.field_fn(grid, spec.seed), np.float32)
            if field.shape != grid:
                raise ValueError(
                    f"{spec.name}: the field builder returned {field.shape}, expected "
                    f"the tile grid {grid}."
                )
        por_map = {}
        for iz in range(grid[0]):
            for iy in range(grid[1]):
                for ix in range(grid[2]):
                    por_map[(iz, iy, ix)] = (
                        float(field[iz, iy, ix]) if field is not None else value
                    )
        return field, por_map

    def _material_request(self, spec: CaseSpec):
        """(voxel envelope, latent-cell envelope) or ``(None, None)`` for 'full'."""
        if spec.material_fn is None:
            return None, None
        voxel = np.asarray(spec.material_fn(spec.volume_shape), bool)
        if voxel.shape != tuple(spec.volume_shape):
            raise ValueError(
                f"{spec.name}: the material builder returned {voxel.shape}, expected "
                f"{tuple(spec.volume_shape)}."
            )
        return voxel, latent_material_map(voxel)

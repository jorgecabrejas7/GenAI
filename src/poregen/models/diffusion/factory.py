"""One place that turns ``model.type`` into a denoiser.

Three call sites build a denoiser from a config — training, eval_v4 generation
and `scripts/generate_volumes.py` — and each one hard-coded `UNet3DDenoiser`.
A baseline that is a DIFFERENT denoiser on the same trunk therefore had to be
threaded through all three by hand, which is three chances for a generation run
to build a different network than the one that was trained. The config already
records the answer; this reads it.
"""

from __future__ import annotations

from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser

#: ``model.type`` -> the class. ``unet3d`` is ldm06 and every run before it.
DENOISERS: dict[str, type] = {"unet3d": UNet3DDenoiser}


def _phi_only() -> type:
    from poregen.baselines.ldm_phi_only.networks import PhiOnlyDenoiser  # noqa: PLC0415

    return PhiOnlyDenoiser


#: Imported lazily: a baseline should not be imported by every training run.
LAZY: dict[str, callable] = {"unet3d_phi_only": _phi_only}


def build_denoiser(cfg: dict):
    """The denoiser this config asks for, on the CPU.

    Refuses an unknown type rather than falling back to the production network:
    a typo that silently trained ldm06's architecture under a baseline's name
    would produce a result that cannot be told from a real one.
    """
    name = str((cfg.get("model") or {}).get("type", "unet3d"))
    if name in DENOISERS:
        cls = DENOISERS[name]
    elif name in LAZY:
        cls = LAZY[name]()
    else:
        raise KeyError(
            f"unknown model.type {name!r}; choose from "
            f"{sorted([*DENOISERS, *LAZY])}"
        )
    return cls(UNet3DConfig.from_cfg(cfg))

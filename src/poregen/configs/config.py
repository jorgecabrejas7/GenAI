"""Configuration loading utilities for PoreGen.

Provides :func:`load_config` (returns a plain dict, backward-compatible with
all callers) and typed :class:`PoreGenConfig` dataclasses for IDE support and
runtime validation.  Use :func:`parse_config` to convert a loaded dict into
typed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Typed config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    in_channels: int = 2
    z_channels: int = 8
    base_channels: int = 32
    n_blocks: int = 2
    patch_size: int = 64

    def __post_init__(self) -> None:
        if self.z_channels < 1:
            raise ValueError(f"z_channels must be ≥ 1, got {self.z_channels}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be ≥ 1, got {self.n_blocks}")
        if self.patch_size < 8:
            raise ValueError(f"patch_size must be ≥ 8, got {self.patch_size}")


@dataclass
class LossConfig:
    xct_loss_type: str = "charbonnier"
    xct_weight: float = 1.0
    mask_bce_weight: float = 1.0
    mask_bce_pos_weight: float = 17.16
    mask_dice_weight: float = 1.0
    use_tversky: bool = False
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    kl_free_bits: float = 0.0
    kl_warmup_steps: int = 0
    kl_max_beta: float = 0.05

    def __post_init__(self) -> None:
        valid = {"l1", "mse", "charbonnier"}
        if self.xct_loss_type not in valid:
            raise ValueError(
                f"xct_loss_type must be one of {valid}, got {self.xct_loss_type!r}"
            )
        if not (0.0 < self.tversky_alpha + self.tversky_beta <= 2.0):
            raise ValueError(
                "tversky_alpha + tversky_beta must be in (0, 2], "
                f"got {self.tversky_alpha} + {self.tversky_beta}"
            )


@dataclass
class TrainingConfig:
    seed: int = 42
    lr: float = 2e-4
    weight_decay: float = 0.01
    total_steps: int = 71690
    max_grad_norm: float | None = None
    scheduler: str = "none"
    lr_min: float = 2e-5
    warmup_steps: int = 0
    log_every: int = 1
    eval_every: int = 62
    val_batches: int = 20
    test_every: int = 625
    test_batches: int = 20
    save_every: int = 1000
    image_log_every: int = 62
    montecarlo_every: int = 100
    montecarlo_batch_size: int = 8
    final_full_eval: bool = True
    sample_every: int = 12500
    n_patch_samples: int = 8
    compile: bool = False
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.scheduler not in {"none", "cosine"}:
            raise ValueError(
                f"scheduler must be 'none' or 'cosine', got {self.scheduler!r}"
            )
        if self.total_steps < 1:
            raise ValueError(f"total_steps must be ≥ 1, got {self.total_steps}")


@dataclass
class DataConfig:
    dataset_root: str = "split_v1"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    timeout: int = 0
    split_version: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be ≥ 0, got {self.num_workers}")
        if self.timeout < 0:
            raise ValueError(f"timeout must be ≥ 0, got {self.timeout}")
        if self.split_version is not None and self.split_version not in {"v1", "v2"}:
            raise ValueError(
                f"split_version must be 'v1' or 'v2', got {self.split_version!r}"
            )


@dataclass
class PoreGenConfig:
    """Fully typed representation of a PoreGen YAML config."""

    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    data: DataConfig

    def __post_init__(self) -> None:
        # Cross-section consistency: dataset_root must match split_version when both set
        sv = self.data.split_version
        dr = self.data.dataset_root
        if sv is not None and dr != f"split_{sv}":
            raise ValueError(
                f"data.split_version={sv!r} expects dataset_root='split_{sv}', "
                f"got {dr!r}"
            )


def parse_config(cfg: dict[str, Any]) -> PoreGenConfig:
    """Convert a loaded config dict into a validated :class:`PoreGenConfig`.

    Callers that want IDE auto-complete and ``__post_init__`` validation can
    use this after :func:`load_config`::

        raw  = load_config("src/poregen/configs/vae_default.yaml")
        typed = parse_config(raw)
        print(typed.model.z_channels)   # int, not Any
    """
    return PoreGenConfig(
        model=ModelConfig(**cfg["model"]),
        loss=LossConfig(**cfg["loss"]),
        training=TrainingConfig(**cfg["training"]),
        data=DataConfig(**cfg["data"]),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_data_config(cfg: dict[str, Any]) -> None:
    """Keep ``split_version`` and ``dataset_root`` consistent for old/new callers."""
    data = cfg.get("data")
    if not isinstance(data, dict):
        return

    split_version = data.get("split_version")
    dataset_root = data.get("dataset_root")

    inferred_version: str | None = None
    if isinstance(dataset_root, str):
        root = dataset_root.strip().lower()
        if root in {"split_v1", "split_v2"}:
            inferred_version = root.removeprefix("split_")

    if split_version is None:
        if inferred_version is not None:
            data["split_version"] = inferred_version
        return

    version = str(split_version).strip().lower()
    if version not in {"v1", "v2"}:
        raise ValueError(
            f"Unsupported data.split_version '{split_version}'. Expected 'v1' or 'v2'."
        )

    expected_root = f"split_{version}"
    if dataset_root is not None and dataset_root != expected_root:
        raise ValueError(
            "Config mismatch: data.split_version="
            f"{version!r} expects data.dataset_root={expected_root!r}, "
            f"found {dataset_root!r}."
        )

    data["split_version"] = version
    data.setdefault("dataset_root", expected_root)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Load a YAML config file and apply optional dot-notation overrides.

    Parameters
    ----------
    path : str | Path
        Path to a YAML config file (e.g. ``vae_default.yaml``).
    **overrides : Any
        Dot-notation key=value pairs to override after loading.
        Example::

            cfg = load_config("vae_default.yaml",
                              **{"loss.kl_max_beta": 0.01,
                                 "training.total_steps": 50000})

    Returns
    -------
    dict
        Nested config dict. Top-level keys are the YAML sections
        (``model``, ``loss``, ``training``, ``data``).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If an override key refers to a section or field that does not
        exist in the loaded config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        if len(parts) != 2:
            raise KeyError(
                f"Override key '{dotkey}' must be in 'section.field' format."
            )
        section, fname = parts
        if section not in cfg:
            raise KeyError(f"Config section '{section}' not found.")
        if fname not in cfg[section]:
            raise KeyError(
                f"Field '{fname}' not found in config section '{section}'."
            )
        cfg[section][fname] = value

    _normalise_data_config(cfg)
    return cfg

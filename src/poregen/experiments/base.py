"""Base experiment runtime — shared loading and data-access utilities.

Every experiment subclasses :class:`ExperimentRuntime` and implements
:meth:`_build_model` to specify how the model is constructed from a config
dict.  :meth:`from_checkpoint` then provides a uniform one-call factory:

    runtime = MyRuntime.from_checkpoint(
        "runs/vae/r05/last.ckpt",
        config_path="src/poregen/configs/r05.yaml",
    )

The :func:`build_patch_loader` utility is also re-exported here so that
experiment modules can expose a single import surface to notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from poregen.configs.config import load_config
from poregen.dataset.loader import PatchDataset
from poregen.training import load_checkpoint, select_device


# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) until the ``src/poregen`` tree is found."""
    current = Path(start or ".").resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "poregen").exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer the repo root from the current working directory."
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_patch_loader(
    cfg: dict[str, Any],
    data_root: str | Path,
    split: str,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a :class:`~poregen.dataset.loader.PatchDataset` / DataLoader pair.

    All arguments fall back to the values in ``cfg["data"]`` when not provided,
    so notebook cells only need to specify what they want to override.
    """
    data_cfg = cfg.get("data", {})
    root = Path(data_root)
    dataset = PatchDataset(root / "patch_index.parquet", root, split=split)

    effective_workers = (
        num_workers if num_workers is not None
        else int(data_cfg.get("num_workers", 0))
    )
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size or int(data_cfg.get("batch_size", 128)),
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=bool(
            data_cfg.get("pin_memory", False) if pin_memory is None else pin_memory
        ),
        drop_last=drop_last,
    )
    if effective_workers > 0 and data_cfg.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])

    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Base runtime dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRuntime:
    """Loaded model plus the resolved config / data context for one experiment.

    Subclasses implement :meth:`_build_model` to specify model construction;
    :meth:`from_checkpoint` handles all the common boilerplate.
    """

    model: nn.Module
    cfg: dict[str, Any]
    device: torch.device
    checkpoint_step: int
    checkpoint_meta: dict[str, Any]
    data_root: Path
    repo_root: Path

    # ------------------------------------------------------------------
    # DataLoader convenience
    # ------------------------------------------------------------------

    def build_loader(
        self,
        split: str,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        drop_last: bool = False,
    ) -> DataLoader:
        """Build a DataLoader for *split* using this runtime's config and data root."""
        return build_patch_loader(
            self.cfg,
            self.data_root,
            split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    # ------------------------------------------------------------------
    # Subclass extension point
    # ------------------------------------------------------------------

    @classmethod
    def _build_model(cls, cfg: dict[str, Any]) -> nn.Module:
        """Construct and return an *uninitialised* model from *cfg*.

        Subclasses **must** override this method.  The returned model should
        not yet be moved to a device — :meth:`from_checkpoint` handles that.
        """
        raise NotImplementedError(
            f"{cls.__name__}._build_model is not implemented."
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config_path: str | Path,
        data_root: str | Path | None = None,
        device: torch.device | None = None,
        repo_root: str | Path | None = None,
    ) -> "ExperimentRuntime":
        """Load checkpoint and construct a runtime in one call.

        Parameters
        ----------
        checkpoint_path :
            Path to a ``.ckpt`` file (relative paths are resolved from the
            inferred repo root).
        config_path :
            Path to the experiment YAML (relative paths likewise resolved).
        data_root :
            Override the dataset directory.  Defaults to
            ``<repo>/data/<cfg["data"]["dataset_root"]>``.
        device :
            Target device.  Falls back to :func:`~poregen.training.select_device`.
        repo_root :
            Override repo-root discovery.  Useful in unusual directory layouts.
        """
        repo = find_repo_root(repo_root)

        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = repo / cfg_path

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (repo / ckpt_path).resolve()

        cfg = load_config(cfg_path)
        runtime_device = device or select_device()
        model = cls._build_model(cfg)
        step, meta = load_checkpoint(
            ckpt_path,
            model=model,
            restore_rng=False,
            map_location=runtime_device,
        )
        model = model.to(runtime_device).eval()

        resolved_data_root = (
            Path(data_root)
            if data_root is not None
            else repo / "data" / cfg["data"].get("dataset_root", "split_v1")
        ).resolve()

        return cls(
            model=model,
            cfg=cfg,
            device=runtime_device,
            checkpoint_step=step,
            checkpoint_meta=meta,
            data_root=resolved_data_root,
            repo_root=repo,
        )

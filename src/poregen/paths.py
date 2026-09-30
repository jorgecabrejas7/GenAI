"""ONE place that decides which dataset split everything reads.

The project has been rebuilt from ``split_v1`` to ``split_v2`` to ``split_v3``,
and is about to be rebuilt again as ``split_v4``. Each time, the split
name was written into configs, script defaults, analysis constants and test
fixtures separately, so pointing the pipeline at a new build meant editing
dozens of files and hoping none was missed — and a missed one does not fail,
it quietly measures the old data.

So the name lives here, and everything that needs it asks.

    from poregen.paths import default_split, data_root

    root = data_root()                  # <repo>/data/split_v3 by default
    root = data_root("split_v4")  # an explicit split always wins

Set ``POREGEN_SPLIT`` to redirect the whole chain at once:

    POREGEN_SPLIT=split_v4 python scripts/train_vae.py run r08/base

THE ENVIRONMENT VARIABLE OVERRIDES A CONFIG'S OWN ``data.dataset_root``, and
:func:`poregen.configuration.experiments.resolve_experiment` says so in the
log when it does. That is deliberate: a switch that every config could silently
opt out of would not be a switch. An explicit argument in code still wins over
both, because a caller that names a split means it.

The builds, and the reference segmentation each came from — the name alone does
not say, and two builds of the same scans can differ only in their labels:

    split_v3         the split every published number was measured on
    split_v4         reference set r30_k0.125_min8  (onlypores.ipynb cell 6)
    split_v5         reference set r15_k0.2_min8    (onlypores_batch.ipynb cell 5)

``DEFAULT_SPLIT`` is NOT changed when a new dataset is built. It stays
``split_v3`` until the author moves it, so a rerun of a published number
reproduces that number rather than silently re-measuring on new data.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The split every published number in this repository was measured on. Do not
#: change it to point at a newer build; pass the new one, or set POREGEN_SPLIT.
DEFAULT_SPLIT = "split_v3"

#: The environment variable that redirects the whole pipeline.
SPLIT_ENV = "POREGEN_SPLIT"


def repo_root() -> Path:
    """The repository root, from this file's location."""
    return Path(__file__).resolve().parents[2]


def default_split() -> str:
    """The split name to use when a caller names none."""
    return os.environ.get(SPLIT_ENV, "").strip() or DEFAULT_SPLIT


def split_override() -> str | None:
    """The split forced by the environment, or None when nothing is forced."""
    value = os.environ.get(SPLIT_ENV, "").strip()
    return value or None


def data_root(split: str | None = None, *, repo: str | Path | None = None) -> Path:
    """``<repo>/data/<split>``, with the split resolved as documented above."""
    base = Path(repo) if repo is not None else repo_root()
    return base / "data" / (split or default_split())


def latents_root(store: str, split: str | None = None,
                 *, repo: str | Path | None = None) -> Path:
    """``<repo>/data/<split>/<store>`` — a latent store under the same split."""
    return data_root(split, repo=repo) / store


#: The environment variable that points every LDM config at a different frozen
#: VAE. It exists for the same reason POREGEN_SPLIT does, one level down: an
#: LDM is tied to the VAE whose latents it trains on, and train_ldm REFUSES to
#: start when cfg['vae']['checkpoint'] differs from the store's own recorded
#: encoder. On split_v3 the bring-up script met that by editing and committing
#: ldm06/base.yaml. On a rebuild the config files are not edited, so the chain
#: names the new VAE here once it exists.
VAE_ENV = "POREGEN_VAE_CHECKPOINT"


def vae_checkpoint_override() -> str | None:
    """The VAE checkpoint forced by the environment, or None."""
    value = os.environ.get(VAE_ENV, "").strip()
    return value or None


"""Evaluation suite v4 - one documented, rerunnable measurement of a generated volume.

Three rules hold the suite together:

1. **Every generated volume carries a manifest.**  What model made it, at what
   step, with which weights and sampler settings, and what was asked of it.
2. **Every metric declares the manifest fields it reads and refuses a volume
   that does not carry them.**  A metric cannot be pointed at a volume it does
   not describe, and a request cannot be inferred from a directory name.
3. **Real test volumes go through the metrics first, and their values are the
   floor row of every table.**  A number is only good or bad against what real
   material scores on the same measurement.

Stages, one CLI subcommand each: ``generate`` (GPU, hours), ``measure``
(CPU, repeatable from the volumes alone), ``report`` (results file only),
``manifest-check`` and ``real-floor``.

The seven assessments live in :mod:`poregen.eval_v4.cases` as data; the only
module that knows the sampler API is :mod:`poregen.eval_v4.generate`.
"""

from poregen.eval_v4.io import Case, load_cases, save_case
from poregen.eval_v4.manifest import Manifest, ManifestError, requires

__all__ = [
    "Case",
    "Manifest",
    "ManifestError",
    "load_cases",
    "requires",
    "save_case",
]

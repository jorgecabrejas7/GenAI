#!/usr/bin/env python
"""Campaign 36: the eval_v4 memorisation search on the whole-coupon twins. (GPU.)

Same search as campaign 18's (exhaustive nearest / second-nearest against the
stride-64 TRAIN rows of the twins' own latent store, in latent and decoded-grey
space, Favero ratio, threshold 1/3, floored by held-out val patches), with the
per-volume tile cap lifted: a whole twin holds ~5 600 tiles where campaign 18's
volumes held at most 256. Must run with the card idle (the search refuses a
busy GPU). Author 2026-10-28: run it.

    python scripts/analysis/c36_memorisation.py runs/campaigns/36-pore-morphology/gen_a2s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from poregen.eval_v4.memorisation import memorisation

root = Path(sys.argv[1])
res = memorisation(root, assessments=("twins",), max_tiles=10_000)
out = root / "twins_memorisation.json"
out.write_text(json.dumps(res, indent=2, default=str))
print("wrote", out)

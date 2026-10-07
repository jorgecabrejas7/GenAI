#!/usr/bin/env python
"""Give JI_7, JI_8 and JI_11 the stacking sequence every other JI coupon has.

``data/layup_ground_truth.json`` (gitignored, like all of data/) is the expert
table. It gives JI_4, JI_5, JI_10 and JI_12 one identical record — sequence
"B", 16 plies, im7_m56, 0.25 mm — and has JI_7 and JI_8 as "not in expert
table" and no entry for JI_11. The author's word (2026-10-07, relayed by the
supervisor): every JI coupon shares that sequence.

This script is the record of that edit: it refuses unless the four known JI
records are identical, copies that record to the three, adds the note, and
is a no-op on a second run.

    python scripts/set_ji_stacking_sequence.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "data" / "layup_ground_truth.json"
PREFIX = "MedidasDB__Juan_Ignacio_probetas_"
KNOWN = ("4_volume_eq_rotated_aligned", "5_volume_eq_rotated_aligned",
         "10_volume_eq_aligned", "12_volume_eq_aligned")
TARGETS = ("7_volume_eq_aligned", "8_volume_eq_aligned", "11_volume_eq_aligned")
NOTE = "author 2026-10-07: same stacking sequence as every JI coupon"


def main() -> int:
    gt = json.loads(GT.read_text())
    vols = gt["volumes"]
    known = [vols[PREFIX + k] for k in KNOWN]
    if any(r != known[0] for r in known[1:]):
        raise SystemExit("the four known JI records differ; nothing written")
    ref = known[0]
    changed = []
    for t in TARGETS:
        rec = {**ref, "note": NOTE}
        if vols.get(PREFIX + t) != rec:
            vols[PREFIX + t] = rec
            changed.append(t)
    if changed:
        GT.write_text(json.dumps(gt, indent=2))
    print(f"reference record: {ref}")
    print(f"written: {changed or 'nothing (already set)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

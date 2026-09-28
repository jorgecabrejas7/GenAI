#!/usr/bin/env python
"""Shorten a denoising film by thinning ONE stage, without re-rendering. (CPU.)

The four films are expensive — the 1024-wide one is 65 minutes on the card —
so a length change must not mean generating it again. Every stage is separated
by a title card, and a card is the same frame repeated, so the stage boundaries
can be read straight off the file: a run of identical frames longer than
``--card-min`` is a card.

Only the named stage is thinned. Title cards, and every other stage, are copied
through frame for frame, so the film keeps its structure and the other stages
keep their intended rates.

Usage
-----
    python scripts/analysis/video_short_cut.py --in <film>.mp4 --stage 1 --keep 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np


def signatures(path: Path) -> list[tuple]:
    """A cheap fingerprint per frame: enough to tell a repeat from a change."""
    sig = []
    with imageio.get_reader(str(path)) as r:
        for f in r:
            small = f[::32, ::32].astype(np.int32)
            sig.append((int(small.sum()), int(small[::4, ::4].sum())))
    return sig


def card_runs(sig: list[tuple], card_min: int) -> list[tuple[int, int]]:
    """Start and end of every run of identical frames at least card_min long."""
    runs, i, n = [], 0, len(sig)
    while i < n:
        j = i
        while j + 1 < n and sig[j + 1] == sig[i]:
            j += 1
        if j - i + 1 >= card_min:
            runs.append((i, j))
        i = j + 1
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stage", type=int, default=1, help="1-based stage to thin")
    ap.add_argument("--keep", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--card-min", type=int, default=30)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    src = a.src.resolve()
    out = a.out or src.with_name(src.stem + "_short.mp4")

    sig = signatures(src)
    cards = card_runs(sig, a.card_min)
    if not cards:
        raise RuntimeError(f"{src.name}: no title cards found, so the stages "
                           "cannot be located; refusing to guess at boundaries")
    # Stage k runs from the end of card k to the start of card k+1 (or the end).
    bounds = []
    for k, (_, end) in enumerate(cards):
        start = end + 1
        stop = cards[k + 1][0] if k + 1 < len(cards) else len(sig)
        bounds.append((start, stop))
    print(f"{len(cards)} title cards, stages at "
          + ", ".join(f"{s}-{e}" for s, e in bounds))
    if not 1 <= a.stage <= len(bounds):
        raise RuntimeError(f"stage {a.stage} does not exist in this film")
    lo, hi = bounds[a.stage - 1]

    kept = 0
    with imageio.get_reader(str(src)) as r, \
         imageio.get_writer(str(out), fps=a.fps, quality=8, format="FFMPEG") as w:
        for i, f in enumerate(r):
            if lo <= i < hi and (i - lo) % a.keep:
                continue
            w.append_data(f)
            kept += 1
    print(f"wrote {out}  ({kept} of {len(sig)} frames, {kept / a.fps:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

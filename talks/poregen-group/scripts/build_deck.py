"""Bundle the deck: inline the fixed-stage CSS and inject the numbers.

Reads deck/deck.src.html and deck/media/numbers.json (from extract_numbers.py)
plus the profile/tile arrays from render_media.py, and writes deck/index.html,
which must stay beside deck/media/ (videos and images are linked, not inlined).

Usage: python talks/poregen-group/scripts/build_deck.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DECK, MEDIA = HERE / "deck", HERE / "deck" / "media"
BASE_CSS = DECK / "viewport-base.css"                              # frontend-slides fixed-stage base, vendored

num = json.loads((MEDIA / "numbers.json").read_text())
for name in ("seam_production", "seam_overlapped"):
    profs = np.load(MEDIA / f"{name}_profiles.npy")                  # pore fraction per y and per x position
    # fold onto the distance from the nearest chunk plane (planes every 192 voxels),
    # averaging the 5 interior planes of both axes; 4-voxel bins
    d = np.arange(-96, 96)
    folded = np.mean([p[c + d] for p in profs for c in range(192, 1024, 192) if c + 95 < len(p)], axis=0) / profs.mean()
    folded = folded.reshape(-1, 4).mean(1)
    num[f"{name}_prof"] = {"value": [round(float(x), 4) for x in folded],
                           "source": f"render_media.seam_maps: {name}, seed 101, pore fraction vs distance from the chunk plane, y and x planes pooled, 4-voxel bins, ÷ volume mean"}
for name in ("halves", "checkerboard"):
    t = np.load(MEDIA / f"{name}_tiles.npy")                          # (requested, delivered) x 3x3x3 tiles
    num[f"{name}_tiles"] = {"value": [t[0][1].round(4).tolist(), t[1][1].round(4).tolist()],
                            "source": f"render_media.local_field: {name} seed 101, middle z layer of 3x3x3 tiles"}

src = (DECK / "deck.src.html").read_text()
src = src.replace("/*INLINE:viewport-base.css*/", BASE_CSS.read_text())
src = src.replace("/*NUMBERS*/{}", json.dumps(num, separators=(",", ":")))
(DECK / "index.html").write_text(src)
print(f"deck/index.html  {len(src) // 1024} KiB  {len(num)} numbers")

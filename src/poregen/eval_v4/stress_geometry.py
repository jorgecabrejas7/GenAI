"""Requested shapes the training material never contained. (exploratory)

Every eval-v4 geometry so far has been a plate with something removed: a notch,
a drilled hole, a sphere.  The training coupons ARE plates, so those requests
sit a short distance off the manifold.  These do not: a tube, an L-bracket, a
taper, a gyroid, a hollow shell, two separated coupons, cut-through letters.

**This assessment is off the gates and says so in its own output.**  A model
asked for a shape nothing in its training resembles may fail in ways that mean
nothing about the material it was trained to make, and a number from here must
not be read as a defect of the kind the gated assessments report.  What it is
for is the boundary: which requests the specimen-envelope conditioning honours
at all, and where it stops.

The builders are here rather than in `cases.py` because they are long, purely
geometric, and testable on their own — each one has a closed-form volume or
area that a test can check it against, which is the only way to know a voxel
mask is the shape it claims to be.
"""

from __future__ import annotations

import numpy as np

#: 25 micron voxels, as everywhere in this project.
VOXEL_MM = 0.025


def _grid(shape: tuple[int, int, int]):
    """Centred coordinate grids, one per axis, broadcastable against `shape`."""
    d, h, w = shape
    z = (np.arange(d) - (d - 1) / 2.0)[:, None, None]
    y = (np.arange(h) - (h - 1) / 2.0)[None, :, None]
    x = (np.arange(w) - (w - 1) / 2.0)[None, None, :]
    return z, y, x


def material_tube(shape, *, outer: float = 200.0, wall: float = 120.0,
                  axis: int = 2) -> np.ndarray:
    """A hollow cylinder: material in the wall, air in the bore and outside.

    THE AXIS IS X, on a (448, 448, 1024) canvas, and the choice is the test.
    `theta_deg` varies along z, so a tube lying along x is cut by plies through
    448 voxels of depth — about 23 plies across its own diameter, which is how
    a real curved part is laid up.  Running it along z instead would have put
    the plies across the wall and measured something else.
    """
    if wall >= outer:
        raise ValueError(f"wall {wall} must be less than outer radius {outer}")
    g = _grid(shape)
    radial = [g[a] for a in range(3) if a != axis]
    r2 = radial[0] ** 2 + radial[1] ** 2
    cross = [shape[a] for a in range(3) if a != axis]
    if 2 * outer >= min(cross):
        raise ValueError(
            f"a tube of outer radius {outer} does not fit in cross-section "
            f"{cross} with a margin"
        )
    # r2 spans only the two radial axes; broadcast it back along the tube's own
    # axis, or the mask comes out one voxel thick there.
    wall_mask = (r2 <= outer ** 2) & (r2 >= (outer - wall) ** 2)
    return np.broadcast_to(wall_mask, shape).copy()


def material_l_bracket(shape, *, leg: float = 512.0, thick: float = 192.0,
                       inner_radius: float = 96.0,
                       outer_radius: float = 288.0) -> np.ndarray:
    """Two legs meeting at a filleted right angle, extruded along x.

    The corner is the point: a laminate's plies cannot follow a right angle,
    so a real bracket is laid up over a radius.  The request carries both — an
    inner fillet and an outer round — and the layup readers are run on the two
    legs SEPARATELY, because no ply orientation is defined through the corner.
    """
    d, h, w = shape
    zz = np.arange(d)[:, None]
    yy = np.arange(h)[None, :]
    # Legs from the corner at (0, 0): one along z, one along y.
    leg_z = (zz < thick) & (yy < leg)
    leg_y = (yy < thick) & (zz < leg)
    mask2d = leg_z | leg_y
    # Outer round on the convex corner, inner fillet on the concave one.
    cz, cy = outer_radius, outer_radius
    outer_ok = ((zz - cz) ** 2 + (yy - cy) ** 2 <= outer_radius ** 2) | \
               (zz >= cz) | (yy >= cy)
    mask2d &= outer_ok
    fz, fy = thick + inner_radius, thick + inner_radius
    in_corner = (zz > thick) & (yy > thick)
    fillet = (zz - fz) ** 2 + (yy - fy) ** 2 >= inner_radius ** 2
    mask2d &= ~in_corner | fillet
    return np.broadcast_to(mask2d[:, :, None], (d, h, w)).copy()


def material_taper(shape, *, t0: float = 192.0, t1: float = 80.0) -> np.ndarray:
    """A plate whose thickness falls linearly along y, centred on z.

    A real ply drop-off: the specimen thins, so the number of plies through
    the thickness falls along the length.  The model has only ever seen
    parallel-sided coupons.
    """
    d, h, w = shape
    t = np.linspace(t0, t1, h)[None, :, None]
    zz = (np.arange(d) - (d - 1) / 2.0)[:, None, None]
    return np.broadcast_to(np.abs(zz) <= t / 2.0, (d, h, w)).copy()


def material_two_coupons(shape, *, gap: float = 128.0) -> np.ndarray:
    """Two plates with an air gap along x — one request, two disjoint bodies.

    Nothing in training is disconnected, and the sampler has no notion of a
    component: the gap is simply requested air. Whether the two halves are
    generated as independent material or bleed into each other across the gap
    is the question.
    """
    d, h, w = shape
    half = (w - gap) / 2.0
    xx = np.arange(w)[None, None, :]
    m = (xx < half) | (xx >= half + gap)
    return np.broadcast_to(m, (d, h, w)).copy()


def material_hollow_sphere(shape, *, outer: float = 224.0,
                           shell: float = 80.0) -> np.ndarray:
    """A spherical shell: air inside AND outside, material only in between.

    The solid sphere of assessment 7 has air outside only.  This adds an
    enclosed interior void, which no coupon has and which the six face
    distances cannot describe — every shell voxel is near a surface on two
    sides.
    """
    z, y, x = _grid(shape)
    r2 = z ** 2 + y ** 2 + x ** 2
    if 2 * outer >= min(shape):
        raise ValueError(f"radius {outer} does not fit in {shape} with a margin")
    return (r2 <= outer ** 2) & (r2 >= (outer - shell) ** 2)


#: A gyroid's strut cannot be an arbitrary fraction of its period.  The brief
#: asked for a 2 mm strut (80 voxels) at a 256-voxel period; measured, that is
#: 99.6 % material — 31 % of the period on each side of the surface, so the
#: walls meet and what is left is a solid with pinholes, not a gyroid.  The
#: 2 mm strut is the physically meaningful half of the request (the tube wall
#: is 3 mm), so the PERIOD is doubled to keep it: 80 voxels at period 512 is
#: 48 % material, the canonical near-half-dense gyroid.
GYROID_PERIOD_VOX = 512.0
GYROID_STRUT_VOX = 80.0


def material_gyroid(shape, *, period: float = GYROID_PERIOD_VOX,
                    thickness: float = GYROID_STRUT_VOX) -> np.ndarray:
    """A gyroid minimal surface thickened into struts.

    ``|sin x cos y + sin y cos z + sin z cos x| <= level`` — triply periodic,
    no flat face anywhere, connected in every direction, and the first request
    with no outer surface at all: every face of the canvas cuts through
    material.

    `thickness` is honoured in VOXELS rather than being a level-set constant
    with an arbitrary scale.  The slab between the two level sets is
    ``2 * level / |grad f|`` thick, and ``|grad f|`` is measured on this
    volume's own surface rather than assumed, so the strut comes out the width
    it was asked for.  See :data:`GYROID_PERIOD_VOX` for why the period is not
    the one first proposed.
    """
    d, h, w = shape
    k = 2.0 * np.pi / period
    z = (np.arange(d) * k)[:, None, None]
    y = (np.arange(h) * k)[None, :, None]
    x = (np.arange(w) * k)[None, None, :]
    f = (np.sin(z) * np.cos(y) + np.sin(y) * np.cos(x) + np.sin(x) * np.cos(z))
    near = np.abs(f) < 0.05
    if not near.any():
        raise ValueError(f"no gyroid surface inside {shape} at period {period}")
    grad = np.gradient(f)
    gmag = np.sqrt(sum(g ** 2 for g in grad))
    level = float(np.median(gmag[near])) * (thickness / 2.0)
    return np.abs(f) <= level


def material_letters(shape, text: str = "PoreGen", *,
                     height: float = 256.0) -> np.ndarray:
    """A plate with `text` cut through it as air.

    Pure controllability and nothing else: the shape carries no mechanical
    meaning, so if the model can hold it the specimen-envelope conditioning is
    doing what it claims, and if it cannot the failure is unambiguous.
    """
    from PIL import Image, ImageDraw, ImageFont

    d, h, w = shape
    img = Image.new("L", (int(w), int(h)), 0)
    draw = ImageDraw.Draw(img)
    font = None
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(path, int(height))
            break
        except OSError:
            continue
    # "PoreGen" at the requested 256 is 1229 voxels wide on a 1024 canvas: it
    # would clip the last letter and test a word that is not the word.  WRAP
    # rather than shrink — the glyph height is the feature size the request is
    # about, so it is the thing to preserve; the line count is not.
    lines = [text]
    if font is not None:
        margin = 0.92
        while len(lines[0]) > 1 and any(
                (draw.textbbox((0, 0), ln, font=font)[2]
                 - draw.textbbox((0, 0), ln, font=font)[0]) > margin * w
                for ln in lines):
            cut = max(1, len(text) * len(lines) // (len(lines) + 1))
            lines = [text[:cut], text[cut:]] if len(lines) == 1 else lines
            break
        if len(lines) > 1 and sum(
                draw.textbbox((0, 0), ln, font=font)[3] for ln in lines) > margin * h:
            lines = [text]
    if font is None:
        # The default bitmap font cannot be scaled, so draw small and enlarge:
        # a blocky glyph is still a glyph, and the request is a shape, not a
        # typeface.
        small = Image.new("L", (int(w) // 8, int(h) // 8), 0)
        ImageDraw.Draw(small).text((4, 4), text, fill=255,
                                   font=ImageFont.load_default())
        img = small.resize((int(w), int(h)), Image.NEAREST)
    else:
        boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        hs = [b[3] - b[1] for b in boxes]
        total = sum(hs) + int(0.15 * height) * (len(lines) - 1)
        y0 = (h - total) / 2
        for ln, b, bh in zip(lines, boxes, hs):
            draw.text(((w - (b[2] - b[0])) / 2 - b[0], y0 - b[1]),
                      ln, fill=255, font=font)
            y0 += bh + int(0.15 * height)
    # PIL indexes (row, col) = (y, x), which is already the (h, w) of the
    # canvas slice, so no transpose.
    cut = np.asarray(img, dtype=np.uint8) > 127
    return np.broadcast_to(~cut[None, :, :], (d, h, w)).copy()


def field_ramp_and_spots(tile_grid: tuple[int, int, int], seed: int, *,
                         lo: float = 0.005, hi: float = 0.08,
                         peak: float = 0.10, sigma_vox: float = 96.0,
                         tile: int = 64) -> np.ndarray:
    """Requested porosity per 64-voxel tile: a ramp along y plus two spots.

    The ramp spans a factor of sixteen, which is wider than any single training
    volume shows, and the spots sit at 0.10 — just under the 0.107 conditioning
    clamp, so the request is extreme but still inside what `cond_por` can
    express. Anything above the clamp would be measuring the clamp.
    """
    nz, ny, nx = tile_grid
    y_c = (np.arange(ny) + 0.5) * tile
    ramp = lo + (hi - lo) * (y_c / (ny * tile))
    f = np.broadcast_to(ramp[None, :, None], tile_grid).astype(np.float64).copy()
    zz = (np.arange(nz) + 0.5)[:, None, None] * tile
    yy = (np.arange(ny) + 0.5)[None, :, None] * tile
    xx = (np.arange(nx) + 0.5)[None, None, :] * tile
    centres = [(nz * tile * 0.5, ny * tile * 0.3, nx * tile * 0.3),
               (nz * tile * 0.5, ny * tile * 0.7, nx * tile * 0.7)]
    for cz, cy, cx in centres:
        r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        f += (peak - lo) * np.exp(-r2 / (2.0 * sigma_vox ** 2))
    return np.clip(f, 1e-4, peak)

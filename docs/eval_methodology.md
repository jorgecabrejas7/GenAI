# Evaluation methodology — eval v4

How a generated volume is measured, and why each number means something. This is
the authoritative reference for the paper methods section and for anyone reading
a `runs/campaigns/<NN>-<name>/` result.

The suite is the package `src/poregen/eval_v4/`, driven by the `eval_v4` CLI.
It replaces `scripts/eval_generated_volumes.py`, which is **deleted**. That
script measured a generated set against a real set with distribution distances —
porosity W1, PSD W1, S₂(r) RMSE, Ripley K, FID on 2-D crops. Those answer "does
this look like the training data". They cannot answer the questions ldm06 is
built to answer, which are all *conditional*: did the volume deliver the porosity
it was **asked** for, in the cells it was asked for, with the layup it was asked
for, inside the material envelope it was asked for. A distribution distance has
no notion of a request, so it scores a model that ignores its conditioning
exactly as well as one that obeys it.

---

## The three rules

**1. Every generated volume carries a manifest.** `manifest.json` sits beside
`volume.tif` and `label.tif` and records what made the volume and what was asked
of it. Nothing is inferred from a directory name. Eval v3 recovered a volume's
requested porosity from its path, so renaming a directory changed the answer.

**2. Every metric declares the manifest fields it reads, and refuses a volume
that does not carry them** — or whose array shape contradicts the manifest.
The declaration is enforced by a decorator, so it cannot go stale:

```python
@requires("requested_global_phi")
def porosity_error(label, material, *, manifest): ...

porosity_error.requires        # ("requested_global_phi",)
```

Positional array arguments are volume-shaped and are checked against
`manifest.volume_shape`; an array on another grid (a tile-grid field, a second
volume) goes in by keyword and is checked by the metric against its own grid.

**3. Real test volumes go through the metrics first, and their values are the
floor row of every table.** A seam ratio of 0.96 is not "nearly perfect" — it is
what a real scan scores, and a generated volume that reaches it has nothing left
to fix. A cell-to-cell porosity spread of 0.003 is not "good local control" — it
is what real material does with no request at all.

---

## Stages

| Command | Needs | Produces |
|---|---|---|
| `eval_v4 real-floor --root <campaign>` | the dataset | `real_floor/` — run this first |
| `eval_v4 generate <assessment> --model <run_dir> --ckpt <step> --out <campaign>` | GPU, hours | `<assessment>/volumes/<case>/` |
| `eval_v4 measure <assessment> --root <campaign>` | the volumes only | `<assessment>/results.json` |
| `eval_v4 report --root <campaign>` | the results file only | `findings.md`, `figures/*.{pdf,png}` |
| `eval_v4 manifest-check --root <campaign>` | the manifests | a pass/fail listing; exit 1 on any fault |

The stages are separate because their costs are. Measurement must be repeatable
from the volumes alone — a metric that needed the model back could not be
re-run after a checkpoint moved — and the report must be rebuildable after the
volumes are deleted.

`generate --dry-run` lists every case and what it asks for, without a GPU.
`generate` skips a case whose `manifest.json` already exists, so an interrupted
run resumes.

### Campaign layout

```
runs/campaigns/<NN>-<name>/
    <assessment>/
        volumes/<case>/
            volume.tif              uint8, raw-scan grey
            label.tif               uint8, 0 material / 1 pore / 2 air
            probs.npz               pore log-odds (+ class probs when small)
            requested_field.npy     requested phi per 64-voxel tile
            requested_material.npy  requested envelope per latent cell
            manifest.json
        results.json
        findings.md
        figures/*.pdf, *.png        300 dpi
    real_floor/                     same shape; sampler = "real"
    README.md
```

`volume.tif` and `label.tif` go in and come out on their **native** scale. The
loader refuses anything that is not already `uint8` rather than guessing a
conversion: a silent rescale cost a whole evaluation campaign once
(`runs/campaigns/03-eval-v2-buggy-decode`).

The manifest is written **last**, so a half-written case has no manifest and
every reader refuses it instead of measuring a truncated volume.

---

## The manifest

| Field | Meaning |
|---|---|
| `assessment`, `case` | where the volume belongs |
| `sampler` | `hybrid_chunked` for a generated volume, `real` for a crop of a scan |
| `model_run`, `checkpoint_step`, `weights` | the LDM run directory, the step, `raw` or `ema` |
| `objective` | what the denoiser predicts — `eps` or `v` |
| `cfg_rescale` | guidance rescaling; 0 is plain classifier-free guidance |
| `ddim_steps`, `chunk_tiles`, `window_stride` | sampler geometry |
| `decode`, `decode_overlap` | `overlapped` with 32-voxel overlap, or `tiled` with 0 |
| `s_por`, `s_nb` | the two guidance scales |
| `seed` | the seed of the LOCAL generator every random draw of the reverse process is taken from; the global torch generator is left untouched |
| `requested_global_phi` | the uniform target, when there was one |
| `requested_field` | `.npy` of the requested phi per 64-voxel tile |
| `requested_layup`, `requested_ply_thickness_vox` | the stacking sequence and its pitch |
| `requested_material` | `"full"`, or the `.npy` of the envelope per latent cell |
| `region_offset`, `region_shape` | the sub-block the case is about (assessment 6) |
| `volume_shape`, `git_commit`, `wall_time_s`, `peak_gpu_memory_bytes` | what it is and what it cost |

A `hybrid_chunked` manifest must carry every generation field and at least one
porosity request; a `real` manifest carries none of them, so any metric that
needs a request refuses it. That is why the floor is reported for the
request-free metrics only.

Self-consistency the schema enforces: `decode` and `decode_overlap` cannot
contradict each other, `objective` must be `eps` or `v`, `region` must lie inside
the volume, an unknown field is an error rather than being ignored.

---

## Shared definitions

**material** — the requested specimen envelope at voxel resolution. The stored
map is the per-latent-cell envelope *fraction*, which is what `cond_material`
means; the voxel-level request is that map upsampled by 4 and thresholded at half
a cell. Every fraction is taken inside it, because a volume asked for a hole
should not be scored for the pores it does not put in the hole. For a real crop
the material is the crop's own `sample_mask`.

**interior** — more than **32 voxels** from every face. Air near a face may be
the specimen surface that was asked for; air in the interior cannot be. The 32
matches `scripts/analysis/air_audit_v2.EDGE_VOX`, and the detector loader asserts
they still agree, so a v3 and a v4 interior number mean the same thing.

**tile** — 64 voxels, the patch the model was trained on, the unit the requested
porosity field is defined on, and the period of the window seam.

**mean ± sd** — always over the three seeds (101, 202, 303) of one cell, sample
standard deviation.

---

## The metrics

### Porosity and air

```
phi_pore      = mean(label == PORE)  inside the material
air_fraction  = mean(label == AIR)   inside the material
air_interior  = mean(label == AIR)   inside the material and the interior
error         = phi_pore - requested_global_phi
```

Gate: `|error| < 0.005` (decision D39, carried over from v3).

**Failure** is not the same as inaccuracy. A volume has failed when
`phi < 1e-4`, `phi > 0.5`, or more than 20 % of the requested material is air.
The failure rate is the share of a cell's seeds that failed.

**Degenerate cells** apply the same two porosity bounds per 64-voxel tile, so a
volume that fails outright and a volume where a tenth of its tiles fail are
described on one scale. Tiles holding less than half material are excluded.

### Local obedience — the within-volume fit

Per 64-voxel tile: `delivered[c] = pore voxels / material voxels`, `requested[c]`
from the field. Then, **after subtracting each volume's own mean from both
sides**, OLS of delivered on requested gives the slope and R². Slope 1 is exact
obedience; slope 0 is a volume that ignored its field.

The mean removal is the whole point. Pooling cells from volumes at different
global targets and quoting the R² is what made eval v3 look obedient: most of
that variance is the global dose response, which assessment 2 already measures.
A unit test builds two volumes that ignore their field entirely, at two
different global levels, and shows they pool to R² ≈ 1.0 while each scores a
within-volume slope of 0. The pooled fit is still reported, under the key
`pooled_r2_not_obedience`.

A real volume has no request, so it reports only its **cell-to-cell spread** —
the noise floor a slope has to beat.

### Assembly seams

`seam_discontinuity` is imported from `poregen.diffusion.sampler`, so the number
the sampler logs and the number the suite reports are the same function. For each
axis it takes the mean absolute slice-to-slice difference at the **seam** planes
and at every **interior** plane, and reports the ratio. Ratio ≈ 1 means the plane
is indistinguishable from ordinary internal texture change.

Two periods, both judged against the same interior baseline (the interior
excludes the 64-voxel window planes in both cases, so the two ratios are
comparable):

| period | where | what it catches |
|---|---|---|
| **window**, 64 voxels | every tile face | the decoder-side seam and the LDM disagreeing between neighbouring latents |
| **chunk**, `64 × chunk_tiles` | where two independently denoised canvases meet | the joint-denoising seam |

Every chunk plane is also a window plane, which is why both are reported: the
window metric averages one bad plane over five and dilutes it; the chunk metric
looks only where two canvases met. A 192³ volume is one chunk and therefore has
**no** chunk plane — its chunk columns are empty by construction, not by failure.

Measured on the decoder's own continuous outputs: the grey level, and the pore
log-odds `log p_pore − log(1 − p_pore)` from the blended 3-class logits. The
log-odds is read from `probs.npz`; when a case did not store it the metric
reports `pore_logit_available: false` rather than substituting the argmax, which
is not a continuous field.

### Cross-head disagreement

The share of requested material where the **grey** head renders a dark void and
the **class** head calls the same voxels material. One of them is wrong, and no
single-head metric can see it.

The detector is imported from `scripts/analysis/_eval_v3`, not re-implemented:
`u8 < t_abs` with connected components below `min_cc` voxels dropped, where
`t_abs = 182` is the Dice-optimal absolute threshold calibrated inside
`sample_mask` on 8 real volumes (campaign 04, Dice 0.842) and `min_cc = 300`.
Both are read from `runs/campaigns/04-measurement-limits/air_audit_v2/results.json`
at call time.

### Layup recovery

Two readers, both imported from `scripts/analysis/layup_roundtrip.measure_volume`
— the T-I estimators, unchanged, the same code campaign 08 measured the real
floor with. The import is **inside the function**: that module's generation half
is pinned to the dead ldm05 sampler API, and a module-scope import would tie the
suite to it.

* `fft_slice` — the angular-spectrum reader on the grey volume.
* `pore_axes` — the pore principal-axis reader on the **predicted** pore class.

Scoring is **direct**: `theta_from_layup` writes the requested angles in image
coordinates, so measured == requested is the null hypothesis and no offset, sign
flip or face reversal is granted. Reported per reader: median and max |error|,
the fraction within 10°, strict 4-class accuracy (both rounded to the nearest of
0/45/90/135), the per-ply hit, and the **recovered ply count** — one more than
the number of 4-class changes between consecutive ply blocks, so two identical
adjacent plies read as one, which is the honest answer because nothing in the
volume distinguishes them.

The readers need a 1024×1024 in-plane window, which is why the layup assessment
generates at that size.

Floors, read from `runs/campaigns/08-pre-ldm06-diagnostics/angle_reader_floor/results.json`
and never hard-coded — a measurement typed into source drifts away from the file
that produced it:

| reader | median \|err\| | strict 4-class |
|---|---:|---:|
| `fft_slice` | 8.2° | 74 % |
| `pore_axes` (STORED mask) | 4.2° | 86 % |

`pore_axes` read the stored mask on real data, so its floor is a **ceiling** no
predicted-mask reader can beat.

### Requested geometry

Dice of the predicted air class against the requested air (`material == 0`), plus
the air fraction inside and outside the requested material. A model that obeys
the map scores a high Dice, a high air fraction outside and a low one inside.
There is no real floor for the Dice: a real volume was never asked for a hole.

---

## The seven assessments

Seeds 101 / 202 / 303 throughout; 93 cases in total. `chunk_tiles = (3, 3, 3)`,
`window_stride = 32` and `decode_stride = 32` unless a case says otherwise.

| # | Assessment | Cases | Asks |
|---|---|---:|---|
| 1 | `sampler` | 18 | DDIM {50, 100, 200} × {192³, 1024×1024×192}, target 0.03, layup A. Porosity error, interior air, both seam periods, wall time, failure rate. |
| 2 | `porosity_global` | 24 | targets {0.005 … 0.10} plus an off-manifold 0.15, 192³, DDIM-200. OLS slope/intercept/R², the gate per level. |
| 3 | `porosity_local` | 9 | three painted fields on the 3×3×3 tile grid — two halves 0.01/0.05, checkerboard 0.01/0.05, and the coherent field from `poregen.diffusion.porosity_field`. Within-volume slope and R². |
| 4 | `cfg` | 24 | `s_por` {1.0, 1.5, 2.0} × targets {0.02, 0.05}; plus `s_nb` {0, 1} at target 0.03. |
| 5 | `layup` | 9 | 1024×1024×192, target 0.03, DDIM-200. A (training), C (a permutation of A), B16 (the 16-ply 0.25 mm sequence). |
| 6 | `assembly` | 6 | window vs chunk seams and cross-head disagreement **on the sampler volumes**, the window-phase pair generated here, and the campaign-08 VAE control row. |
| 7 | `geometry` | 3 | 192×512×512 with a 64-voxel notch and a 200-voxel cylindrical hole through z. |

**The off-manifold request (assessment 2)** asks for φ = 0.15. `cond_por` clamps
at the training maximum of 0.107, so this is a failure-mode row and is excluded
from the dose-response fit. The manifest records the request; `notes` records the
value after the clamp.

**Layup C** is a fixed permutation of A with the same ply population — three
−45, two 0, three 45, two 90 — so any difference in recovery is the stacking
*order* and not the mix of angles. It is deliberately not the reverse of A,
because reversing a stack is the face-order freedom the direct scoring already
refuses to grant. A and B16 are read from `data/layup_ground_truth.json`.

**The `s_nb` arm (assessment 4)** runs on a 256³ volume, not 192³. The metric is
the pore Dice across the **chunk plane** between `s_nb = 0` and `s_nb = 1` at the
same seed, and a 192³ volume with 3×3×3-tile chunks has no chunk plane at all.
A Dice near 1 means turning the neighbour arm off changed nothing where it
could first act, so the arm is inert.

**The window-phase pair (assessment 6)** asks for the same 192³ region twice,
same seed, at two positions in a 256³ canvas. The specimen box, the orientation
profile and the uniform porosity request all move with the region, so the two
runs ask for the same thing and differ only in the grid they are assembled on;
at offset 32 the chunk plane at canvas voxel 192 runs through the region at
region coordinate 160. The seam columns of this assessment come from the
`sampler` volumes, whose grid is in canvas coordinates.

---

## The real floor

`eval_v4 real-floor` cuts crops from the **split_v3 TEST panels** — the panels
the model never saw — at the shapes the generated cases use, and runs every
request-free metric on them: phase fractions, degenerate cells, failure flags,
both seam periods, the cell-to-cell spread, and cross-head disagreement.

A real laminate holds only ~185–212 voxels of continuous material and its outer
z slices are the specimen surface, so a **192-deep box entirely inside
`sample_mask` does not exist** on any test volume — the best window scores a
usable-cell fraction of 0.000. Campaign 08 hit the same wall and dropped to a
128-deep box. The floor therefore reduces the depth one tile at a time until a
clean box fits, never below 128 (the interior needs more than two 32-voxel
shells), and records both the requested and the taken shape. Every metric here
is a ratio or a fraction, so a shallower crop does not bias it. Where no clean
box exists at any depth, the best window is taken and its `sample_mask` is
carried as the crop's requested material, so fractions stay inside real
specimen; a window less than 90 % usable is skipped instead.

Measured on the current dataset:

| shape | n | φ | seam xct | chunk seam xct | cell φ sd | cross-head |
|---|---:|---:|---:|---:|---:|---:|
| 128×192×192 | 4 | 0.0007 | 0.966 ± 0.005 | — | 0.0010 | 0.0000 |
| 128×1024×1024 | 3 | 0.0045 | 0.963 ± 0.009 | 0.902 ± 0.026 | 0.0031 | 0.0000 |

The grey seam ratio lands on campaign 08's real-volume control (0.963), which is
the validation that this implementation measures what that one did.

Three metrics have no floor here, on purpose:

* **porosity error** — a real volume was not asked for a porosity;
* **geometry Dice** — it was not asked for a hole;
* **layup recovery** — campaign 08 already measured that floor on real scans
  with the nominal ply sequence as truth.

---

## What the suite does not measure

No distribution distances: no FID, no PSD W1, no S₂(r) RMSE, no Ripley K. They
were the whole of v1–v3 and none of them can express a request. If a
"does it look like the data" number is wanted later it belongs beside these, not
instead of them, and it needs its own real-volume floor — a held-out real set
scored against another held-out real set — before any generated number is read
against it.

## Failure modes of the method

* **`pore_axes` reads the predicted mask**, so its score confounds angle
  recovery with segmentation quality. Its floor was measured on the stored mask
  and is therefore a ceiling.
* **The grey detector is calibrated inside `sample_mask`** on real volumes. A
  generated volume has no `sample_mask`, so the requested material stands in for
  it; near an outer face that mixes specimen surface with true exterior.
* **The layup truth is the nominal sequence**, which excludes measured per-ply
  deviations, so part of every residual is the truth's own error.
* **Three seeds** bound the sd loosely. A cell whose sd matters should be
  re-run with more.
* **The window-phase pair changes the chunk boundary as well as the window
  phase**, because the sampler anchors windows at the chunk origin. A Dice below
  1 there says the assembly grid matters; it does not separate the two causes.

## What eval v4 does not measure

`poregen.eval_v4` is an **operational** suite: does the model deliver the
porosity, layup, geometry and assembly quality it was asked for. It says
nothing about whether the generated microstructure has the right *statistics*.

Those live in `scripts/eval_generated_volumes.py`, which is kept for exactly
that reason: two-point correlation S2 with a Wasserstein-1 distance, Ripley's
K, FID on 2-D slices, pore-size distribution, and a memorisation check. None of
them is implemented in `eval_v4`.

This matters for Paper 1. Naiff et al. (*Computers & Geosciences* 206, 2026),
the designated paper to beat, reports FID on 2-D slices, W1 pore-size-
distribution distance and TPCF. Three of those four are only available from the
older script. Deleting it would have removed the head-to-head comparison the
paper is built on.

Its memorisation step, and only that step, expects the pre-`latents_r07z4`
latent layout; it warns and skips rather than failing. Porting these metrics
into `eval_v4` is worth doing, and until it happens the older script is the
implementation of record.

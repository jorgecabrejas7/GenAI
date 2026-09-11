# Evaluation methodology — eval v4

How a generated volume is measured, and why each number means something. This is
the authoritative reference for the paper methods section and for anyone reading
a `runs/campaigns/<NN>-<name>/` result.

The suite is the package `src/poregen/eval_v4/`, driven by the `eval_v4` CLI.
It replaces `scripts/eval_generated_volumes.py`, which is **deleted** — its five
distribution statistics were ported into assessment 8 and the rest of it was
not worth keeping.

Most of what the suite asks is *conditional*: did the volume deliver the
porosity it was **asked** for, in the cells it was asked for, with the layup it
was asked for, inside the material envelope it was asked for. A distribution
distance has no notion of a request, so it scores a model that ignores its
conditioning exactly as well as one that obeys it, and that is why the old
script could not answer the questions ldm06 exists to answer.

It could not be dropped either. A model can obey every request and still draw
the wrong microstructure, and Naiff et al. (*Computers & Geosciences* 206,
2026) — the paper to beat — report FID on 2-D slices, a W1 pore-size distance
and the two-point correlation, so those numbers are the head-to-head. They live
in assessment 8, on the same three rules as everything else: a manifest, a
declared requirement, and a **real-vs-real floor**. That floor is what the old
script never had. A Wasserstein distance between two finite samples of the same
material is not zero, so a generated distance is only readable as a multiple of
what real material scores against itself.

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

**Which latent store a run is evaluated against** is the run's own
`data.latents_root`, read from its `resolved_config.yaml`. There is no default
and no `--latents-root` flag. Before the first model call the suite checks that
the store's latent width equals the model's `z_channels` and that the store was
built by the run's `vae.checkpoint`; either disagreement is a hard stop naming
both values. A store is not interchangeable between runs — it fixes the latent
width, the normalisation the sampler works in and the decoder — and the wrong
one produces a plausible volume rather than an error. `measure` takes the same
store from the manifest, so the memorisation check compares generated patches
against the latents the model was actually trained on.

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

**φ — porosity is always `pore / material`.** Every requested and every
delivered φ in this document is the pore fraction of the *specimen envelope*:
`pore voxels / material voxels`, air excluded from the denominator. A request is
a material porosity, and `phi_pore`, `delivered_phi` and `delivered[c]` are all
measured that way, over the whole volume and per tile alike. This is **not** the
number the LDM is conditioned on. The latent store carries
`phi = pore / 64³` — the full patch, with any air outside the specimen counted in
the denominator — because a training patch's conditioning must describe the patch
the encoder saw. The two agree only for a patch that is entirely material. See
`ARCHITECTURE.md`, "Material porosity in, full-patch φ out", for the conversion
the sampler applies.

**requested φ of a window** — the sampler's denoising windows are tile-sized but
step 32 voxels, so a window straddles up to eight tiles of the requested field.
Its request is the RAW tile φ averaged over the window's voxel footprint,
weighted by the volume each tile covers; that material porosity is then
multiplied by the window's own material fraction to get the full-patch φ the
model was trained on, clamped to `[0.002, 0.107]`, and log-standardised into
`cond_por`. So a painted step in the field is asked for as a ramp one window
wide, not as a step: assessment 3 measures how well the model follows the field
it was actually given, and the field it is given is the footprint mean.

**mean ± sd** — always over the three seeds (101, 202, 303) of one cell, sample
standard deviation.

---

## The metrics

### Porosity and air

```
phi_pore      = mean(label == PORE)  inside the material   # pore / material
air_fraction  = mean(label == AIR)   inside the material
air_interior  = mean(label == AIR)   inside the material and the interior
error         = phi_pore - requested_global_phi            # both pore / material
```

`phi_pore_all` and `air_fraction_all` are the same counts over the WHOLE volume,
air included — reported beside the material fractions, never in place of them.

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

#### The same seams, per chunk along the generation order

`seam_metrics` answers "is there a seam in this volume". `chunk_profile` answers
"which chunk's seam, and does it get worse the further the chunk is from the
first one" — the shape a chunked sampler's failure actually has, because chunk
*k* assembles against material chunk *k−1* already produced, and an error that
compounds shows as a trend in *k* rather than as a worse volume average.

`chunk_blocks` cuts the volume into the reference chunk grid in RASTER order,
which is the order the sampler generates in, and cuts a short last block exactly
as `VolumeGenerator._chunk_ranges` does. Each block owns the planes at **its own
origin** — the planes where it met material that already existed — so every
plane has exactly one owner and chunk 0 owns none. Inside a block the planes
split into the tile family (multiples of 64, where two overlapping WINDOWS of
one solve meet) and the interior baseline (everything else), and both the chunk
and tile ratios are divided by that same baseline, the convention `seam_metrics`
already applies volume-wide.

Beside the seams, each chunk reports its own material porosity, and `chunk_s2`
compares S₂ inside the chunk with S₂ in a window centred ON each of its lower
planes — half the window is the previous chunk's material and half is this
one's. `s2_relative_distance` is `mean|across − inside| / mean(inside)`: 0 for
identical curves, and scale-free, so chunks at different porosity stay
comparable. A window that does not fit, or that is not almost entirely
requested specimen, reports `null` rather than a number measured on the
envelope.

The grid is a PARAMETER of the measurement and is never read from the volume's
own manifest — see assessment 11, where four arms with four chunk geometries
have to be read at the same planes.

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

### Microstructure statistics

`src/poregen/eval_v4/microstructure.py`. Four two-sample distances, each
reported three ways — **generated vs real**, the **real-vs-real floor**, and
their **ratio**. A ratio of 1 means the generated set is as close to real
material as two disjoint crops of one real panel are to each other, which is as
close as the measurement can tell.

| statistic | what it is | units |
|---|---|---|
| **S₂(r) W1** | Hann-windowed FFT autocorrelation of the pore phase, debiased by the window's own autocorrelation so `S₂(0) = φ`, radially binned; the two mean curves normalised to unit mass and compared as distributions over r | voxels |
| **pore-size W1** | Wasserstein-1 between the pooled equivalent diameters `(6V/π)^(1/3)` of every connected pore | voxels |
| **Ripley's K** | K(r) of the pore centroids, border-corrected; the distance is the mean \|log(K_gen/K_real)\| over r | dimensionless |
| **FID** | Fréchet distance on 2-D slices along all three axes, mean of the three | dimensionless |

### Memorisation — a full-store search, not a sample

The fifth statistic of assessment 8 is reported in the same `results.json` but
is measured on other volumes and against a different floor, so it is described
on its own.

**Queries.** Every 64³ whole-material patch of every 192³ volume in
`sampler` and `porosity_global` — the production operating point, 57 volumes
and 1 539 patches, rather than the nine microstructure volumes. Latent queries
are the volume RE-ENCODED through the frozen VAE, not its saved `latents.npy`:
ldm06 trains on a posterior draw (`data.latent_mode: sampled`) while the store
holds posterior means, so comparing the two directly would add `σ·ε` to every
distance.

**Bank.** ALL stride-64 rows of the **train** split — 219 580 of the 1 598 000
rows in it. Not a sample. The earlier implementation searched a random 10 000,
which gives an *upper bound* on the distance to the nearest training patch —
the wrong side of the question, because it can only make a copy look further
away than it is. The stride-64 restriction is not subsampling either: the store
is built at a 32-voxel stride, so eight interleaved copies of the 64-voxel grid
sit in it and a patch's "second-nearest neighbour" would otherwise be the same
material shifted by 32 voxels. On the stride-64 grid no two bank rows share a
voxel.

**Spaces.** Both. Latent space is per-channel normalised with the store's own
train statistics (raw channel σ spans 0.41–0.78, so a raw L₂ would be a
distance in whichever channel is widest). Grey space is the RAW source patches
`data/split_v3/patches_xct.bin` the store was built from, addressed by each
row's `source_row`.

**Statistic.** Favero's ratio `‖x − x′‖ / ‖x − x″‖` — nearest over
second-nearest — with **< 1/3 the memorisation verdict**. A copy sits on one
training patch and a normal distance from every other, so its ratio collapses
toward 0; a fresh sample sits a typical distance from both, so its ratio
approaches 1. The ratio is dimensionless, which is what lets the same number be
read in two spaces with no common unit.

**Floor.** The same ratio for 512 real **validation** patches against the same
train bank. Real held-out material is not memorised by construction, yet it
still has near neighbours in the train set because the panels are the same
material — so what the val patches score is the value the ratio takes when
nothing has been copied. Without it a generated ratio of 0.6 means nothing.

**Breakdown.** By requested porosity, and by whether the volume's windows ever
saw a neighbour: `neighbour_census` rebuilds each volume's EXISTS / OOB /
UNKNOWN window-face counts from the manifest's `volume_shape`, `chunk_tiles`
and `window_stride` with the sampler's own rule. A 192³ volume is exactly one
chunk, so it has no UNKNOWN face — the breakdown records that rather than
assuming it.

**Known failure mode.** The grey bank is raw scan data and the generated query
is decoded, so a generated patch carries the VAE decoder's reconstruction error
that a raw validation patch does not. A volume that reproduced a training
latent EXACTLY would still sit one reconstruction error away from the real
patch in grey space, so **the grey ratio is an upper bound** on how memorised a
volume is. The latent ratio is the sharp one; grey is the corroborating
pixel-level check. A second, smaller one: `squared_distances` selects by the
`|q|² + |b|² − 2q·b` expansion, which cancels badly near zero and rounds
differently at different chunk widths, so the two chosen distances are
recomputed from an explicit float64 difference before anything reads them.

**Analysis geometry.** The generated cases are 192³; no test panel holds a
clean 192-deep box, so the reference crops are 128³. Every statistic is
therefore defined on geometry both can supply: S₂ on **128³ windows** (r up to
48 voxels, one bin per voxel) so the FFT support, the Hann debias and the bin
edges are identical for both sets; the pore-size distribution and K on the
whole requested material, both being size-normalised; FID on 64×64 native
crops. (Memorisation is on 64³ patches, the size the VAE was trained on, and
on other volumes entirely — see above.)

**Connected components are 6-connected** everywhere. At a median pore diameter
of 1.79 voxels, 26-connectivity fuses voids that meet at a single corner.

**The FID feature extractor** is torchvision's `inception_v3` with
`Inception_V3_Weights.DEFAULT` (ImageNet IMAGENET1K_V1), 2048-d pool3
(`avgpool`) features. A full slice resized to 299 would be a ~10× downscale,
which shrinks a 1.79-voxel pore to 0.18 pixels — the structure the metric
exists to see would be gone before Inception saw it — so the crops are 64×64
native. 5000 crops per axis per set, because a 2048-dimensional covariance
estimated from fewer samples than that is singular.

**The preprocessing convention is `pytorch-fid`'s**, which is the one every
published FID number uses: replicate the grey crop to three channels,
bilinearly resize to 299×299 (no centre crop — a crop would throw away part of
the field of view), and put the network's input on the TF range **[-1, 1]**.
The range is reached the way torchvision wants it. torchvision's `inception_v3`
carries `transform_input=True` with the pretrained weights, and that flag
remaps an **ImageNet-normalised** input to [-1, 1] itself:

```
transform_input((v - mean) / std) == 2v - 1        # per channel, for the
                                                   # preset's mean/std
```

So `fid_preprocess` normalises with the weights' own preset `mean`/`std`
(read from `Inception_V3_Weights.DEFAULT.transforms()`, not retyped) and lets
`transform_input` finish the job. Normalising *and* scaling to [-1, 1] by hand
would apply the remap twice.

Before 2026-09-08 the crops went in on [0, 1] with no normalisation at all.
`transform_input` then mapped them to about [-0.19, 0.43] — under a third of
the range the network was trained on — so the 2048-d features were
off-distribution and the FID was comparable to nothing. **Any FID recorded
before that date is void.**

One difference from `pytorch-fid` remains and is deliberate: `pytorch-fid`
loads the TF-ported *FID Inception* weights, this suite loads torchvision's
`IMAGENET1K_V1`. The two networks give different absolute values on the same
images. FID here is therefore comparable **within this implementation only**,
which is why every table carries the real-vs-real floor beside it —
`results.json` records the exact extractor string in `fid.extractor`.

**Ripley's K is border-corrected** (reduced-sample): only pores further than r
from every face contribute, so the estimate is unbiased and converges to the
complete-spatial-randomness value `(4/3)πr³` — the volume of a ball, not the
2-D `πr²`. `K(r) / (4/3)πr³` above 1 is clustering, below 1 is regularity. The
old script applied no edge correction, so its K was biased low by a factor that
grew with r and had no known value to be validated against.

FID needs torchvision and the memorisation check needs the latent store AND
the raw source patches beside it; each reports an explicit skip with the reason
when its dependency is absent, so the other statistics are still measured.

---

The same defect was in the eval-v3 VAE reconstruction FID
(`poregen.eval.metrics`), which fed Inception unnormalised grey the same
way. It now calls the one shared `fid_input_from_grey`, so there is a single
convention rather than two that can drift. **Every eval-v3 VAE FID recorded
before 2026-09-08 is void** for the same reason the eval-v4 ones are: the
features were off-distribution, so the number was comparable to nothing.

### Field statistics — the delivered porosity field

`poregen.eval_v4.field_stats`. Naiff, Ramos and Wang ("Large-Scale Porous Media
Generation Through Field-Controlled Latent Diffusion Models", SSRN
10.2139/ssrn.7161201) claim that a porosity field is a **sufficient descriptor
of large-scale heterogeneity**. That is a claim about a measurable quantity, and
this is the measurement of it: the marginal distribution of local porosity per
window, and how far that field stays correlated along each axis, generated
against real.

**Delivered, not requested.** Every field is read out of a `label.tif` — pore
voxels over material voxels — never out of the `requested_field.npy` beside it.
The request is measured too and reported as its own row, because the failure
worth catching is a request with the right statistics that the model does not
deliver.

**One window on both sides.** 64-voxel windows every 32 voxels, on real crops
and generated volumes alike; a window under half material is dropped. That is
the grid campaign 01's T-D measured the real correlation lengths on (z 79.45,
y 413.58, x 900.95 voxels), which are in turn the lengths
`build_porosity_field` smooths the coherent request with — so the request, the
delivery, the real material and the target the request was built from are all
the same kind of number. The T-D lengths are printed as the first row of the
correlation table.

**Per axis.** The material is a laminate: it decorrelates in about 80 voxels
through the thickness and in several hundred in plane. A single isotropic
correlation length would average away the structure most likely to be lost.

**A length longer than the crop is not a length.** The 1/e crossing is
interpolated between lags (on a 32-voxel grid, whole lags alone are a 20 % error
on an 80-voxel length). When the curve does not cross inside the lags a field
holds, the report prints `> reach` rather than a number, and `corr_length_gap`
refuses to difference two lengths found over different reaches. A 192³ volume
reaches 128 voxels of lag, so it can measure z and can only ever say "longer
than this crop" in y and x; the `r(axis, lag)` columns at fixed voxel-disjoint
lags are the comparison that holds at every crop size. A crossing found close to
the reach rests on few independent samples and should be read with the reach
beside it.

Distances between marginals are W1 on the porosities and W1 after each sample is
divided by its own mean — the second removes the global porosity the two sets
happen to sit at and leaves the shape of the heterogeneity. Both carry a
real-vs-real floor: the real crops are split in half and scored against
themselves, because two halves of real material do not score zero either.

---

## The eleven assessments

Seeds 101 / 202 / 303 throughout; 177 cases in total. `chunk_tiles = (3, 3, 3)`,
`window_stride = 32` and `decode_stride = 32` unless a case says otherwise.
The case counts are those `eval_v4.cases.build_cases` produces.

| # | Assessment | Cases | Asks |
|---|---|---:|---|
| 1 | `sampler` | 18 | DDIM {50, 100, 200} × {192³, 1024×1024×192}, target 0.03, layup A. Porosity error, interior air, both seam periods, wall time, failure rate. |
| 2 | `porosity_global` | 48 | targets {0.005 … 0.10} plus an off-manifold 0.15, 192³, DDIM {50, 200}. OLS slope/intercept/R², the gate per level. |
| 3 | `porosity_local` | 18 | three painted fields on the 3×3×3 tile grid — two halves 0.01/0.05, checkerboard 0.01/0.05, and the coherent field from `poregen.diffusion.porosity_field`. Within-volume slope and R². |
| 4 | `cfg` | 24 | `s_por` {1.0, 1.5, 2.0} × targets {0.02, 0.05}; plus `s_nb` {0, 1} at target 0.03. |
| 5 | `layup` | 9 | 1024×1024×192, target 0.03, DDIM-200. A (training), C (a permutation of A), B16 (the 16-ply 0.25 mm sequence). |
| 6 | `assembly` | 9 | window vs chunk seams and cross-head disagreement **on the sampler volumes**, the offset triple generated here (offsets 0 / 16 / 32 in a 256³ canvas), and the campaign-08 VAE control row. |
| 7 | `geometry` | 8 | 192×512×512 with a 64-voxel notch and a 200-voxel cylindrical hole through z, plus the off-manifold sphere. |
| 8 | `microstructure` | 9 | 192³, DDIM-200, layup A, targets {0.01, 0.03, 0.06}. S₂(r) W1, pore-size W1, Ripley's K and FID on 2-D slices, each against the matched real-vs-real floor. The memorisation block is reported here too but is measured on the assessment-1 and assessment-2 volumes against the whole train store, with its own real-val floor. |
| 9 | `surface` | 12 | a flat z-surface request (controllability) and a rough one matched to the real floor's Sa and correlation length (realism), at 192³ and 1024 wide. |
| 10 | `multichunk` | 5 | 384³ — two chunks on EVERY axis, so the z chunk planes exist — as a box, a sphere and a rough slab. |
| 11 | `assembly_modes` | 17 | four ways to assemble the SAME request from the SAME seeds, at 384³ and 1024×1024×192, reported per chunk index. See below. |
| — | `field_stats` | 0 | **Measure-only: it generates nothing.** Re-reads the coherent-field volumes `porosity_local` and `multichunk` already wrote, and the real crops, for the marginal and the per-axis correlation length of the delivered field. `eval_v4 measure field_stats` and `eval_v4 report`; there is no `generate field_stats`. |

**Assessment 8 runs at three porosity levels and no more** because every
statistic in it is confounded by pore fraction, and each level needs its own
matched real reference. Reading a generated set against real material at a
different porosity would show the porosity gap and call it a texture gap.

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

**The offset triple (assessment 6)** asks for the same 192³ region three times,
same seed, at three positions in a 256³ canvas. The specimen box, the
orientation profile, the uniform porosity request and the frame every noise
draw is taken in all move with the region, so the runs ask for the same thing
from the same noise and differ only in the grid they are assembled on. Offset 0
is the reference and every other offset is scored against it; the seam columns
of this assessment come from the `sampler` volumes, whose grid is in canvas
coordinates.

*The region-relative noise frame.* An offset only isolates the assembly grid if
the noise moves with the request. The sampler therefore takes every random draw
of the reverse process — the canvas the chunks start from, and the fresh noise
that re-noises each finished chunk at every timestep — as ONE canvas-sized field
per draw, rolled by `request_offset` before use
(`poregen.diffusion.sampler.region_noise_field`). The value used at canvas cell
`p` is the value the request sees at region cell `p − offset`, so translating
the request translates its noise with it. `torch.roll` is a permutation of one
draw, so the field is still exactly iid standard normal and the wrap reaches
only canvas cells outside the requested region. With the draw anchored to the
canvas instead — which is what the sampler did until this fix — the two runs of
a pair differed in the noise realisation as well as in the grid, and the pore
Dice across them could not attribute the difference to either. `request_offset`
must be a whole number of latent cells; the sampler refuses anything else.

*What each offset isolates.* An offset moves two independent things, and one
offset cannot tell them apart:

| offset | window phase | chunk alignment |
|---:|---|---|
| 0 | window origins start on the region origin | the region IS chunk zero: no chunk plane crosses it |
| 32 | unchanged — 32 is a whole window stride, so region-relative window origins are still 0, 32, 64 … | the chunk plane at canvas voxel 192 crosses the region at region coordinate 160 |
| 16 | half a stride out: region-relative window origins are 16, 48, 80 … | a chunk plane at region coordinate 176 |

Read 0 against 32 for chunk alignment and 32 against 16 for window phase. The
report gives the pore Dice and the φ difference per offset and never pools
them: the two offsets answer different questions.

**The four arms (assessment 11).** Assessment 6 asks whether the answer depends
on where the assembly grid falls. Assessment 11 asks the prior question: how
much of the quality is the hybrid chunked sampler at all, and how far is it from
a ceiling. Four arms, one request, one seed set per scale:

| arm | `chunk_tiles` | neighbours | what it is |
|---|---|---|---|
| `joint` | the whole volume | UNKNOWN (the CFG null) | the ldm05 MultiDiffusion sampler: one canvas, neighbour conditioning inert, no chunk plane anywhere |
| `autoregressive` | (1, 1, 1) | the canvas | the ldm05 sequential sampler: one patch at a time against finished material |
| `hybrid` | (3, 3, 3) | the canvas | production |
| `teacher_forced` | (3, 3, 3) | a REAL test volume's encodings at the same canvas positions | a control, not a sampler — the ceiling the hybrid would reach with perfect neighbours |

Only the neighbour source and the chunk geometry move. The porosity target is
uniform (so any drift from chunk to chunk is a defect and not the request), the
step count is 50 for every arm, and every arm starts from the identical noise
field: the initial canvas draw is the first draw of the reverse process, so it
does not depend on how many re-noising draws an arm goes on to make.

*The teacher-forced canvas* is assembled by pasting one stored latent per
64-voxel tile from the run's own store — the rows whose origins are 64 apart
tile a block with no overlap — and what is pasted is a posterior draw
`mu + sigma·eps`, per-channel normalised, seeded by the case. The store is never
re-encoded: an encoder other than the one that built it would produce latents
the denoiser has never seen, which would measure the encoder and call it a
ceiling.

*Why the ceiling is missing at 384.* Test patches in the r08 store reach
`z0 = 128`, so the deepest real block on the 64-voxel tile grid is 192 voxels.
There is no real material to teach with at 384 deep, and repeating a block to
fill the depth would put a fake join exactly on a chunk plane — the one place
this assessment measures. `teacher.find_reference_block` raises rather than
faking it, and the 384 cells carry no ceiling row.

*Every arm is measured on the same grid.* The arms have different chunk
geometries by construction, so reading each on its own `chunk_tiles` would put
four different sets of planes in one table. The reference grid is the production
period, 192 voxels: for `hybrid` it is also the generation grid, and for the
others it is "what happens at the planes the production sampler would have had
to assemble across". The `joint` arm has no chunk planes at all, so its row is
the measurement's own no-seam reading, beside the real-volume floor.

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

### The matched-porosity reference pairs

`--shapes micro` cuts what assessment 8 is read against: for every requested
porosity level and every test **panel**, two 128³ crops of that panel whose
measured porosity matches the level and which **share no material**. Crop `a`
is the reference the generated set is scored against; `a` against `b` is the
floor. Two crops of one panel, never two panels: panels differ in cure and in
void population, so a cross-panel pair would fold the between-panel spread into
the floor and flatter every generated number read against it.

Every candidate box is scored exactly, and for free, from a one-pass cell
summary of the pore mask — the same trick `cell_ok_by_slice` uses for the
specimen mask — so the search never reads a volume once per candidate. A box
must lie entirely inside `sample_mask`, which excludes exterior air and the
three drilled registration holes.

Measured on the current dataset — three test panels, Na_05, Na_09 and JI_8:

| requested φ | panels matched exactly | worst miss |
|---|---|---:|
| 0.01 | all three | 0.0000 |
| 0.03 | all three | 0.0000 |
| 0.06 | Na_09 only | 0.0194 |

**φ = 0.06 does not exist in Na_05 or JI_8** at this crop size: the best boxes
those panels hold are 0.041 and 0.043. The crops are written anyway, with
`phi_miss` in the manifest and a warning line in `findings.md`, because the
alternative — silently comparing against real material at another porosity — is
what turns a porosity gap into a reported texture gap.

---

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
* **Every non-zero offset moves the chunk boundary**, because the sampler
  anchors windows at the chunk origin and the only shift available is a
  translation of the request. Offset 32 therefore isolates chunk alignment
  cleanly (it keeps the window phase), but offset 16 carries a chunk plane as
  well as the half-window phase, at a different region coordinate than 32 does.
  A Dice below 1 at 16 that is not matched at 32 points at the window phase; it
  does not prove it on its own.
* **The microstructure floor is two crops per panel per level.** A distance
  between two samples that small is itself noisy, so a ratio near 1 means
  "indistinguishable at this sample size" and not "identical".
* **φ = 0.06 is not reachable on two of the three test panels** (see below), so
  part of that row's distance is a porosity gap. The miss is reported beside
  the number; it is not corrected for.
* **The generated volumes are 192³ and the real crops 128³.** Every
  microstructure statistic is defined on the fixed 128³ analysis window or on a
  size-normalised quantity, but a residual shape effect cannot be ruled out by
  construction alone.

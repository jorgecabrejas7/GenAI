# Speaker guide — PoreGen group meeting

What to say on each slide, what to be careful about, and likely questions.
Times come from the speaker notes. The total is about 42 min, which is over the
40-min target; possible cuts are at the end.

The numbers here are the same as in the speaker notes of `deck/deck.src.html`,
which give the source of each one.

---

## 1 · Title — PoreGen (~40 s)

- "This is a real XCT slice of one of our carbon-fibre coupons."
- The animation runs diffusion backwards: noise at t = 1000, clean image at t = 0.
  At the end the labels appear: grey = material, amber = pore, blue = air (here a drilled hole).
- The one-sentence version of the talk: **"We train a model that writes new volumes like this one,
  with the labels already attached."**

**Careful:** the noise is computed in the browser from the real slice. It is not a model output.
If someone asks "is this generated?", say no: it is a real scan, and the noise only shows the idea.

The scale bar: the drilled hole is about 200 voxels at 25 µm, so 5 mm.

---

## 2 · Why generate porosity? (~50 s)

Go left to right.

1. "In these laminates, porosity is the main defect from manufacture. It makes the part weaker."
2. Point at the coupon: pores sit between the plies and are flat.
3. "To see them we need X-ray CT. A scan takes hours, then someone must segment it. So we have few volumes."
4. "We get a 3-D grey volume with 25 µm voxels."
5. The dashed amber volume is the main point: "We want as many volumes as we need, with the porosity
   we ask for, and with every pore already labelled."
6. Read the four cards quickly: training data for segmentation, known pores to test NDT,
   digital twins ("we come back to this at the end"), and "what if" questions.

**Careful:** do not give a number of hours for a scan. There is no source for one on the slide.

**Next:** "The model type we use is diffusion. Here is the idea in one picture."

---

## 3 · Destroy with noise, then learn to undo it (~40 s)

- Forward (→): "Add a little Gaussian noise many times. At t = 1000 nothing is left. This needs no learning."
- Reverse (←): "The hard way is back. A network learns, at each noise level, what noise to remove."
- Key point: "If it can do that at every level, we start from pure noise and end at a new, realistic slice."

**Careful:** the reverse half of the animation is the same known noise played backwards. It returns to
exactly the same real slice. Say "this is the direction the model learns", not "this is the model generating".

**Next:** "Now precisely how much noise there is at each step."

---

## 4 · Adding noise: one formula for any step (~80 s)

The only equation in this part. Go slowly.

1. Read it in words: "the clean image, scaled down, plus a noise image, scaled up."
2. Define each symbol once: x₀ clean, ε noise, x_t the result, ᾱ_t ("alpha bar") the share of image left, 1 → 0.
3. Square roots: "they keep the total variance the same; only the mix changes." (Name: variance preserving.)
4. Why it matters: "We never add noise 1000 times. In training: pick a random t, one noise draw, one line."
5. Top row: "at t ≈ 600 you can still guess the plies, at 800 almost not."
6. Bottom left: ᾱ_t, cosine schedule — slow at the ends, fast in the middle.
7. Bottom right: SNR on a log axis.

**Careful:** the notes say SNR is "exactly 0" at t = 1000. The measured value is ᾱ_T = 3.75·10⁻³³.
Say "practically zero, about 10⁻³³".

**Question "why cosine?":** a linear schedule destroys the image well before the last step, so many steps
are wasted. Cosine spreads the loss of information more evenly (Nichol & Dhariwal 2021).

---

## 5 · Removing noise: a network guesses the noise (~60 s)

- "The network is ε_θ: it outputs a guess of ε. θ are its weights."
- Training is normal supervised learning: "We made the noisy image ourselves, so we know the right answer.
  The loss is the squared error between the true and the guessed noise."
- Generating: "Start from pure noise. At each step the network guesses the noise, we remove a small part,
  and the image gets a little cleaner."
- Say at the end: "Our model actually predicts a different quantity, called v. Two slides from now I say why."

**Careful:** the animation uses the true noise, not a network. Same as slide 3.

**Question "why does a small step work and not one big step?":** at high noise the guess of the noise is
poor in detail. Many small steps let each guess correct the last one.

---

## 6 · Why it works: images on a probability landscape (~80 s)

The main intuition for the ML side. Take your time.

- "Picture every image as a point in a huge space. Realistic images are rare: they sit on hills."
- "Noise pushes a point downhill, away from the hills."
- "So a network that predicts the noise knows, at every point, which way is downhill.
  Turn it around and you have the uphill direction. That is called the score."
- The animation: two hills, a toy example in 2-D. σ is the noise level.
  - "With much noise, the landscape is smooth and the arrows point to the centre."
  - "As σ falls, they point to each hill."
  - "The amber dots follow the arrows and end on a hill. That is exactly what sampling does with images."

**Question "why does the noise level matter?":** at low noise the score is only known near the hills.
Far away there is no training data. High noise gives a smooth field everywhere, so we start there and
reduce it step by step.

---

## 7 · The theory in three results (~70 s)

Read the words, not the equations. One sentence each:

1. "Removing noise and knowing the uphill direction are the same task" (Vincent 2011).
2. "The best guess of the clean image is the noisy image moved one step uphill" — Tweedie's formula (Efron 2011).
   This is how each step makes its clean-image guess.
3. "With very small steps, adding noise is a continuous random motion. To run it backwards you need only the score"
   (Song et al. 2021). DDPM and DDIM are two ways to take those backward steps.

**Careful:** do not derive anything. If someone wants details, offer to go through it after the talk.

**In equation 3:** f is the drift, g the noise strength, w̄ a Brownian motion that runs backwards in time.

---

## 8 · Sampling: 1000 noisy steps or 50 clean ones (~50 s)

- Same toy landscape.
- Grey: DDPM. "1000 steps, fresh noise each time, so the paths shake."
- Amber: DDIM. "The same network, no fresh noise, big steps: 50 are enough. The paths are smooth."
- "Both end on the hills."
- "DDIM is deterministic: the same start noise always gives the same image. We use DDIM with 50 steps,
  so 20 times fewer network calls."

**Question "do you lose quality with DDIM?":** our evaluation ran both 50 and 200 steps. Porosity control was
inside ±0.005 for all 42 volumes with both (slide 30).

---

## 9 · Conditioning: telling the model what to make (~50 s)

- "A model that makes 'some coupon' is not useful. We want to say what to make. That extra input is the condition."
- Classifier-free guidance: "In training we hide the condition at random. So the same network learns
  'a coupon' and 'a coupon with 2 % porosity'."
- "At sampling we ask both and go further in the direction of the difference, by a factor s."
- s = 1: the plain conditional answer. s > 1: follow the condition more strongly.
- Give the example: "Asked for 0.02, s = 1 gives 0.0176 and s = 2 gives 0.0199. More on slide 31."

---

## 10 · What should the network output? Three choices (~70 s)

- "The network can be trained to output the noise ε, the clean image x₀, or a mix of both called v."
- Watch the panels as t grows:
  - ε is always noise.
  - x₀ is always the clean image.
  - v changes: at small t it looks like the noise, at large t like minus the image.
- "Given x_t and any one of them, you get the other two. So they carry the same information."
- "What differs is how easy each one is to learn, and how errors grow. That decides it. Next slide."

---

## 11 · Why we predict v (~80 s)

The most technical slide. Its message is simple: **at the last step, ε-prediction divides by zero.**

- Left: "a and b are like cos and sin of an angle. As t grows, x_t turns from the image to the noise.
  v is the direction of that turn."
- Right: "To get an image from a noise guess, you divide by a. At the last step a is 6·10⁻¹⁷.
  Any error is multiplied by 10¹⁶."
- "With v there is no division. At the last step the network just outputs the image, with a minus sign."
- Zero terminal SNR: "Our schedule removes the image completely at the last step. So training ends at pure noise,
  exactly where sampling starts. Many older schedules leave a faint trace of the image there (Lin et al. 2024)."
- Close with: "But zero signal is exactly where ε fails. That is why we predict v."

**Bottom right plot:** SNR over the last 10 % of steps, ours (falls to 0) against the classic linear schedule
(stays above 0).

---

## 12 · Diffuse in a compressed latent, not in voxels (~50 s)

- "Diffusion on full 3-D volumes costs too much. So we first train an autoencoder that compresses each 64³ patch."
- The numbers: "3 × 64³ input values become 4 × 16³ latent values: 48 times fewer.
  That is the same factor as Stable Diffusion."
- In Rombach's names: f = 4 (each side shrinks 4×), c = 4 latent channels.
- "Every denoising step works on the small latent. The decoder runs once at the end, and gives grey and labels."
- The denoiser is a 3-D U-Net with 83 M parameters.

**Why 3 inputs?** grey, pore map and air map. Explained on slide 17.

---

## 13 · The road to modern diffusion (~50 s)

A timeline. Do not stop on each box. Read left to right in one breath:

"DDPM made diffusion work; the score-SDE view explained it; DDIM made it fast; guidance made it controllable;
v fixed what the network outputs; latent diffusion made it cheap; EDM cleaned up the design;
MultiDiffusion fused overlapping windows, which is the idea behind our tiling; zero-SNR fixed the end of the schedule."

"Amber is what our model uses directly."

**Cut candidate:** you can skip this slide if time is short. Everything on it was said already.

---

## 14 · Generating microstructures: what is missing (~60 s)

- "GANs gave the first 3-D microstructures (Mosser 2017). SliceGAN learns 3-D from 2-D images only (2021)."
- "Diffusion then took over: first 2-D, then 3-D."
- "The closest work is Naiff et al. 2026: latent diffusion for 3-D porous media, with porosity as the condition.
  A 2026 preprint adds a porosity map."
- Then the right column: "What we need is all four at once: large volumes, local control, labels, and specimen shape."

**Careful:** do not say "nobody does X". Say: "Some papers have one of these. The point is all four in one model,
for composites, with labels." The notes say this too.

---

## 15 · Our data: 81 coupons at 25 µm (~50 s)

- Left: a real slice; the line wipes from grey to labels.
- Right top: a section through the thickness. "You can see the plies and the pore layers between them."
- "The coupons come from several panels with different porosity."
- "The labels come from one fixed method, a local Sauvola threshold. Not by hand."

**Careful:** say this clearly: **"The model learns to reproduce that segmentation. Its labels are only as good
as the reference."** Someone will ask about it otherwise.

z is always the through-thickness direction. Say it once here; it helps later slides.

---

## 16 · From volumes to patches, split by panel (~50 s)

- "A panel is one manufactured plate. All coupons from one panel go to the same set."
- "18 panels: 12 train, 3 validation, 3 test. So we test on plates the model never saw."
- Why: "Coupons cut from the same plate share their porosity pattern. A split by coupon would leak."
- "Each volume is cut into 64³ patches, a new one every 32 voxels."
- Holes: "Patches that touch the drilled holes are removed. Otherwise the holes would be the only air inside
  the material, and the model learns to draw air inside material."

Totals if asked: 59 / 11 / 11 volumes; about 2.18 M patches after the hole removal.

---

## 17 · Our autoencoder: grey in, grey and labels out (~50 s)

- "Three 64³ maps go in: grey, pore mask, air mask."
- "The encoder shrinks each side by 4. For each latent number it gives a mean and a spread, and z is drawn from them.
  That is what makes it a variational autoencoder."
- "The decoder has two heads: the grey image, and three class probabilities per voxel."
- It is small: 0.76 M parameters. The big network is the denoiser.

**Question "why give it the masks, since they come from the grey?":** they add little new information,
but they make the model output labelled volumes directly. That is the point of the work. (Also on slide 19.)

---

## 18 · Training the autoencoder: five terms (~70 s)

One line per row of the table:

- L1: grey values come back right.
- LPIPS: compares slices through the eyes of an image network, so texture counts, not only the average.
- Cross-entropy + Dice: the labels. Dice = overlap of two masks, 1 = perfect.
- KL, very small weight: keeps the latent smooth and near a standard normal, which suits diffusion.
- 3-D discriminator, from step 15 000: a critic tries to tell real from rebuilt patches. This gives sharper texture.

"This is the Stable Diffusion autoencoder recipe, made 3-D."

The bottom bar shows when each term is on.

**Cut candidate:** you can go through it in 30 s, or skip it.

---

## 19 · How much can we compress? f = 4, c = 4 (~70 s)

- "We trained six autoencoders, all at f = 4, with c = 32, 16, 8, 4, 2, 1."
- Amber line: pore Dice. "It stays above 0.93 down to c = 4 and falls clearly at c = 2 and c = 1."
- Dashed line: grey L1 error.
- "c = 4 is the smallest latent that keeps pore Dice above 0.93 and air Dice above 0.99. That is 48×, like Stable Diffusion."

**Careful:** one run per size. If someone asks about the variance between runs, say there is only one run each.
Also, the model in use is the c = 4 autoencoder at a later checkpoint than the one in this table.

---

## 20 · What the denoiser sees (~70 s)

Two types of input:

- **Maps as channels** (like red, green, blue in a photo): ply orientation per depth (as cos 2θ and sin 2θ,
  so 0° and 180° are the same), the material envelope, the six neighbour latents, a flag per neighbour
  (exists / unknown / outside), and each neighbour's noise level.
- **Numbers through normalisation**, together with the timestep: the porosity request, the depth,
  and the distance from each face to the specimen surface.

Key sentence: **"The porosity request is per patch. A map of requests gives local control."**

"Every input is always present. There are no switches."

If asked: 127 input channels in total.

---

## 21 · Neighbours, and dropping one face at a time (~60 s)

- "A large volume is made patch by patch. Each patch sees its six touching neighbours."
- Each face has three states: exists, unknown, outside.
- Training: "Each face becomes 'unknown' with probability 0.10. All six with probability 0.05;
  that is the 'no neighbour' answer for guidance."
- Why: "In an earlier version we only had all-or-nothing. So training never showed 'five known, one unknown'.
  But that is exactly the state at the front of a volume made chunk by chunk.
  The model then put a layer with too few pores along every chunk join."
- "A matched fine-tune that changed only this one setting removed it: the strip went from 0.19 of the mean to 1.03."

**Careful:** that result is from the earlier model and dataset. It was not measured again on the final model.
The notes also write "0.19 for the control", which can confuse; just say "0.19 → 1.03".

**Link to slide 26:** per-face dropout fixed the band in training. The overlapped sampler (slide 26) fixes
what remains at generation time. If someone asks "why both?", that is the answer.

---

## 22 · Patch, tile, window, chunk (~50 s)

"One size, 64³ voxels, used four ways." The animation goes through them; say one sentence per phase:

- **Patch:** a training example cut from a real scan, a new one every 32 voxels.
- **Tile:** a cell of the grid of a new volume, every 64 voxels, no overlap. One porosity request and one latent block per tile.
- **Window:** one network call during generation. Also 64³, but they start every 32 voxels and are averaged.
- **Chunk:** 3 × 3 × 3 tiles, 192³ voxels, that go through all 50 steps together. Then the next chunk.

64 voxels = 1.6 mm. Say it here; slide 40 uses "per 1.6 mm cube".

---

## 23 · How a large volume is generated (~60 s)

Go left to right along the boxes:

1. Maps: where the specimen and the air are, the ply angle per depth, a porosity value per tile.
2. Random noise fills the whole latent canvas.
3. Take the chunks in order: depth, then y, then x.
4. Inside the chunk, each window (up to 125) gets its noisy latent, its maps and its six neighbours, and predicts v.
5. Average the predictions with the window weights, then one DDIM step.
   Steps 4 and 5 repeat 50 times.
6. The finished chunk is written to the canvas and becomes a neighbour.
   At the end, the decoder turns the canvas into grey and labels.

Scale: **"One whole coupon is 900 chunks and took 4.3 h on one GPU."**

---

## 24 · Large volumes: tiles, chunks, order (~50 s)

- Background: one generated slice, 1024 × 1024, at mid-depth. The thin grid is the 64-voxel tiles.
- "The amber chunk goes through all 50 steps. Then it is fixed and becomes a known neighbour."
- "Each chunk starts two tiles after the one before, so they share one tile. The next slides explain why."
- Re-noising: "When a new chunk is made, finished neighbours are noised to the current step.
  So the model always sees neighbours at a matching noise level, as in training."

**Cut candidate:** much of this was on slide 23. You can merge the two in your talk and spend 20 s here.

---

## 25 · Blending: overlap, weight, normalise (~60 s)

- Left top: windows every 32 voxels, each with a sin² weight, high in the centre, low at the edges.
- Left bottom: the summed weights. "We divide by this sum, so it is a weighted average everywhere.
  No window border is a hard edge."
- "The decoder does the same with its 64³ passes."
- Right: "Why the decoder overlap matters: tile-by-tile decoding leaves visible seams, ratio 1.30.
  Overlapped decoding gives 0.96, the same as the real volume."

**Careful:** the seam ratios were measured with the earlier autoencoder, not the final one.

---

## 26 · The overlapped sampler: keep, see, redraw (~80 s)

Follow the five phases of the animation:

1. **Standard:** "Chunk k+1 starts where chunk k ends. The last voxels of chunk k were made while that side
   was unknown. They get too few pores: a band at every join."
2. **Start earlier:** "Chunk k+1 starts 64 voxels before chunk k ends."
3. **Keep and see:** "The first 32 shared voxels are kept. At every step they are reset to chunk k's result,
   with the noise of that step. The new chunk sees them, so it continues chunk k." (This is the RePaint idea.)
4. **Redraw:** "The last 32 are the poor rim of chunk k. They are free, and chunk k+1 draws them again,
   now with known material on both sides."
5. **Result:** no band, at 2.6–2.7× the time.

**Question "why not keep all 64?":** the other 32 are the poor rim. If they were kept, at every step they would
push the new chunk to copy the band.

"This is the sampler we use for every volume larger than one chunk."

---

## 27 · The overlapped sampler removes the band (~50 s)

- Left: pore fraction against the distance from a chunk plane, divided by the mean.
  "Standard drops to about 0.6 at the plane. Overlapped stays near 1."
- Middle: seam ratio per axis. Standard: y 0.61–0.70, x 0.80–0.89. Overlapped: 0.97–1.04. Target is 1.
- Right: the cost, 2.6–2.7× the time, measured on the five whole coupons.

**Careful:** say it: **"The overlapped sampler is measured on only 3 of the 18 evaluation cases so far.
The full evaluation is running."**

---

## 28 · The same on a whole coupon (~40 s)

- "One real coupon at 2.62 %, generated twice from its maps."
- "Each line is the pore fraction at each position along the coupon. The ticks are the standard chunk joins, every 192 voxels."
- "The real scan has nothing at the ticks. Standard drops at every tick. Overlapped has no drops."
- Right: folded onto the distance from a join. At the join: real 1.01, standard 0.84, overlapped 0.99.

Why a profile and not an image: the band is only a few voxels wide, so you do not see it in a picture of the pores.

**Possible question "why are the lines smoothed?":** both generated twins have a small pattern that repeats
every 4 voxels, the size of one latent cell. The 4-voxel average removes it exactly. The real scan does not have it.
If asked "what causes it?": it comes from the latent grid / decoder. It is not yet examined.

**Cut candidate:** you can merge 27 and 28 into one minute.

---

## 29 · First, measure real against real (~50 s)

- "A raw distance means nothing alone. How far is far?"
- "So for each statistic we also compare two separate real crops of the same panel. That is the floor."
- "We report the ratio: distance(generated, real) ÷ distance(real, real)."
- "Ratio 1: as close to real as real is to itself. Ratio 3: three times the natural variation."
- "Each porosity level is compared with real crops of the same porosity, so a porosity difference is not read as texture."

Say this well: every result slide after this uses the ratio.

---

## 30 · Porosity control: you ask, it delivers (~70 s)

- Left: requested against delivered, each dot one 192³ volume. Band = ±0.005.
- "Slope 1.03, R² 0.998, and all 42 volumes are inside the band." (7 levels × 3 seeds × 2 step counts.)
- "It delivers a bit less at the lowest level: 0.0037 for 0.005."
- Right: all the pores of three volumes, asked for 0.005, 0.03, 0.10. "The pores sit in layers between plies, as in real coupons."

If asked about outside the range: a request of 0.15 gave 0.160. It is not in the fit.

---

## 31 · Local control: where the pores go (~70 s)

- "We paint a porosity value on each 64³ tile and measure what each tile gets."
- "Halves": low on one side, high on the other. Slope 0.98, R² 0.87. "Large patterns are followed."
- "Checkerboard": alternating tiles. Slope 0.47, R² 0.63. "The contrast is followed, but at half strength,
  because neighbouring tiles pull towards each other."
- Bottom right: guidance. "s = 1 gives 0.0176 for 0.02, s = 2 gives 0.0199. It is a free knob at sampling time."

**Link to slide 41:** the checkerboard result is the same limit as the "local field" item: patches stay similar
over about 50–60 voxels.

---

## 32 · Four ways to compare microstructure (~80 s)

One sentence each:

1. **S₂(r):** the probability that two points r apart are both pore. Captures amount, size and spacing at once.
2. **Pore-size distribution:** sizes of connected pores.
3. **Ripley's K:** for pore centres, how many others lie within r, against a random pattern. Measures clustering.
4. **FID:** distance between image-network features of 2-D slices. Measures how the slices look.

"To compare two curves we use W₁, the area between the two cumulative curves: how much mass you must move,
and how far."

**Cut candidate:** shorten to 40 s; the audience only needs the idea before the next slide.

---

## 33 · Microstructure: 1 to 3× the real floor (~60 s)

- "Bars are the ratio to the real floor. The line is 1."
- "At 3 %, S₂ (0.90) and Ripley (0.77) are at the floor."
- "Pore size is about 2.2–2.3× at every level: the pores are not sized exactly like real ones."
- "FID is 2.7–3.4×: slices look different. That matches the decoder blur."

**Careful:** do not hide that 0.01 is worse (S₂ 2.42). If asked why, a possible reason (not measured):
low porosity means few pores per volume, so the statistics are noisier.

---

## 34 · Shape and plies: drawn where we ask (~70 s)

- Left: notch and 200-voxel hole, and a sphere. "Air Dice 0.992 against the requested air."
- Right: two stacking sequences. "The circle shows the requested angle at this depth. The elongated pores turn with it."
- Bottom: requested angle against the angle read back from the labels.
  "Median error 2.2° for layup A and 1.8° for C. The same reader on real scans has 4.2°."
- "A sequence with thin double plies is harder: 17.8°."

**Careful:** the reader itself has a 4.2° error on real scans, so errors under that are at the reader's limit.
Do not claim "better than real".

---

## 35 · Does the model use each input? (~70 s)

"Swap one input for a valid value from somewhere else, keep the rest, and see what changes." Four tests:

- **Ply profile:** give layup C, score A. Error on A goes from 2.2° to 54°; C is read back at 2.5°. "The plies follow the profile."
- **Material map:** replace with "all material". Air Dice goes from 0.99 to 0. "It does not invent geometry."
- **Porosity field:** replace with its mean. Local R² collapses (0.63 → 0.01, 0.87 → 0.13).
- **Position:** from another specimen. Nothing changes (0.992 → 0.992). "The model barely uses it."

The last one is honest and good to say: one input does almost nothing.

---

## 36 · Is it copying the training set? (~60 s)

- "For each generated patch, find its nearest and second-nearest training patch. A copy is very close to one
  and far from the rest, so d₁/d₂ is small. We flag below 1/3."
- "Real patches the model never saw are our reference: they cannot be copies."
- "Both sit near 1. None of 4 847 generated patches is below 1/3, searched against all 222 976 training patches."

Key word: "the whole training bank, not a sample."

---

## 37 · Three baselines, three questions (~60 s)

"Each baseline removes one of our ideas, so the comparison says what that idea buys."

- **SliceGAN:** does diffusion beat a GAN?
- **Pixel-space 3-D diffusion:** our recipe on voxels, no autoencoder. What does the latent buy?
- **Porosity-only latent diffusion:** our networks, but one porosity value as the only condition (Naiff et al.'s recipe).
  What do maps and neighbours buy?

"The first two have no condition, so they enter only the comparisons that need no request."

If asked: pixel diffusion stopped at a 24 h limit, at step 55 546.

---

## 38 · Against baselines: control is where we win (~80 s)

- Four small plots, ratio to the real floor, log axis. Amber = ours.
- Be honest, as the slide is: **"On texture, the porosity-only model is as good as ours or better
  at 0.01 and 0.03."**
- "SliceGAN and pixel diffusion are clearly worse on S₂ and pore size."
- "Where we win is control: 42 of 42 inside ±0.005, against 38 of 42 for porosity-only. It over-delivers at high porosity.
  And local maps, plies, shape and assembly, which no baseline offers."

**Careful:** the baselines will be run again on the final setup. Say "these numbers will be updated".

---

## 39 · Real or generated? (~60 s)

- Top: all pores of a real 192³ crop (0.105) next to a generated one asked for 0.10 (got 0.104), rotating.
  "Both show pore layers between plies. The generated pores are smoother and less broken up."
- Bottom strip: slices at 0.03, grey and labels. "The generated grey is smoother: decoder blur."
- Right: pore-by-pore comparison, **"results coming"**.

**Careful:** the right panel is still a placeholder. Fill it, or say in one sentence that it is running.

You can ask the audience first: "which one is real?" and wait 2 s.

---

## 40 · A real coupon and its generated twin (~90 s)

Start with the sentence on the slide: **"The generator got the shape and air, the ply angles, and the porosity
per 1.6 mm cube. It did not get the pores. It drew every pore itself."**

- Keys 1–5 change the coupon. Pick one or two live; you do not need all five.
- Four coupons are from the test panels. The 8.7 % one is from training, because no test coupon has that much porosity.
  The 1.01 % coupon is a different material system.
- Each video: left real, right generated, same slice, same grey scale. Three parts:
  1. The whole coupon in three planes.
  2. The same with labels: pores amber, air blue.
  3. A 512 × 512 window at full resolution, where the real coupon has the most pores.
- The sweep slows down at depths with many pores.
- What to look for: same shape and air (given), same ply directions (given), pores in the same places and amounts
  (only the amount per cube was given). "The generated grey is smoother: decoder blur."

**Careful:**
- The 0.10 % coupon's twin has only 0.07 %. That is the low-porosity under-delivery of slide 30. Say so if someone reads the numbers.
- The air in the generated grey was set to one value per volume (the mean grey of its air). The labels are unchanged.
  If someone asks why the air looks so flat, that is the reason.

---

## 41 · Limitations and next steps (~60 s)

Three limits, all measured:

- **Decoder blur:** rebuilt patches keep about 3/4 of the real sharpness; FID stays about 3× the floor.
- **Chunk seam:** standard chunks leave 0.6–0.9 of the pores at joins. The overlapped sampler removes it, at 2.6–2.7× the time.
- **Local field:** generated porosity stays similar over about 50–60 voxels in-plane; in real coupons, over hundreds.

Next: the full evaluation with the overlapped sampler, pore-by-pore morphology, baselines on the final setup,
and a sharper decoder that keeps the same latent space.

End with "Questions?" and stop.

---

## If you need to cut time (about 2–5 min)

- Skip slide 13 (timeline): about 50 s.
- Do slide 18 (five loss terms) in 30 s: about 40 s.
- Merge 23 and 24 (flow + order): about 30 s.
- Merge 27 and 28 (seam results): about 40 s.
- Shorten slide 32 (four statistics) to 40 s: about 40 s.
- On slide 40, show two coupons, not five.

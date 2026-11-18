# Speaker notes — PoreGen group meeting

*Text in quotes is what you say. Text in italics is for you only: what to point at, cautions, and answers to likely questions. Total is about 42 min; the cuts to get under 40 min are at the end. The source of every number is in the speaker notes of the deck (key N).*

---

## 1 · Title (~40 s)

"This is a real XCT slice of one of our carbon-fibre coupons."

*The animation runs diffusion backwards: noise at t = 1000, clean image at t = 0. At the end the labels appear.*

"Grey is material, amber is pore, and blue is air — here, a drilled hole."

"In one sentence: we train a model that writes new volumes like this one, with the labels already attached."

*The noise is computed in the browser from the real slice. It is not a model output. If asked "is this generated?": no, it is a real scan. The drilled hole is about 200 voxels at 25 µm, so 5 mm.*

---

## 2 · Why generate porosity? (~50 s)

*Go left to right.*

"In these laminates, porosity is the main defect from manufacture. It makes the part weaker."

*Point at the coupon.*

"The pores sit between the plies, and they are flat."

"To see them we need X-ray CT. A scan takes hours, and then someone must segment it. So we have few volumes."

"From the scan we get a 3-D grey volume with 25 µm voxels."

*Point at the dashed amber volume. This is the main point.*

"We want as many volumes as we need, with the porosity we ask for, and with every pore already labelled."

*Read the four cards quickly.*

"That gives us training data for segmentation networks, known pores to test NDT methods, digital twins of real coupons — we come back to this at the end — and 'what if' questions."

*Do not give a number of hours for a scan. There is no source for one.*

"The type of model we use is a diffusion model. Here is the idea in one picture."

---

## 3 · Destroy with noise, then learn to undo it (~40 s)

*Forward half of the animation (→).*

"We add a little Gaussian noise, many times. At t = 1000 nothing of the slice is left. This direction needs no learning."

*Reverse half (←).*

"The hard direction is back. We train a network to predict, at each noise level, what noise to remove."

"If it can do that at every level, we can start from pure noise and end at a new, realistic slice. That is how we generate."

*The reverse half is the same known noise played backwards, so it returns to exactly the same real slice. Say "this is the direction the model learns", never "this is the model generating".*

"Now let's see precisely how much noise there is at each step."

---

## 4 · Adding noise: one formula for any step (~80 s)

*The only equation in this part. Go slowly.*

"The noisy image is the clean image, scaled down, plus a noise image, scaled up."

*Point at each symbol as you say it.*

"x₀ is the clean slice, ε is pure noise of the same size, and xₜ is the result at step t. ᾱₜ — 'alpha bar' — is how much of the clean image is left. It goes from 1 to 0."

"The square roots keep the total variance the same at every step. The image does not get brighter or darker; only the mix changes."

"So we never add noise 1000 times. In training we pick a random t, draw one noise image, and build xₜ in one line. That makes training cheap."

*Top row.*

"At about t = 600 you can still guess the plies. At 800, almost not."

*Bottom left.*

"This is ᾱₜ for our cosine schedule. It falls slowly at the start and the end, and fast in the middle."

*Bottom right.*

"And this is the signal-to-noise ratio, on a log scale. At the last step it is practically zero — about 10⁻³³ — so nothing of the image is left."

*The deck notes say "exactly 0"; the measured value is ᾱ_T = 3.75·10⁻³³. Use "practically zero".*

*If asked "why cosine?":* "With a linear schedule the image is destroyed well before the last step, so many steps are wasted. Cosine spreads the loss of information more evenly. That is from Nichol and Dhariwal, 2021."

"That is the easy direction. Now the network."

---

## 5 · Removing noise: a network guesses the noise (~60 s)

"The network is called ε_θ, because it outputs a guess of ε. θ are the weights we train."

"Training is ordinary supervised learning. We made the noisy image ourselves, so we know the right answer: the noise. The loss is the squared error between the true and the guessed noise."

"To generate, we start from pure noise. At each step the network guesses the noise, we remove a small part of it, and the image gets a little cleaner. After all the steps, we have a new image."

"Our model actually predicts a different quantity, called v. In two slides I say why."

*The animation uses the true noise, not a network, as on slide 3.*

*If asked "why many small steps and not one big step?":* "At high noise the guess is poor in detail. With many small steps, each guess can correct the one before."

---

## 6 · Why it works: images on a probability landscape (~80 s)

*The main intuition for the ML side. Take your time.*

"Picture every possible image as a point in a huge space. Realistic images are rare: they sit on hills of high probability."

"Adding noise pushes a point downhill, away from the hills."

"So a network that predicts the noise knows, at every point, which way is downhill. Turn that around and you have the way uphill. That direction is called the score."

*The animation: two hills, a toy example in 2-D. σ is the noise level.*

"With much noise, the landscape is smooth and the arrows point to the centre. As the noise falls, they point to each hill."

"The amber dots start anywhere, follow the arrows, and end on a hill. That is exactly what sampling does with images."

*If asked "why does the noise level matter?":* "At low noise we only know the score near the hills — far away there is no training data. With high noise the field is smooth everywhere, so we start there and reduce the noise step by step."

---

## 7 · The theory in three results (~70 s)

*Read the words, not the equations. One sentence each.*

"First: removing noise and knowing the uphill direction are the same task. That is Vincent, 2011."

"Second: the best guess of the clean image is the noisy image, moved one step uphill. That is Tweedie's formula, in Efron 2011. This is how every step makes its guess of the clean image."

"Third: with very small steps, adding noise is a continuous random motion. To run it backwards in time, you need only the score. That is Song and colleagues, 2021. DDPM and DDIM are two ways to take those backward steps."

*Do not derive anything. If someone wants details, offer to go through it after the talk. In equation 3: f is the drift, g the noise strength, w̄ a Brownian motion that runs backwards in time.*

---

## 8 · Sampling: 1000 noisy steps or 50 clean ones (~50 s)

*Same toy landscape.*

"In grey, DDPM: 1000 small steps, with fresh noise at each one, so the paths shake."

"In amber, DDIM: the same trained network, but no fresh noise, and big steps. 50 are enough, and the paths are smooth."

"Both end on the hills."

"DDIM is deterministic: the same starting noise always gives the same image. We generate with DDIM at 50 steps, so 20 times fewer network calls."

*If asked "do you lose quality with DDIM-50?":* "We evaluated both 50 and 200 steps. Porosity control was inside ±0.005 for all 42 volumes with both. That is on slide 30."

---

## 9 · Conditioning: telling the model what to make (~50 s)

"A model that makes 'some coupon' is not useful. We want to say what to make. That extra input is the condition — for us, the porosity we want, the ply angles, and the shape."

"In training, we hide the condition at random. So the same network learns both 'a coupon' and 'a coupon with 2 % porosity'."

"At sampling we ask both questions, and step beyond the conditional answer, in the direction of the difference, by a factor s."

"s = 1 is the plain conditional answer. Above 1, the model follows the condition more strongly."

"For example: asked for 0.02, s = 1 gives 0.0176, and s = 2 gives 0.0199. More on that later."

---

## 10 · What should the network output? Three choices (~70 s)

"A network can be trained to output the noise ε, the clean image x₀, or a mix of the two called v."

*Watch the panels as t grows.*

"ε is always pure noise. x₀ is always the clean image. v changes: at small t it looks like the noise, and at large t it looks like minus the image."

"Given xₜ and any one of them, you get the other two. So they carry the same information."

"What differs is how easy each one is to learn, and how errors grow. That decides the choice — next slide."

---

## 11 · Why we predict v (~80 s)

*The most technical slide. Its one message: at the last step, ε-prediction divides by zero.*

*Left diagram.*

"a and b behave like the cosine and sine of an angle. As t grows, xₜ turns from the image towards the noise. v is the direction of that turn."

*Right side.*

"To get an image from a noise guess, you divide by a. At the last step, a is 6·10⁻¹⁷. So any error is multiplied by 10¹⁶ — in practice, a division by zero."

"With v there is no division. At the last step, the network just outputs the image, with a minus sign."

"Our schedule removes the image completely at the last step. That is good: training then ends at pure noise, exactly where sampling starts. Lin and colleagues showed that many older schedules leave a faint trace of the image there."

"But zero signal is exactly where ε fails. That is why our model predicts v."

*Bottom right plot: the SNR over the last 10 % of steps — ours falls to 0, the classic linear schedule stays above 0.*

---

## 12 · Diffuse in a compressed latent, not in voxels (~50 s)

"Diffusion on full 3-D volumes is too expensive. So first we train an autoencoder that compresses each 64³ patch."

"3 × 64³ input values become 4 × 16³ latent values: 48 times fewer. That is the same factor as Stable Diffusion's autoencoder."

"In Rombach's notation: f = 4, so each side shrinks 4 times, and c = 4 latent channels."

"Every denoising step works on the small latent. The decoder runs once at the end, and gives the grey image and the labels."

"The denoiser is a 3-D U-Net with 83 million parameters."

*Three inputs — grey, pore map, air map — are explained on slide 17.*

---

## 13 · The road to modern diffusion (~50 s)

*Do not stop on each box. Read left to right in one breath.*

"Each idea of the last slides, with its paper. DDPM made diffusion work. The score-SDE view explained it. DDIM made it fast. Guidance made it controllable. v fixed what the network outputs. Latent diffusion made it cheap. EDM cleaned up the design. MultiDiffusion fused overlapping windows — the idea behind our tiling. And zero-SNR fixed the end of the schedule."

"Amber is what our model uses directly."

*Cut candidate: everything here was already said.*

---

## 14 · Generating microstructures: what is missing (~60 s)

"GANs gave the first 3-D microstructures — Mosser, 2017. SliceGAN, in 2021, learns 3-D from 2-D images only."

"Then diffusion took over: first in 2-D, then in 3-D."

"The closest work to ours is Naiff and colleagues, 2026: latent diffusion for 3-D porous media, with porosity as the condition. A 2026 preprint adds a porosity map."

*Point at the right column.*

"What we need is all four at once: large volumes, local control, labels, and the specimen shape."

"Some papers have one of these. The point is to have all four in one model, for composites, with labels."

*Never say "nobody does X".*

---

## 15 · Our data: 81 coupons at 25 µm (~50 s)

*Left: a real slice; the line wipes from grey to labels.*

"This is a real slice. Every voxel gets one label: material, pore, or air."

*Right top.*

"This is a section through the thickness. You can see the plies, and the pore layers between them. z is always the through-thickness direction."

"The coupons come from several panels with different porosity levels."

"The labels come from one fixed method, a local Sauvola threshold — not by hand. So the model learns to reproduce that segmentation, and its labels are only as good as the reference."

*Say the last sentence clearly; otherwise someone will ask.*

---

## 16 · From volumes to patches, split by panel (~50 s)

"A panel is one manufactured plate. All coupons from one panel go to the same set."

"We have 18 panels: 12 for training, 3 for validation, 3 for test. So we test on plates the model never saw."

"Why: coupons cut from the same plate share their porosity pattern. A split by coupon would leak."

"Each volume is cut into 64³ patches, a new one every 32 voxels."

"Patches that touch the drilled holes are removed. Otherwise the holes are the only air inside the material that the model sees, and it learns to draw air inside material."

*If asked: 59 / 11 / 11 volumes; about 2.18 million patches after hole removal.*

---

## 17 · Our autoencoder: grey in, grey and labels out (~50 s)

"Three 64³ maps go in: the grey image, the pore mask, and the air mask."

"The encoder shrinks each side by 4. For each latent number it gives a mean and a spread, and z is drawn from them. That is what makes it a variational autoencoder."

"The decoder has two heads: one for the grey image, and one for three class probabilities per voxel."

"It is small — 0.76 million parameters. The big network is the denoiser."

*If asked "why give it the masks, if they come from the grey?":* "They add little new information, but they make the model output labelled volumes directly. That is the point of the work."

---

## 18 · Training the autoencoder: five terms (~70 s)

*One sentence per row.*

"L1 checks that the grey values come back right."

"LPIPS compares slices through the eyes of an image-recognition network, so the texture must be right, not only the average."

"Cross-entropy and Dice train the labels. Dice is the overlap of two masks; 1 is perfect."

"KL, with a very small weight, keeps the latent smooth and close to a standard normal. That suits diffusion."

"And from step 15 000, a 3-D discriminator — a critic — tries to tell real patches from rebuilt ones. That gives sharper texture."

"This is the Stable Diffusion autoencoder recipe, made 3-D."

*The bar at the bottom shows when each term is on. Cut candidate: do it in 30 s.*

---

## 19 · How much can we compress? f = 4, c = 4 (~70 s)

"We trained six autoencoders, all at f = 4, with c = 32, 16, 8, 4, 2 and 1."

*Amber line.*

"Pore Dice stays above 0.93 down to c = 4, and falls clearly at c = 2 and c = 1."

*Dashed line.*

"The dashed line is the grey error."

"c = 4 is the smallest latent that keeps pore Dice above 0.93 and air Dice above 0.99. It compresses 48 times, like Stable Diffusion."

*One run per size — if asked about variance, say so. The model in use is the c = 4 autoencoder at a later checkpoint than the one in this table.*

---

## 20 · What the denoiser sees (~70 s)

"Besides the noisy latent, the network gets two kinds of input."

"Maps go in as extra channels, like red, green and blue in a photo. These are the ply orientation at each depth, the material envelope, the six neighbour latents, a flag for each neighbour — exists, unknown or outside — and each neighbour's noise level."

"The ply angle is given as cos 2θ and sin 2θ, so 0° and 180° are the same."

"Numbers go in through the normalisation layers, together with the time step: the porosity request, the depth, and the distance from each face to the specimen surface."

"The porosity request is per patch. So a map of requests gives local control."

"Every input is always present. There are no switches."

*If asked: 127 input channels in total.*

---

## 21 · Neighbours, and dropping one face at a time (~60 s)

"A large volume is made patch by patch. Each patch sees its six touching neighbours, so the two sides continue each other."

"Each face has one of three states: exists, unknown, or outside."

"In training, each face becomes 'unknown' with probability 0.10. All six together with probability 0.05 — that is the 'no neighbour' answer that guidance uses."

"Why: in an earlier version we only had all or nothing. So training never showed 'five known, one unknown'. But that is exactly the state at the front of a volume made chunk by chunk. The model then put a layer with too few pores along every chunk join."

"A matched fine-tune that changed only this one setting removed it: the pores in that strip went from 0.19 of the mean to 1.03."

*That result is from the earlier model and dataset; not measured again on the final model. Say only "0.19 → 1.03".*

*If asked "why do you also need the overlapped sampler (slide 26)?":* "Per-face dropout fixes the band in training. The overlapped sampler removes what remains at generation time."

---

## 22 · Patch, tile, window, chunk (~50 s)

"The network always works on cubes of 64 voxels — 1.6 mm. These four words say how we use those cubes."

*One sentence per phase of the animation.*

"A patch is a training example, cut from a real scan, a new one every 32 voxels."

"A tile is a cell of the grid of a new volume, every 64 voxels, with no overlap. Each tile has one porosity request and one latent block."

"A window is one call of the network during generation. Also 64³, but windows start every 32 voxels, so they overlap by half, and we average them."

"A chunk is 3 × 3 × 3 tiles — 192³ voxels — that go through all 50 denoising steps together. Then the next chunk starts."

*Say "1.6 mm" clearly: slide 40 uses "per 1.6 mm cube".*

---

## 23 · How a large volume is generated (~60 s)

*Go left to right along the boxes.*

"First we say what we want, as maps: where the specimen and the air are, the ply angle at every depth, and a porosity value for every tile."

"Random noise fills the whole latent canvas — the compressed volume."

"Then we take the chunks in order: depth, then y, then x."

"Inside a chunk, each window gets its noisy latent, its maps and its six neighbours, and the network predicts v."

"We average the predictions with the window weights, and take one DDIM step. That repeats 50 times."

"The finished chunk is written to the canvas, and becomes a neighbour for the next chunks."

"When every chunk is done, the decoder turns the canvas into grey values and labels."

"For scale: one whole coupon is 900 chunks, and took 4.3 hours on one GPU."

---

## 24 · Large volumes: tiles, chunks, order (~50 s)

*Background: one generated slice, 1024 × 1024 at mid-depth. The thin grid is the 64-voxel tiles.*

"The amber chunk goes through all 50 steps. Then it is fixed, and becomes a known neighbour."

"Each chunk starts two tiles after the one before, so they share one tile. The next slides explain why."

"When a new chunk is made, its finished neighbours are noised to the current step. So the model always sees neighbours at a matching noise level, as in training."

*Cut candidate: merge with slide 23 and spend 20 s here.*

---

## 25 · Blending: overlap, weight, normalise (~60 s)

*Left top.*

"Windows start every 32 voxels. Each has a sin² weight: high in the centre, low at the edges."

*Left bottom.*

"This is the sum of the weights. We divide by it, so the result is a weighted average everywhere, and no window border is a hard edge."

"The decoder does the same with its 64³ passes."

*Right.*

"This is why the decoder overlap matters. Decoding tile by tile leaves visible seams — a ratio of 1.30. Overlapped decoding gives 0.96, the same as the real volume."

*These seam ratios were measured with the earlier autoencoder, not the final one.*

---

## 26 · The overlapped sampler: keep, see, redraw (~80 s)

*Follow the five phases of the animation.*

"In the standard way, chunk k+1 starts where chunk k ends. But the last voxels of chunk k were made while that side was still unknown. They get too few pores: a band at every join."

"In the sampler we use, chunk k+1 starts 64 voxels earlier."

"The first 32 shared voxels are kept. At every step, they are reset to chunk k's result, with the noise of that step. The new chunk sees them, so it continues chunk k. This is the RePaint idea."

"The last 32 are the poor rim of chunk k. They are free, and chunk k+1 draws them again — now with known material on both sides."

"The result: no band, at 2.6 to 2.7 times the time."

"This is the sampler we use for every volume larger than one chunk."

*If asked "why not keep all 64?":* "The other 32 are the poor rim. If we kept them, at every step they would push the new chunk to copy the band."

---

## 27 · The overlapped sampler removes the band (~50 s)

*Left.*

"This is the pore fraction against the distance from a chunk join, divided by the mean. Standard chunks drop to about 0.6 at the join. Overlapped chunks stay near 1."

*Middle.*

"The same per axis: standard is 0.6 to 0.9; overlapped is 0.97 to 1.04. The target is 1."

*Right.*

"The cost is 2.6 to 2.7 times the time, measured on five whole coupons."

"I should say: the overlapped sampler is measured on only 3 of the 18 evaluation cases so far. The full evaluation is running."

---

## 28 · The same on a whole coupon (~40 s)

"This is one real coupon at 2.62 % porosity, generated twice from its maps."

"Each line is the pore fraction at each position along the coupon. The ticks are the standard chunk joins, every 192 voxels."

"The real scan has nothing at the ticks. Standard chunks drop at every tick. Overlapped chunks have no drops."

*Right.*

"Folded onto the distance from a join: at the join, real is 1.01, standard 0.84, overlapped 0.99."

*Why a profile and not an image: the band is only a few voxels wide, so you cannot see it in a picture.*

*If asked "why are the lines smoothed?":* "Both generated twins have a small pore pattern that repeats every 4 voxels — one latent cell. A 4-voxel average removes it exactly. The real scan does not have it. We have not examined the cause yet."

*Cut candidate: merge 27 and 28 into one minute.*

---

## 29 · First, measure real against real (~50 s)

"A raw distance between generated and real means nothing alone. How far is far?"

"So, for each statistic, we also compare two separate real crops from the same panel. That distance is the floor."

"We report the ratio: generated-to-real distance, divided by real-to-real distance."

"A ratio of 1 means the generated set is as close to real as real is to itself. A ratio of 3 means three times the natural variation."

"Each porosity level is compared with real crops of the same porosity, so a porosity difference is not read as a texture difference."

*Say this well: every result slide after this uses the ratio.*

---

## 30 · Porosity control: you ask, it delivers (~70 s)

*Left: each dot is one 192³ volume; the band is ±0.005.*

"Requested against delivered porosity. Seven levels, three seeds, two step counts: 42 volumes."

"The slope is 1.03, R² is 0.998, and all 42 volumes are inside ±0.005."

"It delivers a little less at the lowest level: 0.0037 for 0.005."

*Right.*

"These are all the pores of three volumes, asked for 0.005, 0.03 and 0.10. The pores sit in layers between the plies, as in real coupons."

*If asked about higher requests: 0.15 gave 0.160. It is not in the fit.*

---

## 31 · Local control: where the pores go (~70 s)

"Now we paint a porosity value on each 64³ tile, and measure what each tile gets."

"Halves — low on one side, high on the other: slope 0.98, R² 0.87. Large patterns are followed."

"Checkerboard — alternating tiles: slope 0.47, R² 0.63. The contrast is followed, but at half strength, because neighbouring tiles pull towards each other."

*Bottom right.*

"And guidance: for a request of 0.02, s = 1 gives 0.0176, and s = 2 gives 0.0199. It is a free knob at sampling time."

*The checkerboard result is the same limit as "local field" on the last slide.*

---

## 32 · Four ways to compare microstructure (~80 s)

"Each statistic turns a volume into a curve or a distribution."

"S₂ of r: the probability that two points a distance r apart are both pore. It captures the amount, the size and the spacing at once."

"The pore-size distribution: the sizes of connected pores."

"Ripley's K: for the pore centres, how many others lie within r, compared with a random pattern. It measures clustering."

"And FID: a distance between image-network features of 2-D slices. It measures how the slices look."

"To compare two distributions we use W₁: the area between the two cumulative curves — how much mass you must move, and how far."

*Cut candidate: 40 s is enough; the audience only needs the idea.*

---

## 33 · Microstructure: 1 to 3× the real floor (~60 s)

"The bars are the ratio to the real floor. The line is 1."

"At 3 % porosity, S₂ at 0.90 and Ripley at 0.77 are at the floor."

"Pore size is about 2.2 to 2.3 times the floor at every level. The generated pores are not sized exactly like real ones."

"FID is 2.7 to 3.4 times: the slices look different. That matches the decoder blur."

*Do not hide that 1 % is worse (S₂ 2.42).*

*If asked why:* "A possible reason — we have not measured it — is that at low porosity there are few pores per volume, so the statistics are noisier."

---

## 34 · Shape and plies: drawn where we ask (~70 s)

*Left.*

"The material map carves a notch and a 200-voxel hole here, and a sphere here. Air Dice against the requested air is 0.992."

*Right.*

"Two stacking sequences. The circle shows the requested angle at this depth, and the elongated pores turn with it."

*Bottom.*

"Requested angle against the angle read back from the labels. The median error is 2.2° for layup A and 1.8° for C. The same reader on real scans has 4.2°."

"A sequence with thin double plies is harder: 17.8°."

*The reader itself has 4.2° error on real scans, so these errors are at the reader's limit. Do not claim "better than real".*

---

## 35 · Does the model use each input? (~70 s)

"We swap one input for a valid value from somewhere else, keep everything else, and see what changes."

"Ply profile: we give layup C and score A. The error on A jumps from 2.2° to 54°, and C is read back at 2.5°. So the plies follow the profile."

"Material map: we replace it with 'all material'. Air Dice drops from 0.99 to 0. The model does not invent geometry."

"Porosity field: we replace it with its mean. The local fit collapses — R² from 0.63 to 0.01 for the checkerboard."

"Position: from another specimen. Nothing changes — 0.992 to 0.992. The model barely uses it."

*The last result is honest and worth saying clearly.*

---

## 36 · Is it copying the training set? (~60 s)

"For each generated patch, we find its nearest and second-nearest training patch. A copy is very close to one and far from the rest, so the ratio d₁ over d₂ is small. We flag anything below one third."

"Real patches the model never saw are our reference: they cannot be copies."

"Both sit near 1. None of 4 847 generated patches is below one third — searched against all 222 976 training patches. The whole bank, not a sample."

---

## 37 · Three baselines, three questions (~60 s)

"Each baseline removes one of our ideas, so the comparison says what that idea buys."

"SliceGAN asks: does diffusion beat a GAN?"

"Pixel-space 3-D diffusion is our recipe directly on voxels, with no autoencoder. It asks: what does the latent buy?"

"Porosity-only latent diffusion is our networks, with one porosity value as the only condition — the recipe of Naiff and colleagues. It asks: what do the maps and the neighbours buy?"

"The first two have no condition, so they enter only the comparisons that need no request."

*If asked: pixel diffusion stopped at a 24-hour limit, at step 55 546.*

---

## 38 · Against baselines: control is where we win (~80 s)

*Four small plots, ratio to the real floor, log axis. Amber is ours.*

"To be honest: on texture, the porosity-only model is as good as ours, or better, at 1 % and 3 %."

"SliceGAN and pixel diffusion are clearly worse on S₂ and pore size."

"Where we win is control. 42 of 42 volumes inside ±0.005, against 38 of 42 for porosity-only, which over-delivers at high porosity. And local maps, plies, shape and large-volume assembly, which no baseline offers."

"These baselines will run again on the final setup, so these numbers will be updated."

---

## 39 · Real or generated? (~60 s)

*Optional: ask "which one is real?" and wait 2 s.*

"On the left, all the pores of a real 192³ crop at 0.105. On the right, a generated volume asked for 0.10 — it gave 0.104."

"Both show pore layers between the plies. The generated pores are smoother and less broken up."

*Bottom strip.*

"Slices at 3 %, grey and labels. The generated grey is smoother: that is the decoder blur."

"The pore-by-pore comparison against real coupons is running now."

*The right panel still says "results coming". Fill it before the talk, or use the sentence above.*

---

## 40 · A real coupon and its generated twin (~90 s)

"The generator got the coupon's shape and air, its ply angles, and its porosity per 1.6 mm cube. It did not get the pores. It drew every pore itself."

*Keys 1–5 change the coupon. Show one or two, not all five.*

"Four of these coupons are from the test panels. The 8.7 % one is from training, because no test coupon has that much porosity. The 1.01 % coupon is a different material system."

"Left is the real scan, right is the generated twin — the same slice, the same grey scale."

"First the whole coupon, in three planes. Then the same with labels: pores in amber, air in blue. Then a 512 × 512 window at full resolution, where the real coupon has the most pores."

"The sweep slows down at depths with many pores."

"Look for the same shape and the same ply directions — those were given — and pores in the same places and amounts, where only the amount per cube was given."

"The generated grey is smoother: decoder blur again."

*The 0.10 % coupon's twin has only 0.07 % — the low-porosity under-delivery of slide 30. Say so if someone reads the numbers.*

*If asked why the air looks so flat:* "In the generated grey, all air is set to one value per volume — the mean grey of its air. The labels are unchanged."

---

## 41 · Limitations and next steps (~60 s)

"Three limits, all measured."

"Decoder blur: rebuilt patches keep about three quarters of the real sharpness, and FID stays about 3 times the floor."

"Chunk seams: standard chunks leave 0.6 to 0.9 of the pores at the joins. The overlapped sampler removes the band, but takes 2.6 to 2.7 times the time."

"Local field: generated porosity stays similar over only about 50 to 60 voxels in-plane. In real coupons, over hundreds."

"Next: the full evaluation with the overlapped sampler, the pore-by-pore comparison, the baselines on the final setup, and a sharper decoder that keeps the same latent space."

"Thank you. Questions?"

---

## If you need to cut time (2–5 min)

- *Skip slide 13 (timeline): about 50 s.*
- *Do slide 18 (five loss terms) in 30 s: about 40 s.*
- *Merge slides 23 and 24: about 30 s.*
- *Merge slides 27 and 28: about 40 s.*
- *Do slide 32 in 40 s: about 40 s.*
- *On slide 40, show two coupons, not five.*

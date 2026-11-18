# Speaker notes — PoreGen group meeting

*Text in quotes is what you say. Text in italics is for you only: what to point at, cautions, and answers to likely questions. The talk is 23 slides, about 22 min of speaking. After "Questions?" there are 18 backup slides (B1–B18) with the method detail; use them only to answer questions. The source of every number is in the speaker notes of the deck (key N).*

*The audience is mostly not from ML. Keep the "how" short and spend the time on what the model can do.*

---

## 1 · Title (~30 s)

"This is a real XCT slice of one of our carbon-fibre coupons."

*The animation runs the noise backwards. At the end the labels appear.*

"Grey is material, amber is pore, and blue is air — here, a drilled hole."

"In one sentence: we train a model that writes new volumes like this one, with the labels already attached."

*The noise is computed in the browser from the real slice. If asked "is this generated?": no, it is a real scan.*

---

## 2 · Why generate porosity? (~50 s)

"In these laminates, porosity is the main defect from manufacture. It makes the part weaker."

*Point at the coupon.*

"The pores sit between the plies, and they are flat."

"To see them we need X-ray CT. A scan takes hours, and then someone must segment it. So we have few volumes."

*Point at the dashed amber volume.*

"We want as many volumes as we need, with the porosity we ask for, and with every pore already labelled."

"That gives us training data for segmentation networks, known pores to test NDT methods, digital twins of real coupons — I show one at the end — and 'what if' questions."

*Do not give a number of hours for a scan. There is no source for one.*

---

## 3 · Destroy with noise, then learn to undo it (~50 s)

*This is the only slide on how diffusion works. Keep it to the picture.*

"Here is the whole idea of a diffusion model."

*Forward half (→).*

"We take a real slice and add a little random noise, many times. At the end nothing of the slice is left. That direction is easy: it is just arithmetic."

*Reverse half (←).*

"The hard direction is back. We train a neural network to remove the noise, one small step at a time."

"Once it can do that, we start from pure noise — noise that no real scan ever made — and after 50 steps we have a new, realistic slice."

*The reverse half is the same noise played backwards, so it returns to the same real slice. Say "this is the direction the model learns", not "this is the model generating".*

*If an ML person asks for detail: backup slides B1–B7 have the formula, the loss, the theory, and why we predict v.*

---

## 4 · Conditioning: telling the model what to make (~50 s)

"A model that makes 'some coupon' is not useful. We want to say what to make: this porosity, these ply angles, this shape. That extra input is called the condition."

"In training, we sometimes hide the condition. So the same network learns both 'a coupon' and 'a coupon with 2 % porosity'."

*Point at the arrows.*

"When we generate, we ask both, and push further in the direction the request points. How hard we push is a knob, s."

"For example: asked for 2 %, s = 1 gives 1.76 %, and s = 2 gives 1.99 %."

---

## 5 · Diffuse in a compressed latent (~50 s)

"3-D volumes are big. Running diffusion directly on voxels costs too much."

"So first we train a second network, an autoencoder, that compresses each 64-voxel cube 48 times. Diffusion runs on the compressed version, and at the end the decoder turns it back into a volume."

"48 times is the same compression as Stable Diffusion, the well-known image generator."

"And the decoder does not only give the grey image. It gives the labels too: material, pore and air."

---

## 6 · Generating microstructures: what is missing (~50 s)

"Others have generated microstructures before. GANs did 3-D rock in 2017. SliceGAN learns 3-D from 2-D images. Diffusion models came after, and in 2026 Naiff and colleagues conditioned one on porosity."

*Point at the right column.*

"What we need is all four at once: whole coupons, control of where the pores go, labels, and the specimen shape."

"Some papers have one of these. The point is all four in one model, for composites."

*Never say "nobody does X".*

---

## 7 · Our data: 81 coupons at 25 µm (~50 s)

"This is a real slice. Every voxel gets one label: material, pore, or air."

*Point at the section on the right.*

"This is a cut through the thickness. You can see the plies, and the pore layers between them. z is always the through-thickness direction."

"The labels come from one fixed method, a local threshold — not by hand. So the model learns to reproduce that segmentation. Its labels are only as good as the reference."

---

## 8 · From volumes to patches, split by panel (~40 s)

"A panel is one manufactured plate. All coupons from one panel go to the same set: 12 panels for training, 3 for validation, 3 for test. So we test on plates the model never saw."

"Each scan is cut into 64-voxel cubes, a new one every 32 voxels."

"We remove cubes that touch the drilled holes. Otherwise the model learns to draw air inside the material."

*If asked: 59 / 11 / 11 volumes, about 2.2 million cubes.*

---

## 9 · Our autoencoder: grey in, grey and labels out (~50 s)

"Three maps go in: the grey image, the pore mask, and the air mask."

"The encoder shrinks each side by 4. The decoder grows it back and gives two things: the grey image, and a label for every voxel."

"How much can we compress? We trained six versions, each smaller. This one is the smallest that still keeps the pores: the rebuilt pores match the real ones with a Dice of 0.93. Smaller than this, the pores start to disappear."

*Dice: overlap of two masks, 1 = perfect. The compression study is on backup B10.*

---

## 10 · What we can control (~60 s)

"This is everything we can tell the network."

*Left column.*

"Maps say where things are: the ply angle at each depth, the shape — what is specimen and what is air — and the six touching cubes, so that a large volume continues across them."

*Right column.*

"Numbers say how much. The most important one is the porosity request."

"And the porosity request is given per 1.6 mm cube. So if we give a map of requests, we control where the pores go."

*If asked how they enter: the maps are stacked as extra input channels; the numbers enter through the normalisation layers.*

---

## 11 · How a large volume is generated (~70 s)

*Go left to right along the boxes.*

"First we say what we want, as maps: the shape and air, the ply angle at every depth, and a porosity value for every cube."

"Then we fill the whole compressed volume with random noise."

"The network can only see one 64-voxel cube at a time. So we work in chunks of 3 × 3 × 3 cubes."

"Inside a chunk, the network looks at each window, we average the windows, and take one step. 50 steps, and the chunk is finished."

"The finished chunk becomes a neighbour for the next one. Chunk after chunk, the whole volume is made."

"At the end the decoder turns everything into grey values and labels."

"For scale: one whole coupon is 900 chunks, and takes about 4.3 hours on one GPU."

---

## 12 · The overlapped sampler: keep, see, redraw (~70 s)

*Follow the five phases of the animation.*

"One problem. The edge of a chunk is made before the next chunk exists. The model does not know what will be there, and it puts too few pores. So you get a thin band with few pores at every join."

"Our fix: the next chunk starts 64 voxels earlier, inside the previous one."

"The first half of that overlap is kept, so the new chunk sees what is already there and continues it."

"The second half is the poor edge. The new chunk draws it again — now with material known on both sides."

"Result: no band. The cost is about 2.6 times the time."

*If asked "why not keep all 64?": the other 32 are the poor edge; keeping them would make the new chunk copy the band.*

---

## 13 · The overlapped sampler removes the band (~60 s)

"This is one real coupon at 2.6 % porosity, generated twice from its maps."

"Each line is the amount of pores along the coupon. The ticks are where the standard chunks join."

"The real scan has nothing special at the ticks. The standard way drops at every tick. The overlapped way has no drops."

*Right plot.*

"Averaged over all joins: real 1.01, standard 0.84, overlapped 0.99. 1 means no band."

*Why a profile and not an image: the band is a few voxels wide, so you cannot see it in a picture.*

*Say if asked: the overlapped sampler is measured on 3 of the 18 evaluation cases so far.*

*If asked "why are the lines smoothed?": both generated twins have a small pattern that repeats every 4 voxels, one compressed cell. A 4-voxel average removes it. The cause is not examined yet.*

---

## 14 · First, measure real against real (~50 s)

"How do we judge if a generated volume is realistic? A distance alone means nothing: how far is far?"

"So we first compare two real samples of the same material with each other. They are not identical either. That distance is the floor — the natural variation."

"Every result is a ratio to that floor. 1 means the generated volume is as close to real as real is to itself. 3 means three times the natural variation."

---

## 15 · Porosity control: you ask, it delivers (~60 s)

"We asked for seven porosity levels, from 0.5 % to 10 %, with different seeds: 42 volumes."

*Left plot.*

"Asked against delivered. All 42 are within half a percentage point of the request. The slope is 1.03."

"It gives a bit less at the lowest level: 0.37 % for 0.5 %."

*Right video.*

"These are all the pores in three of them. The pores sit in layers between the plies, as in real coupons."

---

## 16 · Local control: where the pores go (~60 s)

"Now we ask for different porosity in different places."

"Halves — low on one side, high on the other: the model follows it almost exactly. Slope 0.98."

"Checkerboard — alternating cubes: it follows the pattern, but at half strength. Neighbouring cubes pull towards each other."

*Bottom right.*

"And the knob from the conditioning slide: s = 2 makes it hit the request more exactly."

---

## 17 · Microstructure: 1 to 3× the real floor (~70 s)

"Four ways to compare the structure." *Read the strip at the bottom: one line each.*

"S₂ looks at amount, size and spacing of the pores. Pore size, the sizes of single pores. Ripley K, how much they cluster. FID, how the slices look."

"At 3 % porosity, two of them are at the floor: as good as real against real."

"Pore size is about 2.2 times the floor: the pores are not sized exactly like real ones."

"FID is about 3 times: the slices look smoother. That is the decoder blur."

*Do not hide that 1 % is worse (S₂ 2.42). If asked why, a possible reason (not measured): few pores per volume, so the statistics are noisier.*

---

## 18 · Shape and plies: drawn where we ask (~60 s)

*Left.*

"We ask for a notch and a hole here, and a sphere here. The air comes out where we asked: Dice 0.992."

*Right.*

"Two stacking sequences. The circle shows the ply angle we asked for at this depth, and the elongated pores turn with it."

"We read the angle back from the generated pores: the median error is about 2°. The same reader on real scans has 4°, so that is as good as we can measure."

"One harder case: thin double plies, 18°."

*Do not claim "better than real".*

---

## 19 · Is it copying the training set? (~45 s)

"A fair question for any generator: is it just copying?"

"For each generated cube, we find the closest training cube and the second closest. A copy is very close to one of them and far from the rest."

"None of almost 5 000 generated cubes looks like a copy. We searched all 223 000 training cubes, not a sample."

---

## 20 · Against baselines: control is where we win (~70 s)

"We compared with three other methods. Each one removes one of our ideas." *Point at the strip at the bottom.*

"An older GAN method. Diffusion without compression. And our own model with only one porosity number as its input."

"To be honest: on texture, the porosity-only model is as good as ours, or better."

"The GAN and the uncompressed diffusion are clearly worse."

"Where we win is control: 42 of 42 volumes on target, against 38 of 42. And local maps, plies, shape and whole coupons, which none of them can do."

"These baselines will run again on the final setup, so the numbers will be updated."

---

## 21 · Real or generated? (~50 s)

*Optional: ask "which one is real?" and wait 2 s.*

"Left, all the pores of a real crop at 10.5 %. Right, a generated one asked for 10 % — it gave 10.4 %."

"Both have pore layers between the plies. The generated pores are smoother and less broken up."

*Bottom strip.*

"Slices, grey and labels. The generated grey is smoother: decoder blur."

"The pore-by-pore comparison against real coupons is running now."

*The right panel still says "results coming". Fill it before the talk, or use the sentence above.*

---

## 22 · A real coupon and its generated twin (~90 s)

"Now a whole coupon. The generator got the coupon's shape and air, its ply angles, and its porosity per 1.6 mm cube. It did not get the pores. It drew every pore itself."

*Keys 1–5 change the coupon. Show one or two.*

"Left is the real scan, right is the generated twin — the same slice, the same grey scale."

"First the whole coupon. Then the labels: pores in amber, air in blue. Then a full-resolution window where the real coupon has the most pores."

"Look for the same shape and ply directions — those were given — and pores in the same places and amounts, where only the amount per cube was given."

*The 0.10 % twin has only 0.07 %: the same low-porosity under-delivery as slide 15.*

*If asked why the air looks so flat: in the generated grey, all air is set to one value per volume, the mean grey of its air. Labels unchanged.*

---

## 23 · Limitations and next steps (~60 s)

"Three limits, all measured."

"Blur: the decoder keeps about three quarters of the real sharpness. That is why FID is about 3 times the floor."

"Seams: standard chunks leave a band at the joins. The overlapped sampler removes it, at 2.6 times the time."

"Local control: generated porosity stays similar over only about 1.2 to 1.5 mm. In real coupons, over much more."

"Next: finish the evaluation with the overlapped sampler, the pore-by-pore comparison, the baselines on the final setup, and a sharper decoder."

"Thank you. Questions?"

*1.2–1.5 mm = 48–60 voxels at 25 µm.*

---

## Backup slides (for questions only)

*The divider slide comes first. Press End to jump to the last backup slide, or use the arrows.*

| Slide | Content | Use it when someone asks… |
|---|---|---|
| B1 | Noise formula | "How much noise at each step?" |
| B2 | The loss | "What does the network learn?" |
| B3 | Probability landscape | "Why does this work at all?" |
| B4 | Theory in three results | "What is the theory?" |
| B5 | DDPM vs DDIM | "Why only 50 steps?" |
| B6 | Three output choices | "What does the network output?" |
| B7 | Why we predict v | "Why v and not ε?" |
| B8 | Timeline of diffusion | "Which papers is this built on?" |
| B9 | Five autoencoder loss terms | "How is the autoencoder trained?" |
| B10 | Compression study | "Why 48×? Why 4 channels?" |
| B11 | Neighbour dropout | "How do chunks know about each other?" |
| B12 | Patch, tile, window, chunk | "What is the difference between these?" |
| B13 | Chunk order | "In what order are chunks made?" |
| B14 | Blending windows | "Why no seams between windows?" |
| B15 | Seam plots on 1024² volumes | "Is the seam result only on one coupon?" |
| B16 | The four statistics in detail | "What is S₂ / Ripley K / W₁?" |
| B17 | Input-swap test | "Does the model really use each input?" |
| B18 | The three baselines | "What exactly are the baselines?" |

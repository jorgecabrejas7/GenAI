# PoreGen — group-meeting deck

A 36-slide, ~37-minute talk (speaker-note estimates) for a mixed ML and materials audience: diffusion from
scratch, the state of the art, our data, autoencoder, latent diffusion model,
large-volume assembly, evaluation and results of the final model.

## Build

```bash
python talks/poregen-group/scripts/extract_numbers.py   # numbers + sources -> deck/media/numbers.json
python talks/poregen-group/scripts/render_media.py      # images and videos -> deck/media/ (~10 min, CPU)
python talks/poregen-group/scripts/build_deck.py        # -> deck/index.html
```

Open `deck/index.html` in a browser; it must stay beside `deck/media/`.
Keys: ← → Space navigate · N speaker notes · F fullscreen · E edit text.

## Rules the deck follows

- No repository names on slides or in notes; experiments carry technical titles.
- No term is used before it is explained: each slide carries a "new words" strip that defines every term it introduces.
- Every on-slide number has its source in the speaker notes; the file and JSON
  path of each is in `deck/media/numbers.json`.
- Autoencoder named as in Rombach et al.: f = 4, c = 4, 48x reduction.

## Inputs

- Final model's full evaluation and the overlapped-scheme, conditioning-test,
  baseline and autoencoder-study campaign folders under `runs/campaigns/`
  (paths at the top of `scripts/extract_numbers.py` and `scripts/render_media.py`).
- One real scan and its reference segmentation from `raw_data/MedidasDB/`.

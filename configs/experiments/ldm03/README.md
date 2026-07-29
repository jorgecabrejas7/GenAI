# ldm03 — Classifier-Free Guidance

## Base
`ldm02/base` (Rombach-style sampled latents, `z = μ+σε`, `latent_std ≈ 0.9985`).
Architecture, schedule, optimiser, EMA, data splits, and eval/checkpoint cadence are
identical to ldm01/ldm02.

---

## Conditioning buckets

| Bucket      | Guided? | Null mechanism                          |
|-------------|---------|------------------------------------------|
| Position    | No      | Always-on; never dropped or guided       |
| Porosity    | Yes     | Learned `null_por` (`nn.Parameter(C=512)`) |
| Neighbours  | Yes     | All-UNKNOWN availability (state 2)       |

The null mechanisms are **asymmetric**:
- `0` is a real porosity value, so it cannot be the null. A learned parameter is swapped
  in for the `por_mlp` output when a sample is marked `drop_por=True`.
- Forcing `nb_avail = UNKNOWN` (state 2 for all 6 directions) reproduces the parity-0
  anchor state the model trains on ~50% of the time. No new parameter is needed.

---

## Training dropout rates

Mutually exclusive per sample (drawn from uniform [0, 1]):

| Category                   | Rate  | Effect                                      |
|----------------------------|-------|---------------------------------------------|
| Porosity-only drop         | 0.10  | `drop_por=True`, neighbours unchanged       |
| Joint drop (unconditional) | 0.05  | `drop_por=True` AND all-UNKNOWN neighbours  |
| Neighbours-only drop       | 0.05  | `drop_por=False`, all-UNKNOWN neighbours    |
| Full conditioning          | ~0.80 | No dropout                                  |

---

## Guided sampler formula (DDIM, 3 passes per step)

```
eps_uncond = model(z_t, t, nb, ALL_UNKNOWN, pos, por, drop_por=True )
eps_por    = model(z_t, t, nb, ALL_UNKNOWN, pos, por, drop_por=False)
eps_full   = model(z_t, t, nb, REAL_AVAIL,  pos, por, drop_por=False)

eps = eps_uncond
    + s_por * (eps_por  - eps_uncond)   # porosity guidance
    + s_nb  * (eps_full - eps_por)      # neighbour guidance
```

Position is on in all three calls. At `s_por = s_nb = 1.0` the formula telescopes
to `eps_full` (exact un-guided full-conditional), reproduced with a single model call.

---

## Guidance scale parameters

Configured under `guidance:` in the resolved config:

```yaml
guidance:
  s_por: 1.0   # porosity guidance scale  (sweep: 1.5–4.0 typical)
  s_nb:  1.0   # neighbour guidance scale (sweep: 1.2–2.0 typical)
```

Override at generation time via:

```bash
python scripts/generate_volumes.py \
    --checkpoint runs/ldm/ldm03-run-NNNN-.../checkpoints/best.ckpt \
    --vae-run    runs/vae/r05-run-0001-... \
    --ddim-steps 50 \
    --s-por 2.0 \
    --s-nb  1.5
```

Defaults fall back to the `guidance:` block in `resolved_config.yaml` when the CLI flags
are omitted.

---

## Launch

```bash
python scripts/train_ldm.py run ldm03/base
```

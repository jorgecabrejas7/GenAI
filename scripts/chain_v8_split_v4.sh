#!/usr/bin/env bash
# THE WHOLE PAPER, REBUILT ON data/split_v4.  PREPARED — NOT LAUNCHED.
#
# Same architecture as split_v3 throughout. One thing is new and it is a
# SIMPLIFICATION, not an addition: per-face neighbour dropout runs from step 0
# of the 130k base training (ldm06/facedrop_from_start), so there is no
# fine-tune stage. Campaign 29 established that the dropout and not the
# fine-tuning clears the chunk-plane band — a matched control at drop_nb_face 0
# left the -8 slab at 0.187 against the untouched 0.192 — so the correction can
# move to the start without losing its cause.
#
# LAUNCH ONLY ON THE AUTHOR'S WORD, and only once data/split_v4 exists. The
# chain refuses to start without it.
#
# STOPPING. Never signal it. Touch runs/campaigns/STOP_CHAIN and it halts at
# the next stage boundary with every job intact. A trainer signalled mid-step
# lost 22 steps AND its whole U_f finalisation on 2026-09-27.
#
# ETAs are the split_v3 wall times from docs/PAPER_RUNBOOK.md, which are
# measured on this GB10. They are a guide for planning, not a budget the chain
# enforces: split_v4 has its own patch count and the rungs stop on their own
# early-stopping criterion, not on the clock.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
STOP="$REPO/runs/campaigns/STOP_CHAIN"
SPLIT="${POREGEN_SPLIT:-split_v4}"
export POREGEN_SPLIT="$SPLIT"
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75
#: Campaign directories for this build. NEVER the split_v3 ones: those hold
#: every published number and a rerun must not be able to overwrite them.
SUF="-v4"
cd "$REPO" || exit 1
say() { printf '%s  V8 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mem_line() {
    local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
    local cuda; cuda=$(python - <<'PY' 2>/dev/null || echo "n/a"
import torch
if torch.cuda.is_available():
    f,t = torch.cuda.mem_get_info(); print(f"{f/2**30:.1f}/{t/2**30:.1f} GiB free")
else: print("no cuda")
PY
)
    printf 'host available %s GB; cuda %s' "$avail" "$cuda"
}
gate() {
    if [ -e "$STOP" ]; then
        say "STOP FILE PRESENT — halting cleanly before: $*"
        say "  every job is intact; remove $STOP and rerun to continue."
        exit 0
    fi
}
step() {   # a stage with no +5min watch: short, or CPU
    local label="$1" tag="$2" eta="$3"; shift 3
    gate "$label"
    say "$label start (ETA ${eta})"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v8_$tag.log" 2>&1
    local rc=$?
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    [ "$rc" -ne 0 ] && { say "STOP NOTICE: $tag failed — see $S/v8_$tag.log"; exit "$rc"; }
    return 0
}
run_watched() {  # a training stage: watched for host pressure at +5 min
    local label="$1" tag="$2" eta="$3"; shift 3
    gate "$label"
    say "$label start (ETA ${eta})"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v8_$tag.log" 2>&1 &
    local pid=$!
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
          say "$tag +5min: $(mem_line)"
          if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
              say "$tag STOPPING: host available ${avail} GB under ${MIN_AVAIL_GB} GB"
              kill -TERM "$pid" 2>/dev/null
          fi
      fi ) &
    local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    [ "$rc" -ne 0 ] && { say "STOP NOTICE: $tag failed — see $S/v8_$tag.log"; exit "$rc"; }
    return 0
}
smoke() {
    local exp="$1" tag="$2"
    say "SMOKE $exp (cap $SMOKE_FRACTION, $SPLIT)"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION \
        choom -n 1000 -- python scripts/analysis/vae_smoke_step.py --experiment "$exp" \
        > "$S/v8_smoke_$tag.log" 2>&1
    local rc=$?
    say "SMOKE $exp rc=$rc — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/v8_smoke_$tag.log" | head -2 | tr '\n' ' ')"
    return $rc
}

# ── 0. refuse to start on a dataset that is not there ──────────────────────
say "=== v8: the paper on $SPLIT === $(mem_line)"
if [ ! -d "$REPO/data/$SPLIT" ]; then
    say "STOP NOTICE: data/$SPLIT does not exist. Nothing is launched."
    exit 1
fi
for f in patch_index.parquet splits.json volumes.zarr; do
    if [ ! -e "$REPO/data/$SPLIT/$f" ]; then
        say "STOP NOTICE: data/$SPLIT/$f is missing — the build is incomplete."
        exit 1
    fi
done
say "data/$SPLIT present; POREGEN_SPLIT=$SPLIT exported for every stage"

# ── 1. the VAE, exactly rf-8 ───────────────────────────────────────────────
if ! smoke r08/reduction-factor-8 rf8; then
    say "STOP NOTICE: rf-8 does not fit at its own batch under the cap."
    exit 1
fi
run_watched "r08 rf-8 on $SPLIT" r08_rf8 "30 h" \
    python scripts/train_vae.py run r08/reduction-factor-8
VAE=$(ls -dt "$REPO"/runs/vae/r08-run-*archv2-conv_noattn_dualbranch_cls-z8-*/ 2>/dev/null | head -1)
say "VAE = $(basename "${VAE%/}")"

# ── 2. the latent store and its conditioning ───────────────────────────────
step "latent store" latents "6 h" \
    python scripts/build_latent_dataset.py --checkpoint "${VAE%/}/best.ckpt" \
    --output "data/$SPLIT/latents_r08z8"
step "conditioning" conditioning "1 h" \
    python scripts/build_conditioning.py --store "data/$SPLIT/latents_r08z8"

# ── 3. the LDM, with the dropout from step 0 — no fine-tune stage ──────────
run_watched "ldm06 facedrop_from_start (130k)" ldm06 "16 h" \
    python scripts/train_ldm.py run ldm06/facedrop_from_start
LDM=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)
say "LDM = $(basename "${LDM%/}")"

# ── 4. the paper's campaigns, in the runbook's order ───────────────────────
# real-floor FIRST in every campaign: every ratio is measured against it.
C18="$REPO/runs/campaigns/18-eval-v4-final$SUF"
for A in real-floor sampler porosity_global porosity_local microstructure \
         geometry surface layup assembly multichunk cfg; do
    step "c18 generate $A" "c18_gen_$A" "varies" \
        python -m poregen.eval_v4.cli generate "$A" --model "${LDM%/}" \
        --ckpt latest --weights ema --out "$C18"
    step "c18 measure $A" "c18_meas_$A" "varies" \
        python -m poregen.eval_v4.cli measure "$A" --root "$C18"
done
step "c18 report" c18_report "1 min" \
    python -m poregen.eval_v4.cli report --root "$C18"

# ── 5. the three baselines ─────────────────────────────────────────────────
step "slicegan train" slicegan "11 h" python scripts/train_slicegan.py
step "slicegan measure" slicegan_meas "8 min" \
    python -m poregen.eval_v4.cli measure slicegan \
    --root "$REPO/runs/campaigns/22-slicegan-baseline$SUF"
step "ddpm3d train" ddpm3d "24 h" \
    python scripts/train_ddpm3d.py --data-root "data/$SPLIT"
step "ddpm3d measure" ddpm3d_meas "8 min" \
    python -m poregen.eval_v4.cli measure ddpm3d \
    --root "$REPO/runs/campaigns/23-ddpm3d-baseline$SUF"
run_watched "ldm25 phi-only (40k)" ldm25 "19 h" \
    python scripts/train_ldm.py run ldm25/phi_only
LDM25=$(ls -dt "$REPO"/runs/ldm/ldm25-run-*/ 2>/dev/null | head -1)
C25="$REPO/runs/campaigns/25-ldm-phi-only$SUF"
for A in real-floor porosity_global microstructure cfg; do
    step "c25 generate $A" "c25_gen_$A" "varies" \
        python -m poregen.eval_v4.cli generate "$A" --model "${LDM25%/}" \
        --ckpt 40000 --weights ema --out "$C25"
    step "c25 measure $A" "c25_meas_$A" "varies" \
        python -m poregen.eval_v4.cli measure "$A" --root "$C25"
done
step "c25 report" c25_report "1 min" \
    python -m poregen.eval_v4.cli report --root "$C25"

# ── 6. the ablation ────────────────────────────────────────────────────────
C24="$REPO/runs/campaigns/24-ablation$SUF"
step "c24 generate" c24_gen "6 h" \
    python -m poregen.eval_v4.cli generate ablation --model "${LDM%/}" \
    --ckpt latest --weights ema --out "$C24"
step "c24 measure" c24_meas "20 min" \
    python -m poregen.eval_v4.cli measure ablation --root "$C24"
step "c24 report" c24_report "1 min" \
    python -m poregen.eval_v4.cli report --root "$C24"

# ── 7. the budget-matched row ──────────────────────────────────────────────
# ldm06 at the phi-only baseline's own 40 000 steps. On split_v4 the base run
# IS the facedrop-from-start run, so the step checkpoint comes from it.
C27="$REPO/runs/campaigns/27-budget-matched-40k$SUF"
for A in real-floor porosity_global microstructure sampler; do
    step "c27 generate $A" "c27_gen_$A" "varies" \
        python -m poregen.eval_v4.cli generate "$A" --model "${LDM%/}" \
        --ckpt 40000 --weights ema --out "$C27"
    step "c27 measure $A" "c27_meas_$A" "varies" \
        python -m poregen.eval_v4.cli measure "$A" --root "$C27"
done
step "c27 report" c27_report "1 min" \
    python -m poregen.eval_v4.cli report --root "$C27"

# ── 8. downstream utility ──────────────────────────────────────────────────
step "c14 downstream" c14 "10 h" \
    python scripts/analysis/downstream_utility.py \
    --out "$REPO/runs/campaigns/14-downstream-utility$SUF"

say "BLOCK COMPLETE — the paper is rebuilt on $SPLIT — $(mem_line)"
say "TOTAL ETA from the split_v3 wall times: about 130 h of card time."

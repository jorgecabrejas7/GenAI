#!/usr/bin/env bash
# The GPU order after the facedrop trial, as the supervisor set it:
#
#   campaign 18 regen -> stress DDIM-50 -> ood_conditioning
#   -> downstream_utility -> stress DDIM-200
#
# and then it STOPS. The decoder fine-tune is next in that order, but its gate
# (`runs/campaigns/decoder_ft_go`, which `facedrop_post.sh` is already blocked
# on) is the author's go-ahead and not an ordering device, so this script never
# creates it. It says the gate is next and exits.
#
# TWO WAITS, and neither is a timeout.
#
#   1. `facedrop_post.sh` must have reached its own gate. Until then it is
#      running the fd_baseline / fd_a2s trial arms and the mixed-set rim test
#      on the card, and two generation jobs on one 121 GB unified pool is how a
#      CUDA allocation fails while `free` still reports tens of GB.
#   2. `runs/campaigns/regen_go` must exist. It is a shell fragment naming
#      CKPT_RUN and SAMPLER, e.g.
#
#         CKPT_RUN=runs/ldm/ldm06-run-0002-20260912-085535-z8-c128-bs256-lr2e-05
#         SAMPLER=production
#
#      Those two choices come from the trial table, which does not exist yet
#      when this script is armed. Guessing them would spend a day of GPU on a
#      checkpoint/sampler pairing nobody chose.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
GO="$REPO/runs/campaigns/regen_go"
FT_GO="$REPO/runs/campaigns/decoder_ft_go"
FD_LOG="$REPO/runs/campaigns/09-r08-latent-sweep/post_ldm06.log"
C18="$REPO/runs/campaigns/18-eval-v4-final"
C19="$REPO/runs/campaigns/19-stress-geometry"
C20="$REPO/runs/campaigns/20-ood-conditioning"
C14="$REPO/runs/campaigns/14-downstream-utility"
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/eval_queue.log"
mkdir -p "$S" "$C14" "$C20"; cd "$REPO" || exit 1
say() { printf '%s  EVALQ %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

say "armed"

# ── wait 1: facedrop_post must be parked at its own gate ────────────────────
# The log line it writes there is the signal. Process absence is NOT the
# signal: facedrop_post may not have started its watch yet when this arms.
while ! grep -q "GATE blocked\|GATE released\|POST COMPLETE" "$FD_LOG" 2>/dev/null; do
    sleep 120
done
say "facedrop_post has reached its gate — the card is free"

# ── wait 2: the regeneration choice ─────────────────────────────────────────
if [ ! -e "$GO" ]; then
    say "waiting for $GO (CKPT_RUN and SAMPLER, from the trial table)"
    while [ ! -e "$GO" ]; do sleep 60; done
fi
# shellcheck disable=SC1090
. "$GO"
: "${CKPT_RUN:?regen_go must set CKPT_RUN}"
: "${SAMPLER:?regen_go must set SAMPLER}"
say "GO: CKPT_RUN=$CKPT_RUN SAMPLER=$SAMPLER"

# ── 1. campaign 18 — the whole eval-v4 suite on the chosen pairing ──────────
say "REGEN start (campaign 18)"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" bash scripts/regen_eval_v4.sh \
    > "$S/evalq_regen.log" 2>&1
say "REGEN done rc=$? -> $C18"

# ── 2. campaign 19 — the screening pass ────────────────────────────────────
say "STRESS ddim50 start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" PASS=ddim50 \
    bash scripts/run_stress_geometry.sh > "$S/evalq_stress50.log" 2>&1
say "STRESS ddim50 done rc=$? -> $C19"

# ── 3. campaign 20 — conditioning out of distribution ──────────────────────
# One pass, DDIM-50 throughout, so there is no second visit to schedule. Per
# case for the same reason as the stress requests: a case the sampler cannot
# hold costs that case and not the 35 after it.
#
# It runs on the PRODUCTION sampler whatever SAMPLER says. All four axes it
# moves are in the CONDITIONING, and generating them under an overlap sampler
# would mean a failure at phi 0.20 could be the request or could be the
# assembly, with no way to tell which.
say "OOD start (campaign 20)"
mapfile -t OOD_CASES < <(python - <<'PYCASES'
from poregen.eval_v4.cases import build_cases
for c in build_cases("ood_conditioning"):
    print(c.name)
PYCASES
)
say "OOD ${#OOD_CASES[@]} cases"
for c in "${OOD_CASES[@]}"; do
    python -m poregen.eval_v4.cli generate ood_conditioning \
        --model "$CKPT_RUN" --ckpt "$CKPT" --out "$C20" --only "$c" \
        > "$S/evalq_ood_${c}.log" 2>&1
    say "OOD GEN $c rc=$?"
done
python -m poregen.eval_v4.cli measure ood_conditioning --root "$C20" \
    > "$S/evalq_ood_measure.log" 2>&1
say "OOD MEASURE rc=$?"
python -m poregen.eval_v4.cli report --root "$C20" --assessment ood_conditioning \
    > "$S/evalq_ood_report.log" 2>&1
say "OOD done rc=$? -> $C20/ood_conditioning/findings.md"

# ── 4. downstream utility, on the set campaign 18 just wrote ───────────────
# It reads the `sampler`, `porosity_global`, `microstructure` and `surface`
# volumes and REFUSES to start, naming the missing cases, if any are absent —
# which is the check that says the regen finished, so it is not repeated here.
say "DOWNSTREAM start (campaign-root $C18)"
python scripts/analysis/downstream_utility.py --campaign-root "$C18" \
    --out "$C14" > "$S/evalq_downstream.log" 2>&1
say "DOWNSTREAM done rc=$? -> $C14"

# ── 5. campaign 19 — the DDIM-200 pass ─────────────────────────────────────
say "STRESS ddim200 start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" PASS=ddim200 \
    bash scripts/run_stress_geometry.sh > "$S/evalq_stress200.log" 2>&1
say "STRESS ddim200 done rc=$? -> $C19"

say "EVAL QUEUE COMPLETE — the decoder-ft gate is next."
say "facedrop_post.sh is blocked on $FT_GO and this script does not create it:"
say "that gate is the author's go-ahead, not an ordering device."

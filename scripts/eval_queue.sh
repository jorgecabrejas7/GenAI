#!/usr/bin/env bash
# The GPU order after the facedrop trial, as the supervisor set it:
#
#   campaign 18 regen -> stress DDIM-50 -> downstream_utility -> stress DDIM-200
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
C14="$REPO/runs/campaigns/14-downstream-utility"
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/eval_queue.log"
mkdir -p "$S" "$C14"; cd "$REPO" || exit 1
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

# ── 3. downstream utility, on the set campaign 18 just wrote ───────────────
# It reads the `sampler`, `porosity_global`, `microstructure` and `surface`
# volumes and REFUSES to start, naming the missing cases, if any are absent —
# which is the check that says the regen finished, so it is not repeated here.
say "DOWNSTREAM start (campaign-root $C18)"
python scripts/analysis/downstream_utility.py --campaign-root "$C18" \
    --out "$C14" > "$S/evalq_downstream.log" 2>&1
say "DOWNSTREAM done rc=$? -> $C14"

# ── 4. campaign 19 — the DDIM-200 pass ─────────────────────────────────────
say "STRESS ddim200 start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" PASS=ddim200 \
    bash scripts/run_stress_geometry.sh > "$S/evalq_stress200.log" 2>&1
say "STRESS ddim200 done rc=$? -> $C19"

say "EVAL QUEUE COMPLETE — the decoder-ft gate is next."
say "facedrop_post.sh is blocked on $FT_GO and this script does not create it:"
say "that gate is the author's go-ahead, not an ordering device."

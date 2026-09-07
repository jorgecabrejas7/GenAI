#!/usr/bin/env bash
# Everything the GPU does after ldm06/base exits, in one chain.
#
# Armed BEFORE ldm06/base finishes so nothing waits on a human seeing an exit —
# a missed process exit already cost 3 h 31 m of GPU once in this campaign.
#
# Order (D43):
#   [gate] eval_v4 generate, all assessments, --save-latents, ORIGINAL decoder
#   decoder-ft            (r08/decoder-ft, option 1)
#   decoder_ft_redecode   (val arm + generated arm on the saved eval-v4 latents)
#   gate table            written to the campaign, flagged for the supervisor
#   owed rung reports     (base, rf-8, rf-32, rf-4)
#   rf-2, rf-64           (the bracket rungs, lowest priority)
#
# The eval-v4 block is gated on runs/campaigns/ldm06_go, which the supervisor
# creates after reading the 40k gate. Absent when ldm06 exits, the block is
# skipped and the chain continues at decoder-ft — the supervisor triggers
# generation separately. The file is checked ONCE, when ldm06 exits, so a late
# touch does not retro-insert hours of generation ahead of the fine-tune.
#
# Usage:  bash scripts/ldm06_post.sh
set -uo pipefail          # NOT -e: one failed stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
EVAL_CAMP="$REPO/runs/campaigns/12-eval-v4"
FT_CAMP="$REPO/runs/campaigns/11-decoder-ft"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/post_ldm06.log"
GO="$REPO/runs/campaigns/ldm06_go"
ASSESSMENTS=(sampler microstructure porosity_global porosity_local geometry layup assembly cfg)
OWED=(base reduction-factor-8 reduction-factor-32 reduction-factor-4)
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)

mkdir -p "$CAMP" "$FT_CAMP" "$SCRATCH"
cd "$REPO" || exit 1
say() { printf '%s  POST %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  POST %s\n' "$(date -Is)" "$*"; }

run_dir_for() {
    local want="r08/$1" d got
    for d in $(ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null); do
        [ -f "${d}best.ckpt" ] || continue
        got=$(python - "$d" <<'PY'
import sys, json, pathlib
try:
    print(json.loads((pathlib.Path(sys.argv[1]) / "run_metadata.json").read_text())
          .get("experiment_id", ""))
except Exception:
    print("")
PY
)
        [ "$got" = "$want" ] && { printf '%s' "$d"; return 0; }
    done
    return 1
}

# ── 0. wait for ldm06/base ────────────────────────────────────────────────────
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)
say "armed; watching ${LDM_RUN:-<no ldm06 run>}"
while pgrep -f "scripts/train_ldm\.py run " >/dev/null 2>&1; do sleep 60; done
say "ldm06/base has exited"
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)

# ── 1. eval v4 generation, gated ──────────────────────────────────────────────
# Read the gate ONCE, here, so a file touched later cannot push hours of
# generation in front of the fine-tune that is already running.
if [ -e "$GO" ]; then
    say "GO present -> eval v4 generation on $LDM_RUN (original r08 decoder, --save-latents)"
    for a in "${ASSESSMENTS[@]}"; do
        say "EVALV4 generate $a start"
        python -m poregen.eval_v4.cli generate "$a" \
            --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents \
            > "$SCRATCH/evalv4_gen_${a}.log" 2>&1
        say "EVALV4 generate $a done rc=$?"
    done
else
    say "GO absent -> skipping eval v4 generation; supervisor triggers it separately"
fi

# ── 2. decoder fine-tune, option 1 ────────────────────────────────────────────
say "DECODER-FT start (r08/decoder-ft)"
python scripts/train_vae.py run r08/decoder-ft > "$SCRATCH/decoder_ft.log" 2>&1
ft_rc=$?
say "DECODER-FT done rc=$ft_rc"
# The run dir is named from experiment.name ("r08"), not the variant, so the
# fine-tune lands in r08-run-NNNN-... beside the sweep rungs. Resolve it by
# experiment_id, the same way every other rung is resolved here.
FT_RUN=$(run_dir_for decoder-ft)
say "DECODER-FT run dir: ${FT_RUN:-<none>}"

# ── 3. re-decode comparison ───────────────────────────────────────────────────
BASE_CKPT="$(run_dir_for reduction-factor-8)best.ckpt"
if [ "$ft_rc" -eq 0 ] && [ -n "$FT_RUN" ] && [ -f "${FT_RUN}best.ckpt" ]; then
    say "REDECODE start (baseline $BASE_CKPT)"
    python scripts/analysis/decoder_ft_redecode.py \
        --baseline "$BASE_CKPT" --finetuned "${FT_RUN}best.ckpt" \
        --latents "$EVAL_CAMP/*/*/latents.npy" \
        --out "$FT_CAMP/redecode" \
        > "$SCRATCH/redecode.log" 2>&1
    say "REDECODE done rc=$?"
    say "GATE TABLE -> $FT_CAMP/redecode/results.json — SEND TO SUPERVISOR"
else
    say "REDECODE skipped: fine-tune rc=$ft_rc, run=${FT_RUN:-<none>}"
fi

# ── 4. the owed full-split rung reports ───────────────────────────────────────
for exp in "${OWED[@]}"; do
    d=$(run_dir_for "$exp")
    if [ -z "$d" ]; then say "OWED $exp skipped: no run directory"; continue; fi
    say "OWED $exp report start"
    python scripts/analysis/r08_rung_report.py --run "$d" > "$SCRATCH/report_${exp}.log" 2>&1
    say "OWED $exp report done rc=$?"
done
say "COMPARE start (final paper table)"
python scripts/analysis/r08_rung_report.py --compare > "$SCRATCH/r08_compare_final.log" 2>&1
say "COMPARE done rc=$?"

# ── 5. the bracket rungs, last ────────────────────────────────────────────────
for exp in "${TAIL_RUNGS[@]}"; do
    say "START r08/$exp"
    python scripts/train_vae.py run "r08/$exp" > "$SCRATCH/r08_${exp}.log" 2>&1
    rc=$?
    say "DONE r08/$exp rc=$rc"
    [ "$rc" -ne 0 ] && say "FAIL r08/$exp rc=$rc — chain continues"
done

say "CHAIN COMPLETE"

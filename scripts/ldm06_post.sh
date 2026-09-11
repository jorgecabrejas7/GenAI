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
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/post_ldm06.log"
GO="$REPO/runs/campaigns/ldm06_go"
ASSESSMENTS=(sampler microstructure porosity_global porosity_local geometry surface layup assembly cfg)

mkdir -p "$CAMP" "$SCRATCH"
cd "$REPO" || exit 1
say() { printf '%s  POST %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  POST %s\n' "$(date -Is)" "$*"; }

# ── 0. wait for ldm06/base ────────────────────────────────────────────────────
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)
say "armed; watching ${LDM_RUN:-<no ldm06 run>}"
# "No training process" is NOT the same as "training finished". It also means
# the run was stopped deliberately for a minute — to change a config, say — and
# a runner that cannot tell the difference will start ten hours of generation
# on a card that is about to go back to training. That happened: a two-minute
# stop to widen sample_grid launched the whole eval-v4 set against a step-77000
# checkpoint, halving training throughput for eleven hours to produce volumes
# that were never the paper set.
#
# So the exit test is the RUN'S OWN PROGRESS: gone from the process table AND
# the last logged step has reached total_steps. That is true only when training
# is really over, and it survives a crash, which a marker file written on a
# clean exit would not.
reached_total_steps() {
    python "$REPO/scripts/_ldm_reached_total.py" "$LDM_RUN"
}

while true; do
    if pgrep -f "scripts/train_ldm\.py (run|resume) " >/dev/null 2>&1; then
        sleep 60
        continue
    fi
    if reached_total_steps; then
        break
    fi
    say "no training process, but the run has not reached total_steps — treating this as a pause, not the end"
    sleep 60
done
say "ldm06/base has exited"
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)

# ── 0a. the full diagnostic, once, with the card to itself ────────────────────
# The every-20k diagnostic was disarmed: on this unified-memory box a CUDA
# context could not reliably be created while the run held ~44 GB RSS and its
# dataloader kept the page cache full, so the 60k attempt died at its first
# CUDA call and each retry cost 10-15 minutes of training throughput. The
# trend is covered by the trainer's OWN gen-eval every 2k and sample volumes
# every 10k, which run inside its process and need no second context.
#
# So it runs here instead, first, when the memory is free — both references,
# per-bucket porosity, the boundary material probe and the inset-box surface
# sample, at the final weights.
say "DIAG final start (both references, inset box, DDIM-200)"
python scripts/diag_ldm_samples.py --run-dir "$LDM_RUN" --ddim200 \
    > "$SCRATCH/ldm06_diag_final.log" 2>&1
diag_rc=$?
if [ "$diag_rc" -eq 0 ]; then
    say "DIAG final done rc=0 -> $LDM_RUN/convergence_check.jsonl — SEND TABLE TO SUPERVISOR"
else
    say "DIAG final FAILED rc=$diag_rc — NOTHING MEASURED; see $SCRATCH/ldm06_diag_final.log"
fi

# ── 0b. refresh the sampled-latent std reference ──────────────────────────────
# Deferred to here on purpose. The store's channel_stats.sampled_std is
# PROVISIONAL at 4000 rows, and a 50k random-read pass over latents.bin
# competes with the training dataloader — doing it live slowed ldm06 from 1.74
# to 2.43 s/step. With training finished the read is free.
say "STDREF refresh start (50k rows, CPU)"
python scripts/analysis/latent_std_reference.py \
    --store "$REPO/data/split_v3/latents_r08z8" --n 50000 \
    > "$SCRATCH/latent_std_reference.log" 2>&1
say "STDREF refresh done rc=$?"

# ── 1. eval v4 generation, gated ──────────────────────────────────────────────
# Read the gate ONCE, here, so a file touched later cannot push hours of
# generation in front of the fine-tune that is already running.
if [ -e "$GO" ]; then
    say "GO present -> eval v4 generation on $LDM_RUN (original r08 decoder, --save-latents)"
    # The real floor first, and on CPU. Every generated table is read against
    # it, and a table with no floor row says only that a number exists — the
    # surface roughness in particular is meaningless without the real one,
    # since no real surface is perfectly flat either.
    say "EVALV4 real-floor start (CPU; small, large, micro, surface)"
    python -m poregen.eval_v4.cli real-floor --root "$EVAL_CAMP" \
        --shapes small large micro surface \
        > "$SCRATCH/evalv4_real_floor.log" 2>&1
    say "EVALV4 real-floor done rc=$?"
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

# ── 1b. everything after generation belongs to ldm06_post_tail.sh ────────────
# The gate, the decoder fine-tune, the re-decode, the owed rung reports and the
# bracket rungs USED TO LIVE HERE.  They cannot any more: the tail script stops
# this runner at the `surface` boundary to move layup to the end, so anything
# written below that point would never run.  Two scripts that both launch
# r08/decoder-ft is worse than one — the fine-tune would run twice if a kill
# ever missed.  ldm06_post_tail.sh owns the queue from `surface` onward.
say "GENERATION COMPLETE — ldm06_post_tail.sh owns the queue from here"

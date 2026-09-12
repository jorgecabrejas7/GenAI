#!/usr/bin/env bash
# Regenerate the whole eval-v4 campaign into a NEW directory.
#
# PREPARED, NOT LAUNCHED.  Campaign 12 is never written to: it is the record of
# what ldm06 @ 119k produced under the production sampler, and the paper's
# numbers are currently read from it.  This writes campaign 18.
#
# Two choices, and they are independent:
#
#   CKPT_RUN   which weights.  ldm06 run-0001 latest = 130k EMA, the run that
#              finished; or the facedrop fine-tune, which puts the sampler's
#              chunk-frontier neighbour state into the training distribution.
#   SAMPLER    production (no overlap) or a2s (overlap 64, 32 pinned, the free
#              part written from the successor's own prediction).  a2s clears
#              the chunk-plane band on 130k without retraining; on a facedrop
#              checkpoint it may be unnecessary, which is the point of being
#              able to run either.
#
# The sampler override reaches every case on the production neighbour path.
# The joint and teacher-forced arms of assembly_modes keep their own settings —
# those settings are what those arms ARE, and overriding them would turn a
# four-way comparison into four copies of one thing.
#
# Usage:
#   CKPT_RUN=runs/ldm/ldm06-run-0001-... SAMPLER=production bash scripts/regen_eval_v4.sh
#   CKPT_RUN=runs/ldm/ldm06-run-0002-... SAMPLER=a2s        bash scripts/regen_eval_v4.sh
set -uo pipefail          # NOT -e: a failing stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
OUT="${OUT:-$REPO/runs/campaigns/18-eval-v4-final}"
SRC="$REPO/runs/campaigns/12-eval-v4"
SCRATCH="${SCRATCH:-/tmp/regen_eval_v4}"
CKPT="${CKPT:-latest}"
SAMPLER="${SAMPLER:-production}"
: "${CKPT_RUN:?set CKPT_RUN to the ldm run directory to generate from}"

#: The a2s sampler, as the trial ran it.
A2S_OVERLAP=64
A2S_PINNED=32
A2S_WRITE=successor
#: The caps the memorisation pass is held to, as in the ldm06 queue.
MEMO_MAX_MIN=60
MEMO_MAX_GB=20

case "$SAMPLER" in
    production) OVERRIDE=() ;;
    a2s)        OVERRIDE=(--chunk-overlap "$A2S_OVERLAP"
                          --chunk-overlap-pinned "$A2S_PINNED"
                          --chunk-overlap-write "$A2S_WRITE") ;;
    *) echo "SAMPLER must be 'production' or 'a2s', got '$SAMPLER'" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$SCRATCH"; cd "$REPO" || exit 1
LOG="$OUT/regen.log"
say() { printf '%s  REGEN %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

say "checkpoint run : $CKPT_RUN ($CKPT)"
say "sampler        : $SAMPLER ${OVERRIDE[*]:-(no override)}"
say "output         : $OUT"

# ── 0. the real floor ───────────────────────────────────────────────────────
# Its inputs are the split_v3 TEST panels and the detector calibration — none
# of which depend on the checkpoint or the sampler — so recutting it would burn
# 7 minutes of CPU to produce the same arrays. Linked, not copied, and the link
# is what says the two campaigns share one floor.
if [ -d "$SRC/real_floor" ] && [ ! -e "$OUT/real_floor" ]; then
    ln -s "$SRC/real_floor" "$OUT/real_floor"
    say "real_floor symlinked from campaign 12 (same panels, same detector)"
elif [ ! -e "$OUT/real_floor" ]; then
    say "REAL-FLOOR cut start (CPU)"
    python -m poregen.eval_v4.cli real-floor --root "$OUT" \
        --shapes small large micro surface > "$SCRATCH/real_floor.log" 2>&1
    say "REAL-FLOOR done rc=$?"
fi

# ── 1. generation ───────────────────────────────────────────────────────────
# layup last: 9 cases at 1024x1024x192 and DDIM-200 is about a third of the
# whole campaign, and the inspection pack should exist before it starts.
GEN_ORDER=(sampler microstructure porosity_global porosity_local geometry
           surface multichunk assembly cfg assembly_modes layup)
for a in "${GEN_ORDER[@]}"; do
    say "GEN $a start"
    python -m poregen.eval_v4.cli generate "$a" \
        --model "$CKPT_RUN" --ckpt "$CKPT" --out "$OUT" --save-latents \
        "${OVERRIDE[@]}" > "$SCRATCH/gen_${a}.log" 2>&1
    say "GEN $a done rc=$?"
    if [ "$a" = "cfg" ]; then
        say "INSPECT pack start (before layup, so inspection overlaps it)"
        python scripts/analysis/eval_v4_inspection_pack.py --root "$OUT" \
            > "$SCRATCH/inspection.log" 2>&1
        say "INSPECT pack done rc=$? -> $OUT/inspection"
    fi
done

# ── 2. measure ──────────────────────────────────────────────────────────────
# field_stats is MEASURE-ONLY: it re-reads what porosity_local and multichunk
# wrote and has no generate verb, so a loop over the case list drops it.
# microstructure carries the FULL memorisation pass and is capped; the card is
# idle by now, which is the condition that pass requires.
MEASURERS=$(python -c "
from poregen.eval_v4.measure import MEASURERS
print(' '.join(sorted(MEASURERS)))")
for a in $MEASURERS; do
    if [ "$a" = microstructure ]; then
        say "MEASURE microstructure start (carries the FULL memorisation pass, cap ${MEMO_MAX_MIN} min)"
        timeout --signal=KILL "$(( MEMO_MAX_MIN * 60 ))" \
            python -m poregen.eval_v4.cli measure microstructure --root "$OUT" \
            > "$SCRATCH/measure_${a}.log" 2>&1
        rc=$?
        [ "$rc" -eq 137 ] && say "MEASURE microstructure STOPPED at the cap — REPORT" \
                          || say "MEASURE microstructure done rc=$rc"
        continue
    fi
    say "MEASURE $a start"
    python -m poregen.eval_v4.cli measure "$a" --root "$OUT" \
        > "$SCRATCH/measure_${a}.log" 2>&1
    say "MEASURE $a done rc=$?"
done

# ── 3. report, the final pack, and the notebook ─────────────────────────────
say "REPORT start"
python -m poregen.eval_v4.cli report --root "$OUT" > "$SCRATCH/report.log" 2>&1
say "REPORT done rc=$?"
say "INSPECT pack (final) start"
python scripts/analysis/eval_v4_inspection_pack.py --root "$OUT" \
    > "$SCRATCH/inspection_final.log" 2>&1
say "INSPECT pack (final) done rc=$? -> $OUT/inspection"
say "NOTEBOOK start"
python scripts/analysis/build_eval_v4_notebook.py --root "$OUT" \
    > "$SCRATCH/notebook.log" 2>&1
say "NOTEBOOK done rc=$? -> $OUT/inspect_eval_v4.ipynb"

say "REGEN COMPLETE — $OUT"

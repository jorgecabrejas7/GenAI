#!/usr/bin/env bash
# Owns the eval-v4 queue from the `surface` stage onward.
#
# WHY IT TAKES OVER. layup is 9 cases at 1024x1024x192 and DDIM-200 — about 6
# hours, 48% of the remaining generation. The user wants to inspect the volumes
# while that runs, so layup moves to the END and the inspection pack is built
# before it. The main runner's order is fixed (… surface layup assembly cfg),
# so the only way to reorder is to stop it at a stage boundary and run the rest
# here. assembly and cfg come AFTER layup in the runner's list, so they would
# be skipped too — they are picked up here.
#
# Order: [runner finishes surface] -> stop runner -> multichunk -> assembly ->
# cfg -> INSPECTION PACK (user inspects) -> the five -> layup (full 3 seeds) ->
# measure/report -> pack again.
#
# Usage:  bash scripts/ldm06_post_tail.sh
set -uo pipefail          # NOT -e: a failing stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
EVAL_CAMP="$REPO/runs/campaigns/12-eval-v4"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/post_ldm06.log"
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)
#: The runner stage immediately before layup.
STOP_AFTER=surface

mkdir -p "$SCRATCH"; cd "$REPO" || exit 1
say() { printf '%s  TAIL %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  TAIL %s\n' "$(date -Is)" "$*"; }

gen() {
    local a="$1"
    say "GEN $a start"
    python -m poregen.eval_v4.cli generate "$a" \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents \
        > "$SCRATCH/tail_gen_${a}.log" 2>&1
    say "GEN $a done rc=$?"
}

pack() {
    say "INSPECT pack start ($1)"
    python scripts/analysis/eval_v4_inspection_pack.py --root "$EVAL_CAMP" \
        > "$SCRATCH/tail_inspection_$1.log" 2>&1
    say "INSPECT pack done rc=$? ($1) -> $EVAL_CAMP/inspection"
}

# ── 0. wait for the runner to finish the stage before layup ─────────────────
# Match the RUNNER's "POST " prefix, and only lines newer than our own start:
# this log is append-only, still holds the accidental 2026-09-09 run, and say()
# writes into it, so a bare phrase match hits either history or itself.
START_LINE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
say "armed at log line $START_LINE; will take over once the runner completes stage $STOP_AFTER"
while ! tail -n "+$((START_LINE + 1))" "$LOG" 2>/dev/null \
        | grep -q "POST EVALV4 generate $STOP_AFTER done"; do
    sleep 30
done
say "runner completed $STOP_AFTER — taking over the queue"

# ── 0b. stop the runner at this boundary ────────────────────────────────────
# The runner starts layup within seconds of logging "surface done", so a race
# is unavoidable. Killing a layup that is seconds old costs seconds and the
# tail regenerates it at full seeds anyway; killing one that had run for hours
# would not be acceptable, which is why this fires at the boundary and not
# later.
for p in $(pgrep -f "bash scripts/ldm06_post\.sh" 2>/dev/null); do
    kill "$p" 2>/dev/null && say "stopped the main runner (pid $p)"
done
sleep 3
for p in $(pgrep -f "eval_v4\.cli generate layup" 2>/dev/null); do
    kill "$p" 2>/dev/null && say "stopped a layup that had just started (pid $p) — it is re-run later at full seeds"
done
sleep 5
while pgrep -f "eval_v4\.cli generate|r08_rung_report\.py" >/dev/null 2>&1; do sleep 20; done
say "GPU free; tail owns the queue"

# ── 1. the cheap stages the runner would have done after layup ──────────────
gen multichunk
gen assembly
gen cfg

# ── 2. the pack, BEFORE layup, so inspection happens while layup runs ───────
pack before_layup
say "PACK READY for inspection -> $EVAL_CAMP/inspection — SEND PATH TO SUPERVISOR"

# ── 3. the five approved assessments, cheapest first ────────────────────────
# Skipped loudly when not yet implemented; re-running this script picks up
# whatever has landed since.
run_stage() {
    local name="$1" kind="$2" probe="$3"; shift 3
    case "$kind" in
        assessment)
            python -c "import sys; from poregen.eval_v4.cases import ASSESSMENTS; sys.exit(0 if '$probe' in ASSESSMENTS else 1)" 2>/dev/null \
                || { say "SKIP $name — not implemented yet"; return; } ;;
        script)
            [ -f "$probe" ] || { say "SKIP $name — not implemented yet ($probe absent)"; return; } ;;
    esac
    say "STAGE $name start"
    "$@" > "$SCRATCH/tail_${name}.log" 2>&1
    say "STAGE $name done rc=$?"
}

run_stage memorisation_full script scripts/analysis/memorisation_full_store.py \
    python scripts/analysis/memorisation_full_store.py --root "$EVAL_CAMP"
run_stage assembly_modes assessment assembly_modes \
    python -m poregen.eval_v4.cli generate assembly_modes \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents
run_stage field_stats assessment field_stats \
    python -m poregen.eval_v4.cli generate field_stats \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents
run_stage label_uncertainty script scripts/analysis/label_uncertainty.py \
    python scripts/analysis/label_uncertainty.py --out "$EVAL_CAMP/label_uncertainty"
run_stage downstream_utility script scripts/analysis/downstream_utility.py \
    python scripts/analysis/downstream_utility.py --out "$EVAL_CAMP/downstream_utility"

# ── 4. layup LAST, full 3 seeds — the paper needs the 3-seed table ──────────
gen layup

# ── 5. measure, report, and a final pack that includes everything ───────────
for a in $(python -c "from poregen.eval_v4.cases import ASSESSMENTS; print(' '.join(sorted(ASSESSMENTS)))"); do
    if [ -d "$EVAL_CAMP/$a/volumes" ]; then
        say "MEASURE $a start"
        python -m poregen.eval_v4.cli measure "$a" --root "$EVAL_CAMP" \
            > "$SCRATCH/tail_measure_${a}.log" 2>&1
        say "MEASURE $a done rc=$?"
    else
        say "MEASURE $a skipped: no volumes"
    fi
done
say "REPORT start"
python -m poregen.eval_v4.cli report --root "$EVAL_CAMP" > "$SCRATCH/tail_report.log" 2>&1
say "REPORT done rc=$?"
pack final
say "TAIL COMPLETE"

#!/usr/bin/env bash
# Everything that must follow the main post-runner's eval-v4 generation.
#
# It exists as a SEPARATE script because scripts/ldm06_post.sh is already
# running: bash reads a script incrementally by file offset, so editing a live
# one makes it resume mid-line in rewritten content. This waits for the
# runner's own log line instead.
#
# Order: multichunk (missing from the runner's ASSESSMENTS) -> the five
# approved assessments, cheapest first -> measure/report -> a regenerated
# inspection pack so the 384-cubed and sphere-r160 slices are in it.
#
# Anything not yet implemented is SKIPPED loudly and the chain continues. The
# queue is not blocked on unfinished code; the skipped ones are run later, as
# they land, by re-running this script.
#
# Usage:  bash scripts/ldm06_post_tail.sh
set -uo pipefail          # NOT -e: a failing stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
EVAL_CAMP="$REPO/runs/campaigns/12-eval-v4"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/post_ldm06.log"
RUNNER_LOG="$LOG"
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)

mkdir -p "$SCRATCH"; cd "$REPO" || exit 1
say() { printf '%s  TAIL %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  TAIL %s\n' "$(date -Is)" "$*"; }

# ── 0. wait for the main runner to finish its last generation stage ──────────
# The runner emits `EVALV4 generate cfg done rc=N` from `say "EVALV4 generate
# $a done rc=$?"`. cfg is the last entry in its ASSESSMENTS array.
# Only lines written AFTER this script arms. The log is append-only and still
# holds the accidental 2026-09-09 run, which contains its own
# "EVALV4 generate cfg done rc=1" — a plain grep matches that and fires
# immediately, which is exactly what happened on the first attempt.
START_LINE=$(wc -l < "$RUNNER_LOG" 2>/dev/null || echo 0)
say "armed at log line $START_LINE; waiting for a NEW 'EVALV4 generate cfg done'"
while ! tail -n "+$((START_LINE + 1))" "$RUNNER_LOG" 2>/dev/null \
        | grep -q "EVALV4 generate cfg done"; do
    sleep 60
done
say "cfg generation finished"

# One GPU job at a time: the runner goes on to its own inspection pack and the
# owed rung reports after cfg, so wait for the card rather than racing it.
while pgrep -f "eval_v4\.cli generate|r08_rung_report\.py|train_vae\.py run" >/dev/null 2>&1; do
    sleep 30
done
say "GPU is free"

# ── 1. multichunk, missing from the runner's ASSESSMENTS ────────────────────
say "EVALV4 generate multichunk start"
python -m poregen.eval_v4.cli generate multichunk \
    --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents \
    > "$SCRATCH/evalv4_gen_multichunk.log" 2>&1
say "EVALV4 generate multichunk done rc=$?"

# ── 2. the five approved assessments, cheapest first ────────────────────────
# name|kind|entry point to test for|command
#   kind 'assessment' = an eval_v4 assessment (needs a cases entry)
#   kind 'script'     = a standalone script
run_stage() {
    local name="$1" kind="$2" probe="$3"; shift 3
    case "$kind" in
        assessment)
            if ! python -c "import sys; from poregen.eval_v4.cases import ASSESSMENTS; sys.exit(0 if '$probe' in ASSESSMENTS else 1)" 2>/dev/null; then
                say "SKIP $name — not implemented yet (no '$probe' assessment); re-run this script when it lands"
                return
            fi ;;
        script)
            if [ ! -f "$probe" ]; then
                say "SKIP $name — not implemented yet (no $probe); re-run this script when it lands"
                return
            fi ;;
    esac
    say "STAGE $name start"
    "$@" > "$SCRATCH/tail_${name}.log" 2>&1
    say "STAGE $name done rc=$?"
}

run_stage label_uncertainty script scripts/analysis/label_uncertainty.py \
    python scripts/analysis/label_uncertainty.py --out "$EVAL_CAMP/label_uncertainty"

run_stage field_stats assessment field_stats \
    python -m poregen.eval_v4.cli generate field_stats \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents

run_stage assembly_modes assessment assembly_modes \
    python -m poregen.eval_v4.cli generate assembly_modes \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents

run_stage memorisation_full script scripts/analysis/memorisation_full_store.py \
    python scripts/analysis/memorisation_full_store.py --root "$EVAL_CAMP"

run_stage downstream_utility script scripts/analysis/downstream_utility.py \
    python scripts/analysis/downstream_utility.py --out "$EVAL_CAMP/downstream_utility"

# ── 3. measure and report every assessment that has volumes ─────────────────
for a in $(python -c "from poregen.eval_v4.cases import ASSESSMENTS; print(' '.join(sorted(ASSESSMENTS)))"); do
    if [ -d "$EVAL_CAMP/$a/volumes" ]; then
        say "EVALV4 measure $a start (CPU)"
        python -m poregen.eval_v4.cli measure "$a" --root "$EVAL_CAMP" \
            > "$SCRATCH/tail_measure_${a}.log" 2>&1
        say "EVALV4 measure $a done rc=$?"
    else
        say "EVALV4 measure $a skipped: no volumes"
    fi
done
say "EVALV4 report start (CPU)"
python -m poregen.eval_v4.cli report --root "$EVAL_CAMP" > "$SCRATCH/tail_report.log" 2>&1
say "EVALV4 report done rc=$?"

# ── 4. regenerate the inspection pack, LAST ─────────────────────────────────
# The main runner builds a pack before multichunk exists, so its version cannot
# contain the 384-cubed box, the r=160 sphere or the rough surface. This
# overwrites it with the complete one.
say "INSPECT pack regenerate start (CPU)"
python scripts/analysis/eval_v4_inspection_pack.py --root "$EVAL_CAMP" \
    > "$SCRATCH/tail_inspection.log" 2>&1
say "INSPECT pack regenerate done rc=$? -> $EVAL_CAMP/inspection — SEND PATH TO SUPERVISOR"

say "TAIL COMPLETE"

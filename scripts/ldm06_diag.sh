#!/usr/bin/env bash
# Run diag_ldm_samples every DIAG_EVERY steps while ldm06/base trains.
#
# The run's own metrics.jsonl logs a single val `loss`. Every number the D43
# gate is written in — std-ratio, por_cond_mae, x0_sat, degenerate fraction,
# the per-neighbour-bucket table — comes from diag_ldm_samples, which SAMPLES.
# Nothing produces them unless this is run, so a run can look healthy on its
# loss curve and be failing every gate.
#
# It shares the GPU with training on purpose: a diagnostic that waits for the
# card to be free arrives after the decision it was meant to inform.
#
# Usage:  bash scripts/ldm06_diag.sh [run_dir]
set -uo pipefail

REPO=/home/jorgecabrejas/Dev/GenAI
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
DIAG_EVERY=${DIAG_EVERY:-20000}
#: Steps to wait past a target before sampling. DIAG_EVERY is a multiple of the
#: run's gen_eval_every (2000), so firing on the target guarantees a collision
#: with the run's full validation and 64-sample generation eval.
SETTLE_STEPS=${SETTLE_STEPS:-800}
RUN_DIR="${1:-$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)}"
LOG="$REPO/runs/campaigns/09-r08-latent-sweep/ldm06_diag.log"

cd "$REPO" || exit 1
mkdir -p "$SCRATCH" "$(dirname "$LOG")"
say() { printf '%s  DIAG %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

[ -n "$RUN_DIR" ] || { say "ABORT no ldm06 run directory"; exit 1; }
say "watching $RUN_DIR every $DIAG_EVERY steps"

current_step() {
    python - "$RUN_DIR" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "log.jsonl"
last = 0
try:
    with p.open() as f:
        for line in f:
            if '"step"' in line:
                try:
                    last = max(last, int(json.loads(line).get("step", 0)))
                except Exception:
                    pass
except FileNotFoundError:
    pass
print(last)
PY
}

next_target=$DIAG_EVERY
while true; do
    step=$(current_step)
    # SETTLE_STEPS past the target, not on it. The 60k run died with CUDA OOM
    # loading its own 83M-parameter UNet, and the cause is arithmetic rather
    # than luck: gen_eval_every is 2000 and DIAG_EVERY is 20000, so every
    # single diagnostic fires exactly when the run is doing its heaviest thing
    # — a full validation plus a 64-sample generation eval. Waiting a few
    # hundred steps puts the diagnostic in the quiet part of the cycle.
    if [ "${step:-0}" -ge "$((next_target + SETTLE_STEPS))" ]; then
        tag=$(printf '%dk' $((next_target / 1000)))
        rc=1
        for attempt in $(seq 1 30); do
            # Cheap probe first. On this unified-memory box a CUDA context
            # cannot always be created while the training run holds ~44 GB RSS
            # and the page cache holds most of the rest — the 60k diagnostic
            # died at its FIRST cuda call, before loading anything. The probe
            # costs a second; discovering it by loading a checkpoint costs
            # minutes and produces a traceback that looks like a code fault.
            if ! python -c "import torch,sys; torch.cuda.mem_get_info(); sys.exit(0)" >/dev/null 2>&1; then
                say "diag $tag attempt $attempt: no CUDA context (memory pressure), waiting"
                sleep 120
                continue
            fi
            say "step $step >= $next_target+$SETTLE_STEPS -> diag_ldm_samples ($tag, raw + EMA, DDIM-200, attempt $attempt)"
            python scripts/diag_ldm_samples.py --run-dir "$RUN_DIR" --ddim200 \
                > "$SCRATCH/ldm06_diag_${tag}.log" 2>&1
            rc=$?
            [ "$rc" -eq 0 ] && break
            say "FAIL diag $tag attempt $attempt rc=$rc — see $SCRATCH/ldm06_diag_${tag}.log"
            sleep 300
        done
        if [ "$rc" -eq 0 ]; then
            say "diag $tag done rc=0 -> $RUN_DIR/convergence_check.jsonl"
            say "REPORT $tag ready — send to supervisor"
        else
            # NOT "ready". A failed diagnostic that announces itself as ready is
            # worse than one that is simply missing: it says a gate was read
            # when nothing was measured.
            say "FAILED diag $tag after 30 attempts rc=$rc — NOTHING MEASURED, REPORT THE FAILURE"
        fi
        next_target=$((next_target + DIAG_EVERY))
    fi
    # Training exited and we are past the last target: nothing more will come.
    if ! pgrep -f "scripts/train_ldm\.py (run|resume) " >/dev/null 2>&1; then
        say "training has exited at step ${step:-?}; final diag"
        python scripts/diag_ldm_samples.py --run-dir "$RUN_DIR" --ddim200 \
            > "$SCRATCH/ldm06_diag_final.log" 2>&1
        say "final diag done rc=$?"
        break
    fi
    sleep 300
done
say "watcher done"

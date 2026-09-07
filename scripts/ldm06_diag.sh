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
    if [ "${step:-0}" -ge "$next_target" ]; then
        tag=$(printf '%dk' $((next_target / 1000)))
        say "step $step >= $next_target -> diag_ldm_samples ($tag, raw + EMA, DDIM-200)"
        python scripts/diag_ldm_samples.py --run-dir "$RUN_DIR" --ddim200 \
            > "$SCRATCH/ldm06_diag_${tag}.log" 2>&1
        say "diag $tag done rc=$? -> $RUN_DIR/convergence_check.jsonl"
        say "REPORT $tag ready — send to supervisor"
        next_target=$((next_target + DIAG_EVERY))
    fi
    # Training exited and we are past the last target: nothing more will come.
    if ! pgrep -f "scripts/train_ldm\.py run " >/dev/null 2>&1; then
        say "training has exited at step ${step:-?}; final diag"
        python scripts/diag_ldm_samples.py --run-dir "$RUN_DIR" --ddim200 \
            > "$SCRATCH/ldm06_diag_final.log" 2>&1
        say "final diag done rc=$?"
        break
    fi
    sleep 300
done
say "watcher done"

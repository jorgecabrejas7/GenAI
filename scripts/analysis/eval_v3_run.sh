#!/bin/bash
# eval v3 runner — regenerate the porosity/air volumes with the fixed decode
# path and redo the analyses on them.  Appends everything to
# runs/campaigns/05-eval-v3-fixed-decode/run.log and ends with EVAL_V3_DONE (or EVAL_V3_ABORTED).
#
# Order: the cheap sets land first (dose response ~45 min, DDIM probe ~25 min)
# together with a full analysis pass, then the expensive layup set (~4 h,
# 1024x1024x192 volumes) runs and every analysis is refreshed over it.
#
# Launch:
#   nohup bash scripts/analysis/eval_v3_run.sh > /dev/null 2>&1 &
#   tail -f runs/campaigns/05-eval-v3-fixed-decode/run.log
#
# Log line format:
#   [YYYY-MM-DD HH:MM:SS] STEP nn/NN <name> START
#   [YYYY-MM-DD HH:MM:SS] STEP nn/NN <name> DONE rc=0 elapsed=123s
#   [YYYY-MM-DD HH:MM:SS] STEP nn/NN <name> FAILED rc=1 elapsed=12s
set -uo pipefail

REPO=/home/jorgecabrejas/Dev/GenAI
cd "$REPO" || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POREGEN_EVAL_ROOT="$REPO/runs/campaigns/05-eval-v3-fixed-decode"
export TQDM_DISABLE=1
export MPLBACKEND=Agg

PY=python
LOG="$POREGEN_EVAL_ROOT/run.log"
mkdir -p "$POREGEN_EVAL_ROOT"

N_STEPS=12
STEP_I=0
FAILED=""

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# run <name> <critical:yes|no> <command...>
run() {
  local name="$1"; shift
  local critical="$1"; shift
  STEP_I=$((STEP_I + 1))
  local idx
  idx=$(printf '%02d/%02d' "$STEP_I" "$N_STEPS")
  local t0
  t0=$(date +%s)
  echo "[$(ts)] STEP $idx $name START" >> "$LOG"
  echo "[$(ts)] STEP $idx $name CMD: $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  local el=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 0 ]; then
    echo "[$(ts)] STEP $idx $name DONE rc=0 elapsed=${el}s" >> "$LOG"
    return 0
  fi
  echo "[$(ts)] STEP $idx $name FAILED rc=$rc elapsed=${el}s" >> "$LOG"
  FAILED="$FAILED $name(rc=$rc)"
  if [ "$critical" = "yes" ]; then
    echo "[$(ts)] critical step failed — aborting" >> "$LOG"
    echo "EVAL_V3_ABORTED at $name rc=$rc" >> "$LOG"
    exit "$rc"
  fi
  return "$rc"
}

echo "=== eval v3 start $(date -Is) ===" >> "$LOG"
echo "POREGEN_EVAL_ROOT=$POREGEN_EVAL_ROOT" >> "$LOG"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" >> "$LOG"
echo "git HEAD: $(git rev-parse --short HEAD 2>/dev/null)" >> "$LOG"

# ---- phase 1: cheap generation --------------------------------------------
run gen_dose_response yes \
  "$PY" scripts/analysis/eval_v2_dose_response.py
run gen_ddim_probe yes \
  "$PY" scripts/analysis/eval_v3_ddim_probe.py

# ---- phase 2: first analysis pass (192^3 sets only) ------------------------
run air_audit_pass1 no \
  "$PY" scripts/analysis/eval_v3_air_audit.py
run onlypores_pass1 no \
  "$PY" scripts/analysis/eval_v3_onlypores.py
run ddim_analysis_pass1 no \
  "$PY" scripts/analysis/eval_v3_ddim_analysis.py
run compare_pass1 no \
  "$PY" scripts/analysis/eval_v3_compare.py

# ---- phase 3: expensive layup generation (~4 h) ---------------------------
run gen_layup yes \
  "$PY" scripts/analysis/eval_v2_layup.py --seeds 101

# ---- phase 4: final analysis pass over everything -------------------------
run air_audit_final no \
  "$PY" scripts/analysis/eval_v3_air_audit.py
run onlypores_final no \
  "$PY" scripts/analysis/eval_v3_onlypores.py --skip-real-validation
run ddim_analysis_final no \
  "$PY" scripts/analysis/eval_v3_ddim_analysis.py
run compare_final no \
  "$PY" scripts/analysis/eval_v3_compare.py

STEP_I=$((STEP_I + 1))
echo "[$(ts)] STEP $(printf '%02d/%02d' "$STEP_I" "$N_STEPS") summary DONE rc=0 elapsed=0s" >> "$LOG"

if [ -z "$FAILED" ]; then
  echo "[$(ts)] all steps succeeded" >> "$LOG"
else
  echo "[$(ts)] non-critical failures:$FAILED" >> "$LOG"
fi
echo "EVAL_V3_DONE" >> "$LOG"

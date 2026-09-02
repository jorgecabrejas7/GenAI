#!/bin/bash
# Phase 1 runner: dose-response v2 then CFG sweep v2. Appends to phase1.log.
set -uo pipefail
cd /home/jorgecabrejas/Dev/GenAI
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/eval_v2/phase1.log
echo "=== eval_v2 phase 1 start $(date -Is) ===" >> "$LOG"
python scripts/analysis/eval_v2_dose_response.py >> "$LOG" 2>&1
RC_A=$?
echo "=== task A (dose_response) exit $RC_A $(date -Is) ===" >> "$LOG"
python scripts/analysis/eval_v2_cfg_sweep.py >> "$LOG" 2>&1
RC_B=$?
echo "=== task B (cfg_sweep) exit $RC_B $(date -Is) ===" >> "$LOG"
if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]; then
  echo "PHASE1_DONE" >> "$LOG"
else
  echo "PHASE1_FAILED A=$RC_A B=$RC_B" >> "$LOG"
fi

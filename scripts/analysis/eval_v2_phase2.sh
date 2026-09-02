#!/bin/bash
# Phase 2 runner: layup round-trip v2 (three arms). Appends to phase2.log.
set -uo pipefail
cd /home/jorgecabrejas/Dev/GenAI
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/eval_v2/phase2.log
echo "=== eval_v2 phase 2 start $(date -Is) ===" >> "$LOG"
python scripts/analysis/eval_v2_layup.py >> "$LOG" 2>&1
RC=$?
echo "=== layup round-trip v2 exit $RC $(date -Is) ===" >> "$LOG"
if [ "$RC" -eq 0 ]; then
  echo "PHASE2_DONE" >> "$LOG"
else
  echo "PHASE2_FAILED rc=$RC" >> "$LOG"
fi

#!/usr/bin/env bash
# Second r08 runner: the two rungs that do not feed the compressor decision,
# plus every full-split rung report OWED by the first runner.
#
# Why it is separate: rf-2 and rf-64 bracket the sweep for the paper, but the
# decision only needs base / rf-8 / rf-32 / rf-4. Keeping them here means the
# decision is not queued behind ~26 h of training. The full-split reports are
# here for the same reason — the decision reads the engine's own final
# val_full/test_full, and the standalone reports are for the final table.
#
# Start it AFTER ldm06/base is training: it waits for the GPU to go quiet, so
# starting it early would make it jump into ldm06's gap, not wait for it.
#
# Usage:  bash scripts/r08_queue_tail.sh
set -uo pipefail          # deliberately NOT -e: a failing rung must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/queue.log"
RUNGS=(reduction-factor-2 reduction-factor-64)
# Every rung that should end up in the final paper table, in sweep order.
ALL_RUNGS=(reduction-factor-2 base reduction-factor-8 reduction-factor-32 reduction-factor-4 reduction-factor-64)

mkdir -p "$CAMP" "$SCRATCH"
cd "$REPO" || exit 1

say() { printf '%s  %s\n' "$(date -Is)" "TAIL $*" >> "$LOG"; printf '%s  %s\n' "$(date -Is)" "TAIL $*"; }

newest_run() { ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null | head -1; }

# The run directory for a rung: newest run whose run_metadata.json names that
# variant AND that actually produced a best.ckpt. Matching on run index would
# be wrong — base has three runs, two of which are the crashed and the
# untempered attempts, and the index says nothing about which is which.
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

post_reports() {
    local run_dir="$1" tag="$2"
    [ -z "$run_dir" ] && { say "REPORTS $tag skipped: no run dir"; return; }
    [ -f "${run_dir}best.ckpt" ] || { say "REPORTS $tag skipped: no best.ckpt in $run_dir"; return; }

    say "REPORTS $tag probe start (CPU) $run_dir"
    ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 \
      python scripts/analysis/r08_calibration_probe.py \
        --run "$run_dir" --device cpu --per-bin 96 \
        --out-name "calibration_probe_r08_${tag}" \
        > "$SCRATCH/probe_${tag}.log" 2>&1
      say "REPORTS $tag probe done rc=$?" ) &

    say "REPORTS $tag tile-seam start (GPU, gap)"
    python scripts/analysis/vae_tile_seam.py \
        --checkpoint "${run_dir}best.ckpt" \
        --out "$CAMP/r08_${tag}/tile_seam" --no-save-volumes \
        > "$SCRATCH/tileseam_${tag}.log" 2>&1
    say "REPORTS $tag tile-seam done rc=$?"

    say "REPORTS $tag sanity start (GPU, gap)"
    python scripts/diag_mask_sanity.py --checkpoint "${run_dir}best.ckpt" \
        > "$SCRATCH/sanity_${tag}.log" 2>&1
    say "REPORTS $tag sanity done rc=$?"
}

# Wait for whatever holds the GPU (ldm06/base, or a rung that overran).
if pgrep -f "scripts/train_(vae|ldm)\.py run " >/dev/null 2>&1; then
    say "WAIT for the job on the GPU: $(pgrep -af 'scripts/train_.*\.py run ' | head -1)"
    while pgrep -f "scripts/train_(vae|ldm)\.py run " >/dev/null 2>&1; do sleep 60; done
    say "WAIT done"
fi

say "QUEUE start; rungs: ${RUNGS[*]}"
for exp in "${RUNGS[@]}"; do
    say "START r08/$exp"
    python scripts/train_vae.py run "r08/$exp" > "$SCRATCH/r08_${exp}.log" 2>&1
    rc=$?
    say "DONE r08/$exp rc=$rc"
    [ "$rc" -ne 0 ] && say "FAIL r08/$exp rc=$rc — chain continues; see $SCRATCH/r08_${exp}.log"
    post_reports "$(newest_run)" "$exp"
done

# The debt the first runner logged as OWED. ~70 min of GPU each, which is why
# they are here and not on the decision path. A rung whose run cannot be found
# is skipped loudly rather than silently dropped from the table.
say "OWED start: full-split rung reports for ${ALL_RUNGS[*]}"
for exp in "${ALL_RUNGS[@]}"; do
    d=$(run_dir_for "$exp")
    if [ -z "$d" ]; then say "OWED $exp skipped: no run directory found"; continue; fi
    say "OWED $exp report start $d"
    python scripts/analysis/r08_rung_report.py --run "$d" > "$SCRATCH/report_${exp}.log" 2>&1
    say "OWED $exp report done rc=$?"
done

say "START rung report --compare (final table)"
python scripts/analysis/r08_rung_report.py --compare > "$SCRATCH/r08_compare_final.log" 2>&1
say "DONE rung report --compare rc=$?"
say "QUEUE complete: sweep closed, all six rungs reported."

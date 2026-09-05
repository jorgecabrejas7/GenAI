#!/usr/bin/env bash
# Sequential runner for the r08 compression sweep.
#
# Why this exists: rf-8 finished at 09:51 and the next rung was not launched
# until 13:22 because a completion notification was missed. 3 h 31 m of GPU
# went nowhere. A human (or an agent) noticing a process exit is not a
# scheduling mechanism; this is.
#
# Rules:
#   - the next rung starts the moment the previous process exits, pass or fail
#   - a FAILED rung is logged and the chain continues; only a rung that cannot
#     be launched at all stops it
#   - CPU-able end-of-run reports run in parallel with the next rung
#   - every transition is appended to queue.log with an rc
#
# Usage:  bash scripts/r08_queue.sh
set -uo pipefail          # deliberately NOT -e: a failing rung must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/queue.log"
# rf-2 and rf-64 are NOT here: they move to a second runner that starts after
# ldm06/base, so the compressor decision is not waiting behind two more rungs.
RUNGS=(reduction-factor-4)
# The rungs whose full-split report feeds the decision table.
DECIDING=(base reduction-factor-8 reduction-factor-32 reduction-factor-4)

mkdir -p "$CAMP" "$SCRATCH"
cd "$REPO" || exit 1

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  %s\n' "$(date -Is)" "$*"; }

newest_run() { ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null | head -1; }

# Reports for a finished rung, run in the GAP before the next rung starts.
#
# tile-seam and sanity need the GPU, so they run SEQUENTIALLY here with nothing
# else on the card — ~20 min per rung, which is cheaper than a missing gate
# column, and it is the only way to touch the GPU without risking the OOM that
# killed a job earlier today. The calibration probe has a CPU path so it is
# backgrounded and overlaps the next rung. The full-split rung report stays
# OWED until after the chain: the engine's own final val_full/test_full already
# give the per-bin tables, so it would buy little for an hour of GPU.
post_reports() {
    local run_dir="$1" tag="$2"
    [ -z "$run_dir" ] && { say "REPORTS $tag skipped: no run dir"; return; }
    [ -f "${run_dir}best.ckpt" ] || { say "REPORTS $tag skipped: no best.ckpt in $run_dir"; return; }

    # CPU, overlaps the next rung.
    say "REPORTS $tag probe start (CPU) $run_dir"
    ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 \
      python scripts/analysis/r08_calibration_probe.py \
        --run "$run_dir" --device cpu --per-bin 96 \
        --out-name "calibration_probe_r08_${tag}" \
        > "$SCRATCH/probe_${tag}.log" 2>&1
      say "REPORTS $tag probe done rc=$?" ) &

    # GPU, in the gap, one at a time, before the next rung is launched.
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

    say "OWED $tag: full-split rung report (GPU, after the chain)"
}

say "QUEUE start; rungs: ${RUNGS[*]}"

# A rung may already be running (rf-32 was launched by hand). Wait it out
# rather than launching a second job onto the same GPU.
if pgrep -f "train_vae.py run r08/" >/dev/null 2>&1; then
    running=$(pgrep -af "train_vae.py run r08/" | head -1)
    say "WAIT for the rung already on the GPU: $running"
    while pgrep -f "train_vae.py run r08/" >/dev/null 2>&1; do sleep 30; done
    say "WAIT done"
    post_reports "$(newest_run)" "reduction-factor-32"
    # rf-8 finished before this runner existed, so its GPU reports were never
    # run. Clear that debt in the same gap rather than leaving a hole in the
    # decision table.
    rf8dir=$(ls -d "$REPO"/runs/vae/r08-run-0004-*/ 2>/dev/null | head -1)
    if [ -n "$rf8dir" ] && [ ! -f "$CAMP/r08_reduction-factor-8/tile_seam/results.json" ]; then
        say "REPORTS reduction-factor-8 tile-seam start (backfill, GPU, gap)"
        python scripts/analysis/vae_tile_seam.py --checkpoint "${rf8dir}best.ckpt" \
            --out "$CAMP/r08_reduction-factor-8/tile_seam" --no-save-volumes \
            > "$SCRATCH/tileseam_reduction-factor-8.log" 2>&1
        say "REPORTS reduction-factor-8 tile-seam done rc=$?"
        say "REPORTS reduction-factor-8 sanity start (backfill, GPU, gap)"
        python scripts/diag_mask_sanity.py --checkpoint "${rf8dir}best.ckpt" \
            > "$SCRATCH/sanity_reduction-factor-8.log" 2>&1
        say "REPORTS reduction-factor-8 sanity done rc=$?"
    fi
fi

for exp in "${RUNGS[@]}"; do
    tag="$exp"
    say "START r08/$exp"
    python scripts/train_vae.py run "r08/$exp" > "$SCRATCH/r08_${tag}.log" 2>&1
    rc=$?
    say "DONE r08/$exp rc=$rc"
    [ "$rc" -ne 0 ] && say "FAIL r08/$exp rc=$rc — chain continues; see $SCRATCH/r08_${tag}.log"
    post_reports "$(newest_run)" "$tag"
done

# Full-split rung reports for the deciding rungs. GPU, sequential, nothing else
# on the card. Each is val 2080 + test 2270 batches at ~1 batch/s, so ~70 min
# per rung — ~4.7 h for four. They are the decision input, so the cost is
# deliberate, not incidental.
for exp in "${DECIDING[@]}"; do
    dir=$(ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null | while read -r d; do
              grep -q "variant: $exp\b" "$d/resolved_config.yaml" 2>/dev/null && echo "$d" && break
          done)
    if [ -z "$dir" ]; then say "REPORT $exp skipped: no run dir found"; continue; fi
    say "REPORT $exp start (full split, GPU) $dir"
    python scripts/analysis/r08_rung_report.py --run "$dir" \
        > "$SCRATCH/rungreport_${exp}.log" 2>&1
    say "REPORT $exp done rc=$?"
done

say "START rung report --compare"
python scripts/analysis/r08_rung_report.py --compare > "$SCRATCH/r08_compare.log" 2>&1
say "DONE rung report --compare rc=$?"
say "QUEUE STOPPED for the compressor decision. rf-2 and rf-64 are in the second runner (scripts/r08_queue_tail.sh), which starts after ldm06/base."

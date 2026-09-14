#!/usr/bin/env bash
# Insert the corrected generated-arm re-run between the campaign-14 arms and
# the r08 rungs, then finish the queue.
#
# `queue_tail.sh` is already running and holds its own inode, so its step B
# cannot be reordered by editing it — the same reason it had to take over from
# `final_queue.sh`. This waits for its step A, stops it, and runs the rest:
#
#   1. redecode DDIM-50 and DDIM-200 on the CORRECTED latent scaling
#   2. r08 rf-2 and rf-64
#   3. the notebooks into campaigns 18 and 12
#
# The redecode goes first because every generated-arm number of the D43 gate is
# currently withdrawn, and the rungs are the outer question.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
TLOG="$REPO/runs/campaigns/queue_tail.log"
LOG="$REPO/runs/campaigns/queue_tail2.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C18="$REPO/runs/campaigns/18-eval-v4-final"
FT_CAMP="$REPO/runs/campaigns/11-decoder-ft"
BASE=$(ls -d "$REPO"/runs/vae/r08-run-0004-*/best.ckpt | head -1)
FT=$(ls -d "$REPO"/runs/vae/r08-run-0007-*/best.ckpt | head -1)
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)
cd "$REPO" || exit 1
say() { printf '%s  TAIL2 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

say "armed — waiting for the campaign-14 arms (queue_tail.sh step A)"
while ! grep -q "TAIL A done" "$TLOG" 2>/dev/null; do sleep 60; done
say "step A finished"

for pid in $(pgrep -f "bash .*queue_tail\.sh" || true); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *queue_tail2*) continue ;;
        *queue_tail.sh*) say "stopping queue_tail.sh pid $pid"; kill -TERM "$pid" 2>/dev/null ;;
    esac
done
sleep 5
for pid in $(pgrep -f "train_vae\.py run r08/reduction-factor" || true); do
    say "stopping a just-started rung, pid $pid"; kill -TERM "$pid" 2>/dev/null
done
sleep 5
say "old tail stopped"

# ── 1. the corrected generated arm ──────────────────────────────────────────
for steps in 50 200; do
    out="$FT_CAMP/redecode_ddim${steps}"
    say "REDECODE DDIM-$steps start (corrected latent scaling)"
    python scripts/analysis/decoder_ft_redecode.py \
        --baseline "$BASE" --finetuned "$FT" \
        --latents "$C18/*/volumes/*/latents.npy" --ddim-steps "$steps" \
        --out "$out" > "$S/fix_redecode_${steps}.log" 2>&1
    say "REDECODE DDIM-$steps done rc=$? -> $out"
done

# ── 2. the rungs ────────────────────────────────────────────────────────────
for exp in "${TAIL_RUNGS[@]}"; do
    say "r08/$exp start"
    python scripts/train_vae.py run "r08/$exp" > "$S/t2_r08_${exp}.log" 2>&1
    say "r08/$exp done rc=$?"
done

# ── 3. the notebooks ────────────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    say "notebook into $(basename "$root") start"
    python scripts/analysis/build_eval_v4_notebook.py --root "$root" \
        > "$S/t2_nb_$(basename "$root").log" 2>&1
    say "notebook into $(basename "$root") done rc=$?"
done
say "QUEUE TAIL2 COMPLETE"

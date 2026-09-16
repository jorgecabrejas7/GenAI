#!/usr/bin/env bash
# THE ONE CHAIN. Everything from the SliceGAN training exit to the notebook.
#
#   1. wait for SliceGAN training to finish       (run progress, not process absence)
#   2. generate its eval_v4 cases
#   3. eval_v4 measure + report + inspection PNGs, request-free
#   4. downstream arms slicegan_synthetic + real_plus_slicegan_aug
#   5. 3D pixel-space DDPM training       (campaign 23)
#   6. DDPM generation + measure/report + its downstream arms
#   7. resume rf-2 from its step-22000 checkpoint
#   8. rf-64
#   9. notebook rebuild into campaigns 18 and 12
#
# RULES THIS CHAIN KEEPS, each one learned the hard way:
#   - exit tests are RUN PROGRESS, never process absence: a run paused for two
#     minutes is not a finished run.
#   - pids are resolved and VERIFIED against /proc before any signal, and this
#     script excludes itself. A pattern kill matches the watcher and its own
#     shell; that has cost two aborted runs and five exit-144s.
#   - nothing edits a running script in place.
#   - NO store-heavy work runs beside a training stage — that deadlocked rf-2
#     at step 22999 and cost 999 steps. Stage 4 reads the store and therefore
#     runs alone, after stage 3 and before stage 5.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"
C22="$REPO/runs/campaigns/22-slicegan-baseline"
C23="$REPO/runs/campaigns/23-ddpm3d-baseline"
SG_TRAIN="$C22/train"
RF2_RUN=$(ls -dt "$REPO"/runs/vae/r08-run-0010-*/ 2>/dev/null | head -1)
cd "$REPO" || exit 1
say() { printf '%s  CHAIN %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

alive() {  # $1 = pattern; true if a process OTHER than this script matches
    local pid cmd
    for pid in $(pgrep -f "$1" 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        case "$cmd" in *chain.sh*) continue ;; esac
        case "$cmd" in *"$2"*) return 0 ;; esac
    done
    return 1
}

# ── 1. wait for SliceGAN ────────────────────────────────────────────────────
say "armed; waiting for SliceGAN training"
while true; do
    if alive "train_slicegan\.py" "train_slicegan.py"; then sleep 120; continue; fi
    # Gone. Finished, or stopped? The log says which.
    if grep -q "done ->" "$S/slicegan.log" 2>/dev/null; then break; fi
    say "train_slicegan is not running and its log has no completion line — stopping"
    say "(resume it by hand; the chain will not run stages on a half-trained model)"
    exit 1
done
say "SliceGAN training finished"

# ── 2. generation ───────────────────────────────────────────────────────────
say "STAGE 2 generation start"
python scripts/analysis/slicegan_sample.py \
    --checkpoint "$SG_TRAIN/latest.ckpt" --root "$C22" \
    > "$S/chain_sg_sample.log" 2>&1
say "STAGE 2 done rc=$?"

# ── 3. measure + report, request-free ───────────────────────────────────────
say "STAGE 3 measure start"
python -m poregen.eval_v4.cli measure slicegan --root "$C22" \
    > "$S/chain_sg_measure.log" 2>&1
say "STAGE 3 measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C22" --assessment slicegan \
    > "$S/chain_sg_report.log" 2>&1
say "STAGE 3 done rc=$?"
say "STAGE 3 inspection start"
python scripts/analysis/slicegan_inspection.py --root "$C22" \
    > "$S/chain_sg_inspect.log" 2>&1
say "STAGE 3 inspection rc=$? -> $C22/inspection"

# ── 4. downstream arms (READS THE STORE — runs alone) ───────────────────────
say "STAGE 4 downstream arms start (store-heavy; nothing else runs)"
python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --slicegan-root "$C22" --out "$C14" \
    --arms slicegan_synthetic real_plus_slicegan_aug \
    > "$S/chain_downstream.log" 2>&1
say "STAGE 4 done rc=$? -> $C14"

# ── 5. the 3-D pixel-space DDPM (STREAMS THE STORE — runs alone) ───────────
say "STAGE 5 DDPM training start (campaign 23, cap 24 h)"
python scripts/train_ddpm3d.py --out "$C23/train" --max-hours 24 \
    > "$S/chain_ddpm_train.log" 2>&1
say "STAGE 5 done rc=$?"

# ── 6. DDPM generation, measure, downstream ────────────────────────────────
say "STAGE 6 DDPM generation start"
python scripts/analysis/ddpm3d_sample.py \
    --checkpoint "$C23/train/latest.ckpt" --root "$C23" \
    > "$S/chain_ddpm_sample.log" 2>&1
say "STAGE 6 generation rc=$?"
python -m poregen.eval_v4.cli measure ddpm3d --root "$C23" \
    > "$S/chain_ddpm_measure.log" 2>&1
say "STAGE 6 measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C23" --assessment ddpm3d \
    > "$S/chain_ddpm_report.log" 2>&1
say "STAGE 6 report rc=$?"
say "STAGE 6 downstream arms start (store-heavy; nothing else runs)"
python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --ddpm3d-root "$C23" --out "$C14" \
    --arms ddpm3d_synthetic real_plus_ddpm3d_aug \
    > "$S/chain_ddpm_downstream.log" 2>&1
say "STAGE 6 done rc=$?"

# ── 7. rf-2, resumed from step 22000 ────────────────────────────────────────
if [ -n "$RF2_RUN" ]; then
    say "STAGE 7 rf-2 resume from $(basename "$RF2_RUN")"
    python scripts/train_vae.py resume "$(basename "$RF2_RUN")" latest.ckpt \
        > "$S/chain_rf2.log" 2>&1
    say "STAGE 7 done rc=$?"
else
    say "STAGE 7 SKIPPED: no r08-run-0010 directory found"
fi

# ── 8. rf-64 ────────────────────────────────────────────────────────────────
say "STAGE 8 rf-64 start"
python scripts/train_vae.py run r08/reduction-factor-64 > "$S/chain_rf64.log" 2>&1
say "STAGE 8 done rc=$?"

# ── 9. notebooks ────────────────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    say "STAGE 9 notebook into $(basename "$root")"
    python scripts/analysis/build_eval_v4_notebook.py --root "$root" \
        > "$S/chain_nb_$(basename "$root").log" 2>&1
    say "STAGE 9 $(basename "$root") rc=$?"
done
say "CHAIN COMPLETE"

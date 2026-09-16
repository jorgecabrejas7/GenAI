#!/usr/bin/env bash
# Everything after the DDPM training, in the order the supervisor decided.
#
# WHY A SECOND SCRIPT AND NOT AN EDIT. chain.sh was already running, and bash
# reads a script by file offset: an edit gets a new inode while the running
# shell keeps reading the old one, so the running instance would have done the
# old thing anyway. This takes the queue instead. chain.sh's own bash is
# stopped when this starts; its child `train_ddpm3d.py` is NOT — killing the
# parent shell leaves the training reparented to init and still writing to its
# own log, which is the point. This script waits for THAT pid.
#
# THE ORDER, AND WHY IT IS THIS ORDER:
#   1. wait for the DDPM training to exit           (run progress, not absence)
#   2. SliceGAN FID — MINUTES, and the only moment the card is idle. It is the
#      one number campaign 22 still owes, and it must not run beside a training
#      job: see DEVELOPMENT.md, "no GPU job beside a dataloader either".
#   3. DDPM generation + measure + report
#   4. ONE downstream invocation carrying ALL FOUR baseline arms, alone on the
#      machine. One pass builds the real pool and the test loader once instead
#      of twice, and store-heavy work runs with nothing beside it — the rule
#      that was broken twice already.
#   5. rf-2 resume, rf-64, notebooks
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"
C22="$REPO/runs/campaigns/22-slicegan-baseline"
C23="$REPO/runs/campaigns/23-ddpm3d-baseline"
RF2_RUN=$(ls -dt "$REPO"/runs/vae/r08-run-0010-*/ 2>/dev/null | head -1)
: "${DDPM_PID:?set DDPM_PID to the running train_ddpm3d.py process}"
cd "$REPO" || exit 1
say() { printf '%s  TAIL %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

# ── 1. wait for the DDPM training ───────────────────────────────────────────
# By PID, verified against /proc/<pid>/cmdline — a bare pid can be reused, and
# a pattern match here would catch this script's own command line. Then by RUN
# PROGRESS: a vanished process that never logged a completion line is a crash,
# and the stages below must not run on a half-trained model.
say "armed; waiting for train_ddpm3d pid $DDPM_PID"
while [ -r "/proc/$DDPM_PID/cmdline" ] \
      && tr '\0' ' ' < "/proc/$DDPM_PID/cmdline" | grep -q "train_ddpm3d\.py"; do
    sleep 120
done
if ! grep -q "done ->" "$S/chain_ddpm_train.log" 2>/dev/null; then
    say "train_ddpm3d exited with no completion line — STOPPING"
    say "(the volumes below would come from a half-trained model; resume by hand)"
    exit 1
fi
say "DDPM training finished; the card is idle"

# ── 2. the SliceGAN FID, while nothing holds the card ───────────────────────
say "STAGE A SliceGAN FID start (the only owed number in campaign 22)"
python -m poregen.eval_v4.cli measure slicegan --root "$C22" \
    > "$S/tail_sg_measure.log" 2>&1
say "STAGE A measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C22" --assessment slicegan \
    > "$S/tail_sg_report.log" 2>&1
say "STAGE A done rc=$? -> $C22/slicegan/findings.md"

# ── 3. DDPM generation, measure, report ────────────────────────────────────
say "STAGE B DDPM generation start"
python scripts/analysis/ddpm3d_sample.py \
    --checkpoint "$C23/train/latest.ckpt" --root "$C23" \
    > "$S/tail_ddpm_sample.log" 2>&1
say "STAGE B generation rc=$?"
python -m poregen.eval_v4.cli measure ddpm3d --root "$C23" \
    > "$S/tail_ddpm_measure.log" 2>&1
say "STAGE B measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C23" --assessment ddpm3d \
    > "$S/tail_ddpm_report.log" 2>&1
say "STAGE B done rc=$?"

# ── 4. all four baseline arms, in ONE pass, alone ──────────────────────────
say "STAGE C four downstream arms start (store-heavy; NOTHING runs beside this)"
python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --slicegan-root "$C22" --ddpm3d-root "$C23" \
    --out "$C14" \
    --arms slicegan_synthetic real_plus_slicegan_aug \
           ddpm3d_synthetic real_plus_ddpm3d_aug \
    > "$S/tail_downstream.log" 2>&1
say "STAGE C done rc=$? -> $C14"

# ── 5. the owed VAE rungs, then the notebooks ──────────────────────────────
if [ -n "$RF2_RUN" ]; then
    say "STAGE D rf-2 resume from $(basename "$RF2_RUN")"
    python scripts/train_vae.py resume "$(basename "$RF2_RUN")" latest.ckpt \
        > "$S/tail_rf2.log" 2>&1
    say "STAGE D done rc=$?"
else
    say "STAGE D SKIPPED: no r08-run-0010 directory found"
fi
say "STAGE E rf-64 start"
python scripts/train_vae.py run r08/reduction-factor-64 > "$S/tail_rf64.log" 2>&1
say "STAGE E done rc=$?"
for root in "$C18" "$C12"; do
    say "STAGE F notebook into $(basename "$root")"
    python scripts/analysis/build_eval_v4_notebook.py --root "$root" \
        > "$S/tail_nb_$(basename "$root").log" 2>&1
    say "STAGE F $(basename "$root") rc=$?"
done
say "TAIL COMPLETE"

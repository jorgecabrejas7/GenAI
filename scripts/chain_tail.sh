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
#   5. campaign 25, the porosity-only LDM: train, generate, measure.
#   6. campaign 24, the conditioning ablations.
#   7. campaign 25's own arms and the second-segmenter arms, again in ONE
#      store-heavy pass with nothing beside them.
#   8. rf-2 resume, rf-64, notebooks
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"
C22="$REPO/runs/campaigns/22-slicegan-baseline"
C23="$REPO/runs/campaigns/23-ddpm3d-baseline"
C24="$REPO/runs/campaigns/24-ablation"
C25="$REPO/runs/campaigns/25-ldm-phi-only"
#: The ablations are DIFFERENCES against campaign 18's rows, so they must come
#: from the checkpoint campaign 18 itself used. That is READ OUT OF ITS OWN
#: MANIFESTS rather than globbed for: a glob resolves by luck of directory
#: naming, and a fine-tune that produced a differently named run would silently
#: make every ablation a difference from a model that produced no baseline row.
read -r LDM06_RUN LDM06_CKPT LDM06_WEIGHTS <<EOF
$(python - "$C18" <<'PY'
import json, sys, glob, collections
c = collections.Counter()
for f in glob.glob(f"{sys.argv[1]}/*/volumes/*/manifest.json"):
    m = json.load(open(f))
    if m.get("model_run"):
        c[(m["model_run"], m.get("checkpoint_step"), m.get("weights"))] += 1
if c:
    run, step, w = c.most_common(1)[0][0]
    print(run, step, w or "ema")
PY
)
EOF
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

# ── 5. campaign 25: the porosity-only LDM ──────────────────────────────────
# ~18 h at the measured 1589 ms/step, and the config caps the cosine schedule
# at the same 40 000 steps so it COMPLETES rather than stopping part-way.
say "STAGE G phi-only training start (40k steps, ~18 h)"
python scripts/train_ldm.py run ldm25/phi_only > "$S/tail_phi_train.log" 2>&1
say "STAGE G done rc=$?"

PHI_RUN=$(ls -dt "$REPO"/runs/ldm/ldm25-run-*/ 2>/dev/null | head -1)
if [ -z "$PHI_RUN" ]; then
    say "STAGE H SKIPPED: no ldm25 run directory — training did not start"
else
    say "STAGE H phi-only generation from $(basename "$PHI_RUN")"
    # --joint: one chunk over the whole canvas, every neighbour UNKNOWN. This
    # model has no neighbour input, so chunking it would partition the canvas
    # into independent solves and put a seam at every chunk plane for nothing.
    for a in sampler porosity_global microstructure cfg; do
        say "STAGE H generate $a"
        python -m poregen.eval_v4.cli generate "$a" \
            --model "$PHI_RUN" --ckpt latest --out "$C25" --joint \
            > "$S/tail_phi_gen_$a.log" 2>&1
        say "STAGE H generate $a rc=$?"
    done
    # The real floor depends on the test panels and the detector and on no
    # model, so campaign 25 shares campaign 12's rather than recutting it.
    [ -e "$C25/real_floor" ] || ln -s "$C12/real_floor" "$C25/real_floor"
    for a in sampler porosity_global microstructure cfg; do
        say "STAGE I measure $a"
        python -m poregen.eval_v4.cli measure "$a" --root "$C25" \
            > "$S/tail_phi_measure_$a.log" 2>&1
        say "STAGE I measure $a rc=$?"
    done
    python -m poregen.eval_v4.cli report --root "$C25" > "$S/tail_phi_report.log" 2>&1
    say "STAGE I done rc=$? -> $C25"
fi

# ── 6. campaign 24: the conditioning ablations ─────────────────────────────
say "STAGE J ablation generation start (30 cases, DDIM-50)"
if [ -z "${LDM06_RUN:-}" ]; then
    say "STAGE J SKIPPED: campaign 18's manifests name no model run, so there is "
    say "  nothing for these ablations to be a difference FROM"
else
    say "STAGE J model $(basename "$LDM06_RUN") ckpt $LDM06_CKPT ($LDM06_WEIGHTS) — campaign 18's own"
    python -m poregen.eval_v4.cli generate ablation \
        --model "$LDM06_RUN" --ckpt "$LDM06_CKPT" --weights "$LDM06_WEIGHTS" \
        --out "$C24" > "$S/tail_ablation_gen.log" 2>&1
    say "STAGE J generate rc=$?"
fi
[ -e "$C24/real_floor" ] || ln -s "$C12/real_floor" "$C24/real_floor"
python -m poregen.eval_v4.cli measure ablation --root "$C24" \
    > "$S/tail_ablation_measure.log" 2>&1
say "STAGE J measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C24" --assessment ablation \
    > "$S/tail_ablation_report.log" 2>&1
say "STAGE J done rc=$? -> $C24"

# ── 7. the last store-heavy pass ───────────────────────────────────────────
say "STAGE K phi-only arms start (store-heavy; NOTHING runs beside this)"
python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --phi-only-root "$C25" --out "$C14" \
    --arms phi_only_synthetic real_plus_phi_only_aug \
    > "$S/tail_phi_downstream.log" 2>&1
say "STAGE K phi-only arms rc=$?"
# The second segmenter writes to its OWN directory. The arm names are the same
# ones the plain U-Net produced, and results merge by (arm, seed) — writing
# both into one directory would have the ResUNet silently overwrite campaign
# 14's headline numbers.
say "STAGE K second-segmenter arms start (ResUNet, separate directory)"
python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --slicegan-root "$C22" --out "$C14/resunet" \
    --architecture resunet \
    --arms real synthetic slicegan_synthetic \
    > "$S/tail_resunet_downstream.log" 2>&1
say "STAGE K done rc=$? -> $C14/resunet"

# ── 8. the owed VAE rungs, then the notebooks ──────────────────────────────
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

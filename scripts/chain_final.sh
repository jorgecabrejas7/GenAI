#!/usr/bin/env bash
# THE ONE REMAINING QUEUE. Takes over from chain_tail.sh at stage D.
#
# chain_tail.sh is stopped when this starts; its rf-2 child is NOT — killing
# the parent shell leaves the training reparented to init and still stepping,
# which is what this waits on. Same handover chain.sh -> chain_tail.sh used.
#
# ORDER, and why it is this order:
#   1  wait for rf-2                                   ~36 h from 2026-09-21 10:00
#   2  the OWED short items, which the chain passed over when it went K -> D:
#      the phi-only microstructure measure, campaign 23's DDPM measure and
#      report, and the budget-matched ldm06@40k comparator row. All short, all
#      blocked only on the card being free.
#   3  the review's GPU items 9, 8, 7, 10 in that order: 9 first because the
#      regenerations feed every table after them, 7 late because it is the
#      longest single job.
#   4  rf-64
#   5  the VRRAE study: V0, A, B, A0, each with a memory smoke test in its own
#      slot and the family table re-rendered after it.
#
# NOTHING RUNS BESIDE A TRAINING STAGE. Every stage here is sequential for
# that reason: a GPU job beside a dataloader killed one downstream arm with
# SIGBUS already, and store I/O beside a training run deadlocked rf-2 once.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"; C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"; C22="$REPO/runs/campaigns/22-slicegan-baseline"
C23="$REPO/runs/campaigns/23-ddpm3d-baseline"; C25="$REPO/runs/campaigns/25-ldm-phi-only"
C27="$REPO/runs/campaigns/27-budget-matched-40k"; C28="$REPO/runs/campaigns/28-vrrae-family"
: "${RF2_PID:?set RF2_PID to the running rf-2 process}"
cd "$REPO" || exit 1
say() { printf '%s  FINAL %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
run() { say "$1 start"; shift; local tag="$1"; shift; "$@" > "$S/$tag.log" 2>&1; say "$tag rc=$?"; }

read -r LDM06_RUN LDM06_CKPT LDM06_W <<EOF2
$(python - "$C18" <<'PY'
import json, sys, glob, collections
c = collections.Counter()
for f in glob.glob(f"{sys.argv[1]}/*/volumes/*/manifest.json"):
    m = json.load(open(f))
    if m.get("model_run"): c[(m["model_run"], m.get("checkpoint_step"), m.get("weights"))] += 1
if c:
    r, s, w = c.most_common(1)[0][0]; print(r, s, w or "ema")
PY
)
EOF2

# ── 1. wait for rf-2, by PID and then by its own completion ────────────────
say "armed; waiting for rf-2 pid $RF2_PID (~36 h)"
while [ -r "/proc/$RF2_PID/cmdline" ] \
      && tr '\0' ' ' < "/proc/$RF2_PID/cmdline" | grep -q "train_vae\.py"; do
    sleep 300
done
say "rf-2 exited; the card is free"

# ── 2. the owed short items ────────────────────────────────────────────────
run "OWED phi-only microstructure" owed_phi_micro \
    python -m poregen.eval_v4.cli measure microstructure --root "$C25"
run "OWED phi-only report" owed_phi_report \
    python -m poregen.eval_v4.cli report --root "$C25"
run "OWED ddpm3d measure" owed_ddpm_measure \
    python -m poregen.eval_v4.cli measure ddpm3d --root "$C23"
run "OWED ddpm3d report" owed_ddpm_report \
    python -m poregen.eval_v4.cli report --root "$C23" --assessment ddpm3d

# The budget-matched comparator: the FULL model at the phi-only model's step
# count, so the dose-response rows can be read side by side without the
# 130k+15k versus 40k confound.
say "OWED budget-matched ldm06@40k (porosity_global, 48 volumes)"
mkdir -p "$C27"
[ -e "$C27/real_floor" ] || ln -s "$C12/real_floor" "$C27/real_floor"
run "OWED 40k generate" owed_40k_gen \
    python -m poregen.eval_v4.cli generate porosity_global \
    --model "$LDM06_RUN" --ckpt 40000 --weights "$LDM06_W" --out "$C27"
run "OWED 40k sampler" owed_40k_sampler \
    python -m poregen.eval_v4.cli generate sampler \
    --model "$LDM06_RUN" --ckpt 40000 --weights "$LDM06_W" --out "$C27" \
    --only 192_ddim50_seed101 192_ddim50_seed202 192_ddim50_seed303
for a in porosity_global sampler; do
    run "OWED 40k measure $a" "owed_40k_measure_$a" \
        python -m poregen.eval_v4.cli measure "$a" --root "$C27"
done
run "OWED 40k report" owed_40k_report python -m poregen.eval_v4.cli report --root "$C27"

# ── 3. the review's GPU items ──────────────────────────────────────────────
# 9 first: every table after it reads these volumes.
say "ITEM 9 regeneration on the refitted priors and the corrected window phi"
for a in porosity_local field_stats multichunk; do
    [ "$a" = field_stats ] && continue
    run "ITEM9 generate $a" "item9_gen_$a" \
        python -m poregen.eval_v4.cli generate "$a" \
        --model "$LDM06_RUN" --ckpt "$LDM06_CKPT" --weights "$LDM06_W" --out "$C18"
done
for a in porosity_local field_stats multichunk stress_geometry ood_conditioning; do
    run "ITEM9 measure $a" "item9_measure_$a" \
        python -m poregen.eval_v4.cli measure "$a" --root "$C18"
done
run "ITEM9 report" item9_report python -m poregen.eval_v4.cli report --root "$C18"

# 8: the missing factorial cell — production chunking, no neighbour content.
run "ITEM 8 chunked-without-neighbours at 1024" item8 \
    python -m poregen.eval_v4.cli generate assembly_modes \
    --model "$LDM06_RUN" --ckpt "$LDM06_CKPT" --weights "$LDM06_W" --out "$C18" \
    --chunk-overlap 0

# 7: the matched control for the facedrop causal claim.
run "ITEM 7 matched control fine-tune (drop_nb_face 0)" item7_train \
    python scripts/train_ldm.py run ldm06/facedrop_control

# 10: the downstream arms, store-heavy, alone.
say "ITEM 10 downstream arms (store-heavy; NOTHING runs beside this)"
run "ITEM10 low-real-budget pair" item10_lowreal \
    python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --out "$C14" \
    --arms real_2k real_2k_plus_synthetic_14k
# Exposure-preserving: the same 16k real patches as the `real` arm PLUS 16k
# synthetic, at DOUBLE the steps so each patch is seen as often as in `real`.
# A separate invocation and a separate directory because the STEP BUDGET
# differs, and a training setting must never live on an Arm.
run "ITEM10 exposure-preserving (16k steps)" item10_exposure \
    python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --out "$C14/exposure_preserved" --steps 16000 \
    --arms real real_plus_synthetic_aug

# ── 4. rf-64 ───────────────────────────────────────────────────────────────
run "rf-64" rf64 python scripts/train_vae.py run r08/reduction-factor-64

# ── 5. the VRRAE study ─────────────────────────────────────────────────────
# Order set by the author with the collaborator: V0, A, B, A0.
vrrae_row() {  # re-render the family table with whatever runs now exist
    python scripts/analysis/vrrae_family_table.py --out "$C28" \
        > "$S/vrrae_table_$1.log" 2>&1
    say "family table re-rendered after $1 rc=$?"
}
smoke() {      # one training step, for memory, before a queue slot is spent
    say "SMOKE $1"
    python scripts/analysis/vae_smoke_step.py --experiment "$1" \
        > "$S/smoke_$2.log" 2>&1
    local rc=$?; say "SMOKE $1 rc=$rc"; return $rc
}

run "VRRAE V0 (vrrae/beta0)" vrrae_v0 python scripts/train_vae.py run vrrae/beta0
vrrae_row v0

if smoke vrrae/a a; then
    run "VRRAE A (vrrae/a)" vrrae_a python scripts/train_vae.py run vrrae/a
else
    say "VRRAE A SKIPPED: the smoke step did not fit; see $S/smoke_a.log"
fi
vrrae_row a

if smoke vrrae/b b; then
    run "VRRAE B (vrrae/b)" vrrae_b python scripts/train_vae.py run vrrae/b
else
    say "VRRAE B at batch 1536 did not fit — running the fallback file"
    run "VRRAE B fallback" vrrae_b_fb python scripts/train_vae.py run vrrae/b_fallback
fi
vrrae_row b

run "VRRAE A0 (vrrae/a0)" vrrae_a0 python scripts/train_vae.py run vrrae/a0
vrrae_row a0

# ── 6. the notebooks ───────────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    run "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "FINAL COMPLETE"

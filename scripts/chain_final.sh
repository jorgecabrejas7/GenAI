#!/usr/bin/env bash
# THE ONE REMAINING QUEUE. Takes over from chain_tail.sh at stage D.
#
# chain_tail.sh is stopped when this starts; its rf-2 child is NOT — killing
# the parent shell leaves the training reparented to init and still stepping,
# which is what this waits on. Same handover chain.sh -> chain_tail.sh used.
#
# ORDER — REVISED 2026-09-21. THE PAPER'S NUMBERS COME FIRST.
#
# rf-2 was PAUSED at a checkpoint to let this run. It had 35.7 h left and every
# number the paper is waiting on sat behind it. Its resume is verified — this
# run has already been resumed once, from step 22000 — and it is resumed below
# with the same command, from `latest.ckpt`.
#
#   1  the OWED short items: the phi-only microstructure measure, campaign 23's
#      DDPM measure and report, and the budget-matched ldm06@40k comparator.
#   2  the review's GPU items 9, 8, 7, 10: 9 first because the regenerations
#      feed every table after them, 8 next because it is half an hour, 7 late
#      because it is the longest, 10 last because it is store-heavy.
#   3  rf-2, resumed to completion.
#   4  the VRRAE study: V0, A, B, A0, each with a memory smoke test in its own
#      slot and the family table re-rendered after it.
#   5  rf-64 LAST. It is 55 h and nothing depends on it — it brackets the sweep
#      for the paper and does not feed the compressor decision.
#   6  the notebooks.
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
#: The rf-2 run to resume once the paper's work is done. Resolved by name, and
#: checked to exist before anything else runs: discovering at hour 20 that the
#: resume target is gone would waste the whole reordering.
RF2_RUN=$(ls -dt "$REPO"/runs/vae/r08-run-0010-*/ 2>/dev/null | head -1)
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

# ── 0. the card must be free before anything starts ────────────────────────
if [ -z "$RF2_RUN" ]; then
    say "ABORT: no r08-run-0010 directory, so rf-2 could not be resumed later"
    exit 1
fi
for f in latest.ckpt resolved_config.yaml; do
    [ -f "$RF2_RUN/$f" ] || { say "ABORT: $RF2_RUN/$f is missing"; exit 1; }
done
say "rf-2 will resume from $(basename "$RF2_RUN")/latest.ckpt after the paper work"
while pgrep -f "train_vae\.py resume" > /dev/null 2>&1; do
    say "waiting: rf-2 is still stopping"
    sleep 30
done
say "the card is free; starting the paper's queue"

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
#
# IT IS ldm06/BASE, NOT THE FACEDROP RUN campaign 18 used. $LDM06_RUN above is
# run-0002, the 15 000-step facedrop fine-tune, which HAS NO STEP 40 000 — it
# never ran that far. The step-40k weights only exist in run-0001, ldm06/base,
# which is the run that trained to 130 000. Resolved separately and CHECKED,
# because a missing checkpoint here would have failed the stage three hours in.
LDM06_BASE=$(ls -dt "$REPO"/runs/ldm/ldm06-run-0001-*/ 2>/dev/null | head -1)
say "OWED budget-matched ldm06@40k (porosity_global, 48 volumes)"
mkdir -p "$C27"
[ -e "$C27/real_floor" ] || ln -s "$C12/real_floor" "$C27/real_floor"
# The file is `ldm_step00040000.ckpt` — the LDM checkpoints are NOT prefixed
# with the run name the way the VAE ones are, and guessing that cost one
# wrong guard already.
if [ -z "$LDM06_BASE" ] || [ ! -f "$LDM06_BASE/checkpoints/ldm_step00040000.ckpt" ]; then
    say "OWED 40k SKIPPED: no step-40000 checkpoint under ${LDM06_BASE:-<no ldm06/base run>}"
else
run "OWED 40k generate" owed_40k_gen \
    python -m poregen.eval_v4.cli generate porosity_global \
    --model "$LDM06_BASE" --ckpt 40000 --weights ema --out "$C27"
run "OWED 40k sampler" owed_40k_sampler \
    python -m poregen.eval_v4.cli generate sampler \
    --model "$LDM06_BASE" --ckpt 40000 --weights ema --out "$C27" \
    --only 192_ddim50_seed101 192_ddim50_seed202 192_ddim50_seed303
for a in porosity_global sampler; do
    run "OWED 40k measure $a" "owed_40k_measure_$a" \
        python -m poregen.eval_v4.cli measure "$a" --root "$C27"
done
run "OWED 40k report" owed_40k_report python -m poregen.eval_v4.cli report --root "$C27"
fi

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

# ── 3. rf-2, resumed to completion ─────────────────────────────────────────
run "rf-2 resumed to completion" rf2_resume \
    python scripts/train_vae.py resume "$(basename "$RF2_RUN")" latest.ckpt

# ── 4. the VRRAE study ─────────────────────────────────────────────────────
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

# ── 5. rf-64, last ─────────────────────────────────────────────────────────
run "rf-64" rf64 python scripts/train_vae.py run r08/reduction-factor-64

# ── 6. the notebooks ───────────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    run "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "FINAL COMPLETE"

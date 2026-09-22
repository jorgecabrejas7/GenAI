#!/usr/bin/env bash
# The fixes the chain's own log exposed, then everything after item 10.
#
# WHAT WENT WRONG, so the next reader does not repeat it. Four stages of
# chain_final.sh reported rc=0 and did nothing, and two failed outright:
#
#  * item9_gen_* finished in 3-4 s. `eval_v4 generate` SKIPS a case whose
#    manifest.json exists — the property that makes a long campaign resumable —
#    so a "regeneration" stage rebuilds nothing unless the old cases are moved
#    aside first. They are moved here, never deleted.
#  * item8 passed `--chunk-overlap 0`, which is the DEFAULT and therefore a
#    no-op, and is not the missing factorial cell in any case. That cell is
#    production chunking with the neighbour CONTENT removed; it is now a real
#    case in assembly_modes (`chunked_no_neighbours`).
#  * item9 measured stress_geometry and ood_conditioning against campaign 18.
#    They live in campaigns 19 and 20. Both raised FileNotFoundError.
#  * item7 trained the control and never measured it. The band profile is the
#    whole point of that control and is run here.
#  * ddpm3d had no measurer at all — the same gap slicegan had — so campaign 23
#    had eleven volumes and no numbers. Fixed in code and already measured.
#
# THE RULE THIS ADDS: a stage that should generate and finishes in under a
# minute is a FAILURE, and the chain now says so itself.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"; C17="$REPO/runs/campaigns/17-chunk-band-trial"
C18="$REPO/runs/campaigns/18-eval-v4-final"; C19="$REPO/runs/campaigns/19-stress-geometry"
C20="$REPO/runs/campaigns/20-ood-conditioning"; C28="$REPO/runs/campaigns/28-vrrae-family"
C29="$REPO/runs/campaigns/29-facedrop-control"
RF2_RUN=$(ls -dt "$REPO"/runs/vae/r08-run-0010-*/ 2>/dev/null | head -1)
CTRL_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-0003-*/ 2>/dev/null | head -1)
: "${WAIT_PID:?set WAIT_PID to the running item-10 process}"
cd "$REPO" || exit 1
say() { printf '%s  FIX %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

#: Run a stage and CHECK IT DID SOMETHING. `min_seconds` is how long the stage
#: must take to be credible: a generate that returns instantly skipped its
#: cases, which is exactly the failure this script exists to correct.
run() {
    local label="$1" tag="$2" min="$3"; shift 3
    say "$label start"
    local t0=$SECONDS
    "$@" > "$S/$tag.log" 2>&1
    local rc=$? dt=$((SECONDS - t0))
    if [ "$rc" -ne 0 ]; then
        say "$tag rc=$rc FAILED after ${dt}s — see $S/$tag.log"
    elif [ "$min" -gt 0 ] && [ "$dt" -lt "$min" ]; then
        say "$tag rc=0 but finished in ${dt}s, under the ${min}s this stage needs"
        say "  SUSPECT NO-OP — check $S/$tag.log before believing it"
    else
        say "$tag rc=0 in ${dt}s"
    fi
    return $rc
}

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

say "armed; waiting for item 10 (pid $WAIT_PID) to finish"
while [ -r "/proc/$WAIT_PID/cmdline" ] \
      && tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" | grep -q "downstream_utility"; do
    sleep 120
done
say "item 10 finished; the card is free"

# ── 9b. move the stale cases aside, THEN regenerate ────────────────────────
for pair in "$C18:porosity_local" "$C18:multichunk" \
            "$C19:stress_geometry" "$C20:ood_conditioning"; do
    root="${pair%%:*}"; a="${pair##*:}"
    run "9b archive stale $a" "fix9_archive_$a" 0 \
        python scripts/analysis/archive_stale_cases.py --root "$root" --assessment "$a" --apply
done
# 60 s minimum: 28 cases at DDIM-50/200 cannot be rebuilt faster than that, so
# anything quicker means the archive step did not take.
for pair in "$C18:porosity_local" "$C18:multichunk" \
            "$C19:stress_geometry" "$C20:ood_conditioning"; do
    root="${pair%%:*}"; a="${pair##*:}"
    run "9b regenerate $a" "fix9_gen_$a" 60 \
        python -m poregen.eval_v4.cli generate "$a" \
        --model "$LDM06_RUN" --ckpt "$LDM06_CKPT" --weights "$LDM06_W" --out "$root"
done
# Measured against the campaign each assessment actually lives in.
for pair in "$C18:porosity_local" "$C18:field_stats" "$C18:multichunk" \
            "$C19:stress_geometry" "$C20:ood_conditioning"; do
    root="${pair%%:*}"; a="${pair##*:}"
    run "9b measure $a" "fix9_measure_$a" 0 \
        python -m poregen.eval_v4.cli measure "$a" --root "$root"
done
for root in "$C18" "$C19" "$C20"; do
    run "9b report $(basename "$root")" "fix9_report_$(basename "$root")" 0 \
        python -m poregen.eval_v4.cli report --root "$root"
done

# ── 8b. the factorial cell, as a real case this time ───────────────────────
run "8b chunked_no_neighbours (5 cases, 2 at 1024)" fix8_gen 60 \
    python -m poregen.eval_v4.cli generate assembly_modes \
    --model "$LDM06_RUN" --ckpt "$LDM06_CKPT" --weights "$LDM06_W" --out "$C18"
run "8b measure assembly_modes" fix8_measure 0 \
    python -m poregen.eval_v4.cli measure assembly_modes --root "$C18"
run "8b report" fix8_report 0 \
    python -m poregen.eval_v4.cli report --root "$C18" --assessment assembly_modes

# ── 7b. the matched control's band profile ─────────────────────────────────
if [ -z "$CTRL_RUN" ]; then
    say "7b SKIPPED: no ldm06-run-0003 (the facedrop control) on disk"
else
    say "7b band profile from $(basename "$CTRL_RUN") — the control, drop_nb_face 0"
    run "7b control band volumes" fix7_gen 60 \
        python scripts/analysis/chunk_band_trial.py --model "$CTRL_RUN" \
        --ckpt latest --out "$C29" --arms fd_baseline --only-1024
    run "7b control band report" fix7_report 0 \
        python scripts/analysis/chunk_band_trial_report.py --trial "$C29" --baseline "$C12"
fi

# ── then everything the original chain still owes ──────────────────────────
run "rf-2 resumed to completion" rf2_resume 0 \
    python scripts/train_vae.py resume "$(basename "$RF2_RUN")" latest.ckpt

vrrae_row() { python scripts/analysis/vrrae_family_table.py --out "$C28" \
    > "$S/vrrae_table_$1.log" 2>&1; say "family table re-rendered after $1 rc=$?"; }
smoke() { say "SMOKE $1"; python scripts/analysis/vae_smoke_step.py --experiment "$1" \
    > "$S/smoke_$2.log" 2>&1; local rc=$?; say "SMOKE $1 rc=$rc"; return $rc; }

run "VRRAE V0" vrrae_v0 0 python scripts/train_vae.py run vrrae/beta0; vrrae_row v0
if smoke vrrae/a a; then run "VRRAE A" vrrae_a 0 python scripts/train_vae.py run vrrae/a
else say "VRRAE A SKIPPED: smoke step did not fit"; fi
vrrae_row a
if smoke vrrae/b b; then run "VRRAE B" vrrae_b 0 python scripts/train_vae.py run vrrae/b
else say "VRRAE B at 1536 did not fit — fallback"
     run "VRRAE B fallback" vrrae_b_fb 0 python scripts/train_vae.py run vrrae/b_fallback; fi
vrrae_row b
run "VRRAE A0" vrrae_a0 0 python scripts/train_vae.py run vrrae/a0; vrrae_row a0

run "rf-64" rf64 0 python scripts/train_vae.py run r08/reduction-factor-64
for root in "$C18" "$C12"; do
    run "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" 0 \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "FIX CHAIN COMPLETE"

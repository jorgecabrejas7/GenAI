#!/usr/bin/env bash
# Owns the eval-v4 queue outright, from generation to the bracket rungs.
#
# WHY IT OWNS IT. layup is 9 cases at 1024x1024x192 and DDIM-200 — about 6
# hours, 48% of the remaining generation. The user wants to inspect the volumes
# while that runs, so layup moves to the END and the inspection pack is built
# before it. The main runner's stage order is fixed and layup sits in the
# middle of it, so the reorder cannot be done from there.
#
# An earlier version of this script waited for the runner to reach a stage
# boundary and killed it there. That is gone: `eval_v4 generate` skips any case
# that already has a manifest.json, and save_case writes the manifest LAST, so
# stopping the runner costs one in-flight case and nothing else. Taking the
# queue outright removes the kill race, the log-tailing and the dependence on
# the runner reaching any particular line.
#
# ORDER, and what fixes each position:
#
#   label_uncertainty    starts NOW, in parallel with generation. CPU only,
#                        nice 19, one thread: it must not slow the card.
#   GENERATE             the runner's own list first, then the three stages it
#                        had queued behind layup
#   INSPECTION PACK      before layup, so inspection happens while layup runs
#   field_stats          MEASURE-ONLY: it generates nothing and re-reads what
#                        porosity_local and multichunk wrote, so it cannot run
#                        before multichunk exists.
#   memorisation smoke   the memorisation code has never touched hardware. Two
#                        volumes under a hard time/memory cap prove the VAE
#                        load, the encode, decode_grey and both store passes
#                        before the full search is given the GPU for hours.
#   assembly_modes       GPU. Must precede the full memorisation pass, which
#                        searches assembly_modes volumes among others.
#   layup                full 3 seeds — the paper needs the 3-seed table
#   measure / report     the FULL memorisation pass rides inside `measure
#                        microstructure`; there is no separate full stage,
#                        because a standalone one would search the 272 GB
#                        store twice. That measure runs under the cap.
#   pack final
#   OWED rung reports, COMPARE
#   GATE -> decoder-ft -> redecode   decoder-ft preempts downstream_utility
#   downstream_utility   LAST of the five: it trains a network on the GPU and
#                        is the slowest. It must not delay the second pack or
#                        the fine-tune.
#   rf-2, rf-64          ~26 h each, lowest priority
#
# Usage:  bash scripts/ldm06_post_tail.sh
set -uo pipefail          # NOT -e: a failing stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
EVAL_CAMP="$REPO/runs/campaigns/12-eval-v4"
FT_CAMP="$REPO/runs/campaigns/11-decoder-ft"
FT_GO="$REPO/runs/campaigns/decoder_ft_go"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/post_ldm06.log"
LDM_RUN=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*/ 2>/dev/null | head -1)
#: Generation order. The runner's own list, then the stages it had queued
#: behind layup. layup and assembly_modes are NOT here — they run later, and
#: for reasons the header gives.
GEN_ORDER=(sampler microstructure porosity_global porosity_local geometry surface
           multichunk assembly cfg)
#: The caps the memorisation passes are held to.
MEMO_MAX_MIN=60
MEMO_MAX_GB=20
#: Rungs owed a full-split report, and the bracket rungs, from the runner.
OWED=(base reduction-factor-8 reduction-factor-32 reduction-factor-4)
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)

mkdir -p "$SCRATCH" "$FT_CAMP"; cd "$REPO" || exit 1
say() { printf '%s  TAIL %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  TAIL %s\n' "$(date -Is)" "$*"; }

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

gen() {
    local a="$1"
    say "GEN $a start"
    python -m poregen.eval_v4.cli generate "$a" \
        --model "$LDM_RUN" --ckpt best --out "$EVAL_CAMP" --save-latents \
        > "$SCRATCH/tail_gen_${a}.log" 2>&1
    say "GEN $a done rc=$?"
}

pack() {
    say "INSPECT pack start ($1)"
    python scripts/analysis/eval_v4_inspection_pack.py --root "$EVAL_CAMP" \
        > "$SCRATCH/tail_inspection_$1.log" 2>&1
    say "INSPECT pack done rc=$? ($1) -> $EVAL_CAMP/inspection"
}

# ── 0. label_uncertainty starts immediately, on the CPU ─────────────────────
# Nine threshold variants over three full test volumes, ~20 min. It reads
# scans and writes masks; it never touches the card. Running it here costs the
# generation nothing and takes it off the critical path entirely.
if [ -f scripts/analysis/label_uncertainty.py ]; then
    say "STAGE label_uncertainty start (CPU, background, nice 19)"
    # Its own campaign (13-label-uncertainty) is the script's default; an
    # --out into the eval-v4 tree would put one question's answer inside
    # another's campaign.
    ( nice -n 19 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        python scripts/analysis/label_uncertainty.py \
        > "$SCRATCH/tail_label_uncertainty.log" 2>&1
      say "STAGE label_uncertainty done rc=$?" ) &
else
    say "SKIP label_uncertainty — scripts/analysis/label_uncertainty.py absent"
fi

# ── 1. take the queue from the runner ───────────────────────────────────────
# Kill the runner, but LET ITS IN-FLIGHT GENERATE FINISH. Killing the parent
# shell does not kill the python child, and that child is hours into a case.
# Nothing is gained by killing it: `generate` regenerates only cases with no
# manifest.json and the manifest is written last, so the finished cases are
# kept either way — but the one still running would have to start over.
# Waiting costs the queue nothing, because the card is busy with it regardless.
for p in $(pgrep -f "bash scripts/ldm06_post\.sh" 2>/dev/null); do
    kill "$p" 2>/dev/null && say "stopped the main runner (pid $p); its in-flight generate is left to finish"
done
sleep 3
if pgrep -f "eval_v4\.cli generate" >/dev/null 2>&1; then
    say "waiting for the runner's in-flight generate to finish — it keeps the case it is on"
fi
while pgrep -f "eval_v4\.cli generate|r08_rung_report\.py" >/dev/null 2>&1; do sleep 20; done
say "GPU free; tail owns the queue"

# ── 2. generation, in the order the header gives ────────────────────────────
for a in "${GEN_ORDER[@]}"; do
    gen "$a"
done

# ── 3. the pack, BEFORE layup, so inspection happens while layup runs ───────
pack before_layup
say "PACK READY for inspection -> $EVAL_CAMP/inspection — SEND PATH TO SUPERVISOR"

# ── 4. field_stats: measure-only, and only once multichunk exists ───────────
# It is NOT in cases.ASSESSMENTS and has no `generate` verb; asking for one
# would have skipped it silently for the whole campaign.
say "STAGE field_stats start (measure-only, CPU)"
nice -n 19 env CUDA_VISIBLE_DEVICES= python -m poregen.eval_v4.cli measure field_stats \
    --root "$EVAL_CAMP" > "$SCRATCH/tail_field_stats.log" 2>&1
say "STAGE field_stats done rc=$?"

# ── 5. memorisation smoke, under the cap ────────────────────────────────────
say "STAGE memorisation_smoke start (2 volumes, cap ${MEMO_MAX_MIN} min / ${MEMO_MAX_GB} GB)"
python scripts/analysis/memorisation_smoke.py --root "$EVAL_CAMP" \
    --max-minutes "$MEMO_MAX_MIN" --max-gb "$MEMO_MAX_GB" \
    > "$SCRATCH/tail_memorisation_smoke.log" 2>&1
smoke_rc=$?
say "STAGE memorisation_smoke done rc=$smoke_rc — REPORT TO SUPERVISOR"
if [ "$smoke_rc" -ne 0 ]; then
    say "MEMO SMOKE FAILED rc=$smoke_rc — the full pass inside measure microstructure will be attempted anyway, under the same cap, and reported"
fi

# ── 6. assembly_modes, then layup ───────────────────────────────────────────
# assembly_modes before the measure stage, because the full memorisation pass
# that rides inside `measure microstructure` searches its volumes.
gen assembly_modes
gen layup

# ── 7. measure, report, and a final pack that includes everything ───────────
# Driven by MEASURERS, not by cases.ASSESSMENTS: field_stats is a measurer with
# no case list and no volumes directory, so a loop over the case list drops it.
MEASURE_LIST=$(python -c "
from poregen.eval_v4.measure import MEASURERS
print(' '.join(sorted(MEASURERS)))")
MEASURE_ONLY=$(python -c "
from poregen.eval_v4.cases import MEASURE_ONLY
print(' '.join(MEASURE_ONLY))")
for a in $MEASURE_LIST; do
    case " $MEASURE_ONLY " in
        *" $a "*) say "MEASURE $a already done above (measure-only)"; continue ;;
    esac
    if [ ! -d "$EVAL_CAMP/$a/volumes" ]; then
        say "MEASURE $a skipped: no volumes at $EVAL_CAMP/$a/volumes"
        continue
    fi
    if [ "$a" = microstructure ]; then
        # This one carries the FULL memorisation pass (measure.py calls
        # memorisation() with its defaults), so it is the long GPU stage the
        # cap was asked for — not a CPU measure like the rest.
        say "MEASURE microstructure start — carries the FULL memorisation pass, cap ${MEMO_MAX_MIN} min / ${MEMO_MAX_GB} GB"
        timeout --signal=KILL "$(( MEMO_MAX_MIN * 60 ))" \
            python -m poregen.eval_v4.cli measure microstructure --root "$EVAL_CAMP" \
            > "$SCRATCH/tail_measure_${a}.log" 2>&1
        rc=$?
        if [ "$rc" -eq 137 ]; then
            say "MEASURE microstructure STOPPED at the ${MEMO_MAX_MIN} min cap — the memorisation pass exceeded it. REPORT TO SUPERVISOR; the queue continues"
        else
            say "MEASURE microstructure done rc=$rc"
        fi
        continue
    fi
    say "MEASURE $a start"
    python -m poregen.eval_v4.cli measure "$a" --root "$EVAL_CAMP" \
        > "$SCRATCH/tail_measure_${a}.log" 2>&1
    say "MEASURE $a done rc=$?"
done
say "REPORT start"
python -m poregen.eval_v4.cli report --root "$EVAL_CAMP" > "$SCRATCH/tail_report.log" 2>&1
say "REPORT done rc=$?"
pack final
say "PACK FINAL READY -> $EVAL_CAMP/inspection — SEND PATH TO SUPERVISOR"

# ── 8. the owed full-split rung reports (GPU) ───────────────────────────────
for exp in "${OWED[@]}"; do
    d=$(run_dir_for "$exp")
    if [ -z "$d" ]; then say "OWED $exp skipped: no run directory"; continue; fi
    say "OWED $exp report start"
    python scripts/analysis/r08_rung_report.py --run "$d" > "$SCRATCH/report_${exp}.log" 2>&1
    say "OWED $exp report done rc=$?"
done
say "COMPARE start (final paper table)"
python scripts/analysis/r08_rung_report.py --compare > "$SCRATCH/r08_compare_final.log" 2>&1
say "COMPARE done rc=$?"

# ── 9. decoder-ft preempts downstream_utility ───────────────────────────────
# If the go-ahead is already here, the fine-tune goes first: it is the gated
# decision the user is waiting on, and downstream_utility would hold the card
# for hours in front of it. If it is not here, downstream_utility runs now
# rather than leaving the card idle, and the gate is waited on afterwards.
decoder_ft() {
    say "DECODER-FT start (r08/decoder-ft)"
    python scripts/train_vae.py run r08/decoder-ft > "$SCRATCH/decoder_ft.log" 2>&1
    local ft_rc=$?
    say "DECODER-FT done rc=$ft_rc"
    local FT_RUN; FT_RUN=$(run_dir_for decoder-ft)
    say "DECODER-FT run dir: ${FT_RUN:-<none>}"
    local BASE_CKPT="$(run_dir_for reduction-factor-8)best.ckpt"
    if [ "$ft_rc" -eq 0 ] && [ -n "$FT_RUN" ] && [ -f "${FT_RUN}best.ckpt" ]; then
        say "REDECODE start (baseline $BASE_CKPT)"
        python scripts/analysis/decoder_ft_redecode.py \
            --baseline "$BASE_CKPT" --finetuned "${FT_RUN}best.ckpt" \
            --latents "$EVAL_CAMP/*/*/latents.npy" \
            --out "$FT_CAMP/redecode" \
            > "$SCRATCH/redecode.log" 2>&1
        say "REDECODE done rc=$?"
        say "GATE TABLE -> $FT_CAMP/redecode/results.json — SEND TO SUPERVISOR"
    else
        say "REDECODE skipped: fine-tune rc=$ft_rc, run=${FT_RUN:-<none>}"
    fi
}

downstream_utility() {
    if [ ! -f scripts/analysis/downstream_utility.py ]; then
        say "SKIP downstream_utility — scripts/analysis/downstream_utility.py absent"
        return
    fi
    say "STAGE downstream_utility start (GPU, trains a network — the slowest of the five)"
    # Reads the synthetic volumes from the eval-v4 campaign, writes to its own
    # (14-downstream-utility) — both are the script's defaults.
    python scripts/analysis/downstream_utility.py --campaign-root "$EVAL_CAMP" \
        > "$SCRATCH/tail_downstream_utility.log" 2>&1
    say "STAGE downstream_utility done rc=$?"
}

if [ -e "$FT_GO" ]; then
    say "GATE already released: $FT_GO present — decoder-ft goes before downstream_utility"
    decoder_ft
    downstream_utility
else
    downstream_utility
    if [ ! -e "$FT_GO" ]; then
        say "GATE blocked: waiting for $FT_GO (polling 60 s, no timeout)"
        while [ ! -e "$FT_GO" ]; do sleep 60; done
    fi
    say "GATE released: $FT_GO present"
    decoder_ft
fi

# ── 10. the bracket rungs, last: ~26 h each ─────────────────────────────────
for exp in "${TAIL_RUNGS[@]}"; do
    say "START r08/$exp"
    python scripts/train_vae.py run "r08/$exp" > "$SCRATCH/r08_${exp}.log" 2>&1
    rc=$?
    say "DONE r08/$exp rc=$rc"
    [ "$rc" -ne 0 ] && say "FAIL r08/$exp rc=$rc — chain continues"
done

wait
say "TAIL COMPLETE"

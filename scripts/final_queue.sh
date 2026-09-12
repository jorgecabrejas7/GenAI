#!/usr/bin/env bash
# THE ONE QUEUE. Everything the card does from the end of the facedrop trial to
# the final notebook, in the author's order, chained with no gaps.
#
#   1. (not here) fd_a2s + the rim test — facedrop_post.sh owns those and is
#      STOPPED at the rim-test boundary by scripts/stop_fdpost_after_rim.sh.
#   2. campaign 18 regeneration      generate -> pack -> measure -> report -> notebook
#   3. campaign 19 stress, DDIM-50
#   4. campaign 20 ood_conditioning
#   5. decoder fine-tune, then redecode + the D43 gate table on CAMPAIGN 18
#      volumes (not 12), DDIM-50 and DDIM-200 rows
#   6. downstream utility on the campaign-18 grey+label
#   7. campaign 19 stress, DDIM-200
#   8. r08 rf-2 and rf-64
#   9. final notebook rebuild into 18 and 12
#
# WHY THE DECODER FINE-TUNE MOVED HERE. It used to sit in facedrop_post.sh
# behind runs/campaigns/decoder_ft_go. The gate mechanism is KEPT — stage 5
# still waits on that file — but the stage runs here, where it cannot start
# before the regeneration it is supposed to be measured on. In the old script
# it would have started the moment the gate file appeared, which on this
# timeline is hours before campaign 18 exists.
#
# STAGE 6 USES THE ORIGINAL r08 DECODER, not the fine-tuned one. The redecode
# is a COMPARISON until its gates are read; making it the default before that
# would put an unmeasured decoder into the one experiment that asks whether the
# synthetic data is useful.
set -uo pipefail          # NOT -e: a failing stage must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
GO="$REPO/runs/campaigns/regen_go"
FT_GO="$REPO/runs/campaigns/decoder_ft_go"
FD_LOG="$REPO/runs/campaigns/09-r08-latent-sweep/post_ldm06.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"
C19="$REPO/runs/campaigns/19-stress-geometry"
C20="$REPO/runs/campaigns/20-ood-conditioning"
FT_CAMP="$REPO/runs/campaigns/11-decoder-ft"
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/final_queue.log"
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)
mkdir -p "$S" "$C14" "$C18" "$C19" "$C20" "$FT_CAMP"; cd "$REPO" || exit 1
say() { printf '%s  QUEUE %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

#: Resolve an r08 experiment id to its run directory. Copied in behaviour from
#: facedrop_post.sh, which is being retired for the stages below.
run_dir_for() {
    local want="r08/$1" d got
    for d in $(ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null); do
        [ -f "${d}best.ckpt" ] || continue
        got=$(python - "$d" <<'PYRUN'
import sys, json, pathlib
try:
    print(json.loads((pathlib.Path(sys.argv[1]) / "run_metadata.json").read_text())
          .get("experiment_id", ""))
except Exception:
    print("")
PYRUN
)
        [ "$got" = "$want" ] && { printf '%s' "$d"; return 0; }
    done
    return 1
}

say "armed"

# ── wait: the rim test must be finished and the card free ───────────────────
# The rim test is the last stage facedrop_post.sh owns. Two generation jobs on
# one 121 GB unified pool is how a CUDA allocation fails while `free` still
# reports tens of GB.
while ! grep -q "RIM mixed-set done" "$FD_LOG" 2>/dev/null; do sleep 60; done
say "rim test finished"
while pgrep -f "bash scripts/facedrop_post\.sh" >/dev/null 2>&1; do
    say "waiting for facedrop_post.sh to stop"
    sleep 30
done
say "facedrop_post.sh is stopped — the card is free"

: "${CKPT_RUN:=}"; : "${SAMPLER:=}"
if [ -e "$GO" ]; then . "$GO"; fi
: "${CKPT_RUN:?regen_go must set CKPT_RUN}"
: "${SAMPLER:?regen_go must set SAMPLER}"
say "GO: CKPT_RUN=$CKPT_RUN SAMPLER=$SAMPLER"

# ── 2. campaign 18 ──────────────────────────────────────────────────────────
# regen_eval_v4.sh already runs generate -> inspection pack -> measure ->
# report -> notebook, and writes latents, which stage 5 needs.
say "STAGE 2 campaign 18 regeneration start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" bash scripts/regen_eval_v4.sh \
    > "$S/q_regen.log" 2>&1
say "STAGE 2 done rc=$? -> $C18"

# ── 3. campaign 19, the screening pass ──────────────────────────────────────
say "STAGE 3 stress DDIM-50 start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" PASS=ddim50 \
    bash scripts/run_stress_geometry.sh > "$S/q_stress50.log" 2>&1
say "STAGE 3 done rc=$? -> $C19"

# ── 4. campaign 20 ──────────────────────────────────────────────────────────
# Production sampler whatever SAMPLER says: all four axes it moves are in the
# CONDITIONING, and an overlap sampler would make a failure at phi 0.20
# unattributable between the request and the assembly.
say "STAGE 4 ood_conditioning start"
mapfile -t OOD_CASES < <(python - <<'PYCASES'
from poregen.eval_v4.cases import build_cases
for c in build_cases("ood_conditioning"):
    print(c.name)
PYCASES
)
say "STAGE 4 ${#OOD_CASES[@]} cases"
for c in "${OOD_CASES[@]}"; do
    python -m poregen.eval_v4.cli generate ood_conditioning \
        --model "$CKPT_RUN" --ckpt "${CKPT:-latest}" --out "$C20" --only "$c" \
        > "$S/q_ood_${c}.log" 2>&1
    say "STAGE 4 GEN $c rc=$?"
done
python -m poregen.eval_v4.cli measure ood_conditioning --root "$C20" \
    > "$S/q_ood_measure.log" 2>&1
say "STAGE 4 MEASURE rc=$?"
python -m poregen.eval_v4.cli report --root "$C20" --assessment ood_conditioning \
    > "$S/q_ood_report.log" 2>&1
say "STAGE 4 done rc=$? -> $C20"

# ── 5. the decoder fine-tune and the D43 gate table ─────────────────────────
# The gate file is the author's go-ahead and this script never creates it.
if [ ! -e "$FT_GO" ]; then
    say "STAGE 5 blocked: waiting for $FT_GO (polling 60 s, no timeout)"
    while [ ! -e "$FT_GO" ]; do sleep 60; done
fi
say "STAGE 5 gate released"
say "STAGE 5 decoder fine-tune start (r08/decoder-ft)"
python scripts/train_vae.py run r08/decoder-ft > "$S/q_decoder_ft.log" 2>&1
ft_rc=$?
say "STAGE 5 fine-tune done rc=$ft_rc"
FT_RUN=$(run_dir_for decoder-ft)
BASE_CKPT="$(run_dir_for reduction-factor-8)best.ckpt"
if [ "$ft_rc" -eq 0 ] && [ -n "$FT_RUN" ] && [ -f "${FT_RUN}best.ckpt" ]; then
    # CAMPAIGN 18, not 12: the gate must be read on the volumes the paper will
    # report, and on the checkpoint and sampler those were made with.
    for steps in 50 200; do
        say "STAGE 5 redecode DDIM-$steps start"
        python scripts/analysis/decoder_ft_redecode.py \
            --baseline "$BASE_CKPT" --finetuned "${FT_RUN}best.ckpt" \
            --latents "$C18/*/volumes/*ddim${steps}*/latents.npy" \
            --out "$FT_CAMP/redecode_ddim${steps}" \
            > "$S/q_redecode_${steps}.log" 2>&1
        say "STAGE 5 redecode DDIM-$steps done rc=$? -> $FT_CAMP/redecode_ddim${steps}"
    done
else
    say "STAGE 5 redecode SKIPPED: fine-tune rc=$ft_rc run=${FT_RUN:-<none>}"
fi

# ── 6. downstream utility, on the ORIGINAL decoder's volumes ────────────────
say "STAGE 6 downstream utility start (campaign-root $C18, original r08 decoder)"
python scripts/analysis/downstream_utility.py --campaign-root "$C18" \
    --out "$C14" > "$S/q_downstream.log" 2>&1
say "STAGE 6 done rc=$? -> $C14"

# ── 7. campaign 19, the DDIM-200 pass ───────────────────────────────────────
say "STAGE 7 stress DDIM-200 start"
CKPT_RUN="$CKPT_RUN" SAMPLER="$SAMPLER" PASS=ddim200 \
    bash scripts/run_stress_geometry.sh > "$S/q_stress200.log" 2>&1
say "STAGE 7 done rc=$? -> $C19"

# ── 8. the two remaining VAE rungs ──────────────────────────────────────────
for exp in "${TAIL_RUNGS[@]}"; do
    say "STAGE 8 r08/$exp start"
    python scripts/train_vae.py run "r08/$exp" > "$S/q_r08_${exp}.log" 2>&1
    say "STAGE 8 r08/$exp done rc=$?"
done

# ── 9. the final notebooks ──────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    say "STAGE 9 notebook into $root start"
    python scripts/analysis/build_eval_v4_notebook.py --root "$root" \
        > "$S/q_notebook_$(basename "$root").log" 2>&1
    say "STAGE 9 notebook into $root done rc=$?"
done

say "FINAL QUEUE COMPLETE"

#!/usr/bin/env bash
# chain v8, RESUMABLE: the same stages as chain_v8_split_v4.sh, able to start
# at any of them.
#
# WHY A SEPARATE FILE. chain_v8_split_v4.sh is RUNNING, and bash reads a script
# by byte offset while it executes it — editing it would corrupt the live run.
# This copy exists so that a stage failing overnight can be relaunched FROM
# THAT STAGE. Relaunching the original from the top would retrain rf-8 — thirty
# hours — and leave a second rf-8 run beside the first.
#
#     START_AT=store POREGEN_SPLIT=split_v4 bash scripts/chain_v8_resume.sh
#
# Stages, in order: rf8 store ldm06 c18 baselines c24 c27 c14 grey.
# A skipped stage's outputs are read back from disk: the VAE and the LDM by
# their -ds<split> run names, so a resume past rf-8 still exports
# POREGEN_VAE_CHECKPOINT and still cannot pick split_v3's r08-run-0004.
#
# Stages are NOT otherwise changed. Keep this file in step with the original
# if the original is ever edited again.
# THE WHOLE PAPER, REBUILT ON data/split_v4.  PREPARED — NOT LAUNCHED.
#
# Same architecture as split_v3 throughout. One thing is new and it is a
# SIMPLIFICATION, not an addition: per-face neighbour dropout runs from step 0
# of the 130k base training (ldm06/facedrop_from_start), so there is no
# fine-tune stage. Campaign 29 established that the dropout and not the
# fine-tuning clears the chunk-plane band — a matched control at drop_nb_face 0
# left the -8 slab at 0.187 against the untouched 0.192 — so the correction can
# move to the start without losing its cause.
#
# LAUNCH ONLY ON THE AUTHOR'S WORD, and only once data/split_v4 exists. The
# chain refuses to start without it.
#
# STOPPING. Never signal it. Touch runs/campaigns/STOP_CHAIN and it halts at
# the next stage boundary with every job intact. A trainer signalled mid-step
# lost 22 steps AND its whole U_f finalisation on 2026-09-27.
#
# ETAs are the split_v3 wall times from docs/PAPER_RUNBOOK.md, which are
# measured on this GB10. They are a guide for planning, not a budget the chain
# enforces: split_v4 has its own patch count and the rungs stop on their own
# early-stopping criterion, not on the clock.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
STOP="$REPO/runs/campaigns/STOP_CHAIN"
SPLIT="${POREGEN_SPLIT:-split_v4}"
export POREGEN_SPLIT="$SPLIT"
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75
#: Campaign directories for this build. NEVER the split_v3 ones: those hold
#: every published number and a rerun must not be able to overwrite them.
SUF="-v4"
cd "$REPO" || exit 1

#: Resume point. Everything before it is skipped; everything from it runs.
START_AT="${START_AT:-rf8}"
case " rf8 store ldm06 c18 baselines c24 c27 c14 grey " in
    *" $START_AT "*) ;;
    *) echo "START_AT must be one of: rf8 store ldm06 c18 baselines c24 c27 c14 grey" >&2; exit 2 ;;
esac
_reached=0
at() {
    [ "$_reached" -eq 1 ] && return 0
    if [ "$1" = "$START_AT" ]; then _reached=1; return 0; fi
    say "SKIP $1 (resuming at $START_AT)"
    return 1
}
#: Campaign directories and the store, defined up front so a later stage can
#: use them when the stage that normally sets them was skipped.
STORE="data/$SPLIT/latents_r08z8"
C18="$REPO/runs/campaigns/18-eval-v4-final$SUF"
link_floor() { mkdir -p "$1"; [ -e "$1/real_floor" ] || ln -s "$C18/real_floor" "$1/real_floor"; }
say() { printf '%s  V8R %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mem_line() {
    local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
    local cuda; cuda=$(python - <<'PY' 2>/dev/null || echo "n/a"
import torch
if torch.cuda.is_available():
    f,t = torch.cuda.mem_get_info(); print(f"{f/2**30:.1f}/{t/2**30:.1f} GiB free")
else: print("no cuda")
PY
)
    printf 'host available %s GB; cuda %s' "$avail" "$cuda"
}
gate() {
    if [ -e "$STOP" ]; then
        say "STOP FILE PRESENT — halting cleanly before: $*"
        say "  every job is intact; remove $STOP and rerun to continue."
        exit 0
    fi
}
step() {   # a stage with no +5min watch: short, or CPU
    local label="$1" tag="$2" eta="$3"; shift 3
    gate "$label"
    say "$label start (ETA ${eta})"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v8_$tag.log" 2>&1
    local rc=$?
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    [ "$rc" -ne 0 ] && { say "STOP NOTICE: $tag failed — see $S/v8_$tag.log"; exit "$rc"; }
    return 0
}
run_watched() {  # a training stage: watched for host pressure at +5 min
    local label="$1" tag="$2" eta="$3"; shift 3
    gate "$label"
    say "$label start (ETA ${eta})"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v8_$tag.log" 2>&1 &
    local pid=$!
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
          say "$tag +5min: $(mem_line)"
          if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
              say "$tag STOPPING: host available ${avail} GB under ${MIN_AVAIL_GB} GB"
              kill -TERM "$pid" 2>/dev/null
          fi
      fi ) &
    local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    [ "$rc" -ne 0 ] && { say "STOP NOTICE: $tag failed — see $S/v8_$tag.log"; exit "$rc"; }
    return 0
}
#: A CHECK, not a stage: it reports and, for all but the fatal gates, carries
#  on. The labels changed between split_v3 and split_v4, so a moved number may
#  be the improvement the rebuild is for. Only a degenerate cell, a kill switch
#  that cannot separate two requested porosities, and a non-finite loss mean the
#  run is not working — those stop the chain.
check() {
    local label="$1" tag="$2" fatal="$3"; shift 3
    gate "check $label"
    say "CHECK $label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v8_check_$tag.log" 2>&1
    local rc=$?
    local verdict; [ "$rc" -eq 0 ] && verdict=PASS || verdict=FAIL
    say "CHECK $label $verdict (rc=$rc) in $((SECONDS-t0))s — see $S/v8_check_$tag.log"
    if [ "$rc" -ne 0 ] && [ "$fatal" = "fatal" ]; then
        say "STOP NOTICE: $label is a fatal gate — the run is not working."
        exit "$rc"
    fi
    return 0
}
tb() {   # TensorBoard for a family, in its own tmux, never on the card
    local sess="$1" logdir="$2" port="$3"
    if tmux has-session -t "$sess" 2>/dev/null; then
        say "TB $sess already up"
    else
        tmux new-session -d -s "$sess" \
            "tensorboard --logdir $logdir --port $port --bind_all" 2>/dev/null \
            && say "TB $sess on :$port ($logdir)" \
            || say "TB $sess could NOT start — monitoring only, not fatal"
    fi
}
smoke() {
    local exp="$1" tag="$2"
    say "SMOKE $exp (cap $SMOKE_FRACTION, $SPLIT)"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION \
        choom -n 1000 -- python scripts/analysis/vae_smoke_step.py --experiment "$exp" \
        > "$S/v8_smoke_$tag.log" 2>&1
    local rc=$?
    say "SMOKE $exp rc=$rc — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/v8_smoke_$tag.log" | head -2 | tr '\n' ' ')"
    return $rc
}

# ── 0. refuse to start on a dataset that is not there ──────────────────────
say "=== v8: the paper on $SPLIT === $(mem_line)"
if [ ! -d "$REPO/data/$SPLIT" ]; then
    say "STOP NOTICE: data/$SPLIT does not exist. Nothing is launched."
    exit 1
fi
# THE SAME FILE SET split_v3 HAS, plus the two the build writes LAST. The
# build's own rc line cannot be trusted — it reports tee, not python — so its
# success is judged by what it left behind: build_report.md is the final stage
# of --stage all, and patches_meta.json the extractor's.
for f in patch_index.parquet splits.json volumes.zarr class_weights.json \
         build_report.md patches_meta.json patches_xct.bin patches_label.bin; do
    if [ ! -e "$REPO/data/$SPLIT/$f" ]; then
        say "STOP NOTICE: data/$SPLIT/$f is missing — the build is incomplete."
        exit 1
    fi
done
say "data/$SPLIT present; POREGEN_SPLIT=$SPLIT exported for every stage"
# The split_v3 numbers every check below is read against, regenerated from the
# split_v3 runs themselves so the comparison cannot drift from what they did.
step "split_v3 gate reference" gates "5 s" \
    env POREGEN_SPLIT=split_v3 python scripts/analysis/split_v3_gate_reference.py

if at rf8; then
    # ── 1. the VAE, exactly rf-8 ───────────────────────────────────────────────
    tb tb-vae "$REPO/runs/vae" 6006
    # rc=1 IS A MEMORY RESULT, rc=2 IS NOT. The smoke script says so itself
    # ("NOT A MEMORY RESULT") and the chain used to read both as "does not fit" —
    # the false negative that sent B to its fallback on 2026-09-24, and that stopped
    # this chain's first launch on a script fault. Each is now reported as what it is.
    smoke r08/reduction-factor-8 rf8; SRC=$?
    if [ "$SRC" -eq 1 ]; then
        say "STOP NOTICE: rf-8 does not fit at its own batch under the cap (a real OOM)."
        exit 1
    elif [ "$SRC" -ne 0 ]; then
        say "STOP NOTICE: the rf-8 smoke FAILED WITHOUT MEASURING MEMORY (rc=$SRC) — a fault"
        say "  in the smoke script or the config, not a memory result. See $S/v8_smoke_rf8.log"
        exit "$SRC"
    fi
    run_watched "r08 rf-8 on $SPLIT" r08_rf8 "30 h" \
        python scripts/train_vae.py run r08/reduction-factor-8
    # The run name carries -ds<split> on any split but the published one
    # (poregen.runtime.runs), so this glob cannot pick r08-run-0004.
    VAE=$(ls -dt "$REPO"/runs/vae/r08-run-*-z8-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
    [ -z "$VAE" ] && { say "STOP NOTICE: no rf-8 run on $SPLIT was found after training."; exit 1; }
    # EVERY LDM BELOW MUST DECODE WITH THIS VAE. train_ldm refuses to start when
    # cfg['vae']['checkpoint'] differs from the store's recorded encoder, and the
    # LDM configs name r08-run-0004 — split_v3's VAE. On split_v3 the bring-up met
    # that by editing and committing ldm06/base.yaml; on the rebuild no config file
    # is edited, so the new VAE is named here once and every resolution follows it.
    export POREGEN_VAE_CHECKPOINT="${VAE%/}/best.ckpt"
    say "POREGEN_VAE_CHECKPOINT=$POREGEN_VAE_CHECKPOINT for every LDM stage"
    say "VAE = $(basename "${VAE%/}")"

    # ── 1b. the r08 acceptance checks, exactly the split_v3 ones ───────────────
    # Read against docs/SPLIT_V3_GATES.md: r08-run-0004 scored val/test pore Dice
    # 0.9175/0.9033, air Dice 0.9964/0.9886, porosity MAE 0.00112/0.00214, dense
    # pore Dice 0.8162 with a 0.1050 gap, and L1 0.0346 with texture +0.359 and
    # sharpness 0.816 on the harness.
    check "r08 rung report" rung_report ok \
        python scripts/analysis/r08_rung_report.py --run "${VAE%/}"
    check "r08 calibration probe (dense panels)" calib ok \
        python scripts/analysis/r08_calibration_probe.py --run "${VAE%/}"
    check "vae_val_l1 harness (L1, texture, sharpness)" val_l1 ok \
        python scripts/analysis/vae_val_l1.py --run "${VAE%/}" \
        --split "$SPLIT" --n-batches 20
    check "recon figure" recon_fig ok \
        python scripts/analysis/vae_recon_figure.py \
        --run "v4 rf-8=${VAE%/}" --split "$SPLIT" \
        --out "$REPO/runs/campaigns/09-r08-latent-sweep/figures/recon_v4_rf8"

fi
[ -z "${VAE:-}" ] && VAE=$(ls -dt "$REPO"/runs/vae/r08-run-*-z8-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
[ -z "$VAE" ] && { say "STOP NOTICE: resuming past rf-8 but no rf-8 run on $SPLIT exists"; exit 1; }
export POREGEN_VAE_CHECKPOINT="${VAE%/}/best.ckpt"
say "VAE = $(basename "${VAE%/}") (POREGEN_VAE_CHECKPOINT set)"
if at store; then
    # ── 2. the latent store and its conditioning ───────────────────────────────
    STORE="data/$SPLIT/latents_r08z8"
    step "latent store" latents "6 h" \
        python scripts/build_latent_dataset.py --checkpoint "${VAE%/}/best.ckpt" \
        --output "$STORE"
    # THE SPLIT_V3 BRING-UP'S OWN CHECKS (scripts/ldm06_bringup.sh), not new ones:
    # the eight store files, then — after the conditioning — its three sidecars and
    # a LatentDataset that serves a real batch. A store that loads is not the same
    # as a store that is correct.
    for f in metadata.json train/latents.bin train/index.parquet train/material.bin \
             train/air.bin train/pore.bin val/latents.bin test/latents.bin; do
        [ -e "$STORE/$f" ] || { say "STOP NOTICE: store missing $f"; exit 1; }
    done
    say "CHECK store files PASS (the eight the split_v3 bring-up required)"
    # The SAMPLED std reference, before anything is scored against it. ldm06 trains
    # with latent_mode sampled, so the target's per-channel std is
    # sqrt(1 + (sigma_rms/per_channel_std)^2) — 1.863 on split_v3, NOT 1.0. Scoring
    # against 1.0 charges the model for the posterior width it was trained on.
    check "latent std reference" latent_std ok \
        python scripts/analysis/latent_std_reference.py \
        --store "data/$SPLIT/latents_r08z8"
    step "conditioning" conditioning "1 h" \
        python scripts/build_conditioning.py --store "$STORE"
    for sp in train val test; do
        [ -e "$STORE/$sp/cond.parquet" ] || { say "STOP NOTICE: conditioning missing $sp/cond.parquet"; exit 1; }
    done
    say "CHECK conditioning files PASS"
    check "LatentDataset serves a real batch (split_v3 bring-up VERIFY)" store_verify fatal \
        python -c "
    import json, pathlib, sys
    from poregen.diffusion.latents import LatentDataset
    store = pathlib.Path('$STORE')
    meta = json.loads((store / 'metadata.json').read_text())
    print('z_channels', meta['latent_shape'][0], 'pore_status',
          meta.get('material', {}).get('pore_status', 'MISSING'))
    ds = LatentDataset(store, split='train')
    assert len(ds) > 0, 'empty dataset'
    b = ds[0]
    print('batch keys', sorted(b), 'n rows', len(ds))
    print('VERIFY OK')
    "

fi
if at ldm06; then
    # ── 3. the LDM, with the dropout from step 0 — no fine-tune stage ──────────
    tb tb-ldm "$REPO/runs/ldm" 6007
    run_watched "ldm06 facedrop_from_start (130k)" ldm06 "16 h" \
        python scripts/train_ldm.py run ldm06/facedrop_from_start
    LDM=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
    [ -z "$LDM" ] && { say "STOP NOTICE: no ldm06 run on $SPLIT was found after training."; exit 1; }
    say "LDM = $(basename "${LDM%/}")"

    # ── 3b. the LDM convergence gates, the same ones run-0001 was read on ──────
    # run-0001 scored por_mae 0.001119 (ema_ddim50), std ratio 1.8006 against the
    # sampled reference 1.863, x0_sat 1.1e-07, degen 0.0, and a kill switch that
    # separated 0.005 from 0.05 by 0.0442. degen and direction_ok are FATAL.
    check "ldm convergence check" ldm_converge fatal \
        python scripts/diag_ldm_samples.py --run "${LDM%/}" --ckpt latest

fi
[ -z "${LDM:-}" ] && LDM=$(ls -dt "$REPO"/runs/ldm/ldm06-run-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
if at c18; then
    # ── 4. the paper's campaigns, BY THE SPLIT_V3 PROCEDURE ────────────────────
    # Campaign 18 is run by scripts/regen_eval_v4.sh — the script that produced the
    # split_v3 campaign 18 — not by a loop written here. Its generation order, its
    # measure pass (including the measure-only field_stats), the capped
    # memorisation pass, the inspection pack, the report and the notebook are then
    # the split_v3 ones by construction.
    #
    # ONE deliberate difference, and it follows from the rebuild: regen symlinks
    # the real floor from split_v3's campaign 12, whose labels are split_v3's. The
    # v4 floor is cut from split_v4 FIRST, so regen finds it present and does not
    # link the old one. Every other campaign then shares that floor by symlink, as
    # the split_v3 campaigns shared campaign 12's.
    C18="$REPO/runs/campaigns/18-eval-v4-final$SUF"
    mkdir -p "$C18"
    step "real floor on $SPLIT (shared by every v4 campaign)" real_floor "10 min" \
        python -m poregen.eval_v4.cli real-floor --root "$C18" \
        --shapes small large micro surface
    link_floor() { mkdir -p "$1"; [ -e "$1/real_floor" ] || ln -s "$C18/real_floor" "$1/real_floor"; }

    step "campaign 18 via regen_eval_v4.sh (production sampler)" c18 "16 h" \
        env CKPT_RUN="${LDM%/}" CKPT=latest SAMPLER=production OUT="$C18" \
        SCRATCH="$S/v8_regen18" bash scripts/regen_eval_v4.sh
    # regen does NOT exit non-zero when a sub-stage fails — it is written so one
    # failing assessment cannot kill the rest — so its own log is read for them.
    check "campaign 18 sub-stages all rc=0" c18_substages ok \
        bash -c "! grep -E 'rc=[1-9]' '$C18/regen.log'"

fi
if at baselines; then
    # ── 5. the three baselines, trained INTO the v4 directories ────────────────
    # train_slicegan.py and train_ddpm3d.py default --out to the split_v3 campaign
    # directories. Without --out they would OVERWRITE the split_v3 baselines.
    C22="$REPO/runs/campaigns/22-slicegan-baseline$SUF"; link_floor "$C22"
    step "slicegan train" slicegan "11 h" \
        python scripts/train_slicegan.py --out "$C22/train"
    [ -e "$C22/train/latest.ckpt" ] || { say "STOP NOTICE: slicegan left no latest.ckpt"; exit 1; }
    step "slicegan sample" slicegan_sample "30 min" \
        python scripts/analysis/slicegan_sample.py --checkpoint "$C22/train/latest.ckpt" --root "$C22"
    step "slicegan measure" slicegan_meas "8 min" \
        python -m poregen.eval_v4.cli measure slicegan --root "$C22"
    step "slicegan report" slicegan_report "1 min" \
        python -m poregen.eval_v4.cli report --root "$C22"

    C23="$REPO/runs/campaigns/23-ddpm3d-baseline$SUF"; link_floor "$C23"
    step "ddpm3d train" ddpm3d "24 h" \
        python scripts/train_ddpm3d.py --data-root "data/$SPLIT" --out "$C23/train"
    [ -e "$C23/train/latest.ckpt" ] || { say "STOP NOTICE: ddpm3d left no latest.ckpt"; exit 1; }
    step "ddpm3d sample" ddpm3d_sample "2 h" \
        python scripts/analysis/ddpm3d_sample.py --checkpoint "$C23/train/latest.ckpt" --root "$C23"
    step "ddpm3d measure" ddpm3d_meas "8 min" \
        python -m poregen.eval_v4.cli measure ddpm3d --root "$C23"
    step "ddpm3d report" ddpm3d_report "1 min" \
        python -m poregen.eval_v4.cli report --root "$C23"

    run_watched "ldm25 phi-only (40k)" ldm25 "19 h" \
        python scripts/train_ldm.py run ldm25/phi_only
    LDM25=$(ls -dt "$REPO"/runs/ldm/ldm25-run-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
    [ -z "$LDM25" ] && { say "STOP NOTICE: no ldm25 run on $SPLIT was found after training."; exit 1; }
    C25="$REPO/runs/campaigns/25-ldm-phi-only$SUF"; link_floor "$C25"
    for A in porosity_global microstructure cfg; do
        step "c25 generate $A" "c25_gen_$A" "varies" \
            python -m poregen.eval_v4.cli generate "$A" --model "${LDM25%/}" \
            --ckpt 40000 --weights ema --out "$C25"
        step "c25 measure $A" "c25_meas_$A" "varies" \
            python -m poregen.eval_v4.cli measure "$A" --root "$C25"
    done
    step "c25 report" c25_report "1 min" \
        python -m poregen.eval_v4.cli report --root "$C25"

fi
[ -z "${LDM25:-}" ] && LDM25=$(ls -dt "$REPO"/runs/ldm/ldm25-run-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
if at c24; then
    # ── 6. the ablation ────────────────────────────────────────────────────────
    C24="$REPO/runs/campaigns/24-ablation$SUF"; link_floor "$C24"
    step "c24 generate" c24_gen "6 h" \
        python -m poregen.eval_v4.cli generate ablation --model "${LDM%/}" \
        --ckpt latest --weights ema --out "$C24"
    step "c24 measure" c24_meas "20 min" \
        python -m poregen.eval_v4.cli measure ablation --root "$C24"
    step "c24 report" c24_report "1 min" \
        python -m poregen.eval_v4.cli report --root "$C24"

fi
if at c27; then
    # ── 7. the budget-matched row ──────────────────────────────────────────────
    # ldm06 at the phi-only baseline's own 40 000 steps. On split_v3 that was the
    # base run, run-0001; on the rebuild the base run IS facedrop_from_start, so its
    # step-40000 checkpoint is the analogue. save_every 10000 keeps that step.
    C27="$REPO/runs/campaigns/27-budget-matched-40k$SUF"; link_floor "$C27"
    for A in porosity_global microstructure sampler; do
        step "c27 generate $A" "c27_gen_$A" "varies" \
            python -m poregen.eval_v4.cli generate "$A" --model "${LDM%/}" \
            --ckpt 40000 --weights ema --out "$C27"
        step "c27 measure $A" "c27_meas_$A" "varies" \
            python -m poregen.eval_v4.cli measure "$A" --root "$C27"
    done
    step "c27 report" c27_report "1 min" \
        python -m poregen.eval_v4.cli report --root "$C27"

fi
if at c14; then
    # ── 8. downstream utility ──────────────────────────────────────────────────
    step "c14 downstream" c14 "10 h" \
        python scripts/analysis/downstream_utility.py \
        --out "$REPO/runs/campaigns/14-downstream-utility$SUF"

fi
if at grey; then
    # ── 9. the single-channel sweep: GREY ONLY, six rungs ──────────────────────
    # Which channel limits compression. Each rung is the r08 rung it is named after
    # with the grey-only model swapped in; the stopping rule is r08's, in steps.
    # Pore and air are NOT queued: they need a model and a loss that do not exist,
    # and which reading to build is the author's decision. When it comes they join
    # here, after grey, as their own stage.
    # The rung report and the calibration probe are 3-class instruments — pore and
    # air Dice — and have nothing to measure on a grey-only model, so the check is
    # the harness: L1, texture and sharpness, the columns that refereed campaign 28.
    for pair in "2:32" "4:16" "8:8" "16:4" "32:2" "64:1"; do
        RF=${pair%%:*}; Z=${pair#*:}
        smoke "r08_grey/reduction-factor-$RF" "grey_rf$RF"; SRC=$?
        if [ "$SRC" -eq 1 ]; then
            say "grey rf-$RF SKIPPED: does not fit at its own batch under the cap (a real OOM)."
            continue
        elif [ "$SRC" -ne 0 ]; then
            say "grey rf-$RF SKIPPED: its smoke FAILED WITHOUT MEASURING MEMORY (rc=$SRC) — a fault,"
            say "  not a memory result. See $S/v8_smoke_grey_rf$RF.log"
            continue
        fi
        run_watched "grey-only VAE rf-$RF (z=$Z)" "grey_rf$RF" "15 h" \
            python scripts/train_vae.py run "r08_grey/reduction-factor-$RF"
        GREY=$(ls -dt "$REPO"/runs/vae/r08_grey-run-*-z$Z-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
        if [ -z "$GREY" ]; then
            say "grey rf-$RF: no run directory found after training — check not run"
            continue
        fi
        check "grey rf-$RF harness (L1, texture, sharpness)" "grey_rf${RF}_l1" ok \
            python scripts/analysis/vae_val_l1.py --run "${GREY%/}" \
            --split "$SPLIT" --n-batches 20
    done

fi
say "BLOCK COMPLETE — the paper is rebuilt on $SPLIT — $(mem_line)"
say "TOTAL ETA from the split_v3 wall times: about 130 h for the paper, then about 90 h for the six grey rungs."

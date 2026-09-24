#!/usr/bin/env bash
# The A0 trial, then B. Written fresh; chain_vrrae2.sh is finished and is not
# edited.
#
# ORDER, and why:
#   1  A0 TRIAL, 400 steps. A's own 476 steps already served as A's trial and
#      it never learned, so rerunning A tells us nothing. What is untested is
#      whether A's ARCHITECTURE can learn at all with the KL off — A collapsed
#      to 14 active dimensions inside 100 steps while beta was still 0.0013,
#      so KL pressure is not established as the cause and V0's beta-0 result
#      does not automatically transfer.
#   2  B: smoke 1536 under the cap (expected to raise CLEANLY now, not to take
#      the host down), then the written fallback at 1024/rank 768. B is the arm
#      that isolates the FC and its SVD matrix is 1024x4096 — RECTANGULAR, like
#      vrrae03's, which trained — so it is the likeliest of the three to work.
#   3  A0 in full ONLY if the trial learned. The gate is mechanical and is
#      applied here rather than by hand, so the card does not idle waiting for
#      a human to read a number.
#
# Guards, all three now: per-process CUDA cap, logvar clamp, and non-finite
# gradients skipped with an abort after 20 in a row (1bc51b8).
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"; C18="$REPO/runs/campaigns/18-eval-v4-final"
C28="$REPO/runs/campaigns/28-vrrae-family"
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75
#: xct_loss at step 400 below this, and falling, means the architecture learns.
TRIAL_PASS_XCT=0.12
cd "$REPO" || exit 1
say() { printf '%s  V3 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
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
run_watched() {
    local label="$1" tag="$2"; shift 2
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/$tag.log" 2>&1 &
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
    local dt=$((SECONDS - t0))
    if [ "$rc" -ne 0 ]; then
        say "$tag rc=$rc FAILED after ${dt}s — $(mem_line) — see $S/$tag.log"
        say "STOP NOTICE: the block halts here on the first failure inside a run."
        exit "$rc"
    fi
    say "$tag rc=0 in ${dt}s — $(mem_line)"
    return 0
}
smoke() {
    local exp="$1" tag="$2"
    say "SMOKE $exp (cap $SMOKE_FRACTION)"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION \
        choom -n 1000 -- python scripts/analysis/vae_smoke_step.py --experiment "$exp" \
        > "$S/smoke_$tag.log" 2>&1
    local rc=$?
    say "SMOKE $exp rc=$rc — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/smoke_$tag.log" | head -2 | tr '\n' ' ')"
    return $rc
}
table() {
    choom -n 1000 -- python scripts/analysis/vrrae_family_table.py --out "$C28" \
        > "$S/tbl_$1.log" 2>&1
    say "campaign-28 table after $1 rc=$? — $(grep -oE '\*\*[0-9.]+%\*\*' "$C28/family_table.md" 2>/dev/null | tr '\n' ' ')"
}

say "=== A0 trial, then B === $(mem_line)"

# ── 1. the A0 trial ────────────────────────────────────────────────────────
run_watched "A0 TRIAL (vrrae/a0_trial, 400 steps, beta 0)" vrrae_a0_trial \
    python scripts/train_vae.py run vrrae/a0_trial

# Read the trial out of its own log rather than by eye.
TRIAL_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*a0_trial*/ "$REPO"/runs/vae/vrrae-run-*/ 2>/dev/null | head -1)
TRIAL_VERDICT=$(python - "$TRIAL_RUN" "$TRIAL_PASS_XCT" <<'PY'
import json, sys, pathlib
run, thresh = pathlib.Path(sys.argv[1]), float(sys.argv[2])
rows = []
p = run / "log.jsonl"
if p.exists():
    for l in p.read_text().splitlines():
        l = l.strip()
        if l:
            try: rows.append(json.loads(l))
            except Exception: pass
tr = [r for r in rows if r.get("split") == "train" and r.get("xct_loss") is not None]
at = {}
for r in tr:
    s = r.get("step", -1)
    if s in (100, 200, 300, 399, 400):
        at[s] = r
last = tr[-1] if tr else None
line = " ".join(f"s{s}={at[s]['xct_loss']:.4f}" for s in sorted(at))
act = " ".join(f"s{s}:mu{at[s].get('mu_n_active')}" for s in sorted(at))
final = (at.get(400) or at.get(399) or last)
val = final["xct_loss"] if final else 9.9
reached = last["step"] if last else -1
print(f"{'PASS' if val < thresh else 'FAIL'}|{val:.4f}|{reached}|{line}|{act}")
PY
) || TRIAL_VERDICT="FAIL|na|na||"
say "A0 TRIAL RESULT: $TRIAL_VERDICT"

# ── 2. B, smoked then run ──────────────────────────────────────────────────
if smoke vrrae/b b; then
    run_watched "B (vrrae/b, batch 1536)" vrrae_b python scripts/train_vae.py run vrrae/b
    table b
elif smoke vrrae/b_fallback b_fb; then
    say "B at 1536 does not fit under the cap; the written fallback does"
    run_watched "B fallback (batch 1024, rank 768)" vrrae_b_fb \
        python scripts/train_vae.py run vrrae/b_fallback
    table b
else
    say "B SKIPPED: neither 1536 nor the written fallback fits. No third "
    say "  configuration is invented here."
fi

# ── 3. A0 in full, only if the trial learned ───────────────────────────────
case "$TRIAL_VERDICT" in
    PASS*)
        say "trial PASSED — running A0 in full"
        run_watched "A0 (vrrae/a0, beta 0)" vrrae_a0 python scripts/train_vae.py run vrrae/a0
        table a0
        ;;
    *)
        say "trial FAILED — A0 and A are NOT launched. The architecture question"
        say "  goes to the author; the card moves to rf-64."
        ;;
esac

# ── 4. rf-64 and the notebooks ─────────────────────────────────────────────
RF64=$(ls -dt "$REPO"/runs/vae/r08-run-0012-*/ 2>/dev/null | head -1)
if [ -n "$RF64" ]; then
    run_watched "rf-64 resume from step 1000" rf64_resume \
        python scripts/train_vae.py resume "$(basename "${RF64%/}")" latest.ckpt
fi
for root in "$C18" "$C12"; do
    run_watched "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "BLOCK COMPLETE — $(mem_line)"

#!/usr/bin/env bash
# Campaign 19 - the stress geometries.  EXPLORATORY: nothing here is gated.
#
# Two passes, run at different times, because they cost very different things:
#
#   PASS=ddim50    all nine requests at DDIM-50.  The screening pass: it says
#                  which requests the conditioning honours at all.
#   PASS=ddim200   the same nine at DDIM-200, for the requests worth the cost.
#                  Four times the sampling work of the first pass.
#
# NO --save-latents.  The decoder fine-tune gate compares two decoders on
# identical latents and it reads campaign 18, not this one; a 1024-cubed latent
# canvas is 2 GB per case and nothing here would ever read them back.
#
# Usage:
#   CKPT_RUN=runs/ldm/ldm06-run-0002-... PASS=ddim50 bash scripts/run_stress_geometry.sh
set -uo pipefail          # NOT -e: a failing request must not kill the rest

REPO=/home/jorgecabrejas/Dev/GenAI
OUT="${OUT:-$REPO/runs/campaigns/19-stress-geometry}"
SRC="$REPO/runs/campaigns/12-eval-v4"
SCRATCH="${SCRATCH:-/tmp/run_stress_geometry}"
CKPT="${CKPT:-latest}"
SAMPLER="${SAMPLER:-production}"
PASS="${PASS:-ddim50}"
: "${CKPT_RUN:?set CKPT_RUN to the ldm run directory to generate from}"

#: The a2s sampler, as the campaign-17 trial ran it.
A2S_OVERLAP=64
A2S_PINNED=32
A2S_WRITE=successor

#: Requests that are NOT run at DDIM-200 under the a2s sampler.  a2s costs
#: 2.05x, DDIM-200 costs 4x, and cube1024 is 1024 cubed on all three axes: the
#: product is most of a day of GPU for the one request that asks nothing the
#: other eight do not.  It runs at DDIM-50 like everything else, and campaign 18
#: already carries 1024-cubed cases under both samplers.
A2S_DDIM200_SKIP=(cube1024_ddim200)

case "$SAMPLER" in
    production) OVERRIDE=() ;;
    a2s)        OVERRIDE=(--chunk-overlap "$A2S_OVERLAP"
                          --chunk-overlap-pinned "$A2S_PINNED"
                          --chunk-overlap-write "$A2S_WRITE") ;;
    *) echo "SAMPLER must be 'production' or 'a2s', got '$SAMPLER'" >&2; exit 2 ;;
esac

case "$PASS" in
    ddim50|ddim200) ;;
    *) echo "PASS must be 'ddim50' or 'ddim200', got '$PASS'" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$SCRATCH"; cd "$REPO" || exit 1
LOG="$OUT/run.log"
say() { printf '%s  STRESS %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

# The case list comes from the case builder, never from a list typed here: a
# hand-written list stops matching the moment a request is added.
mapfile -t CASES < <(python - "$PASS" <<'PY'
import sys
from poregen.eval_v4.cases import build_cases
want = sys.argv[1]
for c in build_cases("stress_geometry"):
    if c.name.endswith("_" + want):
        print(c.name)
PY
)
if [ "$SAMPLER" = a2s ] && [ "$PASS" = ddim200 ]; then
    for skip in "${A2S_DDIM200_SKIP[@]}"; do
        CASES=("${CASES[@]/$skip}")
    done
    # Drop the empty slots the substitution above leaves behind.
    KEPT=(); for c in "${CASES[@]}"; do [ -n "$c" ] && KEPT+=("$c"); done
    CASES=("${KEPT[@]}")
    say "a2s at DDIM-200 skips: ${A2S_DDIM200_SKIP[*]}"
fi

say "checkpoint run : $CKPT_RUN ($CKPT)"
say "sampler        : $SAMPLER ${OVERRIDE[*]:-(no override)}"
say "pass           : $PASS — ${#CASES[@]} cases: ${CASES[*]}"
say "output         : $OUT"

# ── the real floor ──────────────────────────────────────────────────────────
# Same panels and same detector as campaign 12, so it is linked rather than
# recut; the link is what says the two campaigns share one floor.  Nothing in
# this campaign is gated against it — there is no real tube — but the report
# header reads it, and a findings file that says "floor: not measured" invites
# the reader to supply a target of their own.
if [ -d "$SRC/real_floor" ] && [ ! -e "$OUT/real_floor" ]; then
    ln -s "$SRC/real_floor" "$OUT/real_floor"
    say "real_floor symlinked from campaign 12"
fi

# ── generate, one request at a time ─────────────────────────────────────────
# Per case, not one call for the whole pass: a request the sampler cannot hold
# in memory then costs that request and not the eight after it.
for c in "${CASES[@]}"; do
    say "GEN $c start"
    python -m poregen.eval_v4.cli generate stress_geometry \
        --model "$CKPT_RUN" --ckpt "$CKPT" --out "$OUT" --only "$c" \
        "${OVERRIDE[@]}" > "$SCRATCH/gen_${c}.log" 2>&1
    say "GEN $c done rc=$?"
done

# ── measure and report ──────────────────────────────────────────────────────
# Both run over every case the campaign holds, so after the second pass the
# findings carry DDIM-50 and DDIM-200 side by side without re-measuring the
# first pass by hand.
say "MEASURE start"
python -m poregen.eval_v4.cli measure stress_geometry --root "$OUT" \
    > "$SCRATCH/measure.log" 2>&1
say "MEASURE done rc=$?"
say "REPORT start"
python -m poregen.eval_v4.cli report --root "$OUT" --assessment stress_geometry \
    > "$SCRATCH/report.log" 2>&1
say "REPORT done rc=$? -> $OUT/stress_geometry/findings.md"

say "STRESS $PASS COMPLETE — $OUT"

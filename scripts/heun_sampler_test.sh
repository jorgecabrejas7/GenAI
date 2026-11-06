#!/usr/bin/env bash
# Campaign 38 (author 2026-11-06: "Build heun-a2s and run it before the campaign 18 [regeneration]; if it's better
# in some way, we run both heun and a2s, if not, only a2s"). The Heun (second-order) sampler on the a2s chunk
# assembly, at 10 / 18 / 25 ladder steps (about 18 / 34 / 48 model calls), against a DDIM-50 control generated with
# the same code. Four roots under runs/campaigns/38-heun-sampler-v4/: ddim50, heun10, heun18, heun25. Each holds
# the campaign-18 "sampler" cases at DDIM-50 only (192^3 and 192x1024x1024, 3 seeds; the case NAMES keep "ddim50",
# the manifest records the true solver and step count) and the 9 "microstructure" cases (192^3, phi 0.01/0.03/0.06),
# measured against the campaign-18 real floor. Read: porosity error, seams, S2/PSD/Ripley/FID ratios, wall time.
#
#     bash scripts/heun_sampler_test.sh        (GPU idle; ~3-4 h)
cd "$(dirname "$0")/.."
export POREGEN_SPLIT=split_v4
R=runs/campaigns/38-heun-sampler-v4
LDM=$(ls -d runs/ldm/ldm06-run-0005-*)
FLOOR=$(readlink -f runs/campaigns/18-eval-v4-final-v4-run0005/real_floor)
SIX="192_ddim50_seed101 192_ddim50_seed202 192_ddim50_seed303 1024_ddim50_seed101 1024_ddim50_seed202 1024_ddim50_seed303"
say() { printf '%s  C38 %s\n' "$(date -Is)" "$*" | tee -a runs/campaigns/chain.log; }
run_root() {
    root=$R/$1; shift
    mkdir -p "$root"; [ -e "$root/real_floor" ] || ln -s "$FLOOR" "$root/real_floor"
    say "$(basename "$root") generate sampler start"
    python -m poregen.eval_v4.cli generate sampler --model "$LDM" --ckpt latest --weights ema --out "$root" --only $SIX "$@" > "$root/gen_sampler.log" 2>&1
    say "$(basename "$root") generate sampler rc=$?"
    python -m poregen.eval_v4.cli measure sampler --root "$root" > "$root/meas_sampler.log" 2>&1
    say "$(basename "$root") measure sampler rc=$?"
    python -m poregen.eval_v4.cli generate microstructure --model "$LDM" --ckpt latest --weights ema --out "$root" "$@" > "$root/gen_micro.log" 2>&1
    say "$(basename "$root") generate microstructure rc=$?"
    python -m poregen.eval_v4.cli measure microstructure --root "$root" > "$root/meas_micro.log" 2>&1
    say "$(basename "$root") measure microstructure rc=$?"
    python -m poregen.eval_v4.cli report --root "$root" > "$root/report.log" 2>&1
}
run_root ddim50 --ddim-steps 50
for N in 10 18 25; do run_root heun$N --solver heun --ddim-steps $N; done
say "HEUN TEST COMPLETE"

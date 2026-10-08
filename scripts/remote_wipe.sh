#!/usr/bin/env bash
# Remove everything this project put on a BORROWED GB10, after its results are
# mirrored here. Run only on the author's word.
#
#   bash scripts/remote_wipe.sh <host>
#
# Refuses unless runs/remote/<host>/MIRROR_OK (written by
# `pull_remote_results.sh <host> --full`) exists AND is newer than the remote's
# newest write under runs/, and no training or chain process runs there.
# Removes ~/Dev/GenAI (code, data, runs), ~/miniforge3/envs/poregen and
# ~/Miniforge3.sh, then prints what is left in the remote home and on its disk.
set -euo pipefail
HOST="${1:?usage: remote_wipe.sh <host>}"
RREPO="${RREPO:-Dev/GenAI}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OK="$REPO/runs/remote/$HOST/MIRROR_OK"
[ -f "$OK" ] || { echo "REFUSED: no $OK — run pull_remote_results.sh $HOST --full first"; exit 1; }
mirrored=$(date -d "$(cut -d' ' -f1 "$OK")" +%s)
remote_last=$(ssh "$HOST" "cd $RREPO 2>/dev/null && find runs -type f -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1" || echo 0)
if [ "${remote_last:-0}" -gt "$mirrored" ]; then
    echo "REFUSED: the remote wrote under runs/ at $(date -d @"$remote_last" -Is), after the mirror ($(cut -d' ' -f1 "$OK")). Pull --full again."
    exit 1
fi
busy=$(ssh "$HOST" "pgrep -af 'train_vae|train_ldm|chain_rungs_remote|vae_smoke_step|r08_rung_report|r08_calibration_probe|vae_val_l1|vae_recon_figure' || true")
[ -z "$busy" ] || { echo "REFUSED: still running on $HOST:"; echo "$busy"; exit 1; }
echo "mirror: $(cat "$OK")"
echo "removing on $HOST: ~/$RREPO ~/miniforge3/envs/poregen ~/Miniforge3.sh"
ssh "$HOST" "rm -rf ~/$RREPO ~/miniforge3/envs/poregen ~/Miniforge3.sh; echo '--- left in home:'; ls -la ~; echo '--- disk:'; df -h ~ | tail -1"

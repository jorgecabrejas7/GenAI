#!/usr/bin/env bash
# connect_jupyter.sh — one-command SSH tunnel for Jupyter + TensorBoard
#
# Usage:
#   ./scripts/connect_jupyter.sh <SLURM_JOB_ID> [--tb-port PORT]
#
# What it does:
#   1. SSHs to LOGIN_NODE, tails the Jupyter SLURM log to extract Node/Port/Token
#   2. Opens SSH tunnels for both Jupyter and TensorBoard (Jupyter port + 1)
#   3. Prints ready-to-use VS Code and TensorBoard URLs
#
# ── Configure these for your cluster ──────────────────────────────────────
LOGIN_NODE="<your-hpc-login-node>"                # e.g. user@hpc.university.ac.uk
HPC_LOG_PATH="\$HOME/Dev/GenAI/jupyter/output/Jupyter.log"  # path on HPC
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" ]]; then
    echo "Usage: $0 <SLURM_JOB_ID> [--tb-port PORT]" >&2
    exit 1
fi

TB_PORT_OVERRIDE=""
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tb-port) TB_PORT_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Fetching Jupyter info for job ${JOB_ID} from ${LOGIN_NODE}..."

# ── Extract Node, Port, Token from the HPC log ────────────────────────────
LOG_CONTENT=$(ssh "${LOGIN_NODE}" "tail -40 ${HPC_LOG_PATH}" 2>/dev/null)

NODE=$(echo "$LOG_CONTENT"  | grep -m1 "^Node"  | awk '{print $NF}')
PORT=$(echo "$LOG_CONTENT"  | grep -m1 "^Port"  | awk '{print $NF}')
TOKEN=$(echo "$LOG_CONTENT" | grep -m1 "^Token" | awk '{print $NF}')

if [[ -z "$NODE" || -z "$PORT" || -z "$TOKEN" ]]; then
    echo "ERROR: Could not parse Node/Port/Token from log." >&2
    echo "Log tail:" >&2
    echo "$LOG_CONTENT" >&2
    echo "" >&2
    echo "Hint: Wait a few seconds for the Jupyter server to start, then retry." >&2
    exit 1
fi

# ── TensorBoard port ──────────────────────────────────────────────────────
TB_PORT="${TB_PORT_OVERRIDE:-$(( PORT + 1 ))}"

echo "  Node  : ${NODE}"
echo "  Port  : ${PORT}  (Jupyter)"
echo "  Port  : ${TB_PORT}  (TensorBoard)"
echo ""

# ── Kill any existing local processes holding those ports ─────────────────
for P in "$PORT" "$TB_PORT"; do
    PIDS=$(lsof -ti:"$P" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        echo "Freeing local port ${P} (PID ${PIDS})..."
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
done

# ── Start SSH tunnels in background ───────────────────────────────────────
ssh -f -N -L "${PORT}:${NODE}:${PORT}"          "${LOGIN_NODE}"
ssh -f -N -L "${TB_PORT}:127.0.0.1:${TB_PORT}" "${LOGIN_NODE}"

echo "Tunnels established. Connect with:"
echo ""
echo "  VS Code Jupyter Server URL:"
echo "    http://127.0.0.1:${PORT}/lab?token=${TOKEN}"
echo ""
echo "  TensorBoard:"
echo "    http://127.0.0.1:${TB_PORT}"
echo ""
echo "To stop the tunnels:"
echo "  lsof -ti:${PORT},${TB_PORT} | xargs kill"

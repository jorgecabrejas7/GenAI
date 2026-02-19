#!/bin/bash
#SBATCH --job-name=JupyterHexcel
#SBATCH --output=jupyter/output/Jupyter.log
#SBATCH --mem=128G
#SBATCH --partition=dual-26-cpu-dual-a6000
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

# Be strict, but not stupid
set -eo pipefail

# ------------------ Environment ------------------
module load Miniforge3/24.7.1-2-base
eval "$(conda shell.bash hook)"
conda activate ml

python -m pip install -U ipykernel
python -m ipykernel install --user --name ml --display-name "Python (ml)"


# Conda activation scripts are NOT nounset-safe
set +u
conda activate ml
set -u

# Keep this minimal and consistent
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# ------------------ Jupyter config ------------------
NODE="$(hostname -s)"
PORT=$(( 8000 + (${SLURM_JOB_ID} % 1000) ))
TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"

echo "================= JUPYTER (SLURM) ================="
echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : ${NODE}"
echo "Port  : ${PORT}"
echo "Token : ${TOKEN}"
echo
echo "On YOUR LAPTOP, run:"
echo "ssh -N -L ${PORT}:${NODE}:${PORT} <user>@<login-node>"
echo
echo "Then open in VS Code (Existing Jupyter Server) or browser:"
echo "http://127.0.0.1:${PORT}/lab?token=${TOKEN}"
echo "===================================================="
echo

# ------------------ Launch ------------------
python -m jupyterlab --no-browser --ip=127.0.0.1 --port="${PORT}" \
  --IdentityProvider.token="${TOKEN}" \
  --notebook-dir="$HOME"
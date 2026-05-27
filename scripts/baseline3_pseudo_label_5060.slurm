#!/bin/bash
#SBATCH --job-name=baseline3-pseudo-5060
#SBATCH --account=cil_jobs
#SBATCH --time=08:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --output=/home/mdealvaro/monocular-depth-estimation/logs/slurm/slurm-%x-%j.out
#SBATCH --error=/home/mdealvaro/monocular-depth-estimation/logs/slurm/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="/home/mdealvaro/monocular-depth-estimation"
VENV_DIR="${REPO_DIR}/.venv"

# Default to a unique cache directory per job; override by exporting CACHE_DIR.
CACHE_DIR="${CACHE_DIR:-/work/scratch/mdealvaro/outputs/baseline3/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1}"

cd "${REPO_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-/work/scratch/mdealvaro/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/work/scratch/mdealvaro/cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${REPO_DIR}/logs/slurm"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Pseudo-label cache directory: ${CACHE_DIR}"

"${PYTHON}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
PY

"${PYTHON}" src/bs3_pseudo_label_DA3.py \
  --pseudo-label \
  --cache "${CACHE_DIR}"

echo "Pseudo-label generation finished at $(date)"
echo "Pseudo-labels written to ${CACHE_DIR}"

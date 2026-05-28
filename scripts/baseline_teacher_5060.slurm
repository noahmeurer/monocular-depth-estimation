#!/bin/bash
#SBATCH --job-name=baseline-teacher-5060
#SBATCH --account=cil_jobs
#SBATCH --time=01:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --output=/home/nmeurer/monocular-depth-estimation/logs/slurm/slurm-%x-%j.out
#SBATCH --error=/home/nmeurer/monocular-depth-estimation/logs/slurm/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="/home/nmeurer/monocular-depth-estimation"
VENV_DIR="${REPO_DIR}/.venv"
OUTPUT_DIR="/work/scratch/nmeurer/outputs/baseline_teacher"
PRED_DIR="${OUTPUT_DIR}/preds"
SUBMISSION_CSV="${OUTPUT_DIR}/baseline_teacher_submission.csv"

cd "${REPO_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-/work/scratch/nmeurer/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/work/scratch/nmeurer/cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${REPO_DIR}/logs/slurm"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Writing outputs to ${OUTPUT_DIR}"

"${PYTHON}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
PY

"${PYTHON}" src/bs_teacher_zero_shot_DA3.py

"${PYTHON}" utils/create_submission.py \
  --pred-dir "${PRED_DIR}" \
  --out-csv "${SUBMISSION_CSV}"

echo "Submission written to ${SUBMISSION_CSV}"
echo "Job finished at $(date)"

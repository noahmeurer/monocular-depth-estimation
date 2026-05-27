#!/bin/bash
#SBATCH --job-name=baseline-teacher-da3mono
#SBATCH --account=cil_jobs
#SBATCH --time=00:30:00
#SBATCH --gpus=gb10:1
#SBATCH --output=/home/mdealvaro/monocular-depth-estimation/logs/slurm/slurm-%x-%j.out
#SBATCH --error=/home/mdealvaro/monocular-depth-estimation/logs/slurm/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="/home/mdealvaro/monocular-depth-estimation"
VENV_DIR="${REPO_DIR}/.venv-gb10"
OUTPUT_DIR="/work/scratch/mdealvaro/outputs/baseline_teacher"
PRED_DIR="${OUTPUT_DIR}/preds"
SUBMISSION_CSV="${OUTPUT_DIR}/baseline_teacher_submission.csv"

cd "${REPO_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-/work/scratch/mdealvaro/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/work/scratch/mdealvaro/cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Writing outputs to ${OUTPUT_DIR}"

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# drop-caches uses sudo and fails in batch (no TTY); harmless to skip
if /usr/bin/drop-caches 2>/dev/null; then
  echo "Dropped filesystem cache"
else
  echo "Warning: drop-caches unavailable in batch; continuing"
fi

"${PYTHON}" src/bs_teacher_zero_shot_DA3.py

"${PYTHON}" utils/create_submission.py \
  --pred-dir "${PRED_DIR}" \
  --out-csv "${SUBMISSION_CSV}"

echo "Submission written to ${SUBMISSION_CSV}"
echo "Job finished at $(date)"

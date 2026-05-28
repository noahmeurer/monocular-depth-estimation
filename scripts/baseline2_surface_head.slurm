#!/bin/bash
#SBATCH --job-name=baseline2-surface-head
#SBATCH --account=cil_jobs
#SBATCH --time=14:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=24G
#SBATCH --output=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.out
#SBATCH --error=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.err

set -euo pipefail

USER_NAME="${USER:-nmeurer}"
REPO_DIR="${REPO_DIR:-/home/${USER_NAME}/monocular-depth-estimation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/${USER_NAME}}"
RUN_NAME="baseline2-surface_head"
OUTPUT_DIR="${SCRATCH_ROOT}/outputs/baseline2/${RUN_NAME}"
PRED_DIR="${OUTPUT_DIR}/preds"
SUBMISSION_CSV="${OUTPUT_DIR}/baseline2_surface_head_submission.csv"

cd "${REPO_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SCRATCH_ROOT
export BS2_MODE=surface_head
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${REPO_DIR}/.venv/bin/activate"
PYTHON="${REPO_DIR}/.venv/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Writing outputs to ${OUTPUT_DIR}"

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${PYTHON}" -u src/bs2_finetune_DA3.py

"${PYTHON}" utils/create_submission.py \
  --pred-dir "${PRED_DIR}" \
  --out-csv "${SUBMISSION_CSV}"

echo "Submission written to ${SUBMISSION_CSV}"
echo "Job finished at $(date)"

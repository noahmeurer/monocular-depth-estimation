#!/bin/bash
#SBATCH --job-name=baseline3-da3-features
#SBATCH --account=cil_jobs
#SBATCH --time=04:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=24G
#SBATCH --output=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.out
#SBATCH --error=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.err

set -euo pipefail

USER_NAME="${USER:-nmeurer}"
REPO_DIR="${REPO_DIR:-/home/${USER_NAME}/monocular-depth-estimation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/${USER_NAME}}"
STUDENT_MODEL="${STUDENT_MODEL:-DA3MONO-LARGE}"
VAL_FEATURE_BATCH="${VAL_FEATURE_BATCH:-4}"
FEATURE_DIR="${FEATURE_DIR:-${SCRATCH_ROOT}/outputs/baseline3/feature_cache/da3_backbone_${STUDENT_MODEL}}"

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
export STUDENT_MODEL
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${REPO_DIR}/.venv/bin/activate"
PYTHON="${REPO_DIR}/.venv/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Writing feature cache to ${FEATURE_DIR}"
echo "Student/backbone model: ${STUDENT_MODEL}"
echo "Feature batch size: ${VAL_FEATURE_BATCH}"

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${PYTHON}" -u src/bs3_extract_da3_features.py \
  --model "${STUDENT_MODEL}" \
  --output-dir "${FEATURE_DIR}" \
  --batch-size "${VAL_FEATURE_BATCH}"

echo "Feature cache written to ${FEATURE_DIR}"
echo "Job finished at $(date)"

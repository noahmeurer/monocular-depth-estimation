#!/bin/bash
#SBATCH --job-name=bs4-novel-views
#SBATCH --account=cil_jobs
#SBATCH --time=06:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --output=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.out
#SBATCH --error=/home/%u/monocular-depth-estimation/logs/slurm-%x-%j.err

set -euo pipefail

USER_NAME="${USER:-nmeurer}"
REPO_DIR="${REPO_DIR:-/home/${USER_NAME}/monocular-depth-estimation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/${USER_NAME}}"

TEACHER_MODEL="${TEACHER_MODEL:-DA3-GIANT-1.1}"
TOP_K="${TOP_K:-250}"
MAX_IMAGES="${MAX_IMAGES:-0}"
PREVIEW_LIMIT="${PREVIEW_LIMIT:-12}"
DELTA_FRACTION="${DELTA_FRACTION:-0.05}"
KEEP_PERCENT="${KEEP_PERCENT:-50}"

TOPK_MANIFEST="${TOPK_MANIFEST:-/work/scratch/nmeurer/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv}"
PSEUDO_DIR="${PSEUDO_DIR:-/work/scratch/nmeurer/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1}"
RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/outputs/baseline4/novel_views_${TEACHER_MODEL}_cosine_top${TOP_K}_${RUN_STAMP}}"

cd "${REPO_DIR}"

if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SCRATCH_ROOT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${REPO_DIR}/.venv/bin/activate"
PYTHON="${REPO_DIR}/.venv/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Teacher model: ${TEACHER_MODEL}"
echo "Top-k manifest: ${TOPK_MANIFEST}"
echo "Pseudo-label dir: ${PSEUDO_DIR}"
echo "Top-k: ${TOP_K}"
echo "Max images: ${MAX_IMAGES}"
echo "Delta fraction: ${DELTA_FRACTION}"
echo "Keep percent: ${KEEP_PERCENT}"

if [ ! -r "${TOPK_MANIFEST}" ]; then
    echo "ERROR: top-k manifest is not readable: ${TOPK_MANIFEST}" >&2
    exit 2
fi
if [ ! -d "${PSEUDO_DIR}" ]; then
    echo "ERROR: pseudo-label directory does not exist: ${PSEUDO_DIR}" >&2
    exit 2
fi

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${PYTHON}" -u src/bs4_generate_novel_views.py \
  --manifest "${TOPK_MANIFEST}" \
  --pseudo-dir "${PSEUDO_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --teacher-model "${TEACHER_MODEL}" \
  --top-k "${TOP_K}" \
  --max-images "${MAX_IMAGES}" \
  --delta-fraction "${DELTA_FRACTION}" \
  --keep-percent "${KEEP_PERCENT}" \
  --preview-limit "${PREVIEW_LIMIT}"

echo "Novel-view dataset written to ${OUTPUT_DIR}"
echo "Job finished at $(date)"

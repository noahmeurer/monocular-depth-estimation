#!/bin/bash
#SBATCH --job-name=bs4-ablationA-top250
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
TEACHER_MODEL="${TEACHER_MODEL:-DA3-GIANT-1.1}"
STUDENT_MODEL="${STUDENT_MODEL:-DA3MONO-LARGE}"
BS3_MODE="${BS3_MODE:-full_head}"

CACHE_DIR="${CACHE_DIR:-/work/scratch/nmeurer/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1}"
RESUME_CKPT="${RESUME_CKPT:-/work/scratch/cdeubel/models/baseline3-DA3MONO-LARGE-full_head-2026-05-29_00-09-52/checkpoints/best.pth}"
TOPK_MANIFEST="${TOPK_MANIFEST:-/work/scratch/nmeurer/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv}"
TOP_K="${TOP_K:-250}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-3}"
ADAPT_BATCH="${ADAPT_BATCH:-8}"
ADAPT_LR="${ADAPT_LR:-1e-8}"
ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY:-1e-4}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/models/baseline4-ablationA-top${TOP_K}-adapt-${TEACHER_MODEL}-${BS3_MODE}-${RUN_STAMP}}"

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
export TEACHER_MODEL
export STUDENT_MODEL
export BS3_MODE
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${REPO_DIR}/.venv/bin/activate"
PYTHON="${REPO_DIR}/.venv/bin/python"

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Resume checkpoint: ${RESUME_CKPT}"
echo "Pseudo-label cache: ${CACHE_DIR}"
echo "Top-k manifest: ${TOPK_MANIFEST}"
echo "Top-k: ${TOP_K}"
echo "Epochs: ${ADAPT_EPOCHS}"
echo "LR: ${ADAPT_LR}"
echo "Mode: ${BS3_MODE}"

if [ ! -r "${RESUME_CKPT}" ]; then
    echo "ERROR: resume checkpoint is not readable: ${RESUME_CKPT}" >&2
    exit 2
fi
if [ ! -d "${CACHE_DIR}" ]; then
    echo "ERROR: pseudo-label cache does not exist: ${CACHE_DIR}" >&2
    exit 2
fi
if [ ! -r "${TOPK_MANIFEST}" ]; then
    echo "ERROR: top-k manifest is not readable: ${TOPK_MANIFEST}" >&2
    exit 2
fi

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${PYTHON}" -u src/bs4_ablationA_topk_adapt.py \
  --cache "${CACHE_DIR}" \
  --resume "${RESUME_CKPT}" \
  --manifest "${TOPK_MANIFEST}" \
  --top-k "${TOP_K}" \
  --epochs "${ADAPT_EPOCHS}" \
  --batch-size "${ADAPT_BATCH}" \
  --lr "${ADAPT_LR}" \
  --weight-decay "${ADAPT_WEIGHT_DECAY}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${BS3_MODE}"

if [ "${RUN_INFERENCE}" = "1" ]; then
    INFER_OUTPUT_DIR="${OUTPUT_DIR}/test_inference_last"
    SUBMISSION_CSV="${OUTPUT_DIR}/baseline4_ablationA_top${TOP_K}_adapt_submission.csv"
    "${PYTHON}" -u utils/predict_da3_checkpoint.py \
      --ckpt "${OUTPUT_DIR}/checkpoints/last.pth" \
      --output-dir "${INFER_OUTPUT_DIR}" \
      --model "${STUDENT_MODEL}" \
      --infer-batch 32 \
      --debug-vis-limit 16

    "${PYTHON}" utils/create_submission.py \
      --pred-dir "${INFER_OUTPUT_DIR}/preds" \
      --out-csv "${SUBMISSION_CSV}"

    echo "Submission written to ${SUBMISSION_CSV}"
fi

echo "Job finished at $(date)"

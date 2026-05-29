#!/bin/bash
#SBATCH --job-name=bs4-ablationB-aug
#SBATCH --account=cil
#SBATCH --time=02:00:00
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
BS4_MODE="${BS4_MODE:-${BS3_MODE:-full_head}}"

RESUME_CKPT="${RESUME_CKPT:-/work/scratch/cdeubel/models/baseline3-DA3MONO-LARGE-full_head-2026-05-29_00-09-52/checkpoints/best.pth}"
AUG_DATASET_DIR="${AUG_DATASET_DIR:-}"
AUG_MANIFEST="${AUG_MANIFEST:-}"
EXPECTED_SOURCES="${EXPECTED_SOURCES:-250}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1250}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-1}"
ADAPT_BATCH="${ADAPT_BATCH:-8}"
ADAPT_LR="${ADAPT_LR:-1e-8}"
ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY:-1e-4}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/models/baseline4-ablationB-aug-top${EXPECTED_SOURCES}-${TEACHER_MODEL}-${BS4_MODE}-${RUN_STAMP}}"

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
export BS3_MODE="${BS4_MODE}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

. /etc/profile.d/modules.sh
module add cuda/13.0

source "${REPO_DIR}/.venv/bin/activate"
PYTHON="${REPO_DIR}/.venv/bin/python"

if [ -z "${AUG_MANIFEST}" ] && [ -n "${AUG_DATASET_DIR}" ]; then
    AUG_MANIFEST="${AUG_DATASET_DIR}/dataset_manifest.csv"
fi
if [ -z "${AUG_MANIFEST}" ]; then
    AUG_MANIFEST=$(find "${SCRATCH_ROOT}/outputs/baseline4" -maxdepth 2 -name "dataset_manifest.csv" \
        -path "*/novel_views_DA3-GIANT-1.1_cosine_top250_*" ! -path "*/smoke_*" | sort | tail -1)
fi

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Resume checkpoint: ${RESUME_CKPT}"
echo "Augmented manifest: ${AUG_MANIFEST}"
echo "Teacher model metadata: ${TEACHER_MODEL}"
echo "Expected sources: ${EXPECTED_SOURCES}"
echo "Expected samples: ${EXPECTED_SAMPLES}"
echo "Epochs: ${ADAPT_EPOCHS}"
echo "LR: ${ADAPT_LR}"
echo "Mode: ${BS4_MODE}"

if [ ! -r "${RESUME_CKPT}" ]; then
    echo "ERROR: resume checkpoint is not readable: ${RESUME_CKPT}" >&2
    exit 2
fi
if [ -z "${AUG_MANIFEST}" ] || [ ! -r "${AUG_MANIFEST}" ]; then
    echo "ERROR: augmented dataset manifest is not readable: ${AUG_MANIFEST}" >&2
    exit 2
fi

"${PYTHON}" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

"${PYTHON}" -u src/bs4_ablationB_aug_adapt.py \
  --manifest "${AUG_MANIFEST}" \
  --resume "${RESUME_CKPT}" \
  --epochs "${ADAPT_EPOCHS}" \
  --batch-size "${ADAPT_BATCH}" \
  --lr "${ADAPT_LR}" \
  --weight-decay "${ADAPT_WEIGHT_DECAY}" \
  --expected-samples "${EXPECTED_SAMPLES}" \
  --expected-sources "${EXPECTED_SOURCES}" \
  --output-dir "${OUTPUT_DIR}" \
  --mode "${BS4_MODE}"

if [ "${RUN_INFERENCE}" = "1" ]; then
    INFER_OUTPUT_DIR="${OUTPUT_DIR}/test_inference_last"
    SUBMISSION_CSV="${OUTPUT_DIR}/baseline4_ablationB_aug_top${EXPECTED_SOURCES}_submission.csv"
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

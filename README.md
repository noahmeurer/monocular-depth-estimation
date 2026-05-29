# Monocular Depth Estimation

This repository contains the code used for the CIL monocular depth project:
zero-shot DA3 evaluation, ground-truth fine-tuning, teacher pseudo-label
adaptation, distribution-aware validation selection, and novel-view
augmentation ablations.

## Environment

The project uses `uv` and installs Depth Anything 3 from the pinned git source
in `pyproject.toml`.

```bash
cd ~/monocular-depth-estimation
module add cuda/13.0
export UV_CACHE_DIR=/work/scratch/$USER/uv-cache
uv sync
source .venv/bin/activate
```

The scripts assume the CIL cluster dataset layout:

```text
/cluster/courses/cil/monocular-depth-estimation/train
/cluster/courses/cil/monocular-depth-estimation/test
```

Most jobs write to scratch. The usual convention is:

```bash
export SCRATCH_ROOT=/work/scratch/$USER
export HF_HOME=$SCRATCH_ROOT/hf_cache
export XDG_CACHE_HOME=$SCRATCH_ROOT/cache
```

The Slurm wrappers set offline Hugging Face cache mode by default, so the DA3
model weights should already be present in the user's scratch cache.

## Experiment Map

| Experiment | Purpose | Main code | Slurm / utility |
|---|---|---|---|
| Baseline 1 | DA3MONO-LARGE zero-shot test prediction | `src/bs1_zero_shot_DA3.py` | `scripts/baseline1.slurm` |
| Teacher zero-shot | Larger DA3 teacher zero-shot reference | `src/bs_teacher_zero_shot_DA3.py` | `scripts/baseline_teacher_5060.slurm`, `scripts/baseline_teacher_gb10.slurm` |
| Baseline 2 | Fine-tune DA3MONO-LARGE on provided GT depth | `src/bs2_finetune_DA3.py` | `scripts/baseline2.slurm`, `scripts/baseline2_surface_head.slurm` |
| Baseline 3 pseudo-label cache | Generate teacher pseudo-labels for train images | `src/bs3_pseudo_label_DA3.py` | `scripts/baseline3_pseudo_label_5060.slurm` |
| Baseline 3 training | Fine-tune student on teacher pseudo-labels and validate on GT | `src/bs3_pseudo_label_DA3_train.py` | `scripts/baseline3_train.slurm` |
| Baseline 3 feature extraction | Extract pooled DA3/DINO backbone features for train/test images | `src/bs3_extract_da3_features.py` | `scripts/baseline3_extract_features.slurm` |
| Validation manifests | Build cosine/PCA validation subsets and HTML previews | `scripts/baseline3_build_val_manifest.py` | standalone Python utility |
| Baseline 4 ablation A | Resume best Baseline 3 checkpoint and adapt on top-k target-similar pseudo-labeled images | `src/bs4_ablationA_topk_adapt.py` | `scripts/baseline4_ablationA_top250_adapt.slurm` |
| Baseline 4 novel views | Generate original plus left/right/up/down samples for top-k images | `src/bs4_generate_novel_views.py` | `scripts/baseline4_generate_novel_views.slurm` |
| Baseline 4 ablation B | Fine-tune on generated original plus novel-view augmented dataset | `src/bs4_ablationB_aug_adapt.py` | `scripts/baseline4_ablationB_aug_adapt.slurm` |
| Submission helpers | Run checkpoint inference and create Kaggle CSV | `utils/predict_da3_checkpoint.py`, `utils/create_submission.py` | standalone utilities |

## Reproducibility Guide

### Baseline 1: zero-shot student

```bash
sbatch scripts/baseline1.slurm
```

Expected output:

```text
$SCRATCH_ROOT/outputs/baseline1/
```

### Baseline 2: GT fine-tuning

```bash
SCRATCH_ROOT=/work/scratch/$USER \
BS2_MODE=full_head \
sbatch scripts/baseline2.slurm
```

The training script saves `best.pth`, `last.pth`, predictions, and a submission
CSV under:

```text
$SCRATCH_ROOT/outputs/baseline2/
```

### Baseline 3: teacher pseudo-label adaptation

First generate or point to a teacher pseudo-label cache. The expected default
location is:

```text
$SCRATCH_ROOT/outputs/baseline3/pseudo_labels_<TEACHER_MODEL>/
```

Representative pseudo-label generation:

```bash
TEACHER_MODEL=DA3-GIANT-1.1 \
SCRATCH_ROOT=/work/scratch/$USER \
sbatch scripts/baseline3_pseudo_label_5060.slurm
```

Then train the student on the pseudo-labels:

```bash
TEACHER_MODEL=DA3-GIANT-1.1 \
CACHE_DIR=/work/scratch/$USER/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \
BS3_MODE=full_head \
VAL_SUBSET_STRATEGY=random \
EPOCHS=3 \
sbatch scripts/baseline3_train.slurm
```

For manifest-based validation, pass a CSV produced by the validation manifest
utility:

```bash
VAL_SUBSET_STRATEGY=manifest \
VAL_MANIFEST=/work/scratch/$USER/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv \
TEACHER_MODEL=DA3-GIANT-1.1 \
CACHE_DIR=/work/scratch/$USER/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \
BS3_MODE=full_head \
VAL_INTERVAL_STEPS=500 \
sbatch scripts/baseline3_train.slurm
```

### Validation set filtering

Extract pooled DA3/DINO features:

```bash
SCRATCH_ROOT=/work/scratch/$USER \
STUDENT_MODEL=DA3MONO-LARGE \
sbatch scripts/baseline3_extract_features.slurm
```

Build cosine and PCA validation manifests from the cached features:

```bash
python scripts/baseline3_build_val_manifest.py \
  --feature-dir /work/scratch/$USER/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE \
  --methods cosine,pca64 \
  --val-fraction 0.10 \
  --preview-count 500 \
  --write-pca2d-plot \
  --write-test-gallery
```

The utility writes `val_top10pct.csv`, `preview_top500.html`, and optional report
figures under:

```text
$SCRATCH_ROOT/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/
```

### Baseline 4 ablations

Ablation A resumes from a saved Baseline 3 checkpoint and trains only on the
top-k cosine-ranked pseudo-labeled images:

```bash
RESUME_CKPT=/path/to/baseline3/checkpoints/best.pth \
CACHE_DIR=/work/scratch/$USER/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \
TOPK_MANIFEST=/work/scratch/$USER/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv \
TOP_K=250 \
ADAPT_EPOCHS=3 \
ADAPT_LR=1e-8 \
sbatch scripts/baseline4_ablationA_top250_adapt.slurm
```

Generate the novel-view augmented dataset for ablation B:

```bash
TEACHER_MODEL=DA3-GIANT-1.1 \
PSEUDO_DIR=/work/scratch/$USER/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \
TOPK_MANIFEST=/work/scratch/$USER/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv \
TOP_K=250 \
sbatch scripts/baseline4_generate_novel_views.slurm
```

Then fine-tune on the generated manifest:

```bash
AUG_MANIFEST=/work/scratch/$USER/outputs/baseline4/<novel_view_run>/dataset_manifest.csv \
RESUME_CKPT=/path/to/baseline3/checkpoints/best.pth \
EXPECTED_SOURCES=250 \
EXPECTED_SAMPLES=1250 \
ADAPT_EPOCHS=3 \
ADAPT_LR=1e-8 \
sbatch scripts/baseline4_ablationB_aug_adapt.slurm
```

## Creating a Submission From a Checkpoint

```bash
python utils/predict_da3_checkpoint.py \
  --ckpt /path/to/checkpoints/best.pth \
  --output-dir /work/scratch/$USER/outputs/my_submission_run \
  --model DA3MONO-LARGE \
  --infer-batch 32 \
  --debug-vis-limit 16

python utils/create_submission.py \
  --pred-dir /work/scratch/$USER/outputs/my_submission_run/preds \
  --out-csv /work/scratch/$USER/outputs/my_submission_run/submission.csv
```

## Important Scratch Artifacts

The exact scratch locations used during the project depended on the user who
launched each job. These paths are the important artifact types needed to
reproduce the main experiments:

```text
$SCRATCH_ROOT/outputs/baseline1/
$SCRATCH_ROOT/outputs/baseline2/
$SCRATCH_ROOT/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1/
$SCRATCH_ROOT/outputs/baseline3/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1/
$SCRATCH_ROOT/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/
$SCRATCH_ROOT/models/baseline3-*/
$SCRATCH_ROOT/outputs/baseline4/novel_views_DA3-GIANT-1.1_cosine_top250_*/
$SCRATCH_ROOT/models/baseline4-*/
```

`SCRATCH.md` may contain a more specific, team-local index of generated
artifacts. It is useful for internal reproducibility, but should be checked
before public submission because scratch paths are user-specific.

## Candidate Cleanup Before Zip Submission

The following items are candidates for removal or exclusion from the final zip
unless the course staff explicitly expects them:

- Hugging Face scratch sync support: `scripts/hf_scratch_sync.py`,
  `.env.example`, and the related README/SCRATCH private Hub workflow.
- GB10-specific wrappers and notes: `scripts/baseline_teacher_gb10.slurm` and
  GB10-only setup details in `CLUSTER.md`.
- Obsolete or exploratory notebooks: especially
  `notebooks/novel_view_synthesis_messy.ipynb` if the cleaned notebook is kept.
- Local/generated outputs: `outputs/`, `logs/`, W&B run folders, preview panels,
  predictions, checkpoint directories, and generated submissions.
- One-off debug utilities that are not referenced in the paper or README, such
  as `scripts/preview_bs3_pseudo_labels.py` and
  `utils/test_baseline2_checkpoint.py`.
- Cluster-internal documentation (`CLUSTER.md`, `SCRATCH.md`) if the final
  submission should be portable rather than ETH-cluster-specific.

Do not delete source files, Slurm wrappers, or utilities needed by the
experiment map above without first updating this README.

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md`, `CLUSTER.md` | Cluster onboarding and environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |
| 3 | Codex | `scripts/baseline1.slurm` | Slurm job script for DA3MONO-LARGE zero-shot baseline |
| 4 | Claude (Cursor) | `README.md`, `CLUSTER.md`, `pyproject.toml`, `scripts/baseline_teacher_gb10.slurm` | GB10/uv dual-environment setup (`.venv-gb10`, PyTorch cu130 index, xformers x86-only) |
| 5 | Codex | `scripts/preview_bs3_pseudo_labels.py` | Debug script for Baseline 3 pseudo-label inspection |
| 6 | Claude (Cursor) | `scripts/hf_scratch_sync.py`, `.env.example`, `SCRATCH.md`, `README.md` | Private Hub push/pull for scratch artifacts and scratch artifact index |
| 7 | Codex | `src/bs2_finetune_DA3.py`, `scripts/baseline2.slurm`, `scripts/baseline2_surface_head.slurm` | Baseline 2 GT fine-tuning, corrected per-image SiLog validation, checkpoint selection, and full/surface-head modes |
| 8 | Codex | `src/bs3_pseudo_label_DA3_train.py`, `scripts/baseline3_train.slurm` | Baseline 3 pseudo-label adaptation with intra-epoch validation and random/manifest validation selection |
| 9 | Codex | `src/bs3_extract_da3_features.py`, `scripts/baseline3_extract_features.slurm`, `scripts/baseline3_build_val_manifest.py` | DA3/DINO feature extraction, cosine/PCA validation-manifest construction, PCA plot, HTML preview, and test-gallery utilities |
| 10 | Codex | `src/bs4_ablationA_topk_adapt.py`, `scripts/baseline4_ablationA_top250_adapt.slurm` | Baseline 4 ablation A top-k target-similar adaptation from a saved Baseline 3 checkpoint |
| 11 | Claude (Cursor) | `notebooks/novel_view_synthesis.ipynb` | Gaussian-splat rendering and depth reprojection exploration |
| 12 | Claude (Cursor) | `notebooks/novel_view_synthesis.ipynb` | Clean refactor of `novel_view_synthesis_messy.ipynb` and stacked multi-view figures |
| 13 | Codex | `src/bs4_generate_novel_views.py`, `scripts/baseline4_generate_novel_views.slurm` | Baseline 4 top-250 cosine novel-view dataset generation using DA3-GIANT Gaussian splats and confidence-masked reprojected depth |
| 14 | Codex | `src/bs4_ablationB_aug_adapt.py`, `scripts/baseline4_ablationB_aug_adapt.slurm` | Baseline 4 ablation B fine-tuning on the generated original plus left/right/up/down augmented dataset |
| 15 | Codex | `README.md` | Submission-focused README cleanup, experiment-to-file mapping, reproducibility commands, and candidate cleanup list |

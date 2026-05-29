# Monocular Depth Estimation

This repository contains the code used for the CIL monocular depth project:
zero-shot DA3 evaluation, ground-truth fine-tuning, teacher pseudo-label
adaptation, distribution-aware validation selection, and novel-view dataset
generation.

## Baseline Overview

- **Baseline 1:** run DA3MONO-LARGE zero-shot to establish the compact student reference.
- **Baseline 2:** fine-tune the student on provided ground-truth depth to test direct task-label adaptation.
- **Baseline 3:** fine-tune on stronger-teacher pseudo-labels, with cosine/PCA validation manifests for target-aware checkpoint selection.
- **Baseline 4:** generate teacher-guided novel views for the most target-similar images to test geometric augmentation quality.

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
| Teacher zero-shot | Larger DA3 teacher zero-shot reference | `src/bs_teacher_zero_shot_DA3.py` | `scripts/baseline_teacher_5060.slurm` |
| Baseline 2 | Fine-tune DA3MONO-LARGE on provided GT depth | `src/bs2_finetune_DA3.py` | `scripts/baseline2.slurm`, `scripts/baseline2_surface_head.slurm` |
| Baseline 3 pseudo-label cache | Generate teacher pseudo-labels for train images | `src/bs3_pseudo_label_DA3.py` | `scripts/baseline3_pseudo_label_5060.slurm` |
| Baseline 3 training | Fine-tune student on teacher pseudo-labels and validate on GT | `src/bs3_pseudo_label_DA3_train.py` | `scripts/baseline3_train.slurm` |
| Baseline 3 feature extraction | Extract pooled DA3/DINO backbone features for train/test images | `src/bs3_extract_da3_features.py` | `scripts/baseline3_extract_features.slurm` |
| Validation manifests | Build cosine/PCA validation subsets and HTML previews | `scripts/baseline3_build_val_manifest.py` | standalone Python utility |
| Baseline 4 novel views | Generate original plus left/right/up/down samples for top-k images | `src/bs4_generate_novel_views.py` | `scripts/baseline4_generate_novel_views.slurm` |
| Submission helper | Create Kaggle CSV from saved prediction arrays | `utils/create_submission.py` | standalone utility |

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

### Baseline 4: novel-view dataset generation

Baseline 4 generates a small novel-view augmented dataset from the top-k
cosine-ranked training images. Each selected source image keeps its original
teacher pseudo-depth and adds left/right/up/down rendered views with
confidence-masked reprojected depth:

```bash
TEACHER_MODEL=DA3-GIANT-1.1 \
PSEUDO_DIR=/work/scratch/$USER/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \
TOPK_MANIFEST=/work/scratch/$USER/outputs/baseline3/feature_cache/da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv \
TOP_K=250 \
sbatch scripts/baseline4_generate_novel_views.slurm
```

## Creating a Submission From Prediction Arrays

```bash
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
```

## Candidate Cleanup Before Zip Submission

The following items are candidates for removal or exclusion from the final zip
unless the course staff explicitly expects them:

- Obsolete or exploratory notebooks: especially
  `notebooks/novel_view_synthesis_messy.ipynb` if the cleaned notebook is kept.
- Local/generated outputs: `outputs/`, `logs/`, W&B run folders, preview panels,
  predictions, checkpoint directories, and generated submissions.
- One-off debug utilities that are not referenced in the paper or README, such
  as `scripts/preview_bs3_pseudo_labels.py` and
  `utils/test_baseline2_checkpoint.py`.

Do not delete source files, Slurm wrappers, or utilities needed by the
experiment map above without first updating this README.

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md` | Cluster-oriented environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |
| 3 | Codex | `scripts/baseline1.slurm` | Slurm job script for DA3MONO-LARGE zero-shot baseline |
| 4 | Claude (Cursor) | `README.md`, `pyproject.toml` | uv environment setup |
| 5 | Codex | `scripts/preview_bs3_pseudo_labels.py` | Debug script for Baseline 3 pseudo-label inspection |
| 6 | Codex | `src/bs2_finetune_DA3.py`, `scripts/baseline2.slurm`, `scripts/baseline2_surface_head.slurm` | Baseline 2 GT fine-tuning, corrected per-image SiLog validation, checkpoint selection, and full/surface-head modes |
| 7 | Codex | `src/bs3_pseudo_label_DA3_train.py`, `scripts/baseline3_train.slurm` | Baseline 3 pseudo-label adaptation with intra-epoch validation and random/manifest validation selection |
| 8 | Codex | `src/bs3_extract_da3_features.py`, `scripts/baseline3_extract_features.slurm`, `scripts/baseline3_build_val_manifest.py` | DA3/DINO feature extraction, cosine/PCA validation-manifest construction, PCA plot, HTML preview, and test-gallery utilities |
| 9 | Claude (Cursor) | `notebooks/novel_view_synthesis.ipynb` | Gaussian-splat rendering and depth reprojection exploration |
| 10 | Claude (Cursor) | `notebooks/novel_view_synthesis.ipynb` | Clean refactor of `novel_view_synthesis_messy.ipynb` and stacked multi-view figures |
| 11 | Codex | `src/bs4_generate_novel_views.py`, `scripts/baseline4_generate_novel_views.slurm` | Baseline 4 top-250 cosine novel-view dataset generation using DA3-GIANT Gaussian splats and confidence-masked reprojected depth |
| 12 | Codex | `README.md` | Submission-focused README cleanup, experiment-to-file mapping, reproducibility commands, and candidate cleanup list |

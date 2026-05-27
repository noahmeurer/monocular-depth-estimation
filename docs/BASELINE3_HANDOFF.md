# Baseline 3 pseudo-label job — cluster handoff (mdealvaro)

Instructions for running Noah’s baseline-3 pseudo-label generation on the ETH student cluster using paths preconfigured for **`mdealvaro`**.

Use branch **`handoff/mdealvaro`** (do not use `main` for this run).

## Goal

Generate teacher pseudo-depth labels for the CIL training set (~22,605 images):

- **Input:** `/cluster/courses/cil/monocular-depth-estimation/train/*_rgb.png`
- **Output:** `/work/scratch/mdealvaro/outputs/baseline3/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1/*_depth.npy`
- **Teacher:** `DA3NESTED-GIANT-LARGE-1.1` (see `src/bs3_pseudo_label_DA3.py`)
- **Runtime:** ~2–3 h on a 5060 Ti (Slurm limit 8 h)

## Prerequisites

- CIL cluster account with **`cil_jobs`** GPU hours
- **`uv`** installed ([docs](https://docs.astral.sh/uv/))
- Write access to `/work/scratch/mdealvaro`

## One-time setup

```bash
git clone -b handoff/mdealvaro \
  https://github.com/noahmeurer/monocular-depth-estimation.git \
  ~/monocular-depth-estimation

cd ~/monocular-depth-estimation

export UV_CACHE_DIR=/work/scratch/mdealvaro/uv-cache
mkdir -p "$UV_CACHE_DIR" logs/slurm

. /etc/profile.d/modules.sh
module add cuda/13.0

uv sync
source .venv/bin/activate

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Submit job

```bash
cd ~/monocular-depth-estimation
sbatch scripts/baseline3_pseudo_label_5060.slurm
```

Monitor:

```bash
squeue --me
tail -f logs/slurm/slurm-baseline3-pseudo-5060-<JOBID>.out
```

Progress (done when count is **22605**):

```bash
ls /work/scratch/mdealvaro/outputs/baseline3/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1 | wc -l
```

Success: log ends with `Pseudo-label generation finished` and file count is 22605.

## Optional: preview first 10 samples (CPU)

After some `.npy` files exist:

```bash
python scripts/preview_bs3_pseudo_labels.py --num 10
```

Previews: `/work/scratch/mdealvaro/outputs/baseline3/pseudo_preview/`

## After the job

When the job finishes (22,605 `*_depth.npy` files), copy the pseudo-label folder to your laptop, then upload it for Noah.

**1. Download from the cluster** (run on your laptop, not on the cluster):

```bash
mkdir -p ~/Downloads/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1

rsync -avz --progress \
  mdealvaro@student-cluster.inf.ethz.ch:/work/scratch/mdealvaro/outputs/baseline3/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1/ \
  ~/Downloads/pseudo_labels_DA3NESTED-GIANT-LARGE-1.1/
```

This may take a while (~22k files). Use ETH VPN if you are off campus.

**2. Upload to Google Drive**

- Upload the folder `pseudo_labels_DA3NESTED-GIANT-LARGE-1.1` to Google Drive (zipped is fine if the UI struggles with many small files).
- Share it with Noah (he will send you the Google account to share with).
- Message Noah when the share is ready and confirm the file count is 22,605.

## Troubleshooting

| Issue | Fix |
|--------|-----|
| `Cache directory ... already exists` | Remove empty/partial dir only after confirming with Noah |
| Job pending | Normal; first GPU job of the day may take ~5 min |
| CUDA errors | `module add cuda/13.0` before `uv sync`; job requests `5060ti:1` |
| HF download slow | First run caches model under `/work/scratch/mdealvaro/hf_cache` |

# Monocular Depth Estimation

## Setup

```bash
git clone https://github.com/noahmeurer/monocular-depth-estimation.git ~/monocular-depth-estimation
cd ~/monocular-depth-estimation
```

Depth Anything 3 is installed automatically by `uv sync` (git dependency in `pyproject.toml`).

### Environment (uv)

Uses [uv](https://docs.astral.sh/uv/) with CUDA 13.0 PyTorch wheels from `pyproject.toml`.

Copy `.env.example` to `.env` and set `CLUSTER_USERNAME` to your student-cluster login (used for `/work/scratch` paths):

```bash
cp .env.example .env
# edit CLUSTER_USERNAME, HF_TOKEN, etc.
source .env
```

On x86 GPU nodes, load CUDA before `uv sync`:

```bash
module add cuda/13.0
```

Keep uv’s cache on scratch (20GB home quota):

```bash
export UV_CACHE_DIR=/work/scratch/${CLUSTER_USERNAME}/uv-cache
```

#### x86 GPUs (5060 Ti, 2080 Ti, 1080 Ti)

On the login node or any x86 GPU node:

```bash
cd ~/monocular-depth-estimation
module add cuda/13.0
uv sync
source .venv/bin/activate
```

#### GB10 (ARM)

GB10 is **aarch64** — use `.venv-gb10`, not `.venv`. Interactive setup and `drop-caches`: **`CLUSTER.md` → GB10 (interactive)**.

#### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

**Train:** `/cluster/courses/cil/monocular-depth-estimation/train`

**Test:** `/cluster/courses/cil/monocular-depth-estimation/test`

## Running Jobs

Interactive session (x86 GPU):

```bash
srun --pty --gpus 5060ti:1 -A cil_jobs -t 120 bash --login
```

Batch job:

```bash
sbatch scripts/baseline1.slurm
```

Slurm accounts, storage, Jupyter: `CLUSTER.md`.

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md`, `CLUSTER.md` | Cluster onboarding, environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |
| 3 | Codex | `scripts/baseline1.slurm` | Slurm job script for DA3MONO-LARGE zero-shot baseline |
| 4 | Claude (Cursor) | `README.md`, `CLUSTER.md`, `pyproject.toml`, `scripts/baseline_teacher_gb10.slurm` | GB10/uv dual-env setup (`.venv-gb10`, PyTorch cu130 index, xformers x86-only) |
| 5 | Codex | `scripts/preview_bs3_pseudo_labels.py` | Debug script for baseline3 pseudo-label generation |

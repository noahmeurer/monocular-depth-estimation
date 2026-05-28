# Monocular Depth Estimation

## Setup

```bash
git clone https://github.com/noahmeurer/monocular-depth-estimation.git ~/monocular-depth-estimation
cd ~/monocular-depth-estimation
```

Depth Anything 3 is installed automatically by `uv sync` (git dependency in `pyproject.toml`).

### Environment (uv)

Uses [uv](https://docs.astral.sh/uv/) with CUDA 13.0 PyTorch wheels from `pyproject.toml`.

Copy `.env.example` to `.env` and fill in your username under the `/work/scratch/<your_username>/...` paths and your `HF_TOKEN`:

```bash
cp .env.example .env
# edit HF_TOKEN, HF_HOME, XDG_CACHE_HOME
source .env
```

On x86 GPU nodes, load CUDA before `uv sync`:

```bash
module add cuda/13.0
```

Keep uv’s cache on scratch (20GB home quota):

```bash
export UV_CACHE_DIR=/work/scratch/<your_username>/uv-cache
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

## Sharing scratch artifacts

Scratch dirs are world-readable on the cluster, so the simplest way to share between teammates is to `rsync`/`cp` straight from another user's `/work/scratch/<user>/...`. See `SCRATCH.md` for the index of known artifacts (baseline outputs, pseudo-labels, etc.).

For sharing off-cluster (e.g. to a teammate without cluster access, or as a durable backup), `scripts/hf_scratch_sync.py` pushes/pulls scratch paths to a private Hub repo (`HF_SCRATCH_REPO_ID` in `.env`):

```bash
# upload outputs/baseline1 from your scratch to the team's private HF repo
python scripts/hf_scratch_sync.py push outputs/baseline1

# pull it back on another machine into the same scratch-relative path
python scripts/hf_scratch_sync.py pull outputs/baseline1

# list what's on the Hub
python scripts/hf_scratch_sync.py list outputs/
```

For folders with thousands of files (e.g. pseudo-labels), use `--chunk-size 1000 --sleep-between-chunks 30` to stay under HF's commit-rate limits. `--scratch-root /work/scratch/<other_user>` lets you push someone else's tree without touching your own.

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md`, `CLUSTER.md` | Cluster onboarding, environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |
| 3 | Codex | `scripts/baseline1.slurm` | Slurm job script for DA3MONO-LARGE zero-shot baseline |
| 4 | Claude (Cursor) | `README.md`, `CLUSTER.md`, `pyproject.toml`, `scripts/baseline_teacher_gb10.slurm` | GB10/uv dual-env setup (`.venv-gb10`, PyTorch cu130 index, xformers x86-only) |
| 5 | Codex | `scripts/preview_bs3_pseudo_labels.py` | Debug script for baseline3 pseudo-label generation |
| 6 | Claude (Cursor) | `scripts/hf_scratch_sync.py`, `.env.example`, `SCRATCH.md`, `README.md` | Private Hub push/pull for scratch artifacts (`cil-mono-depth-26`); scratch artifact index |

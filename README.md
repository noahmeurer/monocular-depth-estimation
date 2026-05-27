# Monocular Depth Estimation

## Setup

Clone both repos side by side:

```bash
git clone https://github.com/noahmeurer/monocular-depth-estimation.git ~/monocular-depth-estimation
git clone https://github.com/bytedance-seed/depth-anything-3 ~/depth-anything-3
cd ~/depth-anything-3 && git submodule update --init --recursive
```

### Environment (uv)

This project uses [uv](https://docs.astral.sh/uv/). CUDA 13.0 PyTorch wheels are configured in `pyproject.toml`. 

Before launching interactive 5060 Ti sessions, always load the matching CUDA module before `uv sync`. Do not load the module before launching interactive GB10 sessions.

```bash
module add cuda/13.0
```

Keep uv’s cache on scratch so the 20GB home quota is not filled:

```bash
export UV_CACHE_DIR=/work/scratch/mdealvaro/uv-cache
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

GB10 nodes are **aarch64**. A `.venv` built on the login node (x86) will not work there — use a separate `.venv-gb10` and run `uv sync` **only on a GB10 node**.

1. Request an interactive GB10 session (always use `--login`):

   ```bash
   srun --gpus gb10:1 --pty -A cil_jobs -t 120 bash --login
   ```

2. On the GB10 node, check which CUDA modules exist (`module avail` — not every version is available on GB10).

3. Create the environment:

   ```bash
   cd ~/monocular-depth-estimation
   . /etc/profile.d/modules.sh
   module add cuda/13.0   # or a version listed by module avail on GB10

   export UV_PROJECT_ENVIRONMENT="$PWD/.venv-gb10"
   export UV_CACHE_DIR=/work/scratch/mdealvaro/uv-cache
   uv sync
   source .venv-gb10/bin/activate
   ```

`xformers` is only installed on x86_64 (no official ARM wheels). Depth Anything 3 falls back to pure PyTorch on GB10.

> **Later (baseline 4 / 3DGS novel views):** DA3’s `gsplat` extra is not in the default env. If you need Gaussian rendering on GB10, you’ll likely have to add `depth-anything-3[gs]` and build or JIT-install `gsplat` on a GB10 node (no Linux aarch64 wheels today) — prototype on x86 first.

Before GPU-heavy work on GB10 (interactive sessions), free RAM used as filesystem cache (CPU and GPU share memory):

```bash
/usr/bin/drop-caches
```

This may fail under `sbatch` (no TTY for sudo); `baseline_teacher.slurm` skips it non-fatally in that case.

GB10 batch jobs: `scripts/baseline_teacher.slurm` (`source .venv-gb10`).

#### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Cluster reference (Slurm, storage, quotas): see `CLUSTER.md`.

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

See `CLUSTER.md` for full cluster reference (accounts, `squeue`, Jupyter, storage).

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md`, `CLUSTER.md` | Cluster onboarding, environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |
| 3 | Codex | `scripts/baseline1.slurm` | Slurm job script for DA3MONO-LARGE zero-shot baseline |
| 4 | Claude (Cursor) | `README.md`, `CLUSTER.md`, `pyproject.toml`, `scripts/baseline_teacher_gb10.slurm` | GB10/uv dual-env setup (`.venv-gb10`, PyTorch cu130 index, xformers x86-only) |
| 5 | Codex | `scripts/preview_bs3_pseudo_labels.py` | Debug script for baseline3 pseudo-label generation |

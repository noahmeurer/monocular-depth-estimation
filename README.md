# Monocular Depth Estimation

## Setup

Clone both repos side by side:

```bash
git clone https://github.com/noahmeurer/monocular-depth-estimation.git ~/monocular-depth-estimation
git clone https://github.com/bytedance-seed/depth-anything-3 ~/depth-anything-3
cd ~/depth-anything-3 && git submodule update --init --recursive
```

Set up the environment (on the student cluster login node):

```bash
cd ~/monocular-depth-estimation
module add cuda/13.0
module save default
uv sync
source .venv/bin/activate
```

## Data

**Train:** `/cluster/courses/cil/monocular-depth-estimation/train`

**Test:** `/cluster/courses/cil/monocular-depth-estimation/test`

## Running Jobs

Interactive session:
```bash
srun --pty --gpus 5060ti:1 -A cil_jobs -t 120 bash --login
```

Batch job:
```bash
sbatch train.sh
```

See `CLUSTER.md` for full cluster reference.

## AI Usage Declaration

| # | Tool | Files affected | Purpose |
|---|------|----------------|---------|
| 1 | Claude Sonnet | `README.md`, `CLUSTER.md` | Cluster onboarding, environment setup |
| 2 | Claude (Cursor) | `notebooks/visualize_dataset.ipynb` | Matplotlib syntax for `visualize_random_batch` dataset inspector |

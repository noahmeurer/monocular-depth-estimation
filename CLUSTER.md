# ETH Student Cluster — Onboarding Guide

Quick reference for running CIL monocular depth estimation work on the D-INFK student cluster.

## Resource Budget

You have two separate GPU time pools:

| Tag | Hours | Max Runtime | Use For |
|---|---|---|---|
| `cil` | 100h | 60 min | Jupyter, short interactive sessions |
| `cil_jobs` | 200h | 24h | Training runs, batch jobs |

Check remaining time at any time:
```bash
courses
```

> Use `cil_jobs` for real training — don't burn your `cil` budget on long runs.

---

## Running Jobs

### Interactive session (stays in your current terminal)
```bash
srun --pty -A cil -t 60 bash --login
```

For longer interactive work (up to 24h):
```bash
srun --pty -A cil_jobs -t 480 bash --login
```

### Batch job (fire and forget — preferred for training)

Create a script `train.sh`:
```bash
#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=cil_jobs
#SBATCH --output=train_output.out

. /etc/profile.d/modules.sh
module add cuda/13.0

python train.py
```

Submit it:
```bash
sbatch train.sh
```

### Job management
```bash
squeue --me --start # your jobs + estimated start time for pending
squeue            # check if job is running or waiting
scancel <job_id>  # cancel a job
```

> The cluster powers down idle nodes. First job of the day may take up to **5 minutes** to start — this is normal.

---

## Jupyter Notebooks

**Easiest path:** https://student-jupyter.inf.ethz.ch — select the `cil` course and an environment. GPU is automatically attached.

**Critical:** Closing your browser tab does NOT stop the server. It keeps consuming your `cil` hours. Always go to **Home → Stop My Server** when done.

---

## CUDA and PyTorch Setup

Project environment setup (uv, `.venv` / `.venv-gb10`): see **README.md**.

```bash
module avail              # see available CUDA versions
module add cuda/13.0      # load a specific version
module save default       # make it the default for future sessions
```

Without uv, install PyTorch manually (match the CUDA version in the URL):

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Always use `--no-cache-dir` with pip — home directory is only 20GB.

Verify GPU is working:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

In batch scripts, modules aren't loaded automatically — add this after your `#SBATCH` lines:
```bash
. /etc/profile.d/modules.sh
module add cuda/13.0
```

---

## Storage

| Location | Size | Use For |
|---|---|---|
| `~` (home) | 20GB | Code, environments, small files |
| `/work/scratch/mdealvaro` | 100GB | Datasets, model checkpoints |
| `/tmp` ($TMPDIR) | 40GB | Fast local SSD — deleted when job ends |

Keep datasets and checkpoints in `/work/scratch/mdealvaro` — it's accessible from all nodes. Auto-deletion applies based on age:

| Used Space | Max Age |
|---|---|
| < 10GB | 7 days |
| 10–50GB | 2 days |
| > 50GB | 1 day |

---

## Copying Data to the Cluster

**From terminal (Mac):**
```bash
scp /local/path/to/file mdealvaro@student-cluster.inf.ethz.ch:~/destination/
```

**Mount as network drive (Mac Finder):**
Go → Connect to Server → `smb://student-files.inf.ethz.ch`

---

## GPU Options

| Type | VRAM | Tag |
|---|---|---|
| RTX 5060 Ti | 16GB | `5060ti` |
| RTX 2080 Ti | 11GB | `2080ti` |
| GTX 1080 Ti | 11GB | `1080ti` |
| GB10 (ARM) | 128GB | `gb10` |

Request a specific GPU:
```bash
srun --pty -A cil_jobs -t 120 --gpus 2080ti:1 bash --login
```

GB10 (ARM, unified CPU/GPU memory):
```bash
srun --gpus gb10:1 --pty -A cil_jobs -t 120 bash --login
```

Default (no `--gpus` flag) assigns based on availability, priority order: 5060 Ti → 2080 Ti → 1080 Ti.

### GB10 nodes (ARM)

GB10 systems are **aarch64** with unified CPU/GPU memory (~116 GB usable). Environment setup: **README.md** (`.venv-gb10`, `uv sync` on a GB10 node only).

Cluster-specific notes:

- Always use `bash --login` in `srun` (see command above).
- Not all CUDA modules available on x86 are available on GB10 — run `module avail` on the node.
- Before GPU-heavy work: `/usr/bin/drop-caches` (Linux uses free RAM as buffer cache).
- Example batch script: `scripts/baseline_teacher.slurm`.

References: [GB10 nodes](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobsGB10), [CUDA and PyTorch](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda).

---

## Common Pitfalls

- **Don't pip install without `--no-cache-dir`** — you'll fill your 20GB home directory fast.
- **Don't leave Jupyter servers running** when you're not using them.
- **Don't do compute on the login node** — it has hard limits (0.5 CPU cores sustained) and will get throttled. Login nodes are for setup, compilation, and file management only.
- **Batch jobs can get cancelled** if the cluster is under heavy load — save checkpoints regularly so you can resume.
- **Hugging Face / git-lfs models:** after cloning, run `lfs-hardlink path_to_checkout` to halve the disk usage (every file exists twice in git-lfs checkouts by default).
- **CUDA version mismatch:** if you get `RuntimeError: detected CUDA version mismatches`, make sure you loaded the same CUDA module version that you used when installing torch.
- **GB10 / ARM:** see README — use `.venv-gb10`, not `.venv`.

import os
import argparse
from pathlib import Path
from typing import List, Literal
from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import wandb

from depth_anything_3.api import DepthAnything3

### Configs
SCRATCH_ROOT = Path("/work/scratch/nmeurer")
DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation")

TEACHER_MODEL = "DA3NESTED-GIANT-LARGE-1.1"
STUDENT_MODEL = "DA3MONO-LARGE"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE     = 560
TRAIN_BATCH  = 8
INFER_BATCH  = 32
EPOCHS       = 3
LR           = 1e-6
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0
VAL_SPLIT    = 0.1
NUM_WORKERS  = 0
AMP          = True
SEED         = 42
LOG_INTERVAL = 20

WANDB_PROJECT = "monocular-depth-estimation"

# ---- switch here to change training mode ----
MODE           : Literal["full_head", "lora_dpt_blocks"] = "full_head"
WANDB_RUN_NAME = f"baseline3-{STUDENT_MODEL}-{MODE}"
# ---------------------------------------------

# LoRA on the 4 DPT extraction blocks (out_layers=[4,11,17,23]) — attn + MLP
_DPT_LORA_TARGETS = r"model\.backbone\.pretrained\.blocks\.(4|11|17|23)\.(attn\.(qkv|proj)|mlp\.fc[12])"
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    init_lora_weights="gaussian",
    lora_dropout=0.0,
    bias="none",
    target_modules=_DPT_LORA_TARGETS,
)
###


def debug_vis(path: Path, rgb: np.ndarray, depth: np.ndarray, debug_dir: Path):
    valid = np.isfinite(depth)
    if not np.any(valid):
        print("Warning (debug_vis): No valid depth values found")
        return

    vis_depth = depth.copy()
    lo, hi = np.percentile(vis_depth[valid], [1, 99])
    depth_norm = np.clip((vis_depth - lo) / max(hi - lo, 1e-6), 0, 1)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    depth_rgb = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    gap_width = 12
    gap = np.full((rgb.shape[0], gap_width, 3), 255, dtype=np.uint8)
    side_by_side = np.concatenate(
        [rgb.astype(np.uint8), gap, depth_rgb.astype(np.uint8)],
        axis=1,
    )

    out_name = path.stem.replace("_rgb", "_depth_vis") + ".png"
    Image.fromarray(side_by_side).save(debug_dir / out_name)


class PseudoLabelDataset(Dataset):
    """Images from train_dir, pseudo-label depths from cache_dir."""

    def __init__(self, train_dir: Path, cache_dir: Path, img_size: int = IMG_SIZE):
        self.image_paths = sorted(train_dir.glob("*_rgb.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No *_rgb.png images in {train_dir}")
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def _depth_path(self, img_path: Path) -> Path:
        depth_name = img_path.name.replace("_rgb.png", "_depth.npy")
        return self.cache_dir / depth_name

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.LANCZOS)
        depth = np.load(self._depth_path(img_path)).astype(np.float32)
        if depth.shape != (self.img_size, self.img_size):
            depth = np.array(
                Image.fromarray(depth).resize((self.img_size, self.img_size), Image.NEAREST)
            )

        image = self.normalize(TF.to_tensor(image))
        depth = torch.from_numpy(depth).unsqueeze(0)
        return image, depth


def silog_loss(pred: torch.Tensor, target: torch.Tensor, lambda_: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    valid = (target > eps) & (pred > eps)
    d = torch.log(pred[valid]) - torch.log(target[valid])
    return torch.sqrt((torch.mean(d ** 2) - lambda_ * torch.mean(d) ** 2).clamp(min=1e-8))


def load_model(device: torch.device, mode: Literal["full_head", "lora_dpt_blocks"]) -> DepthAnything3:
    model = DepthAnything3.from_pretrained(f"depth-anything/{STUDENT_MODEL}")

    if mode == "full_head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True

    elif mode == "lora_dpt_blocks":
        model = get_peft_model(model, lora_config)

    else:
        raise ValueError(f"Invalid mode {mode}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,} params")

    return model.to(device)


def forward_train(model: DepthAnything3, images: torch.Tensor) -> torch.Tensor:
    out = model.model(images.unsqueeze(1))  # [B, C, H, W] -> [B, 1, C, H, W]
    depth = out["depth"] if isinstance(out, dict) else out
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    return F.interpolate(depth, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).clamp(min=1e-3)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int) -> float:
    model.train()
    if MODE == "full_head":
        model.model.backbone.eval()
    total = 0.0
    for i, (images, depths) in enumerate(loader):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=AMP):
            preds = forward_train(model, images)
            loss  = silog_loss(preds, depths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        if (i + 1) % LOG_INTERVAL == 0:
            global_step = (epoch - 1) * len(loader) + i
            print(f"  [{i+1}/{len(loader)}] loss={loss.item():.4f}")
            wandb.log({"train_loss_step": loss.item()}, step=global_step)
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        preds = forward_train(model, images)
        total += silog_loss(preds, depths).item()
    return total / len(loader)


def generate_pseudo_labels(train_dir: Path, cache_dir: Path, device: str) -> bool:
    model = DepthAnything3.from_pretrained(f"depth-anything/{TEACHER_MODEL}")
    model = model.to(device).eval()

    image_paths: List[Path] = sorted(train_dir.glob("*_rgb.png"))
    n_written = 0
    for p in image_paths:
        prediction = model.inference(
            image=[str(p)],
            process_res=560,
            process_res_method="upper_bound_resize"
        )

        depth = prediction.depth[0]
        assert depth.shape == (560, 560), f"Depth map size is {depth.shape} instead of 560x560"

        cache_depth = depth.astype(np.float32)
        valid = np.isfinite(cache_depth) & (cache_depth > 0)
        cache_depth = np.where(valid, cache_depth, 0.0).astype(np.float32)

        depth_name = p.name.replace("_rgb.png", "_depth.npy")
        np.save(cache_dir / depth_name, cache_depth)
        n_written += 1

    return n_written == len(image_paths)


def main():
    args = parse_args()

    if not os.path.exists(args.cache) and not args.pseudo_label:
        raise FileNotFoundError(f"Cache directory {args.cache} does not exist. "
                                "Set --pseudo-label to generate pseudo-labels first.")

    train_dir = Path(DATA_ROOT, "train")
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dataset directory {train_dir} does not exist")
    test_dir = Path(DATA_ROOT, "test")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset directory {test_dir} does not exist")

    # Init device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(f"Warning: CUDA is not available, using CPU.")
    elif "GB10" in torch.cuda.get_device_name(0):
        torch.backends.cudnn.enabled = False
        print("Disabled cuDNN backend for GB10")

    cache_dir = Path(args.cache)
    if cache_dir.exists():
        print(f"Using existing pseudo-labels from {cache_dir}")
    else:
        print(f"Generating pseudo-labels with {TEACHER_MODEL}. Saving to {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=False)
        success = generate_pseudo_labels(train_dir, cache_dir, device)
        if not success:
            raise RuntimeError("Failed to generate all pseudo-labels. Stopping...")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(SCRATCH_ROOT, f"models/baseline3-{STUDENT_MODEL}-{MODE}-{current_datetime}")
    ckpt_dir   = output_dir / "checkpoints"
    debug_dir  = output_dir / "depth_vis"
    pred_dir   = output_dir / "preds"
    for d in [ckpt_dir, debug_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"baseline3-{STUDENT_MODEL}-{MODE}-{current_datetime}",
        config={
            "teacher_model": TEACHER_MODEL,
            "student_model": STUDENT_MODEL,
            "mode": MODE,
            "img_size": IMG_SIZE,
            "train_batch": TRAIN_BATCH,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "val_split": VAL_SPLIT,
            "amp": AMP,
            "seed": SEED,
            "cache_dir": str(cache_dir),
        },
    )

    full_ds = PseudoLabelDataset(train_dir, cache_dir)
    n   = len(full_ds)
    rng = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=rng).tolist()
    val_n = int(n * VAL_SPLIT)
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=TRAIN_BATCH,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=TRAIN_BATCH,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    model = load_model(device, mode=MODE)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    scaler = GradScaler("cuda", enabled=AMP)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_metric = validate(model, val_loader, device)
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        wandb.log({"train_loss": train_loss, "val_si_rmse": val_metric, "lr": scheduler.get_last_lr()[0]}, step=epoch)

        ckpt = {"epoch": epoch, "model": model.state_dict(), "val_si_rmse": val_metric}
        torch.save(ckpt, ckpt_dir / "last.pth")
        if val_metric < best_val:
            best_val = val_metric
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  --> new best (si_rmse={best_val:.4f})")

    print(f"\nDone. Best val SI-RMSE: {best_val:.4f}")

    best = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best["model"])
    model.eval()

    image_paths: List[Path] = sorted(test_dir.glob("*_rgb.png"))
    for i in range(0, len(image_paths), INFER_BATCH):
        img_batch = image_paths[i:i+INFER_BATCH]
        predictions = model.inference(
            image=[str(p) for p in img_batch],
            process_res=560,
            process_res_method="upper_bound_resize"
        )

        for p, rgb, depth in zip(img_batch, predictions.processed_images, predictions.depth):
            debug_vis(p, rgb, depth, debug_dir)

            assert depth.shape == (560, 560), f"Depth map size is {depth.shape} instead of 560x560"

            submit_depth = depth.astype(np.float32)
            valid = np.isfinite(submit_depth) & (submit_depth > 0)
            if not np.all(valid):
                fill = np.median(submit_depth[valid]) if np.any(valid) else 1.0
                submit_depth = np.where(valid, submit_depth, fill).astype(np.float32)
            submit_depth = np.clip(submit_depth, 1e-6, None)

            pred_name = p.stem.replace("_rgb", "") + ".npy"
            np.save(pred_dir / pred_name, submit_depth)

    print(f"Predictions saved to {pred_dir}")
    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline 3 experiment to fine-tune the student on pseudo-labeled teacher outputs.")
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Folder containing pseudo-labeled training data."
    )
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="If set, the teacher model will be used to generate pseudo-labels for the student model."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

import os
import argparse
import csv
from pathlib import Path
from typing import List, Literal, cast
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
SCRATCH_ROOT = Path(os.environ.get("SCRATCH_ROOT", "/work/scratch/nmeurer"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/cluster/courses/cil/monocular-depth-estimation"))

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "DA3-GIANT-1.1")
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "DA3MONO-LARGE")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE     = 560
TRAIN_BATCH  = 8
INFER_BATCH  = 32
EPOCHS       = 3
LR           = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
VAL_SPLIT    = 0.1
NUM_WORKERS  = 2
AMP          = True
SEED         = 42
LOG_INTERVAL = 20
VAL_INTERVAL_STEPS = int(os.environ.get("VAL_INTERVAL_STEPS", "100"))

WANDB_PROJECT = "monocular-depth-estimation"
TrainMode = Literal["full_head", "surface_head", "lora_dpt_blocks"]
ValSubsetStrategy = Literal["random", "manifest"]

# ---- switch here to change training mode ----
_MODE = os.environ.get("BS3_MODE", "full_head")
if _MODE not in ("full_head", "surface_head", "lora_dpt_blocks"):
    raise ValueError(f"Invalid BS3_MODE={_MODE}")
MODE           : TrainMode = cast(TrainMode, _MODE)
WANDB_RUN_NAME = f"baseline3-{STUDENT_MODEL}-{MODE}"
# ---------------------------------------------

_VAL_SUBSET_STRATEGY = os.environ.get("VAL_SUBSET_STRATEGY", "random")
if _VAL_SUBSET_STRATEGY not in ("random", "manifest"):
    raise ValueError(f"Invalid VAL_SUBSET_STRATEGY={_VAL_SUBSET_STRATEGY}")
VAL_SUBSET_STRATEGY: ValSubsetStrategy = cast(ValSubsetStrategy, _VAL_SUBSET_STRATEGY)
VAL_MANIFEST = os.environ.get("VAL_MANIFEST", "").strip()
if VAL_INTERVAL_STEPS < 0:
    raise ValueError("VAL_INTERVAL_STEPS must be non-negative")

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
        missing = [p.name.replace("_rgb.png", "_depth.npy") for p in self.image_paths if not self._depth_path(p).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Pseudo-label cache {cache_dir} is missing {len(missing)} files. "
                f"Examples: {preview}"
            )
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


class GroundTruthDepthDataset(Dataset):
    """Ground-truth labels from train_dir, used only for validation/model selection."""

    def __init__(self, train_dir: Path, img_size: int = IMG_SIZE):
        self.image_paths = sorted(train_dir.glob("*_rgb.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No *_rgb.png images in {train_dir}")
        self.img_size = img_size
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def _depth_path(self, img_path: Path) -> Path:
        return img_path.parent / img_path.name.replace("_rgb.png", "_depth.npy")

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


def scale_invariant_rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Kaggle-style SI-RMSE: remove one log-scale bias per image, then average images."""
    valid = (
        torch.isfinite(pred)
        & torch.isfinite(target)
        & (pred > 0)
        & (target > 0)
    )
    losses = []
    for item_idx in range(pred.shape[0]):
        item_valid = valid[item_idx]
        if not torch.any(item_valid):
            continue
        log_pred = torch.log(torch.clamp(pred[item_idx][item_valid], min=eps))
        log_target = torch.log(torch.clamp(target[item_idx][item_valid], min=eps))
        diff = log_pred - log_target
        variance = torch.mean(diff ** 2) - torch.mean(diff) ** 2
        losses.append(torch.sqrt(variance.clamp(min=1e-8)))
    if not losses:
        raise RuntimeError("No valid pixels remained after masking invalid depth values.")
    return torch.stack(losses).mean()


def load_model(device: torch.device, mode: TrainMode) -> DepthAnything3:
    model = DepthAnything3.from_pretrained(f"depth-anything/{STUDENT_MODEL}")

    if mode == "full_head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True

    elif mode == "surface_head":
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only the final depth prediction stack, leaving DPT fusion blocks fixed.
        for p in model.model.head.scratch.output_conv2.parameters():
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


def random_validation_indices(n: int, val_n: int) -> List[int]:
    rng = torch.Generator().manual_seed(SEED)
    return torch.randperm(n, generator=rng).tolist()[:val_n]


def image_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rgb"):
        return stem[:-4]
    if stem.endswith("_depth"):
        return stem[:-6]
    return stem


def manifest_validation_indices(
    gt_dataset: GroundTruthDepthDataset,
    manifest_path: Path,
) -> tuple[List[int], dict[str, float]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest does not exist: {manifest_path}")

    id_to_idx = {image_id_from_path(p): i for i, p in enumerate(gt_dataset.image_paths)}
    path_to_idx = {str(p.resolve()): i for i, p in enumerate(gt_dataset.image_paths)}
    indices: List[int] = []
    seen = set()

    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Validation manifest has no header: {manifest_path}")
        for row in reader:
            idx = None
            image_id = row.get("train_id") or row.get("image_id") or row.get("id")
            if image_id:
                idx = id_to_idx.get(image_id)
            if idx is None and row.get("rgb_path"):
                rgb_path = str(Path(row["rgb_path"]).resolve())
                idx = path_to_idx.get(rgb_path)
                if idx is None:
                    idx = id_to_idx.get(image_id_from_path(Path(row["rgb_path"])))
            if idx is None:
                raise ValueError(f"Manifest row does not match a training image: {row}")
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)

    if not indices:
        raise ValueError(f"Validation manifest did not contain any usable rows: {manifest_path}")
    return indices, {"val_manifest_rows": len(indices)}


def select_validation_indices(gt_dataset: GroundTruthDepthDataset, val_n: int) -> tuple[List[int], dict[str, float]]:
    val_n = min(max(1, val_n), len(gt_dataset))
    if VAL_SUBSET_STRATEGY == "random":
        print("Selecting GT validation subset randomly.")
        return random_validation_indices(len(gt_dataset), val_n), {}
    if VAL_SUBSET_STRATEGY == "manifest":
        if not VAL_MANIFEST:
            raise ValueError("VAL_SUBSET_STRATEGY=manifest requires VAL_MANIFEST=/path/to/manifest.csv")
        print(f"Selecting GT validation subset from manifest: {VAL_MANIFEST}")
        return manifest_validation_indices(gt_dataset, Path(VAL_MANIFEST))
    raise ValueError(f"Invalid VAL_SUBSET_STRATEGY={VAL_SUBSET_STRATEGY}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    val_loader,
    ckpt_dir: Path,
    best_val: float,
) -> tuple[float, float]:
    model.train()
    if MODE in ("full_head", "surface_head"):
        model.model.backbone.eval()
    total = 0.0
    for i, (images, depths) in enumerate(loader):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=AMP):
            preds = forward_train(model, images)
            loss  = scale_invariant_rmse_loss(preds, depths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        global_step = (epoch - 1) * len(loader) + i + 1
        if (i + 1) % LOG_INTERVAL == 0:
            print(f"  [{i+1}/{len(loader)}] loss={loss.item():.4f}")
            wandb.log({"train_loss_step": loss.item(), "epoch": epoch}, step=global_step)
        if VAL_INTERVAL_STEPS > 0 and global_step % VAL_INTERVAL_STEPS == 0:
            best_val, val_metric = validate_and_checkpoint(
                model=model,
                loader=val_loader,
                device=device,
                ckpt_dir=ckpt_dir,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val,
                lr=optimizer.param_groups[0]["lr"],
                trigger="interval",
            )
            print(f"  [val step {global_step}] val_si_rmse={val_metric:.4f}")
            model.train()
            if MODE in ("full_head", "surface_head"):
                model.model.backbone.eval()
    return total / len(loader), best_val


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for images, depths in loader:
        images, depths = images.to(device), depths.to(device)
        preds = forward_train(model, images)
        total += scale_invariant_rmse_loss(preds, depths).item()
    return total / len(loader)


def validate_and_checkpoint(
    model,
    loader,
    device,
    ckpt_dir: Path,
    epoch: int,
    global_step: int,
    best_val: float,
    lr: float,
    trigger: str,
) -> tuple[float, float]:
    val_metric = validate(model, loader, device)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "val_si_rmse": val_metric,
        "val_trigger": trigger,
    }
    torch.save(ckpt, ckpt_dir / "last.pth")

    is_best = val_metric < best_val
    if is_best:
        best_val = val_metric
        torch.save(ckpt, ckpt_dir / "best.pth")
        print(f"  --> new best (si_rmse={best_val:.4f}, step={global_step}, trigger={trigger})")

    wandb.log(
        {
            "val_si_rmse": val_metric,
            f"val_si_rmse_{trigger}": val_metric,
            "best_val_si_rmse": best_val,
            "epoch": epoch,
            "lr": lr,
        },
        step=global_step,
    )
    return best_val, val_metric


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
            "val_interval_steps": VAL_INTERVAL_STEPS,
            "val_subset_strategy": VAL_SUBSET_STRATEGY,
            "val_manifest": VAL_MANIFEST,
            "amp": AMP,
            "seed": SEED,
            "cache_dir": str(cache_dir),
            "train_target": "teacher_pseudo_labels",
            "validation_target": "ground_truth_depth",
            "train_on_all_pseudo_labels": True,
        },
    )

    pseudo_train_ds = PseudoLabelDataset(train_dir, cache_dir)
    gt_val_ds = GroundTruthDepthDataset(train_dir)
    model = load_model(device, mode=MODE)

    n = len(gt_val_ds)
    val_n = max(1, int(n * VAL_SPLIT))
    val_idx, val_stats = select_validation_indices(gt_val_ds, val_n)
    wandb.config.update({
        "val_size": len(val_idx),
        "val_indices_preview": val_idx[:10],
        **val_stats,
    })
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    train_loader = DataLoader(pseudo_train_ds, batch_size=TRAIN_BATCH,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(gt_val_ds, val_idx),   batch_size=TRAIN_BATCH,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(
        f"Train pseudo-labels: {len(pseudo_train_ds)}  Val GT: {len(val_idx)} "
        f"({VAL_SUBSET_STRATEGY})"
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    scaler = GradScaler("cuda", enabled=AMP)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, best_val = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            val_loader=val_loader,
            ckpt_dir=ckpt_dir,
            best_val=best_val,
        )
        global_step = epoch * len(train_loader)
        best_val, val_metric = validate_and_checkpoint(
            model=model,
            loader=val_loader,
            device=device,
            ckpt_dir=ckpt_dir,
            epoch=epoch,
            global_step=global_step,
            best_val=best_val,
            lr=optimizer.param_groups[0]["lr"],
            trigger="epoch_end",
        )
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        wandb.log(
            {"train_loss": train_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch},
            step=global_step,
        )

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

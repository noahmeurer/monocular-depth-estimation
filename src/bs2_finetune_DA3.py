from pathlib import Path
from typing import List, Literal
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

from depth_anything_3.api import DepthAnything3


### Configs
SCRATCH_ROOT    = Path("/work/scratch/cdeubel")
TRAIN_DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/train")
TEST_DATA_ROOT  = Path("/cluster/courses/cil/monocular-depth-estimation/test")
DATA_ROOT       = TEST_DATA_ROOT

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE     = 560
TRAIN_BATCH  = 8
INFER_BATCH  = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0
VAL_SPLIT    = 0.1
NUM_WORKERS  = 4
AMP          = True
SEED         = 42
LOG_INTERVAL = 20


lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    init_lora_weights="gaussian",
    dropout=0.1,
    bias="none",
    # TODO: specify target modules
    target_modules=[],
)
###

def debug_vis(path: Path, rgb: np.ndarray, depth: np.ndarray, debug_dir: Path):
    """
    Visualize a depth map side-by-side with it's original (processed) image 
    for visually debugging model predictions.

    Inputs:
        path: Path          unprocessed image path
        rgb: np.ndarray     processed rgb image [H, W, 3]
        depth: np.ndarray   predicted depth map [H, W]
        debug_dir: Path     save directory

    Outputs:
        None
    """
    valid = np.isfinite(depth)
    if not np.any(valid):
        print("Warning (debug_vis): No valid depth values found")
        return

    vis_depth = depth.copy()

    # TODO: Optional add sky and conf based masking for visualization

    # Normalize the display range to the 1st and 99th percentiles for a robust (not flat) colormap
    lo, hi = np.percentile(vis_depth[valid], [1, 99])
    depth_norm = np.clip((vis_depth - lo) / max(hi - lo, 1e-6), 0, 1)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 1))  # NaNs render black
    depth_rgb = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    # Save debugging image with a side-by-side of the original (processed) rgb image and the visualized predicted depth map
    gap_width = 12
    gap = np.full((rgb.shape[0], gap_width, 3), 255, dtype=np.uint8)

    side_by_side = np.concatenate(
        [rgb.astype(np.uint8), gap, depth_rgb.astype(np.uint8)],
        axis=1,
    )

    out_name = path.stem.replace("_rgb", "_depth_vis") + ".png"
    Image.fromarray(side_by_side).save(debug_dir / out_name)


class DepthDataset(Dataset):
    def __init__(self, data_dir: Path, img_size: int = IMG_SIZE):
        self.image_paths = sorted(data_dir.glob("*_rgb.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No *_rgb.png images in {data_dir}")
        self.img_size = img_size
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def _depth_path(self, img_path: Path) -> Path:
        return img_path.parent / (img_path.stem.replace("_rgb", "_depth") + ".npy")

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


# Model helpers
def load_model(device: torch.device, mode: Literal["full_head", "lora_head"]) -> DepthAnything3:
    model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

    if mode == "full_head":
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze head only
        for p in model.depth_head.parameters():
            p.requires_grad = True

    elif mode == "lora_head":
        model = get_peft_model(model, lora_config)

    else:
        raise ValueError(f"Invalid mode {mode}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"Trainable: {n_train:,} / {n_total:,} params")

    return model.to(device)


def forward_train(model: DepthAnything3, images: torch.Tensor) -> torch.Tensor:
    # TODO: verify raw forward call on cluster — may differ depending on DA3 internals
    # Likely options if this fails:
    #   out = model.model(pixel_values=images)   # if DepthAnything3 wraps a HF model
    #   out = model.forward(images)
    out = model(images)
    depth = out.predicted_depth if hasattr(out, "predicted_depth") else out
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    return F.interpolate(depth, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).clamp(min=1e-3)


# ---------- Train / validate ----------

def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
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
            print(f"  [{i+1}/{len(loader)}] loss={loss.item():.4f}")
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Setup input and output directories
    input_dir = DATA_ROOT
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
    output_dir = Path(SCRATCH_ROOT, "outputs/baseline2")
    ckpt_dir   = output_dir / "checkpoints"
    debug_dir  = output_dir / "depth_vis"
    pred_dir   = output_dir / "preds"
    for d in [ckpt_dir, debug_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Init device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(f"Warning: CUDA is not available, using CPU.")


    # Dataset split
    if not TRAIN_DATA_ROOT.exists():
        raise FileNotFoundError(f"Train dir not found: {TRAIN_DATA_ROOT}")
    full_ds = DepthDataset(TRAIN_DATA_ROOT)
    val_ds  = DepthDataset(TRAIN_DATA_ROOT)
    n   = len(full_ds)
    rng = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=rng).tolist()
    val_n = int(n * VAL_SPLIT)
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=TRAIN_BATCH,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,  val_idx),   batch_size=TRAIN_BATCH,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")


    # Initialize the DepthAnything3 single-view monocular student model
    model = load_model(device, mode="lora_head")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    scaler = GradScaler("cuda", enabled=AMP)


    # Training loop
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_metric = validate(model, val_loader, device)
        scheduler.step()
        print(f"  train_loss={train_loss:.4f}  val_si_rmse={val_metric:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {"epoch": epoch, "model": model.state_dict(), "val_si_rmse": val_metric}
        torch.save(ckpt, ckpt_dir / "last.pth")
        if val_metric < best_val:
            best_val = val_metric
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  --> new best (si_rmse={best_val:.4f})")

    print(f"\nDone. Best val SI-RMSE: {best_val:.4f}")

    # Load best weights for inference
    best = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best["model"])
    model.eval()

    # Inference on test set
    if not TEST_DATA_ROOT.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DATA_ROOT}")

    # Merge LoRA weights into base model so .inference() is available
    model = model.merge_and_unload()
    model.eval()

    image_paths: List[Path] = sorted(input_dir.glob("*_rgb.png"))
    for i in range(0, len(image_paths), INFER_BATCH):
        img_batch = image_paths[i:i+INFER_BATCH]
        predictions = model.inference(
            image=[str(p) for p in img_batch], 
            process_res=560, 
            process_res_method="upper_bound_resize"
        )

        # Process model outputs
        for p, rgb, depth in zip(img_batch, predictions.processed_images, predictions.depth):
            # Visualize depth map for debugging
            debug_vis(p, rgb, depth, debug_dir)

            # Assert depth map size is 560x560
            assert depth.shape == (560, 560), f"Depth map size is {depth.shape} instead of 560x560"

            # Save depth map for baseline evaluation
            submit_depth = depth.astype(np.float32)
            valid = np.isfinite(submit_depth) & (submit_depth > 0)
            if not np.all(valid):
                fill = np.median(submit_depth[valid]) if np.any(valid) else 1.0
                submit_depth = np.where(valid, submit_depth, fill).astype(np.float32)
            submit_depth = np.clip(submit_depth, 1e-6, None)

            pred_name = p.stem.replace("_rgb", "") + ".npy"
            np.save(pred_dir / pred_name, submit_depth)

    print(f"Predictions saved to {pred_dir}")
if __name__ == "__main__":
    main()
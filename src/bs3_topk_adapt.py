from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import wandb

from bs3_pseudo_label_DA3_train import (
    AMP,
    DATA_ROOT,
    GRAD_CLIP,
    IMG_SIZE,
    MODE,
    NUM_WORKERS,
    PseudoLabelDataset,
    SCRATCH_ROOT,
    SEED,
    STUDENT_MODEL,
    TEACHER_MODEL,
    TrainMode,
    forward_train,
    image_id_from_path,
    load_model,
    scale_invariant_rmse_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a Baseline3 checkpoint and adapt on the top-k target-similar pseudo-labeled train images."
    )
    parser.add_argument("--cache", type=Path, required=True, help="Directory with teacher pseudo-label *_depth.npy files.")
    parser.add_argument("--resume", type=Path, required=True, help="Checkpoint to resume from.")
    parser.add_argument("--manifest", type=Path, required=True, help="Similarity-ranked train manifest CSV.")
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("ADAPT_TOP_K", "250")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("ADAPT_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ADAPT_BATCH", "8")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("ADAPT_LR", "1e-8")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("ADAPT_WEIGHT_DECAY", "1e-4")))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["full_head", "surface_head", "lora_dpt_blocks"], default=MODE)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "monocular-depth-estimation"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_RUN_NAME", ""))
    return parser.parse_args()


def read_topk_manifest_rows(manifest_path: Path, top_k: int) -> list[dict[str, str]]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        for row in reader:
            rows.append(row)
            if len(rows) >= top_k:
                break

    if len(rows) < top_k:
        raise ValueError(f"Manifest has only {len(rows)} rows, but top_k={top_k}")
    return rows


def manifest_rows_to_indices(dataset: PseudoLabelDataset, rows: list[dict[str, str]]) -> List[int]:
    id_to_idx = {image_id_from_path(p): i for i, p in enumerate(dataset.image_paths)}
    path_to_idx = {str(p.resolve()): i for i, p in enumerate(dataset.image_paths)}

    indices: List[int] = []
    seen = set()
    for row in rows:
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
            raise ValueError(f"Manifest row does not match a pseudo-label training image: {row}")
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)

    if len(indices) != len(rows):
        raise ValueError(f"Expected {len(rows)} unique train images but found {len(indices)}")
    return indices


def write_selected_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    mode: TrainMode,
) -> tuple[float, int]:
    model.train()
    if mode in ("full_head", "surface_head"):
        model.model.backbone.eval()

    total = 0.0
    steps = 0
    for i, (images, depths) in enumerate(loader, start=1):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=AMP):
            preds = forward_train(model, images)
            loss = scale_invariant_rmse_loss(preds, depths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        steps += 1
        wandb.log({"train_loss_step": loss.item(), "epoch": epoch}, step=(epoch - 1) * len(loader) + i)
        if i == 1 or i % 10 == 0 or i == len(loader):
            print(f"  [{i}/{len(loader)}] loss={loss.item():.4f}")

    return total / max(1, steps), steps


def save_checkpoint(
    model,
    ckpt_dir: Path,
    epoch: int,
    global_step: int,
    train_loss: float,
    args: argparse.Namespace,
    source_meta: dict,
) -> Path:
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "train_loss": train_loss,
        "lr": args.lr,
        "top_k": args.top_k,
        "source_checkpoint": str(args.resume),
        "source_checkpoint_epoch": source_meta.get("epoch"),
        "source_checkpoint_global_step": source_meta.get("global_step"),
        "source_checkpoint_val_si_rmse": source_meta.get("val_si_rmse"),
        "source_checkpoint_val_trigger": source_meta.get("val_trigger"),
        "train_target": "teacher_pseudo_labels_topk",
        "validation_target": "none",
    }
    epoch_path = ckpt_dir / f"epoch_{epoch:02d}.pth"
    torch.save(ckpt, epoch_path)
    torch.save(ckpt, ckpt_dir / "last.pth")
    return epoch_path


def main() -> None:
    args = parse_args()
    if not args.cache.exists():
        raise FileNotFoundError(f"Pseudo-label cache not found: {args.cache}")
    if not args.resume.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    train_dir = DATA_ROOT / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dataset directory not found: {train_dir}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available, using CPU.")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (
        SCRATCH_ROOT / f"models/baseline3-top{args.top_k}-adapt-{STUDENT_MODEL}-{args.mode}-{current_datetime}"
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    print(f"Output dir: {output_dir}")
    print(f"Resume checkpoint: {args.resume}")
    print(f"Pseudo-label cache: {args.cache}")
    print(f"Manifest: {args.manifest}")
    print(f"Top-k: {args.top_k}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr:.2e}")

    pseudo_ds = PseudoLabelDataset(train_dir, args.cache)
    rows = read_topk_manifest_rows(args.manifest, args.top_k)
    indices = manifest_rows_to_indices(pseudo_ds, rows)
    write_selected_manifest(rows, output_dir / f"top{args.top_k}_manifest.csv")
    train_ds = Subset(pseudo_ds, indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    mode: TrainMode = args.mode  # type: ignore[assignment]
    model = load_model(device, mode=mode)
    source_ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(source_ckpt["model"])
    source_meta = {
        "epoch": source_ckpt.get("epoch"),
        "global_step": source_ckpt.get("global_step"),
        "val_si_rmse": source_ckpt.get("val_si_rmse"),
        "val_trigger": source_ckpt.get("val_trigger"),
    }
    (output_dir / "source_checkpoint_metadata.json").write_text(json.dumps(source_meta, indent=2) + "\n")

    run_name = args.wandb_name or f"baseline3-top{args.top_k}-adapt-{STUDENT_MODEL}-{mode}-{current_datetime}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "experiment": "baseline3_topk_adapt",
            "teacher_model": TEACHER_MODEL,
            "student_model": STUDENT_MODEL,
            "mode": mode,
            "img_size": IMG_SIZE,
            "train_batch": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": GRAD_CLIP,
            "amp": AMP,
            "seed": SEED,
            "cache_dir": str(args.cache),
            "resume": str(args.resume),
            "source_checkpoint": source_meta,
            "manifest": str(args.manifest),
            "top_k": args.top_k,
            "train_size": len(train_ds),
            "validation_target": "none",
            "train_target": "teacher_pseudo_labels_topk",
        },
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=AMP)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, steps = train_epoch(model, train_loader, optimizer, scaler, device, epoch, mode)
        global_step += steps
        epoch_path = save_checkpoint(model, ckpt_dir, epoch, global_step, train_loss, args, source_meta)
        print(f"  train_loss={train_loss:.4f}  lr={args.lr:.2e}  saved={epoch_path}")
        wandb.log({"train_loss": train_loss, "lr": args.lr, "epoch": epoch}, step=global_step)

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cache": str(args.cache),
                "resume": str(args.resume),
                "manifest": str(args.manifest),
                "top_k": args.top_k,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "student_model": STUDENT_MODEL,
                "teacher_model": TEACHER_MODEL,
                "mode": mode,
                "final_checkpoint": str(ckpt_dir / "last.pth"),
            },
            indent=2,
        )
        + "\n"
    )
    wandb.finish()
    print(f"\nDone. Final checkpoint: {ckpt_dir / 'last.pth'}")


if __name__ == "__main__":
    main()

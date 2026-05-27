#!/usr/bin/env python3
"""Debug script for baseline3 pseudo-label generation."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(depth[valid], [1, 99])
    depth_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    depth_norm = np.where(valid, depth_norm, np.nan)
    return (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)


def make_panel(rgb_path: Path, gt_path: Path, pseudo_path: Path, out_path: Path) -> None:
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    gt = np.load(gt_path).astype(np.float32)
    pseudo = np.load(pseudo_path).astype(np.float32)

    gt_rgb = depth_to_rgb(gt)
    pseudo_rgb = depth_to_rgb(pseudo)

    h, w = rgb.shape[:2]
    if gt_rgb.shape[:2] != (h, w):
        gt_rgb = np.array(Image.fromarray(gt_rgb).resize((w, h), Image.NEAREST))
    if pseudo_rgb.shape[:2] != (h, w):
        pseudo_rgb = np.array(Image.fromarray(pseudo_rgb).resize((w, h), Image.NEAREST))

    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate([rgb, gap, gt_rgb, gap, pseudo_rgb], axis=1)
    Image.fromarray(panel).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview pseudo labels vs RGB and GT depth.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/cluster/courses/cil/monocular-depth-estimation/train"),
        help="Directory with *_rgb.png and *_depth.npy ground truth files.",
    )
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path("/work/scratch/mdealvaro/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1"),
        help="Directory with generated pseudo *_depth.npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/work/scratch/mdealvaro/outputs/baseline3/pseudo_preview"),
        help="Directory where side-by-side PNG previews are written.",
    )
    parser.add_argument("--num", type=int, default=10, help="Number of samples to visualize.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(args.train_dir.glob("*_rgb.png"))[: args.num]
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png files found in {args.train_dir}")

    written = 0
    for rgb_path in rgb_files:
        gt_path = args.train_dir / rgb_path.name.replace("_rgb.png", "_depth.npy")
        pseudo_path = args.pseudo_dir / rgb_path.name.replace("_rgb.png", "_depth.npy")
        if not gt_path.exists() or not pseudo_path.exists():
            continue

        out_name = rgb_path.name.replace("_rgb.png", "_preview.png")
        make_panel(rgb_path, gt_path, pseudo_path, args.out_dir / out_name)
        written += 1

    print(f"Wrote {written} preview images to: {args.out_dir}")
    print("Panel order: RGB | GT depth | Pseudo depth")


if __name__ == "__main__":
    main()

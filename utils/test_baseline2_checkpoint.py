from pathlib import Path
from typing import List
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3

import base64
import zlib
import pandas as pd


def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    return base64.b64encode(compressed).decode("utf-8")


### Configs
SCRATCH_ROOT   = Path("/work/scratch/cdeubel")
TEST_DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/test")

CKPT_PATH      = SCRATCH_ROOT / "outputs/baseline2/checkpoints/best.pth"
OUTPUT_DIR     = SCRATCH_ROOT / "outputs/test_baseline2"
PRED_DIR       = OUTPUT_DIR / "preds"
DEBUG_DIR      = OUTPUT_DIR / "depth_vis"
SUBMISSION_CSV = OUTPUT_DIR / "baseline2_submission.csv"

INFER_BATCH = 32
###


def debug_vis(path: Path, rgb: np.ndarray, depth: np.ndarray, debug_dir: Path):
    valid = np.isfinite(depth)
    if not np.any(valid):
        print(f"Warning (debug_vis): no valid depth values for {path.name}")
        return
    lo, hi = np.percentile(depth[valid], [1, 99])
    depth_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    depth_rgb = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)
    gap = np.full((rgb.shape[0], 12, 3), 255, dtype=np.uint8)
    side_by_side = np.concatenate([rgb.astype(np.uint8), gap, depth_rgb], axis=1)
    Image.fromarray(side_by_side).save(debug_dir / (path.stem.replace("_rgb", "_depth_vis") + ".png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    args = parser.parse_args()

    for d in [PRED_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, using CPU.")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print(f"Loading model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_si_rmse={ckpt['val_si_rmse']:.4f})")

    if not TEST_DATA_ROOT.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DATA_ROOT}")

    image_paths: List[Path] = sorted(TEST_DATA_ROOT.glob("*_rgb.png"))
    print(f"Running inference on {len(image_paths)} test images...")

    for i in range(0, len(image_paths), INFER_BATCH):
        img_batch = image_paths[i:i + INFER_BATCH]
        predictions = model.inference(
            image=[str(p) for p in img_batch],
            process_res=560,
            process_res_method="upper_bound_resize",
        )
        for p, rgb, depth in zip(img_batch, predictions.processed_images, predictions.depth):
            debug_vis(p, rgb, depth, DEBUG_DIR)

            assert depth.shape == (560, 560), f"Depth shape {depth.shape} != (560, 560)"

            submit_depth = depth.astype(np.float32)
            valid = np.isfinite(submit_depth) & (submit_depth > 0)
            if not np.all(valid):
                fill = np.median(submit_depth[valid]) if np.any(valid) else 1.0
                submit_depth = np.where(valid, submit_depth, fill).astype(np.float32)
            submit_depth = np.clip(submit_depth, 1e-6, None)

            np.save(PRED_DIR / (p.stem.replace("_rgb", "") + ".npy"), submit_depth)

        print(f"  [{min(i + INFER_BATCH, len(image_paths))}/{len(image_paths)}]")

    # Build submission CSV
    rows = []
    for pred_path in sorted(PRED_DIR.glob("test_*.npy")):
        idx = pred_path.stem.split("_")[-1]
        rows.append({"id": f"test_{idx}_depth", "Depths": encode_depth(np.load(pred_path))})

    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission saved to {SUBMISSION_CSV} ({len(df)} predictions)")


if __name__ == "__main__":
    main()

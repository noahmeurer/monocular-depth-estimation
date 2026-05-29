from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from depth_anything_3.api import DepthAnything3

### Configs
SCRATCH_ROOT = Path("/work/scratch/nmeurer")
DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/test")

TEACHER_MODEL = "DA3-GIANT-1.1"
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

def main():
    # Setup input and output directories
    input_dir = DATA_ROOT
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
    output_dir = Path(SCRATCH_ROOT, "outputs/baseline_teacher")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(output_dir, "depth_vis")
    debug_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Init device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(f"Warning: CUDA is not available, using CPU.")

    # Initialize the DepthAnything3 single-view monocular student model
    model = DepthAnything3.from_pretrained(f"depth-anything/{TEACHER_MODEL}")
    model = model.to(device).eval()

    # Do zero-shot depth prediction on images
    image_paths: List[Path] = sorted(input_dir.glob("*_rgb.png"))
    for p in image_paths:
        prediction = model.inference(
            image=[str(p)], 
            process_res=560, 
            process_res_method="upper_bound_resize"
        )

        # Process model outputs
        rgb = prediction.processed_images[0]
        depth = prediction.depth[0]

        # Assert depth map size is 560x560
        assert depth.shape == (560, 560), f"Depth map size is {depth.shape} instead of 560x560"
            
        # Visualize depth map for debugging
        debug_vis(p, rgb, depth, debug_dir)

        # Save depth map for baseline evaluation
        submit_depth = depth.astype(np.float32)
        valid = np.isfinite(submit_depth) & (submit_depth > 0)
        if not np.all(valid):
            fill = np.median(submit_depth[valid]) if np.any(valid) else 1.0
            submit_depth = np.where(valid, submit_depth, fill).astype(np.float32)
        submit_depth = np.clip(submit_depth, 1e-6, None)

        pred_name = p.stem.replace("_rgb", "") + ".npy"
        np.save(pred_dir / pred_name, submit_depth)

if __name__ == "__main__":
    main()

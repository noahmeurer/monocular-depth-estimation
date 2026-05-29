import os
import argparse
from pathlib import Path
from typing import List
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3

### Configs
SCRATCH_ROOT = Path("/work/scratch/nmeurer")
DATA_ROOT = Path("/cluster/courses/cil/monocular-depth-estimation")

TEACHER_MODEL = "DA3NESTED-GIANT-LARGE-1.1"
STUDENT_MODEL = "DA3MONO-LARGE"
###

def main():
    args = parse_args()

    if not os.path.exists(args.cache) and not args.pseudo_label:
        raise FileNotFoundError(f"Cache directory {args.cache} does not exist. Did you already generate the pseudo-labels? "
                                "If not, set the --pseudo-label flag to generate them. "
                                "Otherwise, provide the correct path to the cache directory.")
    if args.pseudo_label and os.path.exists(args.cache):
        raise FileExistsError(f"Cache directory {args.cache} already exists. "
                              "To generate new pseudo-labels, specify a new directory or remove the existing one.")

    # Setup data directories
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
    
    # Setup cache directory
    cache_dir = Path(args.cache)
    if args.pseudo_label:
        # Generate pseudo-labels
        print(f"Generating fresh pseudo-labels with {TEACHER_MODEL}. Saving to {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=False)
        success = generate_pseudo_labels(train_dir, cache_dir, device)
        if not success:
            raise RuntimeError("Failed to generate all pseudo-labels. Stopping...")
    else:
        print(f"Using cached pseudo-labels from {cache_dir}")

    # Setup output directory
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(SCRATCH_ROOT, f"models/baseline3-{STUDENT_MODEL}-{current_datetime}")
    output_dir.mkdir(parents=True, exist_ok=False)

    # Fine-tune the student model on the pseudo-labeled data
    # TODO: Implement fine-tuning

def generate_pseudo_labels(train_dir: Path, cache_dir: Path, device: str) -> bool:
    # Initialize the DepthAnything3 single-view monocular student model
    model = DepthAnything3.from_pretrained(f"depth-anything/{TEACHER_MODEL}")
    model = model.to(device).eval()

    # Do zero-shot depth prediction on images
    image_paths: List[Path] = sorted(train_dir.glob("*_rgb.png"))
    n_written = 0
    for p in image_paths:
        prediction = model.inference(
            image=[str(p)], 
            process_res=560, 
            process_res_method="upper_bound_resize"
        )

        # Process model outputs
        depth = prediction.depth[0]

        # Assert depth map size is 560x560
        assert depth.shape == (560, 560), f"Depth map size is {depth.shape} instead of 560x560"

        # Save depth map to cache directory
        # - float32 .npy
        # - filename paired with *_rgb.png -> *_depth.npy
        # - invalid values set to 0 so the training mask (depth > 0) ignores them
        cache_depth = depth.astype(np.float32)
        valid = np.isfinite(cache_depth) & (cache_depth > 0)
        cache_depth = np.where(valid, cache_depth, 0.0).astype(np.float32)

        depth_name = p.name.replace("_rgb.png", "_depth.npy")
        np.save(cache_dir / depth_name, cache_depth)
        n_written += 1

    return n_written == len(image_paths)

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

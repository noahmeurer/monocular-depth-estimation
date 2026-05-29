from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from depth_anything_3.api import DepthAnything3


def debug_vis(path: Path, rgb: np.ndarray, depth: np.ndarray, debug_dir: Path) -> None:
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
    out_name = path.stem.replace("_rgb", "_depth_vis") + ".png"
    Image.fromarray(side_by_side).save(debug_dir / out_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DA3MONO test inference from a saved training checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("/cluster/courses/cil/monocular-depth-estimation/test"),
    )
    parser.add_argument("--model", default="DA3MONO-LARGE")
    parser.add_argument("--infer-batch", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--debug-vis-limit", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {args.test_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type != "cuda":
        print("Warning: CUDA is not available, using CPU.")

    pred_dir = args.output_dir / "preds"
    debug_dir = args.output_dir / "depth_vis"
    pred_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading depth-anything/{args.model}")
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(
        "Loaded checkpoint "
        f"epoch={ckpt.get('epoch')} "
        f"global_step={ckpt.get('global_step')} "
        f"val_si_rmse={ckpt.get('val_si_rmse')}"
    )

    image_paths: List[Path] = sorted(args.test_dir.glob("*_rgb.png"))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    print(f"Running inference on {len(image_paths)} test images")

    written = 0
    with torch.inference_mode():
        for i in range(0, len(image_paths), args.infer_batch):
            img_batch = image_paths[i:i + args.infer_batch]
            predictions = model.inference(
                image=[str(p) for p in img_batch],
                process_res=560,
                process_res_method="upper_bound_resize",
            )

            for p, rgb, depth in zip(img_batch, predictions.processed_images, predictions.depth):
                assert depth.shape == (560, 560), f"Depth shape {depth.shape} != (560, 560)"

                if written < args.debug_vis_limit:
                    debug_vis(p, rgb, depth, debug_dir)

                submit_depth = depth.astype(np.float32)
                valid = np.isfinite(submit_depth) & (submit_depth > 0)
                if not np.all(valid):
                    fill = np.median(submit_depth[valid]) if np.any(valid) else 1.0
                    submit_depth = np.where(valid, submit_depth, fill).astype(np.float32)
                submit_depth = np.clip(submit_depth, 1e-6, None)

                pred_name = p.stem.replace("_rgb", "") + ".npy"
                np.save(pred_dir / pred_name, submit_depth)
                written += 1

            print(f"  [{min(i + args.infer_batch, len(image_paths))}/{len(image_paths)}]")

    metadata = {
        "checkpoint": str(args.ckpt),
        "model": args.model,
        "test_dir": str(args.test_dir),
        "pred_dir": str(pred_dir),
        "debug_dir": str(debug_dir),
        "num_predictions": written,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_global_step": ckpt.get("global_step"),
        "checkpoint_val_si_rmse": ckpt.get("val_si_rmse"),
        "checkpoint_val_trigger": ckpt.get("val_trigger"),
    }
    (args.output_dir / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Predictions saved to {pred_dir}")


if __name__ == "__main__":
    main()

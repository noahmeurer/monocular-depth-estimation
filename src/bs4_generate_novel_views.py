from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from gsplat import rasterization

from depth_anything_3.api import DepthAnything3


DEFAULT_TRAIN_DIR = Path("/cluster/courses/cil/monocular-depth-estimation/train")
DEFAULT_MANIFEST = Path(
    "/work/scratch/nmeurer/outputs/baseline3/feature_cache/"
    "da3_backbone_DA3MONO-LARGE/validation_subsets/cosine/val_top10pct.csv"
)
DEFAULT_PSEUDO_DIR = Path("/work/scratch/nmeurer/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1")
DEFAULT_SCRATCH_ROOT = Path(os.environ.get("SCRATCH_ROOT", "/work/scratch/nmeurer"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Baseline4 novel-view RGB/depth pairs for the top-k cosine-similar train images."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--teacher-model", default="DA3-GIANT-1.1")
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "250")))
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("MAX_IMAGES", "0")))
    parser.add_argument("--process-res", type=int, default=560)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--delta-fraction", type=float, default=0.05)
    parser.add_argument("--keep-percent", type=float, default=50.0)
    parser.add_argument("--scene-conf-percentile", type=float, default=30.0)
    parser.add_argument("--preview-limit", type=int, default=int(os.environ.get("PREVIEW_LIMIT", "12")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rgb"):
        return stem[:-4]
    if stem.endswith("_depth"):
        return stem[:-6]
    return stem


def model_repo_id(model_name: str) -> str:
    return model_name if "/" in model_name else f"depth-anything/{model_name}"


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def read_top_manifest_rows(path: Path, top_k: int, max_images: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    limit = min(top_k, max_images) if max_images > 0 else top_k

    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        for row in reader:
            method = (row.get("method") or "").strip()
            if method and method != "cosine":
                raise ValueError(f"Expected cosine manifest rows, got method={method!r}")
            rows.append(row)
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"Manifest has only {len(rows)} usable rows, expected {limit}")
    return rows


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(up_hint, f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    d /= np.linalg.norm(d)
    return np.stack([r, d, f], axis=1)


def c2w_to_w2c_4x4(r_c2w: np.ndarray, c: np.ndarray) -> np.ndarray:
    r_w2c = r_c2w.T
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r_w2c
    mat[:3, 3] = -r_w2c @ c
    return mat


def build_shifted_cameras(
    extr_w2c_3x4: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    delta_fraction: float,
    scene_conf_percentile: float,
) -> dict[str, np.ndarray]:
    r_w2c = extr_w2c_3x4[:, :3]
    t_w2c = extr_w2c_3x4[:, 3]
    r_c2w = r_w2c.T
    c_w = -r_w2c.T @ t_w2c

    right_w = r_c2w[:, 0]
    down_w = r_c2w[:, 1]
    forward_w = r_c2w[:, 2]
    up_w = -down_w

    finite_conf = conf[np.isfinite(conf)]
    if finite_conf.size == 0:
        raise RuntimeError("No finite DA3 confidence values for scene-depth estimation.")
    conf_cutoff = np.percentile(finite_conf, scene_conf_percentile)
    valid = (conf > conf_cutoff) & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise RuntimeError("No valid positive DA3 depth values for scene-depth estimation.")

    scene_depth = float(np.median(depth[valid]))
    target = c_w + scene_depth * forward_w
    delta = delta_fraction * scene_depth
    up_hint = down_w

    shifts = {
        "left": (-delta, 0.0),
        "right": (delta, 0.0),
        "up": (0.0, delta),
        "down": (0.0, -delta),
    }
    cameras: dict[str, np.ndarray] = {}
    for label, (dx, dy) in shifts.items():
        c_new = c_w + dx * right_w + dy * up_w
        r_c2w_new = look_at_opencv(c_new, target, up_hint)
        cameras[label] = c2w_to_w2c_4x4(r_c2w_new, c_new).astype(np.float32)
    return cameras


def splat_scalar(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    extr: np.ndarray,
    k_t: torch.Tensor,
    values_3ch: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    img, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=values_3ch,
        sh_degree=None,
        viewmats=torch.from_numpy(extr).float().to(device).unsqueeze(0),
        Ks=k_t,
        width=width,
        height=height,
    )
    return img[0, ..., 0], alpha[0, ..., 0]


def splat_rgb(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors_sh: torch.Tensor,
    extr: np.ndarray,
    k_t: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    img, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_sh,
        sh_degree=2,
        viewmats=torch.from_numpy(extr).float().to(device).unsqueeze(0),
        Ks=k_t,
        width=width,
        height=height,
    )
    return img[0].clamp(0, 1), alpha[0, ..., 0]


def camera_space_depth_values(means: torch.Tensor, extr: np.ndarray, device: torch.device) -> torch.Tensor:
    r_w2c = torch.from_numpy(extr[:3, :3]).float().to(device)
    t_w2c = torch.from_numpy(extr[:3, 3]).float().to(device)
    z = (means @ r_w2c.T)[:, 2] + t_w2c[2]
    return z.unsqueeze(-1).expand(-1, 3).contiguous()


def sanitize_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    return np.where(valid, depth, 0.0).astype(np.float32), valid


def render_novel_views(
    prediction,
    delta_fraction: float,
    keep_percent: float,
    scene_conf_percentile: float,
    device: torch.device,
) -> list[dict[str, object]]:
    extr_w2c = as_numpy(prediction.extrinsics[0]).astype(np.float64)
    k = as_numpy(prediction.intrinsics[0]).astype(np.float32)
    depth = as_numpy(prediction.depth[0]).astype(np.float32)
    conf = as_numpy(prediction.conf[0]).astype(np.float32)
    height, width = depth.shape

    cameras = build_shifted_cameras(extr_w2c, depth, conf, delta_fraction, scene_conf_percentile)

    gaussians = prediction.gaussians
    means = gaussians.means[0].to(device).contiguous()
    quats = gaussians.rotations[0].to(device).contiguous()
    scales = gaussians.scales[0].to(device).contiguous()
    opacities = gaussians.opacities[0].to(device).contiguous()
    colors_sh = gaussians.harmonics[0].permute(0, 2, 1).to(device).contiguous()

    if conf.size != means.shape[0]:
        raise RuntimeError(f"Confidence has {conf.size} values but DA3 returned {means.shape[0]} Gaussians.")

    conf_per_g = torch.from_numpy(conf.reshape(-1)).float().to(device)
    conf_color = conf_per_g.unsqueeze(-1).expand(-1, 3).contiguous()
    k_t = torch.from_numpy(k).float().to(device).unsqueeze(0)

    results: list[dict[str, object]] = []
    for label, extr in cameras.items():
        rgb_t, _ = splat_rgb(means, quats, scales, opacities, colors_sh, extr, k_t, width, height, device)
        conf_img, alpha_c = splat_scalar(
            means, quats, scales, opacities, extr, k_t, conf_color, width, height, device
        )
        depth_img, _ = splat_scalar(
            means,
            quats,
            scales,
            opacities,
            extr,
            k_t,
            camera_space_depth_values(means, extr, device),
            width,
            height,
            device,
        )

        conf_norm = conf_img / alpha_c.clamp(min=1e-6)
        trust = (alpha_c * conf_norm).detach().cpu().numpy()
        depth_np = depth_img.detach().cpu().numpy().astype(np.float32)
        rgb_np = (rgb_t.detach().cpu().numpy() * 255).astype(np.uint8)

        valid_depth = np.isfinite(depth_np) & (depth_np > 0) & np.isfinite(trust)
        if np.any(valid_depth):
            thresh = float(np.percentile(trust[valid_depth], keep_percent))
            mask = valid_depth & (trust >= thresh)
        else:
            thresh = float("nan")
            mask = np.zeros_like(depth_np, dtype=bool)
        masked_depth = np.where(mask, depth_np, 0.0).astype(np.float32)

        results.append(
            {
                "view": label,
                "rgb": rgb_np,
                "depth": masked_depth,
                "mask": mask,
                "trust_threshold": thresh,
                "kept_fraction": float(mask.mean()),
                "kept_valid_fraction": float(mask.sum() / max(1, int(valid_depth.sum()))),
            }
        )
    return results


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [1, 99])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    norm = np.where(valid, norm, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


def make_preview(sample_rows: Iterable[dict[str, object]], out_path: Path) -> None:
    tiles = []
    for row in sample_rows:
        rgb = np.array(Image.open(row["rgb_path"]).convert("RGB"))
        depth = np.load(row["depth_path"]).astype(np.float32)
        label = str(row["view"])
        depth_rgb = depth_to_rgb(depth)

        tile = np.concatenate([rgb, depth_rgb], axis=1)
        label_band = np.full((28, tile.shape[1], 3), 255, dtype=np.uint8)
        label_img = Image.fromarray(label_band)
        ImageDraw.Draw(label_img).text((8, 7), label, fill=(0, 0, 0))
        label_band = np.array(label_img)
        tiles.append(np.concatenate([label_band, tile], axis=0))

    if not tiles:
        return
    gap = np.full((8, tiles[0].shape[1], 3), 255, dtype=np.uint8)
    panel = tiles[0]
    for tile in tiles[1:]:
        panel = np.concatenate([panel, gap, tile], axis=0)
    Image.fromarray(panel).save(out_path)


def save_rgb(path: Path, image: np.ndarray | Image.Image, size: int) -> None:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB").resize((size, size), Image.LANCZOS)
    else:
        pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if pil.size != (size, size):
            pil = pil.resize((size, size), Image.LANCZOS)
    pil.save(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DA3-GIANT + gsplat novel-view generation.")
    if not args.train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {args.train_dir}")
    if not args.pseudo_dir.exists():
        raise FileNotFoundError(f"Pseudo-label dir not found: {args.pseudo_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            DEFAULT_SCRATCH_ROOT
            / "outputs/baseline4"
            / f"novel_views_DA3-GIANT-1.1_cosine_top{args.top_k}_{stamp}"
        )
    if output_dir.exists() and args.overwrite:
        raise ValueError("--overwrite is intentionally not destructive; choose a fresh --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=not args.overwrite)

    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    mask_dir = output_dir / "mask"
    preview_dir = output_dir / "previews"
    for path in (rgb_dir, depth_dir, mask_dir, preview_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows = read_top_manifest_rows(args.manifest, args.top_k, args.max_images)
    device = torch.device("cuda")

    cache_dir = os.environ.get("HF_HUB_CACHE") or None
    model = DepthAnything3.from_pretrained(model_repo_id(args.teacher_model), cache_dir=cache_dir)
    model = model.to(device).eval()

    print(f"Output dir: {output_dir}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Manifest: {args.manifest}")
    print(f"Pseudo-label dir: {args.pseudo_dir}")
    print(f"Generating {len(rows)} source images x 5 views")

    dataset_rows: list[dict[str, object]] = []
    source_rows_path = output_dir / "source_topk_manifest.csv"
    with source_rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for source_idx, row in enumerate(rows, start=1):
        train_id = row.get("train_id") or image_id_from_path(Path(row["rgb_path"]))
        rgb_path = Path(row["rgb_path"])
        if not rgb_path.exists():
            rgb_path = args.train_dir / f"{train_id}_rgb.png"
        pseudo_path = args.pseudo_dir / f"{train_id}_depth.npy"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB not found for {train_id}: {rgb_path}")
        if not pseudo_path.exists():
            raise FileNotFoundError(f"Pseudo-label depth not found for {train_id}: {pseudo_path}")

        sample_rows: list[dict[str, object]] = []

        orig_rgb_out = rgb_dir / f"{train_id}_orig_rgb.png"
        orig_depth_out = depth_dir / f"{train_id}_orig_depth.npy"
        orig_mask_out = mask_dir / f"{train_id}_orig_mask.npy"
        orig_depth, orig_mask = sanitize_depth(np.load(pseudo_path).astype(np.float32))
        save_rgb(orig_rgb_out, Image.open(rgb_path), args.process_res)
        np.save(orig_depth_out, orig_depth)
        np.save(orig_mask_out, orig_mask.astype(np.uint8))

        base_meta = {
            "source_rank": row.get("rank", source_idx),
            "source_train_id": train_id,
            "source_rgb_path": str(rgb_path),
            "source_pseudo_depth_path": str(pseudo_path),
            "nearest_test_id": row.get("nearest_test_id", ""),
            "cosine_score": row.get("score", ""),
            "nearest_test_score": row.get("nearest_test_score", ""),
            "teacher_model": args.teacher_model,
            "delta_fraction": args.delta_fraction,
            "keep_percent": args.keep_percent,
        }
        orig_row = {
            **base_meta,
            "view": "orig",
            "rgb_path": str(orig_rgb_out),
            "depth_path": str(orig_depth_out),
            "mask_path": str(orig_mask_out),
            "kept_fraction": float(orig_mask.mean()),
            "kept_valid_fraction": 1.0,
            "trust_threshold": "",
        }
        dataset_rows.append(orig_row)
        sample_rows.append(orig_row)

        with torch.inference_mode():
            prediction = model.inference(
                image=[str(rgb_path)],
                infer_gs=True,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
            )
            novel_views = render_novel_views(
                prediction=prediction,
                delta_fraction=args.delta_fraction,
                keep_percent=args.keep_percent,
                scene_conf_percentile=args.scene_conf_percentile,
                device=device,
            )

        for novel in novel_views:
            view = str(novel["view"])
            rgb_out = rgb_dir / f"{train_id}_{view}_rgb.png"
            depth_out = depth_dir / f"{train_id}_{view}_depth.npy"
            mask_out = mask_dir / f"{train_id}_{view}_mask.npy"
            save_rgb(rgb_out, novel["rgb"], args.process_res)
            np.save(depth_out, novel["depth"])
            np.save(mask_out, novel["mask"].astype(np.uint8))

            novel_row = {
                **base_meta,
                "view": view,
                "rgb_path": str(rgb_out),
                "depth_path": str(depth_out),
                "mask_path": str(mask_out),
                "kept_fraction": novel["kept_fraction"],
                "kept_valid_fraction": novel["kept_valid_fraction"],
                "trust_threshold": novel["trust_threshold"],
            }
            dataset_rows.append(novel_row)
            sample_rows.append(novel_row)

        if source_idx <= args.preview_limit:
            make_preview(sample_rows, preview_dir / f"{train_id}_preview.png")

        if source_idx == 1 or source_idx % 10 == 0 or source_idx == len(rows):
            print(f"  [{source_idx}/{len(rows)}] {train_id}")

        del prediction
        torch.cuda.empty_cache()

    manifest_path = output_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = list(dataset_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)

    metadata = {
        "output_dir": str(output_dir),
        "train_dir": str(args.train_dir),
        "manifest": str(args.manifest),
        "pseudo_dir": str(args.pseudo_dir),
        "teacher_model": args.teacher_model,
        "ranking_method": "cosine",
        "top_k": args.top_k,
        "max_images": args.max_images,
        "num_source_images": len(rows),
        "num_dataset_samples": len(dataset_rows),
        "views_per_source": ["orig", "left", "right", "up", "down"],
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "delta_fraction": args.delta_fraction,
        "keep_percent": args.keep_percent,
        "scene_conf_percentile": args.scene_conf_percentile,
        "dataset_manifest": str(manifest_path),
        "source_topk_manifest": str(source_rows_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset manifest: {manifest_path}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

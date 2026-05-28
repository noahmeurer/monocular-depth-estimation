import argparse
import csv
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from depth_anything_3.api import DepthAnything3


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], img_size: int):
        self.image_paths = image_paths
        self.img_size = img_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.LANCZOS
        )
        return self.normalize(TF.to_tensor(image))


def image_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rgb"):
        return stem[:-4]
    if stem.endswith("_depth"):
        return stem[:-6]
    return stem


@torch.no_grad()
def extract_features(
    model: DepthAnything3,
    image_paths: List[Path],
    device: torch.device,
    img_size: int,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> torch.Tensor:
    dataset = RGBImageDataset(image_paths, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    embeddings = []
    amp_context = autocast("cuda", enabled=amp) if device.type == "cuda" else nullcontext()

    for batch_idx, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with amp_context:
            feats, _ = model.model.backbone(images.unsqueeze(1))
        patch_tokens = feats[-1][0]
        if patch_tokens.dim() != 4:
            raise RuntimeError(f"Unexpected DA3 backbone feature shape: {tuple(patch_tokens.shape)}")
        pooled = patch_tokens.float().mean(dim=(1, 2))
        embeddings.append(pooled.cpu())
        if batch_idx % 20 == 0:
            print(f"  extracted {min(batch_idx * batch_size, len(image_paths))}/{len(image_paths)}")

    return torch.cat(embeddings, dim=0)


def write_manifest(paths: List[Path], csv_path: Path, split: str) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "split", "image_id", "rgb_path", "depth_path"])
        writer.writeheader()
        for idx, path in enumerate(paths):
            image_id = image_id_from_path(path)
            depth_path = path.parent / f"{image_id}_depth.npy" if split == "train" else ""
            writer.writerow(
                {
                    "index": idx,
                    "split": split,
                    "image_id": image_id,
                    "rgb_path": str(path),
                    "depth_path": str(depth_path),
                }
            )


def save_feature_split(output_dir: Path, split: str, paths: List[Path], features: torch.Tensor) -> None:
    payload = {
        "split": split,
        "features": features.contiguous(),
        "image_ids": [image_id_from_path(p) for p in paths],
        "paths": [str(p) for p in paths],
    }
    torch.save(payload, output_dir / f"{split}_features.pt")
    write_manifest(paths, output_dir / f"{split}_manifest.csv", split=split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache pooled DA3/DINO backbone features for Baseline3 validation subset selection."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/cluster/courses/cil/monocular-depth-estimation")),
    )
    parser.add_argument("--model", default=os.environ.get("STUDENT_MODEL", "DA3MONO-LARGE"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=560)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("VAL_FEATURE_BATCH", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch_root = Path(os.environ.get("SCRATCH_ROOT", f"/work/scratch/{os.environ.get('USER', 'nmeurer')}"))
    output_dir = args.output_dir or scratch_root / "outputs/baseline3/feature_cache" / f"da3_backbone_{args.model}"
    expected = [output_dir / "train_features.pt", output_dir / "test_features.pt"]
    if any(path.exists() for path in expected) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains feature files. Pass --overwrite to regenerate.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.data_root / "train"
    test_dir = args.data_root / "test"
    train_paths = sorted(train_dir.glob("*_rgb.png"))
    test_paths = sorted(test_dir.glob("*_rgb.png"))
    if not train_paths:
        raise FileNotFoundError(f"No train RGB images found in {train_dir}")
    if not test_paths:
        raise FileNotFoundError(f"No test RGB images found in {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available; feature extraction will be slow.")

    print(f"Writing features to {output_dir}")
    print(f"Loading depth-anything/{args.model}")
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}").to(device).eval()

    print(f"Extracting train features: {len(train_paths)} images")
    train_features = extract_features(
        model=model,
        image_paths=train_paths,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    save_feature_split(output_dir, "train", train_paths, train_features)

    print(f"Extracting test features: {len(test_paths)} images")
    test_features = extract_features(
        model=model,
        image_paths=test_paths,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    save_feature_split(output_dir, "test", test_paths, test_features)

    metadata = {
        "model": args.model,
        "data_root": str(args.data_root),
        "img_size": args.img_size,
        "pooling": "mean over final DA3/DINO patch tokens",
        "normalized": False,
        "train_count": len(train_paths),
        "test_count": len(test_paths),
        "feature_dim": int(train_features.shape[1]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    F.normalize(train_features[:1], dim=1)  # fail early if feature dtype/shape is unusable downstream
    print("Done.")


if __name__ == "__main__":
    main()

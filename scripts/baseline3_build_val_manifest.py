import argparse
import csv
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def load_feature_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    required = {"features", "image_ids", "paths"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return payload


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1)


def apply_pca(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim < 8:
        raise ValueError("PCA dimension must be at least 8 for validation selection.")
    combined = torch.cat([train_features, test_features], dim=0).float()

    # Center the data
    centered = combined - combined.mean(dim=0, keepdim=True)

    # Use low-rank SVD to find the principal eigenvectors
    _, _, v = torch.pca_lowrank(centered, q=dim, center=False)
    
    # Project the data onto the principal eigenvectors
    projected = centered @ v[:, :dim]
    projected = normalize_features(projected)
    return projected[: len(train_features)], projected[len(train_features):]


def project_pca_2d(train_features: torch.Tensor, test_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([train_features, test_features], dim=0).float()
    centered = combined - combined.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    projected = centered @ v[:, :2]
    return projected[: len(train_features)], projected[len(train_features):]


def score_train_images(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k = min(max(1, top_k), test_features.shape[0])
    scores = []
    nearest_test = []
    nearest_score = []

    for start in range(0, train_features.shape[0], chunk_size):
        train_chunk = train_features[start:start + chunk_size]
        sims = train_chunk @ test_features.T
        values, indices = sims.topk(k=top_k, dim=1, largest=True)
        scores.append(values.mean(dim=1).cpu())
        nearest_test.append(indices[:, 0].cpu())
        nearest_score.append(values[:, 0].cpu())

    return torch.cat(scores), torch.cat(nearest_test), torch.cat(nearest_score)


def preview_asset_link(method_dir: Path, source_path: str, split: str) -> str:
    source = Path(source_path).resolve()
    asset_dir = method_dir / "preview_assets" / split
    asset_dir.mkdir(parents=True, exist_ok=True)
    link = asset_dir / source.name

    if link.is_symlink():
        if link.resolve() != source:
            link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink preview asset: {link}")

    if not link.exists():
        link.symlink_to(source)

    return link.relative_to(method_dir).as_posix()


def write_manifest(
    method_dir: Path,
    method: str,
    order: torch.Tensor,
    scores: torch.Tensor,
    nearest_test: torch.Tensor,
    nearest_score: torch.Tensor,
    train_payload: dict,
    test_payload: dict,
    val_count: int,
    preview_count: int,
) -> Path:
    method_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = method_dir / "val_top10pct.csv"
    preview_csv_path = method_dir / "preview_top500.csv"
    selected = order[:val_count].tolist()
    preview = order[:preview_count].tolist()

    fieldnames = [
        "rank",
        "train_index",
        "train_id",
        "rgb_path",
        "depth_path",
        "score",
        "nearest_test_index",
        "nearest_test_id",
        "nearest_test_path",
        "nearest_test_score",
        "method",
    ]

    def row_for(rank: int, train_idx: int) -> dict:
        test_idx = int(nearest_test[train_idx].item())
        train_id = train_payload["image_ids"][train_idx]
        return {
            "rank": rank,
            "train_index": train_idx,
            "train_id": train_id,
            "rgb_path": train_payload["paths"][train_idx],
            "depth_path": str(Path(train_payload["paths"][train_idx]).parent / f"{train_id}_depth.npy"),
            "score": float(scores[train_idx].item()),
            "nearest_test_index": test_idx,
            "nearest_test_id": test_payload["image_ids"][test_idx],
            "nearest_test_path": test_payload["paths"][test_idx],
            "nearest_test_score": float(nearest_score[train_idx].item()),
            "method": method,
        }

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, train_idx in enumerate(selected, start=1):
            writer.writerow(row_for(rank, train_idx))

    with preview_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, train_idx in enumerate(preview, start=1):
            writer.writerow(row_for(rank, train_idx))

    preview_rows = [row_for(rank, train_idx) for rank, train_idx in enumerate(preview, start=1)]
    write_html_preview(
        method_dir,
        method=method,
        rows=preview_rows,
    )
    return manifest_path


def write_html_preview(method_dir: Path, method: str, rows: Iterable[dict]) -> None:
    cards = []
    for row in rows:
        train_uri = preview_asset_link(method_dir, row["rgb_path"], split="train")
        test_uri = preview_asset_link(method_dir, row["nearest_test_path"], split="test")
        cards.append(
            f"""
            <section class="pair">
              <div class="meta">
                <strong>#{row['rank']}</strong>
                <span>score {row['score']:.4f}</span>
                <span>nearest {row['nearest_test_score']:.4f}</span>
              </div>
              <div class="images">
                <figure>
                  <img src="{html.escape(train_uri)}" loading="lazy" />
                  <figcaption>{html.escape(row['train_id'])}</figcaption>
                </figure>
                <figure>
                  <img src="{html.escape(test_uri)}" loading="lazy" />
                  <figcaption>{html.escape(row['nearest_test_id'])}</figcaption>
                </figure>
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Baseline3 Validation Preview - {html.escape(method)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #171717; }}
    h1 {{ font-size: 20px; margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 14px; }}
    .pair {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
    .meta {{ display: flex; gap: 10px; align-items: baseline; font-size: 13px; margin-bottom: 8px; }}
    .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 4px; display: block; }}
    figcaption {{ margin-top: 4px; font-size: 12px; overflow-wrap: anywhere; color: #444; }}
  </style>
</head>
<body>
  <h1>{html.escape(method)} validation preview: selected train image, nearest test image</h1>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (method_dir / "preview_top500.html").write_text(page)


def write_pca_2d_plot(
    output_dir: Path,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    selected_by_method: dict[str, torch.Tensor],
    max_train_points: int,
    seed: int,
) -> Path:
    train_xy, test_xy = project_pca_2d(train_features, test_features)
    train_xy = train_xy.numpy()
    test_xy = test_xy.numpy()

    if len(train_xy) > max_train_points:
        generator = torch.Generator().manual_seed(seed)
        background_idx = torch.randperm(len(train_xy), generator=generator)[:max_train_points].numpy()
    else:
        background_idx = slice(None)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    ax.scatter(
        train_xy[background_idx, 0],
        train_xy[background_idx, 1],
        s=5,
        c="#9a9a9a",
        alpha=0.25,
        linewidths=0,
        label="train RGBs",
    )
    ax.scatter(
        test_xy[:, 0],
        test_xy[:, 1],
        s=14,
        c="#1f77b4",
        alpha=0.8,
        linewidths=0,
        label="test RGBs",
    )

    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for color, (method, selected) in zip(colors, selected_by_method.items()):
        selected_np = selected.numpy()
        ax.scatter(
            train_xy[selected_np, 0],
            train_xy[selected_np, 1],
            s=10,
            alpha=0.55,
            linewidths=0,
            c=color,
            label=f"{method} selected train",
        )

    ax.set_title("2D PCA of DA3/DINO Backbone Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()

    out_path = output_dir / "pca2d_validation_selection.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def parse_methods(methods: str) -> list[tuple[str, int | None]]:
    parsed = []
    for raw in methods.split(","):
        method = raw.strip().lower()
        if not method:
            continue
        if method == "cosine":
            parsed.append(("cosine", None))
        elif method.startswith("pca"):
            dim_text = method[3:]
            dim = int(dim_text) if dim_text else 64
            parsed.append((f"pca{dim}", dim))
        else:
            raise ValueError(f"Unknown method {raw!r}; use cosine,pca64")
    if not parsed:
        raise ValueError("No methods requested.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Baseline3 GT validation manifests and HTML previews from cached DA3/DINO features."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", default="cosine,pca64")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--preview-count", type=int, default=500)
    parser.add_argument("--top-k-test", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-pca2d-plot",
        action="store_true",
        help="Write a plotting-only 2D PCA view of train/test embeddings and selected validation points.",
    )
    parser.add_argument(
        "--pca2d-max-train-points",
        type=int,
        default=5000,
        help="Maximum random background train points to draw in the 2D PCA plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = args.output_dir or args.feature_dir / "validation_subsets"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = load_feature_payload(args.feature_dir / "train_features.pt")
    test_payload = load_feature_payload(args.feature_dir / "test_features.pt")
    train_features_raw = train_payload["features"].float()
    test_features_raw = test_payload["features"].float()
    val_count = max(1, int(len(train_features_raw) * args.val_fraction))
    preview_count = min(max(1, args.preview_count), len(train_features_raw))

    outputs = {}
    selected_by_method = {}
    for method, pca_dim in parse_methods(args.methods):
        print(f"Building {method} manifest")
        if pca_dim is None:
            train_features = normalize_features(train_features_raw)
            test_features = normalize_features(test_features_raw)
        else:
            train_features, test_features = apply_pca(train_features_raw, test_features_raw, pca_dim)

        scores, nearest_test, nearest_score = score_train_images(
            train_features=train_features,
            test_features=test_features,
            top_k=args.top_k_test,
            chunk_size=max(1, args.chunk_size),
        )
        order = torch.argsort(scores, descending=True)
        selected_by_method[method] = order[:val_count].cpu()
        manifest_path = write_manifest(
            method_dir=output_dir / method,
            method=method,
            order=order,
            scores=scores,
            nearest_test=nearest_test,
            nearest_score=nearest_score,
            train_payload=train_payload,
            test_payload=test_payload,
            val_count=val_count,
            preview_count=preview_count,
        )
        outputs[method] = str(manifest_path)
        print(f"  wrote {manifest_path}")

    pca2d_plot = None
    if args.write_pca2d_plot:
        pca2d_plot = write_pca_2d_plot(
            output_dir=output_dir,
            train_features=train_features_raw,
            test_features=test_features_raw,
            selected_by_method=selected_by_method,
            max_train_points=max(1, args.pca2d_max_train_points),
            seed=args.seed,
        )
        print(f"  wrote {pca2d_plot}")

    metadata = {
        "feature_dir": str(args.feature_dir),
        "output_dir": str(output_dir),
        "methods": outputs,
        "val_fraction": args.val_fraction,
        "val_count": val_count,
        "preview_count": preview_count,
        "top_k_test": args.top_k_test,
        "pca2d_plot": str(pca2d_plot) if pca2d_plot else "",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

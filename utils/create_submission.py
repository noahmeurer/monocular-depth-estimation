from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import base64
import zlib


def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Kaggle submission CSV from depth .npy predictions.")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Folder containing test_*.npy prediction files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Path where the submission CSV should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir
    out_csv = args.out_csv

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    rows = []
    pred_files = sorted(pred_dir.glob("test_*.npy"))

    if not pred_files:
        raise FileNotFoundError(f"No test_*.npy prediction files found in {pred_dir}")

    for pred_path in pred_files:
        depth = np.load(pred_path)

        idx = pred_path.stem.split("_")[-1]
        img_id = f"test_{idx}_depth"

        encoded_depth = encode_depth(depth)

        rows.append({
            "id": img_id,
            "Depths": encoded_depth,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(out_csv, index=False)

    print(f"Saved submission to {out_csv}")
    print(f"Number of predictions: {len(df)}")


if __name__ == "__main__":
    main()

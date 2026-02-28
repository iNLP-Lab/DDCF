#!/usr/bin/env python3
"""
Download binary_correctness dataset from Hugging Face Hub and save as CSV files
in the same layout as binary_correctness_data (one CSV per model).

Usage:
  python download_binary_correctness_from_hf.py --repo_id YOUR_USERNAME/binary-correctness-math

  # Custom output dir and/or specific models
  python download_binary_correctness_from_hf.py --repo_id USER/repo --out_dir binary_correctness_data --models "Qwen__Qwen2.5-7B,Qwen__Qwen2.5-14B"
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Download binary_correctness dataset from HF and save as CSV files per model")
    p.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face dataset repo id, e.g. 'iNLP-Lab/binary-correctness-math'",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="binary_correctness_math",
        help="Output directory for CSV files (one per model)",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated list of model names to download. If not set, downloads all.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.repo_id}...")
    ds = load_dataset(args.repo_id, split=args.split)

    if "model" not in ds.column_names:
        raise ValueError("Dataset has no 'model' column. " "Use a repo that was uploaded with upload_binary_correctness_to_hf.py (default mode).")

    models_to_save = None
    if args.models:
        models_to_save = {s.strip() for s in args.models.split(",")}

    # Group by model and write one CSV per model
    model_names = ds.unique("model")
    for model in sorted(model_names):
        if models_to_save is not None and model not in models_to_save:
            continue
        sub = ds.filter(lambda ex, m=model: ex["model"] == m)
        # Convert to pandas and drop 'model' column so CSV matches original format
        df = sub.to_pandas()
        df = df.drop(columns=["model"], errors="ignore")
        path = out_dir / f"{model}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows -> {path}")

    print(f"Done. CSVs written to {out_dir}/")


if __name__ == "__main__":
    main()

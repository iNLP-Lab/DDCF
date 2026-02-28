#!/usr/bin/env python3
"""
Download DDCF MATH datasets from Hugging Face Hub and save as CSV files.

Downloads:
  - DDCF-seed-math -> MATH/seed_math.csv (from Dataset)
  - DDCF-fullcorpus-math -> MATH/fullcorpus_math.csv (single file)

Usage:
  # Download both to default paths (MATH/seed_math.csv, MATH/fullcorpus_math.csv)
  python download_ddcf_math_from_hf.py

  # Download only one
  python download_ddcf_math_from_hf.py --which seed
  python download_ddcf_math_from_hf.py --which fullcorpus

  # Custom repo IDs and output paths
  python download_ddcf_math_from_hf.py --seed_repo_id USERNAME/DDCF-seed-math --seed_out my_seed.csv
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_SEED_REPO = "iNLP-Lab/DDCF-seed-math"
DEFAULT_FULLCORPUS_REPO = "iNLP-Lab/DDCF-fullcorpus-math"
DEFAULT_SEED_OUT = "MATH/seed_math.csv"
DEFAULT_FULLCORPUS_OUT = "MATH/fullcorpus_math.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Download DDCF MATH (seed and/or fullcorpus) from Hugging Face")
    p.add_argument(
        "--which",
        type=str,
        choices=("seed", "fullcorpus", "both"),
        default="both",
        help="Which dataset(s) to download (default: both)",
    )
    p.add_argument(
        "--seed_repo_id",
        type=str,
        default=DEFAULT_SEED_REPO,
        help=f"HF repo for seed (default: {DEFAULT_SEED_REPO})",
    )
    p.add_argument(
        "--fullcorpus_repo_id",
        type=str,
        default=DEFAULT_FULLCORPUS_REPO,
        help=f"HF repo for full corpus (default: {DEFAULT_FULLCORPUS_REPO})",
    )
    p.add_argument(
        "--seed_out",
        type=str,
        default=DEFAULT_SEED_OUT,
        help=f"Output path for seed CSV (default: {DEFAULT_SEED_OUT})",
    )
    p.add_argument(
        "--fullcorpus_out",
        type=str,
        default=DEFAULT_FULLCORPUS_OUT,
        help=f"Output path for full corpus CSV (default: {DEFAULT_FULLCORPUS_OUT})",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for seed (default: train)",
    )
    return p.parse_args()


def download_seed(repo_id: str, out_path: Path, split: str) -> None:
    """Download seed MATH from HF Dataset and save as CSV."""
    print(f"Loading dataset {repo_id}...")
    ds = load_dataset(repo_id, split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = ds.to_pandas()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")


def download_fullcorpus(repo_id: str, out_path: Path) -> None:
    """Download full corpus MATH CSV from HF repo."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename="fullcorpus_math.csv",
        repo_type="dataset",
        local_dir=out_path.parent,
        local_dir_use_symlinks=False,
    )
    downloaded = Path(local_path)
    if downloaded.resolve() != out_path.resolve():
        downloaded.rename(out_path)
    print(f"Saved -> {out_path}")


def main():
    args = parse_args()
    seed_out = Path(args.seed_out)
    fullcorpus_out = Path(args.fullcorpus_out)

    if args.which in ("seed", "both"):
        download_seed(args.seed_repo_id, seed_out, args.split)

    if args.which in ("fullcorpus", "both"):
        download_fullcorpus(args.fullcorpus_repo_id, fullcorpus_out)

    print("Done.")


if __name__ == "__main__":
    main()

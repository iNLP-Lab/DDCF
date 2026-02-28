import argparse
from typing import Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 128


def get_question_embeddings(
    csv_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    """
    Generate embeddings for questions in the order specified in the CSV file.

    Args:
        csv_path: Path to a CSV file containing a 'prompt' column.
        model_name: Name/path of the sentence-transformer model to use.
        device: Torch device to run the model on (e.g., "cuda:0", "cpu").
        batch_size: Batch size used during encoding.

    Returns:
        Tensor of shape (num_questions, embedding_dim) with question embeddings.
    """
    df = pd.read_csv(csv_path)
    questions = df["prompt"].tolist()

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={
            # "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )

    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    return torch.tensor(embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate question embeddings from CSV files.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help='Torch device to use, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--seed_csv",
        type=str,
        default="MATH/seed_math.csv",
        help="Path to CSV file for seed MATH questions.",
    )
    parser.add_argument(
        "--full_csv",
        type=str,
        default="MATH/fullcorpus_math.csv",
        help="Path to CSV file for full MATH corpus.",
    )
    parser.add_argument(
        "--seed_out",
        type=str,
        default="DDCF_data/seed_math_embeddings.pth",
        help="Output path for seed embeddings tensor.",
    )
    parser.add_argument(
        "--full_out",
        type=str,
        default="DDCF_data/fullcorpus_math_embeddings.pth",
        help="Output path for full corpus embeddings tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seed set
    print(f"Loading model '{args.model_name}' on device '{args.device}'")

    # Seed set embeddings
    seed_embeddings = get_question_embeddings(
        csv_path=args.seed_csv,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    torch.save(seed_embeddings, args.seed_out)
    print(f"Generated seed embeddings of shape: {seed_embeddings.shape}")
    print(f"Saved seed embeddings to: {args.seed_out}")

    # Full corpus embeddings
    full_embeddings = get_question_embeddings(
        csv_path=args.full_csv,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    torch.save(full_embeddings, args.full_out)
    print(f"Generated full corpus embeddings of shape: {full_embeddings.shape}")
    print(f"Saved full corpus embeddings to: {args.full_out}")


if __name__ == "__main__":
    main()

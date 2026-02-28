import argparse
import random

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training and validation data for DDCF.")
    parser.add_argument(
        "--question_csv",
        type=str,
        default="MATH/seed_math.csv",
        help="Path to seed MATH question CSV.",
    )
    parser.add_argument(
        "--model_order_csv",
        type=str,
        default="DDCF_data/model_order.csv",
        help="Path to model order CSV (with model_id and model_name).",
    )
    parser.add_argument(
        "--train_out",
        type=str,
        default="DDCF_data/DDCF_traindata.csv",
        help="Output path for training data CSV.",
    )
    parser.add_argument(
        "--val_out",
        type=str,
        default="DDCF_data/DDCF_valdata.csv",
        help="Output path for validation data CSV.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of questions to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    q_df = pd.read_csv(args.question_csv)

    prompt_ids = np.array(q_df["prompt_id"].tolist())
    val_size = int(args.val_fraction * len(prompt_ids))
    val_index = np.random.choice(prompt_ids, size=val_size, replace=False)

    m_df = pd.read_csv(args.model_order_csv)
    model_ids = m_df["model_id"].tolist()
    model_names = m_df["model_name"].tolist()

    train_data, val_data = None, None
    for model_id, model_name in zip(model_ids, model_names):
        model_name_safe = model_name.replace("/", "__")
        df = pd.read_csv(f"binary_correctness_data/{model_name_safe}.csv")
        df["label"] = df["check"].apply(int)
        df["model_id"] = model_id
        df = df[["model_id", "label", "prompt"]]
        df = df.merge(q_df, on="prompt")[["prompt_id", "model_id", "label", "prompt"]]

        val_mask = df["prompt_id"].isin(val_index)
        val_df = df[val_mask]
        train_df = df[~val_mask]

        if train_data is None:
            train_data = train_df
            val_data = val_df
        else:
            train_data = pd.concat([train_data, train_df], ignore_index=True)
            val_data = pd.concat([val_data, val_df], ignore_index=True)

    print(train_data.shape, val_data.shape)

    train_data.to_csv(args.train_out, index=False)
    val_data.to_csv(args.val_out, index=False)


if __name__ == "__main__":
    main()

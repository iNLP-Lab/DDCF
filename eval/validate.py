import argparse
import os
from math import comb

import pandas as pd
import swifter

from eval.grader import math_equal
from eval.parser import extract_answer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"
DEFAULT_NUM_SELECT = "1000"
DEFAULT_LAMDA = 0.2
DEFAULT_SEED = 42
DEFAULT_PREDICTIONS_ROOT = "predictions"
AIME24_N = 64
AIME24_K = 32
TASKS = [
    "math500",
    "olympiadbench",
    "gsm8k",
    "sat",
    "minervamath",
    "gaokao",
    "stem",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate predictions and compute per-task accuracy and Pass@k.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name (used in prediction path).",
    )
    parser.add_argument(
        "--num_select",
        type=str,
        default=DEFAULT_NUM_SELECT,
        help="Selection size or identifier (e.g. '1000' or 'base').",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=DEFAULT_LAMDA,
        help="Lambda used in selection (used in path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (used in path).",
    )
    parser.add_argument(
        "--predictions_root",
        type=str,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Root directory for prediction inputs and outputs.",
    )
    return parser.parse_args()


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass@k = 1 - ((n - c) choose k) / (n choose k).

    Args:
        n: Total number of attempts.
        c: Number of correct solutions.
        k: Number of samples considered.

    Returns:
        Pass@k value in [0, 1].
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    total = comb(n, k)
    wrong = comb(n - c, k)
    return 1.0 - wrong / total


def main() -> None:
    args = parse_args()

    model_name_safe = args.model_name.replace("/", "__")
    name = f"ddcf/{args.num_select}/{model_name_safe}_{args.lamda}_{args.seed}"
    base_dir = os.path.join(args.predictions_root, name)

    csv_path = os.path.join(base_dir, "all_passk.csv")
    df = pd.read_csv(csv_path)

    df["answer"] = df["answer"].apply(lambda x: str(x).replace("np.arcsin", "\\arcsin"))
    df["final_pred"] = df["prediction"].swifter.apply(lambda x: extract_answer(x))

    golds = df["answer"].tolist()
    final_preds = df["final_pred"].tolist()
    df["pair"] = [{"gold": str(gold), "pred": str(pred)} for gold, pred in zip(golds, final_preds)]
    df["check"] = df["pair"].swifter.apply(lambda x: math_equal(x["pred"], x["gold"], timeout=True))

    # AIME24: Pass@k per problem then average
    aime = df[df["task"] == "aime24"][["problem", "check"]].groupby("problem").sum()
    aime_passk = aime["check"].apply(lambda c: pass_at_k(AIME24_N, int(c), AIME24_K)).mean() * 100

    txt_path = os.path.join(base_dir, "all_passk.txt")
    with open(txt_path, "w") as f:
        print(f"The accuracy at the task aime24 is {aime_passk:.2f}", file=f)
        for task in TASKS:
            acc = df[df["task"] == task]["check"].mean() * 100
            print(f"The accuracy at the task {task} is {acc:.2f}", file=f)

    out_csv_path = os.path.join(base_dir, "all_passk_validated.csv")
    df.to_csv(out_csv_path, index=False)
    print(f"Validated results saved to {out_csv_path}")


if __name__ == "__main__":
    main()

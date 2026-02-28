import argparse
import os
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"
DEFAULT_NUM_SELECT = 1000
DEFAULT_LAMDA = 0.2
DEFAULT_METHOD = "ddcf"
DEFAULT_DIFFICULTY_DIR = "factorized_data"
DEFAULT_EMBEDDINGS_PATH = "DDCF_data/fullcorpus_math_embeddings.pth"
DEFAULT_FULL_CORPUS_PATH = "MATH/fullcorpus_math.csv"
DEFAULT_OUTPUT_ROOT = "selected_data"
DEFAULT_MAX_SCORE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="k-greedy selection with ddcf and dataset conversion.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model name used for difficulty estimation.",
    )
    parser.add_argument(
        "--num_select",
        type=int,
        default=DEFAULT_NUM_SELECT,
        help="Number of items to select (k).",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=DEFAULT_LAMDA,
        help="Trade-off between difficulty and redundancy.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        help="Selection method name (used in output paths).",
    )
    parser.add_argument(
        "--difficulty_dir",
        type=str,
        default=DEFAULT_DIFFICULTY_DIR,
        help="Directory containing per-model difficulty CSVs.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to full-corpus question embeddings tensor (.pth).",
    )
    parser.add_argument(
        "--full_corpus_path",
        type=str,
        default=DEFAULT_FULL_CORPUS_PATH,
        help="Path to full-corpus MATH CSV (with prompt_id).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory to save Hugging Face datasets.",
    )
    parser.add_argument(
        "--max_score",
        type=float,
        default=DEFAULT_MAX_SCORE,
        help="Maximum difficulty score to keep in the candidate pool.",
    )
    return parser.parse_args()


def load_factorized_embeddings(pt_path: str) -> np.ndarray:
    """
    Load a torch tensor of shape (n, d) onto CPU and return as NumPy array.
    """
    tensor = torch.load(pt_path, map_location="cpu")
    return tensor.numpy()


def ddcf_greedy_min(
    embed: np.ndarray,
    diff: np.ndarray,
    k: int,
    lam: float = 0.5,
    mask: Union[None, Sequence[int], np.ndarray] = None,
    return_costs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy minimisation of cost_i = lam * diff_i + (1 - lam) * max_sim_to_S.

    Returns:
        sel_idx: (k,) absolute indices in the original arrays.
        costs:   (k,) realised cost for each chosen item (if return_costs=True).
    """
    n, _ = embed.shape
    assert diff.shape[0] == n

    if mask is None:
        pool = np.arange(n, dtype=np.int64)
    else:
        m = np.asarray(mask)
        pool = np.where(m)[0] if m.dtype == bool else m.astype(np.int64)

    emb_pool = embed[pool].astype(np.float32, copy=False)
    emb_pool /= np.linalg.norm(emb_pool, axis=1, keepdims=True).clip(min=1e-12)

    selected, costs = [], []
    redundancy = np.zeros(pool.size, dtype=np.float32)
    active = np.ones(pool.size, dtype=bool)

    for _ in range(min(k, pool.size)):
        cost = lam * diff[pool] + (1.0 - lam) * redundancy
        cost[~active] = np.inf

        best_in_pool = int(np.argmin(cost))
        best_global = int(pool[best_in_pool])

        selected.append(best_global)
        costs.append(float(cost[best_in_pool]))
        active[best_in_pool] = False

        sim_new = emb_pool @ emb_pool[best_in_pool]
        redundancy = np.maximum(redundancy, sim_new)

    sel_idx = np.array(selected, dtype=np.int64)
    return (sel_idx, np.array(costs)) if return_costs else sel_idx


def select_indices(args: argparse.Namespace) -> np.ndarray:
    """
    Run ddcf-based k-greedy selection and return selected indices.
    """
    model_name_safe = args.model_name.replace("/", "__")

    diff_path = os.path.join(args.difficulty_dir, f"{model_name_safe}.csv")
    df = pd.read_csv(diff_path)
    scores = df["prediction"].to_numpy(dtype=float)

    embeddings = load_factorized_embeddings(args.embeddings_path)

    mask = scores <= args.max_score

    print(f"Start selection: model={model_name_safe}, lamda={args.lamda}, k={args.num_select}")
    sel_idx = ddcf_greedy_min(
        embed=embeddings,
        diff=scores,
        k=args.num_select,
        lam=args.lamda,
        mask=mask,
    )

    return sel_idx


def prepare_template(question: str) -> str:
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final "
        "answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
    )


def build_and_save_dataset(args: argparse.Namespace, selected_idx: np.ndarray) -> None:
    """
    Build a Hugging Face Dataset from selected prompt IDs and save to disk.
    """
    model_name_safe = args.model_name.replace("/", "__")

    df = pd.read_csv(args.full_corpus_path)
    chosen = df[df["prompt_id"].isin(selected_idx)].copy()
    chosen = chosen[["prompt_id", "prompt", "response", "gold", "subject", "len"]].drop_duplicates()

    chosen["text"] = chosen["prompt"].apply(prepare_template) + chosen["response"]
    chosen["original_index"] = chosen.index
    data_for_ds = chosen[["text", "original_index"]].reset_index(drop=True)

    hf_dataset = Dataset.from_pandas(data_for_ds)
    dataset_dict = DatasetDict({"train": hf_dataset})

    out_dir = os.path.join(
        args.output_root,
        args.method,
        str(args.num_select),
        f"{model_name_safe}_{args.lamda}",
    )
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    dataset_dict.save_to_disk(out_dir)
    print(f"Saved dataset to {out_dir}")


def main() -> None:
    args = parse_args()
    selected_idx = select_indices(args)
    build_and_save_dataset(args, selected_idx)


if __name__ == "__main__":
    main()

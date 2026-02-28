"""
Evaluate a fine-tuned model on math benchmarks (AIME, MATH500, Minerva, etc.)
using SGLang. Supports pass-k sampling for selected tasks.
"""

import argparse
import os
import re
from parser import extract_answer
from pathlib import Path

import pandas as pd
import swifter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASKS = [
    "aime24",
    "math500",
    "minervamath",
    "olympiadbench",
    "gsm8k",
    "sat",
    "gaokao",
    "stem",
]

# Tasks that use pass-k > 1 (multiple samples per question)
PASS_K_TASKS = {"aime24", "amc23"}

EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
SYSTEM_PROMPT = "<|im_start|>system\n" "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
STOP_TOKENS = ["<|im_end|>", "<|end▁of▁sentence|>"]

DEFAULT_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "stop": STOP_TOKENS,
}


def load_eval_data(data_dir: Path) -> pd.DataFrame:
    """Load all task JSONL files into a single DataFrame with a 'task' column."""
    frames = []
    for task in TASKS:
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Eval data not found: {path}")
        df = pd.read_json(path, lines=True)[["problem", "answer"]]
        df["task"] = task
        pass_k = 64 if task in PASS_K_TASKS else 1
        # Repeat rows for pass-k sampling
        frames.extend([df] * pass_k)
    data = pd.concat(frames, ignore_index=True)
    data["answer"] = data["answer"].apply(lambda x: str(x).replace("np.arcsin", "\\arcsin"))
    return data


def prepare_template(question: str) -> str:
    """Wrap question in the chat template."""
    return f"{SYSTEM_PROMPT}" "<|im_start|>user\n" f"{question}" "<|im_end|>\n<|im_start|>assistant\n"


def _ensure_boxed(completions: list[str], prompts: list[str], llm, base_params: dict):
    """
    For completions missing \\boxed{...}, run a short follow-up with a hint.
    Returns the same list with missing entries filled.
    """
    pattern = re.compile(r"\\boxed\{.*?\}")
    result = list(completions)
    to_fix = []
    indices = []
    prefixes = []
    for idx, comp in enumerate(completions):
        if pattern.search(comp):
            continue
        to_fix.append(prompts[idx] + comp + "\nThe final answer is \\boxed{")
        indices.append(idx)
        prefixes.append(comp + "\nThe final answer is \\boxed{")
    if not to_fix:
        return result
    params = {**base_params, "max_new_tokens": 32}
    new_completions = llm.generate(to_fix, params)
    for i, idx in enumerate(indices):
        result[idx] = prefixes[i] + new_completions[i]["text"]
    return result


def _ensure_im_end(completions: list[str], prompts: list[str], llm, base_params: dict):
    """
    For completions missing </think>, run a short follow-up to get the rest.
    Returns the same list with missing entries filled.
    """
    result = list(completions)
    to_fix = []
    indices = []
    prefixes = []
    for idx, comp in enumerate(completions):
        if "</think>" in comp:
            continue
        to_fix.append(prompts[idx] + comp + "\n</think>")
        indices.append(idx)
        prefixes.append(comp + "\n</think>")
    if not to_fix:
        return result
    params = {**base_params, "max_new_tokens": 2048}
    new_completions = llm.generate(to_fix, params)
    for i, idx in enumerate(indices):
        result[idx] = prefixes[i] + new_completions[i]["text"]
    return result


def run_inference(
    prompts: list[str],
    model_path: str,
    dp_size: int = 8,
    seed: int = 42,
) -> list[str]:
    """Run SGLang generation and post-process to ensure \\boxed{} and </think>."""
    import sglang as sgl

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        random_seed=seed,
        context_length=16384,
        dp_size=dp_size,
    )
    base_params = {**DEFAULT_SAMPLING, "max_new_tokens": 12288}

    responses = llm.generate(prompts, base_params)
    completions = [r["text"] for r in responses]

    completions = _ensure_im_end(completions, prompts, llm, base_params)
    completions = _ensure_boxed(completions, prompts, llm, base_params)

    llm.shutdown()
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on math benchmarks via SGLang.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B",
        help="Base model name (for display/paths).",
    )
    parser.add_argument(
        "--num_select",
        type=str,
        default="1000",
        help="Selection size, e.g. 1000, 2000, or custom name.",
    )
    parser.add_argument("--dp_size", type=int, default=8, help="SGLang data parallel size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lamda",
        type=float,
        default=0.2,
        dest="lamda",
        help="Trade-off between linear and quadratic terms (for ddcf path).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing task JSONL files. Default: eval/data",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else EVAL_DATA_DIR
    df = load_eval_data(data_dir)
    prompts = [prepare_template(q) for q in df["problem"].tolist()]
    print("Num questions:", len(prompts))

    model_name_safe = args.model_name.replace("/", "__")
    ddcf_sizes = {"1000", "2000", "4000", "8000", "16000", "32000", "64000", "128000"}
    if args.num_select in ddcf_sizes:
        run_name = f"ddcf/{args.num_select}/{model_name_safe}_{args.lamda}_{args.seed}"
    else:
        run_name = f"{args.num_select}/{model_name_safe}_{args.seed}"
    model_path = f"outdir/{run_name}"

    preds = run_inference(
        prompts,
        model_path=model_path,
        dp_size=args.dp_size,
        seed=args.seed,
    )

    df["prediction"] = preds
    df["final_pred"] = df["prediction"].swifter.apply(extract_answer)

    out_dir = Path(f"predictions/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_passk.csv"
    df.to_csv(out_path, index=False)
    print(f"Predictions and extracted answers saved to {out_path}")


if __name__ == "__main__":
    main()

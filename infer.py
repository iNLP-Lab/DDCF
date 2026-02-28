import argparse
import os
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import swifter
import torch
from transformers import AutoTokenizer

from eval.grader import math_equal
from eval.parser import extract_answer
from prepare import prepare_fewshot

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"
DEFAULT_SEED = 42
MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 2700


def parse_args() -> argparse.Namespace:
    # python infer.py --model_name Qwen/Qwen2.5-Math-7B
    parser = argparse.ArgumentParser(description="Inference setting details")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name or path for the base model.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size used by sglang.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def truncate_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    """Truncate prompts to the last MAX_PROMPT_TOKENS tokens if they are too long."""
    truncated: List[str] = []
    for prompt in prompts:
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(tokens) >= MAX_PROMPT_TOKENS:
            cutoff_prompt = tokenizer.decode(tokens[-MAX_PROMPT_TOKENS:])
            truncated.append(cutoff_prompt)
        else:
            truncated.append(prompt)
    return truncated


def build_stop_tokens(tokenizer: AutoTokenizer) -> List[str]:
    stop_tokens = ["<|im_end|>", "<｜end▁of▁sentence｜>", "Problem:"]
    if tokenizer.eos_token is not None:
        stop_tokens.append(tokenizer.eos_token)
    return stop_tokens


def run_verification(model_name: str) -> None:
    """
    Post-process generated completions:
    - merge with gold data
    - extract answers
    - compute correctness using math_equal
    - compute completion token lengths
    - overwrite CSV with enriched data and print mean accuracy.
    """
    tokenizer_model_name = model_name.replace("_short", "")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    model_name_safe = model_name.replace("/", "__")

    df = pd.read_csv(f"binary_correctness_data/{model_name_safe}.csv")
    df_gold = pd.read_csv("MATH/seed_math.csv")

    df = df.merge(df_gold, left_on="question", right_on="prompt", how="left")
    df = df[["prompt_id", "prompt", "completion", "gold"]]

    df["prediction"] = df["completion"].swifter.apply(lambda x: extract_answer(x, use_last_number=False))
    df["prediction"] = df["prediction"].fillna(value="Cannot extract result.")

    golds = df["gold"].tolist()
    predictions = df["prediction"].tolist()
    df["pair"] = [{"gold": str(gold), "pred": str(pred)} for gold, pred in zip(golds, predictions)]
    df["check"] = df["pair"].swifter.apply(lambda x: math_equal(x["pred"], x["gold"], timeout=True))

    df["len"] = df["completion"].swifter.apply(lambda x: len(tokenizer.encode(x)))

    print(model_name_safe, df["check"].mean())

    df.to_csv(f"binary_correctness_data/{model_name_safe}.csv", index=False)


def main() -> None:
    args = parse_args()

    import sglang as sgl

    start = time.time()

    model_name_safe = args.model_name.replace("/", "__")

    llm = sgl.Engine(
        model_path=args.model_name,
        tp_size=args.tensor_parallel_size,
        context_length=4096,
        dp_size=8,
    )

    set_seed(args.seed)

    df = pd.read_csv("MATH/seed_math.csv")
    problems = df["prompt"].tolist()
    types = df["type"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prompts = [prepare_fewshot(problem, type_) for problem, type_ in zip(problems, types)]

    print(args.model_name)

    prompts = truncate_prompts(prompts, tokenizer)

    print("Done data processing")

    stop_tokens = build_stop_tokens(tokenizer)

    sampling_params = {"temperature": 0.0, "stop": stop_tokens, "max_new_tokens": MAX_NEW_TOKENS}
    responses = llm.generate(prompts, sampling_params)
    pred_texts = [response["text"] for response in responses]

    initial_completions = pred_texts
    final_completions: List[str] = [None] * len(pred_texts)  # type: ignore[list-item]
    modified_questions: List[str] = []
    indices_to_modify: List[int] = []
    first_completions: List[str] = []

    for idx, comp in enumerate(initial_completions):
        if "\\boxed{" in comp:
            final_completions[idx] = comp
        else:
            modified_question = prompts[idx] + comp + " The final answer is $\\boxed{"
            modified_questions.append(modified_question)
            indices_to_modify.append(idx)
            first_completions.append(comp + " The final answer is $\\boxed{")

    if modified_questions:
        sampling_params = {"temperature": 0.0, "stop": stop_tokens, "max_new_tokens": 30}
        new_completions = llm.generate(modified_questions, sampling_params)
        new_completions_text = [item["text"] for item in new_completions]
        for index, new_comp, first_com in zip(indices_to_modify, new_completions_text, first_completions):
            final_completions[index] = first_com + new_comp

    preds = final_completions

    llm.shutdown()

    os.makedirs("binary_correctness_data", exist_ok=True)
    out_df = pd.DataFrame({"question": problems, "completion": preds})
    out_df.to_csv(f"binary_correctness_data/{model_name_safe}.csv", index=False)

    end = time.time()
    runtime = end - start
    torch.save(runtime, f"binary_correctness_data/{model_name_safe}.pt")

    # Run verification on the generated completions.
    run_verification(args.model_name)


if __name__ == "__main__":
    main()

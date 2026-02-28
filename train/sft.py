import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
import trl
from datasets import load_from_disk
from transformers.trainer_utils import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_BLOCK_SIZE = 16384
INSTRUCTION_TEMPLATE = "<|im_start|>user"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-Math-7B")
    block_size: int = field(default=DEFAULT_BLOCK_SIZE)
    train_file_path: Optional[str] = field(default="selective_data/1000/Qwen__Qwen2.5-Math-7B_0.5")
    dagger: bool = field(default=False)
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed; if None, inferred from output_dir."})


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _get_seed(config: TrainingConfig, output_dir: str) -> int:
    """Infer seed from config or from output_dir (e.g. path ending in _42)."""
    if config.seed is not None:
        return config.seed
    try:
        return int(output_dir.rstrip("/").split("_")[-1])
    except (ValueError, IndexError):
        return 42


def _ensure_pad_token(tokenizer: transformers.PreTrainedTokenizerBase, model_name: str) -> None:
    if "Llama" in model_name:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "mistral" in model_name.lower():
        tokenizer.pad_token = "<|pad_token|>"
    elif "Qwen" in model_name:
        tokenizer.pad_token = "<|fim_pad|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _maybe_extend_context_length(model_config: transformers.PretrainedConfig, block_size: int) -> None:
    if model_config.max_position_embeddings < block_size:
        model_config.max_position_embeddings = block_size
        model_config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": block_size,
        }


def main() -> None:
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config: TrainingConfig
    args: trl.SFTConfig
    config, args = parser.parse_args_into_dataclasses()

    seed = _get_seed(config, args.output_dir)
    if seed != 42:
        setup_seed(seed)

    logging.info("Training config: %s", {**asdict(config), **asdict(args)})

    model_config = transformers.AutoConfig.from_pretrained(config.model_name)
    _maybe_extend_context_length(model_config, config.block_size)

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, config=model_config)

    dataset = load_from_disk(config.train_file_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    _ensure_pad_token(tokenizer, config.model_name)

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=INSTRUCTION_TEMPLATE,
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )
    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size

    eval_dataset = dataset["test"] if "test" in dataset else dataset["train"]
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        args=args,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

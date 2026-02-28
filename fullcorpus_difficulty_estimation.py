import argparse
import os

import pandas as pd
import torch
from torch import nn

from models import CustomDataset, TextMF

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_PATH = "DDCF_data/DDCF_traindata.csv"
DEFAULT_MODEL_ORDER_PATH = "DDCF_data/model_order.csv"
DEFAULT_FULL_CORPUS_PATH = "MATH/fullcorpus_math.csv"
DEFAULT_SEED_EMBEDDINGS_PATH = "DDCF_data/seed_math_embeddings.pth"
DEFAULT_FULL_EMBEDDINGS_PATH = "DDCF_data/fullcorpus_math_embeddings.pth"
DEFAULT_CHECKPOINT_PATH = "DDCF_data/correctness_predictor.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate difficulty scores for the full MATH corpus.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model family name (should match entries in model_order.csv).",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to training data CSV used to infer num_models.",
    )
    parser.add_argument(
        "--model-order-path",
        type=str,
        default=DEFAULT_MODEL_ORDER_PATH,
        help="Path to model order CSV.",
    )
    parser.add_argument(
        "--full-corpus-path",
        type=str,
        default=DEFAULT_FULL_CORPUS_PATH,
        help="Path to full corpus MATH CSV (with prompt_id).",
    )
    parser.add_argument(
        "--seed-embeddings-path",
        type=str,
        default=DEFAULT_SEED_EMBEDDINGS_PATH,
        help="Path to seed question embeddings tensor (.pth).",
    )
    parser.add_argument(
        "--full-embeddings-path",
        type=str,
        default=DEFAULT_FULL_EMBEDDINGS_PATH,
        help="Path to full-corpus question embeddings tensor (.pth).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained correctness predictor checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name_safe = args.model_name.replace("/", "__")

    train_data = pd.read_csv(args.train_data_path)
    num_models = train_data["model_id"].max() + 1

    seed_question_embeddings = torch.load(args.seed_embeddings_path, map_location=device).to(device)
    num_prompts = seed_question_embeddings.shape[0]
    text_dim = seed_question_embeddings.shape[1]

    model = TextMF(
        question_embeddings=seed_question_embeddings,
        num_prompts=num_prompts,
        alpha=0,
        num_models=num_models,
        text_dim=text_dim,
    )

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    full_question_embeddings = torch.load(args.full_embeddings_path, map_location=device).to(device)
    num_prompts = full_question_embeddings.shape[0]
    model.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
    model.Q.weight.data.copy_(full_question_embeddings)

    model = model.to(device)

    model_ids_df = pd.read_csv(args.model_order_path)
    full_corpus_df = pd.read_csv(args.full_corpus_path)

    model_id = model_ids_df[model_ids_df["model_name"] == args.model_name]["model_id"].tolist()[0]

    ques_ids = full_corpus_df["prompt_id"].tolist()

    infer_ds = CustomDataset(
        [model_id] * len(ques_ids),
        ques_ids,
        [0] * len(ques_ids),
        shuffle=False,
    )
    infer_loader = infer_ds.get_dataloaders(args.batch_size)

    infer_model = []
    infer_ques = []
    prediction = []

    for models, questions, _ in infer_loader:
        infer_model.extend(models.tolist())
        infer_ques.extend(questions.tolist())
        models, questions = models.to(device), questions.to(device)
        logits = model(models, questions, test_mode=True)
        probs = logits.softmax(dim=-1)
        pred_labels = probs[:, 1].cpu().tolist()
        prediction.extend(pred_labels)

    df = pd.DataFrame({"model_id": infer_model, "prompt_id": infer_ques, "prediction": prediction}).sort_values("prompt_id")

    mean_pred = df["prediction"].mean()
    print(f"Mean predicted correctness for {args.model_name}: {mean_pred:.4f}")

    os.makedirs("factorized_data", exist_ok=True)
    df.to_csv(f"factorized_data/{model_name_safe}.csv", index=False)


if __name__ == "__main__":
    main()

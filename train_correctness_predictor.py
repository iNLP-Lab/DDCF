import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from models import CustomDataset, TextMF

DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.01
DEFAULT_BATCH_SIZE = 1028
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_TRAIN_PATH = "DDCF_data/DDCF_traindata.csv"
DEFAULT_VAL_PATH = "DDCF_data/DDCF_valdata.csv"
DEFAULT_MODEL_SAVE_PATH = "DDCF_data/correctness_predictor.pt"
DEFAULT_EMBEDDINGS_PATH = "DDCF_data/seed_math_embeddings.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train correctness predictor model.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--train-data-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to training CSV produced by create_training_data.py",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default=DEFAULT_VAL_PATH,
        help="Path to validation CSV produced by create_training_data.py",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to question embeddings tensor (.pth).",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default=DEFAULT_MODEL_SAVE_PATH,
        help="Path to save the best correctness predictor checkpoint.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(net, test_loader, device: torch.device) -> tuple[float, float]:
    """Standard evaluator for correctness prediction"""
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for models, questions, labels in test_loader:
            # models, labels = models.to(device), labels.to(device)
            models, questions, labels = models.to(device), questions.to(device), labels.to(device)
            logits = net(models, questions, test_mode=True)
            loss = loss_fn(logits, labels)
            pred_labels = net.predict(models, questions)
            correct += (pred_labels == labels).sum().item()
            total_loss += loss.item()
            num_samples += labels.shape[0]

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    net.train()
    return mean_loss, accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    # test_data = test_data[test_data["model_id"] == 4]

    num_models = train_data["model_id"].max() + 1

    train_dataset = CustomDataset(
        train_data["model_id"].tolist(),
        train_data["prompt_id"].tolist(),
        train_data["label"].tolist(),
    )
    test_dataset = CustomDataset(
        test_data["model_id"].tolist(),
        test_data["prompt_id"].tolist(),
        test_data["label"].tolist(),
        shuffle=False,
    )

    train_loader = train_dataset.get_dataloaders(args.batch_size)
    test_loader = test_dataset.get_dataloaders(args.batch_size * 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing model...")

    question_embeddings = torch.load(args.embeddings_path).to(device)
    num_prompts = question_embeddings.shape[0]

    model = TextMF(
        question_embeddings=question_embeddings,
        num_prompts=num_prompts,
        alpha=args.alpha,
        text_dim=question_embeddings.shape[1],
        num_models=num_models,
    )

    model.to(device)
    model.device = device

    print("Training model...")

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=args.num_epochs)
    num_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best = -1.0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for models_batch, questions_batch, labels_batch in train_loader:
            models_batch, questions_batch, labels_batch = (
                models_batch.to(device),
                questions_batch.to(device),
                labels_batch.to(device),
            )

            optimizer.zero_grad()
            logits = model(models_batch, questions_batch)
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.4f}")

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
        progress_bar.update(1)

        if best < test_accuracy:
            best = test_accuracy
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Best model saved to {args.model_save_path}")

    progress_bar.close()
    print(f"Best validation accuracy: {best:.4f}")


if __name__ == "__main__":
    main()

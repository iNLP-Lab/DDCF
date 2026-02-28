import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from timm.models.layers import DropPath  # or your own implementation
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class CustomDataset(Dataset):
    def __init__(self, model_ids, questions, labels, shuffle=True):

        self.model_ids = torch.tensor(model_ids, dtype=torch.int64)
        # self.questions = questions
        self.questions = torch.tensor(questions, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.model_ids[index], self.questions[index], self.labels[index]

    def get_dataloaders(self, batch_size):
        return DataLoader(self, batch_size, shuffle=self.shuffle)


class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=0.1, drop=0.8, drop_path=0.2):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)  # or nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # zero‐init final:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, test_mode=False):
        u = self.norm(x)
        u = self.fc1(u)
        u = self.act(u)
        if not test_mode:
            u = self.drop1(u)
        u = self.fc2(u)
        if not test_mode:
            u = self.drop2(u)
            return x + self.drop_path(u)
        else:
            return x + u


class TextMF(nn.Module):
    def __init__(self, question_embeddings, alpha, num_models, num_prompts, text_dim=2560, num_classes=2):
        super(TextMF, self).__init__()
        # Model embedding network
        self.P = nn.Embedding(num_models, text_dim)

        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)

        self.model_proj = MLPBlock(text_dim)

        self.text_proj = MLPBlock(text_dim)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(text_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in model and question embedding to reduce overfitting
            p += torch.randn_like(p) * self.alpha
            q += torch.randn_like(q) * self.alpha

        p = self.model_proj(p, test_mode=test_mode)
        q = self.text_proj(q, test_mode=test_mode)
        # print(p.shape, q.shape)
        # exit()
        return self.classifier(p * q)
        # return self.classifier(torch.cat([p, q], -1))

    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True)  # During inference no noise is applied
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def get_trained_ques_embedding(self, prompt):
        self.eval()
        q = self.Q(prompt)
        q = self.text_proj(q, test_mode=True)
        return q

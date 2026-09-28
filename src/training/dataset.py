"""Task-specific datasets. Rows only carry the label of their own task (no fake joint labels)."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.settings import project_path


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts, self.labels = list(texts), list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def load_split(task: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(project_path("data", "processed", f"{task}_{split}.parquet"))


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch, strict=True)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def make_loader(df: pd.DataFrame, tokenizer, max_length: int, batch_size: int,
                shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TextDataset(df["text"].tolist(), df["label"].astype(int).tolist()),
                      batch_size=batch_size, shuffle=shuffle, generator=g,
                      collate_fn=make_collate(tokenizer, max_length))

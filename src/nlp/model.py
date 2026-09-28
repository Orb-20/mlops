"""Multi-task DistilBERT: shared encoder, sentiment head (3-way) and toxicity head (2-way).

Artifact layout (a directory, logged to MLflow as-is):
    encoder/          DistilBERT weights + config (save_pretrained)
    tokenizer/        tokenizer files
    heads.pt          state dict of the two classification heads
    metadata.json     label maps, max_length, pretrained name, pooling
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS
from src.nlp.embeddings import mean_pool

REQUIRED_ARTIFACT_FILES = ("encoder/config.json", "tokenizer", "heads.pt", "metadata.json")


class MultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden, len(SENTIMENT_LABELS))
        self.toxicity_head = nn.Linear(hidden, len(TOXICITY_LABELS))

    @classmethod
    def from_pretrained_encoder(cls, name: str, dropout: float = 0.1) -> MultiTaskModel:
        encoder = AutoModel.from_pretrained(name)
        return cls(encoder, encoder.config.hidden_size, dropout)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled last hidden state: the shared representation used for drift monitoring."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return mean_pool(hidden, attention_mask)

    def forward(self, input_ids, attention_mask, task: str | None = None):
        emb = self.embed(input_ids, attention_mask)
        x = self.dropout(emb)
        out = {"embedding": emb}
        if task in (None, "sentiment"):
            out["sentiment_logits"] = self.sentiment_head(x)
        if task in (None, "toxicity"):
            out["toxicity_logits"] = self.toxicity_head(x)
        return out

    # ---- persistence -------------------------------------------------------------------------
    def save(self, directory: Path, tokenizer: PreTrainedTokenizerBase, metadata: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / "encoder")
        tokenizer.save_pretrained(directory / "tokenizer")
        torch.save(
            {"sentiment_head": self.sentiment_head.state_dict(),
             "toxicity_head": self.toxicity_head.state_dict()},
            directory / "heads.pt",
        )
        meta = {
            "sentiment_labels": list(SENTIMENT_LABELS),
            "toxicity_labels": list(TOXICITY_LABELS),
            **metadata,
        }
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: str | torch.device = "cpu"):
        directory = Path(directory)
        missing = [f for f in REQUIRED_ARTIFACT_FILES if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(f"model artifact at {directory} missing {missing}")
        encoder = AutoModel.from_pretrained(directory / "encoder")
        model = cls(encoder, encoder.config.hidden_size)
        heads = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
        model.sentiment_head.load_state_dict(heads["sentiment_head"])
        model.toxicity_head.load_state_dict(heads["toxicity_head"])
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return model.to(device).eval(), tokenizer, metadata

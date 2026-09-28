"""Batched inference. Load a model once per process and reuse it (spec §50)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config.settings import get_config
from src.data.schemas import Prediction
from src.nlp.model import MultiTaskModel

log = logging.getLogger(__name__)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BatchOutput:
    predictions: list[Prediction]
    embeddings: np.ndarray  # (n, hidden) float32
    sentiment_probs: np.ndarray  # (n, 3)
    toxicity_scores: np.ndarray  # (n,)


class Predictor:
    def __init__(self, model_dir: Path, version: str = "local", device: str | None = None):
        self.device = torch.device(device) if device else pick_device()
        self.model, self.tokenizer, self.metadata = MultiTaskModel.load(model_dir, self.device)
        self.version = version
        self.model_dir = Path(model_dir)
        cfg = get_config()
        self.max_length = self.metadata.get("max_length", cfg["model"]["max_length"])
        self.batch_size = cfg["inference"]["batch_size"]
        # per-model calibrated threshold (validation-chosen) if the artifact has one
        self.threshold = float(self.metadata.get("toxicity_threshold",
                                                 cfg["inference"]["toxicity_threshold"]))
        self.sent_labels = self.metadata["sentiment_labels"]
        self.tox_labels = self.metadata["toxicity_labels"]
        log.info("Loaded model %s from %s on %s", version, model_dir, self.device)

    @torch.inference_mode()
    def run(self, texts: list[str]) -> BatchOutput:
        embs, sprobs, tscores = [], [], []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            out = self.model(enc["input_ids"], enc["attention_mask"])
            embs.append(out["embedding"].float().cpu().numpy())
            sprobs.append(torch.softmax(out["sentiment_logits"].float(), -1).cpu().numpy())
            tscores.append(torch.softmax(out["toxicity_logits"].float(), -1)[:, 1].cpu().numpy())
        if not texts:
            hidden = self.model.encoder.config.hidden_size
            return BatchOutput([], np.zeros((0, hidden), np.float32), np.zeros((0, 3)), np.zeros(0))
        E, S, T = np.concatenate(embs), np.concatenate(sprobs), np.concatenate(tscores)
        preds = []
        for s, t in zip(S, T, strict=True):
            k = int(s.argmax())
            preds.append(Prediction(
                sentiment=self.sent_labels[k],
                sentiment_confidence=round(float(s[k]), 4),
                sentiment_probs={lbl: round(float(p), 4)
                                 for lbl, p in zip(self.sent_labels, s, strict=True)},
                toxicity_label=self.tox_labels[int(t >= self.threshold)],
                toxicity_score=round(float(t), 4),
            ))
        return BatchOutput(preds, E, S, T)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.run(texts).embeddings

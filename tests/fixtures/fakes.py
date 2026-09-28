"""Test doubles. Used only in unit tests - never in the application path."""

from __future__ import annotations

import numpy as np

from src.data.schemas import Prediction
from src.nlp.inference import BatchOutput


class RulePredictor:
    """Keyword predictor with the same `.run()` contract as src.nlp.inference.Predictor."""

    version = "fake"

    def __init__(self, toxic_words=("idiot", "clown", "kill", "stupid", "shut up", "moron"),
                 dim: int = 8):
        self.toxic_words = toxic_words
        self.dim = dim

    def run(self, texts):
        preds, embs = [], []
        for t in texts:
            low = t.lower()
            tox = any(w in low for w in self.toxic_words)
            if any(w in low for w in ("hate", "worst", "bad", "poor", "waste", "useless")):
                s = "negative"
            elif any(w in low for w in ("love", "best", "great", "thank", "amazing", "peak")):
                s = "positive"
            else:
                s = "neutral"
            preds.append(Prediction(sentiment=s, sentiment_confidence=0.9,
                                    sentiment_probs={"negative": 0.05, "neutral": 0.05,
                                                     "positive": 0.9},
                                    toxicity_label="toxic" if tox else "safe",
                                    toxicity_score=0.9 if tox else 0.1))
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            embs.append(rng.normal(size=self.dim))
        e = np.array(embs, dtype=np.float32).reshape(len(texts), self.dim)
        return BatchOutput(preds, e, np.zeros((len(texts), 3)),
                           np.array([p.toxicity_score for p in preds]))

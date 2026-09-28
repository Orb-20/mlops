"""DriftDecisionEngine: combine drift signals into one structured, auditable decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.config.settings import get_config
from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS
from src.monitoring.drift import SignalResult, categorical_drift, embedding_drift, proportions
from src.monitoring.vocabulary import vocabulary_report
from src.nlp.language import REPORTED_LANGUAGES


@dataclass
class WindowData:
    """Everything the engine needs about one window (reference or current)."""

    embeddings: np.ndarray
    languages: list[str]
    sentiments: list[str]
    toxicity_labels: list[str]
    texts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.embeddings)


@dataclass
class DriftDecision:
    drift_detected: bool
    severity: str  # none | low | medium | high
    overall_status: str  # stable | warning | drift | insufficient_data
    reasons: list[str]
    scores: dict[str, float]
    thresholds: dict[str, float]
    sample_count: int
    weighted_score: float
    retraining_recommended: bool
    signals: dict[str, dict]
    vocabulary: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


REASON_NAMES = {
    "embedding": "embedding_drift",
    "language": "language_distribution_shift",
    "prediction": "prediction_distribution_shift",
}


class DriftDecisionEngine:
    def __init__(self, cfg: dict | None = None, seed: int | None = None):
        full = get_config()
        self.cfg = cfg or full["drift"]
        self.seed = full["project"]["seed"] if seed is None else seed

    def evaluate(self, reference: WindowData, current: WindowData) -> DriftDecision:
        c = self.cfg
        wr = c["warning_ratio"]
        n = len(current)
        signals: dict[str, SignalResult] = {}

        if n < c["minimum_samples"]:
            return DriftDecision(
                False,
                "none",
                "insufficient_data",
                [f"only {n} samples (< minimum_samples={c['minimum_samples']})"],
                {},
                {},
                n,
                0.0,
                False,
                {},
            )

        if c["embedding"]["enabled"]:
            signals["embedding"] = embedding_drift(
                reference.embeddings, current.embeddings, c["embedding"], wr, self.seed
            )
        if c["language"]["enabled"]:
            langs = list(REPORTED_LANGUAGES)
            signals["language"] = categorical_drift(
                "language",
                {"language": proportions(reference.languages, langs)},
                {"language": proportions(current.languages, langs)},
                c["language"],
                wr,
            )
        if c["prediction"]["enabled"]:
            s, t = list(SENTIMENT_LABELS), list(TOXICITY_LABELS)
            signals["prediction"] = categorical_drift(
                "prediction",
                {
                    "sentiment": proportions(reference.sentiments, s),
                    "toxicity": proportions(reference.toxicity_labels, t),
                },
                {
                    "sentiment": proportions(current.sentiments, s),
                    "toxicity": proportions(current.toxicity_labels, t),
                },
                c["prediction"],
                wr,
            )

        total_w = sum(c[k]["weight"] for k in signals) or 1.0
        weighted = sum(c[k]["weight"] * sig.drifted for k, sig in signals.items()) / total_w
        reasons = [REASON_NAMES[k] for k, sig in signals.items() if sig.drifted]
        emb_drift = signals.get("embedding") is not None and signals["embedding"].drifted
        drift = weighted >= c["overall_threshold"]
        if drift:
            severity = "high" if emb_drift and len(reasons) >= 2 else "medium"
            status = "drift"
        elif reasons or any(s.status == "warning" for s in signals.values()):
            severity, status = "low", "warning"
        else:
            severity, status = "none", "stable"

        vocab = None
        if c.get("vocabulary", {}).get("enabled") and reference.texts and current.texts:
            vocab = vocabulary_report(
                reference.texts,
                current.texts,
                c["vocabulary"]["top_k"],
                c["vocabulary"]["min_count"],
            )

        return DriftDecision(
            drift_detected=drift,
            severity=severity,
            overall_status=status,
            reasons=reasons,
            scores={k: s.score for k, s in signals.items()},
            thresholds={k: s.threshold for k, s in signals.items()},
            sample_count=n,
            weighted_score=round(weighted, 4),
            retraining_recommended=drift,
            signals={k: s.to_dict() for k, s in signals.items()},
            vocabulary=vocab,
        )

"""Aggregate statistics for an analysed batch. Every number comes from observed data or
model predictions; the executive summary is generated from these numbers only."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.data.schemas import SENTIMENT_LABELS, CommentRecord, Prediction
from src.nlp.language import LanguageResult, language_distribution

HIGH_RISK_TOXICITY = 0.9


def _dist(values: list[str], labels) -> dict[str, float]:
    n = len(values)
    c = Counter(values)
    return {lbl: round(c.get(lbl, 0) / n, 4) if n else 0.0 for lbl in labels}


def compute_summary(
    comments: list[CommentRecord],
    langs: list[LanguageResult],
    preds: list[Prediction],
    n_buckets: int = 10,
) -> dict:
    n = len(preds)
    sentiments = [p.sentiment for p in preds]
    tox = [p.toxicity_label for p in preds]
    scores = [p.toxicity_score for p in preds]
    summary = {
        "comment_count": n,
        "sentiment": _dist(sentiments, SENTIMENT_LABELS),
        "sentiment_avg_confidence": (
            round(sum(p.sentiment_confidence for p in preds) / n, 4) if n else 0.0
        ),
        "toxicity": _dist(tox, ("safe", "toxic")),
        "toxicity_avg_score": round(sum(scores) / n, 4) if n else 0.0,
        "high_risk_count": sum(s >= HIGH_RISK_TOXICITY for s in scores),
        "high_risk_threshold": HIGH_RISK_TOXICITY,
        "languages": language_distribution([r.language for r in langs]),
        "language_count": len({r.language for r in langs}),
        "trend": _trend(comments, preds, n_buckets),
    }
    summary["executive_summary"] = executive_summary(summary)
    return summary


def _trend(comments: list[CommentRecord], preds: list[Prediction], n_buckets: int) -> list[dict]:
    """Sentiment/toxicity by chronological bucket (timestamp if available, else arrival order)."""
    if not preds:
        return []
    df = pd.DataFrame(
        {
            "ts": [c.timestamp for c in comments],
            "sent": [p.sentiment for p in preds],
            "tox": [p.toxicity_score for p in preds],
            "toxic": [p.toxicity_label == "toxic" for p in preds],
        }
    )
    has_ts = df["ts"].notna().all()
    if has_ts:
        df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    k = max(1, min(n_buckets, len(df) // 10 or 1))
    df["bucket"] = pd.qcut(df.index, k, labels=False) if k > 1 else 0
    out = []
    for b, g in df.groupby("bucket"):
        out.append(
            {
                "bucket": int(b),
                "start": g["ts"].min().isoformat() if has_ts else None,
                "end": g["ts"].max().isoformat() if has_ts else None,
                "n": int(len(g)),
                "positive": round(float((g["sent"] == "positive").mean()), 4),
                "neutral": round(float((g["sent"] == "neutral").mean()), 4),
                "negative": round(float((g["sent"] == "negative").mean()), 4),
                "toxic_rate": round(float(g["toxic"].mean()), 4),
                "avg_toxicity": round(float(g["tox"].mean()), 4),
            }
        )
    return out


LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "hinglish": "Hinglish",
    "es": "Spanish",
    "other": "Other",
}


def executive_summary(s: dict) -> str:
    n = s["comment_count"]
    if n == 0:
        return "No comments were available to analyse."
    sent = s["sentiment"]
    top, share = max(sent.items(), key=lambda kv: kv[1])
    if share >= 0.5:
        tone = f"predominantly {top}"
    else:
        second = sorted(sent.items(), key=lambda kv: -kv[1])[1][0]
        tone = f"mixed (mostly {top}, then {second})"
    langs = [
        LANG_NAMES.get(k, k)
        for k, v in sorted(s["languages"].items(), key=lambda kv: -kv[1])
        if v > 0
    ]
    return (
        f"The model predicts the audience reaction is {tone}: "
        f"{sent['positive']:.0%} positive, {sent['neutral']:.0%} neutral, "
        f"{sent['negative']:.0%} negative across {n} comments. "
        f"{s['toxicity']['toxic']:.0%} of comments are predicted toxic "
        f"({s['high_risk_count']} high-risk, score >= {s['high_risk_threshold']}). "
        f"Detected languages: {', '.join(langs[:4])}."
    )

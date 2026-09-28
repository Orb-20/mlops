"""Lightweight language detection with an explicit Hinglish (romanised Hindi-English) rule.

langdetect has no notion of Hinglish and tends to label romanised Hindi as some random
Latin-script language (so, id, tl, ...). We therefore add a transparent lexicon rule:
if a Latin-script text contains enough common romanised-Hindi function words, it is Hinglish.
This is a heuristic and is documented as such in docs/limitations.md.

Language detection is a monitoring *signal*; it is not drift detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # langdetect is non-deterministic without this

REPORTED_LANGUAGES = ("en", "hi", "hinglish", "es", "other")

HINGLISH_MARKERS = frozenset(
    """
    hai hain ho tha thi the nahi nahin nhi kya kyu kyun kaise kaisa kuch bahut bohot bhai yaar
    acha accha achha bhi aur mera meri mere tera teri tere apna apni hum tum aap ko ka ki ke se
    mein main mai hoga hogi raha rahi rahe wala wali wale sab abhi kab jab tab toh to bas matlab
    dekho dekh samajh pyaar pyar dil zindagi log logo sahi galat bilkul ekdum mast bekar bakwas
    """.split()
)
# Words that are also frequent English tokens: count them only as weak evidence.
_AMBIGUOUS = frozenset({"to", "main", "the", "me", "he", "ho", "bas", "ko", "sab"})

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[a-zA-Z]+")


@dataclass(frozen=True)
class LanguageResult:
    language: str
    confidence: float


def _hinglish_score(text: str) -> tuple[int, int]:
    words = [w.lower() for w in _LATIN_WORD.findall(text)]
    strong = sum(1 for w in words if w in HINGLISH_MARKERS and w not in _AMBIGUOUS)
    return strong, len(words)


def detect_language(text: str) -> LanguageResult:
    text = (text or "").strip()
    if not text:
        return LanguageResult("other", 0.0)

    deva = len(_DEVANAGARI.findall(text))
    letters = sum(ch.isalpha() for ch in text)
    if letters and deva / letters > 0.5:
        return LanguageResult("hi", round(deva / letters, 3))

    strong, n_words = _hinglish_score(text)
    if strong >= 2 or (n_words and n_words <= 4 and strong >= 1):
        conf = min(1.0, 0.5 + strong / max(n_words, 1))
        return LanguageResult("hinglish", round(conf, 3))

    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return LanguageResult("other", 0.0)
    lang = best.lang if best.lang in ("en", "hi", "es") else "other"
    # langdetect is unreliable on very short texts; low-confidence Latin text defaults to "en"
    # only when it is ASCII and short, and we keep the low confidence so it is visible.
    if lang == "other" and n_words <= 3 and text.isascii():
        return LanguageResult("en", round(min(best.prob, 0.5), 3))
    return LanguageResult(lang, round(float(best.prob), 3))


def detect_many(texts: list[str]) -> list[LanguageResult]:
    return [detect_language(t) for t in texts]


def language_distribution(languages: list[str]) -> dict[str, float]:
    n = len(languages)
    dist = {k: 0.0 for k in REPORTED_LANGUAGES}
    if not n:
        return dist
    for lang in languages:
        key = lang if lang in dist else "other"
        dist[key] += 1 / n
    return {k: round(v, 4) for k, v in dist.items()}

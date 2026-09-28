import pytest

from src.nlp.language import detect_language, language_distribution


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This movie is absolutely amazing, I loved every minute of it.", "en"),
        ("यह फिल्म बहुत अच्छी है", "hi"),
        ("bhai ye video bahut mast hai yaar", "hinglish"),
        ("kya bakwas hai", "hinglish"),
        ("Este video es increíble, me encantó muchísimo la música", "es"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text).language == expected


def test_empty_text_is_other():
    r = detect_language("   ")
    assert r.language == "other" and r.confidence == 0.0


def test_deterministic():
    t = "short one"
    assert detect_language(t) == detect_language(t)


def test_distribution_sums_to_one_and_buckets_unknown():
    d = language_distribution(["en", "en", "hi", "fr"])
    assert d["en"] == 0.5 and d["hi"] == 0.25 and d["other"] == 0.25
    assert abs(sum(d.values()) - 1) < 1e-6


def test_distribution_empty():
    assert all(v == 0 for v in language_distribution([]).values())

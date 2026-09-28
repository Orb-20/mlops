import numpy as np

from src.monitoring.decision import WindowData
from src.monitoring.evidently_report import run_evidently


def _window(n, lang_mix, seed):
    rng = np.random.default_rng(seed)
    langs = rng.choice(["en", "hinglish"], n, p=lang_mix).tolist()
    return WindowData(
        np.zeros((n, 4), np.float32),
        langs,
        rng.choice(["negative", "neutral", "positive"], n).tolist(),
        rng.choice(["safe", "toxic"], n, p=[0.9, 0.1]).tolist(),
        ["a comment of some length"] * n,
    )


def test_evidently_flags_language_shift_and_writes_html(tmp_path):
    ref, cur = _window(300, [0.95, 0.05], 1), _window(300, [0.5, 0.5], 2)
    out = run_evidently(ref, cur, tmp_path / "r.html")
    assert "error" not in out, out
    assert out["columns"]["language"]["p_value"] < 0.05
    assert set(out["columns"]) == {"language", "sentiment", "toxicity_label", "text_length"}
    assert (tmp_path / "r.html").stat().st_size > 10_000


def test_evidently_failure_is_reported_not_raised():
    empty = WindowData(np.zeros((0, 4)), [], [], [], [])
    out = run_evidently(empty, empty)
    assert "error" in out

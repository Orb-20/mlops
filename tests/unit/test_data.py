import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.preprocessing import clean_frame, clean_text
from src.data.schemas import CommentRecord, LabeledExample
from src.data.validation import (
    DataValidationError,
    check_no_split_leakage,
    validate_comment_frame,
    validate_labeled_frame,
)


class TestCleanText:
    def test_html_entities_and_tags(self):
        assert clean_text("it&#39;s <b>great</b><br>really") == "it's great really"

    def test_urls_and_mentions_are_masked(self):
        assert clean_text("see https://x.com/a @bob now") == "see http @user now"

    def test_whitespace_and_control_chars(self):
        assert clean_text("  a\t\nb\x00c  ") == "a b c"

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_truncation(self):
        assert len(clean_text("x" * 50, max_chars=10)) == 10

    def test_preserves_case_and_non_latin(self):
        assert clean_text("Bahut ACCHA hai 😂 बहुत") == "Bahut ACCHA hai 😂 बहुत"


def test_clean_frame_drops_short_and_duplicates():
    df = pd.DataFrame({"text": ["hello", "hello ", "a", "world"], "label": [1, 1, 0, 2]})
    out = clean_frame(df, min_chars=2)
    assert list(out["text"]) == ["hello", "world"]


class TestValidation:
    def test_valid_frame(self):
        df = pd.DataFrame({"text": ["a b", "c d"], "label": [0, 2]})
        assert validate_labeled_frame(df, "x", n_classes=3).ok

    def test_label_out_of_range(self):
        df = pd.DataFrame({"text": ["a b"], "label": [5]})
        rep = validate_labeled_frame(df, "x", n_classes=3)
        assert not rep.ok
        with pytest.raises(DataValidationError):
            rep.raise_if_failed()

    def test_missing_columns(self):
        rep = validate_labeled_frame(pd.DataFrame({"text": ["a"]}), "x", 2)
        assert "missing columns" in rep.problems[0]

    def test_duplicates_flagged(self):
        df = pd.DataFrame({"text": ["same", "same"], "label": [0, 0]})
        assert not validate_labeled_frame(df, "x", 2).ok

    def test_split_leakage(self):
        a = pd.DataFrame({"text": ["x", "y"]})
        b = pd.DataFrame({"text": ["y", "z"]})
        assert check_no_split_leakage({"a": a, "b": b}) == ["1 texts shared between a and b"]

    def test_comment_frame_duplicate_ids(self):
        df = pd.DataFrame(
            {
                "source": ["youtube"] * 2,
                "source_content_id": ["v"] * 2,
                "source_comment_id": ["c1", "c1"],
                "text": ["a", "b"],
            }
        )
        assert not validate_comment_frame(df).ok


class TestSchemas:
    def test_comment_rejects_blank_text(self):
        with pytest.raises(ValidationError):
            CommentRecord(source_content_id="v", source_comment_id="c", text="   ")

    def test_labeled_example_range(self):
        LabeledExample(text="ok", task="sentiment", label=2)
        with pytest.raises(ValidationError):
            LabeledExample(text="ok", task="toxicity", label=2)

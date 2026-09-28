"""Text normalisation shared by training, inference and monitoring (must stay identical)."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str, max_chars: int = 2000) -> str:
    """Normalise a raw comment.

    - HTML-unescape (YouTube returns `&#39;` etc.) and strip simple tags
    - Unicode NFKC, drop control characters
    - replace URLs with `http` and @mentions with `@user` (tweet_eval convention; also avoids
      carrying user handles into storage)
    - collapse whitespace, truncate
    Case is preserved; the tokenizer handles lower-casing.
    """
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = re.sub(r"<br\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = unicodedata.normalize("NFKC", t)
    t = _CTRL_RE.sub(" ", t)
    t = _URL_RE.sub("http", t)
    t = _USER_RE.sub("@user", t)
    t = _WS_RE.sub(" ", t).strip()
    return t[:max_chars]


def clean_frame(
    df: pd.DataFrame, text_col: str = "text", min_chars: int = 2, max_chars: int = 2000
) -> pd.DataFrame:
    """Clean text, drop too-short rows and exact duplicates (after cleaning)."""
    out = df.copy()
    out[text_col] = out[text_col].map(lambda s: clean_text(s, max_chars))
    out = out[out[text_col].str.len() >= min_chars]
    out = out.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return out

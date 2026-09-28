"""Data-quality checks. Each check returns a list of human-readable problems (empty = OK)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    name: str
    rows: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise DataValidationError(f"{self.name}: " + "; ".join(self.problems))


def validate_labeled_frame(
    df: pd.DataFrame, name: str, n_classes: int, min_rows: int = 1, max_dup_ratio: float = 0.0
) -> ValidationReport:
    rep = ValidationReport(name=name, rows=len(df))
    missing = {"text", "label"} - set(df.columns)
    if missing:
        rep.problems.append(f"missing columns {sorted(missing)}")
        return rep
    if len(df) < min_rows:
        rep.problems.append(f"only {len(df)} rows (< {min_rows})")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        rep.problems.append("empty or null text")
    if df["label"].isna().any():
        rep.problems.append("null labels")
    else:
        bad = set(df["label"].astype(int).unique()) - set(range(n_classes))
        if bad:
            rep.problems.append(f"labels out of range: {sorted(bad)}")
    if len(df):
        dup = df["text"].duplicated().mean()
        if dup > max_dup_ratio:
            rep.problems.append(f"duplicate ratio {dup:.3f} > {max_dup_ratio}")
    return rep


def validate_comment_frame(df: pd.DataFrame, name: str = "comments") -> ValidationReport:
    rep = ValidationReport(name=name, rows=len(df))
    required = {"source", "source_content_id", "source_comment_id", "text"}
    missing = required - set(df.columns)
    if missing:
        rep.problems.append(f"missing columns {sorted(missing)}")
        return rep
    if df["source_comment_id"].duplicated().any():
        rep.problems.append("duplicate source_comment_id")
    if (df["text"].astype(str).str.strip() == "").any():
        rep.problems.append("empty text")
    return rep


def check_no_split_leakage(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Texts must not appear in more than one split."""
    problems = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]["text"]) & set(splits[b]["text"])
            if overlap:
                problems.append(f"{len(overlap)} texts shared between {a} and {b}")
    return problems

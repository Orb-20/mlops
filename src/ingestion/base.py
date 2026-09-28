"""Source-agnostic ingestion interface. YouTube is the only live adapter in scope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.data.schemas import CommentRecord, DataMode


class IngestionError(RuntimeError):
    """Base for user-actionable ingestion failures (message is safe to show in the UI)."""


class InvalidURLError(IngestionError):
    pass


class MissingAPIKeyError(IngestionError):
    pass


class QuotaExceededError(IngestionError):
    pass


class CommentsDisabledError(IngestionError):
    pass


class ContentNotFoundError(IngestionError):
    pass


class SourceUnavailableError(IngestionError):
    pass


@dataclass
class FetchResult:
    content_id: str
    comments: list[CommentRecord]
    mode: DataMode
    source: str
    skipped_malformed: int = 0


class SocialMediaAdapter(ABC):
    source: str

    @abstractmethod
    def validate_url(self, url: str) -> bool: ...

    @abstractmethod
    def extract_content_id(self, url: str) -> str: ...

    @abstractmethod
    def fetch_comments(self, content_id: str, max_comments: int = 300,
                       known_ids: set[str] | None = None) -> FetchResult: ...

"""YouTube Data API v3 adapter (official API only) + deterministic demo adapter.

Only comment id, parent id, text and publish time are kept. Author names/channel ids are
deliberately dropped at the adapter boundary (spec §49).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd
from pydantic import ValidationError

from src.config.settings import get_config, get_env, project_path
from src.data.preprocessing import clean_text
from src.data.schemas import CommentRecord
from src.ingestion.base import (
    CommentsDisabledError,
    ContentNotFoundError,
    FetchResult,
    InvalidURLError,
    MissingAPIKeyError,
    QuotaExceededError,
    SocialMediaAdapter,
    SourceUnavailableError,
)

log = logging.getLogger(__name__)

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
          "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
DEMO_CONTENT_ID = "demo"


def parse_video_id(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    url = url.strip()
    if _VIDEO_ID.match(url):
        return url
    if "://" not in url:
        url = "https://" + url
    try:
        p = urlparse(url)
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if host not in _HOSTS:
        return None
    candidate = None
    if host.endswith("youtu.be"):
        candidate = p.path.lstrip("/").split("/")[0]
    elif p.path == "/watch":
        candidate = (parse_qs(p.query).get("v") or [None])[0]
    else:
        parts = [x for x in p.path.split("/") if x]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            candidate = parts[1]
    return candidate if candidate and _VIDEO_ID.match(candidate) else None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class YouTubeAdapter(SocialMediaAdapter):
    source = "youtube"

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        cfg = get_config()["ingestion"]["youtube"]
        self.api_key = get_env().youtube_api_key if api_key is None else api_key
        self.base = cfg["api_base"]
        self.page_size = cfg["page_size"]
        self.max_retries = cfg["max_retries"]
        self.backoff = cfg["backoff_seconds"]
        self.client = client or httpx.Client(timeout=cfg["timeout_seconds"])

    def validate_url(self, url: str) -> bool:
        return parse_video_id(url) is not None

    def extract_content_id(self, url: str) -> str:
        vid = parse_video_id(url)
        if vid is None:
            raise InvalidURLError(f"Not a valid YouTube video URL: {url!r}")
        return vid

    # -- HTTP --------------------------------------------------------------------------------
    def _get(self, params: dict) -> dict:
        params = {**params, "key": self.api_key}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.get(f"{self.base}/commentThreads", params=params)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as exc:
                        raise SourceUnavailableError("YouTube returned malformed JSON") from exc
                reason = self._error_reason(r)
                if r.status_code == 403 and reason in ("quotaExceeded", "dailyLimitExceeded",
                                                       "rateLimitExceeded"):
                    raise QuotaExceededError("YouTube API quota exceeded; try again later "
                                             "or use demo data.")
                if r.status_code == 403 and reason == "commentsDisabled":
                    raise CommentsDisabledError("Comments are disabled for this video.")
                if r.status_code == 404 or reason in ("videoNotFound", "forbidden"):
                    raise ContentNotFoundError("Video not found, private, or unavailable.")
                if r.status_code == 400 and reason in ("keyInvalid", "badRequest"):
                    raise MissingAPIKeyError("YouTube API rejected the key (keyInvalid).")
                if r.status_code < 500 and r.status_code != 429:
                    raise SourceUnavailableError(f"YouTube API error {r.status_code}: {reason}")
                last_exc = SourceUnavailableError(f"YouTube API transient {r.status_code}")
            if attempt < self.max_retries:
                delay = self.backoff * (2**attempt)
                log.warning("YouTube request failed (%s); retry in %.1fs", last_exc, delay)
                time.sleep(delay)
        raise SourceUnavailableError(f"YouTube API unreachable after retries: {last_exc}")

    @staticmethod
    def _error_reason(r: httpx.Response) -> str:
        try:
            errs = r.json().get("error", {}).get("errors", [])
            return errs[0].get("reason", "") if errs else ""
        except (ValueError, AttributeError):
            return ""

    # -- public ------------------------------------------------------------------------------
    def fetch_comments(self, content_id: str, max_comments: int = 300,
                       known_ids: set[str] | None = None) -> FetchResult:
        if not self.api_key:
            raise MissingAPIKeyError(
                "YOUTUBE_API_KEY is not set. Add it to .env for live data, or use demo data."
            )
        known_ids = known_ids or set()
        comments: list[CommentRecord] = []
        skipped = 0
        page_token: str | None = None
        log.info("Fetching YouTube comments for %s (max %d)", content_id, max_comments)
        while len(comments) < max_comments:
            params = {"part": "snippet,replies", "videoId": content_id, "textFormat": "plainText",
                      "order": "time",
                      "maxResults": min(self.page_size, 100)}
            if page_token:
                params["pageToken"] = page_token
            data = self._get(params)
            items = data.get("items")
            if not isinstance(items, list):
                raise SourceUnavailableError("Malformed YouTube response: no 'items' list")
            for rec in self._records_from_thread(items, content_id):
                if rec is None:
                    skipped += 1
                elif rec.source_comment_id not in known_ids:
                    comments.append(rec)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        comments = comments[:max_comments]
        log.info("Retrieved %d new comments (%d malformed skipped)", len(comments), skipped)
        return FetchResult(content_id, comments, "live", self.source, skipped)

    def _records_from_thread(self, items: list, content_id: str):
        for item in items:
            try:
                top = item["snippet"]["topLevelComment"]
                yield self._record(top, content_id, None)
                for reply in (item.get("replies") or {}).get("comments", []):
                    yield self._record(reply, content_id, top["id"])
            except (KeyError, TypeError):
                yield None

    @staticmethod
    def _record(c: dict, content_id: str, parent: str | None) -> CommentRecord | None:
        sn = c.get("snippet", {})
        text = clean_text(sn.get("textDisplay") or sn.get("textOriginal") or "")
        try:
            return CommentRecord(source="youtube", source_content_id=content_id,
                                 source_comment_id=c["id"], parent_comment_id=parent, text=text,
                                 timestamp=_parse_ts(sn.get("publishedAt")))
        except (ValidationError, KeyError):
            return None


class DemoAdapter(SocialMediaAdapter):
    """Deterministic offline source backed by data/sample/youtube_comments.csv.

    Everything it returns is labelled mode="demo" and must be shown as DEMO DATA.
    """

    source = "youtube"

    def __init__(self, path: Path | None = None):
        self.path = path or project_path("data", "sample", "youtube_comments.csv")

    def validate_url(self, url: str) -> bool:
        return True

    def extract_content_id(self, url: str) -> str:
        vid = parse_video_id(url or "")
        return vid or DEMO_CONTENT_ID

    def fetch_comments(self, content_id: str, max_comments: int = 300,
                       known_ids: set[str] | None = None) -> FetchResult:
        df = pd.read_csv(self.path)
        known_ids = known_ids or set()
        out, skipped = [], 0
        for row in df.itertuples(index=False):
            cid = f"{content_id}:{row.comment_id}"
            if cid in known_ids:
                continue
            try:
                out.append(CommentRecord(
                    source="youtube", source_content_id=content_id, source_comment_id=cid,
                    parent_comment_id=None, text=clean_text(str(row.text)),
                    timestamp=_parse_ts(str(row.published_at)),
                ))
            except ValidationError:
                skipped += 1
            if len(out) >= max_comments:
                break
        return FetchResult(content_id, out, "demo", self.source, skipped)

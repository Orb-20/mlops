import httpx
import pytest

from src.ingestion.base import (
    CommentsDisabledError,
    ContentNotFoundError,
    InvalidURLError,
    MissingAPIKeyError,
    QuotaExceededError,
    SourceUnavailableError,
)
from src.ingestion.youtube import DemoAdapter, YouTubeAdapter, parse_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"youtube.com/watch?feature=share&v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/embed/{VID}",
        VID,
    ],
)
def test_parse_valid(url):
    assert parse_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "hello",
        "https://vimeo.com/123",
        f"https://evil.com/watch?v={VID}",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/channel/UCabc",
        None,
    ],
)
def test_parse_invalid(url):
    assert parse_video_id(url) is None


def test_extract_raises_on_invalid():
    with pytest.raises(InvalidURLError):
        YouTubeAdapter(api_key="k").extract_content_id("https://example.com")


def _thread(cid, text, replies=()):
    item = {
        "snippet": {
            "topLevelComment": {
                "id": cid,
                "snippet": {
                    "textDisplay": text,
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "authorDisplayName": "should-not-be-stored",
                },
            }
        }
    }
    if replies:
        item["replies"] = {
            "comments": [
                {"id": rid, "snippet": {"textDisplay": rt, "publishedAt": "2026-01-01T01:00:00Z"}}
                for rid, rt in replies
            ]
        }
    return item


def _adapter(handler, key="test-key"):
    a = YouTubeAdapter(api_key=key, client=httpx.Client(transport=httpx.MockTransport(handler)))
    a.backoff = 0
    return a


def test_pagination_replies_and_max():
    pages = {
        None: {
            "items": [_thread("c1", "great video", [("r1", "agreed")]), _thread("c2", "meh")],
            "nextPageToken": "p2",
        },
        "p2": {"items": [_thread("c3", "third"), _thread("c4", "fourth")]},
    }

    def handler(req):
        assert req.url.params["key"] == "test-key"
        return httpx.Response(200, json=pages[req.url.params.get("pageToken")])

    res = _adapter(handler).fetch_comments(VID, max_comments=4)
    assert [c.source_comment_id for c in res.comments] == ["c1", "r1", "c2", "c3"]
    assert res.comments[1].parent_comment_id == "c1"
    assert res.mode == "live"
    assert "author" not in res.comments[0].model_dump_json()


def test_known_ids_are_skipped_for_incremental_monitoring():
    def handler(req):
        return httpx.Response(200, json={"items": [_thread("c1", "a b"), _thread("c2", "c d")]})

    res = _adapter(handler).fetch_comments(VID, known_ids={"c1"})
    assert [c.source_comment_id for c in res.comments] == ["c2"]


def test_malformed_items_counted_not_crashing():
    def handler(req):
        return httpx.Response(
            200, json={"items": [{"bad": 1}, _thread("c1", "ok"), _thread("c2", "   ")]}
        )

    res = _adapter(handler).fetch_comments(VID)
    assert len(res.comments) == 1 and res.skipped_malformed == 2


def test_missing_items_is_malformed_response():
    res_handler = lambda req: httpx.Response(200, json={"kind": "x"})  # noqa: E731
    with pytest.raises(SourceUnavailableError):
        _adapter(res_handler).fetch_comments(VID)


def test_empty_comments():
    res = _adapter(lambda req: httpx.Response(200, json={"items": []})).fetch_comments(VID)
    assert res.comments == []


def _err(status, reason):
    return lambda req: httpx.Response(status, json={"error": {"errors": [{"reason": reason}]}})


@pytest.mark.parametrize(
    "status,reason,exc",
    [
        (403, "quotaExceeded", QuotaExceededError),
        (403, "commentsDisabled", CommentsDisabledError),
        (404, "videoNotFound", ContentNotFoundError),
        (400, "keyInvalid", MissingAPIKeyError),
    ],
)
def test_api_errors(status, reason, exc):
    with pytest.raises(exc):
        _adapter(_err(status, reason)).fetch_comments(VID)


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"items": [_thread("c1", "ok ok")]})

    res = _adapter(handler).fetch_comments(VID)
    assert calls["n"] == 3 and len(res.comments) == 1


def test_network_failure_exhausts_retries():
    def handler(req):
        raise httpx.ConnectError("down")

    with pytest.raises(SourceUnavailableError):
        _adapter(handler).fetch_comments(VID)


def test_missing_key():
    with pytest.raises(MissingAPIKeyError):
        YouTubeAdapter(api_key="").fetch_comments(VID)


def test_demo_adapter_is_labelled_demo_and_deterministic():
    a = DemoAdapter()
    r1, r2 = a.fetch_comments("demo", 50), a.fetch_comments("demo", 50)
    assert r1.mode == "demo"
    assert [c.text for c in r1.comments] == [c.text for c in r2.comments]
    assert len(r1.comments) == 50

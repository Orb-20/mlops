"""Page 1 — Analyze Video."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    LANG_COLORS,
    LANG_NAMES,
    SENTIMENT_COLORS,
    TOXICITY_COLORS,
    APIError,
    api,
    dist_bars,
    page_header,
    status_badge,
    trend_lines,
)

page_header(
    "Adaptive NLP Monitoring",
    "Sentiment & toxicity analysis of YouTube comments, with embedding-drift "
    "monitoring and gated model retraining.",
)

try:
    health = api("GET", "/health", timeout=15)
except APIError as exc:
    st.error(f"Backend unavailable. {exc}")
    st.stop()

if not health["production_model"]:
    st.warning(
        "No production model is registered yet. Train and promote one first "
        "(see README → Training)."
    )

# ------------------------------------------------------------------ input
with st.container(border=True):
    live_ok = health["youtube_api_configured"]
    c1, c2 = st.columns([3, 1])
    url = c1.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID", disabled=not live_ok
    )
    source = c2.radio(
        "Data source",
        ["Demo data", "Live YouTube"],
        index=1 if live_ok else 0,
        disabled=not live_ok,
        horizontal=False,
    )
    if not live_ok:
        st.info(
            "YOUTUBE_API_KEY is not configured, so live YouTube analysis is unavailable. "
            "Using the bundled **synthetic demo dataset**. "
            "Add the key to `.env` for live data."
        )
    max_comments = st.slider("Maximum comments", 50, 1000, 300, step=50)
    b1, b2, _ = st.columns([1, 1, 3])
    analyze = b1.button("Analyze Video", type="primary", use_container_width=True)
    monitor = b2.button(
        "Monitor Video",
        use_container_width=True,
        help="Poll for new comments periodically and track drift",
    )

mode = "live" if (live_ok and source == "Live YouTube") else "demo"
if analyze:
    if mode == "live" and not url.strip():
        st.error("Enter a YouTube URL for live analysis.")
    else:
        with st.spinner("Fetching comments, running inference and drift checks…"):
            try:
                st.session_state["result"] = api(
                    "POST",
                    "/api/v1/analyze/youtube",
                    json={"url": url or None, "mode": mode, "max_comments": max_comments},
                )
            except APIError as exc:
                st.error(str(exc))
if monitor:
    try:
        m = api("POST", "/api/v1/monitor/start", json={"url": url or None, "mode": mode})
        st.success(
            f"Monitoring `{m['content_id']}` every {m['poll_interval_seconds']} s "
            f"({m['mode']} data). See the MLOps Monitoring page."
        )
    except APIError as exc:
        st.error(str(exc))

# Re-open a previous analysis
with st.expander("Open a previous analysis"):
    try:
        past = api("GET", "/api/v1/analyses?limit=30")
    except APIError:
        past = []
    if past:
        opts = {
            f"#{a['id']} · {a['content_id']} · {a['mode']} · {a['comment_count']} comments "
            f"· {a['created_at'][:16]}": a["id"]
            for a in past
        }
        pick = st.selectbox("Analysis", list(opts))
        if st.button("Open"):
            d = api("GET", f"/api/v1/analyses/{opts[pick]}")
            s = d["summary"]
            st.session_state["result"] = {
                "analysis_id": d["analysis_id"],
                "content_id": d["content_id"],
                "source": d["source"],
                "data_mode": d["data_mode"],
                "data_label": "LIVE YOUTUBE DATA" if d["data_mode"] == "live" else "DEMO DATA",
                "comment_count": d["comment_count"],
                "model_version": d["model_version"],
                "sentiment": s["sentiment"],
                "toxicity": s["toxicity"],
                "languages": s["languages"],
                "summary": s,
                "report_url": d["report_url"],
                "drift": {
                    "status": d["drift"].get("overall_status", "n/a"),
                    "scores": d["drift"].get("scores", {}),
                    "thresholds": d["drift"].get("thresholds", {}),
                    "reasons": d["drift"].get("reasons", []),
                    "sample_count": d["drift"].get("sample_count", 0),
                    "vocabulary": d["drift"].get("vocabulary"),
                    "event_id": d["drift_event_id"],
                },
            }
    else:
        st.caption("No analyses yet.")

res = st.session_state.get("result")
if not res:
    st.stop()

s = res["summary"]
# ------------------------------------------------------------------ overview
label = res["data_label"]
color = "#2a78d6" if res["data_mode"] == "live" else "#52514e"
st.markdown(
    f"<span style='background:{color};color:white;padding:3px 10px;border-radius:4px;"
    f"font-weight:700;letter-spacing:.05em'>{label}</span>",
    unsafe_allow_html=True,
)
st.markdown("### Video overview")
o = st.columns(5)
o[0].metric("Video ID", res["content_id"])
o[1].metric("Comments analysed", res["comment_count"])
o[2].metric("Languages detected", s["language_count"])
o[3].metric("Production model", f"v{res['model_version']}")
ts = s.get("analysed_at", "")
o[4].metric(
    "Analysed (UTC)", ts[11:16], help=ts[:10], delta=ts[:10], delta_color="off", delta_arrow="off"
)
st.info(s["executive_summary"])

# ------------------------------------------------------------------ sentiment / toxicity
left, right = st.columns(2)
with left:
    st.markdown(
        "### Sentiment  <small style='color:#52514e'>model prediction</small>",
        unsafe_allow_html=True,
    )
    m = st.columns(4)
    m[0].metric("Positive", f"{s['sentiment']['positive']:.0%}")
    m[1].metric("Neutral", f"{s['sentiment']['neutral']:.0%}")
    m[2].metric("Negative", f"{s['sentiment']['negative']:.0%}")
    m[3].metric("Avg confidence", f"{s['sentiment_avg_confidence']:.0%}")
    st.plotly_chart(dist_bars(s["sentiment"], SENTIMENT_COLORS), use_container_width=True)
with right:
    st.markdown(
        "### Toxicity  <small style='color:#52514e'>model prediction</small>",
        unsafe_allow_html=True,
    )
    m = st.columns(4)
    m[0].metric("Safe", f"{s['toxicity']['safe']:.0%}")
    m[1].metric("Toxic", f"{s['toxicity']['toxic']:.0%}")
    m[2].metric("Avg score", f"{s['toxicity_avg_score']:.3f}")
    m[3].metric(
        "High-risk", s["high_risk_count"], help=f"toxicity score ≥ {s['high_risk_threshold']}"
    )
    st.plotly_chart(dist_bars(s["toxicity"], TOXICITY_COLORS), use_container_width=True)

trend = pd.DataFrame(s.get("trend", []))
if len(trend) >= 2:
    basis = "comment time" if trend["start"].notna().all() else "arrival order"
    trend["window"] = (
        pd.to_datetime(trend["start"]).dt.strftime("%m-%d %H:%M")
        if basis == "comment time"
        else trend["bucket"] + 1
    )
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Sentiment over {basis}**")
        st.plotly_chart(
            trend_lines(
                trend,
                "window",
                {
                    "positive": SENTIMENT_COLORS["positive"],
                    "negative": SENTIMENT_COLORS["negative"],
                },
                "share of comments",
            ),
            use_container_width=True,
        )
    with t2:
        st.markdown(f"**Toxic share over {basis}**")
        st.plotly_chart(
            trend_lines(trend, "window", {"toxic_rate": TOXICITY_COLORS["toxic"]}, "toxic share"),
            use_container_width=True,
        )

# ------------------------------------------------------------------ language / vocabulary
left, right = st.columns(2)
with left:
    st.markdown(
        "### Language  <small style='color:#52514e'>observed (heuristic detector)</small>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(dist_bars(res["languages"], LANG_COLORS, LANG_NAMES), use_container_width=True)
with right:
    st.markdown(
        "### Emerging vocabulary  <small style='color:#52514e'>vs model reference" "</small>",
        unsafe_allow_html=True,
    )
    v = res["drift"].get("vocabulary") or {}
    if v:
        st.caption(
            f"Out-of-reference vocabulary: **{v.get('oov_rate', 0):.0%}** of tokens. "
            "A new phrase is not evidence of toxicity."
        )
        tabs = st.tabs(["Frequent", "New", "Biggest increases"])
        tabs[0].dataframe(
            pd.DataFrame(v.get("top_current", [])[:15]), hide_index=True, use_container_width=True
        )
        tabs[1].dataframe(
            pd.DataFrame(v.get("new_terms", [])[:15]), hide_index=True, use_container_width=True
        )
        tabs[2].dataframe(
            pd.DataFrame(v.get("frequency_shifts", [])[:15]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Vocabulary comparison unavailable (too few samples).")

# ------------------------------------------------------------------ health / drift
left, right = st.columns(2)
with left:
    st.markdown("### Model health")
    try:
        mi = api("GET", "/api/v1/model", timeout=15)
        st.markdown(
            f"**Model:** sentiment-toxicity-v{mi['version']}  \n"
            f"**Status:** {mi['status'].title()}  \n"
            f"**Last updated:** {str(mi['created'])[:16].replace('T', ' ')} UTC  \n"
            f"**Dataset version:** `{mi['dataset_version']}`  ·  "
            f"**Git commit:** `{mi['git_commit']}`"
        )
    except APIError as exc:
        st.warning(str(exc))
with right:
    st.markdown(
        "### Drift status  <small style='color:#52514e'>drift signal</small>",
        unsafe_allow_html=True,
    )
    d = res["drift"]
    rows = []
    for sig in ("embedding", "language", "prediction"):
        if sig in d["scores"]:
            sc, th = d["scores"][sig], d["thresholds"][sig]
            st_ = "drift" if sc >= th else ("warning" if sc >= 0.6 * th else "stable")
            if sig == "embedding" and sc >= th and "embedding_drift" not in d["reasons"]:
                st_ = "warning"  # above effect size but not statistically significant
            rows.append((sig.title(), sc, th, st_))
    for name, sc, th, st_ in rows:
        st.markdown(
            f"{name} drift: {status_badge(st_)} &nbsp; <small>score {sc:.4f} · "
            f"threshold {th}</small>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"**Overall:** {status_badge(d['status'])} &nbsp;"
        f"<small>{d['sample_count']} samples</small>",
        unsafe_allow_html=True,
    )
    if d["status"] == "drift":
        st.warning(
            "Drift detected → a retraining event was queued. A new model is deployed only "
            "if it passes evaluation, regression and production-comparison gates."
        )

# ------------------------------------------------------------------ report
st.markdown("### Report")
rid = res["analysis_id"]
try:
    html = api("GET", f"/api/v1/report/{rid}")
    md = api("GET", f"/api/v1/report/{rid}?format=md")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button(
        "Download HTML report", html, f"analysis_{rid}.html", "text/html", use_container_width=True
    )
    c2.download_button(
        "Download Markdown", md, f"analysis_{rid}.md", "text/markdown", use_container_width=True
    )
except APIError as exc:
    st.warning(f"Report unavailable: {exc}")

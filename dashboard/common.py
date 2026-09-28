"""Dashboard helpers: API client + chart builders. The dashboard talks to FastAPI only
(same service layer as the CLI); it never runs models itself."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Colour by job (validated with the dataviz palette checker; light theme pinned):
# sentiment = polarity (red - gray midpoint - blue), languages = identity (fixed slots).
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#8f8e8a", "positive": "#2a78d6"}
TOXICITY_COLORS = {"safe": "#2a78d6", "toxic": "#e34948"}
LANG_COLORS = {"en": "#2a78d6", "hi": "#eb6834", "hinglish": "#1baf7a", "es": "#eda100",
               "other": "#8f8e8a"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "es": "Spanish",
              "other": "Other"}
STATUS = {  # reserved status palette, always icon + label
    "stable": ("#0ca30c", "✅", "Stable"),
    "warning": ("#fab219", "⚠️", "Warning"),
    "drift": ("#d03b3b", "🔴", "Drift detected"),
    "insufficient_data": ("#8f8e8a", "⏸️", "Insufficient data"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


class APIError(RuntimeError):
    pass


def api(method: str, path: str, timeout: float = 600, **kw):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        raise APIError(f"API not reachable at {API_URL}: {exc}") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise APIError(f"{r.status_code}: {detail}")
    ctype = r.headers.get("content-type", "")
    return r.json() if "json" in ctype else r.text


def status_badge(status: str) -> str:
    color, icon, label = STATUS.get(status, ("#8f8e8a", "•", status))
    return (f"<span style='background:{color}1f;border:1px solid {color};color:{INK};"
            f"padding:2px 10px;border-radius:12px;font-size:0.9rem'>{icon} {label}</span>")


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=13),
                      showlegend=False, hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def dist_bars(values: dict[str, float], colors: dict[str, str],
              names: dict[str, str] | None = None, height: int | None = None) -> go.Figure:
    """Horizontal share bars, fixed category order, direct % labels (secondary encoding)."""
    keys = list(values)
    labels = [(names or {}).get(k, k.title()) for k in keys]
    vals = [values[k] for k in keys]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=[colors.get(k, "#8f8e8a") for k in keys],
        marker_line=dict(color="#fcfcfb", width=2), text=[f"{v:.1%}" for v in vals],
        textposition="outside", textfont=dict(color=INK), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, max(1e-6, max(vals)) * 1.25], tickformat=".0%", showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, height or 40 + 38 * len(keys))


def trend_lines(df: pd.DataFrame, x: str, series: dict[str, str], y_title: str,
                height: int = 260) -> go.Figure:
    fig = go.Figure()
    for col, color in series.items():
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=col.title(),
                                 line=dict(color=color, width=2), marker=dict(size=8),
                                 hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>"))
    fig = _layout(fig, height)
    fig.update_layout(showlegend=len(series) > 1, hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(tickformat=".0%", title=y_title)
    return fig


def score_with_threshold(df: pd.DataFrame, x: str, y: str, threshold: float, color: str,
                         y_title: str, height: int = 240) -> go.Figure:
    """One signal per chart (never dual axes) with its threshold as a dashed reference."""
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="lines+markers", line=dict(color=color, width=2),
        marker=dict(size=8, color=[STATUS["drift"][0] if v >= threshold else color
                                   for v in df[y]], line=dict(color="#fcfcfb", width=2)),
        hovertemplate="event %{customdata}<br>%{y:.4f}<extra></extra>",
        customdata=df["id"]))
    fig.add_hline(y=threshold, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text=f"threshold {threshold}", annotation_position="top left",
                  annotation_font_color=MUTED)
    fig = _layout(fig, height)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return fig


def score_line(df: pd.DataFrame, x: str, y: str, color: str, y_title: str,
               height: int = 240) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers",
                               line=dict(color=color, width=2), marker=dict(size=8),
                               hovertemplate="event %{x}<br>%{y:.3f}<extra></extra>"))
    fig = _layout(fig, height)
    fig.update_yaxes(title=y_title, rangemode="tozero", tickformat=".0%")
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)

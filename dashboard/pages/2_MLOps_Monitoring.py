"""Page 2 — MLOps Monitoring: production model lineage, drift history, monitored videos."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    LANG_COLORS,
    LANG_NAMES,
    SENTIMENT_COLORS,
    STATUS,
    APIError,
    _layout,
    api,
    page_header,
    score_line,
    score_with_threshold,
    status_badge,
)

st.set_page_config(page_title="MLOps Monitoring", page_icon="📈", layout="wide")
page_header("MLOps Monitoring", "Which model is serving, what it was trained on, and how the "
            "incoming data has moved relative to its reference distribution.")

try:
    model = api("GET", "/api/v1/model", timeout=15)
    events = api("GET", "/api/v1/drift/history?limit=500")
    monitors = api("GET", "/api/v1/monitor/status")
except APIError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ production model
st.markdown("### Current production model")
c = st.columns(5)
c[0].metric("Model version", f"v{model['version']}")
c[1].metric("Dataset version", model["dataset_version"] or "—")
c[2].metric("Git commit", model["git_commit"] or "—")
c[3].metric("Registered (UTC)", str(model["created"])[:16].replace("T", " "))
c[4].metric("Status", model["status"].title())
st.caption(f"MLflow run: `{model['run_id']}` — trace any prediction: prediction → model version "
           "→ MLflow run → dataset version → git commit.")

if not events:
    st.info("No drift evaluations yet. Analyse a video or run `python scripts/simulate_drift.py`.")
    st.stop()

df = pd.DataFrame(events).sort_values("id")
df["when"] = pd.to_datetime(df["created_at"]).dt.strftime("%m-%d %H:%M")

# ------------------------------------------------------------------ latest status
latest = df.iloc[-1]
st.markdown("### Latest drift evaluation")
st.markdown(f"Event #{latest['id']} · `{latest['content_id']}` · model v{latest['model_version']}"
            f" · {latest['sample_count']} samples &nbsp; {status_badge(latest['overall_status'])}"
            f" &nbsp; retraining: **{latest['retraining_status']}**", unsafe_allow_html=True)

content_ids = ["All"] + sorted(df["content_id"].unique().tolist())
pick = st.selectbox("Filter by content", content_ids)
view = df if pick == "All" else df[df["content_id"] == pick]
view = view.dropna(subset=["embedding_score"])

# ------------------------------------------------------------------ drift score charts
st.markdown("### Drift history")
st.caption("One signal per chart; dashed line = configured threshold. Red markers exceed it. "
           "Embedding drift additionally requires permutation-test p < 0.05.")
# thresholds as recorded with the most recent evaluated event (configs/drift.yaml at that time)
thr = next((t for t in reversed(df["thresholds"].tolist()) if isinstance(t, dict) and t), {})
if not thr:
    st.warning("No thresholds recorded yet.")
    st.stop()
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Embedding drift (MMD²)** — primary signal")
    st.plotly_chart(score_with_threshold(view, "id", "embedding_score", thr["embedding"],
                                         "#2a78d6", "MMD²"), use_container_width=True)
with g2:
    st.markdown("**Vocabulary change (out-of-reference token rate)**")
    st.plotly_chart(score_line(view.fillna({"vocabulary_oov_rate": 0}), "id",
                               "vocabulary_oov_rate", "#4a3aa7", "OOV rate"),
                    use_container_width=True)
    st.caption("Supporting evidence only — vocabulary change never triggers retraining alone.")
g3, g4 = st.columns(2)
with g3:
    st.markdown("**Language distribution shift (TV distance)**")
    st.plotly_chart(score_with_threshold(view, "id", "language_score", thr["language"],
                                         "#1baf7a", "TV distance"), use_container_width=True)
with g4:
    st.markdown("**Prediction distribution shift (TV distance)**")
    st.plotly_chart(score_with_threshold(view, "id", "prediction_score", thr["prediction"],
                                         "#eb6834", "TV distance"), use_container_width=True)
st.caption("x-axis: drift event id (chronological)")


def stacked(frame: pd.DataFrame, col: str, colors: dict, names: dict | None = None):
    rows = frame.dropna(subset=[col])
    fig = go.Figure()
    for k, color in colors.items():
        fig.add_trace(go.Bar(x=rows["id"].astype(str), y=[(d or {}).get(k, 0) for d in rows[col]],
                             name=(names or {}).get(k, k.title()), marker_color=color,
                             marker_line=dict(color="#fcfcfb", width=2),
                             hovertemplate=f"{(names or {}).get(k, k)}: %{{y:.1%}}<extra></extra>"))
    fig = _layout(fig, 280)
    fig.update_layout(barmode="stack", showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="drift event id", type="category")
    return fig


g5, g6 = st.columns(2)
with g5:
    st.markdown("**Language distribution per window**")
    st.plotly_chart(stacked(view, "language_distribution", LANG_COLORS, LANG_NAMES),
                    use_container_width=True)
with g6:
    st.markdown("**Predicted sentiment per window**")
    st.plotly_chart(stacked(view, "sentiment_distribution", SENTIMENT_COLORS),
                    use_container_width=True)

# ------------------------------------------------------------------ events over time
st.markdown("### Drift events")
counts = df["overall_status"].value_counts()
c = st.columns(4)
for i, key in enumerate(["stable", "warning", "drift", "insufficient_data"]):
    color, icon, label = STATUS[key]
    c[i].metric(f"{icon} {label}", int(counts.get(key, 0)))
table = df.sort_values("id", ascending=False)[
    ["id", "when", "content_id", "model_version", "sample_count", "overall_status", "severity",
     "embedding_score", "language_score", "prediction_score", "reason", "retraining_status"]]
st.dataframe(table, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ monitored videos
st.markdown("### Monitored videos")
st.caption(f"Background worker running: **{monitors['worker_running']}**")
if monitors["monitors"]:
    st.dataframe(pd.DataFrame(monitors["monitors"]), hide_index=True, use_container_width=True)
    stop = st.selectbox("Stop monitoring", [m["content_id"] for m in monitors["monitors"]
                                            if m["active"]] or ["—"])
    if st.button("Stop") and stop != "—":
        api("POST", "/api/v1/monitor/stop", json={"content_id": stop})
        st.rerun()
else:
    st.caption("No videos are being monitored. Use **Monitor Video** on the Analyze page.")

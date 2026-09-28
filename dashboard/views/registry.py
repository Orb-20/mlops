"""Page 3 — Model Registry & Retraining History: every candidate and why it was promoted or
rejected."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APIError, api, page_header  # noqa: E402

page_header(
    "Model Registry & Retraining History",
    "Drift never deploys a model directly: every candidate passes artifact, "
    "minimum-metric, regression and production-comparison gates, or it is rejected.",
)

try:
    model = api("GET", "/api/v1/model", timeout=15)
    history = api("GET", "/api/v1/retraining/history?limit=200")
except APIError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ registry
st.markdown("### Registered versions — `sentiment-toxicity-model`")
rows = []
for v in model["versions"]:
    t = v["tags"]
    rows.append(
        {
            "version": f"v{v['version']}",
            "aliases": ", ".join(v["aliases"]) or "—",
            "lifecycle": t.get("lifecycle", "—"),
            "trigger": t.get("trigger", "initial training"),
            "scenario": t.get("scenario", "—"),
            "dataset_version": t.get("dataset_version"),
            "git_commit": t.get("git_commit"),
            "run_id": v["run_id"],
            "registered (UTC)": (
                datetime.fromtimestamp(v["created"] / 1000, UTC).strftime("%Y-%m-%d %H:%M")
                if v.get("created")
                else "—"
            ),
            "rejection_reason": t.get("rejection_reason", ""),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "Aliases are the source of truth for serving: the API always loads `production`. "
    "A rejected candidate is tagged `rejected` and never receives the production alias."
)

# ------------------------------------------------------------------ retraining events
st.markdown("### Retraining history")
if not history:
    st.info(
        "No retraining has run yet. Trigger one with `python scripts/simulate_drift.py` "
        "or `python -m src.pipeline.retrain`."
    )
    st.stop()


def pct(x):
    return "—" if x is None else f"{x:.3f}"


for ev in history:
    decision = ev["decision"]
    icon = {"PROMOTE": "✅", "REJECT": "⛔", "FAILED": "💥"}.get(decision, "•")
    title = (
        f"{icon} Event #{ev['id']} · candidate v{ev['model_version']} · {decision} · "
        f"{ev['created_at'][:16].replace('T', ' ')} UTC"
    )
    with st.expander(title, expanded=ev is history[0]):
        cm, pm = ev.get("candidate_metrics") or {}, ev.get("production_metrics") or {}
        reg = ev.get("regression") or {}
        creg, preg = reg.get("candidate") or {}, reg.get("production") or {}
        details = ev.get("details") or {}
        after = details.get("production_after") if isinstance(details, dict) else None

        # lifecycle flow
        steps = [
            f"Drift event #{ev['drift_event_id']}" if ev["drift_event_id"] else "Manual",
            f"Candidate v{ev['model_version']}",
        ]
        if creg:
            steps.append(
                f"Regression tests: {'PASS' if creg.get('passed') else 'FAIL'} "
                f"({creg.get('failed')}/{creg.get('max_failed_cases')} max)"
            )
        if decision == "PROMOTE":
            steps += ["Production comparison: PASS", "Promoted"]
        elif decision == "REJECT":
            steps += ["Rejected", f"Production remains v{ev['production_version']}"]
        else:
            steps.append("Pipeline failed — production unchanged")
        st.markdown("  →  ".join(f"**{s}**" for s in steps))
        st.markdown(f"**Reason:** {ev['reason']}")

        if cm:
            keys = [
                "sentiment_macro_f1",
                "toxicity_f1",
                "toxicity_recall",
                "toxicity_precision",
                "toxicity_pr_auc",
            ]
            table = pd.DataFrame(
                {
                    "metric (held-out test sets)": keys,
                    f"candidate v{ev['model_version']}": [pct(cm.get(k)) for k in keys],
                    f"production v{ev['production_version']}": [pct(pm.get(k)) for k in keys],
                }
            )
            table.loc[len(table)] = [
                "regression failures",
                str(creg.get("failed", "—")),
                str(preg.get("failed", "—")),
            ]
            st.dataframe(table, hide_index=True, use_container_width=True)
        dt = details.get("drift_test") if isinstance(details, dict) else None
        if dt:
            st.markdown("**Held-out drifted data** (labelled controlled drift test set)")
            c, p = dt.get("candidate", {}), dt.get("production", {})
            st.dataframe(
                pd.DataFrame(
                    {
                        "metric (drift test set)": [
                            "sentiment_macro_f1",
                            "toxicity_f1",
                            "toxicity_recall",
                        ],
                        "candidate": [
                            pct(c.get(k))
                            for k in ("sentiment_macro_f1", "toxicity_f1", "toxicity_recall")
                        ],
                        "production": [
                            pct(p.get(k))
                            for k in ("sentiment_macro_f1", "toxicity_f1", "toxicity_recall")
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        gates = details.get("gates") if isinstance(details, dict) else None
        if gates:
            st.markdown("**Gates**")
            for g in gates:
                st.markdown(f"- {'✅' if g['passed'] else '❌'} `{g['name']}`")
        if creg.get("failed_ids"):
            st.caption(
                f"Failed regression cases: {', '.join(creg['failed_ids'])}"
                + (
                    f" · critical: {', '.join(creg['critical_failed_ids'])}"
                    if creg.get("critical_failed_ids")
                    else ""
                )
            )
        if after:
            st.caption(f"Production after this event: v{after}")

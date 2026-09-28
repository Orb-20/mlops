"""Analysis report (spec §9, §43): self-contained HTML (inline SVG charts) and Markdown.

Content is built only from stored rows: the Analysis summary, its DriftEvent, the model
registry tags and any ModelEvents linked to that drift event. Sections separate
OBSERVED DATA, MODEL PREDICTIONS and DRIFT SIGNALS.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from src.config.settings import get_config
from src.data.schemas import utcnow
from src.database.connection import session_scope
from src.database.models import Analysis, DriftEvent, ModelEvent

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "hinglish": "Hinglish",
    "es": "Spanish",
    "other": "Other",
}
COLORS = {
    "positive": "#2f9e6e",
    "neutral": "#8a8f98",
    "negative": "#d1495b",
    "safe": "#2f9e6e",
    "toxic": "#d1495b",
}


class ReportNotFound(LookupError):
    pass


@dataclass
class ReportData:
    analysis: dict
    drift: dict | None
    model: dict
    model_events: list[dict]
    generated_at: datetime


def load_report_data(analysis_id: int) -> ReportData:
    with session_scope() as s:
        a = s.get(Analysis, analysis_id)
        if a is None:
            raise ReportNotFound(f"analysis {analysis_id} not found")
        ev = s.get(DriftEvent, a.drift_event_id) if a.drift_event_id else None
        mevents = []
        if ev is not None:
            mevents = [
                _row(m)
                for m in s.scalars(
                    select(ModelEvent)
                    .where(ModelEvent.drift_event_id == ev.id)
                    .order_by(ModelEvent.id)
                )
            ]
        analysis, drift = _row(a), _row(ev) if ev else None
    return ReportData(analysis, drift, _model_meta(analysis["model_version"]), mevents, utcnow())


def _row(obj) -> dict:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def _model_meta(version: str) -> dict:
    meta = {"registered_model": get_config()["mlflow"]["registered_model"], "version": version}
    try:
        from src.training import registry

        mv = registry.client().get_model_version(registry.registered_name(), version)
        prod = registry.get_version_by_alias("production")
        meta.update(
            {
                "run_id": mv.run_id,
                "dataset_version": mv.tags.get("dataset_version"),
                "git_commit": mv.tags.get("git_commit"),
                "lifecycle": mv.tags.get("lifecycle"),
                "current_production": prod.version if prod else None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report must still render if MLflow is down
        meta["registry_error"] = f"MLflow unavailable: {type(exc).__name__}"
    return meta


# ---------------------------------------------------------------------------- charts (SVG)
def svg_bars(values: dict[str, float], width: int = 420, color_map: dict | None = None) -> str:
    items = list(values.items())
    bar_h, gap, label_w = 22, 8, 90
    h = len(items) * (bar_h + gap)
    parts = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" ' f'style="max-width:{width}px">'
    ]
    for i, (k, v) in enumerate(items):
        y = i * (bar_h + gap)
        w = max(0.0, min(1.0, v)) * (width - label_w - 60)
        col = (color_map or COLORS).get(k, "#4c6ef5")
        parts.append(
            f'<text x="0" y="{y + 16}" font-size="13" fill="currentColor">'
            f"{html.escape(LANG_NAMES.get(k, k))}</text>"
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
            f'fill="{col}"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + 16}" font-size="13" '
            f'fill="currentColor">{v:.1%}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_trend(
    trend: list[dict], key_colors: dict[str, str], width: int = 520, height: int = 160
) -> str:
    if len(trend) < 2:
        return "<p class='muted'>Not enough data points for a trend.</p>"
    pad = 30
    xs = [pad + i * (width - 2 * pad) / (len(trend) - 1) for i in range(len(trend))]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="#999" stroke-width="1"/>',
    ]
    for key, col in key_colors.items():
        pts = " ".join(
            f"{x:.1f},{height - pad - t[key] * (height - 2 * pad):.1f}"
            for x, t in zip(xs, trend, strict=True)
        )
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
    legend = " ".join(f'<tspan fill="{c}">■ {k}</tspan>' for k, c in key_colors.items())
    parts.append(f'<text x="{pad}" y="14" font-size="12">{legend}</text></svg>')
    return "".join(parts)


# ---------------------------------------------------------------------------- renderers
def _status_badge(status: str) -> str:
    col = {"stable": "#2f9e6e", "warning": "#e8a33d", "drift": "#d1495b"}.get(status, "#8a8f98")
    return f'<span class="badge" style="background:{col}">{html.escape(status.upper())}</span>'


def render_html(d: ReportData) -> str:
    a, s, dr, m = d.analysis, d.analysis["summary"], d.drift, d.model
    mode_label = "LIVE YOUTUBE DATA" if a["mode"] == "live" else "DEMO DATA (synthetic)"
    e = html.escape
    vocab = ((dr or {}).get("details") or {}).get("vocabulary") or {}
    rows_new = "".join(
        f"<tr><td>{e(t['term'])}</td><td>{t['count']}</td></tr>"
        for t in vocab.get("new_terms", [])[:12]
    )
    rows_top = "".join(
        f"<tr><td>{e(t['term'])}</td><td>{t['count']}</td></tr>"
        for t in vocab.get("top_current", [])[:12]
    )
    rows_shift = "".join(
        f"<tr><td>{e(t['term'])}</td><td>{t['count']}</td>" f"<td>{t['log2_ratio']:+.2f}</td></tr>"
        for t in vocab.get("frequency_shifts", [])[:12]
    )
    drift_rows = ""
    if dr:
        for key in ("embedding", "language", "prediction"):
            sig = (dr["details"].get("signals") or {}).get(key)
            if sig:
                drift_rows += (
                    f"<tr><td>{key.title()}</td><td>{sig['score']:.4f}</td>"
                    f"<td>{sig['threshold']}</td><td>{_status_badge(sig['status'])}"
                    f"</td></tr>"
                )
    ev_rows = "".join(
        f"<tr><td>{e(me['event_type'])}</td><td>v{e(str(me['model_version']))}</td>"
        f"<td><b>{e(me['decision'])}</b></td><td>{e(me['reason'])}</td></tr>"
        for me in d.model_events
    )
    retrain = (
        f"<table><tr><th>Event</th><th>Candidate</th><th>Decision</th><th>Reason</th></tr>"
        f"{ev_rows}</table>"
        if ev_rows
        else f"<p>Retraining status for this drift event: "
        f"<b>{e((dr or {}).get('retraining_status', 'n/a'))}</b>. "
        f"No candidate model has been evaluated for it yet.</p>"
    )
    trend = s.get("trend", [])
    trend_basis = "comment timestamp" if trend and trend[0].get("start") else "arrival order"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analysis Report #{a['id']}</title><style>
:root{{--bg:#fff;--fg:#1d2330;--muted:#667085;--line:#e4e7ec;--card:#f8f9fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151c;--fg:#e6e8ec;--muted:#98a2b3;
--line:#2a2f3a;--card:#1a1e27}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:900px;
margin:0 auto;padding:24px 16px}}h1{{margin:0 0 4px}}h2{{border-bottom:1px solid var(--line);
padding-bottom:4px;margin-top:32px}}.muted{{color:var(--muted)}}.badge{{color:#fff;padding:2px 8px;
border-radius:10px;font-size:12px;font-weight:600}}.tag{{display:inline-block;font-size:11px;
font-weight:700;letter-spacing:.06em;padding:2px 6px;border:1px solid var(--line);
border-radius:4px;color:var(--muted)}}table{{border-collapse:collapse;width:100%;margin:8px 0}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;font-size:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
</style></head><body>
<h1>Audience Analysis Report #{a['id']}</h1>
<p class="muted">Generated {d.generated_at:%Y-%m-%d %H:%M UTC} · Adaptive NLP MLOps Pipeline</p>
<p><span class="badge" style="background:{'#4c6ef5' if a['mode'] == 'live' else '#8a8f98'}">
{mode_label}</span></p>

<h2>Executive summary</h2>
<div class="card">{e(s['executive_summary'])}<br>
Drift status: {_status_badge((dr or {}).get('overall_status', 'n/a'))}
<p class="muted">Generated from the computed statistics below; sentiment and toxicity figures are
model predictions, not ground truth.</p></div>

<h2>1. Source &amp; observed data <span class="tag">OBSERVED DATA</span></h2>
<table><tr><td>Source</td><td>{e(a['source'])}</td></tr>
<tr><td>Content ID</td><td>{e(a['content_id'])}</td></tr>
<tr><td>Data mode</td><td>{mode_label}</td></tr>
<tr><td>Analysis timestamp</td><td>{a['created_at']:%Y-%m-%d %H:%M:%S UTC}</td></tr>
<tr><td>Comments analysed</td><td>{a['comment_count']}</td></tr>
<tr><td>Languages detected</td><td>{s['language_count']}</td></tr></table>
<h3>Language distribution</h3>
<p class="muted">Language detection is a heuristic signal (langdetect + Hinglish lexicon rule).</p>
{svg_bars(s['languages'], color_map={})}

<h2>2. Sentiment <span class="tag">MODEL PREDICTION</span></h2>
<div class="grid"><div>{svg_bars(s['sentiment'])}</div>
<div class="card">Average confidence: <b>{s['sentiment_avg_confidence']:.1%}</b></div></div>
<h3>Sentiment trend (by {trend_basis})</h3>
{svg_trend(trend, {'positive': COLORS['positive'], 'negative': COLORS['negative']})}

<h2>3. Toxicity <span class="tag">MODEL PREDICTION</span></h2>
<div class="grid"><div>{svg_bars(s['toxicity'])}</div><div class="card">
Average toxicity score: <b>{s['toxicity_avg_score']:.3f}</b><br>
High-risk comments (score ≥ {s['high_risk_threshold']}): <b>{s['high_risk_count']}</b></div></div>
<h3>Toxicity trend</h3>{svg_trend(trend, {'toxic_rate': COLORS['toxic']})}

<h2>4. Emerging vocabulary <span class="tag">DRIFT SIGNAL</span></h2>
<p class="muted">Compared with the model's reference data. A new or frequent phrase is not
evidence of toxicity.</p>
<p>Out-of-reference-vocabulary rate: <b>{vocab.get('oov_rate', 0):.1%}</b></p>
<div class="grid"><div><h3>Frequent phrases</h3><table>{rows_top}</table></div>
<div><h3>New phrases</h3><table>{rows_new or '<tr><td>none</td></tr>'}</table></div></div>
<h3>Largest frequency increases (log2 ratio vs reference)</h3><table>{rows_shift}</table>

<h2>5. Drift status <span class="tag">DRIFT SIGNAL</span></h2>
<table><tr><th>Signal</th><th>Score</th><th>Threshold</th><th>Status</th></tr>{drift_rows}
</table><p>Overall: {_status_badge((dr or {}).get('overall_status', 'n/a'))}
 · reasons: {e((dr or {}).get('reason', 'n/a'))} · samples: {(dr or {}).get('sample_count')}</p>
<p class="muted">Embedding score = MMD² (RBF kernel) vs reference; language/prediction scores =
total-variation distance. Thresholds are configurable demonstration values.</p>

<h2>6. Model health &amp; retraining</h2>
<table><tr><td>Model</td><td>{e(m['registered_model'])} v{e(str(m['version']))}</td></tr>
<tr><td>Status at report time</td><td>{e(str(m.get('lifecycle')))}
 (current production: v{e(str(m.get('current_production')))})</td></tr></table>
<h3>Retraining / candidate status</h3>{retrain}

<h2>7. Technical metadata</h2>
<table><tr><td>MLflow run</td><td><code>{e(str(m.get('run_id')))}</code></td></tr>
<tr><td>Dataset version</td><td><code>{e(str(m.get('dataset_version')))}</code></td></tr>
<tr><td>Git commit</td><td><code>{e(str(m.get('git_commit')))}</code></td></tr>
<tr><td>Analysis id / drift event id</td><td>{a['id']} / {a.get('drift_event_id')}</td></tr>
{'<tr><td>Registry</td><td>' + e(m['registry_error']) + '</td></tr>'
 if 'registry_error' in m else ''}</table>
<p class="muted">No commenter identities are stored or shown in this report.</p>
</body></html>"""


def render_markdown(d: ReportData) -> str:
    a, s, dr, m = d.analysis, d.analysis["summary"], d.drift, d.model
    mode = "LIVE YOUTUBE DATA" if a["mode"] == "live" else "DEMO DATA (synthetic)"

    def pct(dct: dict) -> str:
        return " · ".join(f"{LANG_NAMES.get(k, k)} {v:.1%}" for k, v in dct.items())

    lines = [
        f"# Audience Analysis Report #{a['id']}",
        "",
        f"**{mode}** · generated {d.generated_at:%Y-%m-%d %H:%M UTC}",
        "",
        "## Executive summary",
        s["executive_summary"],
        "",
        "## Observed data",
        f"- Source: {a['source']} / `{a['content_id']}`",
        f"- Analysed: {a['created_at']:%Y-%m-%d %H:%M:%S UTC}",
        f"- Comments: {a['comment_count']}",
        f"- Languages: {pct(s['languages'])}",
        "",
        "## Model predictions",
        f"- Sentiment: {pct(s['sentiment'])} (avg confidence {s['sentiment_avg_confidence']:.1%})",
        f"- Toxicity: {pct(s['toxicity'])} (avg score {s['toxicity_avg_score']:.3f}, "
        f"high-risk {s['high_risk_count']})",
        "",
        "## Drift signals",
    ]
    if dr:
        for k in ("embedding", "language", "prediction"):
            sig = (dr["details"].get("signals") or {}).get(k)
            if sig:
                lines.append(
                    f"- {k}: {sig['score']:.4f} (threshold {sig['threshold']}) "
                    f"→ {sig['status']}"
                )
        lines.append(f"- **Overall: {dr['overall_status']}** ({dr['reason']})")
    lines += [
        "",
        "## Model",
        f"- {m['registered_model']} v{m['version']} ({m.get('lifecycle')})",
        f"- run `{m.get('run_id')}` · dataset `{m.get('dataset_version')}` · "
        f"commit `{m.get('git_commit')}`",
    ]
    if d.model_events:
        lines += ["", "## Retraining"]
        lines += [
            f"- {me['event_type']}: v{me['model_version']} → **{me['decision']}** "
            f"({me['reason']})"
            for me in d.model_events
        ]
    return "\n".join(lines) + "\n"


def generate_report(analysis_id: int, fmt: str = "html") -> str:
    d = load_report_data(analysis_id)
    return render_html(d) if fmt == "html" else render_markdown(d)

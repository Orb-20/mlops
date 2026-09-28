# Demo Script (~12 minutes)

Prerequisites: API on :8000, dashboard on :8501, a production model (`python scripts/setup_local.py`).
Keep a terminal open next to the browser.

## Demo 1 — Analysis (3 min)

1. Open http://localhost:8501 → **Analyze Video**.
2. Without a key the page says so and uses **DEMO DATA** (badge). With a key: paste a YouTube URL,
   choose *Live YouTube* (badge **LIVE YOUTUBE DATA**).
3. Click **Analyze Video**. Walk through: overview (comments, languages, model version) → executive
   summary (generated from the numbers) → sentiment and toxicity (*model prediction*) → language
   (*observed, heuristic*) → emerging vocabulary.
4. Drift status: embedding *warning*, language/prediction above threshold, overall **Warning** —
   auxiliary signals alone never trigger retraining.
5. Download the HTML report; point out OBSERVED DATA / MODEL PREDICTION / DRIFT SIGNAL tags.

## Demo 2 — Drift (3 min)

1. Open **MLOps Monitoring**: production model lineage (version → dataset version → git commit).
2. Terminal: `python scripts/simulate_drift.py --no-retrain`
   → `STABLE window: STABLE` and `SHIFTED window: DRIFT` → `DRIFT DETECTED (event N ...)`.
   (If the current production model was already retrained on this drift, the shifted window reads
   warning — explain that as the adapted state, then use Demo 4 for the gate.)
3. Refresh the page: the embedding MMD² spike crosses the dashed threshold; the drift events table
   shows `retraining_status = pending`. Open `reports/drift/evidently_shifted_*.html` for Evidently's view.

## Demo 3 — Retraining (4 min, ~5 min runtime on GPU; start it early)

1. `python -m src.pipeline.retrain` (oldest pending event).
2. Narrate the log: data quality report → DVC commit → training → evaluation → regression → comparison.
3. Output: `Candidate trained / evaluated / Regression tests executed / Promotion decision made`.
4. **Model Registry** page: the new event, its gates, candidate vs production metrics on the test sets
   and on the held-out drift set. Mention the recorded history: v2 was rejected for a recall drop
   before the threshold-calibration fix; v3 was promoted.

## Demo 4 — Safety gate (2 min; pre-run if short on time)

1. `python -m src.pipeline.retrain --force --scenario corrupted_labels`
2. Result: `REJECT (minimum_metrics_failed, regression_failed, production_comparison_failed)`,
   `Production model: v3 -> v3`; the recorded run failed 40/48 regression cases including critical
   cases r30, r46, r47, r48 (it labels almost everything toxic: recall 1.0, precision 0.10).
3. Registry page: candidate tagged `rejected`, production alias unchanged, API still serving it.

Closing line: drift triggered retraining, but only the gates decided what reached production.

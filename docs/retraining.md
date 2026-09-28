# Retraining & Promotion

```
drift event (retraining_status = pending)
  → collect labelled drift data (data/drift/train.parquet)
  → clean · dedupe · remove overlap with every evaluation set · schema + class-balance checks
  → dvc commit drift_data  (content hash also folded into dataset_version)
  → train candidate (MLflow experiment "drift-retraining", alias "candidate")
  → evaluate on standard test sets + held-out drift test set
  → regression suite
  → compare with production on the SAME data
  → PROMOTE (alias "production" moves; old version tagged "archived")
    or REJECT (tag "rejected"; production untouched)
  → model_events row + drift event status (promoted / rejected / failed)
```

Commands:

```bash
python -m src.pipeline.retrain                              # oldest pending drift event
python -m src.pipeline.retrain --event-id 6                 # a specific event
python -m src.pipeline.retrain --force                      # no drift event (manual)
python -m src.pipeline.retrain --force --scenario corrupted_labels   # safety-gate demo
python -m src.evaluation.promotion --alias candidate [--dry-run]     # gates only
```

Exit codes: 0 promote, 3 reject, 1 usage/data error.

## Data policy

The only labelled data that enters retraining is the **controlled drift dataset**
(`src/monitoring/simulation.py`): synthetic slang / meme / Hinglish templates whose labels are the
template's intended label. Templates are split between train and test so the drift-test metric measures
generalisation to unseen phrasings. Live comments are stored unlabelled and are never used as labels;
pseudo-labelling is not used. In a real deployment, this step would take a manually annotated sample of
the drifted window instead.

## Gates (`configs/gates.yaml`, `PromotionDecisionEngine`)

| Gate | Rule (current config) |
|---|---|
| 1. Artifact validity | files present, loads, inference returns valid labels and finite embeddings |
| 2. Minimum metrics | sentiment macro-F1 ≥ 0.60, toxicity F1 ≥ 0.65, toxicity recall ≥ 0.60 |
| 3. Regression suite | ≤ 15 failed cases of 48 **and** none of the 6 critical cases fails |
| 4. Production comparison | drop ≤ 0.02 macro-F1, ≤ 0.02 toxicity F1, ≤ 0.03 toxicity recall; ≤ 1 extra regression failure |

Minimums are demonstration values: tweet_eval 3-class macro-F1 is ~0.7 even for strong models, so the
spec's example 0.80 would reject every model. `max_failed_cases` was calibrated on v1 (13/48 failures).
Gate 4 is skipped only when no production model exists (bootstrap), and the decision says so
(`bootstrap_no_production_model`).

## Regression suite

`data/regression/regression_cases.csv`: 48 cases in 15 categories (sarcasm, negation, slang,
abbreviation, mixed language, profanity, neutral profanity, positive-toxicity edge cases, negative
without toxicity, toxicity without negative sentiment, ambiguous, …). Each has intended labels and a
rationale; ambiguous cases accept several labels (`negative|neutral`). Critical cases: r26 "You are an
idiot.", r30 "Go kill yourself", r31 stacked insults, r46–r48 unambiguous praise/criticism.

Observed failures shared by v1 and v3: sarcasm (r06, r07), Hinglish (r18–r20), profanity used as
praise (r23, r25, r34), affectionate insults (r32, r40).

## Demonstrated outcomes

See `mlops-lifecycle.md`: v2 REJECT (recall regression), v3 PROMOTE, v4 REJECT (corrupted labels).

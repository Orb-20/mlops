# Drift Detection

Language detection ≠ drift detection. Text can stay "English" while its meaning shifts
("absolute cinema", "mid", "fell off"). Embedding drift is therefore the primary signal.

## Signals (`src/monitoring/`)

| Signal | Statistic | Role | Threshold |
|---|---|---|---|
| Embedding | MMD² (RBF kernel, median-heuristic bandwidth) + 200-permutation test | **primary**, weight 0.70 | MMD² ≥ 0.09 **and** p < 0.05 |
| Language | total-variation distance of en/hi/hinglish/es/other shares | auxiliary, weight 0.15 | TV ≥ 0.10 |
| Prediction | max TV distance of predicted sentiment / toxicity shares | auxiliary, weight 0.15 | TV ≥ 0.10 |
| Vocabulary | new n-grams, log₂ frequency shifts, OOV rate, Jaccard | evidence only | — |
| Evidently `DataDriftPreset` | per-column tests on language / sentiment / toxicity label / length | evidence only | Evidently defaults |

**Decision** (`DriftDecisionEngine`): weighted vote of drifted signals; drift when ≥ 0.5. Because the
embedding weight is 0.70, auxiliary signals alone can raise at most a *warning* — they cannot trigger
retraining. Fewer than `minimum_samples` (100) → `insufficient_data`, never retraining.
Severity: `high` if embedding + ≥1 auxiliary drifted, else `medium`.
Status per signal: `drift` ≥ threshold, `warning` ≥ 0.6 × threshold, else `stable`.

## Why MMD and not Evidently's embedding drift

Evidently 0.7.23 (installed) does offer an `EmbeddingsDrift` metric, but the project needs a
statistic whose magnitude can be calibrated as an *effect size* and reproduced exactly. MMD² gives a
magnitude (0 iff distributions match, for a characteristic kernel) plus a permutation p-value, is
deterministic under a seed, and is ~40 lines of numpy that the unit tests pin down. Evidently runs in
the same pipeline on the tabular view of each window, and its report is saved as supporting evidence
(`reports/drift/evidently_*.html`).

Observed: after adaptation (v3), Evidently still flags all four columns of the slang window as drifted
(p-values down to 1e-42) while MMD² is 0.037. With n = 300, pure significance tests flag any small
difference; that is why the decision votes on calibrated effect sizes.

## Threshold calibration (not universal constants)

Model v1, reference = 500 validation texts (250 tweets + 250 Jigsaw comments), 200 permutations:

| Window | n | MMD² | p |
|---|---|---|---|
| held-out same distribution | 300 / 400 / 600 | −0.001 … 0.000 | 0.46 – 0.94 |
| synthetic demo YouTube comments | 300 | 0.063 | 0.005 |
| controlled slang / Hinglish window | 300 | 0.117 | 0.005 |

0.09 separates "same distribution" from "intended drift" and places the demo data (a real domain
difference from tweets/Wikipedia) in the *warning* band. With another model or domain, recalibrate:
run `scripts/simulate_drift.py --no-retrain` and inspect the stable/shifted scores.

## Reference windows

- Default: seeded 500-text sample of the validation splits, embedded by the model being monitored
  (embeddings from different encoders are not comparable), cached at `artifacts/reference/v<N>.npz`.
- After drift-triggered retraining the model ships its own reference: 50% historical + 50% recent
  drift data, so the new normal is not re-flagged forever. Consequence (observed): the purely
  historical window now reads *warning* (MMD² 0.075) against v3's mixed reference.

## Monitoring window

On-demand analysis: the analysed batch. Continuous monitoring: rolling window of the latest
`monitoring.window_size` (200) stored comments for that video, including the new batch.

## Labels

Live comments have no ground truth. They drive drift signals only; F1/precision/recall are computed
exclusively on labelled data (test splits, drift test set, regression suite). Model predictions are
never used as training labels.

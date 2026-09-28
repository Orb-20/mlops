# Limitations (honest list)

## Unverified in this environment
- **Docker / Docker Compose / PostgreSQL**: not installed on the development machine; the image and
  stack were never built. The same SQLAlchemy code runs on SQLite locally (verified).
- **Live YouTube API**: no key was available. The adapter is tested against mocked HTTP responses
  (pagination, quota, disabled comments, 404, invalid key, retries); a real call has not been made.
- **GitHub Actions**: workflows written; never executed (no remote).

## Data & labels
- YouTube comments carry no ground truth. Live data feeds drift signals only; quality metrics come
  from labelled test sets, the drift test set and the regression suite.
- The drift dataset and demo comments are **synthetic, template-generated** text written for this
  project. They demonstrate the mechanism reproducibly; they are not a sample of real audiences.
  Their labels are intended labels by construction.
- Retraining uses that synthetic labelled drift set. A real deployment needs human annotation of the
  drifted window (active learning / annotation UI are future work).
- Regression-suite labels are the author's intended labels with rationale; some cases are
  deliberately ambiguous and accept multiple labels.
- tweet_eval's official test split has a different class balance from its train split.

## Model
- English-centric encoder (`distilbert-base-uncased`); Hindi (Devanagari) and Hinglish are poorly
  modelled. Regression failures cluster in sarcasm, Hinglish, and profanity used as praise.
- v3's recall on unseen toxic slang templates is 0.37 (precision 0.79): the small drift set does not
  teach toxic slang well.
- v3's promotion came from threshold calibration, not new encoder quality (identical weights to v2).
- Toxicity is binary (any Jigsaw label); raw subtype labels are kept but not modelled.

## Monitoring
- Drift thresholds were calibrated once, on model v1 and this data; they are configuration, not
  universal constants.
- Language detection is heuristic (langdetect + Hinglish lexicon); short texts are unreliable.
- After a drift-triggered retrain the reference becomes 50% historical / 50% recent, so fully
  historical traffic then reads as a *warning*.
- Evidently's per-column tests are significance-only and flag almost everything at n ≥ 300; they are
  evidence, not a trigger.
- The monitor worker is an in-process thread (single API replica). Multiple replicas would poll the
  same video; a real deployment should use one scheduler or a queue.

## Platform-specific (Windows)
- Windows Smart App Control blocked some wheels: pinned `pyarrow==24.0.0`, `scikit-learn==1.6.1`;
  the venv must be created with `--copies`; `black.exe` is blocked (use `python -m black`).
- Loading PyTorch before pyarrow caused access violations in worker threads; `src/__init__.py`
  imports pyarrow first.

## Reproducibility notes
- Training is seeded and was observed to be bit-reproducible on the same GPU/driver; bitwise
  equality across different hardware is not guaranteed.
- v2 and v3 were trained with uncommitted edits in the working tree (git tag `-dirty`).
- No DVC remote is configured; `dvc repro` rebuilds data from the public sources.

## Future work
Kafka/queue ingestion, Kubernetes, Airflow scheduling, X / Instagram adapters (optional, not core),
multilingual encoder (e.g. XLM-R / MuRIL for Hinglish), human feedback + annotation UI, active
learning, GPU inference service, cloud object storage for artifacts, feature store.

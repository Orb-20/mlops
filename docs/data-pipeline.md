# Data Pipeline

## Datasets

| Task | Source | Classes | Label mapping |
|---|---|---|---|
| Sentiment | `cardiffnlp/tweet_eval`, config `sentiment` (SemEval-2017 Task 4A, tweets) | negative / neutral / positive | Native ids 0/1/2 — **no mapping invented** |
| Toxicity | Jigsaw Toxic Comment Classification `train.csv` (HF mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge`) | safe / toxic | `toxic = 1` if **any** of `toxic, severe_toxic, obscene, threat, insult, identity_hate` is 1. Raw columns are kept in the processed parquet. |

Why tweet_eval rather than Sentiment140/Amazon: it natively has 3 classes (Sentiment140 is binary;
Amazon star ratings would need an invented star→class rule) and it is short social-media text,
close to YouTube comments.

Known property: the SemEval-2017 **test** split has a different class balance from train
(neg 32% / neu 48% / pos 19% vs 16% / 45% / 39%). We keep the official test split, so test metrics
include this label shift. Macro-F1 is the headline metric for that reason.

## Stages (`dvc.yaml`)

```
download  →  prepare  →  baseline
                     ↘  drift_data
validate_curated  (hashes the git-tracked regression + demo sets into dvc.lock)
```

- `download`: fetches raw files to `data/raw/` (DVC output).
- `prepare`: `python -m src.data.prepare --step build`
  1. `clean_text` (HTML unescape, NFKC, control chars, URL → `http`, `@handle` → `@user`, whitespace)
  2. drop texts shorter than `data.text.min_chars`, dedupe after cleaning
  3. sentiment: official splits; test built first, texts seen in test are removed from val/train
  4. toxicity: seeded stratified 70/10/20 split of a seeded stratified subsample
  5. subsample caps from `configs/config.yaml` (hardware budget: 4 GB GPU)
  6. validation: columns, label ranges, empties, duplicates, **cross-split leakage** → hard failure
  7. writes `data/processed/{task}_{split}.parquet` + `manifest.json`

## Versioning

- `dvc.lock` records md5 of every stage's deps/outs (data, code, params).
- `manifest.json.dataset_version` = sha256 (first 12 hex) of the processed parquet files. This is
  logged to MLflow on every training run as the `dataset_version` tag, so a model is traceable to
  its exact data even inside containers that don't run DVC.
- Determinism verified: `dvc repro -f prepare` reproduces `dataset_version = 1cbb27cc862a`.
- No DVC remote is configured (₹0 infrastructure). `dvc remote add` any free storage to share data;
  otherwise `dvc repro` rebuilds everything from public sources.

## Current processed sizes

| split | sentiment | toxicity (toxic %) |
|---|---|---|
| train | 12,000 | 14,000 (10.2%) |
| val | 2,000 | 2,000 (10.2%) |
| test | 3,000 | 4,000 (10.2%) |

## Demo / curated data (git-tracked)

- `data/sample/youtube_comments.csv` — 300 **synthetic** template-generated comments
  (`scripts/seed_demo_data.py`, seed 42). Not real YouTube data; always shown as DEMO DATA.
- `data/regression/regression_cases.csv` — curated regression suite (Phase 12).

# Model

## Architecture

Multi-task DistilBERT (`distilbert-base-uncased`, `src/nlp/model.py`):

```
text → tokenizer (max_length 128) → DistilBERT encoder → mean pooling (768-d embedding)
                                                        ├── dropout → Linear(768, 3)  sentiment
                                                        └── dropout → Linear(768, 2)  toxicity
```

The pooled embedding is also the representation used for drift monitoring, so monitoring watches
exactly what the heads see.

## Training (`src/training/trainer.py`)

- Sentiment and toxicity data come from different datasets, so no row has both labels. Each step
  draws **one task's batch** and computes only that head's loss; across an epoch the objective is
  `L = L_sentiment + λ·L_toxicity` (λ = `training.toxicity_loss_weight`, 1.0).
- Batch interleaving is a seeded shuffle proportional to dataset size.
- Toxicity cross-entropy uses inverse-frequency class weights (Jigsaw is ~10% toxic).
- AdamW, lr 3e-5, weight decay 0.01, 6% linear warmup, grad clip 1.0, fp16 autocast on CUDA.
- Defaults: 2 epochs, batch 16, up to 8,000 rows per task (4 GB laptop GPU). CPU works, slowly.
- Fully seeded: re-running with the same data reproduces identical weights (observed: v2 and v3).

## Toxicity decision threshold

Models trained after 2026-09-29 store an F1-optimal threshold chosen on the **validation** split
(`calibrate_threshold`, `configs/model.yaml: inference.calibrate_threshold`). Models without one use
0.5. Every evaluation (including promotion comparisons) uses each model's own threshold.

## Results (held-out test sets: 3,000 tweets / 4,000 Jigsaw comments)

| Model | Sent. acc | Sent. macro-F1 | Tox P | Tox R | Tox F1 | ROC-AUC | PR-AUC | threshold |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR baseline | 0.554 | 0.519 | 0.744 | 0.702 | 0.722 | 0.959 | 0.815 | 0.5 |
| DistilBERT v1 | 0.681 | 0.675 | 0.777 | 0.808 | 0.792 | 0.977 | 0.886 | 0.5 |
| v2 (+drift data) — rejected | 0.682 | 0.676 | 0.813 | 0.771 | 0.791 | 0.973 | 0.882 | 0.5 |
| **v3 (+drift data) — production** | 0.682 | 0.676 | 0.772 | 0.825 | 0.798 | 0.973 | 0.882 | 0.18 |
| v4 (corrupted labels) — rejected | 0.423 | 0.421 | 0.102 | 1.000 | 0.184 | 0.112 | 0.055 | 0.12 |

Why the transformer: +0.156 sentiment macro-F1 and +0.07 toxicity F1 over the baseline, plus it
provides the dense embeddings the drift monitor needs (TF-IDF vectors would make every new slang
token look like drift).

### Held-out drift test set (unseen slang/Hinglish templates, 400 texts)

| Model | Sent. macro-F1 | Tox F1 | Tox recall |
|---|---|---|---|
| v1 | 0.389 | 0.295 | 0.439 |
| v3 | 0.577 | 0.500 | 0.366 |

v3 adapts strongly on sentiment and toxicity precision, but recall on *unseen* toxic slang templates
stays low — see `limitations.md`.

## Metrics policy

Sentiment: accuracy, macro/weighted precision-recall-F1, confusion matrix. Toxicity: precision,
recall, F1, ROC-AUC, PR-AUC, confusion matrix — accuracy is logged but never used as a gate
(95.7% accuracy is achievable by predicting "safe" for almost everything).

## Artifact

`encoder/`, `tokenizer/`, `heads.pt`, `metadata.json` (labels, max_length, run_id, dataset_version,
git_commit, toxicity_threshold), optional `reference_texts.json`; logged as an MLflow pyfunc model
and registered as `sentiment-toxicity-model`.

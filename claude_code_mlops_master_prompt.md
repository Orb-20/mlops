# CLAUDE CODE MASTER IMPLEMENTATION PROMPT
# Adaptive Social Media Sentiment & Toxicity Monitoring
# with Automated Embedding Drift Detection, Retraining, CI Gating, and Reporting

> **Purpose:** Give this entire file to Claude Code as the implementation specification.
>
> **Important:** This is an MLOps project first and an NLP application second. The UI is the demonstration layer. The core academic contribution is the complete, reproducible lifecycle from data ingestion → inference → monitoring → drift detection → retraining → validation → model promotion/rejection → deployment.

---

# 0. EXECUTION CONTRACT

You are the lead:

- ML Engineer
- MLOps Engineer
- Backend Engineer
- DevOps Engineer
- QA Engineer
- Data Engineer
- Technical Architect

You are responsible for implementing this project end-to-end.

Do not merely generate code snippets. Build the actual repository, run commands, test the implementation, fix errors, and leave the repository in a runnable state.

## Non-negotiable rules

1. **Do not build a notebook-only project.**
2. **Do not fake functionality.**
3. **Do not silently skip broken components.**
4. **Do not claim a feature works unless you have actually tested it.**
5. **Prefer a smaller working implementation over a large collection of placeholders.**
6. **Run commands whenever the environment permits.**
7. **Inspect the existing repository before changing anything.**
8. **Do not delete or overwrite useful existing work without understanding it first.**
9. **Keep secrets out of Git.**
10. **Do not require paid infrastructure for the core project.**
11. **Do not make X or Instagram API access a core dependency.**
12. **YouTube is the primary live data source.**
13. **A local deterministic demo-data path must always exist.**
14. **Drift must never directly deploy a new model.**
15. **Every candidate model must pass evaluation and regression gates before promotion.**
16. **Every important behavior must have tests.**
17. **Document implementation decisions and limitations honestly.**
18. **If a library/API has changed, inspect its installed/current documentation or package interface instead of guessing.**
19. **Do not invent API responses, metrics, screenshots, successful deployments, or monitoring results.**
20. **If something cannot be verified, explicitly mark it as unverified.**

---

# 1. PROJECT TITLE

## Adaptive Social Media Sentiment & Toxicity Monitoring with Automated Drift Detection and Safe Model Retraining

Short name:

```text
Adaptive NLP MLOps Pipeline
```

---

# 2. PROJECT OBJECTIVE

Build a production-style MLOps system that can:

1. Train an NLP sentiment + toxicity model.
2. Track experiments with MLflow.
3. Version datasets with DVC.
4. Serve predictions through FastAPI.
5. Accept a YouTube video URL.
6. Fetch available comments using the official YouTube Data API.
7. Detect the language of every comment.
8. Generate sentiment predictions.
9. Generate toxicity predictions.
10. Generate transformer embeddings.
11. Store comments, predictions, metadata, and monitoring information.
12. Monitor changes in incoming data.
13. Detect embedding drift.
14. Monitor language-distribution drift.
15. Monitor prediction-distribution drift.
16. Monitor vocabulary changes.
17. Trigger a retraining workflow when configured drift conditions are met.
18. Train a candidate model.
19. Evaluate the candidate model.
20. Run a curated regression test suite.
21. Compare the candidate against the current production model.
22. Reject unsafe or inferior candidates.
23. Register and promote acceptable candidates through MLflow.
24. Run CI checks through GitHub Actions.
25. Provide an interactive monitoring dashboard.
26. Generate a useful analysis report for an analyzed video.
27. Demonstrate the complete lifecycle locally with Docker Compose.
28. Work without paid infrastructure.
29. Remain deployable later to a free/low-cost host without redesigning the architecture.

---

# 3. WHAT THIS PROJECT IS NOT

Do NOT reduce the project to:

```text
YouTube URL
    ↓
Sentiment model
    ↓
Chart
```

That would be an ordinary NLP application.

The actual project is:

```text
New social-media data
        ↓
Inference
        ↓
Embeddings
        ↓
Monitoring
        ↓
Drift detection
        ↓
Retraining trigger
        ↓
Candidate model
        ↓
Evaluation
        ↓
Regression gate
        ↓
Production comparison
        ↓
Promote OR reject
        ↓
Monitor again
```

The URL analysis UI is only the user-facing demonstration of this lifecycle.

---

# 4. ACADEMIC/MLOPS REQUIREMENT

The project must visibly demonstrate:

| Requirement | Implementation |
|---|---|
| ML training | PyTorch + Transformers |
| Experiment tracking | MLflow |
| Dataset versioning | DVC |
| Model registry | MLflow |
| Data ingestion | YouTube API + deterministic local fallback |
| API serving | FastAPI |
| UI | Streamlit |
| Monitoring | Evidently + custom monitoring logic where required |
| Drift detection | Embedding drift + auxiliary drift signals |
| Automated retraining | Python pipeline |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Containerization | Docker |
| Local orchestration | Docker Compose |
| Reporting | Streamlit dashboard + downloadable report |
| Reproducibility | fixed seeds + versioned data + documented commands |

---

# 5. CORE MLOPS IDEA

The central research/engineering question is:

> Can a deployed NLP system detect when the language and data distribution it receives has changed, automatically trigger retraining, and safely update the production model without blindly deploying a potentially worse model?

The system must therefore distinguish between:

## Language detection

Example:

```text
English
Hindi
Hinglish
Spanish
Other
```

and:

## Language/data drift

Example:

Training/reference language:

```text
"This movie is amazing."
"I really enjoyed this."
"This product is terrible."
```

Later language:

```text
"bro this is peak"
"absolute cinema"
"mid"
"fell off"
"this cooked"
```

The language may still be English, while its embedding distribution and vocabulary have changed.

Therefore:

> **Do not equate language detection with drift detection.**

Language detection is one monitoring signal.

**Embedding drift is the primary automated drift signal.**

---

# 6. FINAL USER EXPERIENCE

The final application should have two major user flows.

## Flow A — Analyze a YouTube video

User enters:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

Then:

```text
Validate URL
    ↓
Extract video ID
    ↓
Fetch comments
    ↓
Normalize/validate data
    ↓
Detect language
    ↓
Run sentiment + toxicity model
    ↓
Generate embeddings
    ↓
Store results
    ↓
Generate analytics
    ↓
Show dashboard
    ↓
Generate report
```

## Flow B — Monitor a video

User selects:

```text
Monitor Video
```

The system periodically checks for new comments.

New comments are processed incrementally.

The monitoring system compares the new window against a reference distribution.

If drift is detected:

```text
DRIFT DETECTED
      ↓
Create retraining event
      ↓
Build candidate training dataset
      ↓
Train candidate
      ↓
Evaluate candidate
      ↓
Regression tests
      ↓
Production comparison
      ↓
PROMOTE or REJECT
```

---

# 7. FINAL UI DESIGN

Build a clean Streamlit dashboard.

Do not make it look like a raw developer tool.

Use clear sections.

## Page 1 — Analyze

```text
Adaptive NLP Monitoring

YouTube URL
[________________________________________]

[ Analyze Video ]
```

Show:

### Video overview

- Video ID
- Source
- Number of comments analyzed
- Number of languages detected
- Analysis timestamp
- Current production model version

### Sentiment

Show:

- Positive %
- Neutral %
- Negative %
- Average confidence
- Sentiment distribution chart
- Sentiment over comment/time order if timestamp data is available

### Toxicity

Show:

- Safe %
- Toxic %
- Average toxicity score
- High-risk comment count
- Toxicity distribution

Do not expose or emphasize personal information about commenters.

### Language

Show:

```text
English
Hindi
Hinglish
Other
```

with percentages.

### Vocabulary/emerging phrases

Show:

- frequently appearing phrases
- new/rare phrases
- phrases with unusually large frequency changes

Do not claim that a phrase is "toxic" or "abusive" solely because it is new.

### Model health

Show:

```text
Model: sentiment-toxicity-vN
Status: Production
Last updated: ...
```

### Drift status

Show:

```text
Embedding drift:      Stable / Warning / Drift
Language drift:       Stable / Warning / Drift
Prediction drift:     Stable / Warning / Drift
Overall status:       Stable / Drift detected
```

---

# 8. MLOPS MONITORING PAGE

Create a dedicated monitoring page.

Show:

## Current production model

```text
Model version
Run ID
Dataset version
Git commit
Training timestamp
```

## Drift history

Charts for:

- embedding drift score
- language distribution
- prediction distribution
- vocabulary change
- drift events over time

## Retraining history

Show:

```text
Event
Reason
Candidate version
Candidate metrics
Regression result
Production comparison
Decision
```

Example:

```text
Drift detected
      ↓
Candidate v5
      ↓
Regression tests: PASS
      ↓
Production comparison: PASS
      ↓
Promoted
```

or:

```text
Drift detected
      ↓
Candidate v6
      ↓
Regression tests: FAIL
      ↓
Rejected
      ↓
Production remains v5
```

---

# 9. FINAL ANALYSIS REPORT

After analyzing a video, generate a report containing:

## Executive Summary

Example structure:

```text
The analyzed content received predominantly positive/negative/mixed
audience reactions. The system analyzed N comments across M detected
languages using production model vX.
```

Do not fabricate the summary. Generate it from actual computed statistics.

## Sections

1. Video/source information
2. Analysis timestamp
3. Model version
4. Number of comments
5. Sentiment distribution
6. Toxicity distribution
7. Language distribution
8. Sentiment trend
9. Toxicity trend
10. Emerging vocabulary
11. Drift status
12. Model health
13. Retraining status, if triggered
14. Candidate model status, if applicable
15. Technical metadata

The report must clearly distinguish:

```text
Observed data
```

from:

```text
Model prediction
```

and:

```text
Drift signal
```

---

# 10. ARCHITECTURE

Implement the following logical architecture:

```text
                         USER
                           |
                           v
                 +--------------------+
                 |    STREAMLIT UI    |
                 |                    |
                 | Analyze Video      |
                 | Monitor Video      |
                 | MLOps Dashboard    |
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 |      FASTAPI       |
                 +---------+----------+
                           |
                +----------+----------+
                |                     |
                v                     v
       +----------------+    +----------------+
       | YouTube Adapter|    | Monitoring API |
       +-------+--------+    +----------------+
               |
               v
        Raw Comments
               |
               v
       Validation/Cleaning
               |
               v
       Language Detection
               |
               v
      +-------------------+
      | Shared Transformer|
      | Encoder           |
      +---------+---------+
                |
         +------+------+
         |             |
         v             v
    Sentiment       Toxicity
       Head           Head
         |             |
         +------+------+
                |
                v
            Embeddings
                |
                v
           PostgreSQL
                |
                v
        Monitoring Layer
                |
       +--------+---------+
       |                  |
       v                  v
  Drift Signals       Analytics
       |
       v
 Drift Decision Engine
       |
  +----+----+
  |         |
Stable     Drift
  |         |
  |         v
  |     Retraining
  |         |
  |         v
  |      DVC Data
  |         |
  |         v
  |    Train Candidate
  |         |
  |         v
  |       MLflow
  |         |
  |         v
  |    Model Evaluation
  |         |
  |         v
  |   Regression Tests
  |         |
  |    +----+----+
  |    |         |
 FAIL        PASS
  |             |
Reject       Compare
  |             |
Old Model   Production
                |
         +------+------+
         |             |
      Reject         Promote
                        |
                        v
                 Production Model
                        |
                        v
                     FastAPI
```

---

# 11. TECHNOLOGY STACK

Use this stack unless a concrete technical incompatibility requires a documented change.

## ML

- Python 3.11
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT
- scikit-learn
- pandas
- NumPy

## NLP

- DistilBERT tokenizer/model
- Lightweight language detection
- Shared transformer encoder
- Sentiment head
- Toxicity head

## MLOps

- MLflow
- DVC
- Evidently
- GitHub Actions

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Database

- PostgreSQL

## Frontend

- Streamlit

## Infrastructure

- Docker
- Docker Compose

## Testing

- pytest
- pytest-asyncio where appropriate
- HTTPX

## Code quality

- Ruff
- Black
- isort if useful
- mypy where practical

Do not introduce Kubernetes, Kafka, Airflow, Spark, Redis, or another orchestration system merely for visual complexity.

Those can be discussed as future production extensions.

---

# 12. REPOSITORY-FIRST WORKFLOW

Before writing code:

1. Inspect the repository.
2. List files and directories.
3. Identify existing code.
4. Identify existing virtual environments.
5. Inspect `README`, configuration, Git status, Docker files, and dependency files.
6. Determine whether this is a new project or an existing partial implementation.
7. Preserve useful existing work.
8. Produce a short implementation plan.
9. Then begin implementation.

Do not start by blindly generating dozens of files.

---

# 13. TARGET REPOSITORY STRUCTURE

Use a clean structure similar to:

```text
adaptive-nlp-mlops/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── Makefile
│
├── configs/
│   ├── config.yaml
│   ├── model.yaml
│   ├── drift.yaml
│   └── gates.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── reference/
│   ├── regression/
│   ├── drift/
│   └── sample/
│
├── dvc.yaml
├── dvc.lock
│
├── notebooks/
│   └── exploration/
│
├── scripts/
│   ├── setup_local.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── simulate_drift.py
│   ├── run_retraining.py
│   └── seed_demo_data.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── data/
│   │   ├── schemas.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   └── versioning.py
│   │
│   ├── ingestion/
│   │   ├── base.py
│   │   └── youtube.py
│   │
│   ├── nlp/
│   │   ├── tokenizer.py
│   │   ├── language.py
│   │   ├── model.py
│   │   ├── inference.py
│   │   └── embeddings.py
│   │
│   ├── training/
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   ├── metrics.py
│   │   └── train.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── vocabulary.py
│   │   ├── prediction_drift.py
│   │   └── decision.py
│   │
│   ├── pipeline/
│   │   ├── analyze.py
│   │   ├── monitor.py
│   │   └── retrain.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── regression.py
│   │   └── promotion.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │
│   ├── database/
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   └── reporting/
│       ├── analytics.py
│       └── report.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   ├── regression/
│   └── fixtures/
│
├── mlruns/
│   └── .gitkeep
│
├── reports/
│   ├── drift/
│   ├── evaluation/
│   └── generated/
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── model-validation.yml
```

Adjust this if the existing repository has a better structure, but preserve the architectural separation.

---

# 14. DATASET STRATEGY

Use separate labeled datasets for the two tasks.

## Sentiment

Choose one practical public dataset as the primary training dataset:

- Sentiment140
- Amazon Reviews subset

Prefer a dataset that supports:

```text
positive
neutral
negative
```

If the selected dataset does not naturally provide three classes, document the label mapping explicitly.

## Toxicity

Use the Jigsaw Toxic Comment Classification dataset.

Raw labels include:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

For the first production implementation:

```text
toxicity:
safe / toxic
```

Keep raw multi-label columns available for future extension.

Do not silently invent label mappings.

---

# 15. MULTI-TASK MODEL DESIGN

Use a shared DistilBERT encoder with separate task heads.

Conceptually:

```text
                    Text
                     |
                 Tokenizer
                     |
                 DistilBERT
                     |
               Shared embedding
                 /         \
                /           \
       Sentiment head     Toxicity head
            |                  |
        3 classes           2 classes
```

Because sentiment and toxicity datasets may come from different sources, do not incorrectly pretend every row has both labels.

Implement one of these properly:

### Preferred

A multi-task training loop with task-specific batches and masked losses:

```text
sentiment batch → sentiment loss
toxicity batch  → toxicity loss

combined loss:
L = sentiment_loss + lambda * toxicity_loss
```

or a similarly defensible approach.

If this becomes unstable or too expensive for the available hardware, implement separate task models while preserving a common embedding interface and document the architectural trade-off.

Do not sacrifice a working, reproducible system merely to force a multi-task architecture.

---

# 16. TRAINING PIPELINE

Provide a reproducible command:

```bash
python -m src.training.train
```

The pipeline must:

```text
Load versioned data
      ↓
Validate
      ↓
Preprocess
      ↓
Split
      ↓
Tokenize
      ↓
Train
      ↓
Validate
      ↓
Evaluate
      ↓
Save artifact
      ↓
Log MLflow
      ↓
Register candidate
```

Use fixed seeds.

Track:

- dataset version
- Git commit
- model name
- tokenizer version
- hyperparameters
- batch size
- learning rate
- epochs
- random seed
- training duration
- validation metrics
- test metrics
- environment/dependency information where practical

---

# 17. BASELINE

Implement a simple baseline before the transformer pipeline.

Recommended:

```text
TF-IDF + Logistic Regression
```

for sentiment and toxicity where appropriate.

Record baseline metrics.

The academic report should show why the transformer pipeline is useful without reducing the project to a model-comparison exercise.

---

# 18. MODEL METRICS

## Sentiment

Track:

- accuracy
- precision
- recall
- macro F1
- weighted F1
- confusion matrix

## Toxicity

Track:

- precision
- recall
- F1
- ROC-AUC where appropriate
- PR-AUC where appropriate
- confusion matrix

Because toxicity is imbalanced:

> Never use accuracy as the only toxicity metric.

---

# 19. MLflow REQUIREMENTS

Run MLflow locally through Docker Compose.

Track:

```text
experiment
run ID
parameters
metrics
artifacts
model
dataset version
Git commit
```

Create experiments:

```text
sentiment-toxicity-training
drift-retraining
```

Use a model registry entry:

```text
sentiment-toxicity-model
```

Maintain explicit lifecycle states/aliases such as:

```text
candidate
staging
production
rejected
```

Do not use filenames as model versioning.

MLflow should be the source of truth for model lineage and promotion.

---

# 20. YOUTUBE INGESTION

YouTube is the primary live source.

Implement:

```text
SocialMediaAdapter
```

with an interface similar to:

```python
class SocialMediaAdapter(ABC):
    def validate_url(self, url: str) -> bool:
        ...

    def extract_content_id(self, url: str) -> str:
        ...

    def fetch_comments(self, content_id: str, ...) -> list:
        ...
```

Implement:

```text
YouTubeAdapter
```

Input:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

Use the official YouTube Data API.

Environment variable:

```text
YOUTUBE_API_KEY
```

Never commit the key.

Support:

- URL validation
- video ID extraction
- pagination
- configurable maximum comments
- retry/backoff for transient failures
- API quota errors
- empty comments
- disabled comments
- malformed responses

Store:

- comment ID
- parent ID where available
- video ID
- text
- timestamp
- source
- ingestion timestamp

Avoid unnecessary personal information.

---

# 21. FREE-TIER / OFFLINE FALLBACK

The project must remain demonstrable without external API access.

Create:

```text
data/sample/youtube_comments.csv
```

and a deterministic demo path.

The UI must clearly indicate:

```text
LIVE YOUTUBE DATA
```

or:

```text
DEMO DATA
```

Never present demo data as live data.

If the YouTube API key is missing, the application should give a useful message and allow the demo dataset path rather than crashing the entire application.

---

# 22. ON-DEMAND ANALYSIS PIPELINE

Implement:

```bash
python -m src.pipeline.analyze --url "<YOUTUBE_URL>"
```

or an equivalent CLI.

Pipeline:

```text
URL
 ↓
Validate
 ↓
Extract video ID
 ↓
Fetch comments
 ↓
Normalize schema
 ↓
Validate records
 ↓
Language detection
 ↓
Sentiment inference
 ↓
Toxicity inference
 ↓
Embedding generation
 ↓
Persist results
 ↓
Generate analytics
 ↓
Generate report
```

The API and Streamlit UI should call the same service-layer functions rather than duplicating business logic.

---

# 23. CONTINUOUS MONITORING

Implement two modes.

## Mode A — On-demand

```text
URL
 ↓
Fetch
 ↓
Analyze
 ↓
Report
```

## Mode B — Monitoring

```text
Monitor video
 ↓
Periodic polling
 ↓
Identify new comments
 ↓
Process only new data where possible
 ↓
Update monitoring window
 ↓
Calculate drift
```

Do not build Kafka for the first version.

Use a simple background/scheduled worker.

Store enough state to avoid repeatedly processing the same comments.

For a class project, a polling interval should be configurable.

Example:

```yaml
monitoring:
  poll_interval_seconds: 300
```

Do not run aggressive polling by default.

---

# 24. LANGUAGE DETECTION

Every incoming text should pass through language detection.

Store:

```text
language
language_confidence
```

Generate aggregate statistics.

Example:

```text
English   64%
Hindi     18%
Hinglish  11%
Other      7%
```

Important:

> Language detection is not drift detection.

Use language distribution as an auxiliary drift signal.

---

# 25. EMBEDDING GENERATION

Use the same transformer representation used by the model where practical.

For each text:

```text
text
 ↓
DistilBERT
 ↓
embedding
```

The system must maintain:

```text
reference embeddings
```

and:

```text
current-window embeddings
```

Do not store massive raw embedding vectors in every UI response.

Store them efficiently in the database or artifact layer.

---

# 26. DATABASE DESIGN

Use PostgreSQL.

At minimum, design tables/entities conceptually similar to:

## comments

```text
id
source
source_content_id
source_comment_id
parent_comment_id
text
timestamp
ingested_at
language
language_confidence
```

## predictions

```text
comment_id
model_version
sentiment
sentiment_confidence
toxicity_score
toxicity_label
created_at
```

## embeddings

```text
comment_id
model_version
embedding_reference
created_at
```

Do not force PostgreSQL to become a vector database unless necessary.

For the first version, efficient serialized/array storage or artifact-based embedding storage is acceptable if clearly documented.

## drift_events

```text
id
source
content_id
window_start
window_end
embedding_score
language_score
prediction_score
overall_status
reason
created_at
```

## model_events

```text
model_version
event_type
candidate_metrics
production_metrics
decision
reason
created_at
```

---

# 27. DRIFT MONITORING

Use Evidently where its installed/current API supports the required monitoring.

Do not hallucinate Evidently APIs.

If a particular version does not directly support the desired embedding metric:

1. Inspect the installed version.
2. Use the supported Evidently monitoring/reporting functionality.
3. Implement the mathematically appropriate embedding drift calculation separately if required.
4. Store the result in the same monitoring pipeline.
5. Document why the implementation is split.

Never create a fake `evidently.detect_drift()` function.

---

# 28. PRIMARY DRIFT SIGNAL: EMBEDDING DRIFT

Compare:

```text
REFERENCE EMBEDDINGS
        vs
CURRENT EMBEDDINGS
```

Use a statistically defensible distance.

Preferred options, depending on available implementation:

- MMD
- distributional statistical tests
- appropriate multivariate distance
- Evidently-supported embedding monitoring

Do not choose a threshold arbitrarily and claim it is scientifically universal.

Make the threshold configurable.

Document how the threshold was selected for the academic demonstration.

---

# 29. AUXILIARY DRIFT SIGNALS

Also monitor:

## A. Language distribution

Compare:

```text
reference language proportions
```

with:

```text
current language proportions
```

## B. Vocabulary drift

Track:

- new terms
- rare terms
- frequency shifts
- n-gram changes

## C. Prediction drift

Compare:

```text
positive / neutral / negative
```

and:

```text
safe / toxic
```

distribution between reference and current windows.

These are supporting signals.

---

# 30. DRIFT DECISION ENGINE

Create a central component:

```text
DriftDecisionEngine
```

Example configuration:

```yaml
drift:
  minimum_samples: 100

  embedding:
    enabled: true
    threshold: 0.10
    weight: 0.70

  language:
    enabled: true
    threshold: 0.10
    weight: 0.15

  prediction:
    enabled: true
    threshold: 0.10
    weight: 0.15
```

The exact values are configuration examples, not scientifically fixed values.

The decision engine should produce a structured result such as:

```json
{
  "drift_detected": true,
  "severity": "high",
  "reasons": [
    "embedding_drift",
    "language_distribution_shift"
  ],
  "scores": {
    "embedding": 0.18,
    "language": 0.14,
    "prediction": 0.07
  },
  "thresholds": {
    "embedding": 0.10,
    "language": 0.10,
    "prediction": 0.10
  },
  "sample_count": 482
}
```

If fewer than `minimum_samples` are available:

```text
Do not trigger retraining.
```

---

# 31. CRITICAL SAFETY RULE

Never implement:

```text
drift detected
    ↓
deploy
```

Correct:

```text
drift detected
    ↓
retraining
    ↓
candidate
    ↓
evaluation
    ↓
regression
    ↓
production comparison
    ↓
promotion decision
```

---

# 32. RETRAINING PIPELINE

Provide:

```bash
python -m src.pipeline.retrain
```

Pipeline:

```text
Drift Event
     ↓
Collect recent data
     ↓
Validate
     ↓
Deduplicate
     ↓
Quality checks
     ↓
Version with DVC
     ↓
Prepare training data
     ↓
Train candidate
     ↓
Evaluate
     ↓
MLflow logging
     ↓
Regression tests
     ↓
Compare with production
     ↓
Promotion decision
```

---

# 33. REAL-TIME DATA LABELING PROBLEM

Do not ignore this.

Fresh YouTube comments normally do not have ground-truth sentiment/toxicity labels.

Therefore:

```text
Unlabeled live data
```

can support:

- embedding drift
- language drift
- vocabulary drift
- prediction drift

but cannot directly provide true:

```text
F1
precision
recall
```

without labels.

The implementation must distinguish:

## Unlabeled monitoring

```text
continuous
```

from:

## Ground-truth evaluation

```text
periodic/manual/controlled
```

Do not falsely calculate live F1 by treating model predictions as ground truth.

---

# 34. RETRAINING DATA POLICY

Never blindly retrain on every prediction.

Recent data must be:

- validated
- deduplicated
- quality checked

For fully automatic retraining:

### Preferred academic demonstration

Use a controlled drift dataset with known labels.

This provides reproducibility.

For real YouTube data:

- store it as unlabeled data
- use it for drift monitoring
- do not claim ground-truth performance
- optionally create a manually labeled evaluation sample

If pseudo-labeling is used, document it explicitly and never describe pseudo-labels as ground truth.

---

# 35. DRIFT SIMULATION — MANDATORY

The project must be able to demonstrate drift without depending on a live YouTube API.

Create:

```text
data/drift/
```

and:

```text
scripts/simulate_drift.py
```

The script must:

1. Load the reference dataset.
2. Construct a deterministic shifted dataset.
3. Include changed vocabulary/slang/code-mixed patterns.
4. Generate embeddings.
5. Run the drift detector.
6. Produce a drift report.
7. Trigger the retraining pipeline.
8. Record the event.

Example:

```text
Reference:
"This movie is very good."

Shifted:
"absolute cinema"

Reference:
"I disliked this."

Shifted:
"nah this ain't it"
```

Do not merely change labels.

The shifted data must actually represent a meaningful distribution change.

The simulation must be deterministic using a fixed seed.

---

# 36. REGRESSION TEST DATASET

Create at least 30–50 carefully reviewed examples.

Categories:

```text
sarcasm
negation
slang
abbreviation
mixed language
profanity
neutral profanity
positive toxicity edge cases
negative sentiment without toxicity
toxicity without negative sentiment
ambiguous statements
```

Examples:

```text
"This is not bad at all."
"Great job, genius."
"bro this is peak"
"what a clown"
"I hate this movie"
"you are an idiot"
```

Do not assume expected labels are universally objective.

Document the intended label and rationale.

Store:

```text
text
expected_sentiment
expected_toxicity
category
rationale
```

Version this dataset with DVC.

---

# 37. MODEL VALIDATION GATES

The candidate model must satisfy all configured gates.

## Gate 1 — Minimum metrics

Example:

```yaml
gates:
  sentiment_macro_f1_min: 0.80
  toxicity_f1_min: 0.75
```

These are configurable demonstration thresholds, not universal quality standards.

## Gate 2 — Regression tests

Candidate must not exceed the configured failure tolerance.

Example:

```yaml
gates:
  regression:
    max_failed_cases: 2
```

## Gate 3 — Production comparison

Candidate must not significantly degrade important production metrics.

Do not use:

```text
candidate_accuracy > production_accuracy
```

as the only condition.

Compare relevant metrics:

```text
sentiment macro F1
toxicity F1
toxicity recall
regression failure count
```

## Gate 4 — Artifact validity

Verify:

- model artifact exists
- tokenizer exists
- metadata exists
- model loads successfully
- inference works

---

# 38. PROMOTION POLICY

Implement a centralized:

```text
PromotionDecisionEngine
```

Return a structured decision:

```json
{
  "decision": "PROMOTE",
  "candidate_version": "v5",
  "production_version": "v4",
  "reasons": [
    "minimum_metrics_passed",
    "regression_tests_passed",
    "no_critical_metric_regression"
  ]
}
```

For rejection:

```json
{
  "decision": "REJECT",
  "candidate_version": "v6",
  "production_version": "v5",
  "reasons": [
    "regression_gate_failed"
  ]
}
```

The old production model must remain available after rejection.

---

# 39. GITHUB ACTIONS

Create:

```text
.github/workflows/ci.yml
.github/workflows/model-validation.yml
```

## ci.yml

Run on:

- push
- pull request

Run:

```text
dependency validation
lint
format check
unit tests
API tests
data validation tests
```

## model-validation.yml

Run model validation when appropriate.

It should demonstrate:

```text
candidate model
    ↓
load
    ↓
evaluate
    ↓
regression suite
    ↓
promotion decision
```

Do not put secrets directly in YAML.

Use GitHub Secrets for actual API credentials.

For CI, prefer deterministic demo data so tests do not fail because YouTube is unavailable.

---

# 40. DOCKER COMPOSE

Provide a working local stack.

At minimum:

```text
postgres
mlflow
api
dashboard
```

Add a worker only if necessary.

The stack must start with a simple command:

```bash
docker compose up --build
```

Document ports clearly.

Do not hardcode host-specific assumptions.

Use health checks.

Services should wait for dependencies where required.

---

# 41. API

Implement FastAPI endpoints similar to:

```text
GET  /health
GET  /api/v1/model
POST /api/v1/analyze/youtube
POST /api/v1/monitor/start
POST /api/v1/monitor/stop
GET  /api/v1/monitor/status
GET  /api/v1/drift/latest
GET  /api/v1/drift/history
GET  /api/v1/retraining/history
GET  /api/v1/report/{id}
```

Do not expose internal secrets.

Return structured Pydantic responses.

---

# 42. API ANALYSIS RESPONSE

A response should contain fields similar to:

```json
{
  "source": "youtube",
  "content_id": "VIDEO_ID",
  "comment_count": 1200,
  "model_version": "v4",
  "sentiment": {
    "positive": 0.61,
    "neutral": 0.24,
    "negative": 0.15
  },
  "toxicity": {
    "safe": 0.87,
    "toxic": 0.13
  },
  "languages": {
    "en": 0.64,
    "hi": 0.18,
    "hinglish": 0.11,
    "other": 0.07
  },
  "drift": {
    "status": "stable"
  }
}
```

Use actual computed values.

---

# 43. REPORT GENERATION

Create a report service that can generate a downloadable report.

Prefer a simple robust format first.

If PDF generation is implemented, ensure:

- tables render correctly
- charts render correctly
- no fake data
- model version is shown
- timestamps are shown
- drift status is shown
- report source is shown

If PDF generation becomes unstable, provide a working HTML/Markdown report first and document the limitation rather than shipping a broken PDF feature.

---

# 44. TESTING STRATEGY

Testing is part of the project, not an afterthought.

## Unit tests

Test:

- URL parsing
- YouTube ID extraction
- preprocessing
- schema validation
- language detection wrapper
- model output validation
- drift calculation
- decision engine
- promotion engine
- report analytics

## Integration tests

Test:

```text
API
 ↓
service
 ↓
database
```

where practical.

## Regression tests

Test the curated 30–50 phrase dataset.

## End-to-end test

The repository should provide a deterministic path:

```text
demo data
 ↓
inference
 ↓
embeddings
 ↓
drift simulation
 ↓
retraining
 ↓
evaluation
 ↓
regression
 ↓
promotion/rejection
```

This should work without YouTube API access.

---

# 45. CONFIGURATION

Use configuration files and environment variables.

Create:

```text
.env.example
```

Include placeholders such as:

```text
YOUTUBE_API_KEY=
DATABASE_URL=
MLFLOW_TRACKING_URI=
ENVIRONMENT=local
```

Never commit `.env`.

Configuration files:

```text
configs/config.yaml
configs/model.yaml
configs/drift.yaml
configs/gates.yaml
```

Do not scatter constants through Python files.

---

# 46. LOGGING

Use structured, useful logs.

Example:

```text
[INFO] Fetching YouTube comments
[INFO] Retrieved 842 comments
[INFO] Running language detection
[INFO] Running inference with model v4
[INFO] Generated embeddings
[INFO] Running drift monitoring
[WARNING] Embedding drift threshold exceeded
[INFO] Retraining triggered
[INFO] Candidate model v5 trained
[INFO] Regression tests passed
[INFO] Candidate promoted
```

Errors should include actionable information.

Do not log API keys or sensitive text unnecessarily.

---

# 47. ERROR HANDLING

Handle gracefully:

- missing API key
- invalid YouTube URL
- private/unavailable video
- disabled comments
- API quota errors
- network failure
- empty comment result
- malformed comments
- model artifact missing
- MLflow unavailable
- PostgreSQL unavailable
- insufficient drift samples
- failed training
- failed regression tests

A temporary MLflow failure must not cause the system to silently claim that the model was registered.

---

# 48. OBSERVABILITY

The project should expose enough metadata to answer:

> What model produced this prediction?

For each prediction, retain or be able to trace:

```text
prediction
 ↓
model version
 ↓
MLflow run
 ↓
dataset version
 ↓
Git commit
```

This is essential for MLOps reproducibility.

---

# 49. SECURITY AND PRIVACY

Do not store unnecessary personal data from commenters.

Prefer:

```text
comment ID
```

over:

```text
author identity
```

unless the author field is genuinely required.

Do not expose API keys.

Do not put secrets in:

- Git
- Docker image
- logs
- Streamlit UI
- reports

Add appropriate `.gitignore` entries.

---

# 50. PERFORMANCE

The project should be usable on a normal development machine.

Do not assume a large GPU.

Use:

- small batch sizes
- configurable max comments
- configurable sequence length
- CPU fallback
- optional GPU acceleration
- caching where appropriate

Do not load the transformer model repeatedly for every individual comment.

Load the model once per service/process.

Batch inference where possible.

---

# 51. FREE-INFRASTRUCTURE REQUIREMENT

The complete core project must run locally for ₹0 infrastructure cost.

Required local services:

```text
Docker
PostgreSQL
MLflow
FastAPI
Streamlit
```

External dependency:

```text
YouTube API
```

with a valid user-provided API key.

For classroom/demo reliability:

```text
Demo dataset
```

must always be available.

Do not make the project dependent on paid cloud APIs.

---

# 52. OPTIONAL DEPLOYMENT

After local implementation is stable, make deployment possible to a free/low-cost environment.

Potential targets can include:

- Hugging Face Spaces
- Render/free-tier where currently available
- another currently available free/low-cost host

Do not assume a free tier without verifying it at deployment time.

The local architecture must remain the source of truth.

---

# 53. WHAT NOT TO BUILD

Do not waste implementation time on:

- Kafka
- Kubernetes
- Airflow
- Spark
- complex microservices
- Redis
- paid vector databases
- expensive cloud GPUs
- elaborate authentication
- social-media scraping that violates platform policies
- fake real-time streaming
- fake drift metrics
- fake MLflow runs
- fake GitHub Actions results

The project should be technically deep because of its lifecycle, not because it contains many technologies.

---

# 54. IMPLEMENTATION PHASES

Follow this exact implementation order.

## PHASE 0 — Repository inspection

Tasks:

1. Inspect repository.
2. Inspect existing code.
3. Check Python version.
4. Check Docker availability.
5. Check Git.
6. Check available GPU if relevant.
7. Identify existing dependencies.
8. Identify existing environment variables.
9. Create implementation plan.

Deliverable:

```text
docs/implementation-plan.md
```

---

## PHASE 1 — Project skeleton

Create:

- package structure
- pyproject.toml
- configuration
- `.env.example`
- `.gitignore`
- README skeleton
- Docker skeleton
- test skeleton

Acceptance:

```bash
python -m pytest
```

runs successfully, even if only basic tests exist.

---

## PHASE 2 — Data pipeline

Implement:

- dataset acquisition/documentation
- normalization
- preprocessing
- schema validation
- train/validation/test split
- DVC tracking

Acceptance:

```bash
python -m src.data.prepare
```

or equivalent produces versioned processed data.

---

## PHASE 3 — Baseline

Implement:

```text
TF-IDF + Logistic Regression
```

Track baseline metrics.

Log them to MLflow.

Acceptance:

- metrics produced
- MLflow run exists
- artifacts exist

---

## PHASE 4 — Transformer model

Implement:

- tokenizer
- DistilBERT
- sentiment head
- toxicity head
- embedding extraction
- training
- evaluation

Acceptance:

```bash
python -m src.training.train
```

completes successfully on a manageable subset/full dataset as hardware permits.

---

## PHASE 5 — MLflow registry

Implement:

- experiment tracking
- artifact logging
- model registration
- candidate/production management
- lineage metadata

Acceptance:

A trained model can be traced to:

```text
model
 ↓
run
 ↓
dataset version
 ↓
Git commit
```

---

## PHASE 6 — YouTube ingestion

Implement:

- adapter
- URL validation
- video ID extraction
- API client
- pagination
- error handling
- local fallback

Acceptance:

Live mode works with a valid API key.

Demo mode works without one.

---

## PHASE 7 — FastAPI

Implement:

- health endpoint
- analysis endpoint
- model endpoint
- monitoring endpoints
- drift endpoint
- retraining history endpoint

Acceptance:

```bash
curl http://localhost:<port>/health
```

returns healthy status.

---

## PHASE 8 — PostgreSQL

Implement:

- schema
- connection
- repositories
- prediction storage
- drift event storage
- model event storage

Acceptance:

An analyzed dataset can be persisted and queried.

---

## PHASE 9 — Drift monitoring

Implement:

- reference embeddings
- current embeddings
- embedding drift
- language drift
- vocabulary drift
- prediction drift
- drift decision engine

Acceptance:

A controlled shifted dataset produces a detectable drift event.

---

## PHASE 10 — Drift simulation

Implement:

```bash
python scripts/simulate_drift.py
```

Acceptance:

The script deterministically demonstrates:

```text
stable dataset
    ↓
shifted dataset
    ↓
drift detected
```

---

## PHASE 11 — Automated retraining

Implement:

```bash
python -m src.pipeline.retrain
```

Acceptance:

A drift event can produce:

```text
candidate model
 ↓
MLflow run
 ↓
evaluation
 ↓
regression tests
 ↓
promotion/rejection
```

---

## PHASE 12 — Regression gate

Implement the 30–50 case test suite.

Acceptance:

You can intentionally create:

### Passing candidate

```text
PASS
→ promote
```

### Failing candidate

```text
FAIL
→ reject
→ keep production model
```

Both paths must be demonstrable.

---

## PHASE 13 — GitHub Actions

Implement:

- CI
- model validation workflow

Acceptance:

A GitHub push/PR triggers visible automated checks.

---

## PHASE 14 — Streamlit dashboard

Implement:

### Page 1

```text
Analyze Video
```

### Page 2

```text
MLOps Monitoring
```

### Page 3

```text
Model Registry / Retraining History
```

Acceptance:

A faculty evaluator can understand the project without opening source code.

---

## PHASE 15 — Reporting

Implement downloadable analysis report.

Acceptance:

A real analyzed dataset produces a real report.

No fake values.

---

## PHASE 16 — Docker Compose

Containerize:

- PostgreSQL
- MLflow
- FastAPI
- Streamlit

Acceptance:

```bash
docker compose up --build
```

starts the complete local system.

---

## PHASE 17 — End-to-end demo

Run:

```text
Demo dataset
 ↓
Analysis
 ↓
Embedding generation
 ↓
Drift simulation
 ↓
Retraining
 ↓
Candidate evaluation
 ↓
Regression
 ↓
Promotion
 ↓
Dashboard update
```

Then run a failing candidate scenario.

Acceptance:

Both promotion and rejection paths work.

---

# 55. CLAUDE CODE BEHAVIOR DURING IMPLEMENTATION

At the end of every major phase:

1. Run relevant tests.
2. Inspect logs.
3. Fix failures.
4. Update documentation.
5. Summarize what changed.
6. State exact commands used.
7. State what remains.

Do not move to the next major phase while the current phase is fundamentally broken.

However, do not get stuck indefinitely on an external dependency.

If an external API is unavailable:

```text
document limitation
use deterministic fallback
continue implementation
```

---

# 56. COMMAND EXECUTION POLICY

Prefer actual execution.

Examples:

```bash
python --version
git status
docker --version
docker compose version
python -m pytest
docker compose config
docker compose up --build
```

If a command fails:

1. Read the actual error.
2. Diagnose it.
3. Fix it.
4. Re-run the command.

Do not merely tell the user that a command "should work."

---

# 57. QUALITY GATE BEFORE DECLARING COMPLETION

Do not declare the project complete until these checks have been attempted.

## Repository

```text
[ ] clean structure
[ ] README
[ ] .env.example
[ ] .gitignore
[ ] configuration
```

## Data

```text
[ ] datasets documented
[ ] preprocessing works
[ ] DVC works
[ ] sample/demo data works
```

## ML

```text
[ ] baseline works
[ ] transformer works
[ ] sentiment metrics
[ ] toxicity metrics
[ ] embeddings
```

## MLOps

```text
[ ] MLflow tracking
[ ] MLflow registry
[ ] DVC
[ ] drift detection
[ ] retraining
[ ] regression gate
[ ] promotion/rejection
```

## Backend

```text
[ ] FastAPI
[ ] health check
[ ] analysis endpoint
[ ] monitoring endpoints
```

## Database

```text
[ ] PostgreSQL
[ ] migrations/schema
[ ] persistence
```

## UI

```text
[ ] YouTube URL input
[ ] analysis dashboard
[ ] MLOps dashboard
[ ] drift visualization
[ ] model history
[ ] report
```

## DevOps

```text
[ ] Docker
[ ] Docker Compose
[ ] GitHub Actions
```

## Tests

```text
[ ] unit tests
[ ] API tests
[ ] integration tests
[ ] regression tests
[ ] end-to-end demo
```

---

# 58. REQUIRED DOCUMENTATION

Create:

```text
README.md
docs/
    architecture.md
    data-pipeline.md
    model.md
    mlops-lifecycle.md
    drift-detection.md
    retraining.md
    deployment.md
    testing.md
    demo-script.md
    limitations.md
```

The README must contain:

1. Project overview
2. Architecture diagram
3. Technology stack
4. Setup
5. Environment variables
6. Dataset preparation
7. Training
8. MLflow
9. DVC
10. YouTube setup
11. API
12. Dashboard
13. Drift simulation
14. Retraining
15. CI/CD
16. Testing
17. Docker
18. Demo instructions
19. Limitations
20. Future work

---

# 59. DEMO SCRIPT

Create:

```text
docs/demo-script.md
```

The demo should fit a short academic presentation.

Suggested flow:

## Demo 1 — Live analysis

```text
Open dashboard
 ↓
Paste YouTube URL
 ↓
Analyze
 ↓
Show sentiment
 ↓
Show toxicity
 ↓
Show language
 ↓
Show report
```

## Demo 2 — Drift

```text
Open monitoring dashboard
 ↓
Show reference distribution
 ↓
Run drift simulation
 ↓
Show embedding drift
 ↓
Show drift event
```

## Demo 3 — Retraining

```text
Drift event
 ↓
Retraining
 ↓
MLflow candidate
 ↓
Regression tests
 ↓
Promotion
```

## Demo 4 — Safety gate

Intentionally use a failing candidate.

Show:

```text
Candidate failed
 ↓
Rejected
 ↓
Production model unchanged
```

This is one of the most important MLOps demonstrations.

---

# 60. ARCHITECTURE DIAGRAM REQUIREMENT

Create a polished architecture diagram in:

```text
docs/architecture.png
```

or an editable diagram source such as Mermaid.

The diagram must visibly show:

```text
YouTube
   ↓
Ingestion
   ↓
Preprocessing
   ↓
Transformer
   ↓
Sentiment + Toxicity
   ↓
Embeddings
   ↓
PostgreSQL
   ↓
Evidently / Drift Engine
   ↓
Retraining
   ↓
DVC
   ↓
MLflow
   ↓
CI Regression Gate
   ↓
Production
   ↓
FastAPI
   ↓
Streamlit
```

---

# 61. IMPORTANT TECHNICAL LIMITATIONS TO DOCUMENT

Do not hide these:

### YouTube API

Requires:

```text
YOUTUBE_API_KEY
```

The project should work in demo mode without it.

### Live labels

YouTube comments do not automatically provide ground-truth sentiment/toxicity labels.

Therefore live monitoring uses:

```text
drift
distribution
embedding
vocabulary
prediction
```

rather than pretending to have real-time F1.

### Drift simulation

Controlled drift is used for reproducible academic demonstration.

### Platform support

YouTube is the primary supported live source.

X and Instagram are optional future adapters, not core dependencies.

### Production scaling

A real high-throughput deployment could use a queue/streaming architecture later, but that is outside the first implementation.

---

# 62. FUTURE EXTENSIONS

Document, but do not require:

```text
Kafka
Kubernetes
Airflow
X API
Instagram API
multilingual transformer
human feedback loop
active learning
manual annotation UI
GPU inference service
cloud object storage
feature store
```

These are future extensions, not reasons to over-engineer the current project.

---

# 63. FINAL DEFINITION OF DONE

The project is complete only when a new user can follow the README and do something close to:

```bash
git clone <repo>
cd adaptive-nlp-mlops

cp .env.example .env

docker compose up --build
```

Then:

```text
Open Streamlit
     ↓
Choose Demo Data
     ↓
Analyze
     ↓
View sentiment/toxicity/language report
```

Then:

```bash
python scripts/simulate_drift.py
```

and see:

```text
DRIFT DETECTED
```

Then:

```bash
python -m src.pipeline.retrain
```

and see:

```text
Candidate trained
Candidate evaluated
Regression tests executed
Promotion decision made
```

Then the dashboard must show the event.

Finally, demonstrate both:

```text
PASS → model promoted
```

and:

```text
FAIL → model rejected → old production model retained
```

---

# 64. FINAL INSTRUCTION TO CLAUDE CODE

Do not try to implement everything blindly in one pass.

Work in phases.

For each phase:

```text
INSPECT
   ↓
PLAN
   ↓
IMPLEMENT
   ↓
RUN
   ↓
TEST
   ↓
FIX
   ↓
DOCUMENT
   ↓
VERIFY
```

Do not fabricate successful results.

Do not skip testing.

Do not silently replace real functionality with mocks.

Mocks are acceptable only in unit tests or explicitly labelled demo fallbacks.

The final system must be:

```text
REPRODUCIBLE
TESTABLE
VERSIONED
MONITORED
CONTAINERIZED
AUTOMATED
DEPLOYABLE
DEMONSTRABLE
```

The central story of the project must remain:

```text
                 NEW DATA
                    ↓
               PREDICTION
                    ↓
                EMBEDDING
                    ↓
              DRIFT MONITOR
                    ↓
               DRIFT EVENT
                    ↓
               RETRAIN MODEL
                    ↓
               EVALUATE MODEL
                    ↓
             REGRESSION GATE
                    ↓
          +---------+---------+
          |                   |
        FAIL                PASS
          |                   |
          ↓                   ↓
       REJECT             PROMOTE
          |                   |
          ↓                   ↓
   KEEP PRODUCTION       NEW PRODUCTION
          \                   /
           \                 /
            └─── MONITOR ───┘
```

**Build the system around this lifecycle. Everything else is supporting infrastructure.**

# SentimentScope

**[▶ Live demo — melkholy-sentiment-scope.hf.space](https://melkholy-sentiment-scope.hf.space)** · [Hugging Face Space](https://huggingface.co/spaces/melkholy/sentiment-scope)

An educational, end-to-end ML engineering app: a React UI talking to a FastAPI
backend that serves transformer sentiment models locally — with token-level
**Integrated Gradients explainability**, model comparison, **AI-text detection with
detector disagreement**, evaluation metrics, error analysis, batch CSV analysis,
tests, Docker, and CI.

Built as an AI/ML portfolio project: the code is deliberately over-commented,
teaching the *why* of each step (tokenization → logits → softmax, GPU batching,
IG attribution) alongside the *what*.

![Analyze tab: 3-class confidence bars plus a per-token Integrated Gradients heatmap](docs/screenshots/analyze-explain.png)

## Architecture

The frontend only ever calls **relative `/api/*` paths**, so the exact same build
runs unchanged across three serving topologies — the interesting engineering trick
that keeps dev, Docker, and the single-container Space in sync.

```
React (Vite + TypeScript + Tailwind v4 + Recharts, React 19 idioms)
        │  relative /api/* calls  (never a hardcoded host)
        ▼
 ┌──────────────────── three serving topologies ────────────────────┐
 │  dev      →  Vite dev server proxies /api → uvicorn               │
 │  compose  →  nginx serves the built SPA, proxies /api → backend   │
 │  Spaces   →  FastAPI serves the SPA itself via StaticFiles        │
 └───────────────────────────────────────────────────────────────────┘
        ▼
FastAPI ── lifespan loads the default SentimentModel (twitter-roberta) ONCE
        ├── analyze / batch / csv / explain : default model only, strict 3-class
        ├── compare()          : lazy-loads sentiment registry models, dynamic label scores
        ├── explain()          : captum LayerIntegratedGradients on roberta.embeddings
        └── ai-detect(/compare): lazy-loads AI text detectors → {human, ai} scores + disagreement
        ▼
Model registry (task-aware): sentiment (RoBERTa · DistilBERT-SST2 · FinBERT · XLM-R)
                             + AI detectors (desklib DeBERTa-v3 · fakespot · oxidane)
```

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Single text → label + class probabilities (**twitter-roberta only**, strict 3-class) |
| `POST /api/analyze/batch` | JSON list (≤500) → per-row results + aggregates (**twitter-roberta only**) |
| `POST /api/analyze/csv` | CSV upload (`text` column, ≤5MB, ≤500 non-empty rows) → same shape as batch |
| `POST /api/explain` | Integrated Gradients token attributions (**twitter-roberta only**) |
| `GET /api/models?task=` | Task-aware model registry catalog with a live `loaded` flag |
| `POST /api/compare` | Sentiment-only side-by-side comparison — per-model dynamic scores + wall-clock latency |
| `POST /api/ai-detect` | Single text → one AI detector's `{human, ai}` scores + verbatim uncertainty warning (desklib by default) |
| `POST /api/ai-detect/compare` | Same text through all detectors → per-model scores + a `disagreement` flag + the warning |
| `GET /api/health` · `GET /api/model` | Readiness (`model_loaded`, device) · default model card |

## Quickstart

### Live demo

The app is deployed to a free CPU **[Hugging Face Space](https://melkholy-sentiment-scope.hf.space)** —
no setup required. Because it is a public, shared box the deployment is intentionally
constrained:

- **Five models enabled.** `ENABLED_MODELS` allowlists the two sentiment models
  (`twitter-roberta` + `distilbert-sst2`) and all three AI detectors (`desklib` / `fakespot` /
  `oxidane`); requesting any other registry model (finbert, xlm-twitter) returns **403**
  (run locally for the full registry).
- **Rate limited.** One shared **30 requests/minute per IP** budget across all `/api/*`
  routes; bursting past it returns **429**. This mainly protects `/api/explain`, which runs
  ~50 forward passes per call.
- **Weights baked into the image**, so cold starts never re-download — see `Dockerfile.spaces`.

### Docker (one command)

```bash
docker compose up --build
# → http://localhost:8080
```

> **First run downloads ~500MB of model weights** into a persistent `hf-cache` volume.
> There is no container healthcheck, so on a cold cache the frontend returns **502s for
> ~6 minutes** while the backend downloads and loads the model. Subsequent runs reuse the
> volume and come up in seconds. `GET /api/health` flips to `{"model_loaded": true}` once
> inference is ready. (Known Phase-1 gap: no healthcheck/readiness gate — see Roadmap.)

### Local dev

```bash
# Backend — any env with torch + transformers + captum + fastapi
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173 (Vite proxies /api to :8000)
```

Try it: upload `sample-data/reviews.csv` on the **Batch** tab, then run **Compare
Sentiment** on a finance-style sentence (e.g. *"Our quarterly revenue outlook improved"*)
to watch a social model, a binary SST-2 model, and FinBERT disagree.

## Screenshots

| Batch — aggregate charts + per-row table | Compare Sentiment — domain & label mismatch |
|---|---|
| ![Batch tab: sentiment-count and mean-confidence bar charts above a results table](docs/screenshots/batch-charts.png) | ![Compare tab: the same finance sentence scored by twitter-roberta, distilbert-sst2, and finbert](docs/screenshots/compare-sentiment.png) |

The Compare screenshot is the teaching moment: on *"Our quarterly revenue outlook improved"*,
the social **twitter-roberta** says *positive* (92.8%), binary **distilbert-sst2** says
*positive* (100%, with **no neutral class** — a label-space mismatch), and finance-tuned
**finbert** says *neutral* (88.2%, reading it as routine reporting — a domain mismatch).
Same sentence, three defensible-but-different answers.

The **[How it works](docs/screenshots/how-it-works.png)** tab walks through the pipeline in
plain language for non-ML readers.

## Evaluation

`evals/run_eval.py` scores the default model on a deliberately tricky 36-example set
(`evals/data/sentiment_eval.csv`) that concentrates sarcasm, mixed sentiment, and ambiguity —
the cases sentiment models fail on. Full report: [`evals/report.md`](evals/report.md).

| Metric | Value |
|---|---|
| Accuracy | **0.694** |
| Macro F1 | **0.689** |
| Latency p50 / p95 | 6.4 ms / 27.0 ms |

Macro F1 sits below accuracy because the model handles the three classes unevenly — the
number that exposes class imbalance rather than hiding it.

**Confusion matrix** (rows = true, columns = predicted; the diagonal is correct):

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| **negative** | 9 | 1 | 4 |
| **neutral** | 2 | 6 | 3 |
| **positive** | 0 | 1 | 10 |

The off-diagonal tells the story: the model over-predicts *positive* (recall 0.91, precision
0.59) and misses negatives and neutrals — exactly the sarcasm ("Yeah, amazing, another crash")
and mixed-sentiment ("great camera though the battery drains too fast") rows in the report.

## AI text detection

A second task family, served alongside sentiment through the same task-aware registry:
paste text and see whether AI-text detectors think it was machine-written. Three local
detectors run behind two endpoints — `POST /api/ai-detect` (one detector, desklib by
default) and `POST /api/ai-detect/compare` (all three, side by side). Each returns a full
`{human, ai}` probability distribution, not just a verdict.

- **desklib** — DeBERTa-v3-large fine-tuned for AI-text detection; emits a single logit, so
  `P(ai) = sigmoid(logit)` (the registry's `single_logit_sigmoid` adapter — one architecture,
  loaded through a custom model class rather than `AutoModelForSequenceClassification`).
- **fakespot** and **oxidane** — RoBERTa-base detectors with standard 2-class softmax heads.

**Disagreement is the point, not a bug.** `/api/ai-detect/compare` sets a `disagreement`
flag whenever the detectors don't all land on the same label, and the AI Detector tab frames
that split as uncertainty rather than a tie to break. On the eval fixture the fleet split on
**60% of rows (18 of 30)** — when models trained on different data disagree about a sentence,
that split *is* the uncertainty signal, not noise to average away. Full analysis:
[`evals/ai_detection_report.md`](evals/ai_detection_report.md).

> **Honest caveat on those numbers.** Every row in that fixture — both the `human`- and
> `ai`-labeled examples — was written by the AI assistant that built this project. So it is a
> *teaching fixture for failure modes and disagreement, not a benchmark* of detector quality,
> and the per-detector accuracies in the report are illustrations of failure modes, never
> scores. See the report's "Why this is not a benchmark" section.

**Warning by construction.** Every detector response carries a verbatim uncertainty warning,
and the UI renders that exact backend string — never a paraphrase — so a probability can never
drift into reading as proof of authorship:

> AI detectors are probabilistic and can be wrong, especially on short, edited, non-native,
> highly formal, or mixed-authorship text. Do not use this as proof of authorship.

![AI Detector tab: one paragraph scored by three detectors with a verbatim uncertainty callout and a disagreement banner](docs/screenshots/ai-detector.png)

On the live Space all three detectors are enabled and baked into the image, so the AI Detector
tab works on first click (its default action scores all three detectors at once).

## Tests

```bash
cd backend && pytest              # 72 unit tests — model mocked, runs anywhere (no torch)
cd backend && pytest -m integration   # 9 real-model tests across the sentiment + detector registry (needs weights)
cd frontend && npm test -- --run  # 25 component/API tests (vitest + Testing Library)
```

CI (GitHub Actions) runs lint + unit tests + the frontend build on every push —
**without installing torch**, because the heavy imports are lazy and the unit tests inject
a fake model via FastAPI dependency overrides.

## What this project demonstrates

- **ML fundamentals in code:** raw `AutoModelForSequenceClassification` inference (no
  `pipeline()` magic) — tokenization, batching, softmax, and device placement are all explicit
  and explained. Uses Apple-Silicon MPS when available, CPU otherwise.
- **Explainability:** Layer Integrated Gradients with a padding baseline; the UI renders
  per-token attributions as a heatmap. **RoBERTa only** — other registry models are compare-only.
- **Model comparison:** one input across social, binary SST-2, finance, and multilingual
  sentiment models to expose domain/label mismatch and latency tradeoffs (`/api/compare` is
  sentiment-only; each row carries the model's *own* label keys, so a binary model never fakes
  a neutral score).
- **A second ML task, cleanly separated:** AI-text detection lives in the same task-aware
  registry but never shares comparison endpoints with sentiment — three detectors (two
  architectures: a DeBERTa-v3 single-logit sigmoid head and two RoBERTa softmax heads), two
  endpoints, a disagreement flag, and a verbatim uncertainty warning on every response.
- **Evaluation:** `evals/run_eval.py` reports accuracy, macro F1, confusion matrix, p50/p95
  latency, and the specific misclassified examples — with a machine-readable JSON summary.
- **Engineering hygiene:** validation at the boundary, a dependency-injected model for
  testability, an integration/unit test split, a CPU-only Docker build, and CI with no GPU deps.

## Engineering decisions

- **Torch-free CI via mocking.** `app/model.py` imports torch lazily and the unit suite sets
  `SKIP_MODEL_LOAD=1`, then overrides the `get_model` dependency with a `FakeModel`. CI installs
  `requirements-dev.txt` (no torch/transformers/captum) and still exercises every route, so the
  pipeline stays fast and free. Real weights are only touched by `pytest -m integration`, run locally.
- **Lazy, memoized, task-aware registry.** The default model loads once at startup; other models
  load on first `/api/compare` use behind per-model locks and are cached, so concurrent requests
  never double-load ~500MB of weights. `resolve_model_source` prefers local `models/` weights and
  falls back to the Hub, so a fresh clone / CI / Docker build works without the untracked weights.
- **Validation at the boundary.** Pydantic caps text length and batch size; CSV upload enforces
  size (413), encoding, a required `text` column, and the row cap — each failure returns a specific
  reason. Routes stay thin: validation in schemas, ML in `SentimentModel`, wiring in between.
- **Strict-3-class contract.** `analyze`/`batch`/`csv`/`explain` reject a non-default `model_id`
  (400) so a 2-class model's output can never silently break the 3-class response schema; polyglot
  scoring is confined to `/api/compare`.
- **Public-deploy guardrails.** A single global rate limiter (not per-route decorators, which would
  force the slowapi import into CI) keyed on the trusted rightmost `X-Forwarded-For` hop, plus an
  `ENABLED_MODELS` allowlist — both off by default in dev.

## Honest limitations

- The default model was trained on tweets: long/formal text is out-of-domain.
- Explainability (Integrated Gradients) is implemented for the default Twitter RoBERTa model only;
  DistilBERT/FinBERT/XLM-R are available via `/api/compare` only.
- Sentiment models are not creativity judges; they estimate polarity, confidence, and disagreement.
- 512-token truncation; sarcasm, mixed sentiment, and missing context remain hard (see the eval).
- IG uses 50 integration steps — a principled approximation, not ground truth.
- AI-text detection is probabilistic, never proof of authorship: short, edited, non-native,
  highly formal, or mixed-authorship text fools detectors, and on the eval fixture they
  disagreed on 60% of rows. Every detector response says so verbatim, and the fixture is a
  single-author teaching set, not a benchmark (see its report).
- The public Space is CPU-only and rate-limited (30 req/min); it serves five models (two
  sentiment + three detectors). Clone and run locally for the full registry and unthrottled use.

## Roadmap

- **Docker healthcheck / readiness gate** so the compose frontend waits for model load instead of
  returning cold-start 502s.
- **A larger, provenance-verified detector evaluation.** The current AI-detection eval is a small,
  single-author teaching fixture (see its report); a genuine benchmark needs large, diverse,
  genuinely human-authored data.

> **Shipped:** Phase 2 — AI text detection (desklib / fakespot / oxidane detectors, the two
> `/api/ai-detect*` endpoints with disagreement reporting and a verbatim uncertainty warning, and
> the AI Detector tab) is live, including on the public Space.

## Repository layout

```
backend/     FastAPI app (routes, schemas, model, task-aware registry) + unit/integration tests
frontend/    React 19 + Vite + Tailwind v4 SPA (Analyze / Batch / Compare / AI Detector / How it works)
evals/       Evaluation harnesses + committed reports (sentiment + AI-detector disagreement)
sample-data/ reviews.csv for the Batch tab demo
docs/        Screenshots
docker-compose.yml · Dockerfile.spaces   Compose (nginx + backend) and single-image Space builds
```

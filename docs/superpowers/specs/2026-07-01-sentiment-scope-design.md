# SentimentScope — Design Spec

**Date:** 2026-07-01
**Status:** Approved (feature scope, rigor level, and captum IG explainability confirmed by owner)

## Purpose

A portfolio-grade, educational sentiment analysis web app demonstrating end-to-end
AI/ML engineering: local transformer inference, a clean API layer, an interactive
React UI, model explainability, and production hygiene (tests, Docker, CI).
Target audience for the code itself: hiring managers and interviewers reviewing
an AI/ML portfolio. Every module carries explanatory comments that teach the ML
concepts involved — the code doubles as a teaching artifact.

## Core Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Already cached locally; 3-class (negative/neutral/positive); well-documented |
| Inference | Raw `AutoTokenizer` + `AutoModelForSequenceClassification` + softmax, **not** `pipeline()` | Shows real understanding of the tokenize → logits → probabilities flow; needed anyway for IG |
| Explainability | Integrated Gradients via `captum` (`LayerIntegratedGradients` on the RoBERTa embedding layer) | Owner's choice; strong resume signal; principled attribution method |
| Device | MPS (Apple Silicon) locally, CPU in Docker/CI | Auto-detected at startup |
| Dev environment | Existing `ai` conda env (torch 2.11, transformers 5.5.3) + `pip install captum` | No new env; Docker image is self-contained |
| Frontend stack | React 18 + Vite + TypeScript + Tailwind + Recharts | Modern, minimal, chart lib for confidence/aggregate visuals |
| Location | `~/Projects/active/sentiment-scope` (monorepo: `backend/` + `frontend/`) | GitHub-ready |

## Architecture

```
React (Vite+TS)  ──HTTP/JSON──▶  FastAPI  ──▶  SentimentModel (singleton, loaded in lifespan)
                                                ├── predict(texts: list[str]) — batched
                                                └── explain(text: str) — Layer Integrated Gradients
```

### Backend (`backend/app/`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app factory, lifespan (load model once), CORS (frontend origin only) |
| `model.py` | `SentimentModel` class: load tokenizer/model, `predict()` (batched tokenize → logits → softmax), `explain()` (captum `LayerIntegratedGradients` on embeddings, attribution per token, summed over embedding dim, L2-normalized) |
| `schemas.py` | Pydantic request/response models; validation at the boundary (non-empty text, ≤ ~2000 chars pre-tokenization; truncation to 512 tokens documented) |
| `routes.py` | API endpoints |

**Endpoints**

- `POST /api/analyze` — `{text}` → `{label, scores: {negative, neutral, positive}}`
- `POST /api/analyze/batch` — CSV upload (one `text` column, ≤ 500 rows) or JSON list → per-row results + aggregates `{counts, mean_scores}`; inference is batched (educational point: GPU batching vs. per-row loop)
- `POST /api/explain` — `{text}` → `{label, scores, tokens: [{token, attribution}]}`
- `GET /api/health` — `{status, model_loaded, device}`
- `GET /api/model` — model card info (name, labels, params, training data summary)

**Error handling:** Pydantic 422s with clear messages; 503 if model failed to load;
CSV parse errors → 400 with row-level detail; row limit enforced before inference.

### Frontend (`frontend/src/`)

- **Analyze tab** — textarea → label badge + per-class confidence bars; "Explain" button reveals token heatmap (tokens colored by IG attribution toward the predicted class, diverging color scale).
- **Batch tab** — CSV drag-and-drop → results table + aggregate charts (sentiment distribution pie/bar, mean confidence).
- **How it works page** — educational walkthrough: tokenization, transformer encoder, softmax, what Integrated Gradients computes. Static content, part of the portfolio pitch.
- Components: `AnalyzeForm`, `ConfidenceBars`, `TokenHeatmap`, `BatchUpload`, `ResultsTable`, `AggregateCharts`. Single API client module with typed responses.

## Rigor

- **Backend tests (pytest):** unit tests with the model mocked (schema validation, endpoint contracts, CSV handling, aggregation math); one `@pytest.mark.integration` test hitting the real model, excluded from CI, run locally.
- **Frontend tests (vitest + testing-library):** ConfidenceBars rendering, API client error paths, AnalyzeForm interaction.
- **Docker:** `docker-compose up` → backend (python slim, CPU-only torch wheel, HF cache volume) + frontend (multi-stage node build → nginx). Local dev remains conda + `npm run dev`.
- **CI (GitHub Actions):** two jobs — backend (ruff + pytest, mocked only), frontend (eslint + vitest + `vite build`). Runs on push/PR.
- **README:** architecture diagram, screenshots, quickstart (conda and Docker paths), "What this project demonstrates" section aimed at reviewers, honest limitations section (512-token truncation, English/Twitter-domain model, IG approximation steps).

## Out of Scope (v1)

- Auth, persistence/history database, rate limiting beyond row caps
- Model comparison / multiple models
- Streaming responses, websockets
- Deployment beyond docker-compose

## Success Criteria

1. `docker-compose up` serves the full app; conda + npm dev path also documented and working.
2. All three endpoints return correct shapes; batch of 500 rows completes without OOM.
3. IG heatmap visibly highlights sentiment-bearing words (e.g., "love", "terrible") on sample inputs.
4. CI green: lint + mocked tests + frontend build.
5. A reviewer can read any backend module top-to-bottom and follow the ML reasoning from comments alone.

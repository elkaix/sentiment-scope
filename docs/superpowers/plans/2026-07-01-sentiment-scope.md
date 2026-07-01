# SentimentScope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build **SentimentScope: Educational Sentiment Analysis, Explainability, and Model Comparison** — a portfolio-grade ML engineering app, not just a sentiment dashboard.

**Architecture:** FastAPI loads the default RoBERTa sentiment classifier once at startup. **Default sentiment endpoints** (`/api/analyze`, `/api/analyze/batch`, `/api/analyze/csv`, `/api/explain`) always use twitter-roberta with strict 3-class responses. **Sentiment comparison** is `/api/compare` only, with dynamic scores and lazy-loaded sentiment registry models. **AI text detection** is a separate task family exposed through `/api/ai-detect` and `/api/ai-detect/compare`, using detector-specific dynamic labels, detector disagreement reporting, and clear uncertainty warnings. A Vite + React + TS frontend calls relative `/api` paths. Unit tests mock all models via FastAPI dependency override so CI never downloads torch; real-model, multi-model, and detector tests are `@pytest.mark.integration` and run locally only. `evals/` produces accuracy, macro F1, confusion matrix, latency, wrong-example analysis, and detector-disagreement reports.

**Tech Stack (latest verified 2026-07-01, re-checked via PyPI/npm API + Context7 + Tavily):** Python 3.13+, FastAPI 0.139.0, torch 2.12.1, transformers 5.12.1, captum 0.9.0, scikit-learn 1.9.0, pytest 9.1.1, ruff 0.15.20 · React 19.2.7, Vite 8.1.2, TypeScript 6.0.3, Tailwind 4.3.2, Recharts 3.9.1, Vitest 4.1.9, Testing Library React 16.3.2 · Node 24 LTS (CI/Docker) · Docker + docker-compose · GitHub Actions.

## Global Constraints

- Project root: `~/Projects/active/sentiment-scope` (git repo already initialized; spec committed).
- Default model: `cardiffnlp/twitter-roberta-base-sentiment-latest`, labels `negative/neutral/positive`, 512-token truncation.
- Model registry adds lazy optional **sentiment** models: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (binary SST-2 baseline), `ProsusAI/finbert` (financial text), and `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual social text). **AI text detectors** (`desklib`, `fakespot`, `oxidane`) are a separate task family in the same registry (Task 8, 19) — never mixed into `/api/compare`.
- Local dev backend runs in the existing `ai` conda env (`ca ai`); upgrade to torch 2.12.1 + transformers 5.12.1 + captum 0.9.0 before integration/eval work.
- Batch limits: ≤ 500 texts per request, ≤ 2000 chars per text. Validation at the boundary (Pydantic / explicit HTTP 400s).
- **Educational tone is a hard requirement:** every backend module and every non-trivial frontend component gets explanatory comments teaching the ML/engineering concept involved (why softmax, why batching, what IG computes, why validation lives at the boundary). Code in this plan already includes them — copy verbatim, don't strip them.
- **Do not copy plan audit tags into source code.** This plan uses `<!-- ✅ VALIDATED ... -->` (markdown) and similar markers to trace review fixes. Those are for the plan document only — never paste them into `.py`, `.ts`, `.tsx`, Dockerfiles, or YAML. Source comments should teach ML/engineering concepts, not document review history.
- No Claude co-author trailers or "Generated with Claude Code" footers in any commit.
- Frontend API calls always use relative `/api/...` paths — never hardcode `localhost:8000` in components.
- CI must not install torch/transformers/captum (unit tests never import them — heavy imports live inside `SentimentModel` methods).
- CSV/evaluation inputs must never be silently mutated. Reject overlong rows with a clear 400 instead of truncating.
- Sentiment models are not creativity judges. The creative-writing eval measures polarity, confidence, disagreement, and explanation quality across writing styles.
- Do not load all models in Docker by default. The default container loads only `twitter-roberta`. Optional models are lazy-loaded and may increase memory usage significantly on a laptop. <!-- ✅ VALIDATED 2026-07-01 (review fix #3): memory/latency warning — Tavily model sizes + review; four resident transformers can OOM/slow Docker on laptop -->
- AI text detection is a separate task family, not sentiment analysis. Never mix AI-detector models into sentiment `/api/compare`.
- AI detector endpoints use dynamic labels because local detector model configs may expose labels as `LABEL_0/LABEL_1`, `human/ai`, `real/fake`, or uppercase variants.
- Before finalizing detector labels, inspect each local detector model's `config.json` and map raw labels into canonical `human` / `ai` keys.
- AI detectors are probabilistic and must not be presented as proof of authorship. The UI and README must show an uncertainty warning, especially when detectors disagree.

## Implementation order

> **✅ VALIDATED 2026-07-01 (review fix #8):** ML core before UI — review milestone reorder; keeps compare/eval solid before frontend polish.

Build the ML core before UI polish. Phase 1 ships a complete, publicly deployed sentiment app; Phase 2 adds AI text detection as clean follow-on tasks (19–22) — do not interleave them.

**Phase 1**

1. Single-model backend (Tasks 1–6)
2. Explainability (Task 7)
3. Model registry (Task 8)
4. Compare API (Task 9)
5. Sentiment evaluation harness (Task 10)
6. Frontend (Tasks 11–15)
7. Docker/CI (Tasks 16–17)
8. Free public deployment — Hugging Face Spaces (Task 16A, after CI is green)
9. README/screenshots (Task 18)

**Phase 2 — AI text detection (tasks live at the end of this plan, after Task 18)**

10. AI text detection backend (Task 19)
11. AI detector evaluation + disagreement report (Task 20)
12. AI Detector frontend tab (Task 21)
13. README/Space update for detection (Task 22)

## Review fixes — validated

| # | Fix | Where | Validated |
|---|-----|-------|-----------|
| 1 | `DynamicAnalyzeResponse` / dynamic scores for compare (DistilBERT is binary) | Task 2, 9, 11, 12 | ✅ 2026-07-01 — Tavily HF registry labels `("negative","positive")`; fixed `Scores` would break Pydantic validation |
| 2 | `DEFAULT_COMPARE_MODELS` = 2 models, not all 4 | Task 9, 15 | ✅ 2026-07-01 — review; avoids lazy-loading 4 models on first compare click |
| 3 | Docker memory warning (default model only) | Global Constraints, Task 8 | ✅ 2026-07-01 — review + ~500MB/model estimate |
| 4 | Per-model `asyncio.Lock` on lazy load | Task 8 | ✅ 2026-07-01 — review; prevents duplicate concurrent downloads |
| 5 | Node 24 LTS (not 26) in CI/Docker | Task 16, 17, Research | ✅ 2026-07-01 — Context7 Vite 8 engines `^20.19 \|\| >=22.12`; Tavily Node 24 LTS until Apr 2028 |
| 6 | `requirements-docker.txt` (torch via CPU index only) | Task 1, 16 | ✅ 2026-07-01 — review; PyPI torch 2.12.1 confirmed |
| 7 | `scikit-learn` in `requirements-eval.txt`, not dev/CI | Task 1, 10, 17 | ✅ 2026-07-01 — review; PyPI scikit-learn 1.9.0 confirmed; CI stays light |
| 8 | Implementation order: backend → compare → eval → frontend | above | ✅ 2026-07-01 — review |
| 9 | `/api/explain` RoBERTa-only (`roberta.embeddings` is not universal) | Task 7, 8 | ✅ 2026-07-01 — Context7 Captum uses `model.bert.embeddings`; plan uses `roberta.embeddings`; Tavily/HF DistilBERT is BERT-family |
| 10 | Lazy load via `await asyncio.to_thread(m.load)` | Task 8 | ✅ 2026-07-01 — review + Context7: blocking work off the event loop |
| 11 | Registry `ModelConfig.labels` as canonical labels (not raw `id2label`) | Task 8 | ✅ 2026-07-01 — HF config: DistilBERT `NEGATIVE`/`POSITIVE`; FinBERT order differs from registry |
| 12 | Eval: fail if dataset labels ⊄ model labels unless `--allow-label-mismatch` | Task 10 | ✅ 2026-07-01 — review; binary model on 3-class CSV is misleading |
| 13 | Plan `✅ VALIDATED` tags stay in the plan only — never copy into source | Global Constraints | ✅ 2026-07-01 — review polish |
| 14 | `/api/analyze` default-model only; optional models in `/api/compare` only | Task 7, 8, README | ✅ 2026-07-01 — review; strict 3-class `Scores` breaks on binary DistilBERT |

**Plan vs source:** HTML `<!-- ✅ VALIDATED ... -->` comments in this document are audit trail only. When implementing, copy code snippets **without** those tags (see Global Constraints).

## Research + version validation notes

**Re-verified 2026-07-01** against live PyPI/npm JSON APIs, Context7 docs, and Tavily web search. All plan pins match current releases — no version bumps needed.

### Python (PyPI — authoritative)

| Package | Plan pin | PyPI latest | Released |
|---------|----------|-------------|----------|
| fastapi | 0.139.0 | 0.139.0 | 2026-07-01 |
| torch | 2.12.1 | 2.12.1 | 2026-06-17 |
| transformers | 5.12.1 | 5.12.1 | 2026-06-15 |
| captum | 0.9.0 | 0.9.0 | 2026-04-17 |
| scikit-learn | 1.9.0 | 1.9.0 | 2026-06-02 |
| pytest | 9.1.1 | 9.1.1 | — |
| ruff | 0.15.20 | 0.15.20 | — |
| uvicorn | 0.49.0 | 0.49.0 | — |
| python-multipart | 0.0.32 | 0.0.32 | — |
| httpx | 0.28.1 | 0.28.1 | — |

### Frontend (npm — authoritative)

| Package | Plan pin | npm latest |
|---------|----------|------------|
| react / react-dom | 19.2.7 | 19.2.7 |
| vite | 8.1.2 | 8.1.2 |
| typescript | 6.0.3 | 6.0.3 |
| tailwindcss / @tailwindcss/vite | 4.3.2 | 4.3.2 |
| recharts | 3.9.1 | 3.9.1 |
| vitest | 4.1.9 | 4.1.9 |
| @testing-library/react | 16.3.2 | 16.3.2 |
| @vitejs/plugin-react | 6.0.3 | 6.0.3 |
| jsdom | 29.1.1 | 29.1.1 |
| @testing-library/jest-dom | 6.9.1 | 6.9.1 |
| @testing-library/user-event | 14.6.1 | 14.6.1 |

### Node.js (CI/Docker)

> **✅ VALIDATED 2026-07-01 (review fix #5):** Node 24 over Node 26 for showcase CI/Docker.

- Plan uses **Node 24** (`node-version: "24"`, `FROM node:24-alpine`) — confirmed good choice.
- Node 24 entered LTS October 2025; supported until **April 2028** (Tavily/Node release schedule).
- Latest Node 24 patch: **24.18.0** (npm registry). Pinning `"24"` in CI is fine; Docker `node:24-alpine` tracks current 24.x.
- **Vite 8** requires `^20.19.0 || >=22.12.0` (Context7 / Vite 8 announcement) — Node 24 satisfies this. Node 26 is not required.

### Context7 (docs/API patterns — indexes may lag PyPI)

- **FastAPI** (`/websites/fastapi_tiangolo`): confirms `lifespan` + `@asynccontextmanager`, `Depends`, `UploadFile`/`File`, TestClient lifespan. Indexed versions top out at 0.128.0; live PyPI is 0.139.0 — API patterns unchanged.
- **Vite** (`/vitejs/vite`): confirms React TS starter, `server.proxy` `/api` syntax, Vitest config. Vite 8.0.10 indexed; live npm is 8.1.2. Node engine: `^20.19.0 || >=22.12.0`.
- **Captum** (`/meta-pytorch/captum`): `LayerIntegratedGradients` attaches to architecture-specific embedding modules — BERT tutorials use `model.bert.embeddings`, not `.roberta`. Explainability is RoBERTa-only unless per-architecture adapters are added (review fix #9).
- **Transformers** (`/websites/huggingface_co_transformers_main`): indexed v5.4.0; live PyPI is 5.12.1 — `AutoModelForSequenceClassification` / `AutoTokenizer` usage unchanged.

### Tavily (model/domain validation)

- Cardiff Twitter RoBERTa: ~124M tweets (Jan 2018–Dec 2021), fine-tuned on TweetEval — default model choice validated.
- DistilBERT SST-2: binary baseline (no neutral) — motivates `DynamicAnalyzeResponse` for compare. HF `config.json` id2label is `NEGATIVE`/`POSITIVE` (uppercase) — registry uses lowercase; use `ModelConfig.labels`, not raw `id2label` (review fix #11).
- FinBERT: financial domain; label order in HF config (`positive`, `negative`, `neutral`) differs from registry tuple — same fix: canonical labels from registry.
- XLM-R Twitter: ~198M tweets, multilingual — registry choice validated.
- scikit-learn: `classification_report(..., output_dict=True)` + `confusion_matrix` for eval harness — validated.

---

### Task 1: Backend scaffold + health endpoint

**Files:**

- Create: `.gitignore`, `backend/pyproject.toml`, `backend/requirements.txt`, `backend/requirements-docker.txt`, `backend/requirements-dev.txt`, `backend/requirements-eval.txt`, `backend/app/__init__.py`, `backend/app/main.py`, `backend/app/model.py` (skeleton), `backend/app/routes.py` (health only)
<!-- ✅ VALIDATED 2026-07-01 (review fix #6, #7): split requirements-docker.txt + requirements-eval.txt — PyPI pins verified -->
- Test: `backend/tests/__init__.py`, `backend/tests/conftest.py`, `backend/tests/test_health.py`

**Interfaces:**

- Produces: `app.main.app` (FastAPI instance, lifespan stores `SentimentModel` on `app.state.model`, honors `SKIP_MODEL_LOAD=1`); `app.model.SentimentModel` with `MODEL_NAME`, `MAX_TOKENS`, `is_loaded`, `device`, `labels`, and `load()` stub; `GET /api/health` → `{status, model_loaded, device}`. All later backend tasks import these.

- [ ] **Step 1: Create scaffold files**

`.gitignore` (repo root):

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.venv/
node_modules/
dist/
.env
.DS_Store
```

`backend/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: needs the real model downloaded — run locally, excluded by default",
]
# Integration tests are opt-in: plain `pytest` (and CI) skips them.
addopts = "-m 'not integration'"

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

`backend/requirements.txt` (full runtime — used by Docker; upgrade the local `ai` conda env to these pins before integration/eval runs):

```
fastapi==0.139.0
uvicorn[standard]==0.49.0
python-multipart==0.0.32
torch==2.12.1
transformers==5.12.1
captum==0.9.0
```

`backend/requirements-docker.txt` (runtime deps for Docker — torch installed separately via CPU wheel index):
<!-- ✅ VALIDATED 2026-07-01 (review fix #6): no duplicate torch in pip -r; install torch==2.12.1 from CPU index first — PyPI 2.12.1 confirmed -->

```
fastapi==0.139.0
uvicorn[standard]==0.49.0
python-multipart==0.0.32
transformers==5.12.1
captum==0.9.0
```

`backend/requirements-dev.txt` (what unit tests + CI need — deliberately NO torch or sklearn, see Global Constraints):
<!-- ✅ VALIDATED 2026-07-01 (review fix #7): scikit-learn removed from CI deps — review; unit tests mock model, no sklearn needed -->

```
fastapi==0.139.0
uvicorn[standard]==0.49.0
python-multipart==0.0.32
httpx==0.28.1
pytest==9.1.1
ruff==0.15.20
```

`backend/requirements-eval.txt` (eval harness only — not installed in CI):
<!-- ✅ VALIDATED 2026-07-01 (review fix #7): scikit-learn 1.9.0 eval-only — PyPI latest confirmed -->

```
scikit-learn==1.9.0
```

`backend/app/__init__.py` and `backend/tests/__init__.py`: empty files.

- [ ] **Step 2: Write the failing test**

`backend/tests/conftest.py`:

```python
"""Shared test fixtures.

SKIP_MODEL_LOAD must be set BEFORE importing the app: the FastAPI lifespan
reads it to decide whether to download/load the ~500MB transformer. Unit
tests never touch the real model — that's what keeps CI fast and free of
torch as a dependency.
"""

import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # TestClient used as a context manager so the lifespan (startup/shutdown) runs.
    with TestClient(app) as c:
        yield c
```

`backend/tests/test_health.py`:

```python
def test_health_reports_model_not_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # SKIP_MODEL_LOAD=1 in tests
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Projects/active/sentiment-scope/backend && pytest tests/test_health.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 4: Write minimal implementation**

`backend/app/model.py`:

```python
"""SentimentModel — a thin, educational wrapper around a HuggingFace classifier.

Design notes (the "why", since the "what" is short):

* Singleton lifecycle: transformer weights are ~500MB in RAM. We load them
  exactly once at server startup (see main.py lifespan) instead of per
  request — loading takes seconds, inference takes milliseconds.
* Lazy imports: torch/transformers are imported INSIDE methods, not at module
  top. That means importing this module (e.g. from unit tests or CI) costs
  nothing, and the heavy libraries are only pulled in when the model actually
  loads. CI runs the whole API test suite without torch installed.
"""


class SentimentModel:
    # A RoBERTa-base encoder fine-tuned on ~124M tweets for 3-class sentiment.
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # RoBERTa's positional embeddings cap sequence length; longer inputs are truncated.
    MAX_TOKENS = 512

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        self.labels: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run only) and load tokenizer + model weights."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Apple Silicon GPU (MPS) when available; plain CPU in Docker/CI.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.to(self.device)
        # eval() disables dropout etc. — we want deterministic inference, not training.
        self._model.eval()
        # Read label names from the model config instead of hardcoding:
        # id2label = {0: "negative", 1: "neutral", 2: "positive"} for this model.
        self.labels = [
            self._model.config.id2label[i] for i in range(self._model.config.num_labels)
        ]
```

`backend/app/routes.py`:

```python
"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")


@router.get("/health")
def health(request: Request):
    """Liveness + readiness in one: 'ok' means the server is up;
    model_loaded tells you whether inference will actually work."""
    model = request.app.state.model
    return {"status": "ok", "model_loaded": model.is_loaded, "device": model.device}
```

`backend/app/main.py`:

```python
"""FastAPI application entrypoint.

Run locally (inside the `ai` conda env):
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import SentimentModel
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE for the process lifetime. Every request shares it —
    # safe because inference is read-only (no weight mutation after eval()).
    model = SentimentModel()
    app.state.model = model
    # Tests set SKIP_MODEL_LOAD=1 so the suite runs in milliseconds without torch.
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        model.load()
    yield


app = FastAPI(title="SentimentScope API", lifespan=lifespan)

# CORS is only needed when the frontend is served from a different origin
# (npm dev server without the proxy, or direct API access). The Vite proxy
# and nginx make requests same-origin, but this keeps direct access working.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Projects/active/sentiment-scope/backend && pip install -r requirements-dev.txt >/dev/null && pytest -v && ruff check .`
(Use the `ai` conda env: `ca ai` first. fastapi/pytest may already be present.)
Expected: 1 passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/active/sentiment-scope && git add -A && git commit -m "feat: backend scaffold with health endpoint and test harness"
```

---

### Task 2: Pydantic schemas

**Files:**

- Create: `backend/app/schemas.py`
- Test: `backend/tests/test_schemas.py`

**Interfaces:**

- Produces: `AnalyzeRequest(text)`, `Scores(negative, neutral, positive)` (strict 3-class shape for default-model endpoints), `AnalyzeResponse(label, scores: Scores)`, `DynamicAnalyzeResponse(label, scores: dict[str, float])` (model-agnostic — used by compare), `BatchRequest(texts)`, `BatchItem(text, label, scores)`, `BatchAggregates(counts, mean_scores)`, `BatchResponse(results, aggregates)`, `TokenAttribution(token, attribution)`, `ExplainResponse(label, scores)`. Tasks 4–7 import the strict shapes; Task 9 uses `DynamicAnalyzeResponse`.
<!-- ✅ VALIDATED 2026-07-01 (review fix #1): strict Scores for default endpoints; dynamic dict for compare — DistilBERT SST-2 labels validated via Tavily/HF -->

- [ ] **Step 1: Write the failing test**

`backend/tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from app.schemas import AnalyzeRequest, BatchRequest


def test_analyze_rejects_blank_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="   ")


def test_analyze_rejects_overlong_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="x" * 2001)


def test_analyze_strips_whitespace():
    assert AnalyzeRequest(text="  hello  ").text == "hello"


def test_batch_rejects_blank_items():
    with pytest.raises(ValidationError):
        BatchRequest(texts=["fine", "   "])


def test_batch_rejects_over_500_items():
    with pytest.raises(ValidationError):
        BatchRequest(texts=["x"] * 501)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/active/sentiment-scope/backend && pytest tests/test_schemas.py -v`
Expected: ERROR — `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Write minimal implementation**

`backend/app/schemas.py`:

```python
"""Request/response contracts.

Principle: validate at the boundary. Bad input is rejected here — with a
descriptive 422 — before any of it reaches the model. The ML code can then
assume clean input, which keeps it simple.
"""

from pydantic import BaseModel, Field, field_validator

# Character cap is a cheap pre-tokenization guard. The tokenizer still
# truncates to 512 tokens, but rejecting huge payloads early protects the
# server from pathological requests.
MAX_CHARS = 2000
MAX_BATCH = 500


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class Scores(BaseModel):
    """Softmax output — one probability per class, summing to ~1.0."""

    negative: float
    neutral: float
    positive: float


class AnalyzeResponse(BaseModel):
    label: str
    scores: Scores


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def no_blank_items(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("texts must not contain blank entries")
        if any(len(t) > MAX_CHARS for t in cleaned):
            raise ValueError(f"each text must be at most {MAX_CHARS} characters")
        return cleaned


class BatchItem(AnalyzeResponse):
    text: str


class BatchAggregates(BaseModel):
    counts: dict[str, int]
    mean_scores: Scores


class BatchResponse(BaseModel):
    results: list[BatchItem]
    aggregates: BatchAggregates


class TokenAttribution(BaseModel):
    """One token and its Integrated Gradients attribution toward the
    predicted class. Positive pushes toward the prediction, negative away."""

    token: str
    attribution: float


class ExplainResponse(AnalyzeResponse):
    tokens: list[TokenAttribution]


class DynamicAnalyzeResponse(BaseModel):
    """Model-agnostic response — scores keys match whatever labels the model emits.
    Use this for /api/compare where binary (DistilBERT) and 3-class models coexist.
    Do NOT fake a neutral score for binary models."""

    label: str
    scores: dict[str, float]
```
<!-- ✅ VALIDATED 2026-07-01 (review fix #1): DynamicAnalyzeResponse for compare — DistilBERT binary labels via Tavily/HF -->

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v` — Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: request/response schemas with boundary validation"
```

---

### Task 3: SentimentModel.predict (real inference)

**Files:**

- Modify: `backend/app/model.py` (add `predict`)
- Test: `backend/tests/test_model_integration.py`

**Interfaces:**

- Produces: `SentimentModel.predict(texts: list[str]) -> list[dict]`. For the default model, each dict is `{"label": str, "scores": {"negative": float, "neutral": float, "positive": float}}` — Tasks 4–6 rely on this shape. After Task 8, registry-backed models return score keys matching `ModelConfig.labels` in logit order (used by compare/eval, not by strict `/api/analyze`).
<!-- ✅ VALIDATED 2026-07-01 (review fix #14): analyze stays 3-class; registry predict shape is compare-only -->

- [ ] **Step 1: Write the (integration) test**

`backend/tests/test_model_integration.py`:

```python
"""Real-model tests. Excluded by default (see pyproject addopts); run with:
    pytest -m integration -v
Requires the `ai` conda env (torch + transformers installed, weights cached).
"""

import pytest


@pytest.mark.integration
def test_predict_on_obvious_sentiment():
    from app.model import SentimentModel

    m = SentimentModel()
    m.load()
    out = m.predict(["I love this so much!", "This is absolutely terrible."])

    assert out[0]["label"] == "positive"
    assert out[1]["label"] == "negative"
    for r in out:
        # Probabilities must behave like probabilities.
        assert abs(sum(r["scores"].values()) - 1.0) < 0.01
        assert set(r["scores"]) == {"negative", "neutral", "positive"}
```

- [ ] **Step 2: Verify it fails**

Run: `cd ~/Projects/active/sentiment-scope/backend && pytest -m integration -v` (inside `ca ai`)
Expected: FAIL — `AttributeError: 'SentimentModel' object has no attribute 'predict'`.

- [ ] **Step 3: Implement predict**

Add to `backend/app/model.py` inside the class:

```python
    def predict(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts. The full flow, spelled out:

        1. Tokenize: text -> subword IDs. padding=True pads the batch to the
           longest member so it forms one rectangular tensor; truncation
           enforces the 512-token model limit.
        2. Forward pass: one batched call. Batching is THE key GPU win —
           classifying 500 texts in one tensor is dramatically faster than
           500 single-text calls, because the per-call overhead is paid once.
        3. Softmax: the model outputs logits (unnormalized scores). Softmax
           maps them to probabilities that sum to 1, which is what humans
           (and our confidence bars) actually want to read.
        """
        import torch

        assert self.is_loaded, "call load() before predict()"
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_TOKENS,
        ).to(self.device)

        # no_grad(): we're not training, so skip building the autograd graph
        # — less memory, more speed.
        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu()
        results = []
        for row in probs:
            scores = {label: round(float(p), 4) for label, p in zip(self.labels, row)}
            results.append({"label": max(scores, key=scores.get), "scores": scores})
        return results
```

- [ ] **Step 4: Verify it passes**

Run: `pytest -m integration -v` — Expected: 1 passed (first run may download weights; they're already cached locally).
Also run `pytest -v` and confirm the integration test is NOT collected there (deselected).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: batched predict with tokenize->logits->softmax flow"
```

---

### Task 4: POST /api/analyze

**Files:**

- Modify: `backend/app/routes.py` (add `get_model` dependency + endpoint)
- Modify: `backend/tests/conftest.py` (add `FakeModel` + `client_with_model` fixture)
- Test: `backend/tests/test_analyze.py`

**Interfaces:**

- Produces: `routes.get_model(request) -> SentimentModel` (503 if not loaded — tests override this dependency); `POST /api/analyze` accepting `AnalyzeRequest`, returning `AnalyzeResponse`. `FakeModel` fixture (labels, device="cpu", `predict`, `explain`) reused by Tasks 5–7 tests.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/conftest.py`:

```python
class FakeModel:
    """Stands in for SentimentModel in unit tests. Same interface, canned
    output — API tests check wiring/validation, not ML quality (the
    integration suite covers that)."""

    is_loaded = True
    device = "cpu"
    labels = ["negative", "neutral", "positive"]

    def predict(self, texts):
        return [
            {"label": "positive", "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.8}}
            for _ in texts
        ]

    def explain(self, text):
        return {
            "label": "positive",
            "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.8},
            "tokens": [
                {"token": "I", "attribution": 0.01},
                {"token": " love", "attribution": 0.92},
                {"token": " this", "attribution": 0.05},
            ],
        }


@pytest.fixture
def client_with_model():
    from app.routes import get_model

    app.dependency_overrides[get_model] = lambda: FakeModel()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

`backend/tests/test_analyze.py`:

```python
def test_analyze_returns_label_and_scores(client_with_model):
    resp = client_with_model.post("/api/analyze", json={"text": "I love this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["scores"]["positive"] == 0.8


def test_analyze_blank_text_is_422(client_with_model):
    resp = client_with_model.post("/api/analyze", json={"text": "  "})
    assert resp.status_code == 422


def test_analyze_without_model_is_503(client):
    # No dependency override + SKIP_MODEL_LOAD → model never loaded → 503.
    resp = client.post("/api/analyze", json={"text": "hello"})
    assert resp.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze.py -v`
Expected: ImportError on `get_model` / 404s — no endpoint yet.

- [ ] **Step 3: Implement**

Replace `backend/app/routes.py` imports/top and add the endpoint (keep the existing `health` route):

```python
"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.model import SentimentModel
from app.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/api")


def get_model(request: Request) -> SentimentModel:
    """Dependency injection: routes ask FOR a model instead of reaching into
    globals. Tests swap in a FakeModel with one line (dependency_overrides),
    and a not-yet-loaded model becomes a clean 503 instead of a crash."""
    model = request.app.state.model
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


@router.get("/health")
def health(request: Request):
    """Liveness + readiness in one: 'ok' means the server is up;
    model_loaded tells you whether inference will actually work."""
    model = request.app.state.model
    return {"status": "ok", "model_loaded": model.is_loaded, "device": model.device}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, model: SentimentModel = Depends(get_model)):
    # predict() is batched by design; a single text is just a batch of one.
    return model.predict([req.text])[0]
```

- [ ] **Step 4: Verify all tests pass**

Run: `pytest -v && ruff check .` — Expected: all pass (health, schemas, analyze), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: /api/analyze with model dependency injection"
```

---

### Task 5: POST /api/analyze/batch + aggregation

**Files:**

- Modify: `backend/app/routes.py`
- Test: `backend/tests/test_batch.py`

**Interfaces:**

- Produces: `routes.aggregate(results: list[dict]) -> dict` (`{"counts": {label: int}, "mean_scores": {label: float}}`) — reused verbatim by Task 6; `POST /api/analyze/batch` accepting `BatchRequest`, returning `BatchResponse`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_batch.py`:

```python
from app.routes import aggregate


def test_aggregate_counts_and_means():
    results = [
        {"label": "positive", "scores": {"negative": 0.1, "neutral": 0.1, "positive": 0.8}},
        {"label": "negative", "scores": {"negative": 0.7, "neutral": 0.2, "positive": 0.1}},
    ]
    agg = aggregate(results)
    assert agg["counts"] == {"negative": 1, "neutral": 0, "positive": 1}
    assert agg["mean_scores"]["positive"] == 0.45
    assert agg["mean_scores"]["negative"] == 0.4


def test_batch_endpoint_returns_items_and_aggregates(client_with_model):
    resp = client_with_model.post("/api/analyze/batch", json={"texts": ["a good day", "another"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["text"] == "a good day"
    assert body["aggregates"]["counts"]["positive"] == 2


def test_batch_rejects_501_texts(client_with_model):
    resp = client_with_model.post("/api/analyze/batch", json={"texts": ["x"] * 501})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch.py -v` — Expected: ImportError (`aggregate` doesn't exist).

- [ ] **Step 3: Implement**

Add to `backend/app/routes.py` (extend the schemas import too):

```python
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
)

LABELS = ("negative", "neutral", "positive")


def aggregate(results: list[dict]) -> dict:
    """Summarize a batch: label counts + mean per-class probability.
    Plain Python (no numpy) on purpose — for ≤500 rows this is instant, and
    the arithmetic stays readable for anyone auditing the math."""
    counts = {label: 0 for label in LABELS}
    sums = dict.fromkeys(LABELS, 0.0)
    for r in results:
        counts[r["label"]] += 1
        for label, score in r["scores"].items():
            sums[label] += score
    n = len(results)
    return {
        "counts": counts,
        "mean_scores": {label: round(s / n, 4) for label, s in sums.items()},
    }


@router.post("/analyze/batch", response_model=BatchResponse)
def analyze_batch(req: BatchRequest, model: SentimentModel = Depends(get_model)):
    # ONE batched model call for the whole list — see predict()'s docstring
    # for why that beats a per-text loop.
    results = model.predict(req.texts)
    items = [{"text": t, **r} for t, r in zip(req.texts, results)]
    return {"results": items, "aggregates": aggregate(results)}
```

- [ ] **Step 4: Verify** — `pytest -v && ruff check .` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: batch analysis endpoint with aggregates"
```

---

### Task 6: POST /api/analyze/csv (file upload)

**Files:**

- Modify: `backend/app/routes.py`
- Test: `backend/tests/test_csv.py`

**Interfaces:**

- Produces: `POST /api/analyze/csv` — multipart upload, field name `file`, CSV must have a `text` column, ≤500 data rows; returns `BatchResponse` (same shape as batch). Frontend Task 14 posts `FormData` here.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_csv.py`:

```python
import io


def _upload(client, content: bytes, name="data.csv"):
    return client.post("/api/analyze/csv", files={"file": (name, io.BytesIO(content), "text/csv")})


def test_csv_happy_path(client_with_model):
    csv_bytes = b"text\nI love this\nawful experience\n"
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][1]["text"] == "awful experience"


def test_csv_missing_text_column_is_400(client_with_model):
    resp = _upload(client_with_model, b"comment\nhello\n")
    assert resp.status_code == 400
    assert "text" in resp.json()["detail"]


def test_csv_over_500_rows_is_400(client_with_model):
    csv_bytes = b"text\n" + b"row\n" * 501
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 400


def test_csv_skips_blank_rows(client_with_model):
    csv_bytes = b"text\nhello\n\n   \nworld\n"
    resp = _upload(client_with_model, csv_bytes)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_csv_non_utf8_is_400(client_with_model):
    resp = _upload(client_with_model, b"text\n\xff\xfe broken \xff\n")
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_csv.py -v` — Expected: 404s (no endpoint).

- [ ] **Step 3: Implement**

Add to `backend/app/routes.py` (add `csv`, `io` to imports at top; add `UploadFile` and `File` to the fastapi import; add `MAX_BATCH`, `MAX_CHARS` to the schemas import):

```python
import csv
import io
```

```python
@router.post("/analyze/csv", response_model=BatchResponse)
async def analyze_csv(file: UploadFile = File(...), model: SentimentModel = Depends(get_model)):
    """CSV variant of batch analysis. File parsing is inherently messier than
    JSON, so each failure mode gets its own explicit 400 — the caller should
    always learn WHY their file was rejected."""
    raw = await file.read()
    try:
        # utf-8-sig also swallows the BOM that Excel loves to prepend.
        text_stream = io.StringIO(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    reader = csv.DictReader(text_stream)
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must have a 'text' column")

    texts: list[str] = []
    for i, row in enumerate(reader):
        if i >= MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"CSV exceeds {MAX_BATCH} row limit")
        t = (row.get("text") or "").strip()
        if len(t) > MAX_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Row {i + 2} exceeds {MAX_CHARS} characters",
            )
        if t:
            texts.append(t)

    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty rows found")

    results = model.predict(texts)
    items = [{"text": t, **r} for t, r in zip(texts, results)]
    return {"results": items, "aggregates": aggregate(results)}
```

- [ ] **Step 4: Verify** — `pytest -v && ruff check .` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: CSV upload endpoint with explicit failure modes"
```

---

### Task 7: Explainability (captum IG) + /api/explain + /api/model

**Files:**

- Modify: `backend/app/model.py` (add `explain`), `backend/app/routes.py` (two endpoints)
- Test: `backend/tests/test_explain.py`, extend `backend/tests/test_model_integration.py`

**Interfaces:**

- Produces: `SentimentModel.explain(text: str) -> dict` (`{label, scores, tokens: [{token, attribution}]}`); `POST /api/explain` (`AnalyzeRequest` → `ExplainResponse`, **twitter-roberta only** — rejects other `model_id` with 400); `GET /api/model` → `{name, labels, max_tokens, device, description}`. Frontend Tasks 13/15 consume these.
<!-- ✅ VALIDATED 2026-07-01 (review fix #9): IG uses self._model.roberta.embeddings — Context7 Captum BERT uses bert.embeddings -->

- [ ] **Step 1: Install captum in the ai env**

Run: `ca ai && pip install captum`
Expected: installs cleanly against torch 2.12.1.

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_explain.py`:

```python
def test_explain_returns_token_attributions(client_with_model):
    resp = client_with_model.post("/api/explain", json={"text": "I love this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    tokens = body["tokens"]
    assert tokens[1]["token"] == " love"
    assert tokens[1]["attribution"] > tokens[0]["attribution"]


def test_model_info(client_with_model):
    resp = client_with_model.get("/api/model")
    assert resp.status_code == 200
    body = resp.json()
    assert "roberta" in body["name"]
    assert body["labels"] == ["negative", "neutral", "positive"]
    assert body["max_tokens"] == 512


def test_explain_rejects_non_roberta_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/explain?model_id=distilbert-sst2", json={"text": "I love this"}
    )
    assert resp.status_code == 400
    assert "twitter-roberta" in resp.json()["detail"].lower()
```
<!-- ✅ VALIDATED 2026-07-01 (review fix #9): 400 when explain requested for non-RoBERTa model -->

Append to `backend/tests/test_model_integration.py`:

```python
@pytest.mark.integration
def test_explain_highlights_sentiment_words():
    from app.model import SentimentModel

    m = SentimentModel()
    m.load()
    out = m.explain("I absolutely love this phone")

    assert out["label"] == "positive"
    attrs = {t["token"].strip().lower(): t["attribution"] for t in out["tokens"]}
    # "love" should drive the positive prediction more than the stopword "this".
    assert attrs["love"] > attrs["this"]
```

- [ ] **Step 3: Verify failures**

Run: `pytest tests/test_explain.py -v` — Expected: 404s.

- [ ] **Step 4: Implement explain()**

Add to `backend/app/model.py` inside the class:

```python
    def explain(self, text: str) -> dict:
        """Token-level attribution via Layer Integrated Gradients (captum).

        What IG computes, in one paragraph: pick a "no information" baseline
        input (here: all padding tokens), then walk a straight line in
        embedding space from that baseline to the real input in n_steps
        increments, accumulating the model's gradient at each step. The
        integral assigns each input token a share of the change in the
        predicted class's logit. Tokens with large positive attribution
        pushed the model TOWARD its prediction; negative pushed away.
        We attach IG at the embedding layer (LayerIntegratedGradients)
        because raw token IDs are discrete — you can't differentiate
        through an integer lookup, but you can through its embedding.

        RoBERTa-only for now: LayerIntegratedGradients is wired to
        self._model.roberta.embeddings. DistilBERT uses distilbert.* and
        BERT uses bert.* — other registry models can be compared but not
        explained until per-architecture embedding hooks are added.
        """
        import torch
        from captum.attr import LayerIntegratedGradients

        assert self.is_loaded, "call load() before explain()"
        prediction = self.predict([text])[0]
        target = self.labels.index(prediction["label"])

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.MAX_TOKENS
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        def forward(ids, mask):
            return self._model(input_ids=ids, attention_mask=mask).logits

        lig = LayerIntegratedGradients(forward, self._model.roberta.embeddings)

        # Baseline = same length, but every content token replaced by <pad>,
        # keeping the <s>/</s> specials in place. "What would the model say
        # about a sentence with no words in it?"
        baseline = torch.full_like(input_ids, self._tokenizer.pad_token_id)
        baseline[0, 0] = input_ids[0, 0]
        baseline[0, -1] = input_ids[0, -1]

        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=50,  # more steps = better integral approximation, slower
        )
        # One attribution per (token, embedding_dim); collapse the embedding
        # axis and L2-normalize so the UI gets comparable magnitudes.
        scores = attributions.sum(dim=-1).squeeze(0)
        scores = scores / (torch.norm(scores) + 1e-9)

        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])
        special = {self._tokenizer.bos_token, self._tokenizer.eos_token, self._tokenizer.pad_token}
        token_attrs = [
            # "Ġ" is the BPE marker for "preceded by a space" — swap it back
            # for display so tokens rejoin into readable text.
            {"token": tok.replace("Ġ", " "), "attribution": round(float(a), 4)}
            for tok, a in zip(tokens, scores)
            if tok not in special
        ]
        return {**prediction, "tokens": token_attrs}
```

Add to `backend/app/routes.py` (extend schemas import with `ExplainResponse`):

```python
EXPLAIN_MODEL_ID = "twitter-roberta"


@router.post("/explain", response_model=ExplainResponse)
def explain(
    req: AnalyzeRequest,
    model_id: str | None = None,
    model: SentimentModel = Depends(get_model),
):
    # IG runs one forward pass per integration step (50x slower than
    # /analyze) — that's why explanation is a separate, opt-in endpoint.
    if model_id and model_id != EXPLAIN_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail="Explainability is currently supported only for twitter-roberta",
        )
    return model.explain(req.text)


@router.get("/model")
def model_info(model: SentimentModel = Depends(get_model)):
    return {
        "name": SentimentModel.MODEL_NAME,
        "labels": model.labels,
        "max_tokens": SentimentModel.MAX_TOKENS,
        "device": model.device,
        "description": (
            "RoBERTa-base fine-tuned on ~124M tweets (2018-2021) for 3-class "
            "sentiment classification, by Cardiff NLP."
        ),
    }
```
<!-- ✅ VALIDATED 2026-07-01 (review fix #9): model_id query rejected in Task 7 — test and route ship together -->

- [ ] **Step 5: Verify**

Run: `pytest -v && ruff check .` — Expected: all unit tests pass.
Run: `pytest -m integration -v` — Expected: 2 passed. **If MPS produces NaN attributions** (captum + MPS edge case), set `self.device = "cpu"` temporarily in `load()` to confirm, then keep MPS for predict and move only `explain` to CPU — do this by adding `.cpu()` copies of model/inputs inside `explain()` only if the NaN case actually occurs. Don't pre-build the fallback otherwise.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: Integrated Gradients explainability and model info endpoints"
```

---

### Task 8: Model registry + lazy model loading (task-aware)

**Files:**

- Create: `backend/app/model_registry.py`, `backend/tests/test_model_registry.py`
- Modify: `backend/app/model.py`, `backend/app/routes.py`, `backend/tests/conftest.py`

**Interfaces:**

- Produces: task-aware `MODEL_REGISTRY`, `ModelTask`, `models_for_task()`, `get_default_model_id()`, `get_model_config(model_id)`, lazy per-model cache in app state (`app.state.model_cache`, `app.state.model_locks`), `GET /api/models?task=`, and lazy optional **sentiment** models for `POST /api/compare` only. Default `/api/analyze`, `/api/analyze/batch`, `/api/analyze/csv`, and `/api/explain` remain twitter-roberta only (strict 3-class on analyze/batch/csv). AI detector models live in the same registry but are served only through `/api/ai-detect` (Task 19).
<!-- ✅ VALIDATED 2026-07-01 (review fix #14): no model_id on analyze — binary DistilBERT would break strict Scores schema -->

**Memory warning:** Do not load all models in Docker by default. The default container should load only `twitter-roberta`. Optional models are lazy-loaded on first request and may increase memory usage — loading four transformer models simultaneously can make Docker on a laptop feel broken.
<!-- ✅ VALIDATED 2026-07-01 (review fix #3): same as Global Constraints — review + lazy-load cost -->

- [ ] **Step 1: Create registry**

`backend/app/model_registry.py`:

```python
"""Explicit model choices for an educational ML app.

The registry is task-aware: sentiment models and AI text detectors share lazy-loading
machinery but never share comparison endpoints.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

# Closed set of output heads the app knows how to decode. A Literal (not a
# plain str) makes a typo like "sigmod" a type-check error instead of a
# silently-misbehaving registry entry.
OutputAdapter = Literal["softmax", "single_logit_sigmoid"]


class ModelTask(StrEnum):
    SENTIMENT = "sentiment"
    AI_TEXT_DETECTION = "ai_text_detection"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    task: ModelTask
    labels: tuple[str, ...]
    domain: str
    note: str
    local_path: str | None = None
    default: bool = False
    # How raw model output maps to canonical label scores. "softmax" covers
    # standard N-class heads; "single_logit_sigmoid" covers detectors like
    # desklib that emit ONE logit where sigmoid(logit) = P(ai).
    output_adapter: OutputAdapter = "softmax"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Sentiment models
    "twitter-roberta": ModelConfig(
        id="twitter-roberta",
        name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        task=ModelTask.SENTIMENT,
        labels=("negative", "neutral", "positive"),
        domain="social / short English text",
        note="Default sentiment model; used by analyze, batch, csv, and explain.",
        local_path="models/twitter-roberta-base-sentiment-latest",
        default=True,
    ),
    "distilbert-sst2": ModelConfig(
        id="distilbert-sst2",
        name="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        task=ModelTask.SENTIMENT,
        labels=("negative", "positive"),
        domain="general binary sentiment",
        note="Fast binary baseline; no neutral class, so compare label mismatch explicitly.",
    ),
    "finbert": ModelConfig(
        id="finbert",
        name="ProsusAI/finbert",
        task=ModelTask.SENTIMENT,
        labels=("positive", "negative", "neutral"),
        domain="financial text",
        note="Useful for finance/news sentences; misleading outside that domain.",
    ),
    "xlm-twitter": ModelConfig(
        id="xlm-twitter",
        name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        task=ModelTask.SENTIMENT,
        labels=("negative", "neutral", "positive"),
        domain="multilingual social text",
        note="Multilingual social-text model trained on tweets.",
    ),

    # AI text detector models
    "desklib-ai-detector": ModelConfig(
        id="desklib-ai-detector",
        name="desklib-ai-text-detector-v1.01",
        task=ModelTask.AI_TEXT_DETECTION,
        labels=("human", "ai"),
        domain="general AI-written text detection",
        note="Default detector. Custom DeBERTa-v3 arch (single sigmoid logit) — loads via DetectorModel, never AutoModelForSequenceClassification.",
        local_path="models/desklib-ai-text-detector-v1.01",
        default=True,
        output_adapter="single_logit_sigmoid",
    ),
    "fakespot-ai-detector": ModelConfig(
        id="fakespot-ai-detector",
        name="fakespot-roberta-base-ai-text-detection-v1",
        task=ModelTask.AI_TEXT_DETECTION,
        labels=("human", "ai"),
        domain="AI-generated review/text detection",
        note="RoBERTa-based detector. Use for detector comparison, not sentiment.",
        local_path="models/fakespot-roberta-base-ai-text-detection-v1",
    ),
    "oxidane-ai-detector": ModelConfig(
        id="oxidane-ai-detector",
        name="oxidane-tmr-ai-text-detector",
        task=ModelTask.AI_TEXT_DETECTION,
        labels=("human", "ai"),
        domain="general AI text detection",
        note="Local detector. Verify label order from config.json before trusting scores.",
        local_path="models/oxidane-tmr-ai-text-detector",
    ),
}


def models_for_task(task: ModelTask) -> dict[str, ModelConfig]:
    return {k: v for k, v in MODEL_REGISTRY.items() if v.task == task}


def get_default_model_id(task: ModelTask) -> str:
    for k, v in MODEL_REGISTRY.items():
        if v.task == task and v.default:
            return k
    # Fail loudly: a missing default is a registry bug. next() would surface
    # it as an opaque StopIteration deep inside a request handler.
    raise ValueError(f"No default model configured for task: {task}")


def get_model_config(model_id: str | None = None) -> ModelConfig:
    key = model_id or get_default_model_id(ModelTask.SENTIMENT)
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown model_id: {key}")


# Registry local_path entries are relative to the REPO ROOT (models/ is a
# sibling of backend/), not to whatever directory uvicorn was started from.
# This file lives at backend/app/model_registry.py → parents[2] is the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_source(config: ModelConfig) -> str:
    """Local weights directory when present, else the HF Hub model name.

    Why the fallback matters: models/ is a local-only, untracked directory.
    A fresh clone, CI, or a Docker build has no local weights — without this
    check, from_pretrained("models/...") dies with a path error instead of
    downloading from the Hub.
    """
    if config.local_path:
        local = _REPO_ROOT / config.local_path
        if local.exists():
            return str(local)
    return config.name
```

- [ ] **Step 2: Adapt SentimentModel + lazy cache with per-model locks**
<!-- ✅ VALIDATED 2026-07-01 (review fix #4): app.state.model_cache + model_locks + double-checked asyncio.Lock — review -->

Change `SentimentModel.__init__` to accept a `ModelConfig` (from `model_registry`) instead of using only `MODEL_NAME`. Keep `MODEL_NAME` as the default constant for backward compatibility in Tasks 1–7. **Labels come from the registry, not raw `config.id2label`:** HF DistilBERT returns `NEGATIVE`/`POSITIVE`; FinBERT label order differs from the registry tuple — zip softmax outputs with `list(config.labels)` for stable API keys.
<!-- ✅ VALIDATED 2026-07-01 (review fix #11): HF distilbert config id2label {'0':'NEGATIVE','1':'POSITIVE'} fetched live; registry uses lowercase -->

```python
from app.model_registry import (
    ModelConfig,
    ModelTask,
    get_default_model_id,
    get_model_config,
    resolve_model_source,
)

class SentimentModel:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MAX_TOKENS = 512

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or get_model_config(get_default_model_id(ModelTask.SENTIMENT))
        self.model_name = self._config.name
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        self.labels: list[str] = list(self._config.labels)

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Local weights dir if it exists, HF Hub name otherwise — never let a
        # missing local folder break a fresh clone or the Docker build.
        source = resolve_model_source(self._config)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = AutoModelForSequenceClassification.from_pretrained(source)
        self._model.to(self.device)
        self._model.eval()
        # Canonical label names from the registry — not config.id2label casing/order.
        self.labels = list(self._config.labels)
```

Do not load all models at startup: startup loads only the default model, and `get_or_load_model` lazily loads/cache-misses the selected model.

Initialize in lifespan:

```python
app.state.model_cache = {}
app.state.model_locks = {}
```

On cache miss, acquire a per-model lock before loading so two concurrent requests for the same unloaded model don't both download/load it. **Run blocking `load()` in a thread** so the event loop stays responsive while ~500MB weights download:
<!-- ✅ VALIDATED 2026-07-01 (review fix #4, #10): asyncio.Lock + asyncio.to_thread — Context7: blocking I/O off event loop -->

```python
import asyncio
from typing import Protocol


class BaseTextModel(Protocol):
    """What the cache and routes actually need from a model. SentimentModel
    satisfies this today; DetectorModel (Task 19) will too. The cache must
    not assume every model is a SentimentModel — detector checkpoints have
    different architectures and output heads."""

    labels: list[str]
    device: str | None
    is_loaded: bool

    def load(self) -> None: ...
    def predict(self, texts: list[str]) -> list[dict]: ...


def build_model(cfg: ModelConfig) -> BaseTextModel:
    if cfg.task == ModelTask.SENTIMENT:
        return SentimentModel(cfg)
    if cfg.task == ModelTask.AI_TEXT_DETECTION:
        # Task 19 replaces this with DetectorModel(cfg). Failing loudly now
        # beats silently mis-loading a custom-architecture detector.
        raise NotImplementedError("DetectorModel arrives in Task 19")
    raise ValueError(f"Unsupported model task: {cfg.task}")


async def get_or_load_model(app, model_id: str) -> BaseTextModel:
    if model_id in app.state.model_cache:
        return app.state.model_cache[model_id]
    if model_id not in app.state.model_locks:
        app.state.model_locks[model_id] = asyncio.Lock()
    async with app.state.model_locks[model_id]:
        if model_id not in app.state.model_cache:
            cfg = get_model_config(model_id)
            m = build_model(cfg)
            await asyncio.to_thread(m.load)
            app.state.model_cache[model_id] = m
        return app.state.model_cache[model_id]
```

- [ ] **Step 3: Add model endpoints**

Routes:

```text
GET /api/models?task=sentiment|ai_text_detection   # optional task filter
POST /api/analyze              # twitter-roberta only, strict 3-class (400 if ?model_id= other)
POST /api/analyze/batch        # twitter-roberta only, strict 3-class
POST /api/analyze/csv          # twitter-roberta only, strict 3-class
POST /api/explain              # twitter-roberta only, IG (400 if ?model_id= other)
POST /api/compare              # sentiment models only, dynamic multi-model, lazy-loaded
```

`GET /api/models` returns `{"models": [...]}` (wrap the list in a `models` key — the frontend `getModels()` type in Task 11 expects this exact shape) with registry metadata plus `loaded: true/false` for each model. Optional `?task=` filters to sentiment or AI-detector entries. Unknown `model_id` on compare returns 404 with a clear message. Sentiment registry models are **compare-only** for sentiment (not analyze/batch/csv/explain). AI detector models are **never** accepted on `/api/compare` (Task 19).
<!-- ✅ VALIDATED 2026-07-01 (review fix #9, #14): single clean API surface -->

Reject stray `model_id` on default endpoints (add to `routes.py` when wiring registry — same pattern as explain):

```python
DEFAULT_SENTIMENT_MODEL_ID = get_default_model_id(ModelTask.SENTIMENT)


def reject_non_default_model_id(model_id: str | None) -> None:
    if model_id and model_id != DEFAULT_SENTIMENT_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail="This endpoint uses the default twitter-roberta model only. Use /api/compare for other models.",
        )
```

Apply `reject_non_default_model_id(model_id)` at the top of analyze, batch, and csv handlers if a `model_id` query param is present.

- [ ] **Step 4: Verify**

Unit tests use fake model instances and assert lazy behavior: default sentiment model exists, optional models are not loaded until `/api/compare` is called, unknown compare IDs are rejected, `/api/models?task=sentiment` returns only sentiment registry entries, `/api/models?task=ai_text_detection` returns detector entries, and `/api/analyze?model_id=distilbert-sst2` returns 400.

Add to `backend/tests/test_model_registry.py` (pure unit test — no torch needed):

```python
from app.model_registry import ModelConfig, ModelTask, resolve_model_source


def test_resolve_model_source_falls_back_to_hub_name_when_local_path_missing():
    cfg = ModelConfig(
        id="x",
        name="hf/name",
        task=ModelTask.SENTIMENT,
        labels=("negative", "positive"),
        domain="test",
        note="test",
        local_path="models/does-not-exist",
    )
    assert resolve_model_source(cfg) == "hf/name"
```

Append to `backend/tests/test_model_integration.py`:

```python
@pytest.mark.integration
@pytest.mark.parametrize("model_id", list(models_for_task(ModelTask.SENTIMENT).keys()))
def test_registry_model_score_keys_match_config(model_id):
    from app.model import SentimentModel
    from app.model_registry import MODEL_REGISTRY, get_model_config

    cfg = get_model_config(model_id)
    m = SentimentModel(cfg)
    m.load()
    out = m.predict(["This is good."])[0]
    assert tuple(out["scores"].keys()) == cfg.labels
```
<!-- ✅ VALIDATED 2026-07-01 (review fix #11): registry labels canonical — protects DistilBERT/FinBERT key order -->

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: model registry with lazy model selection"
```

---

### Task 9: Model comparison API

**Files:**

- Modify: `backend/app/routes.py`, `backend/app/schemas.py`
- Test: `backend/tests/test_compare.py`

**Interfaces:**

- Produces: `POST /api/compare` accepting `{text, model_ids?}` and returning one row per **sentiment** model: `{model_id, name, domain, label, scores: dict[str, float], confidence, latency_ms, note}`. Reject any `model_id` with `task != ModelTask.SENTIMENT`.

- [ ] **Step 1: Add schemas**

Use `DynamicAnalyzeResponse` (not the strict 3-class `AnalyzeResponse`) so binary DistilBERT returns only `negative`/`positive` keys without faking a neutral score:
<!-- ✅ VALIDATED 2026-07-01 (review fix #1): registry distilbert-sst2 labels=("negative","positive") — Tavily/HF -->

```python
class CompareRequest(AnalyzeRequest):
    model_ids: list[str] | None = None


class CompareItem(DynamicAnalyzeResponse):
    model_id: str
    name: str
    domain: str
    confidence: float
    latency_ms: float
    note: str


class CompareResponse(BaseModel):
    results: list[CompareItem]
```

- [ ] **Step 2: Implement endpoint**

Use `time.perf_counter()` around each `predict([text])` call. **Do not default to all registry models** — first compare call would lazily download/load four models, which feels broken in a portfolio demo.
<!-- ✅ VALIDATED 2026-07-01 (review fix #2): default 2 models not 4 — review -->

```python
DEFAULT_COMPARE_MODELS = ["twitter-roberta", "distilbert-sst2"]
```

When `model_ids` is omitted, use `DEFAULT_COMPARE_MODELS`. This endpoint is intentionally plain: it teaches domain mismatch, label mismatch, confidence, and latency tradeoffs better than another chart.

- [ ] **Step 3: Verify**

Tests assert all requested models appear, `confidence == max(scores.values())`, latency is numeric/non-negative, and binary DistilBERT returns only `negative`/`positive` in `scores` (no `neutral` key). Response validation must pass for both 2-class and 3-class models.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: compare sentiment models on one input"
```

---

### Task 10: Sentiment evaluation harness + error analysis seed data

**Files:**

- Create: `evals/run_eval.py`, `evals/data/sentiment_eval.csv`, `evals/data/creative_text_eval.csv`, `evals/data/edge_cases.csv`, `evals/report.md`

**Interfaces:**

- Produces: CLI `python evals/run_eval.py --model-id twitter-roberta --data evals/data/sentiment_eval.csv --out evals/report.md [--allow-label-mismatch]` and JSON/Markdown metrics with accuracy, macro F1, per-class precision/recall/F1, confusion matrix, latency p50/p95, and wrong examples.
<!-- ✅ VALIDATED 2026-07-01 (review fix #12): dataset labels must be subset of model labels unless flag passed — e.g. DistilBERT cannot predict neutral -->

- [ ] **Step 1: Seed labeled eval files**

`evals/data/sentiment_eval.csv`:

```csv
id,text,true_label,category,notes
1,The battery life is incredible,positive,product_review,clear positive
2,"Not bad, not great",neutral,ambiguous,mixed phrase
3,I love the design but the app keeps crashing,neutral,mixed,multi-sentiment
4,"Yeah, amazing, another crash",negative,sarcasm,sarcastic negative
5,Our quarterly revenue outlook improved,positive,finance,domain-specific
6,This headline is emotionally neutral,neutral,formal,low sentiment
```

`evals/data/creative_text_eval.csv` should cover: product copy, social post, short-story sentence, sarcasm, mixed emotion, marketing headline, support complaint, neutral technical writing, finance/news sentence, and ambiguous phrase.

- [ ] **Step 2: Implement run_eval.py**

Install eval deps first: `pip install -r requirements.txt && pip install -r requirements-eval.txt` (scikit-learn is eval-only — not in CI's `requirements-dev.txt`).
<!-- ✅ VALIDATED 2026-07-01 (review fix #7): sklearn 1.9.0 PyPI confirmed; CI uses requirements-dev.txt without it -->

Use the registry + `SentimentModel` directly, not HTTP. Use `sklearn.metrics.classification_report(..., output_dict=True)` and `confusion_matrix`. Record p50/p95 latency with the standard library (`statistics.median`, `statistics.quantiles`).

**Label compatibility (required):** before scoring, collect unique `true_label` values from the CSV and compare to `model.labels`. If dataset labels are not a subset of model labels, exit with a clear error unless `--allow-label-mismatch` is passed — running DistilBERT on a CSV with `neutral` rows would produce misleading metrics.

```python
def validate_label_compatibility(
    dataset_labels: set[str], model_labels: set[str], allow_mismatch: bool
) -> None:
    if dataset_labels.issubset(model_labels):
        return
    extra = sorted(dataset_labels - model_labels)
    if allow_mismatch:
        print(f"WARNING: dataset labels {extra} are not in model labels — metrics may mislead")
        return
    raise SystemExit(
        f"Dataset labels {sorted(dataset_labels)} are not a subset of model labels "
        f"{sorted(model_labels)}. Unpredictable classes: {extra}. "
        f"Use --allow-label-mismatch to run anyway."
    )
```
<!-- ✅ VALIDATED 2026-07-01 (review fix #12): honest eval when binary model meets 3-class CSV -->

Output shape:

```json
{
  "model_id": "twitter-roberta",
  "accuracy": 0.82,
  "macro_f1": 0.79,
  "latency_p50_ms": 74,
  "latency_p95_ms": 141,
  "confusion_matrix": [[...]],
  "wrong_examples": [...]
}
```

- [ ] **Step 3: Write report.md**

Include a short, honest section:

```text
Sentiment models are not creativity judges. This project tests emotional polarity,
confidence, model disagreement, and explanation quality across writing styles.
```

Then list the top failure modes: sarcasm, mixed sentiment, long formal text, finance/domain mismatch, and missing context.

- [ ] **Step 4: Verify**

Run one quick eval on the small CSV, inspect `evals/report.md`, and confirm the wrong-example table includes text/category/true/predicted/confidence.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: evaluation harness with error analysis report"
```

---

### Task 11: Frontend scaffold + typed API client

**Files:**

- Create: `frontend/` (Vite react-ts template), `frontend/src/api.ts`, `frontend/src/test-setup.ts`
- Modify: `frontend/vite.config.ts`, `frontend/src/index.css`, `frontend/package.json`
- Test: `frontend/src/api.test.ts`

**Interfaces:**

- Produces: `api.ts` exports — types `Scores` (strict for default analyze/batch), `AnalyzeResult`, `TokenAttribution`, `ExplainResult`, `BatchItem`, `BatchResult`, `ModelInfo`, `ModelSummary`, `CompareItem` (dynamic `scores: Record<string, number>`), `AiDetectItem`, `AiDetectResponse`, `AiDetectCompareResponse`; functions `analyze(text)`, `explainText(text)`, `analyzeCsv(file)`, `getModelInfo()`, `getModels()`, `compareModels(text, modelIds?)`, `detectAiText(text)`, `compareAiDetectors(text, model_ids?)`. All UI tasks import from here; components never call `fetch` directly.
<!-- ✅ VALIDATED 2026-07-01 (review fix #1): CompareItem uses DynamicScores not Scores — mirrors backend DynamicAnalyzeResponse -->

- [ ] **Step 1: Scaffold**

```bash
cd ~/Projects/active/sentiment-scope
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install react@19.2.7 react-dom@19.2.7 recharts@3.9.1 tailwindcss@4.3.2 @tailwindcss/vite@4.3.2
npm install -D vite@8.1.2 @vitejs/plugin-react@6.0.3 typescript@6.0.3 vitest@4.1.9 @testing-library/react@16.3.2 @testing-library/jest-dom@6.9.1 @testing-library/user-event@14.6.1 jsdom@29.1.1
```

`frontend/vite.config.ts` (replace):

```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev-server proxy: the UI calls relative /api/... paths and Vite
    // forwards them to FastAPI. Same-origin from the browser's view, so no
    // CORS gymnastics; in Docker, nginx plays this exact role.
    proxy: { "/api": "http://localhost:8000" },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    globals: true,
  },
});
```

Replace `frontend/src/index.css` entirely with:

```css
@import "tailwindcss";
```

`frontend/src/test-setup.ts`:

```ts
import "@testing-library/jest-dom";
```

Add to `frontend/package.json` scripts: `"test": "vitest"`.
Delete `frontend/src/App.css` and `frontend/src/assets/react.svg` (template cruft; App.tsx gets replaced in Task 12 — for now remove the `import "./App.css"` line from it so the build stays green).

- [ ] **Step 2: Write the failing test**

`frontend/src/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { analyze } from "./api";

describe("api client", () => {
  afterEach(() => vi.restoreAllMocks());

  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ label: "positive", scores: { negative: 0.1, neutral: 0.1, positive: 0.8 } })),
    ));
    const result = await analyze("great stuff");
    expect(result.label).toBe("positive");
  });

  it("throws the server's detail message on error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Model is not loaded" }), { status: 503 }),
    ));
    await expect(analyze("hi")).rejects.toThrow("Model is not loaded");
  });
});
```

- [ ] **Step 3: Verify failure** — `npm test -- --run` — Expected: FAIL, `./api` not found.

- [ ] **Step 4: Implement api.ts**

`frontend/src/api.ts`:

```ts
/**
 * Typed API client — the single place the frontend talks to the backend.
 * Components import these functions and never touch fetch() directly, so
 * error handling and response typing live in exactly one file.
 *
 * All paths are relative (/api/...): the Vite dev server proxies them to
 * FastAPI locally, and nginx does the same inside Docker. The UI never
 * needs to know where the backend lives.
 */

/** Strict 3-class scores for default-model analyze/batch endpoints. */
export interface Scores {
  negative: number;
  neutral: number;
  positive: number;
}

export interface AnalyzeResult {
  label: string;
  scores: Scores;
}

/** Dynamic scores for compare — keys vary by model (binary vs 3-class). */
export type DynamicScores = Record<string, number>;

export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface ExplainResult extends AnalyzeResult {
  tokens: TokenAttribution[];
}

export interface BatchItem extends AnalyzeResult {
  text: string;
}

export interface BatchResult {
  results: BatchItem[];
  aggregates: { counts: Record<string, number>; mean_scores: Scores };
}

export interface ModelInfo {
  name: string;
  labels: string[];
  max_tokens: number;
  device: string;
  description: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  labels: string[];
  domain: string;
  note: string;
  default: boolean;
  loaded: boolean;
}

export interface CompareItem {
  model_id: string;
  name: string;
  domain: string;
  label: string;
  scores: DynamicScores;
  confidence: number;
  latency_ms: number;
  note: string;
}

export type AiDetectionScores = Record<string, number>;

export interface AiDetectItem {
  model_id: string;
  name: string;
  domain: string;
  label: string;
  scores: AiDetectionScores;
  confidence: number;
  latency_ms: number;
  note: string;
}

export interface AiDetectResponse {
  result: AiDetectItem;
  warning: string;
}

export interface AiDetectCompareResponse {
  results: AiDetectItem[];
  disagreement: boolean;
  warning: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    // FastAPI puts human-readable errors in { detail } — surface that
    // instead of a bare status code whenever it's available.
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const analyze = (text: string) => request<AnalyzeResult>("/api/analyze", postJson({ text }));

export const explainText = (text: string) => request<ExplainResult>("/api/explain", postJson({ text }));

export const analyzeCsv = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  // Note: no Content-Type header — the browser sets multipart boundaries itself.
  return request<BatchResult>("/api/analyze/csv", { method: "POST", body: form });
};

export const getModelInfo = () => request<ModelInfo>("/api/model");

export const getModels = (task?: "sentiment" | "ai_text_detection") =>
  request<{ models: ModelSummary[] }>(
    task ? `/api/models?task=${task}` : "/api/models",
  );

export const compareModels = (text: string, model_ids?: string[]) =>
  request<{ results: CompareItem[] }>("/api/compare", postJson({ text, model_ids }));

export const detectAiText = (text: string) =>
  request<AiDetectResponse>("/api/ai-detect", postJson({ text }));

export const compareAiDetectors = (text: string, model_ids?: string[]) =>
  request<AiDetectCompareResponse>("/api/ai-detect/compare", postJson({ text, model_ids }));
```

- [ ] **Step 5: Verify** — `npm test -- --run && npm run lint && npm run build` — Expected: 2 tests pass, lint + build clean.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/active/sentiment-scope && git add -A && git commit -m "feat: frontend scaffold with typed API client"
```

---

### Task 12: Analyze tab (AnalyzeForm + ConfidenceBars)

**Files:**

- Create: `frontend/src/components/ConfidenceBars.tsx`, `frontend/src/components/AnalyzeForm.tsx`
- Modify: `frontend/src/App.tsx` (render AnalyzeForm for now; tabs arrive in Task 15)
- Test: `frontend/src/components/ConfidenceBars.test.tsx`

**Interfaces:**

- Consumes: `analyze` from `../api`.
- Produces: `<ConfidenceBars scores={Scores | DynamicScores} />` (renders `Object.entries(scores)` — works for 2-class and 3-class); `<AnalyzeForm />` (self-contained: textarea + Analyze/Explain buttons; renders ConfidenceBars and, after Task 13, TokenHeatmap).
<!-- ✅ VALIDATED 2026-07-01 (review fix #1): Object.entries not fixed negative/neutral/positive keys — review -->

- [ ] **Step 1: Write the failing test**

`frontend/src/components/ConfidenceBars.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import ConfidenceBars from "./ConfidenceBars";

it("renders one bar per class with percentages", () => {
  render(<ConfidenceBars scores={{ negative: 0.05, neutral: 0.15, positive: 0.8 }} />);
  expect(screen.getByText("negative")).toBeInTheDocument();
  expect(screen.getByText("neutral")).toBeInTheDocument();
  expect(screen.getByText("positive")).toBeInTheDocument();
  expect(screen.getByText("80.0%")).toBeInTheDocument();
});
```

- [ ] **Step 2: Verify failure** — `npm test -- --run` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

`frontend/src/components/ConfidenceBars.tsx`:

```tsx
import type { DynamicScores, Scores } from "../api";

/**
 * Horizontal bars for class probabilities. Renders Object.entries(scores)
 * so it works for binary (DistilBERT) and 3-class models. Showing the full softmax distribution (not just the winner) is deliberate: "positive 51%"
 * and "positive 98%" are very different answers, and hiding that nuance is
 * how ML demos mislead people.
 */

const BAR_COLOR: Record<string, string> = {
  negative: "bg-red-500",
  neutral: "bg-slate-400",
  positive: "bg-emerald-500",
};

const fallbackColor = "bg-indigo-400";

export default function ConfidenceBars({ scores }: { scores: Scores | DynamicScores }) {
  return (
    <div className="space-y-2">
      {Object.entries(scores).map(([label, value]) => (
        <div key={label} className="flex items-center gap-3">
          <span className="w-20 text-sm capitalize text-slate-600">{label}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-slate-200">
            <div
              className={`h-full ${BAR_COLOR[label] ?? fallbackColor} transition-all`}
              style={{ width: `${value * 100}%` }}
            />
          </div>
          <span className="w-14 text-right text-sm tabular-nums text-slate-600">
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
```

`frontend/src/components/AnalyzeForm.tsx`:

```tsx
import { useState } from "react";
import { analyze, explainText } from "../api";
import type { AnalyzeResult, ExplainResult } from "../api";
import ConfidenceBars from "./ConfidenceBars";

const LABEL_BADGE: Record<string, string> = {
  negative: "bg-red-100 text-red-700",
  neutral: "bg-slate-100 text-slate-700",
  positive: "bg-emerald-100 text-emerald-700",
};

export default function AnalyzeForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Both buttons share one submit path; `withExplain` decides which endpoint.
  // Explain is opt-in because Integrated Gradients costs ~50 forward passes.
  const run = async (withExplain: boolean) => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      if (withExplain) {
        const res = await explainText(text);
        setResult(res);
        setExplanation(res);
      } else {
        setResult(await analyze(text));
      }
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <textarea
        className="w-full rounded-lg border border-slate-300 p-3 focus:border-indigo-500 focus:outline-none"
        rows={4}
        maxLength={2000}
        placeholder="Type or paste text to analyze… e.g. 'The battery life on this phone is incredible'"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex gap-3">
        <button
          className="rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
          disabled={loading || !text.trim()}
          onClick={() => run(false)}
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
        <button
          className="rounded-lg border border-indigo-600 px-4 py-2 font-medium text-indigo-600 hover:bg-indigo-50 disabled:opacity-50"
          disabled={loading || !text.trim()}
          onClick={() => run(true)}
          title="Slower: runs Integrated Gradients to show which words drove the prediction"
        >
          Analyze + Explain
        </button>
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <div className="space-y-4 rounded-lg border border-slate-200 p-4">
          <span
            className={`inline-block rounded-full px-3 py-1 text-sm font-semibold capitalize ${LABEL_BADGE[result.label] ?? ""}`}
          >
            {result.label}
          </span>
          <ConfidenceBars scores={result.scores} />
          {/* TokenHeatmap renders here after Task 13 */}
          {explanation && <div data-explanation-slot>{null}</div>}
        </div>
      )}
    </div>
  );
}
```

`frontend/src/App.tsx` (replace template contents):

```tsx
import AnalyzeForm from "./components/AnalyzeForm";

export default function App() {
  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="mb-6 text-2xl font-bold text-slate-800">SentimentScope</h1>
      <AnalyzeForm />
    </div>
  );
}
```

Also update `frontend/src/main.tsx` if the template referenced `App.css` (it imports only `index.css` by default — leave as is).

- [ ] **Step 4: Verify**

`npm test -- --run && npm run lint && npm run build` — Expected: pass.
Manual smoke test: `ca ai && cd backend && uvicorn app.main:app --port 8000` in one shell, `npm run dev` in another, open <http://localhost:5173>, analyze "I love this" → positive badge + bars.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: analyze tab with confidence bars"
```

---

### Task 13: Token attribution heatmap

**Files:**

- Create: `frontend/src/components/TokenHeatmap.tsx`
- Modify: `frontend/src/components/AnalyzeForm.tsx` (render heatmap)
- Test: `frontend/src/components/TokenHeatmap.test.tsx`

**Interfaces:**

- Consumes: `TokenAttribution` type from `../api`.
- Produces: `<TokenHeatmap tokens={TokenAttribution[]} />`.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/TokenHeatmap.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import TokenHeatmap from "./TokenHeatmap";

it("colors positive attributions green and negative red", () => {
  render(
    <TokenHeatmap
      tokens={[
        { token: " love", attribution: 0.9 },
        { token: " not", attribution: -0.5 },
      ]}
    />,
  );
  const love = screen.getByText("love");
  const not = screen.getByText("not");
  expect(love.style.backgroundColor).toContain("16, 185, 129"); // emerald
  expect(not.style.backgroundColor).toContain("239, 68, 68"); // red
});
```

- [ ] **Step 2: Verify failure** — `npm test -- --run` — Expected: module not found.

- [ ] **Step 3: Implement**

`frontend/src/components/TokenHeatmap.tsx`:

```tsx
import type { TokenAttribution } from "../api";

/**
 * Renders the input text token-by-token, tinted by Integrated Gradients
 * attribution: green = pushed the model toward its prediction, red = pushed
 * against it, intensity = relative magnitude. Magnitudes are scaled against
 * the largest attribution in THIS sentence — colors show relative influence
 * within one input and aren't comparable across inputs.
 */
export default function TokenHeatmap({ tokens }: { tokens: TokenAttribution[] }) {
  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.attribution)), 1e-6);

  return (
    <p className="leading-8">
      {tokens.map((t, i) => {
        const strength = Math.abs(t.attribution) / maxAbs;
        const color =
          t.attribution >= 0
            ? `rgba(16, 185, 129, ${(0.15 + 0.7 * strength).toFixed(2)})`
            : `rgba(239, 68, 68, ${(0.15 + 0.7 * strength).toFixed(2)})`;
        // Tokens keep their leading-space marker from the backend; trim for
        // display but re-add spacing via margin so words don't run together.
        const display = t.token.trimStart();
        const leadingSpace = t.token.startsWith(" ");
        return (
          <span
            key={i}
            className={`rounded px-0.5 ${leadingSpace ? "ml-1" : ""}`}
            style={{ backgroundColor: color }}
            title={`attribution: ${t.attribution.toFixed(3)}`}
          >
            {display}
          </span>
        );
      })}
    </p>
  );
}
```

In `frontend/src/components/AnalyzeForm.tsx`: add `import TokenHeatmap from "./TokenHeatmap";` and replace the placeholder line `{explanation && <div data-explanation-slot>{null}</div>}` with:

```tsx
          {explanation && (
            <div className="space-y-1 border-t border-slate-200 pt-4">
              <p className="text-sm font-medium text-slate-600">
                Which words drove this prediction (Integrated Gradients):
              </p>
              <TokenHeatmap tokens={explanation.tokens} />
            </div>
          )}
```

- [ ] **Step 4: Verify**

`npm test -- --run && npm run lint && npm run build` — pass.
Manual: "Analyze + Explain" on "I absolutely love this phone" → "love" glows green strongest.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: token attribution heatmap for explanations"
```

---

### Task 14: Batch tab (CSV upload + aggregate charts)

**Files:**

- Create: `frontend/src/components/BatchUpload.tsx`, `frontend/src/components/AggregateCharts.tsx`

**Interfaces:**

- Consumes: `analyzeCsv`, types `BatchResult` from `../api`.
- Produces: `<BatchUpload />` (self-contained tab body), `<AggregateCharts aggregates={BatchResult["aggregates"]} />`.

- [ ] **Step 1: Implement AggregateCharts**

`frontend/src/components/AggregateCharts.tsx`:

```tsx
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { BatchResult } from "../api";

const LABEL_COLOR: Record<string, string> = {
  negative: "#ef4444",
  neutral: "#94a3b8",
  positive: "#10b981",
};

/**
 * Two views of the same batch: label counts (how many rows landed in each
 * class) and mean softmax scores (how confident the model was on average).
 * Both matter — 100 barely-positive rows and 100 emphatic ones have the
 * same counts but very different mean scores.
 */
export default function AggregateCharts({ aggregates }: { aggregates: BatchResult["aggregates"] }) {
  const countData = Object.entries(aggregates.counts).map(([label, count]) => ({ label, count }));
  const meanData = Object.entries(aggregates.mean_scores).map(([label, mean]) => ({ label, mean }));

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-600">Sentiment counts</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={countData}>
            <XAxis dataKey="label" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count">
              {countData.map((d) => (
                <Cell key={d.label} fill={LABEL_COLOR[d.label]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-600">Mean confidence per class</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={meanData}>
            <XAxis dataKey="label" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="mean">
              {meanData.map((d) => (
                <Cell key={d.label} fill={LABEL_COLOR[d.label]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Implement BatchUpload**

`frontend/src/components/BatchUpload.tsx`:

```tsx
import { useRef, useState } from "react";
import { analyzeCsv } from "../api";
import type { BatchResult } from "../api";
import AggregateCharts from "./AggregateCharts";

const LABEL_TEXT: Record<string, string> = {
  negative: "text-red-600",
  neutral: "text-slate-500",
  positive: "text-emerald-600",
};

export default function BatchUpload() {
  const [result, setResult] = useState<BatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      // The file goes straight to the backend, which owns CSV parsing and
      // validation — one source of truth for what a valid upload is.
      setResult(await analyzeCsv(file));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border-2 border-dashed border-slate-300 p-8 text-center">
        <p className="mb-3 text-slate-600">
          Upload a CSV with a <code className="rounded bg-slate-100 px-1">text</code> column
          (max 500 rows)
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="mx-auto block text-sm"
          onChange={(e) => onFile(e.target.files?.[0])}
          disabled={loading}
        />
        {loading && <p className="mt-3 text-indigo-600">Analyzing batch…</p>}
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <>
          <AggregateCharts aggregates={result.aggregates} />
          <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-50">
                <tr>
                  <th className="p-2">Text</th>
                  <th className="p-2">Label</th>
                  <th className="p-2 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    <td className="max-w-md truncate p-2" title={r.text}>{r.text}</td>
                    <td className={`p-2 font-medium capitalize ${LABEL_TEXT[r.label] ?? ""}`}>{r.label}</td>
                    <td className="p-2 text-right tabular-nums">
                      {(Math.max(...Object.values(r.scores)) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify**

`npm test -- --run && npm run lint && npm run build` — pass (component is exercised manually + via Task 15 nav; charts are visual).
Manual: create `/tmp/sample.csv` with a `text` column of ~5 mixed-sentiment rows, upload on the Batch tab (after Task 15 wires it in — or temporarily render `<BatchUpload />` in App to check now, then revert).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: batch CSV upload with aggregate charts and results table"
```

---

### Task 15: Compare Sentiment tab, How-it-works page, model info footer

**Files:**

- Create: `frontend/src/components/CompareModels.tsx`, `frontend/src/components/HowItWorks.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**

- Consumes: `AnalyzeForm`, `BatchUpload`, `compareModels`, `getModels`, `getModelInfo`/`ModelInfo` from `../api`.
- Produces: `App` with four tabs in Phase 1: Analyze · Batch · Compare Sentiment · How it works. Task 21 adds the AI Detector tab and renames this tab from "Compare Models" to "Compare Sentiment".

Default compare uses `twitter-roberta` + `distilbert-sst2` only. Optional models (`finbert`, `xlm-twitter`) are off by default with a checkbox and a note that first use may take longer because the model is loaded lazily.
<!-- ✅ VALIDATED 2026-07-01 (review fix #2, #3): mirrors backend DEFAULT_COMPARE_MODELS + lazy-load UX — review -->

- [ ] **Step 1: Implement CompareModels**

`frontend/src/components/CompareModels.tsx`:

```tsx
import { useEffect, useState } from "react";
import { compareModels, getModels } from "../api";
import type { CompareItem, ModelSummary } from "../api";

const DEFAULT_COMPARE = ["twitter-roberta", "distilbert-sst2"];

export default function CompareModels() {
  const [text, setText] = useState("Our quarterly revenue outlook improved");
  const [registry, setRegistry] = useState<ModelSummary[]>([]);
  const [selected, setSelected] = useState<string[]>(DEFAULT_COMPARE);
  const [rows, setRows] = useState<CompareItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModels("sentiment")
      .then((r) => setRegistry(r.models))
      .catch(() => setRegistry([]));
  }, []);

  const toggle = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const run = async () => {
    if (!text.trim() || selected.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      setRows((await compareModels(text, selected)).results);
    } catch (e) {
      setRows([]);
      setError(e instanceof Error ? e.message : "Compare failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <textarea
        className="w-full rounded-lg border border-slate-300 p-3 focus:border-indigo-500 focus:outline-none"
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-slate-600">Models to compare</legend>
        {registry.map((m) => (
          <label key={m.id} className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={selected.includes(m.id)}
              onChange={() => toggle(m.id)}
            />
            <span className="font-medium">{m.id}</span>
            <span className="text-slate-500">— {m.domain}</span>
            {!m.loaded && !m.default && (
              <span className="text-xs text-amber-600">(first use may take longer — lazy load)</span>
            )}
          </label>
        ))}
      </fieldset>
      <button
        className="rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        disabled={loading || !text.trim() || selected.length === 0}
        onClick={run}
      >
        {loading ? "Comparing…" : "Compare models"}
      </button>
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      {rows.length > 0 && (
        <div className="overflow-auto rounded-lg border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="p-2">Model</th>
                <th className="p-2">Domain</th>
                <th className="p-2">Prediction</th>
                <th className="p-2 text-right">Confidence</th>
                <th className="p-2 text-right">Latency</th>
                <th className="p-2">Notes</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model_id} className="border-t border-slate-100">
                  <td className="p-2 font-medium">{r.model_id}</td>
                  <td className="p-2">{r.domain}</td>
                  <td className="p-2 capitalize">{r.label}</td>
                  <td className="p-2 text-right tabular-nums">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="p-2 text-right tabular-nums">{r.latency_ms.toFixed(0)} ms</td>
                  <td className="p-2 text-slate-500">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Implement HowItWorks**

`frontend/src/components/HowItWorks.tsx`:

```tsx
import { useEffect, useState } from "react";
import { getModelInfo } from "../api";
import type { ModelInfo } from "../api";

/**
 * The educational heart of the app: a plain-language walkthrough of what
 * happens between "user types a sentence" and "the UI shows 87% positive".
 */
export default function HowItWorks() {
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  return (
    <article className="prose prose-slate max-w-none space-y-6 text-slate-700">
      <section>
        <h2 className="text-lg font-semibold text-slate-800">1. Tokenization</h2>
        <p>
          Neural networks can't read text — they read numbers. A byte-pair-encoding (BPE)
          tokenizer splits your sentence into subword pieces ("incredible" might become
          "incred" + "ible") and maps each piece to an integer ID from a ~50k-entry
          vocabulary. Rare words split into more pieces; common words stay whole. This is
          why the explanation view highlights sub-word chunks rather than whole words.
        </p>
      </section>
      <section>
        <h2 className="text-lg font-semibold text-slate-800">2. The transformer encoder</h2>
        <p>
          Those IDs pass through RoBERTa — 12 layers of self-attention. Each layer lets
          every token "look at" every other token and update its representation based on
          context: the "bank" in "river bank" and "bank account" ends up with different
          vectors. After 12 rounds of this, the model has a contextual summary of the whole
          sentence.
        </p>
      </section>
      <section>
        <h2 className="text-lg font-semibold text-slate-800">3. Classification head + softmax</h2>
        <p>
          A small linear layer maps that summary to three raw scores (logits), one per
          class. Softmax exponentiates and normalizes them into probabilities that sum
          to 1 — the confidence bars you see on the Analyze tab. High entropy (three
          similar bars) means the model is genuinely unsure.
        </p>
      </section>
      <section>
        <h2 className="text-lg font-semibold text-slate-800">4. Explainability: Integrated Gradients</h2>
        <p>
          To answer "which words made it say that?", we use Integrated Gradients: start
          from an empty baseline sentence (all padding tokens), interpolate step-by-step
          toward the real input in embedding space, and accumulate the gradients of the
          predicted class along the way. Each token gets a share of the credit — green
          tokens pushed the model toward its answer, red pushed away.
        </p>
      </section>
      <section>
        <h2 className="text-lg font-semibold text-slate-800">Honest limitations</h2>
        <ul className="list-disc pl-5">
          <li>The model was trained on tweets — long or formal text is out-of-domain.</li>
          <li>Inputs are truncated to 512 tokens; anything beyond is invisible to the model.</li>
          <li>English only; sarcasm and irony remain hard.</li>
          <li>IG is an approximation (50 integration steps), not a ground-truth explanation.</li>
        </ul>
      </section>
      {info && (
        <footer className="rounded-lg bg-slate-50 p-4 text-sm text-slate-500">
          Model: <code>{info.name}</code> · labels: {info.labels.join(" / ")} · max{" "}
          {info.max_tokens} tokens · running on <strong>{info.device}</strong>.{" "}
          {info.description}
        </footer>
      )}
    </article>
  );
}
```

- [ ] **Step 3: Wire tabs in App.tsx**

Replace `frontend/src/App.tsx`:

```tsx
import { useState } from "react";
import AnalyzeForm from "./components/AnalyzeForm";
import BatchUpload from "./components/BatchUpload";
import CompareModels from "./components/CompareModels";
import HowItWorks from "./components/HowItWorks";

const TABS = ["Analyze", "Batch", "Compare Models", "How it works"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Analyze");

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-3xl p-6">
        <header className="mb-6">
          <h1 className="text-2xl font-bold text-slate-800">SentimentScope</h1>
          <p className="text-sm text-slate-500">
            Transformer sentiment analysis with explainability — RoBERTa served locally via FastAPI.
          </p>
        </header>
        <nav className="mb-6 flex gap-2">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-lg px-4 py-2 text-sm font-medium ${
                tab === t
                  ? "bg-indigo-600 text-white"
                  : "bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
        <main className="rounded-xl bg-white p-6 shadow-sm">
          {tab === "Analyze" && <AnalyzeForm />}
          {tab === "Batch" && <BatchUpload />}
          {tab === "Compare Models" && <CompareModels />}
          {tab === "How it works" && <HowItWorks />}
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify**

`npm test -- --run && npm run lint && npm run build` — pass.
Manual: all four tabs render; Compare Models shows domain/confidence/latency disagreement; Batch tab accepts the sample CSV; How-it-works shows the model footer when the backend is up.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: model comparison tab and educational how-it-works page"
```

---

### Task 16: Docker

**Files:**

- Create: `backend/Dockerfile`, `backend/.dockerignore`, `frontend/Dockerfile`, `frontend/nginx.conf`, `frontend/.dockerignore`, `docker-compose.yml`

**Interfaces:**

- Produces: `docker compose up --build` → app at <http://localhost:8080>, nginx proxying `/api/` to the backend container; HF weights cached in the `hf-cache` volume so rebuilds don't re-download.

- [ ] **Step 1: Backend image**

`backend/.dockerignore`:

```
__pycache__
*.pyc
.pytest_cache
.ruff_cache
tests
```

`backend/Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# CPU-only torch wheel: ~10x smaller than the default CUDA build, and there
# is no GPU inside this container anyway. requirements-docker.txt excludes
# torch so we install it once from the CPU index, then the rest normally.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Frontend image + nginx**

`frontend/.dockerignore`:

```
node_modules
dist
```

`frontend/nginx.conf`:

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;

    # Same-origin API: the SPA calls /api/..., nginx forwards to the backend
    # container. Mirrors the Vite dev proxy — the frontend build is identical
    # in both environments.
    location /api/ {
        proxy_pass http://backend:8000;
    }

    # SPA fallback: client-side routes all serve index.html.
    location / {
        try_files $uri /index.html;
    }
}
```

`frontend/Dockerfile`:

```dockerfile
# Stage 1: build the static bundle (Node 24 LTS — stable for CI/showcase repos)
FROM node:24-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve it with nginx (no node runtime in the final image)
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
```

- [ ] **Step 3: Compose**

`docker-compose.yml` (repo root):

```yaml
services:
  backend:
    build: ./backend
    volumes:
      # Persist HuggingFace downloads across container rebuilds — the model
      # is ~500MB and only needs to be fetched once.
      - hf-cache:/root/.cache/huggingface
  frontend:
    build: ./frontend
    ports:
      - "8080:80"
    depends_on:
      - backend

volumes:
  hf-cache:
```

- [ ] **Step 4: Verify**

Run: `cd ~/Projects/active/sentiment-scope && docker compose up --build -d`
Wait for backend model load (first run downloads weights): `docker compose logs -f backend` until "Application startup complete".
Then: `curl -s http://localhost:8080/api/health` → `{"status":"ok","model_loaded":true,"device":"cpu"}`; open <http://localhost:8080> and analyze a sentence.
Teardown: `docker compose down`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: dockerized deployment with nginx proxy and weight cache"
```

---

### Task 17: GitHub Actions CI

**Files:**

- Create: `.github/workflows/ci.yml`

**Interfaces:**

- Consumes: `backend/requirements-dev.txt` (no torch — see Global Constraints), frontend npm scripts (`lint`, `test`, `build`).

- [ ] **Step 1: Write the workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip
      # requirements-dev.txt deliberately excludes torch/transformers/captum/sklearn:
      # unit tests mock the model, so CI stays fast and light. Integration
      # tests (pytest -m integration) run locally where the weights live.
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: pytest -v

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm test -- --run
      - run: npm run build
```

- [ ] **Step 2: Verify locally (CI dry-run)**

Backend, in a throwaway venv to prove torch isn't needed:

```bash
cd ~/Projects/active/sentiment-scope/backend
python3 -m venv /tmp/ci-venv && /tmp/ci-venv/bin/pip install -q -r requirements-dev.txt
/tmp/ci-venv/bin/python -m ruff check . && /tmp/ci-venv/bin/python -m pytest -v
rm -rf /tmp/ci-venv
```

Expected: all unit tests pass with no torch installed. Frontend: `npm run lint && npm test -- --run && npm run build`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "ci: lint, test, and build pipelines for backend and frontend"
```

---

### Task 16A: Free public deployment — Hugging Face Spaces (Docker SDK)

**Purpose:** Actually serve the app to users for free. HF Spaces is the only free tier that fits a torch + ~500MB-weights backend (free CPU Space: 2 vCPU / 16 GB RAM / 50 GB ephemeral disk). Render's free tier is 512 MB RAM (torch won't fit); Fly.io/Railway no longer have usable free tiers. Verified 2026-07-01.

**Files:**

- Create: `Dockerfile.spaces` (repo root — single image: frontend build + FastAPI serving both static and API)
- Create: `SPACE_README.md` (becomes the Space's `README.md` with HF front-matter)
- Modify: `backend/app/main.py` (conditional static mount), `backend/requirements-docker.txt` (add `slowapi`)

**Why a separate Dockerfile:** docker-compose (nginx + backend as two containers) cannot run on Spaces — a Space is exactly one container. Instead FastAPI serves the built SPA itself via `StaticFiles(html=True)`. Educational angle worth a comment in the code: same app, three serving topologies (Vite proxy in dev, nginx in compose, FastAPI static in Spaces) — the frontend never changes because it only ever calls relative `/api/...` paths.

**Spaces platform constraints (each one bites if skipped):**

- Container must listen on **port 7860** (or set `app_port` in the Space README front-matter).
- The container runs as a **non-root user** — `/root/.cache` is not writable. Set `ENV HF_HOME=/tmp/hf` (or create a uid-1000 user with a writable home) or model download dies with `PermissionError`.
- Ephemeral disk resets on every restart, and free Spaces **sleep after ~48h of inactivity** — so **bake the default model weights into the image at build time** (`RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; ..."`) or every cold start re-downloads 500 MB before the health check passes.

**Public-abuse guards (required — this is a real public endpoint, not a demo on localhost):**

- Rate limiting via `slowapi` (already the minimal standard for FastAPI): e.g. `20/minute` on `/api/analyze`, `5/minute` on `/api/explain` and `/api/compare` (IG costs ~50 forward passes; compare lazy-loads models). Enable only when `PUBLIC_DEPLOY=1` so local dev and tests are unaffected.
- `ENABLED_MODELS` env allowlist consumed by the registry: the public Space ships with `twitter-roberta,distilbert-sst2` (and detectors if desired) so anonymous users can't lazy-load every model and balloon RAM. Requests for disabled models get a clear 403 explaining the public deployment limits.

- [ ] **Step 1: Conditional static mount in `main.py`**

```python
# After include_router(router). Mount order matters: /api routes are matched
# first because the router is registered before the static mount at "/".
# STATIC_DIR is only set in the Spaces image — dev and compose don't use this.
static_dir = os.getenv("STATIC_DIR")
if static_dir:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="spa")
```

Unit test: with `STATIC_DIR` pointing at a tmp dir containing an `index.html`, `GET /` returns it and `GET /api/health` still hits the API.

- [ ] **Step 2: Wire the public-abuse guards**

`backend/requirements-docker.txt` gains:

```text
slowapi==0.1.10
```

Keep slowapi OUT of `requirements-dev.txt`: wire it inside the `PUBLIC_DEPLOY` branch in `main.py` with a lazy import (same discipline as torch), so unit tests and CI never import it. Use the global `SlowAPIMiddleware` default limit rather than per-route `@limiter.limit` decorators — decorators would force the limiter (and therefore the slowapi import) to exist when `routes.py` is imported, which breaks the no-slowapi CI environment:

```python
# main.py, after app creation
if os.getenv("PUBLIC_DEPLOY") == "1":
    # Lazy import: slowapi is only installed in the deployment image —
    # dev and CI never take this branch.
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    # One global per-IP budget. 30/min covers real interactive usage;
    # /api/explain (~50 forward passes per call) is what this protects.
    limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
```

The model allowlist is pure stdlib and always wired — setting `ENABLED_MODELS` IS the opt-in (the public image sets it; dev leaves it unset), so there is no second flag to forget. Read the env per call, not at import, so tests can monkeypatch it:

```python
# routes.py
def enabled_model_ids() -> set[str] | None:
    """Parse the ENABLED_MODELS allowlist. None = no restriction (dev)."""
    raw = os.getenv("ENABLED_MODELS")
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def reject_disabled_model(model_id: str) -> None:
    enabled = enabled_model_ids()
    if enabled is not None and model_id not in enabled:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{model_id}' is disabled on the public deployment.",
        )
```

Call `reject_disabled_model()` for each requested model at the top of `/api/compare` — **before** `get_or_load_model`, so a disabled model is rejected without ever touching the lazy loader. Task 19 applies the same guard to the detector endpoints.

Unit test (runs in CI — no slowapi, no torch):

```python
def test_compare_rejects_disabled_model(monkeypatch, client_with_model):
    monkeypatch.setenv("ENABLED_MODELS", "twitter-roberta")
    resp = client_with_model.post(
        "/api/compare",
        json={"text": "great", "model_ids": ["distilbert-sst2"]},
    )
    assert resp.status_code == 403
```

- [ ] **Step 3: Write `Dockerfile.spaces`**

Multi-stage: `node:24-alpine` builds `frontend/dist`; `python:3.13-slim` installs CPU torch + `requirements-docker.txt`, copies `backend/app` + `dist`, pre-downloads the default model with `HF_HOME=/tmp/hf`, sets `STATIC_DIR=/app/static`, `PUBLIC_DEPLOY=1`, and `CMD uvicorn app.main:app --host 0.0.0.0 --port 7860`.

- [ ] **Step 4: Space metadata + push**

`SPACE_README.md` front-matter:

```yaml
---
title: SentimentScope
emoji: 🎯
colorFrom: indigo
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---
```

Create the Space (`huggingface-cli repo create sentiment-scope --type space --space_sdk docker` or via the website), add it as a git remote, push. Docs: <https://huggingface.co/docs/hub/spaces-sdks-docker>.

- [ ] **Step 5: Verify**

Local first: `docker build -f Dockerfile.spaces -t scope-space . && docker run -p 7860:7860 scope-space`, then `curl localhost:7860/api/health` → `model_loaded: true` **without** any network download (weights baked in), and `localhost:7860` serves the UI. Then verify the live Space URL end-to-end, including a rate-limit 429 after hammering `/api/explain`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: single-image Hugging Face Spaces deployment with rate limiting"
```

Link the live Space at the top of the repo README (Task 18).

---

### Task 18: README + final verification

**Files:**

- Create: `README.md`, `sample-data/reviews.csv`

- [ ] **Step 1: Sample data**

`sample-data/reviews.csv`:

```csv
text
The battery life on this phone is incredible
Absolutely terrible customer service, never again
It arrived on time and works as described
I love the design but the app keeps crashing
Best purchase I have made all year
Mediocre at best, expected more for the price
The screen cracked after two days
Setup was quick and painless
Not bad, not great
This exceeded every expectation I had
```

- [ ] **Step 2: Write README.md**

`README.md` (repo root) — write exactly this structure, filling the screenshots section after taking them:

> **Phase note:** this template documents the final Phase 1 + 2 state. When executing Phase 1 only, omit the `ai-detect` endpoint row, the AI-detection bullets, and the AI Detector screenshot — Task 22 restores them once detection ships.

````markdown
# SentimentScope

An educational, end-to-end ML engineering app: a React UI talking to a FastAPI
backend that serves transformer sentiment models locally — with token-level
**Integrated Gradients explainability**, model comparison, evaluation metrics,
error analysis, batch CSV analysis, tests, Docker, and CI.

Built as an AI/ML portfolio project: the code is deliberately over-commented,
teaching the *why* of each step (tokenization → logits → softmax, GPU batching,
IG attribution) alongside the *what*.

## Architecture

```
React (Vite + TS + Tailwind + Recharts)
        │  relative /api/* calls (Vite proxy in dev, nginx in Docker)
        ▼
FastAPI ── lifespan loads default SentimentModel (twitter-roberta) once
        │       ├── analyze/batch/csv/explain: default sentiment model only, strict 3-class
        │       ├── compare(): lazy-loads sentiment registry models, dynamic scores
        │       ├── ai-detect/compare(): lazy-loads detector models (separate task family)
        │       └── explain(): captum LayerIntegratedGradients on roberta.embeddings
        ▼
Model registry (task-aware): sentiment (RoBERTa, DistilBERT, FinBERT, XLM-R) + AI detectors (desklib, fakespot, oxidane)
```

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Single text → label + class probabilities (**twitter-roberta only**, strict 3-class) |
| `POST /api/analyze/batch` | JSON list (≤500) → per-row results + aggregates (**twitter-roberta only**) |
| `POST /api/analyze/csv` | CSV upload (`text` column) → same as batch (**twitter-roberta only**) |
| `POST /api/explain` | Integrated Gradients token attributions (**twitter-roberta only**) |
| `GET /api/models?task=` · `POST /api/compare` | Task-aware model registry · sentiment-only side-by-side comparison (dynamic scores) |
| `POST /api/ai-detect` · `POST /api/ai-detect/compare` | AI text detection (separate task family) with disagreement reporting |
| `GET /api/health` · `GET /api/model` | Readiness · default model card |

## Quickstart

### Docker (one command)

```bash
docker compose up --build
# → http://localhost:8080  (first run downloads ~500MB of model weights)
```

### Local dev

```bash
# Backend — any env with torch + transformers + captum + fastapi
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173 (proxies /api to :8000)
```

Try it: upload `sample-data/reviews.csv` on the Batch tab, then compare models on a finance-style sentence.

## Tests

```bash
cd backend && pytest            # unit tests — model mocked, runs anywhere
cd backend && pytest -m integration  # real-model tests (needs weights)
cd frontend && npm test -- --run
```

CI (GitHub Actions) runs lint + unit tests + frontend build on every push —
without installing torch, because heavy imports are lazy and unit tests inject
a fake model via FastAPI dependency overrides.

## What this project demonstrates

- **ML fundamentals in code:** raw `AutoModel` inference (no `pipeline()` magic) —
  tokenization, batching, softmax, and device placement are all explicit and explained.
- **Explainability:** Layer Integrated Gradients with a padding baseline; the UI
  renders per-token attributions as a heatmap. **RoBERTa only** — other registry models are compare-only.
- **Model comparison:** one input across social, binary SST-2, finance, and multilingual sentiment models to show domain/label mismatch and latency tradeoffs (`/api/compare` is sentiment-only).
- **AI text detection as a separate ML task:** local detector models are served through dedicated endpoints with detector disagreement reporting and uncertainty warnings.
- **Evaluation:** `evals/run_eval.py` reports accuracy, macro F1, confusion matrix, p50/p95 latency, and wrong examples; `evals/run_ai_detect_eval.py` adds detector disagreement analysis.
- **Engineering hygiene:** validation at the boundary, dependency-injected model for
  testability, integration/unit test split, CPU-only Docker build, CI without GPU deps.

## Honest limitations

- The default model was trained on tweets: long/formal text is out-of-domain.
- Explainability (Integrated Gradients) is implemented for the default Twitter RoBERTa model only; DistilBERT/FinBERT/XLM-R are available via `/api/compare` only.
- Sentiment models are not creativity judges; they estimate polarity, confidence, and disagreement.
- AI text detectors are probabilistic and should not be used as proof of authorship. Edited AI text, formal human writing, short text, and non-native writing can all confuse detectors.
- 512-token truncation; sarcasm, mixed sentiment, and missing context remain hard.
- IG uses 50 integration steps — a principled approximation, not ground truth.

## Screenshots

<!-- Add after first run: Analyze heatmap, Compare Sentiment table, AI Detector tab, Batch charts -->
````

- [ ] **Step 3: Full verification pass**

```bash
cd ~/Projects/active/sentiment-scope/backend && pytest -v && pytest -m integration -v && ruff check .
cd ../frontend && npm test -- --run && npm run lint && npm run build
cd .. && docker compose up --build -d && sleep 5 && curl -s localhost:8080/api/health && docker compose down
```

Expected: everything green; health returns `model_loaded: true` (allow time for first model download in Docker).
Manual: run both dev servers, exercise all tabs including Compare Sentiment, AI Detector (after Task 21), `sample-data/reviews.csv` upload, and an Explain call.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: README with architecture, quickstart, and sample data"
```

---

## Phase 2 — AI text detection (Tasks 19–22)

> Execute only after Task 18 is complete and the Phase 1 app is deployed and stable. These tasks are placed at the end of the plan (rather than interleaved with Tasks 10–15) so an agentic worker executing top-to-bottom builds Phase 1 in one clean pass.

### Task 19: AI text detection backend

**Purpose:** Add AI-written-text detection as a separate ML task family, not as sentiment comparison.

**Files:**

- Modify: `backend/app/model_registry.py`
- Modify: `backend/app/model.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes.py`
- Test: `backend/tests/test_ai_detect.py`

**Interfaces:**

- Produces: `POST /api/ai-detect`, `POST /api/ai-detect/compare`, and `GET /api/models?task=ai_text_detection`.
- AI detector responses use dynamic score dictionaries.
- Detector models are lazy-loaded and cached using the same `get_or_load_model` mechanism.
- Detector outputs must include an uncertainty warning.

**Design rules:**

- Do not mix detector models with sentiment models.
- `/api/compare` remains sentiment-only.
- `/api/ai-detect/compare` is detector-only.
- Detector labels must be canonicalized to `human` / `ai`.
- If a detector model exposes unknown raw labels, fail loudly during integration testing instead of guessing.
- **Detectors are NOT guaranteed to be `AutoModelForSequenceClassification`.** Verified 2026-07-01: `desklib/ai-text-detector-v1.01` is a custom `PreTrainedModel` subclass (DeBERTa-v3-large base + mean pooling + a single-logit head with **sigmoid**, not 2-class softmax). It cannot load through `SentimentModel.load()`. Give the detector family its own `DetectorModel` wrapper with a per-model load/predict path: inspect each local detector's `config.json` + README first, implement desklib's mean-pool/sigmoid head explicitly (copy the class from its model card), and map sigmoid probability `p` to `{"human": 1-p, "ai": p}`. `fakespot` (RoBERTa-base) likely loads via AutoModel; verify `oxidane` the same way before wiring.

**DetectorModel + output adapters (how detectors actually load and score):**

Update `build_model` in Task 8's registry module to return `DetectorModel(cfg)` for `ModelTask.AI_TEXT_DETECTION`, replacing the Phase 1 `NotImplementedError`.

```python
class DetectorModel:
    """Detector-family counterpart of SentimentModel: same interface the
    cache and routes rely on (load/predict/is_loaded/labels/device), but a
    task-appropriate load path and output head.

    Educational point: "binary classifier" hides two different architectures.
    A 2-logit softmax head and a 1-logit sigmoid head produce the same kind
    of answer, but conflating them mangles the probabilities — you can't
    softmax a single logit.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        ...

    def load(self) -> None:
        # Same local-dir → Hub-name fallback as SentimentModel.
        source = resolve_model_source(self._config)
        # desklib: instantiate its custom class (copy from the model card) —
        # AutoModelForSequenceClassification would mis-load the checkpoint.
        # fakespot/oxidane: standard AutoModel path if config.json confirms it.
        ...

    def predict(self, texts: list[str]) -> list[dict]:
        ...
        if self._config.output_adapter == "single_logit_sigmoid":
            # One logit per text: sigmoid(logit) = P(ai). Emit both sides so
            # the response shape matches softmax-model responses exactly.
            p_ai = torch.sigmoid(logits.squeeze(-1))
            return [
                {
                    "label": "ai" if p >= 0.5 else "human",
                    "scores": {"human": round(1 - float(p), 4), "ai": round(float(p), 4)},
                }
                for p in p_ai
            ]
        # softmax adapter: map raw id2label (LABEL_0/human/real/…) onto the
        # canonical ("human", "ai") tuple verified from each config.json.
        ...
```

**Schemas:**

```python
class AiDetectRequest(AnalyzeRequest):
    model_ids: list[str] | None = None


class AiDetectItem(DynamicAnalyzeResponse):
    model_id: str
    name: str
    domain: str
    confidence: float
    latency_ms: float
    note: str


class AiDetectResponse(BaseModel):
    result: AiDetectItem
    warning: str


class AiDetectCompareResponse(BaseModel):
    results: list[AiDetectItem]
    disagreement: bool
    warning: str
```

**Warning text:**

```text
AI detectors are probabilistic and can be wrong, especially on short, edited, non-native, highly formal, or mixed-authorship text. Do not use this as proof of authorship.
```

**Default detector behavior:**

- `/api/ai-detect` uses one default detector, preferably `desklib-ai-detector`.
- `/api/ai-detect/compare` uses all available AI detector models unless `model_ids` is provided.
- If detector predictions disagree, set `disagreement: true`.

**Tests:**

- `POST /api/ai-detect` returns label, scores, confidence, model_id, and warning.
- `POST /api/ai-detect/compare` returns multiple detector rows.
- Sentiment models are rejected from detector endpoints.
- Detector models are rejected from sentiment `/api/compare`.
- Disagreement is `true` when fake detector outputs differ.

- [ ] **Step 1: Implement schemas + routes**

Wire `/api/ai-detect` and `/api/ai-detect/compare`. Reject any `model_id` whose `ModelConfig.task != ModelTask.AI_TEXT_DETECTION` on detector routes; reject detector IDs on `/api/compare`. Apply `reject_disabled_model()` (Task 16A) to every requested detector ID before lazy loading, so the public `ENABLED_MODELS` allowlist covers detectors too.

- [ ] **Step 2: Canonicalize detector labels**

Before scoring, read each detector's `config.json` `id2label` and map to canonical `human` / `ai`. Integration tests must assert mapping for all three local detectors.

- [ ] **Step 3: Verify**

```bash
cd backend && pytest tests/test_ai_detect.py -v
cd backend && pytest -m integration tests/test_ai_detect.py -v  # local only, real weights
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: AI text detection endpoints as separate task family"
```

---

### Task 20: AI detector evaluation + disagreement report

**Purpose:** Evaluate detector behavior honestly instead of pretending AI detection is definitive.

**Files:**

- Create: `evals/data/ai_detection_eval.csv`
- Create: `evals/run_ai_detect_eval.py`
- Create: `evals/ai_detection_report.md`

**Dataset columns:**

```csv
id,text,true_label,source_type,notes
1,"This is a short human-written note.",human,human_short,clear human
2,"As an AI language model, I can provide...",ai,ai_obvious,common AI phrasing
3,"The report was completed after reviewing the dataset.",human,formal_human,formal human text
4,"In conclusion, this product offers a seamless and innovative experience.",ai,marketing_ai,polished AI-like text
```

Four rows is a smoke test, not a showcase. Seed **at least 24 rows** (2–3 per category) across: `human_short`, `human_formal`, `human_non_native`, `human_marketing`, `ai_obvious`, `ai_polished`, `ai_edited`, `ai_paraphrased`, `mixed_human_ai`, `ambiguous`. The point of this module is failure analysis and detector disagreement — those only show up with enough varied rows for detectors to disagree on.

**Metrics:**

- Accuracy
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix
- Latency p50/p95
- Wrong examples
- Detector disagreement rate
- Examples where detectors disagree

**Honest report section:**

```text
AI text detection is not proof of authorship. These models estimate whether text resembles patterns seen in AI-generated or human-written training data. Short text, edited AI text, non-native writing, and formal human writing can all confuse detectors.
```

**Verification:**

- Run one detector eval locally.
- Confirm the report includes wrong examples and disagreement examples.
- Confirm the README does not claim AI detection is definitive.

- [ ] **Step 1: Create seed CSV + eval script**

Mirror `evals/run_eval.py` patterns but call detector models and compute disagreement across all three detectors per row.

- [ ] **Step 2: Write ai_detection_report.md**

Include wrong examples, disagreement examples, and the honest limitations section above.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: AI detector evaluation with disagreement report"
```

---

### Task 21: AI Text Detector tab

**Files:**

- Create: `frontend/src/components/AiTextDetector.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/components/AiTextDetector.test.tsx`

**UI behavior:**

- Textarea for input.
- Button: `Detect AI Text`.
- Optional button: `Compare Detectors`.
- Result card shows top label, confidence, model name, and warning.
- Detector comparison table shows model, prediction, confidence, latency, and note.
- If `disagreement: true`, show a visible warning badge.

**UI copy:**

```text
AI detectors are probabilistic. Treat this as a model signal, not proof of authorship.
```

**Tabs:**

```text
Analyze | Batch | Compare Sentiment | AI Detector | How it works
```

Update `App.tsx` from Task 15: rename "Compare Models" tab to "Compare Sentiment", add "AI Detector" tab rendering `<AiTextDetector />`.

**Tests:**

- Renders warning text.
- Shows detector result.
- Shows disagreement badge when detector outputs differ.

- [ ] **Step 1: Implement AiTextDetector.tsx**

Wire `detectAiText` and `compareAiDetectors` from `../api`. Display `warning` from the API response prominently.

- [ ] **Step 2: Update App.tsx tabs**

```tsx
const TABS = ["Analyze", "Batch", "Compare Sentiment", "AI Detector", "How it works"] as const;
```

- [ ] **Step 3: Verify**

```bash
cd frontend && npm test -- --run && npm run lint && npm run build
```

Manual: AI Detector tab shows single-detector result and compare table; disagreement badge appears when mocked responses differ.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: AI text detector tab with disagreement warning"
```

---

### Task 22: README + Space update for AI detection

**Files:**

- Modify: `README.md` (repo root), `SPACE_README.md` / Space settings (if Task 16A shipped)

- [ ] **Step 1: Restore AI-detection content in README** — the `ai-detect` endpoint row, the "AI text detection as a separate ML task" bullet, and the detector limitation bullets omitted during Phase 1 (see the Phase note in Task 18).
- [ ] **Step 2: Screenshots + Space rollout** — add an AI Detector tab screenshot. If publicly deployed, add detector IDs to `ENABLED_MODELS` one at a time and watch RAM on the free tier (DeBERTa-v3-large is ~1.6GB resident; all detectors + two sentiment models still fit in 16GB, but verify before enabling all three).
- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs: document AI text detection task family"
```

---

## Self-Review Notes

- **Spec coverage:** spec's single `POST /api/analyze/batch` accepting "CSV or JSON" is implemented as two endpoints (`/analyze/batch` JSON, `/analyze/csv` multipart) — FastAPI handles one content type per route cleanly; the spec's intent (both input modes, same response shape) is preserved.
- All spec success criteria map to tasks: docker-compose (16), endpoint shapes (4–9, 19), eval/error analysis (10, 20), IG heatmap on sample inputs (7/13), CI green (17), educational comments (throughout, enforced by Global Constraints).
- **Task-aware registry:** `ModelTask.SENTIMENT` vs `ModelTask.AI_TEXT_DETECTION`; sentiment and detector models share lazy-loading but never share compare endpoints. <!-- Phase 2 -->
- Type consistency verified: `FakeModel` mirrors `SentimentModel`'s interface; frontend types mirror Pydantic response models; `aggregate()` output matches `BatchAggregates`; compare rows use `DynamicAnalyzeResponse` / `DynamicScores` so binary DistilBERT never fakes a neutral class. <!-- ✅ VALIDATED 2026-07-01 (review fix #1) -->
- **Schema split:** strict `Scores` (3-class) for default `/api/analyze`, `/api/analyze/batch`, `/api/analyze/csv`, `/api/explain`; dynamic `dict[str, float]` for `/api/compare` (sentiment) and `/api/ai-detect*` (detectors). No `model_id` on analyze/batch/csv — optional registry models are compare-only within their task family.
- **Compare defaults:** backend and frontend both default to `twitter-roberta` + `distilbert-sst2`; optional models require explicit opt-in. <!-- ✅ VALIDATED 2026-07-01 (review fix #2) -->
- **Lazy loading:** per-model `asyncio.Lock` + `asyncio.to_thread(m.load)` prevents duplicate loads and event-loop blocking; Docker loads only the default model at startup. <!-- ✅ VALIDATED 2026-07-01 (review fix #3, #4, #10) -->
- **Explain RoBERTa-only:** `/api/explain` returns 400 for other `model_id` values; IG wired to `roberta.embeddings`. <!-- ✅ VALIDATED 2026-07-01 (review fix #9) — Context7 Captum -->
- **Registry labels:** `ModelConfig.labels` is canonical; integration test asserts score keys match config for every registry model. <!-- ✅ VALIDATED 2026-07-01 (review fix #11) — HF config fetched live -->
- **Eval honesty:** `--allow-label-mismatch` required when dataset labels ⊄ model labels. <!-- ✅ VALIDATED 2026-07-01 (review fix #12) -->
- **Plan audit tags** (`✅ VALIDATED`) stay in this document only — not in source. <!-- ✅ VALIDATED 2026-07-01 (review fix #13) -->
- **Dependency split:** `requirements-docker.txt` (no torch duplicate), `requirements-eval.txt` (sklearn only), `requirements-dev.txt` (no torch/sklearn). <!-- ✅ VALIDATED 2026-07-01 (review fix #6, #7) — PyPI pins re-checked -->
- **Node 24** in CI/Docker (not 26). <!-- ✅ VALIDATED 2026-07-01 (review fix #5) — Context7 + Tavily -->

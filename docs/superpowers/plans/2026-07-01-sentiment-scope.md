# SentimentScope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-grade educational sentiment analysis app: React UI + FastAPI backend wrapping `cardiffnlp/twitter-roberta-base-sentiment-latest` with Integrated Gradients explainability, tests, Docker, and CI.

**Architecture:** FastAPI backend loads the RoBERTa classifier once at startup (singleton, lifespan) and exposes analyze/batch/csv/explain endpoints; a Vite+React+TS frontend calls them via relative `/api` paths (Vite dev proxy locally, nginx proxy in Docker). Unit tests mock the model via FastAPI dependency override so CI never downloads torch; real-model tests are `@pytest.mark.integration` and run locally only.

**Tech Stack:** Python 3.12+, FastAPI, torch, transformers, captum (Layer Integrated Gradients), pytest, ruff · React 18, Vite, TypeScript, Tailwind v4, Recharts, vitest + testing-library · Docker + docker-compose · GitHub Actions.

## Global Constraints

- Project root: `~/Projects/active/sentiment-scope` (git repo already initialized; spec committed).
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`, labels `negative/neutral/positive`, 512-token truncation.
- Local dev backend runs in the existing `ai` conda env (`ca ai`); torch 2.11 + transformers 5.5.3 already installed; captum must be `pip install`-ed (Task 7).
- Batch limits: ≤ 500 texts per request, ≤ 2000 chars per text. Validation at the boundary (Pydantic / explicit HTTP 400s).
- **Educational tone is a hard requirement:** every backend module and every non-trivial frontend component gets explanatory comments teaching the ML/engineering concept involved (why softmax, why batching, what IG computes, why validation lives at the boundary). Code in this plan already includes them — copy verbatim, don't strip them.
- No Claude co-author trailers or "Generated with Claude Code" footers in any commit.
- Frontend API calls always use relative `/api/...` paths — never hardcode `localhost:8000` in components.
- CI must not install torch/transformers/captum (unit tests never import them — heavy imports live inside `SentimentModel` methods).

---

### Task 1: Backend scaffold + health endpoint

**Files:**
- Create: `.gitignore`, `backend/pyproject.toml`, `backend/requirements.txt`, `backend/requirements-dev.txt`, `backend/app/__init__.py`, `backend/app/main.py`, `backend/app/model.py` (skeleton), `backend/app/routes.py` (health only)
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

`backend/requirements.txt` (full runtime — used by Docker; the local `ai` conda env already satisfies torch/transformers):

```
fastapi
uvicorn[standard]
python-multipart
torch
transformers
captum
```

`backend/requirements-dev.txt` (what unit tests + CI need — deliberately NO torch, see Global Constraints):

```
fastapi
uvicorn[standard]
python-multipart
httpx
pytest
ruff
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
- Produces: `AnalyzeRequest(text)`, `Scores(negative, neutral, positive)`, `AnalyzeResponse(label, scores)`, `BatchRequest(texts)`, `BatchItem(text, label, scores)`, `BatchAggregates(counts, mean_scores)`, `BatchResponse(results, aggregates)`, `TokenAttribution(token, attribution)`, `ExplainResponse(label, scores, tokens)`. Tasks 4–7 import these exact names.

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
```

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
- Produces: `SentimentModel.predict(texts: list[str]) -> list[dict]` where each dict is `{"label": str, "scores": {"negative": float, "neutral": float, "positive": float}}`. Tasks 4–6 rely on this exact shape.

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
- Produces: `POST /api/analyze/csv` — multipart upload, field name `file`, CSV must have a `text` column, ≤500 data rows; returns `BatchResponse` (same shape as batch). Frontend Task 11 posts `FormData` here.

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
        if t:
            texts.append(t[:MAX_CHARS])

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
- Produces: `SentimentModel.explain(text: str) -> dict` (`{label, scores, tokens: [{token, attribution}]}`); `POST /api/explain` (`AnalyzeRequest` → `ExplainResponse`); `GET /api/model` → `{name, labels, max_tokens, device, description}`. Frontend Tasks 10/12 consume these.

- [ ] **Step 1: Install captum in the ai env**

Run: `ca ai && pip install captum`
Expected: installs cleanly against torch 2.11.

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
```

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
@router.post("/explain", response_model=ExplainResponse)
def explain(req: AnalyzeRequest, model: SentimentModel = Depends(get_model)):
    # IG runs one forward pass per integration step (50x slower than
    # /analyze) — that's why explanation is a separate, opt-in endpoint.
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

- [ ] **Step 5: Verify**

Run: `pytest -v && ruff check .` — Expected: all unit tests pass.
Run: `pytest -m integration -v` — Expected: 2 passed. **If MPS produces NaN attributions** (captum + MPS edge case), set `self.device = "cpu"` temporarily in `load()` to confirm, then keep MPS for predict and move only `explain` to CPU — do this by adding `.cpu()` copies of model/inputs inside `explain()` only if the NaN case actually occurs. Don't pre-build the fallback otherwise.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: Integrated Gradients explainability and model info endpoints"
```

---

### Task 8: Frontend scaffold + typed API client

**Files:**
- Create: `frontend/` (Vite react-ts template), `frontend/src/api.ts`, `frontend/src/test-setup.ts`
- Modify: `frontend/vite.config.ts`, `frontend/src/index.css`, `frontend/package.json`
- Test: `frontend/src/api.test.ts`

**Interfaces:**
- Produces: `api.ts` exports — types `Scores`, `AnalyzeResult`, `TokenAttribution`, `ExplainResult`, `BatchItem`, `BatchResult`, `ModelInfo`; functions `analyze(text)`, `explainText(text)`, `analyzeCsv(file)`, `getModelInfo()`. All UI tasks import from here; components never call `fetch` directly.

- [ ] **Step 1: Scaffold**

```bash
cd ~/Projects/active/sentiment-scope
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install recharts tailwindcss @tailwindcss/vite
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom
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
Delete `frontend/src/App.css` and `frontend/src/assets/react.svg` (template cruft; App.tsx gets replaced in Task 9 — for now remove the `import "./App.css"` line from it so the build stays green).

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

export interface Scores {
  negative: number;
  neutral: number;
  positive: number;
}

export interface AnalyzeResult {
  label: string;
  scores: Scores;
}

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
```

- [ ] **Step 5: Verify** — `npm test -- --run && npm run lint && npm run build` — Expected: 2 tests pass, lint + build clean.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/active/sentiment-scope && git add -A && git commit -m "feat: frontend scaffold with typed API client"
```

---

### Task 9: Analyze tab (AnalyzeForm + ConfidenceBars)

**Files:**
- Create: `frontend/src/components/ConfidenceBars.tsx`, `frontend/src/components/AnalyzeForm.tsx`
- Modify: `frontend/src/App.tsx` (render AnalyzeForm for now; tabs arrive in Task 12)
- Test: `frontend/src/components/ConfidenceBars.test.tsx`

**Interfaces:**
- Consumes: `analyze` from `../api`.
- Produces: `<ConfidenceBars scores={Scores} />`; `<AnalyzeForm />` (self-contained: textarea + Analyze/Explain buttons; renders ConfidenceBars and, after Task 10, TokenHeatmap).

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
import type { Scores } from "../api";

/**
 * Horizontal bars for the three class probabilities. Showing the full
 * softmax distribution (not just the winner) is deliberate: "positive 51%"
 * and "positive 98%" are very different answers, and hiding that nuance is
 * how ML demos mislead people.
 */

const BAR_COLOR: Record<keyof Scores, string> = {
  negative: "bg-red-500",
  neutral: "bg-slate-400",
  positive: "bg-emerald-500",
};

export default function ConfidenceBars({ scores }: { scores: Scores }) {
  return (
    <div className="space-y-2">
      {(Object.keys(BAR_COLOR) as (keyof Scores)[]).map((label) => (
        <div key={label} className="flex items-center gap-3">
          <span className="w-20 text-sm capitalize text-slate-600">{label}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-slate-200">
            <div
              className={`h-full ${BAR_COLOR[label]} transition-all`}
              style={{ width: `${scores[label] * 100}%` }}
            />
          </div>
          <span className="w-14 text-right text-sm tabular-nums text-slate-600">
            {(scores[label] * 100).toFixed(1)}%
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
          {/* TokenHeatmap renders here after Task 10 */}
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
Manual smoke test: `ca ai && cd backend && uvicorn app.main:app --port 8000` in one shell, `npm run dev` in another, open http://localhost:5173, analyze "I love this" → positive badge + bars.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: analyze tab with confidence bars"
```

---

### Task 10: Token attribution heatmap

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

### Task 11: Batch tab (CSV upload + aggregate charts)

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

`npm test -- --run && npm run lint && npm run build` — pass (component is exercised manually + via Task 12 nav; charts are visual).
Manual: create `/tmp/sample.csv` with a `text` column of ~5 mixed-sentiment rows, upload on the Batch tab (after Task 12 wires it in — or temporarily render `<BatchUpload />` in App to check now, then revert).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: batch CSV upload with aggregate charts and results table"
```

---

### Task 12: Tabs, How-it-works page, model info footer

**Files:**
- Create: `frontend/src/components/HowItWorks.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `AnalyzeForm`, `BatchUpload`, `getModelInfo`/`ModelInfo` from `../api`.
- Produces: final `App` with three tabs: Analyze · Batch · How it works.

- [ ] **Step 1: Implement HowItWorks**

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

- [ ] **Step 2: Wire tabs in App.tsx**

Replace `frontend/src/App.tsx`:

```tsx
import { useState } from "react";
import AnalyzeForm from "./components/AnalyzeForm";
import BatchUpload from "./components/BatchUpload";
import HowItWorks from "./components/HowItWorks";

const TABS = ["Analyze", "Batch", "How it works"] as const;
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
          {tab === "How it works" && <HowItWorks />}
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify**

`npm test -- --run && npm run lint && npm run build` — pass.
Manual: all three tabs render; Batch tab accepts the sample CSV; How-it-works shows the model footer when the backend is up.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: tabbed layout with educational how-it-works page"
```

---

### Task 13: Docker

**Files:**
- Create: `backend/Dockerfile`, `backend/.dockerignore`, `frontend/Dockerfile`, `frontend/nginx.conf`, `frontend/.dockerignore`, `docker-compose.yml`

**Interfaces:**
- Produces: `docker compose up --build` → app at http://localhost:8080, nginx proxying `/api/` to the backend container; HF weights cached in the `hf-cache` volume so rebuilds don't re-download.

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
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch wheel: ~10x smaller than the default CUDA build, and there
# is no GPU inside this container anyway.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

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
# Stage 1: build the static bundle
FROM node:22-alpine AS build
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
Then: `curl -s http://localhost:8080/api/health` → `{"status":"ok","model_loaded":true,"device":"cpu"}`; open http://localhost:8080 and analyze a sentence.
Teardown: `docker compose down`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: dockerized deployment with nginx proxy and weight cache"
```

---

### Task 14: GitHub Actions CI

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
          python-version: "3.12"
          cache: pip
      # requirements-dev.txt deliberately excludes torch/transformers/captum:
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
          node-version: "22"
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

### Task 15: README + final verification

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

````markdown
# SentimentScope

An educational, end-to-end sentiment analysis app: a React UI talking to a FastAPI
backend that serves `cardiffnlp/twitter-roberta-base-sentiment-latest` locally —
with token-level **Integrated Gradients explainability**, batch CSV analysis,
tests, Docker, and CI.

Built as an AI/ML portfolio project: the code is deliberately over-commented,
teaching the *why* of each step (tokenization → logits → softmax, GPU batching,
IG attribution) alongside the *what*.

## Architecture

```
React (Vite + TS + Tailwind + Recharts)
        │  relative /api/* calls (Vite proxy in dev, nginx in Docker)
        ▼
FastAPI ── lifespan loads SentimentModel once (singleton)
        │       ├── predict(): batched tokenize → logits → softmax
        │       └── explain(): captum LayerIntegratedGradients on embeddings
        ▼
cardiffnlp/twitter-roberta-base-sentiment-latest (MPS locally / CPU in Docker)
```

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Single text → label + class probabilities |
| `POST /api/analyze/batch` | JSON list (≤500) → per-row results + aggregates |
| `POST /api/analyze/csv` | CSV upload (`text` column) → same as batch |
| `POST /api/explain` | Integrated Gradients token attributions |
| `GET /api/health` · `GET /api/model` | Readiness · model card |

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

Try it: upload `sample-data/reviews.csv` on the Batch tab.

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
  renders per-token attributions as a heatmap.
- **Engineering hygiene:** validation at the boundary, dependency-injected model for
  testability, integration/unit test split, CPU-only Docker build, CI without GPU deps.

## Honest limitations

- Trained on tweets: long/formal text is out-of-domain; English only.
- 512-token truncation; sarcasm remains hard.
- IG uses 50 integration steps — a principled approximation, not ground truth.

## Screenshots

<!-- Add after first run: Analyze tab with heatmap, Batch tab with charts -->
````

- [ ] **Step 3: Full verification pass**

```bash
cd ~/Projects/active/sentiment-scope/backend && pytest -v && pytest -m integration -v && ruff check .
cd ../frontend && npm test -- --run && npm run lint && npm run build
cd .. && docker compose up --build -d && sleep 5 && curl -s localhost:8080/api/health && docker compose down
```

Expected: everything green; health returns `model_loaded: true` (allow time for first model download in Docker).
Manual: run both dev servers, exercise all three tabs including `sample-data/reviews.csv` upload and an Explain call.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: README with architecture, quickstart, and sample data"
```

---

## Self-Review Notes

- **Spec coverage:** spec's single `POST /api/analyze/batch` accepting "CSV or JSON" is implemented as two endpoints (`/analyze/batch` JSON, `/analyze/csv` multipart) — FastAPI handles one content type per route cleanly; the spec's intent (both input modes, same response shape) is preserved.
- All spec success criteria map to tasks: docker-compose (13), endpoint shapes (4–7), IG heatmap on sample inputs (7/10), CI green (14), educational comments (throughout, enforced by Global Constraints).
- Type consistency verified: `FakeModel` mirrors `SentimentModel`'s interface; frontend types mirror Pydantic response models; `aggregate()` output matches `BatchAggregates`.

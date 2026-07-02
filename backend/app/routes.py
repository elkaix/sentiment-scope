"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

import csv
import io
import os
import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.model import SentimentModel, get_or_load_model
from app.model_registry import (
    MODEL_REGISTRY,
    ModelTask,
    get_default_model_id,
    get_model_config,
    models_for_task,
)
from app.schemas import (
    MAX_BATCH,
    MAX_CHARS,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    CompareRequest,
    CompareResponse,
    ExplainResponse,
)

router = APIRouter(prefix="/api")

LABELS = ("negative", "neutral", "positive")
EXPLAIN_MODEL_ID = "twitter-roberta"
# A legitimate CSV is bounded by the same limits BatchRequest enforces:
# at most MAX_BATCH non-empty texts, each at most MAX_CHARS characters —
# roughly 500 * 2000 = 1MB, or ~4MB in the worst case of 4-byte UTF-8
# characters throughout. 5MB is generous headroom for real use while still
# bounding memory and parse CPU against a pathological file (e.g. millions
# of blank lines slipping past the row-count check). Resource limits belong
# at the boundary, like every other validation in this file.
MAX_CSV_BYTES = 5 * 1024 * 1024
DEFAULT_SENTIMENT_MODEL_ID = get_default_model_id(ModelTask.SENTIMENT)
# Compare only two models by default: loading all four registry models on the
# first call would download/hold ~2GB and feel broken in a portfolio demo.
DEFAULT_COMPARE_MODELS = ["twitter-roberta", "distilbert-sst2"]


def reject_non_default_model_id(model_id: str | None) -> None:
    """analyze/batch/csv are twitter-roberta only. A stray model_id (e.g. the
    binary distilbert) is refused early — its 2-class output would break the
    strict 3-class Scores schema. Other models live on /api/compare (Task 9)."""
    if model_id and model_id != DEFAULT_SENTIMENT_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail="This endpoint uses the default twitter-roberta model only. "
            "Use /api/compare for other models.",
        )


def enabled_model_ids() -> set[str] | None:
    """Parse the ENABLED_MODELS allowlist. None = no restriction (dev).

    The public Space sets ENABLED_MODELS so anonymous users can't lazy-load
    every registry model and balloon RAM on the free 16GB box; dev and CI
    leave it unset. Read per call, not at import, so tests can monkeypatch it.
    """
    raw = os.getenv("ENABLED_MODELS")
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def reject_disabled_model(model_id: str) -> None:
    enabled = enabled_model_ids()
    if enabled is not None and model_id not in enabled:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{model_id}' is disabled on the public deployment. "
            f"Available models: {', '.join(sorted(enabled))}. Run the app "
            "locally (see the repo README) to compare the full registry.",
        )


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


@router.get("/models")
def list_models(request: Request, task: ModelTask | None = None):
    """Registry metadata for every model, plus a live `loaded` flag. Optional
    ?task=sentiment|ai_text_detection narrows the list.

    Reads app.state directly (like /health) instead of Depends(get_model): the
    catalog must work even before any weights load, and get_model would 503
    the whole list whenever the default model isn't loaded (e.g. under tests)."""
    cache = request.app.state.model_cache
    default_model = request.app.state.model
    configs = models_for_task(task) if task is not None else MODEL_REGISTRY

    def is_loaded(model_id: str) -> bool:
        # Optional models live in the lazy cache; the default sentiment model
        # is eagerly loaded into app.state.model at startup.
        if model_id in cache:
            return True
        return model_id == DEFAULT_SENTIMENT_MODEL_ID and default_model.is_loaded

    models = [
        {
            "id": cfg.id,
            "name": cfg.name,
            "task": cfg.task.value,  # StrEnum -> "sentiment", not "ModelTask.SENTIMENT"
            "labels": list(cfg.labels),
            "domain": cfg.domain,
            "note": cfg.note,
            "default": cfg.default,
            "loaded": is_loaded(model_id),
        }
        for model_id, cfg in configs.items()
    ]
    return {"models": models}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    req: AnalyzeRequest,
    model_id: str | None = None,
    model: SentimentModel = Depends(get_model),
):
    reject_non_default_model_id(model_id)
    # predict() is batched by design; a single text is just a batch of one.
    return model.predict([req.text])[0]


@router.post("/analyze/batch", response_model=BatchResponse)
def analyze_batch(
    req: BatchRequest,
    model_id: str | None = None,
    model: SentimentModel = Depends(get_model),
):
    reject_non_default_model_id(model_id)
    # ONE batched model call for the whole list — see predict()'s docstring
    # for why that beats a per-text loop.
    results = model.predict(req.texts)
    items = [{"text": t, **r} for t, r in zip(req.texts, results)]
    return {"results": items, "aggregates": aggregate(results)}


@router.post("/analyze/csv", response_model=BatchResponse)
async def analyze_csv(
    file: UploadFile = File(...),
    model_id: str | None = None,
    model: SentimentModel = Depends(get_model),
):
    """CSV variant of batch analysis. File parsing is inherently messier than
    JSON, so each failure mode gets its own explicit 400 — the caller should
    always learn WHY their file was rejected."""
    reject_non_default_model_id(model_id)
    raw = await file.read()
    if len(raw) > MAX_CSV_BYTES:
        # 413 Payload Too Large, not 400: the upload isn't malformed — it's
        # simply too big, which is exactly what 413 means. Checked before any
        # decode/parse work so an oversized file never reaches csv.DictReader.
        raise HTTPException(
            status_code=413,
            detail=f"CSV file exceeds {MAX_CSV_BYTES // (1024 * 1024)}MB limit",
        )
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
        t = (row.get("text") or "").strip()
        if len(t) > MAX_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Row {i + 2} exceeds {MAX_CHARS} characters",
            )
        if t:
            if len(texts) >= MAX_BATCH:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV exceeds {MAX_BATCH} non-empty row limit",
                )
            texts.append(t)

    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty rows found")

    results = model.predict(texts)
    items = [{"text": t, **r} for t, r in zip(texts, results)]
    return {"results": items, "aggregates": aggregate(results)}


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


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest, request: Request):
    """Run one input through several sentiment models and return them side by side.

    This endpoint is intentionally plain: it teaches domain mismatch, label
    mismatch, confidence, and latency tradeoffs better than another chart. Each
    row carries the model's own label keys (binary models never fake a neutral),
    its winning-class confidence, and a per-model wall-clock latency.
    """
    model_ids = req.model_ids or DEFAULT_COMPARE_MODELS

    # Public-deployment allowlist first: a disabled model is rejected before
    # get_or_load_model can ever spend memory/bandwidth on its weights.
    for model_id in model_ids:
        reject_disabled_model(model_id)

    # Validate every id BEFORE loading any weights — fail fast at the boundary
    # so a bad id never costs a multi-hundred-MB load first.
    configs = []
    for model_id in model_ids:
        try:
            cfg = get_model_config(model_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model_id '{model_id}'. See /api/models for sentiment models.",
            )
        if cfg.task != ModelTask.SENTIMENT:
            raise HTTPException(
                status_code=400,
                detail=f"'{model_id}' is a {cfg.task.value} model; "
                "/api/compare accepts sentiment models only.",
            )
        configs.append((model_id, cfg))

    results = []
    for model_id, cfg in configs:
        model = await get_or_load_model(request.app, model_id)
        # perf_counter is monotonic — the correct clock for measuring a duration.
        start = time.perf_counter()
        prediction = model.predict([req.text])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        scores = prediction["scores"]
        results.append(
            {
                "model_id": model_id,
                "name": cfg.name,
                "domain": cfg.domain,
                "label": prediction["label"],
                "scores": scores,
                # Confidence is just the winning class probability. No extra
                # rounding — predict() already rounds scores to 4 decimals.
                "confidence": max(scores.values()),
                "latency_ms": latency_ms,
                "note": cfg.note,
            }
        )
    return {"results": results}


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

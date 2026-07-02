"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

import asyncio
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
    AiDetectCompareResponse,
    AiDetectRequest,
    AiDetectResponse,
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
DEFAULT_DETECTOR_MODEL_ID = get_default_model_id(ModelTask.AI_TEXT_DETECTION)
# Attached VERBATIM to every detector response. AI detection is probabilistic;
# the product must never let a score read as proof of authorship.
DETECTOR_WARNING = (
    "AI detectors are probabilistic and can be wrong, especially on short, "
    "edited, non-native, highly formal, or mixed-authorship text. Do not use "
    "this as proof of authorship."
)


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
    # Bounded read: pull at most MAX_CSV_BYTES + 1 bytes off the wire, never the
    # whole upload. The +1 byte is all the size check below needs to fire, so the
    # guard can't be defeated by a body larger than RAM — an attacker streaming a
    # multi-GB file never gets more than ~5MB buffered here.
    raw = await file.read(MAX_CSV_BYTES + 1)
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

    # Inference is CPU-bound and blocking; run it off the event loop so
    # concurrent requests stay responsive.
    results = await asyncio.to_thread(model.predict, texts)
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

    # Validate identity BEFORE the allowlist: an unknown id must return a
    # truthful 400, not masquerade as a 403 "disabled". Every id is checked
    # BEFORE loading any weights, so a bad id never costs a multi-hundred-MB load.
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

    # Public-deployment allowlist next: a known-but-disabled model is rejected
    # before get_or_load_model can ever spend memory/bandwidth on its weights.
    for model_id, _ in configs:
        reject_disabled_model(model_id)

    results = []
    for model_id, cfg in configs:
        model = await get_or_load_model(request.app, model_id)
        # Inference is CPU-bound and blocking; run it off the event loop (like
        # the lazy load() in get_or_load_model) so concurrent requests stay
        # responsive. perf_counter is monotonic — the right clock for a duration
        # — and times the predict call itself (the thread hand-off is sub-ms
        # noise; the expensive one-time load already happened before this loop).
        start = time.perf_counter()
        prediction = (await asyncio.to_thread(model.predict, [req.text]))[0]
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


async def _score_detectors(request: Request, text: str, model_ids: list[str]) -> list[dict]:
    """Validate, load, and score a set of AI text detectors on one input.

    Shared by /api/ai-detect and /api/ai-detect/compare so both apply the same
    guards in the same order: known id (400) → detector-only (400) → allowlist
    (403), ALL before any weights load. Identity is checked before the allowlist
    so an unknown id returns a truthful 400 instead of masquerading as a 403
    "disabled". Sentiment ids are refused here — mixing the two task families is
    the one thing this endpoint exists to prevent.
    """
    configs = []
    for model_id in model_ids:
        try:
            cfg = get_model_config(model_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model_id '{model_id}'. "
                "See /api/models?task=ai_text_detection for detector models.",
            )
        if cfg.task != ModelTask.AI_TEXT_DETECTION:
            raise HTTPException(
                status_code=400,
                detail=f"'{model_id}' is a {cfg.task.value} model; "
                "the AI detection endpoints accept AI text detectors only.",
            )
        configs.append((model_id, cfg))

    # Public-deployment allowlist last: a known-but-disabled detector is rejected
    # before get_or_load_model can ever spend memory/bandwidth on its weights.
    for model_id, _ in configs:
        reject_disabled_model(model_id)

    rows = []
    for model_id, cfg in configs:
        model = await get_or_load_model(request.app, model_id)
        # Inference is CPU-bound and blocking; run it off the event loop so
        # concurrent requests stay responsive. Latency times the predict call
        # (thread hand-off is sub-ms noise; the one-time load already happened).
        start = time.perf_counter()
        prediction = (await asyncio.to_thread(model.predict, [text]))[0]
        latency_ms = (time.perf_counter() - start) * 1000
        scores = prediction["scores"]
        rows.append(
            {
                "model_id": model_id,
                "name": cfg.name,
                "domain": cfg.domain,
                "label": prediction["label"],
                "scores": scores,
                "confidence": max(scores.values()),
                "latency_ms": latency_ms,
                "note": cfg.note,
            }
        )
    return rows


@router.post("/ai-detect", response_model=AiDetectResponse)
async def ai_detect(req: AiDetectRequest, request: Request):
    """Score one text with a single AI detector (desklib by default). A
    model_ids override picks a different detector; only the first is used since
    this endpoint returns exactly one result — use /api/ai-detect/compare for
    several. Every response carries the uncertainty warning."""
    model_id = (req.model_ids or [DEFAULT_DETECTOR_MODEL_ID])[0]
    rows = await _score_detectors(request, req.text, [model_id])
    return {"result": rows[0], "warning": DETECTOR_WARNING}


@router.post("/ai-detect/compare", response_model=AiDetectCompareResponse)
async def ai_detect_compare(req: AiDetectRequest, request: Request):
    """Run one text through several AI detectors side by side. Defaults to ALL
    detectors so users can see them (dis)agree — the teaching point of the tab.
    disagreement is true when the detectors don't all land on the same label."""
    model_ids = req.model_ids or list(models_for_task(ModelTask.AI_TEXT_DETECTION))
    rows = await _score_detectors(request, req.text, model_ids)
    disagreement = len({r["label"] for r in rows}) > 1
    return {"results": rows, "disagreement": disagreement, "warning": DETECTOR_WARNING}


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

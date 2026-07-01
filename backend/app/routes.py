"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.model import SentimentModel
from app.model_registry import (
    MODEL_REGISTRY,
    ModelTask,
    get_default_model_id,
    models_for_task,
)
from app.schemas import (
    MAX_BATCH,
    MAX_CHARS,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    ExplainResponse,
)

router = APIRouter(prefix="/api")

LABELS = ("negative", "neutral", "positive")
EXPLAIN_MODEL_ID = "twitter-roberta"
DEFAULT_SENTIMENT_MODEL_ID = get_default_model_id(ModelTask.SENTIMENT)


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

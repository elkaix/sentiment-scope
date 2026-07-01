"""API routes. Endpoints stay thin: validation lives in schemas (Pydantic),
ML logic lives in SentimentModel — routes just wire the two together."""

import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.model import SentimentModel
from app.schemas import (
    MAX_BATCH,
    MAX_CHARS,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
)

router = APIRouter(prefix="/api")

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


@router.post("/analyze/batch", response_model=BatchResponse)
def analyze_batch(req: BatchRequest, model: SentimentModel = Depends(get_model)):
    # ONE batched model call for the whole list — see predict()'s docstring
    # for why that beats a per-text loop.
    results = model.predict(req.texts)
    items = [{"text": t, **r} for t, r in zip(req.texts, results)]
    return {"results": items, "aggregates": aggregate(results)}


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

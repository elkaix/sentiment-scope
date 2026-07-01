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

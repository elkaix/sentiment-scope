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

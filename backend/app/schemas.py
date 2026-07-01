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

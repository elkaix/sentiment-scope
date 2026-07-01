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
        note=(
            "Default detector. Custom DeBERTa-v3 arch (single sigmoid logit) — loads "
            "via DetectorModel, never AutoModelForSequenceClassification."
        ),
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

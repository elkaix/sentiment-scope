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


class FakeBinaryModel:
    """Binary sentiment stand-in (DistilBERT-SST2 shape): negative/positive
    only, no neutral. Lets compare tests prove that a 2-class model emits just
    its own label keys — never a faked neutral score."""

    is_loaded = True
    device = "cpu"
    labels = ["negative", "positive"]

    def predict(self, texts):
        return [
            {"label": "negative", "scores": {"negative": 0.7, "positive": 0.3}}
            for _ in texts
        ]


@pytest.fixture
def client_with_compare_models(client):
    """Seed the lazy cache with fakes so /api/compare runs without real weights.
    get_or_load_model checks app.state.model_cache first, so a pre-seeded entry
    short-circuits the ~500MB load — no torch, still exercises the cache path."""
    client.app.state.model_cache["twitter-roberta"] = FakeModel()
    client.app.state.model_cache["distilbert-sst2"] = FakeBinaryModel()
    return client


class FakeDetector:
    """AI-detector stand-in: canonical {human, ai} scores, canned label. Lets
    the detector API tests prove wiring/validation/disagreement without torch —
    real label mapping and probabilities are covered by the integration suite."""

    is_loaded = True
    device = "cpu"
    labels = ["human", "ai"]

    def __init__(self, label: str = "ai", scores: dict | None = None) -> None:
        self._label = label
        self._scores = scores or {"human": 0.1, "ai": 0.9}

    def predict(self, texts):
        return [{"label": self._label, "scores": self._scores} for _ in texts]


@pytest.fixture
def client_with_detectors(client):
    """Seed the cache with three fake detectors. desklib + fakespot call it AI,
    oxidane calls it human — so the default (all-detector) compare disagrees."""
    cache = client.app.state.model_cache
    cache["desklib-ai-detector"] = FakeDetector("ai", {"human": 0.05, "ai": 0.95})
    cache["fakespot-ai-detector"] = FakeDetector("ai", {"human": 0.2, "ai": 0.8})
    cache["oxidane-ai-detector"] = FakeDetector("human", {"human": 0.7, "ai": 0.3})
    return client

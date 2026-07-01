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

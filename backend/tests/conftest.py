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

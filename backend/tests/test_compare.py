"""API tests for POST /api/compare.

Unit tests (SKIP_MODEL_LOAD=1) — no torch. The compare endpoint reads models
from app.state.model_cache via get_or_load_model, so client_with_compare_models
pre-seeds that cache with fakes: a 3-class FakeModel for twitter-roberta and a
2-class FakeBinaryModel for distilbert-sst2. Behavior is checked over HTTP only.
"""


def test_compare_defaults_to_two_models(client_with_compare_models):
    resp = client_with_compare_models.post("/api/compare", json={"text": "I love this"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["model_id"] for r in results] == ["twitter-roberta", "distilbert-sst2"]


def test_binary_model_returns_only_its_own_label_keys(client_with_compare_models):
    resp = client_with_compare_models.post("/api/compare", json={"text": "meh"})
    by_id = {r["model_id"]: r for r in resp.json()["results"]}
    # Binary DistilBERT has no neutral class — its scores must not fake one.
    assert set(by_id["distilbert-sst2"]["scores"]) == {"negative", "positive"}
    # The 3-class model keeps all three keys.
    assert set(by_id["twitter-roberta"]["scores"]) == {"negative", "neutral", "positive"}


def test_confidence_equals_max_score(client_with_compare_models):
    resp = client_with_compare_models.post("/api/compare", json={"text": "good"})
    for r in resp.json()["results"]:
        assert r["confidence"] == max(r["scores"].values())


def test_latency_is_numeric_and_non_negative(client_with_compare_models):
    resp = client_with_compare_models.post("/api/compare", json={"text": "good"})
    for r in resp.json()["results"]:
        assert isinstance(r["latency_ms"], (int, float))
        assert r["latency_ms"] >= 0


def test_explicit_model_ids_are_honored(client_with_compare_models):
    resp = client_with_compare_models.post(
        "/api/compare", json={"text": "good", "model_ids": ["distilbert-sst2"]}
    )
    results = resp.json()["results"]
    assert [r["model_id"] for r in results] == ["distilbert-sst2"]


def test_row_carries_registry_metadata(client_with_compare_models):
    resp = client_with_compare_models.post(
        "/api/compare", json={"text": "good", "model_ids": ["twitter-roberta"]}
    )
    row = resp.json()["results"][0]
    # name/domain/note come straight from the registry ModelConfig.
    assert row["name"] == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    assert row["domain"] == "social / short English text"
    assert row["note"]


def test_rejects_non_sentiment_model_id(client_with_compare_models):
    resp = client_with_compare_models.post(
        "/api/compare", json={"text": "good", "model_ids": ["desklib-ai-detector"]}
    )
    assert resp.status_code == 400
    assert "sentiment" in resp.json()["detail"].lower()


def test_rejects_unknown_model_id(client_with_compare_models):
    resp = client_with_compare_models.post(
        "/api/compare", json={"text": "good", "model_ids": ["not-a-real-model"]}
    )
    assert resp.status_code == 400
    assert "not-a-real-model" in resp.json()["detail"]


def test_compare_rejects_disabled_model(monkeypatch, client_with_model):
    # ENABLED_MODELS is the public-deployment allowlist (Task 16A): the free
    # Space must not let anonymous users lazy-load every registry model. The
    # guard runs BEFORE get_or_load_model, so no fake cache entry is needed.
    monkeypatch.setenv("ENABLED_MODELS", "twitter-roberta")
    resp = client_with_model.post(
        "/api/compare",
        json={"text": "great", "model_ids": ["distilbert-sst2"]},
    )
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_compare_allows_models_on_the_allowlist(monkeypatch, client_with_compare_models):
    monkeypatch.setenv("ENABLED_MODELS", "twitter-roberta,distilbert-sst2")
    resp = client_with_compare_models.post("/api/compare", json={"text": "great"})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_lifespan_seeds_default_model_into_cache(monkeypatch):
    """The startup-loaded default must be seeded into model_cache so /api/compare
    reuses that one copy instead of loading a second ~500MB set of weights."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model import SentimentModel

    # Fake a successful load without torch: is_loaded checks self._model.
    monkeypatch.setenv("SKIP_MODEL_LOAD", "0")
    monkeypatch.setattr(SentimentModel, "load", lambda self: setattr(self, "_model", object()))

    with TestClient(app) as c:
        cache = c.app.state.model_cache
        # Same object, not a second copy of the weights.
        assert cache["twitter-roberta"] is c.app.state.model

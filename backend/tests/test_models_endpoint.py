"""API tests for the model registry endpoint and default-endpoint guarding.

These are unit tests (SKIP_MODEL_LOAD=1) — no torch, no real weights. They
check wiring: the /api/models shape the frontend expects, the ?task= filter,
and that analyze/batch/csv reject a stray non-default model_id.
"""


def test_models_endpoint_returns_wrapped_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    # Frontend getModels() (Task 11) expects the list wrapped under "models".
    assert "models" in body
    ids = {m["id"] for m in body["models"]}
    assert ids == {
        "twitter-roberta",
        "distilbert-sst2",
        "finbert",
        "xlm-twitter",
        "desklib-ai-detector",
        "fakespot-ai-detector",
        "oxidane-ai-detector",
    }
    # Public metadata is present; internal local_path is not leaked.
    first = body["models"][0]
    assert {"id", "name", "task", "labels", "domain", "note", "default", "loaded"} <= first.keys()
    assert "local_path" not in first


def test_models_endpoint_filters_to_sentiment(client):
    resp = client.get("/api/models?task=sentiment")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert {m["id"] for m in models} == {
        "twitter-roberta",
        "distilbert-sst2",
        "finbert",
        "xlm-twitter",
    }
    # StrEnum must serialize as its value, not "ModelTask.SENTIMENT".
    assert all(m["task"] == "sentiment" for m in models)
    default_ids = [m["id"] for m in models if m["default"]]
    assert default_ids == ["twitter-roberta"]


def test_models_endpoint_filters_to_detectors(client):
    resp = client.get("/api/models?task=ai_text_detection")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert {m["id"] for m in models} == {
        "desklib-ai-detector",
        "fakespot-ai-detector",
        "oxidane-ai-detector",
    }
    assert all(m["task"] == "ai_text_detection" for m in models)


def test_models_endpoint_rejects_unknown_task(client):
    resp = client.get("/api/models?task=not-a-task")
    assert resp.status_code == 422


def test_models_endpoint_reports_loaded_flag(client):
    # A model shows loaded=true iff it is in the lazy cache. Membership is all
    # the flag checks, so a bare sentinel is enough to simulate a loaded model.
    client.app.state.model_cache["distilbert-sst2"] = object()
    models = client.get("/api/models").json()["models"]
    by_id = {m["id"]: m for m in models}
    assert by_id["distilbert-sst2"]["loaded"] is True
    assert by_id["finbert"]["loaded"] is False


def test_analyze_accepts_default_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/analyze?model_id=twitter-roberta", json={"text": "I love this"}
    )
    assert resp.status_code == 200


def test_analyze_rejects_non_default_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/analyze?model_id=distilbert-sst2", json={"text": "I love this"}
    )
    assert resp.status_code == 400
    assert "compare" in resp.json()["detail"].lower()


def test_batch_rejects_non_default_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/analyze/batch?model_id=finbert", json={"texts": ["a", "b"]}
    )
    assert resp.status_code == 400


def test_csv_rejects_non_default_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/analyze/csv?model_id=finbert",
        files={"file": ("data.csv", "text\nhello\n", "text/csv")},
    )
    assert resp.status_code == 400

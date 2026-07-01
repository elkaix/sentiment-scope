def test_analyze_returns_label_and_scores(client_with_model):
    resp = client_with_model.post("/api/analyze", json={"text": "I love this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["scores"]["positive"] == 0.8


def test_analyze_blank_text_is_422(client_with_model):
    resp = client_with_model.post("/api/analyze", json={"text": "  "})
    assert resp.status_code == 422


def test_analyze_without_model_is_503(client):
    # No dependency override + SKIP_MODEL_LOAD → model never loaded → 503.
    resp = client.post("/api/analyze", json={"text": "hello"})
    assert resp.status_code == 503

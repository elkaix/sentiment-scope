def test_health_reports_model_not_loaded(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # SKIP_MODEL_LOAD=1 in tests

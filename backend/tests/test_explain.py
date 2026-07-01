def test_explain_returns_token_attributions(client_with_model):
    resp = client_with_model.post("/api/explain", json={"text": "I love this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    tokens = body["tokens"]
    assert tokens[1]["token"] == " love"
    assert tokens[1]["attribution"] > tokens[0]["attribution"]


def test_model_info(client_with_model):
    resp = client_with_model.get("/api/model")
    assert resp.status_code == 200
    body = resp.json()
    assert "roberta" in body["name"]
    assert body["labels"] == ["negative", "neutral", "positive"]
    assert body["max_tokens"] == 512


def test_explain_rejects_non_roberta_model_id(client_with_model):
    resp = client_with_model.post(
        "/api/explain?model_id=distilbert-sst2", json={"text": "I love this"}
    )
    assert resp.status_code == 400
    assert "twitter-roberta" in resp.json()["detail"].lower()

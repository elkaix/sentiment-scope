from app.routes import aggregate


def test_aggregate_counts_and_means():
    results = [
        {"label": "positive", "scores": {"negative": 0.1, "neutral": 0.1, "positive": 0.8}},
        {"label": "negative", "scores": {"negative": 0.7, "neutral": 0.2, "positive": 0.1}},
    ]
    agg = aggregate(results)
    assert agg["counts"] == {"negative": 1, "neutral": 0, "positive": 1}
    assert agg["mean_scores"]["positive"] == 0.45
    assert agg["mean_scores"]["negative"] == 0.4


def test_batch_endpoint_returns_items_and_aggregates(client_with_model):
    resp = client_with_model.post("/api/analyze/batch", json={"texts": ["a good day", "another"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["text"] == "a good day"
    assert body["aggregates"]["counts"]["positive"] == 2


def test_batch_rejects_501_texts(client_with_model):
    resp = client_with_model.post("/api/analyze/batch", json={"texts": ["x"] * 501})
    assert resp.status_code == 422

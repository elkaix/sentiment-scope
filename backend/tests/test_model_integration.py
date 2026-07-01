"""Real-model tests. Excluded by default (see pyproject addopts); run with:
    pytest -m integration -v
Requires the `ai` conda env (torch + transformers installed, weights cached).
"""

import pytest

from app.model_registry import ModelTask, models_for_task


@pytest.mark.integration
def test_predict_on_obvious_sentiment():
    from app.model import SentimentModel

    m = SentimentModel()
    m.load()
    out = m.predict(["I love this so much!", "This is absolutely terrible."])

    assert out[0]["label"] == "positive"
    assert out[1]["label"] == "negative"
    for r in out:
        # Probabilities must behave like probabilities.
        assert abs(sum(r["scores"].values()) - 1.0) < 0.01
        assert set(r["scores"]) == {"negative", "neutral", "positive"}


@pytest.mark.integration
def test_explain_highlights_sentiment_words():
    from app.model import SentimentModel

    m = SentimentModel()
    m.load()
    out = m.explain("I absolutely love this phone")

    assert out["label"] == "positive"
    attrs = {t["token"].strip().lower(): t["attribution"] for t in out["tokens"]}
    # "love" should drive the positive prediction more than the stopword "this".
    assert attrs["love"] > attrs["this"]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", list(models_for_task(ModelTask.SENTIMENT).keys()))
def test_registry_model_score_keys_match_config(model_id):
    from app.model import SentimentModel
    from app.model_registry import get_model_config

    cfg = get_model_config(model_id)
    m = SentimentModel(cfg)
    m.load()
    out = m.predict(["This is good."])[0]
    assert tuple(out["scores"].keys()) == cfg.labels

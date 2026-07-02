"""Registry + lazy-loading unit tests.

Pure unit tests: no torch, no transformers. Everything here imports only the
lightweight registry and the (lazy-import) model module, so CI can run it
without the ~500MB ML stack installed.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.model_registry import (
    ModelConfig,
    ModelTask,
    get_default_model_id,
    get_model_config,
    models_for_task,
    resolve_model_source,
)


def test_resolve_model_source_falls_back_to_hub_name_when_local_path_missing():
    cfg = ModelConfig(
        id="x",
        name="hf/name",
        task=ModelTask.SENTIMENT,
        labels=("negative", "positive"),
        domain="test",
        note="test",
        local_path="models/does-not-exist",
    )
    assert resolve_model_source(cfg) == "hf/name"


def test_resolve_model_source_uses_local_dir_when_present():
    # twitter-roberta weights exist locally in this repo; the resolver must
    # prefer the on-disk copy (an absolute path ending in the local_path).
    cfg = get_model_config("twitter-roberta")
    source = resolve_model_source(cfg)
    assert source.endswith("models/twitter-roberta-base-sentiment-latest")
    assert source != cfg.name


def test_default_sentiment_model_is_twitter_roberta():
    assert get_default_model_id(ModelTask.SENTIMENT) == "twitter-roberta"


def test_default_detector_is_desklib():
    assert get_default_model_id(ModelTask.AI_TEXT_DETECTION) == "desklib-ai-detector"


def test_models_for_task_sentiment_returns_only_sentiment():
    sentiment = models_for_task(ModelTask.SENTIMENT)
    assert set(sentiment) == {"twitter-roberta", "distilbert-sst2", "finbert", "xlm-twitter"}
    assert all(c.task == ModelTask.SENTIMENT for c in sentiment.values())


def test_models_for_task_detection_returns_only_detectors():
    detectors = models_for_task(ModelTask.AI_TEXT_DETECTION)
    assert set(detectors) == {
        "desklib-ai-detector",
        "fakespot-ai-detector",
        "oxidane-ai-detector",
    }
    assert all(c.task == ModelTask.AI_TEXT_DETECTION for c in detectors.values())


def test_get_model_config_defaults_to_sentiment_default():
    assert get_model_config().id == "twitter-roberta"


def test_get_model_config_unknown_raises_value_error():
    with pytest.raises(ValueError):
        get_model_config("does-not-exist")


def test_build_model_sentiment_returns_sentiment_model():
    from app.model import SentimentModel, build_model

    m = build_model(get_model_config("finbert"))
    assert isinstance(m, SentimentModel)
    # Labels come from the registry, not raw config.id2label — available even
    # before load() (no torch touched here).
    assert m.labels == ["positive", "negative", "neutral"]


def test_build_model_detector_returns_detector_model():
    from app.model import DetectorModel, build_model

    m = build_model(get_model_config("desklib-ai-detector"))
    assert isinstance(m, DetectorModel)
    # Canonical labels come from the registry — available before load() (no
    # torch touched here), exactly like the sentiment path.
    assert m.labels == ["human", "ai"]


def test_get_or_load_model_lazy_loads_and_caches_once(monkeypatch):
    """A cache miss loads exactly once; a second call is a pure cache hit."""
    from app import model as model_module

    load_count = {"n": 0}

    class _FakeModel:
        labels: list = []
        device = None
        is_loaded = False

        def __init__(self, cfg):
            self.cfg = cfg

        def load(self):
            load_count["n"] += 1
            self.is_loaded = True

        def predict(self, texts):
            return []

    monkeypatch.setattr(model_module, "build_model", lambda cfg: _FakeModel(cfg))

    app = SimpleNamespace(state=SimpleNamespace(model_cache={}, model_locks={}))

    first = asyncio.run(model_module.get_or_load_model(app, "distilbert-sst2"))
    assert first.is_loaded
    assert load_count["n"] == 1
    assert "distilbert-sst2" in app.state.model_cache

    second = asyncio.run(model_module.get_or_load_model(app, "distilbert-sst2"))
    assert second is first
    assert load_count["n"] == 1  # not reloaded

"""AI text detection endpoints (Task 19).

Unit tests (SKIP_MODEL_LOAD=1, no torch): the detector routes read models from
app.state.model_cache via get_or_load_model, so client_with_detectors pre-seeds
that cache with FakeDetector instances. Wiring, validation, and disagreement are
checked over HTTP only.

Integration tests (@pytest.mark.integration, real weights, `ai` conda env)
exercise the two output adapters end to end: desklib's custom single-logit
sigmoid head and the softmax detectors' canonical label mapping.
"""

import pytest

from app.model_registry import ModelTask, models_for_task
from app.routes import DETECTOR_WARNING

# The exact warning the product promises on every detector response. A literal
# copy here guards against an accidental reword drifting from the spec.
EXPECTED_WARNING = (
    "AI detectors are probabilistic and can be wrong, especially on short, "
    "edited, non-native, highly formal, or mixed-authorship text. "
    "Do not use this as proof of authorship."
)


def test_warning_constant_matches_spec():
    # Guard the source constant itself, not just the response, against a reword.
    assert DETECTOR_WARNING == EXPECTED_WARNING


# --- /api/ai-detect -------------------------------------------------------


def test_ai_detect_returns_result_and_warning(client_with_detectors):
    resp = client_with_detectors.post("/api/ai-detect", json={"text": "some text"})
    assert resp.status_code == 200
    body = resp.json()
    result = body["result"]
    assert result["model_id"] == "desklib-ai-detector"
    assert result["label"] in {"human", "ai"}
    assert set(result["scores"]) == {"human", "ai"}
    assert result["confidence"] == max(result["scores"].values())
    assert body["warning"] == EXPECTED_WARNING


def test_ai_detect_defaults_to_desklib(client_with_detectors):
    resp = client_with_detectors.post("/api/ai-detect", json={"text": "x"})
    assert resp.json()["result"]["model_id"] == "desklib-ai-detector"


def test_ai_detect_honors_explicit_detector(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/ai-detect", json={"text": "x", "model_ids": ["fakespot-ai-detector"]}
    )
    assert resp.json()["result"]["model_id"] == "fakespot-ai-detector"


def test_ai_detect_row_carries_registry_metadata(client_with_detectors):
    result = client_with_detectors.post("/api/ai-detect", json={"text": "x"}).json()["result"]
    assert result["name"] == "desklib/ai-text-detector-v1.01"
    assert result["domain"]
    assert result["note"]
    assert isinstance(result["latency_ms"], (int, float))
    assert result["latency_ms"] >= 0


def test_ai_detect_rejects_sentiment_model(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/ai-detect", json={"text": "x", "model_ids": ["twitter-roberta"]}
    )
    assert resp.status_code == 400
    assert "sentiment" in resp.json()["detail"].lower()


def test_ai_detect_rejects_unknown_model(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/ai-detect", json={"text": "x", "model_ids": ["not-a-real-model"]}
    )
    assert resp.status_code == 400
    assert "not-a-real-model" in resp.json()["detail"]


def test_ai_detect_rejects_disabled_model(monkeypatch, client):
    # ENABLED_MODELS is the public-deploy allowlist. The guard runs BEFORE
    # get_or_load_model, so no seeded cache entry is needed.
    monkeypatch.setenv("ENABLED_MODELS", "oxidane-ai-detector")
    resp = client.post(
        "/api/ai-detect", json={"text": "x", "model_ids": ["desklib-ai-detector"]}
    )
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


# --- /api/ai-detect/compare ----------------------------------------------


def test_compare_defaults_to_all_detectors(client_with_detectors):
    resp = client_with_detectors.post("/api/ai-detect/compare", json={"text": "x"})
    assert resp.status_code == 200
    ids = {r["model_id"] for r in resp.json()["results"]}
    assert ids == set(models_for_task(ModelTask.AI_TEXT_DETECTION))


def test_compare_disagreement_true_when_labels_differ(client_with_detectors):
    # Seeded fakes: desklib+fakespot say "ai", oxidane says "human" → disagree.
    body = client_with_detectors.post("/api/ai-detect/compare", json={"text": "x"}).json()
    assert body["disagreement"] is True
    assert body["warning"] == EXPECTED_WARNING


def test_compare_disagreement_false_when_labels_agree(client_with_detectors):
    body = client_with_detectors.post(
        "/api/ai-detect/compare",
        json={"text": "x", "model_ids": ["desklib-ai-detector", "fakespot-ai-detector"]},
    ).json()
    assert [r["label"] for r in body["results"]] == ["ai", "ai"]
    assert body["disagreement"] is False


def test_compare_honors_explicit_model_ids(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/ai-detect/compare",
        json={"text": "x", "model_ids": ["oxidane-ai-detector"]},
    )
    assert [r["model_id"] for r in resp.json()["results"]] == ["oxidane-ai-detector"]


def test_compare_rejects_sentiment_model(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/ai-detect/compare", json={"text": "x", "model_ids": ["finbert"]}
    )
    assert resp.status_code == 400
    assert "sentiment" in resp.json()["detail"].lower()


def test_compare_dedupes_duplicate_detector_ids(client_with_detectors):
    # Same dedupe guard as /api/compare: a repeated detector id collapses to one
    # row, so one request can't queue the same detector many times over.
    resp = client_with_detectors.post(
        "/api/ai-detect/compare",
        json={"text": "x", "model_ids": ["desklib-ai-detector", "desklib-ai-detector"]},
    )
    assert resp.status_code == 200
    assert [r["model_id"] for r in resp.json()["results"]] == ["desklib-ai-detector"]


# --- cross-task rejection: detectors are refused by /api/compare ----------


def test_sentiment_compare_rejects_detector_model(client_with_detectors):
    resp = client_with_detectors.post(
        "/api/compare", json={"text": "x", "model_ids": ["desklib-ai-detector"]}
    )
    assert resp.status_code == 400
    assert "sentiment" in resp.json()["detail"].lower()


# --- integration: real weights, real adapters ----------------------------

# From the desklib model card — an obviously AI-ish paragraph vs a terser,
# human-ish note. Labels on borderline text are noisy, so integration tests
# assert shape (keys, sum≈1, valid label), not a specific verdict.
AI_TEXT = (
    "AI detection refers to the process of identifying whether a given piece of "
    "content, such as text, images, or audio, has been generated by artificial "
    "intelligence. This is achieved using various machine learning techniques, "
    "including perplexity analysis, entropy measurements, linguistic pattern "
    "recognition, and neural network classifiers trained on human and AI-generated "
    "data. Advanced AI detection tools assess writing style, coherence, and "
    "statistical properties to determine the likelihood of AI involvement."
)
HUMAN_TEXT = (
    "It is estimated that a major part of the content in the internet will be "
    "generated by AI / LLMs by 2025. This leads to a lot of misinformation and "
    "credibility related issues. That is why if is important to have accurate "
    "tools to identify if a content is AI generated or human written"
)


def _assert_detector_output(row: dict) -> None:
    assert set(row["scores"]) == {"human", "ai"}
    assert abs(sum(row["scores"].values()) - 1.0) < 0.01
    assert row["label"] in {"human", "ai"}


@pytest.mark.integration
def test_desklib_loads_via_custom_class_and_scores(capsys):
    from app.model import DetectorModel
    from app.model_registry import get_model_config

    m = DetectorModel(get_model_config("desklib-ai-detector"))
    m.load()
    out = m.predict([AI_TEXT, HUMAN_TEXT])
    for row in out:
        _assert_detector_output(row)
    with capsys.disabled():
        print(f"\n[desklib] P(ai) ai_text={out[0]['scores']['ai']} "
              f"human_text={out[1]['scores']['ai']}")


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["fakespot-ai-detector", "oxidane-ai-detector"])
def test_softmax_detectors_canonical_labels(model_id, capsys):
    from app.model import DetectorModel
    from app.model_registry import get_model_config

    m = DetectorModel(get_model_config(model_id))
    m.load()
    # Canonical labels were derived from the checkpoint's own id2label.
    assert set(m._softmax_labels) == {"human", "ai"}
    out = m.predict([AI_TEXT, HUMAN_TEXT])
    for row in out:
        _assert_detector_output(row)
    with capsys.disabled():
        print(f"\n[{model_id}] P(ai) ai_text={out[0]['scores']['ai']} "
              f"human_text={out[1]['scores']['ai']}")

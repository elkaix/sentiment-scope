"""Unit tests for the pure (torch-free, app-free) eval helpers.

These import only ``run_eval`` — never ``app`` or torch — because the heavy
model imports live inside ``run_eval.run()``. That keeps this suite fast and
lets it run anywhere sklearn is installed.
"""

import pytest

import run_eval


# --- Slice A: label-compatibility guard (review fix #12) ---------------------


def test_label_compat_subset_passes():
    # 3-class dataset against a 3-class model: nothing to complain about.
    run_eval.validate_label_compatibility(
        {"positive", "negative"}, {"negative", "neutral", "positive"}, allow_mismatch=False
    )


def test_label_compat_mismatch_raises_system_exit():
    # `neutral` cannot be predicted by a binary model -> hard fail by default.
    with pytest.raises(SystemExit) as exc:
        run_eval.validate_label_compatibility(
            {"negative", "neutral", "positive"}, {"negative", "positive"}, allow_mismatch=False
        )
    assert "neutral" in str(exc.value)
    assert "--allow-label-mismatch" in str(exc.value)


def test_label_compat_mismatch_allowed_warns_and_returns(capsys):
    run_eval.validate_label_compatibility(
        {"negative", "neutral", "positive"}, {"negative", "positive"}, allow_mismatch=True
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "neutral" in out


# --- Slice B: CSV parsing + row validation -----------------------------------

_GOOD_CSV = (
    "id,text,true_label,category,notes\n"
    "1,The battery life is incredible,positive,product_review,clear positive\n"
    '2,"Not bad, not great",neutral,ambiguous,mixed phrase\n'
)


def test_parse_eval_rows_reads_all_fields():
    rows = run_eval.parse_eval_rows(_GOOD_CSV)
    assert len(rows) == 2
    assert rows[0].id == "1"
    assert rows[0].text == "The battery life is incredible"
    assert rows[0].true_label == "positive"
    assert rows[0].category == "product_review"
    # Quoted field with an embedded comma survives intact.
    assert rows[1].text == "Not bad, not great"


def test_parse_eval_rows_missing_required_column_raises():
    bad = "id,text,category\n1,hello,greeting\n"  # no true_label
    with pytest.raises(ValueError) as exc:
        run_eval.parse_eval_rows(bad)
    assert "true_label" in str(exc.value)


def test_parse_eval_rows_rejects_overlong_text_without_truncating():
    long_text = "x" * (run_eval.MAX_TEXT_CHARS + 1)
    bad = f"id,text,true_label\n1,{long_text},positive\n"
    with pytest.raises(ValueError) as exc:
        run_eval.parse_eval_rows(bad)
    msg = str(exc.value)
    assert str(run_eval.MAX_TEXT_CHARS) in msg
    # Error names the offending row so the author can fix it.
    assert "1" in msg


def test_parse_eval_rows_skips_blank_text_rows():
    csv_text = "id,text,true_label\n1,,positive\n2,real,negative\n"
    rows = run_eval.parse_eval_rows(csv_text)
    assert [r.text for r in rows] == ["real"]


def test_parse_eval_rows_defaults_optional_metadata():
    rows = run_eval.parse_eval_rows("id,text,true_label\n1,hi,positive\n")
    assert rows[0].category == ""
    assert rows[0].notes == ""


# --- Slice C: metrics + latency percentiles ----------------------------------

_LABELS = ["negative", "neutral", "positive"]
_Y_TRUE = ["positive", "positive", "negative", "neutral"]
_Y_PRED = ["positive", "negative", "negative", "neutral"]


def test_compute_metrics_accuracy():
    m = run_eval.compute_metrics(_Y_TRUE, _Y_PRED, _LABELS)
    assert m["accuracy"] == pytest.approx(0.75)


def test_compute_metrics_has_macro_f1_in_unit_range():
    m = run_eval.compute_metrics(_Y_TRUE, _Y_PRED, _LABELS)
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_compute_metrics_confusion_matrix_ordered_by_labels():
    m = run_eval.compute_metrics(_Y_TRUE, _Y_PRED, _LABELS)
    # rows = true class, cols = predicted class, both in _LABELS order.
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]


def test_compute_metrics_per_class_support():
    m = run_eval.compute_metrics(_Y_TRUE, _Y_PRED, _LABELS)
    support = {k: v["support"] for k, v in m["per_class"].items()}
    assert support == {"negative": 1, "neutral": 1, "positive": 2}


def test_latency_percentiles_p50_is_median_and_p95_ge_p50():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p = run_eval.latency_percentiles(data)
    assert p["p50"] == pytest.approx(55.0)
    assert p["p95"] >= p["p50"]
    assert p["p95"] <= max(data)


def test_latency_percentiles_single_sample():
    p = run_eval.latency_percentiles([42.0])
    assert p["p50"] == pytest.approx(42.0)
    assert p["p95"] == pytest.approx(42.0)


def test_compute_metrics_handles_class_never_predicted():
    # The --allow-label-mismatch path: a class present in y_true that the model
    # can never emit (e.g. `neutral` under a binary model). zero_division=0 must
    # keep it scoreless instead of raising, and the matrix must stay square so
    # the report renderer (which iterates `labels`) never hits a KeyError.
    labels = ["negative", "positive", "neutral"]
    m = run_eval.compute_metrics(["neutral", "positive"], ["positive", "positive"], labels)
    assert len(m["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in m["confusion_matrix"])
    assert m["per_class"]["neutral"]["support"] == 1
    assert m["per_class"]["neutral"]["precision"] == 0
    assert m["per_class"]["neutral"]["recall"] == 0


# --- Slice D: wrong-example analysis + summary + report ----------------------


def _rows():
    return [
        run_eval.EvalRow("1", "The battery life is incredible", "positive",
                         "product_review", "clear positive"),
        run_eval.EvalRow("4", "Yeah, amazing, another crash", "negative",
                         "sarcasm", "sarcastic negative"),
        run_eval.EvalRow("3", "I love the design but it crashes", "neutral",
                         "mixed", "multi-sentiment"),
    ]


def test_collect_wrong_examples_only_mismatches():
    rows = _rows()
    y_pred = ["positive", "positive", "neutral"]  # row 4 sarcasm misread
    conf = [0.98, 0.91, 0.55]
    wrong = run_eval.collect_wrong_examples(rows, y_pred, conf)
    assert len(wrong) == 1
    w = wrong[0]
    assert w["true"] == "negative"
    assert w["predicted"] == "positive"
    assert w["category"] == "sarcasm"
    assert w["confidence"] == pytest.approx(0.91)
    assert "amazing" in w["text"]


def test_build_summary_shape():
    rows = _rows()
    y_pred = ["positive", "positive", "neutral"]
    conf = [0.98, 0.91, 0.55]
    metrics = run_eval.compute_metrics(
        [r.true_label for r in rows], y_pred, _LABELS
    )
    summary = run_eval.build_summary(
        model_id="twitter-roberta",
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        data_file="evals/data/sentiment_eval.csv",
        rows=rows,
        y_pred=y_pred,
        confidences=conf,
        latencies_ms=[70.0, 72.0, 80.0],
        metrics=metrics,
        labels=_LABELS,
    )
    for key in (
        "model_id", "accuracy", "macro_f1", "latency_p50_ms",
        "latency_p95_ms", "confusion_matrix", "wrong_examples", "labels",
        "n_examples",
    ):
        assert key in summary
    assert summary["n_examples"] == 3
    assert isinstance(summary["latency_p95_ms"], float)


def test_render_report_contains_all_required_sections():
    rows = _rows()
    y_pred = ["positive", "positive", "neutral"]
    conf = [0.98, 0.91, 0.55]
    metrics = run_eval.compute_metrics(
        [r.true_label for r in rows], y_pred, _LABELS
    )
    summary = run_eval.build_summary(
        "twitter-roberta", "cardiffnlp/twitter-roberta", "d.csv",
        rows, y_pred, conf, [70.0, 72.0, 80.0], metrics, _LABELS,
    )
    md = run_eval.render_report(summary)
    # Honest-scope disclaimer (required verbatim theme).
    assert "not creativity judges" in md
    # Metric sections.
    assert "Accuracy" in md and "Macro F1" in md
    assert "Confusion" in md
    assert "p50" in md and "p95" in md
    # Wrong-example table columns.
    for col in ("Text", "Category", "True", "Predicted", "Confidence"):
        assert col in md
    # Top failure modes.
    for mode in ("Sarcasm", "Mixed sentiment", "finance", "context"):
        assert mode in md
    # The sarcasm miss should surface in the table.
    assert "amazing" in md

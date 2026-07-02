"""Unit tests for the torch-free AI-detector eval helpers.

Like ``test_run_eval``, these import only ``run_ai_detect_eval`` — never
``app`` or torch — because the heavy detector imports live inside
``run_ai_detect_eval.run()``. The cross-detector helpers (disagreement,
pairwise agreement, calibration) are the whole reason this module exists, so
they get the most coverage.
"""

import pytest

import run_ai_detect_eval as ade


# --- Slice A: CSV parsing (source_type column, not category) ------------------

_GOOD_CSV = (
    "id,text,true_label,source_type,notes\n"
    "1,ok sounds good,human,human_short,casual\n"
    '2,"As an AI language model, I can help",ai,ai_obvious,self-reference\n'
)


def test_parse_reads_source_type_and_fields():
    rows = ade.parse_detector_rows(_GOOD_CSV)
    assert len(rows) == 2
    assert rows[0].id == "1"
    assert rows[0].text == "ok sounds good"
    assert rows[0].true_label == "human"
    assert rows[0].source_type == "human_short"
    # Quoted field with an embedded comma survives intact.
    assert rows[1].text == "As an AI language model, I can help"
    assert rows[1].source_type == "ai_obvious"


def test_parse_missing_required_column_raises():
    bad = "id,text,source_type\n1,hi,human_short\n"  # no true_label
    with pytest.raises(ValueError) as exc:
        ade.parse_detector_rows(bad)
    assert "true_label" in str(exc.value)


def test_parse_rejects_overlong_text_without_truncating():
    long_text = "x" * (ade.MAX_TEXT_CHARS + 1)
    bad = f"id,text,true_label\n1,{long_text},human\n"
    with pytest.raises(ValueError) as exc:
        ade.parse_detector_rows(bad)
    assert str(ade.MAX_TEXT_CHARS) in str(exc.value)


def test_parse_skips_blank_text_rows():
    csv_text = "id,text,true_label\n1,,human\n2,real text,ai\n"
    rows = ade.parse_detector_rows(csv_text)
    assert [r.text for r in rows] == ["real text"]


def test_parse_defaults_optional_metadata():
    rows = ade.parse_detector_rows("id,text,true_label\n1,hi,human\n")
    assert rows[0].source_type == ""
    assert rows[0].notes == ""


# --- Slice B: disagreement (mirror ai_detect_compare: len({labels}) > 1) ------

_LABELS_BY_DET = {
    # row:      0        1        2        3
    "desklib": ["ai", "human", "ai", "human"],
    "fakespot": ["ai", "ai", "ai", "human"],
    "oxidane": ["ai", "human", "human", "human"],
}


def test_disagreement_flags_per_row():
    d = ade.compute_disagreement(_LABELS_BY_DET)
    # row0 all "ai" -> agree; row1 ai/human/human -> disagree;
    # row2 ai/ai/human -> disagree; row3 all human -> agree.
    assert d["flags"] == [False, True, True, False]


def test_disagreement_rate_is_fraction_of_rows():
    d = ade.compute_disagreement(_LABELS_BY_DET)
    assert d["rate"] == pytest.approx(0.5)


def test_disagreement_single_detector_never_disagrees():
    d = ade.compute_disagreement({"only": ["ai", "human", "ai"]})
    assert d["flags"] == [False, False, False]
    assert d["rate"] == pytest.approx(0.0)


# --- Slice C: pairwise agreement ---------------------------------------------


def test_pairwise_agreement_all_unordered_pairs():
    pairs = ade.pairwise_agreement(_LABELS_BY_DET)
    # 3 detectors -> 3 unordered pairs.
    assert len(pairs) == 3
    lookup = {(p["detector_a"], p["detector_b"]): p["agreement"] for p in pairs}
    # desklib vs fakespot: rows [T,F,T,T] match -> 3/4.
    assert lookup[("desklib", "fakespot")] == pytest.approx(0.75)
    # desklib vs oxidane: rows [T,T,F,T] match -> 3/4.
    assert lookup[("desklib", "oxidane")] == pytest.approx(0.75)
    # fakespot vs oxidane: rows [T,F,F,T] match -> 2/4.
    assert lookup[("fakespot", "oxidane")] == pytest.approx(0.5)


# --- Slice D: calibration hint (mean P(ai) on human rows vs ai rows) ---------


def test_calibration_hint_splits_by_true_label():
    true_labels = ["human", "human", "ai", "ai"]
    p_ai = {"desklib": [0.1, 0.3, 0.8, 0.9]}
    cal = ade.calibration_hint(true_labels, p_ai)
    assert cal["desklib"]["mean_p_ai_on_human"] == pytest.approx(0.2)
    assert cal["desklib"]["mean_p_ai_on_ai"] == pytest.approx(0.85)
    # A well-behaved detector separates the classes -> positive gap.
    assert cal["desklib"]["gap"] == pytest.approx(0.65)


def test_calibration_hint_handles_missing_class():
    # No ai rows in the set -> mean_p_ai_on_ai is None, not a crash.
    cal = ade.calibration_hint(["human", "human"], {"d": [0.2, 0.4]})
    assert cal["d"]["mean_p_ai_on_human"] == pytest.approx(0.3)
    assert cal["d"]["mean_p_ai_on_ai"] is None
    assert cal["d"]["gap"] is None


# --- Slice E: disagreement + wrong-example collectors ------------------------


def _rows():
    return [
        ade.DetectorRow("1", "ok sounds good", "human", "human_short", "casual"),
        ade.DetectorRow("2", "As an AI language model", "ai", "ai_obvious", "self-ref"),
        ade.DetectorRow("3", "sounds-AI human line", "human", "ambiguous", "bait"),
    ]


def test_collect_disagreement_examples_only_flagged_rows():
    rows = _rows()
    labels_by_det = {
        "desklib": ["human", "ai", "ai"],
        "oxidane": ["human", "ai", "human"],
    }
    p_ai_by_det = {"desklib": [0.1, 0.9, 0.8], "oxidane": [0.2, 0.95, 0.4]}
    flags = ade.compute_disagreement(labels_by_det)["flags"]
    ex = ade.collect_disagreement_examples(rows, labels_by_det, p_ai_by_det, flags)
    # Only row 3 (index 2) disagrees.
    assert len(ex) == 1
    assert ex[0]["id"] == "3"
    assert ex[0]["true"] == "human"
    per_det = {d["detector"]: (d["label"], d["p_ai"]) for d in ex[0]["per_detector"]}
    assert per_det["desklib"] == ("ai", pytest.approx(0.8))
    assert per_det["oxidane"] == ("human", pytest.approx(0.4))


def test_collect_wrong_examples_only_mismatches():
    rows = _rows()
    y_pred = ["human", "ai", "ai"]  # row 3 human misread as ai
    p_ai = [0.1, 0.9, 0.8]
    wrong = ade.collect_wrong_examples(rows, y_pred, p_ai)
    assert len(wrong) == 1
    assert wrong[0]["true"] == "human"
    assert wrong[0]["predicted"] == "ai"
    assert wrong[0]["source_type"] == "ambiguous"
    assert wrong[0]["p_ai"] == pytest.approx(0.8)


# --- Slice F: summary + report -----------------------------------------------

_WARNING = "PLACEHOLDER WARNING — sourced from app.routes in the real run."


def _detector_results():
    return {
        "desklib": {
            "name": "desklib-ai-text-detector-v1.01",
            "y_pred": ["human", "ai", "ai"],
            "p_ai": [0.1, 0.9, 0.8],
            "latencies_ms": [40.0, 42.0, 50.0],
        },
        "oxidane": {
            "name": "oxidane-tmr-ai-text-detector",
            "y_pred": ["human", "ai", "human"],
            "p_ai": [0.2, 0.95, 0.4],
            "latencies_ms": [30.0, 31.0, 38.0],
        },
    }


def test_build_summary_shape_and_cross_detector_fields():
    summary = ade.build_detector_summary(
        data_file="evals/data/ai_detection_eval.csv",
        rows=_rows(),
        detector_results=_detector_results(),
        warning=_WARNING,
    )
    for key in (
        "data_file", "n_examples", "labels", "detectors", "disagreement_rate",
        "disagreement_examples", "pairwise_agreement", "calibration", "warning",
    ):
        assert key in summary
    assert summary["n_examples"] == 3
    assert summary["labels"] == ["human", "ai"]
    # Per-detector metrics are present and shaped.
    desk = summary["detectors"]["desklib"]
    for key in ("accuracy", "macro_f1", "latency_p50_ms", "latency_p95_ms",
                "per_class", "confusion_matrix", "wrong_examples"):
        assert key in desk
    # Only row 3 disagrees -> rate 1/3 (rounded to 4dp in the summary).
    assert summary["disagreement_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert len(summary["disagreement_examples"]) == 1
    assert summary["warning"] == _WARNING


def test_render_report_contains_all_required_sections():
    summary = ade.build_detector_summary(
        data_file="d.csv", rows=_rows(),
        detector_results=_detector_results(), warning=_WARNING,
    )
    md = ade.render_detector_report(summary)
    # Both honest texts are required and distinct.
    assert "not proof of authorship" in md  # brief's report paragraph
    assert _WARNING in md  # API warning, verbatim
    # Provenance disclosure: BOTH classes are AI-authored (the "human" rows are
    # AI-written stand-ins), and this is explicitly not a benchmark.
    assert "ai assistant" in md.lower() and "benchmark" in md.lower()
    # Core sections.
    assert "Disagreement" in md
    assert "Pairwise agreement" in md
    assert "Calibration" in md
    # Disagreement example table shows each detector's label + P(ai).
    assert "P(ai)" in md
    # Per-detector accuracy and latency reported.
    assert "Accuracy" in md and "p50" in md and "p95" in md


def test_render_report_no_disagreement_states_it():
    # All detectors agree on every row -> the report should say so, not crash.
    results = {
        "a": {"name": "A", "y_pred": ["human", "ai"], "p_ai": [0.1, 0.9],
              "latencies_ms": [10.0, 11.0]},
        "b": {"name": "B", "y_pred": ["human", "ai"], "p_ai": [0.2, 0.8],
              "latencies_ms": [12.0, 13.0]},
    }
    rows = [
        ade.DetectorRow("1", "x", "human", "human_short", ""),
        ade.DetectorRow("2", "y", "ai", "ai_obvious", ""),
    ]
    summary = ade.build_detector_summary("d.csv", rows, results, _WARNING)
    assert summary["disagreement_rate"] == pytest.approx(0.0)
    md = ade.render_detector_report(summary)
    assert "Disagreement" in md

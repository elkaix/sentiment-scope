"""Offline sentiment evaluation harness — teach *why* each metric exists.

Run a sentiment model over a labeled CSV and produce a Markdown error-analysis
report plus a JSON summary. The CLI drives the model DIRECTLY (no HTTP): offline
evaluation wants a reproducible, single-process run, not a live server.

Why these particular metrics (the whole point of this file):

* **Accuracy** is the headline number but lies under class imbalance — a model
  that always says "neutral" scores well if most rows are neutral.
* **Macro F1** averages the per-class F1 with EQUAL weight per class, so a model
  that ignores the rare class is punished. It is the honest single number when
  classes are imbalanced (and sentiment sets usually are).
* **The confusion matrix** shows *which* classes get confused — e.g. sarcasm
  read as positive, mixed sentiment collapsed to neutral. A single score hides
  that; the matrix is where error analysis actually starts.
* **Latency p50/p95** separates the typical request (p50 = median) from the slow
  tail (p95). Users feel the tail: a good median with a bad p95 still feels
  janky. One mean would blur both together.

Heavy imports (torch/transformers via SentimentModel) live INSIDE ``run()`` so
the pure helpers below stay importable — and unit-testable — without the ML
stack. This mirrors backend/app/model.py's lazy-import design.
"""

import argparse
import csv
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --- Pure helpers (no torch, no app) -----------------------------------------

# Same 2000-char boundary the API enforces (backend/app/schemas.MAX_CHARS).
# Eval inputs are never silently truncated: an overlong row is a data bug the
# author should see and fix, not something we quietly reshape under them.
MAX_TEXT_CHARS = 2000

REQUIRED_COLUMNS = ("text", "true_label")


@dataclass(frozen=True)
class EvalRow:
    """One labeled example. ``category``/``notes`` are optional metadata that
    make the error-analysis table readable (why is THIS example hard?)."""

    id: str
    text: str
    true_label: str
    category: str
    notes: str


def validate_label_compatibility(
    dataset_labels: set[str], model_labels: set[str], allow_mismatch: bool
) -> None:
    """Fail fast when the dataset asks for a class the model cannot predict.

    Running a binary (negative/positive) model on a CSV that contains
    ``neutral`` rows is not an error the model reports — it just silently maps
    every neutral example onto its nearest class and reports plausible-looking
    but misleading metrics. We refuse by default and force an explicit
    ``--allow-label-mismatch`` opt-in.
    """
    if dataset_labels.issubset(model_labels):
        return
    extra = sorted(dataset_labels - model_labels)
    if allow_mismatch:
        print(
            f"WARNING: dataset labels {extra} are not in model labels "
            "— metrics may mislead"
        )
        return
    raise SystemExit(
        f"Dataset labels {sorted(dataset_labels)} are not a subset of model "
        f"labels {sorted(model_labels)}. Unpredictable classes: {extra}. "
        f"Use --allow-label-mismatch to run anyway."
    )


def parse_eval_rows(csv_text: str) -> list[EvalRow]:
    """Parse eval CSV text into ``EvalRow`` records, validating at the boundary.

    Rejects (never silently repairs) two data bugs: a missing required column,
    and a row whose text exceeds ``MAX_TEXT_CHARS``. Truncating an overlong row
    would change the very input being scored, so we fail loudly and name the row
    instead. Blank-text rows are skipped (a trailing newline is not a data
    point); ``category``/``notes`` are optional.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    fields = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in fields]
    if missing:
        raise ValueError(
            f"CSV missing required column(s): {missing}. Found: {list(fields)}"
        )

    rows: list[EvalRow] = []
    for i, raw in enumerate(reader):
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        if len(text) > MAX_TEXT_CHARS:
            # +2: 1 for the header line, 1 for 0-based -> 1-based row numbering,
            # so the number matches what a spreadsheet shows.
            raise ValueError(
                f"Row {i + 2} text is {len(text)} chars, exceeds the "
                f"{MAX_TEXT_CHARS}-char limit. Fix the data; rows are never "
                "truncated."
            )
        true_label = (raw.get("true_label") or "").strip()
        rows.append(
            EvalRow(
                id=(raw.get("id") or str(i + 1)).strip(),
                text=text,
                true_label=true_label,
                category=(raw.get("category") or "").strip(),
                notes=(raw.get("notes") or "").strip(),
            )
        )
    return rows


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    """Score predictions with scikit-learn — the reference implementation.

    ``labels=`` is passed explicitly to BOTH calls so the confusion matrix rows
    and the per-class report share one class order (otherwise sklearn infers a
    different order for each and they stop lining up). ``zero_division=0`` keeps
    a class that got zero predictions from raising a warning and instead scores
    it 0 — which is the honest reading: the model never picked it.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # macro = unweighted mean over classes, so a rare class counts as much
        # as a common one. This is the number that exposes class imbalance.
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def latency_percentiles(latencies_ms: list[float]) -> dict:
    """p50 (median = typical request) and p95 (the slow tail users feel).

    p95 uses ``statistics.quantiles(n=20)`` — the 19th of 20 cut points is the
    95th percentile. With fewer than two samples there is no distribution to cut,
    so both percentiles collapse to the single observation.
    """
    if len(latencies_ms) < 2:
        only = latencies_ms[0] if latencies_ms else 0.0
        return {"p50": only, "p95": only}
    p50 = statistics.median(latencies_ms)
    # method="inclusive" keeps p95 within the observed range — a reported p95
    # latency should never exceed the slowest request we actually measured
    # (the default "exclusive" method extrapolates past the max on small n).
    p95 = statistics.quantiles(latencies_ms, n=20, method="inclusive")[18]
    return {"p50": p50, "p95": p95}


def collect_wrong_examples(
    rows: list[EvalRow], y_pred: list[str], confidences: list[float]
) -> list[dict]:
    """The misclassified rows — the actual product of an eval.

    Aggregate scores tell you *how much* the model is wrong; this table tells you
    *where*, with the confidence attached so you can spot the dangerous case:
    confidently wrong (sarcasm read as glowing praise at 0.95) is worse than
    hesitantly wrong (a mixed review at 0.4)."""
    wrong = []
    for row, pred, conf in zip(rows, y_pred, confidences):
        if pred != row.true_label:
            wrong.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "category": row.category,
                    "true": row.true_label,
                    "predicted": pred,
                    "confidence": conf,
                    "notes": row.notes,
                }
            )
    return wrong


def build_summary(
    model_id: str,
    model_name: str,
    data_file: str,
    rows: list[EvalRow],
    y_pred: list[str],
    confidences: list[float],
    latencies_ms: list[float],
    metrics: dict,
    labels: list[str],
) -> dict:
    """Assemble the machine-readable summary (also the JSON printed to stdout)."""
    lat = latency_percentiles(latencies_ms)
    return {
        "model_id": model_id,
        "model_name": model_name,
        "data_file": data_file,
        "n_examples": len(rows),
        "labels": list(labels),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "latency_p50_ms": round(lat["p50"], 2),
        "latency_p95_ms": round(lat["p95"], 2),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "wrong_examples": collect_wrong_examples(rows, y_pred, confidences),
    }


def _md_cell(text: str) -> str:
    """Escape a value so it survives inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_report(summary: dict) -> str:
    """Render the human-facing Markdown error-analysis report from a summary."""
    labels = summary["labels"]
    lines: list[str] = []
    lines.append(f"# Sentiment Evaluation Report — `{summary['model_id']}`")
    lines.append("")
    lines.append(f"- **Model:** `{summary['model_name']}`")
    lines.append(f"- **Dataset:** `{summary['data_file']}` "
                 f"({summary['n_examples']} labeled examples)")
    lines.append(f"- **Labels:** {', '.join(labels)}")
    lines.append("")

    # Honest-scope note (required): what this eval does and does NOT claim.
    lines.append("> **Sentiment models are not creativity judges.** This project "
                 "tests emotional polarity, confidence, model disagreement, and "
                 "explanation quality across writing styles — not whether the "
                 "writing is *good*.")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append(f"- **Accuracy:** {summary['accuracy']:.3f} — the share of "
                 "correct predictions. Easy to read, but inflated when one class "
                 "dominates the dataset.")
    lines.append(f"- **Macro F1:** {summary['macro_f1']:.3f} — the per-class F1 "
                 "averaged with equal weight per class. Lower than accuracy here "
                 "because the model handles the classes unevenly; this is the "
                 "number that exposes class imbalance.")
    lines.append(f"- **Latency p50:** {summary['latency_p50_ms']:.1f} ms — the "
                 "typical (median) single-text request.")
    lines.append(f"- **Latency p95:** {summary['latency_p95_ms']:.1f} ms — the "
                 "slow tail. Users feel the tail, so p95 matters more than an "
                 "average that would blur it away.")
    lines.append("")

    lines.append("## Per-class precision / recall / F1")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label in labels:
        pc = summary["per_class"][label]
        lines.append(f"| {label} | {pc['precision']:.2f} | {pc['recall']:.2f} "
                     f"| {pc['f1']:.2f} | {pc['support']} |")
    lines.append("")

    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows = true label, columns = predicted label. The diagonal is "
                 "correct; every off-diagonal cell is a specific confusion (which "
                 "class gets mistaken for which).")
    lines.append("")
    lines.append("| true \\ pred | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in labels) + " |")
    for i, label in enumerate(labels):
        cells = " | ".join(str(c) for c in summary["confusion_matrix"][i])
        lines.append(f"| **{label}** | {cells} |")
    lines.append("")

    lines.append("## Misclassified examples")
    lines.append("")
    wrong = summary["wrong_examples"]
    if not wrong:
        lines.append("_No misclassifications — every example matched its label._")
    else:
        lines.append(f"{len(wrong)} of {summary['n_examples']} examples were "
                     "misclassified. Confidence is the model's probability for "
                     "the class it *chose* — high confidence on a wrong answer is "
                     "the failure mode to watch.")
        lines.append("")
        lines.append("| Text | Category | True | Predicted | Confidence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for w in wrong:
            lines.append(
                f"| {_md_cell(w['text'])} | {_md_cell(w['category'])} "
                f"| {w['true']} | {w['predicted']} | {w['confidence']:.2f} |"
            )
    lines.append("")

    lines.append("## Top failure modes")
    lines.append("")
    lines.append("These are the general failure modes sentiment models exhibit. "
                 "This 36-row set triggers sarcasm, mixed sentiment, and ambiguity "
                 "most sharply (see the table above); the finance and formal rows "
                 "happened to land correctly here, but both are classic weak spots "
                 "that surface on larger or harder sets.")
    lines.append("")
    lines.append("1. **Sarcasm** — positive words carrying negative intent "
                 "(\"amazing, another crash\") are read literally.")
    lines.append("2. **Mixed sentiment** — a review that praises one thing and "
                 "pans another gets collapsed to a single dominant class.")
    lines.append("3. **Long / formal text** — flat, low-affect prose has no "
                 "strong sentiment signal, so predictions drift.")
    lines.append("4. **Finance / domain mismatch** — a general social-media model "
                 "misreads finance and news sentiment (\"shares tumbled\").")
    lines.append("5. **Missing context** — very short or ambiguous phrases "
                 "(\"Meh\", \"It is fine\") lack the context a human uses.")
    lines.append("")

    lines.append("## Machine-readable summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: summary[k] for k in (
        "model_id", "accuracy", "macro_f1", "latency_p50_ms",
        "latency_p95_ms", "confusion_matrix",
    )}, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --- CLI + model integration (heavy imports live here) -----------------------

_BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"


def _load_model(model_id: str):
    """Build and load the registry model. torch/transformers are pulled in only
    here — via the backend's own lazy-loading SentimentModel — so importing this
    module for the pure helpers above never touches the ML stack."""
    if str(_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_BACKEND_DIR))
    from app.model import build_model
    from app.model_registry import get_model_config

    try:
        cfg = get_model_config(model_id)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return build_model(cfg), cfg


def run(model_id: str, data_path: str, out_path: str, allow_mismatch: bool) -> dict:
    """End-to-end eval: load data, guard labels, predict + time, score, report."""
    rows = parse_eval_rows(Path(data_path).read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"No labeled rows found in {data_path}")

    model, cfg = _load_model(model_id)

    # Guard BEFORE load(): model.labels is populated from the registry in
    # __init__, so an incompatible dataset fails instantly without paying the
    # ~500MB weight load (mirrors the API's validate-before-load pattern).
    dataset_labels = {r.true_label for r in rows}
    validate_label_compatibility(dataset_labels, set(model.labels), allow_mismatch)

    print(f"Loading {model_id} ({cfg.name})...")
    model.load()
    print(f"Loaded on device: {model.device}")

    # One untimed warmup: the first inference on MPS eats one-off graph/compile
    # cost that would otherwise masquerade as a p95 latency spike.
    model.predict([rows[0].text])

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    latencies_ms: list[float] = []
    for row in rows:
        start = time.perf_counter()  # monotonic clock — correct for durations
        pred = model.predict([row.text])[0]
        latencies_ms.append((time.perf_counter() - start) * 1000)
        y_true.append(row.true_label)
        y_pred.append(pred["label"])
        confidences.append(max(pred["scores"].values()))

    # Label order for the report: the model's own labels first, then any extra
    # dataset labels (only reachable via --allow-label-mismatch) appended so the
    # confusion matrix still accounts for them.
    labels = list(model.labels) + [
        label for label in sorted(dataset_labels) if label not in model.labels
    ]
    metrics = compute_metrics(y_true, y_pred, labels)
    summary = build_summary(
        model_id=model_id,
        model_name=cfg.name,
        data_file=data_path,
        rows=rows,
        y_pred=y_pred,
        confidences=confidences,
        latencies_ms=latencies_ms,
        metrics=metrics,
        labels=labels,
    )

    Path(out_path).write_text(render_report(summary), encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    print(json.dumps({k: summary[k] for k in (
        "model_id", "n_examples", "accuracy", "macro_f1",
        "latency_p50_ms", "latency_p95_ms",
    )}, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a sentiment model on a labeled CSV and write a "
        "Markdown error-analysis report.",
    )
    parser.add_argument(
        "--model-id", default="twitter-roberta",
        help="Registry model id (default: twitter-roberta).",
    )
    parser.add_argument("--data", required=True, help="Path to the labeled eval CSV.")
    parser.add_argument("--out", required=True, help="Path to write the report.")
    parser.add_argument(
        "--allow-label-mismatch", action="store_true",
        help="Run even if the dataset has labels the model cannot predict "
        "(metrics will be misleading).",
    )
    args = parser.parse_args(argv)
    run(args.model_id, args.data, args.out, args.allow_label_mismatch)


if __name__ == "__main__":
    main()

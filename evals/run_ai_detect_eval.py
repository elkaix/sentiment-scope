"""Offline AI-text-detector eval — a DISAGREEMENT report, not a benchmark.

This is the detector-family sibling of ``run_eval.py`` (sentiment). It runs
*every* local detector over the *same* labeled rows and asks the question the
product actually cares about: **do the detectors agree, and when they don't,
where?** It deliberately reuses ``run_eval``'s torch-free helpers
(``compute_metrics``, ``latency_percentiles``, ``_md_cell``) so the two harnesses
share one definition of accuracy/F1/latency — but detector metrics NEVER mix
into a sentiment report and vice-versa.

Three ideas this module is built to teach:

* **Per-detector accuracy here is NOT a benchmark claim.** BOTH classes in the
  eval set were written by the AI assistant that built this project (see
  ``ai_detection_eval.csv``): the ``human`` rows are AI-authored text *styled*
  to read as human, not genuinely human writing. So a high P(ai) on a "human"
  row is not cleanly a false positive — the detector may be correctly smelling
  AI provenance. On top of that, the two hardest categories (``mixed_human_ai``,
  ``ambiguous``) have *contestable* ground truth even setting authorship aside;
  we arbitrate by ORIGIN/intent (mixed → ai, ambiguous → human), so a detector's
  score is partly a function of *our* arbitration. A real benchmark needs large,
  diverse, provenance-verified data with genuinely human text — none of which
  this is.

* **Disagreement is signal.** When three detectors trained on different data
  split on the same sentence, that split is a calibrated statement of
  uncertainty about that sentence — far more honest than any single model's
  confident label. The report surfaces the disagreement RATE and every
  disagreed-on row (each detector's label + P(ai)) as the primary deliverable.

* **Calibration matters for a probabilistic detector.** A useful P(ai) should be
  systematically *low* on genuine human text and *high* on AI text. The
  calibration hint (mean P(ai) on human rows vs ai rows, and the gap between
  them) shows whether a detector's probabilities separate the classes at all —
  a detector with a near-zero or negative gap is guessing, regardless of its
  headline accuracy.

Heavy imports (torch/transformers via DetectorModel) live INSIDE ``run()`` so
the pure helpers below stay importable and unit-testable without the ML stack,
exactly like ``run_eval``.
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

# Shared, torch-free helpers — one definition of the numeric/formatting logic
# across both eval harnesses. Importable because run_eval keeps its heavy
# imports inside run() too.
from run_eval import (
    MAX_TEXT_CHARS,
    _md_cell,
    compute_metrics,
    latency_percentiles,
    validate_label_compatibility,
)

# --- Pure helpers (no torch, no app) -----------------------------------------

# The only two classes any detector in this project emits (registry-canonical).
DETECTOR_LABELS = ["human", "ai"]

REQUIRED_COLUMNS = ("text", "true_label")


@dataclass(frozen=True)
class DetectorRow:
    """One labeled example. ``source_type`` is the failure-analysis axis: it says
    *why* a row is interesting (``ai_edited``, ``ambiguous``, …), which is where
    detector disagreement clusters."""

    id: str
    text: str
    true_label: str
    source_type: str
    notes: str


def parse_detector_rows(csv_text: str) -> list[DetectorRow]:
    """Parse the detector CSV (``id,text,true_label,source_type,notes``).

    Same fail-loud contract as ``run_eval.parse_eval_rows``: reject a missing
    required column or an overlong row (never silently truncate the very text
    being scored), skip blank-text rows. The only difference is the
    ``source_type`` metadata column in place of ``category``.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    fields = reader.fieldnames or []
    missing = [c for c in REQUIRED_COLUMNS if c not in fields]
    if missing:
        raise ValueError(
            f"CSV missing required column(s): {missing}. Found: {list(fields)}"
        )

    rows: list[DetectorRow] = []
    for i, raw in enumerate(reader):
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        if len(text) > MAX_TEXT_CHARS:
            # +2: header line + 0->1-based, so the number matches a spreadsheet.
            raise ValueError(
                f"Row {i + 2} text is {len(text)} chars, exceeds the "
                f"{MAX_TEXT_CHARS}-char limit. Fix the data; rows are never "
                "truncated."
            )
        rows.append(
            DetectorRow(
                id=(raw.get("id") or str(i + 1)).strip(),
                text=text,
                true_label=(raw.get("true_label") or "").strip(),
                source_type=(raw.get("source_type") or "").strip(),
                notes=(raw.get("notes") or "").strip(),
            )
        )
    return rows


def compute_disagreement(labels_by_detector: dict[str, list[str]]) -> dict:
    """Per-row disagreement flag + overall rate.

    A row "disagrees" when the detectors don't all land on the same label —
    ``len({labels}) > 1``, the exact rule ``/api/ai-detect/compare`` uses, so
    the eval and the endpoint tell the same story. The rate is the share of rows
    the fleet split on; a high rate on a hand-picked hard set is expected and is
    the honest headline of this report.
    """
    detectors = list(labels_by_detector)
    n = len(labels_by_detector[detectors[0]]) if detectors else 0
    flags = [
        len({labels_by_detector[d][i] for d in detectors}) > 1 for i in range(n)
    ]
    rate = (sum(flags) / n) if n else 0.0
    return {"flags": flags, "rate": rate}


def pairwise_agreement(labels_by_detector: dict[str, list[str]]) -> list[dict]:
    """Fraction of rows each detector PAIR labels identically.

    The disagreement rate collapses the whole fleet to one number; pairwise
    agreement shows the structure underneath — which two detectors move together
    and which one is the odd one out. Reported for every unordered pair.
    """
    from itertools import combinations

    ids = list(labels_by_detector)
    out: list[dict] = []
    for a, b in combinations(ids, 2):
        la, lb = labels_by_detector[a], labels_by_detector[b]
        n = len(la)
        agree = sum(1 for x, y in zip(la, lb) if x == y)
        out.append(
            {"detector_a": a, "detector_b": b, "agreement": (agree / n) if n else 0.0}
        )
    return out


def calibration_hint(
    true_labels: list[str], p_ai_by_detector: dict[str, list[float]]
) -> dict[str, dict]:
    """Mean P(ai) on human rows vs ai rows, per detector, plus the gap.

    A probabilistic detector is only useful if its P(ai) is systematically lower
    on real human text than on AI text. ``gap = mean_on_ai - mean_on_human``
    measures that separation directly: a large positive gap means the
    probabilities carry real information; a gap near zero (or negative) means the
    detector is effectively guessing even if its thresholded accuracy looks fine.
    ``None`` when a class is absent, so a one-sided set never fabricates a gap.
    """
    human_idx = [i for i, t in enumerate(true_labels) if t == "human"]
    ai_idx = [i for i, t in enumerate(true_labels) if t == "ai"]
    out: dict[str, dict] = {}
    for d, ps in p_ai_by_detector.items():
        mh = statistics.mean(ps[i] for i in human_idx) if human_idx else None
        ma = statistics.mean(ps[i] for i in ai_idx) if ai_idx else None
        gap = (ma - mh) if (mh is not None and ma is not None) else None
        out[d] = {"mean_p_ai_on_human": mh, "mean_p_ai_on_ai": ma, "gap": gap}
    return out


def collect_disagreement_examples(
    rows: list[DetectorRow],
    labels_by_detector: dict[str, list[str]],
    p_ai_by_detector: dict[str, list[float]],
    flags: list[bool],
) -> list[dict]:
    """The disagreed-on rows — the actual product of this eval.

    For each flagged row, attach every detector's label and P(ai) side by side,
    so a reader sees not just *that* the fleet split but *how confidently* each
    side did — the case worth teaching is one detector at 0.9 against another at
    0.1 on the same sentence."""
    detectors = list(labels_by_detector)
    out: list[dict] = []
    for i, (row, flag) in enumerate(zip(rows, flags)):
        if not flag:
            continue
        per = [
            {
                "detector": d,
                "label": labels_by_detector[d][i],
                "p_ai": p_ai_by_detector[d][i],
            }
            for d in detectors
        ]
        out.append(
            {
                "id": row.id,
                "text": row.text,
                "source_type": row.source_type,
                "true": row.true_label,
                "per_detector": per,
                "notes": row.notes,
            }
        )
    return out


def collect_wrong_examples(
    rows: list[DetectorRow], y_pred: list[str], p_ai: list[float]
) -> list[dict]:
    """One detector's misclassified rows, with P(ai) attached.

    Confidently wrong is the dangerous failure: a detector calling genuine human
    text "ai" at P(ai)=0.9 is worse than a hesitant 0.55. The ``source_type``
    tells you which kind of text tripped it (non-native? edited AI?)."""
    wrong: list[dict] = []
    for row, pred, p in zip(rows, y_pred, p_ai):
        if pred != row.true_label:
            wrong.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "source_type": row.source_type,
                    "true": row.true_label,
                    "predicted": pred,
                    "p_ai": p,
                    "notes": row.notes,
                }
            )
    return wrong


def build_detector_summary(
    data_file: str,
    rows: list[DetectorRow],
    detector_results: dict[str, dict],
    warning: str,
) -> dict:
    """Assemble the machine-readable summary from raw per-detector predictions.

    ``detector_results[id]`` carries ``name``/``y_pred``/``p_ai``/
    ``latencies_ms`` (produced by ``run()``); every metric — per-detector scores
    AND the cross-detector disagreement/pairwise/calibration blocks — is derived
    here from those torch-free arrays, so this whole function is unit-testable
    without the model stack.
    """
    y_true = [r.true_label for r in rows]
    labels = list(DETECTOR_LABELS)
    labels_by_detector = {d: res["y_pred"] for d, res in detector_results.items()}
    p_ai_by_detector = {d: res["p_ai"] for d, res in detector_results.items()}

    detectors: dict[str, dict] = {}
    for d, res in detector_results.items():
        metrics = compute_metrics(y_true, res["y_pred"], labels)
        lat = latency_percentiles(res["latencies_ms"])
        detectors[d] = {
            "name": res["name"],
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "latency_p50_ms": round(lat["p50"], 2),
            "latency_p95_ms": round(lat["p95"], 2),
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "wrong_examples": collect_wrong_examples(rows, res["y_pred"], res["p_ai"]),
        }

    dis = compute_disagreement(labels_by_detector)
    return {
        "data_file": data_file,
        "n_examples": len(rows),
        "labels": labels,
        "detectors": detectors,
        "disagreement_rate": round(dis["rate"], 4),
        "disagreement_examples": collect_disagreement_examples(
            rows, labels_by_detector, p_ai_by_detector, dis["flags"]
        ),
        "pairwise_agreement": pairwise_agreement(labels_by_detector),
        "calibration": calibration_hint(y_true, p_ai_by_detector),
        "warning": warning,
    }


# The brief's honest-scope paragraph — a DIFFERENT string from the API's
# DETECTOR_WARNING (both are required in the report). This one explains what the
# eval does and does not claim; the API warning is the product-facing promise.
_HONEST_LIMITATIONS = (
    "AI text detection is not proof of authorship. These models estimate whether "
    "text resembles patterns seen in AI-generated or human-written training data. "
    "Short text, edited AI text, non-native writing, and formal human writing can "
    "all confuse detectors."
)


def _fmt(x: float | None, nd: int = 3) -> str:
    """Format a possibly-None metric for a table cell."""
    return "—" if x is None else f"{x:.{nd}f}"


def render_detector_report(summary: dict) -> str:
    """Render the human-facing Markdown disagreement report from a summary."""
    labels = summary["labels"]
    detectors = summary["detectors"]
    lines: list[str] = []

    lines.append("# AI Text Detector Evaluation — Disagreement Report")
    lines.append("")
    lines.append(f"- **Dataset:** `{summary['data_file']}` "
                 f"({summary['n_examples']} labeled examples)")
    lines.append(f"- **Detectors:** {', '.join(detectors)}")
    lines.append(f"- **Labels:** {', '.join(labels)}")
    lines.append("")

    # Two distinct honest texts, both required.
    lines.append(f"> **Limitations.** {_HONEST_LIMITATIONS}")
    lines.append("")
    lines.append("> **Product warning (verbatim, shown on every detector API "
                 f"response):** {summary['warning']}")
    lines.append("")

    lines.append("## Why this is not a benchmark")
    lines.append("")
    lines.append("Every row in this set — **both the `human`-labeled and the "
                 "`ai`-labeled examples — was written by the AI assistant that "
                 "built this project.** There is no genuinely human-authored text "
                 "here: the `human` rows are AI-authored text *styled* to read as "
                 "human (casual, formal, non-native, marketing, \"sounds-AI\"). "
                 "That makes this a teaching fixture for *failure analysis and "
                 "disagreement*, **not a benchmark** of detector quality, and it "
                 "colors every number below:")
    lines.append("")
    lines.append("1. **The `human` class is an AI-authored stand-in, not ground "
                 "truth.** A high P(ai) on those rows is therefore not cleanly a "
                 "false positive — the detector may be correctly smelling AI "
                 "provenance. Read \"flags a human row as AI\" as \"flags our "
                 "AI-authored human *impersonation* as AI,\" which is as much a "
                 "statement about the fixture as about the detector. The small "
                 "calibration gaps below follow from the same fact: AI-styled-human "
                 "and AI text are *both* AI-authored, so their P(ai) distributions "
                 "naturally sit close together.")
    lines.append("2. **Contestable ground truth.** Even setting authorship aside, "
                 "the `mixed_human_ai` and `ambiguous` rows have no objective gold "
                 "label. We arbitrate by **origin/intent** (`mixed_human_ai` → "
                 "`ai`, `ambiguous` → `human`), so each detector's accuracy is "
                 "partly a function of *our* labeling calls, not just its skill.")
    lines.append("3. **Tiny, single-author, English-only.** Real evaluation needs "
                 "large, diverse, provenance-verified data with genuinely human "
                 "writing. Read the numbers below as *illustrations of failure "
                 "modes and detector disagreement*, never as scores.")
    lines.append("")

    # --- The primary deliverable: disagreement --------------------------------
    lines.append("## Detector disagreement")
    lines.append("")
    rate = summary["disagreement_rate"]
    lines.append(f"The fleet split on **{rate:.1%}** of rows (a row \"disagrees\" "
                 "when the detectors do not all pick the same label — the same rule "
                 "`/api/ai-detect/compare` uses). Disagreement is not noise to "
                 "average away: when models trained on different data split on a "
                 "sentence, that split *is* a statement of uncertainty about it.")
    lines.append("")
    ex = summary["disagreement_examples"]
    if not ex:
        lines.append("_No disagreements — every detector agreed on every row._")
    else:
        det_ids = list(detectors)
        header = ["Text", "Source", "True"] + [f"{d} label / P(ai)" for d in det_ids]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for e in ex:
            per = {d["detector"]: d for d in e["per_detector"]}
            cells = [_md_cell(e["text"]), _md_cell(e["source_type"]), e["true"]]
            for d in det_ids:
                cells.append(f"{per[d]['label']} / {per[d]['p_ai']:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Pairwise agreement")
    lines.append("")
    lines.append("How often each PAIR of detectors lands on the same label. This "
                 "shows the structure the single disagreement rate hides — which "
                 "detectors move together, and which is the outlier.")
    lines.append("")
    lines.append("| Detector A | Detector B | Agreement |")
    lines.append("| --- | --- | --- |")
    for p in summary["pairwise_agreement"]:
        lines.append(f"| {p['detector_a']} | {p['detector_b']} | "
                     f"{p['agreement']:.1%} |")
    lines.append("")

    lines.append("## Calibration hint")
    lines.append("")
    lines.append("Mean P(ai) on the genuinely-human rows vs the AI rows. A useful "
                 "probabilistic detector keeps P(ai) **low on human** text and "
                 "**high on AI** text; the **gap** is that separation. A gap near "
                 "zero means the probabilities carry little information even when "
                 "the thresholded label happens to be right.")
    lines.append("")
    lines.append("| Detector | Mean P(ai) on human | Mean P(ai) on AI | Gap |")
    lines.append("| --- | --- | --- | --- |")
    for d, c in summary["calibration"].items():
        lines.append(f"| {d} | {_fmt(c['mean_p_ai_on_human'])} | "
                     f"{_fmt(c['mean_p_ai_on_ai'])} | {_fmt(c['gap'])} |")
    lines.append("")

    # --- Per-detector metrics -------------------------------------------------
    lines.append("## Per-detector metrics")
    lines.append("")
    lines.append("Accuracy and macro F1 against the seed labels, plus single-text "
                 "latency (p50 = typical, p95 = the slow tail users feel).")
    lines.append("")
    lines.append("| Detector | Accuracy | Macro F1 | Latency p50 (ms) | "
                 "Latency p95 (ms) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for d, m in detectors.items():
        lines.append(f"| {d} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                     f"{m['latency_p50_ms']:.1f} | {m['latency_p95_ms']:.1f} |")
    lines.append("")

    for d, m in detectors.items():
        lines.append(f"### `{d}` — {m['name']}")
        lines.append("")
        lines.append("Per-class precision / recall / F1:")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label in labels:
            pc = m["per_class"][label]
            lines.append(f"| {label} | {pc['precision']:.2f} | {pc['recall']:.2f} "
                         f"| {pc['f1']:.2f} | {pc['support']} |")
        lines.append("")
        lines.append("Confusion matrix (rows = true, cols = predicted):")
        lines.append("")
        lines.append("| true \\ pred | " + " | ".join(labels) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in labels) + " |")
        for i, label in enumerate(labels):
            cells = " | ".join(str(c) for c in m["confusion_matrix"][i])
            lines.append(f"| **{label}** | {cells} |")
        lines.append("")
        wrong = m["wrong_examples"]
        if not wrong:
            lines.append("_No misclassifications for this detector._")
        else:
            lines.append(f"Misclassified ({len(wrong)} of {summary['n_examples']}) — "
                         "P(ai) shows how confidently it was wrong:")
            lines.append("")
            lines.append("| Text | Source | True | Predicted | P(ai) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for w in wrong:
                lines.append(f"| {_md_cell(w['text'])} | {_md_cell(w['source_type'])} "
                             f"| {w['true']} | {w['predicted']} | {w['p_ai']:.2f} |")
        lines.append("")

    lines.append("## Machine-readable summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "n_examples": summary["n_examples"],
        "disagreement_rate": summary["disagreement_rate"],
        "pairwise_agreement": summary["pairwise_agreement"],
        "calibration": summary["calibration"],
        "detectors": {
            d: {k: m[k] for k in ("accuracy", "macro_f1",
                                  "latency_p50_ms", "latency_p95_ms")}
            for d, m in detectors.items()
        },
    }, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --- CLI + model integration (heavy imports live here) -----------------------

_BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"


def _backend_imports():
    """Import the backend registry/model factory + the API's warning string.

    torch/transformers are pulled in only when a detector actually loads (the
    backend's own lazy design), so importing this module for the pure helpers
    stays cheap. The warning is sourced from ``app.routes`` — its single source
    of truth — so the report can never drift from what the API promises."""
    if str(_BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(_BACKEND_DIR))
    from app.model import build_model
    from app.model_registry import ModelTask, get_model_config, models_for_task
    from app.routes import DETECTOR_WARNING

    return build_model, get_model_config, models_for_task, ModelTask, DETECTOR_WARNING


def run(data_path: str, out_path: str, detector_ids: list[str] | None = None) -> dict:
    """End-to-end eval: load data, run every detector over the SAME rows, score
    each, compute cross-detector disagreement, and write the report."""
    (build_model, get_model_config, models_for_task,
     ModelTask, warning) = _backend_imports()

    rows = parse_detector_rows(Path(data_path).read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"No labeled rows found in {data_path}")

    # Guard once: the seed labels must be predictable by the detectors. All
    # detectors share the ("human", "ai") label set, so one check covers them.
    dataset_labels = {r.true_label for r in rows}
    validate_label_compatibility(dataset_labels, set(DETECTOR_LABELS), allow_mismatch=False)

    ids = detector_ids or list(models_for_task(ModelTask.AI_TEXT_DETECTION))

    detector_results: dict[str, dict] = {}
    for model_id in ids:
        cfg = get_model_config(model_id)
        if cfg.task != ModelTask.AI_TEXT_DETECTION:
            raise SystemExit(
                f"Model '{model_id}' is not an AI text detector (task={cfg.task})."
            )
        model = build_model(cfg)
        print(f"Loading {model_id} ({cfg.name})...")
        model.load()
        print(f"  loaded on device: {model.device}")

        # One untimed warmup: the first MPS inference pays one-off graph/compile
        # cost that would otherwise masquerade as a p95 spike.
        model.predict([rows[0].text])

        y_pred: list[str] = []
        p_ai: list[float] = []
        latencies_ms: list[float] = []
        for row in rows:
            start = time.perf_counter()
            pred = model.predict([row.text])[0]
            latencies_ms.append((time.perf_counter() - start) * 1000)
            y_pred.append(pred["label"])
            p_ai.append(pred["scores"]["ai"])

        detector_results[model_id] = {
            "name": cfg.name,
            "y_pred": y_pred,
            "p_ai": p_ai,
            "latencies_ms": latencies_ms,
        }
        # Drop the model before loading the next one — desklib alone is ~1.6GB;
        # we only need its predictions from here on, not its weights.
        del model

    summary = build_detector_summary(data_path, rows, detector_results, warning)
    Path(out_path).write_text(render_detector_report(summary), encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    print(json.dumps({
        "n_examples": summary["n_examples"],
        "disagreement_rate": summary["disagreement_rate"],
        "detectors": {
            d: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            for d, m in summary["detectors"].items()
        },
    }, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run every local AI text detector over a labeled CSV and "
        "write a Markdown disagreement report.",
    )
    parser.add_argument(
        "--data", default="evals/data/ai_detection_eval.csv",
        help="Path to the labeled detector eval CSV.",
    )
    parser.add_argument(
        "--out", default="evals/ai_detection_report.md",
        help="Path to write the report.",
    )
    parser.add_argument(
        "--detectors", nargs="*", default=None,
        help="Detector model ids to run (default: all registry detectors).",
    )
    args = parser.parse_args(argv)
    run(args.data, args.out, args.detectors)


if __name__ == "__main__":
    main()

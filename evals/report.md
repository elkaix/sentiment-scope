# Sentiment Evaluation Report — `twitter-roberta`

- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Dataset:** `evals/data/sentiment_eval.csv` (36 labeled examples)
- **Labels:** negative, neutral, positive

> **Sentiment models are not creativity judges.** This project tests emotional polarity, confidence, model disagreement, and explanation quality across writing styles — not whether the writing is *good*.

## Headline metrics

- **Accuracy:** 0.694 — the share of correct predictions. Easy to read, but inflated when one class dominates the dataset.
- **Macro F1:** 0.689 — the per-class F1 averaged with equal weight per class. Lower than accuracy here because the model handles the classes unevenly; this is the number that exposes class imbalance.
- **Latency p50:** 6.4 ms — the typical (median) single-text request.
- **Latency p95:** 27.0 ms — the slow tail. Users feel the tail, so p95 matters more than an average that would blur it away.

## Per-class precision / recall / F1

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.82 | 0.64 | 0.72 | 14 |
| neutral | 0.75 | 0.55 | 0.63 | 11 |
| positive | 0.59 | 0.91 | 0.71 | 11 |

## Confusion matrix

Rows = true label, columns = predicted label. The diagonal is correct; every off-diagonal cell is a specific confusion (which class gets mistaken for which).

| true \ pred | negative | neutral | positive |
| --- | --- | --- | --- |
| **negative** | 9 | 1 | 4 |
| **neutral** | 2 | 6 | 3 |
| **positive** | 0 | 1 | 10 |

## Misclassified examples

11 of 36 examples were misclassified. Confidence is the model's probability for the class it *chose* — high confidence on a wrong answer is the failure mode to watch.

| Text | Category | True | Predicted | Confidence |
| --- | --- | --- | --- | --- |
| Not bad, not great | ambiguous | neutral | negative | 0.50 |
| I love the design but the app keeps crashing | mixed | neutral | negative | 0.60 |
| Yeah, amazing, another crash | sarcasm | negative | positive | 0.77 |
| There is nothing wrong with the food here | negation | positive | neutral | 0.52 |
| Wow what a fantastic way to waste an afternoon | sarcasm | negative | positive | 0.82 |
| Sure, because waiting two hours is exactly what I wanted | sarcasm | negative | positive | 0.46 |
| The food was delicious but the wait was unbearable | mixed | neutral | positive | 0.77 |
| Great camera though the battery drains too fast | mixed | neutral | positive | 0.60 |
| It is fine I guess | ambiguous | neutral | positive | 0.57 |
| Could be better | ambiguous | negative | positive | 0.64 |
| Meh | ambiguous | negative | neutral | 0.62 |

## Top failure modes

These are the general failure modes sentiment models exhibit. This 36-row set triggers sarcasm, mixed sentiment, and ambiguity most sharply (see the table above); the finance and formal rows happened to land correctly here, but both are classic weak spots that surface on larger or harder sets.

1. **Sarcasm** — positive words carrying negative intent ("amazing, another crash") are read literally.
2. **Mixed sentiment** — a review that praises one thing and pans another gets collapsed to a single dominant class.
3. **Long / formal text** — flat, low-affect prose has no strong sentiment signal, so predictions drift.
4. **Finance / domain mismatch** — a general social-media model misreads finance and news sentiment ("shares tumbled").
5. **Missing context** — very short or ambiguous phrases ("Meh", "It is fine") lack the context a human uses.

## Machine-readable summary

```json
{
  "model_id": "twitter-roberta",
  "accuracy": 0.6944,
  "macro_f1": 0.6886,
  "latency_p50_ms": 6.4,
  "latency_p95_ms": 26.98,
  "confusion_matrix": [
    [
      9,
      1,
      4
    ],
    [
      2,
      6,
      3
    ],
    [
      0,
      1,
      10
    ]
  ]
}
```

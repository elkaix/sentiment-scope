# AI Text Detector Evaluation — Disagreement Report

- **Dataset:** `evals/data/ai_detection_eval.csv` (30 labeled examples)
- **Detectors:** desklib-ai-detector, fakespot-ai-detector, oxidane-ai-detector
- **Labels:** human, ai

> **Limitations.** AI text detection is not proof of authorship. These models estimate whether text resembles patterns seen in AI-generated or human-written training data. Short text, edited AI text, non-native writing, and formal human writing can all confuse detectors.

> **Product warning (verbatim, shown on every detector API response):** AI detectors are probabilistic and can be wrong, especially on short, edited, non-native, highly formal, or mixed-authorship text. Do not use this as proof of authorship.

## Why this is not a benchmark

Every row in this set — **both the `human`-labeled and the `ai`-labeled examples — was written by the AI assistant that built this project.** There is no genuinely human-authored text here: the `human` rows are AI-authored text *styled* to read as human (casual, formal, non-native, marketing, "sounds-AI"). That makes this a teaching fixture for *failure analysis and disagreement*, **not a benchmark** of detector quality, and it colors every number below:

1. **The `human` class is an AI-authored stand-in, not ground truth.** A high P(ai) on those rows is therefore not cleanly a false positive — the detector may be correctly smelling AI provenance. Read "flags a human row as AI" as "flags our AI-authored human *impersonation* as AI," which is as much a statement about the fixture as about the detector. The small calibration gaps below follow from the same fact: AI-styled-human and AI text are *both* AI-authored, so their P(ai) distributions naturally sit close together.
2. **Contestable ground truth.** Even setting authorship aside, the `mixed_human_ai` and `ambiguous` rows have no objective gold label. We arbitrate by **origin/intent** (`mixed_human_ai` → `ai`, `ambiguous` → `human`), so each detector's accuracy is partly a function of *our* labeling calls, not just its skill.
3. **Tiny, single-author, English-only.** Real evaluation needs large, diverse, provenance-verified data with genuinely human writing. Read the numbers below as *illustrations of failure modes and detector disagreement*, never as scores.

## Detector disagreement

The fleet split on **60.0%** of rows (a row "disagrees" when the detectors do not all pick the same label — the same rule `/api/ai-detect/compare` uses). Disagreement is not noise to average away: when models trained on different data split on a sentence, that split *is* a statement of uncertainty about it.

| Text | Source | True | desklib-ai-detector label / P(ai) | fakespot-ai-detector label / P(ai) | oxidane-ai-detector label / P(ai) |
| --- | --- | --- | --- | --- | --- |
| ok sounds good, see you then | human_short | human | ai / 0.55 | human / 0.04 | ai / 0.60 |
| ugh my train is late again wtf | human_short | human | ai / 0.82 | human / 0.01 | ai / 0.86 |
| can u send me the file when ur free? thanks | human_short | human | ai / 0.73 | human / 0.11 | ai / 0.82 |
| Please find attached the minutes from Tuesday's meeting; corrections should be submitted by Friday. | human_formal | human | ai / 0.86 | human / 0.08 | ai / 0.98 |
| I am very like this restaurant, the foods are delicious and staffs are kind to us. | human_non_native | human | ai / 0.89 | human / 0.01 | human / 0.17 |
| Yesterday I go to market but it was close, so I am buying nothing and return home. | human_non_native | human | ai / 0.68 | human / 0.02 | human / 0.27 |
| This phone is good but battery is not so much long I must charge two time one day. | human_non_native | human | ai / 0.66 | human / 0.00 | human / 0.08 |
| Introducing the all-new Aurora blender — power, precision, and a splash of style for your kitchen. | human_marketing | human | ai / 0.63 | human / 0.13 | ai / 0.98 |
| Our handcrafted candles turn any evening into a moment worth savoring. Light one and unwind. | human_marketing | human | ai / 1.00 | human / 0.14 | ai / 0.97 |
| Fresh beans, roasted daily, delivered to your door. Coffee the way it was meant to be. | human_marketing | human | ai / 0.83 | human / 0.04 | ai / 0.86 |
| Leveraging cutting-edge technology, our solution seamlessly integrates into your existing workflow. | ai_polished | ai | ai / 0.99 | human / 0.03 | ai / 0.98 |
| By prioritizing user experience the platform delivers a seamless and intuitive journey end to end. | ai_polished | ai | ai / 0.98 | human / 0.12 | ai / 0.98 |
| honestly this approach ensures optimal outcomes while, um, fostering long-term growth lol | ai_edited | ai | ai / 1.00 | human / 0.07 | ai / 0.98 |
| By prioritizing user experience the platform delivers a smooth journey start to finish — pretty neat tbh | ai_edited | ai | ai / 0.89 | human / 0.00 | ai / 0.98 |
| leveraging modern tech our tool fits into ur existing workflow to boost productivity (mostly) | ai_edited | ai | ai / 0.78 | human / 0.01 | ai / 0.95 |
| With a focus on the end user, the app provides a smooth and easy-to-follow path from beginning to end. | ai_paraphrased | ai | ai / 0.99 | human / 0.18 | ai / 0.98 |
| A thorough strategy such as this secures the best results and encourages steady expansion in every area. | ai_paraphrased | ai | ai / 0.70 | human / 0.19 | ai / 0.98 |
| my two cents plus the draft: The proposed framework offers a robust and scalable foundation for future development. | mixed_human_ai | ai | ai / 1.00 | human / 0.14 | ai / 0.98 |

## Pairwise agreement

How often each PAIR of detectors lands on the same label. This shows the structure the single disagreement rate hides — which detectors move together, and which is the outlier.

| Detector A | Detector B | Agreement |
| --- | --- | --- |
| desklib-ai-detector | fakespot-ai-detector | 40.0% |
| desklib-ai-detector | oxidane-ai-detector | 90.0% |
| fakespot-ai-detector | oxidane-ai-detector | 50.0% |

## Calibration hint

Mean P(ai) on the genuinely-human rows vs the AI rows. A useful probabilistic detector keeps P(ai) **low on human** text and **high on AI** text; the **gap** is that separation. A gap near zero means the probabilities carry little information even when the thresholded label happens to be right.

| Detector | Mean P(ai) on human | Mean P(ai) on AI | Gap |
| --- | --- | --- | --- |
| desklib-ai-detector | 0.836 | 0.939 | 0.102 |
| fakespot-ai-detector | 0.364 | 0.508 | 0.144 |
| oxidane-ai-detector | 0.765 | 0.979 | 0.214 |

## Per-detector metrics

Accuracy and macro F1 against the seed labels, plus single-text latency (p50 = typical, p95 = the slow tail users feel).

| Detector | Accuracy | Macro F1 | Latency p50 (ms) | Latency p95 (ms) |
| --- | --- | --- | --- | --- |
| desklib-ai-detector | 0.500 | 0.333 | 50.7 | 199.0 |
| fakespot-ai-detector | 0.567 | 0.562 | 8.5 | 32.9 |
| oxidane-ai-detector | 0.600 | 0.524 | 6.5 | 8.5 |

### `desklib-ai-detector` — desklib/ai-text-detector-v1.01

Per-class precision / recall / F1:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| human | 0.00 | 0.00 | 0.00 | 15 |
| ai | 0.50 | 1.00 | 0.67 | 15 |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | human | ai |
| --- | --- | --- |
| **human** | 0 | 15 |
| **ai** | 0 | 15 |

Misclassified (15 of 30) — P(ai) shows how confidently it was wrong:

| Text | Source | True | Predicted | P(ai) |
| --- | --- | --- | --- | --- |
| ok sounds good, see you then | human_short | human | ai | 0.55 |
| ugh my train is late again wtf | human_short | human | ai | 0.82 |
| can u send me the file when ur free? thanks | human_short | human | ai | 0.73 |
| The committee reviewed the proposal and recommends deferring the decision until the next fiscal quarter. | human_formal | human | ai | 0.93 |
| Please find attached the minutes from Tuesday's meeting; corrections should be submitted by Friday. | human_formal | human | ai | 0.86 |
| I am writing to confirm receipt of your application, which is now under review by our admissions panel. | human_formal | human | ai | 0.96 |
| I am very like this restaurant, the foods are delicious and staffs are kind to us. | human_non_native | human | ai | 0.89 |
| Yesterday I go to market but it was close, so I am buying nothing and return home. | human_non_native | human | ai | 0.68 |
| This phone is good but battery is not so much long I must charge two time one day. | human_non_native | human | ai | 0.66 |
| Introducing the all-new Aurora blender — power, precision, and a splash of style for your kitchen. | human_marketing | human | ai | 0.63 |
| Our handcrafted candles turn any evening into a moment worth savoring. Light one and unwind. | human_marketing | human | ai | 1.00 |
| Fresh beans, roasted daily, delivered to your door. Coffee the way it was meant to be. | human_marketing | human | ai | 0.83 |
| In today's fast-paced world, staying organized is more important than ever for reaching your goals. | ambiguous | human | ai | 1.00 |
| There are many factors to consider, and ultimately the right choice depends on your own needs. | ambiguous | human | ai | 1.00 |
| Overall, it was a memorable trip that offered both challenges and a few lessons I won't forget. | ambiguous | human | ai | 1.00 |

### `fakespot-ai-detector` — fakespot-ai/roberta-base-ai-text-detection-v1

Per-class precision / recall / F1:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| human | 0.56 | 0.67 | 0.61 | 15 |
| ai | 0.58 | 0.47 | 0.52 | 15 |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | human | ai |
| --- | --- | --- |
| **human** | 10 | 5 |
| **ai** | 8 | 7 |

Misclassified (13 of 30) — P(ai) shows how confidently it was wrong:

| Text | Source | True | Predicted | P(ai) |
| --- | --- | --- | --- | --- |
| The committee reviewed the proposal and recommends deferring the decision until the next fiscal quarter. | human_formal | human | ai | 0.90 |
| I am writing to confirm receipt of your application, which is now under review by our admissions panel. | human_formal | human | ai | 0.98 |
| In today's fast-paced world, staying organized is more important than ever for reaching your goals. | ambiguous | human | ai | 1.00 |
| There are many factors to consider, and ultimately the right choice depends on your own needs. | ambiguous | human | ai | 1.00 |
| Overall, it was a memorable trip that offered both challenges and a few lessons I won't forget. | ambiguous | human | ai | 1.00 |
| Leveraging cutting-edge technology, our solution seamlessly integrates into your existing workflow. | ai_polished | ai | human | 0.03 |
| By prioritizing user experience the platform delivers a seamless and intuitive journey end to end. | ai_polished | ai | human | 0.12 |
| honestly this approach ensures optimal outcomes while, um, fostering long-term growth lol | ai_edited | ai | human | 0.07 |
| By prioritizing user experience the platform delivers a smooth journey start to finish — pretty neat tbh | ai_edited | ai | human | 0.00 |
| leveraging modern tech our tool fits into ur existing workflow to boost productivity (mostly) | ai_edited | ai | human | 0.01 |
| With a focus on the end user, the app provides a smooth and easy-to-follow path from beginning to end. | ai_paraphrased | ai | human | 0.18 |
| A thorough strategy such as this secures the best results and encourages steady expansion in every area. | ai_paraphrased | ai | human | 0.19 |
| my two cents plus the draft: The proposed framework offers a robust and scalable foundation for future development. | mixed_human_ai | ai | human | 0.14 |

### `oxidane-ai-detector` — Oxidane/tmr-ai-text-detector

Per-class precision / recall / F1:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| human | 1.00 | 0.20 | 0.33 | 15 |
| ai | 0.56 | 1.00 | 0.71 | 15 |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | human | ai |
| --- | --- | --- |
| **human** | 3 | 12 |
| **ai** | 0 | 15 |

Misclassified (12 of 30) — P(ai) shows how confidently it was wrong:

| Text | Source | True | Predicted | P(ai) |
| --- | --- | --- | --- | --- |
| ok sounds good, see you then | human_short | human | ai | 0.60 |
| ugh my train is late again wtf | human_short | human | ai | 0.86 |
| can u send me the file when ur free? thanks | human_short | human | ai | 0.82 |
| The committee reviewed the proposal and recommends deferring the decision until the next fiscal quarter. | human_formal | human | ai | 0.98 |
| Please find attached the minutes from Tuesday's meeting; corrections should be submitted by Friday. | human_formal | human | ai | 0.98 |
| I am writing to confirm receipt of your application, which is now under review by our admissions panel. | human_formal | human | ai | 0.97 |
| Introducing the all-new Aurora blender — power, precision, and a splash of style for your kitchen. | human_marketing | human | ai | 0.98 |
| Our handcrafted candles turn any evening into a moment worth savoring. Light one and unwind. | human_marketing | human | ai | 0.97 |
| Fresh beans, roasted daily, delivered to your door. Coffee the way it was meant to be. | human_marketing | human | ai | 0.86 |
| In today's fast-paced world, staying organized is more important than ever for reaching your goals. | ambiguous | human | ai | 0.98 |
| There are many factors to consider, and ultimately the right choice depends on your own needs. | ambiguous | human | ai | 0.98 |
| Overall, it was a memorable trip that offered both challenges and a few lessons I won't forget. | ambiguous | human | ai | 0.98 |

## Machine-readable summary

```json
{
  "n_examples": 30,
  "disagreement_rate": 0.6,
  "pairwise_agreement": [
    {
      "detector_a": "desklib-ai-detector",
      "detector_b": "fakespot-ai-detector",
      "agreement": 0.4
    },
    {
      "detector_a": "desklib-ai-detector",
      "detector_b": "oxidane-ai-detector",
      "agreement": 0.9
    },
    {
      "detector_a": "fakespot-ai-detector",
      "detector_b": "oxidane-ai-detector",
      "agreement": 0.5
    }
  ],
  "calibration": {
    "desklib-ai-detector": {
      "mean_p_ai_on_human": 0.8364733333333333,
      "mean_p_ai_on_ai": 0.9386266666666667,
      "gap": 0.10215333333333343
    },
    "fakespot-ai-detector": {
      "mean_p_ai_on_human": 0.3635333333333333,
      "mean_p_ai_on_ai": 0.50798,
      "gap": 0.14444666666666667
    },
    "oxidane-ai-detector": {
      "mean_p_ai_on_human": 0.76546,
      "mean_p_ai_on_ai": 0.97906,
      "gap": 0.2136
    }
  },
  "detectors": {
    "desklib-ai-detector": {
      "accuracy": 0.5,
      "macro_f1": 0.3333,
      "latency_p50_ms": 50.72,
      "latency_p95_ms": 198.96
    },
    "fakespot-ai-detector": {
      "accuracy": 0.5667,
      "macro_f1": 0.5623,
      "latency_p50_ms": 8.47,
      "latency_p95_ms": 32.89
    },
    "oxidane-ai-detector": {
      "accuracy": 0.6,
      "macro_f1": 0.5238,
      "latency_p50_ms": 6.48,
      "latency_p95_ms": 8.48
    }
  }
}
```

---
title: SentimentScope
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Educational sentiment analysis + AI-text detection
models:
  - cardiffnlp/twitter-roberta-base-sentiment-latest
  - distilbert/distilbert-base-uncased-finetuned-sst-2-english
  - desklib/ai-text-detector-v1.01
  - fakespot-ai/roberta-base-ai-text-detection-v1
  - Oxidane/tmr-ai-text-detector
---

# SentimentScope

An educational sentiment-analysis showcase: paste text (or upload a CSV) and
see what a transformer classifier actually does — class probabilities, token
attributions, and side-by-side model comparison.

- **Analyze** — 3-class sentiment (negative / neutral / positive) from
  RoBERTa fine-tuned on ~124M tweets, with per-class confidence bars.
- **Explain** — token-level attributions via Layer Integrated Gradients:
  which words pushed the model toward its prediction.
- **Batch** — CSV upload with aggregate charts.
- **Compare** — the same text through different models (3-class social-media
  RoBERTa vs binary SST-2 DistilBERT) to see domain and label-space mismatch.
- **AI Detector** — one paragraph run through three AI-text detectors at once
  (desklib / fakespot / oxidane), with a disagreement flag and a verbatim
  uncertainty warning: detector disagreement *is* the uncertainty signal.
- **How it works** — a plain-language walkthrough of the pipeline.

## Public deployment limits

This free CPU Space is rate-limited (30 requests/min per IP) and serves an
allowlist of five models — two sentiment models plus all three AI detectors.
Clone the repo and run it locally (docker compose or the dev servers) for the
full model registry.

Backend: FastAPI + PyTorch + transformers + captum. Frontend: React + Vite.
The container serves both — the SPA via FastAPI `StaticFiles`, the API under
`/api/*`, weights baked into the image so cold starts never re-download.

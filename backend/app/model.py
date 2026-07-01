"""SentimentModel — a thin, educational wrapper around a HuggingFace classifier.

Design notes (the "why", since the "what" is short):

* Singleton lifecycle: transformer weights are ~500MB in RAM. We load them
  exactly once at server startup (see main.py lifespan) instead of per
  request — loading takes seconds, inference takes milliseconds.
* Lazy imports: torch/transformers are imported INSIDE methods, not at module
  top. That means importing this module (e.g. from unit tests or CI) costs
  nothing, and the heavy libraries are only pulled in when the model actually
  loads. CI runs the whole API test suite without torch installed.
"""


class SentimentModel:
    # A RoBERTa-base encoder fine-tuned on ~124M tweets for 3-class sentiment.
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # RoBERTa's positional embeddings cap sequence length; longer inputs are truncated.
    MAX_TOKENS = 512

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        self.labels: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run only) and load tokenizer + model weights."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Apple Silicon GPU (MPS) when available; plain CPU in Docker/CI.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.to(self.device)
        # eval() disables dropout etc. — we want deterministic inference, not training.
        self._model.eval()
        # Read label names from the model config instead of hardcoding:
        # id2label = {0: "negative", 1: "neutral", 2: "positive"} for this model.
        self.labels = [
            self._model.config.id2label[i] for i in range(self._model.config.num_labels)
        ]

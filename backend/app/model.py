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

import asyncio
from typing import Protocol

from app.model_registry import (
    ModelConfig,
    ModelTask,
    get_default_model_id,
    get_model_config,
    resolve_model_source,
)


class SentimentModel:
    # A RoBERTa-base encoder fine-tuned on ~124M tweets for 3-class sentiment.
    # Kept as a class constant for backward compatibility (Tasks 1–7 / /api/model);
    # per-instance model choice now flows through the registry ModelConfig below.
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # RoBERTa's positional embeddings cap sequence length; longer inputs are truncated.
    MAX_TOKENS = 512

    def __init__(self, config: ModelConfig | None = None) -> None:
        # Default to the registry's sentiment default so `SentimentModel()`
        # (no args) still means twitter-roberta, exactly as in Tasks 1–7.
        self._config = config or get_model_config(get_default_model_id(ModelTask.SENTIMENT))
        self.model_name = self._config.name
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        # Canonical labels from the registry — available before load() and
        # decoupled from raw HF config.id2label casing/order.
        self.labels: list[str] = list(self._config.labels)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run only) and load tokenizer + model weights."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Apple Silicon GPU (MPS) when available; plain CPU in Docker/CI.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Local weights dir if it exists, HF Hub name otherwise — never let a
        # missing local folder break a fresh clone or the Docker build.
        source = resolve_model_source(self._config)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = AutoModelForSequenceClassification.from_pretrained(source)
        self._model.to(self.device)
        # eval() disables dropout etc. — we want deterministic inference, not training.
        self._model.eval()
        # Canonical label names from the registry — NOT config.id2label. HF
        # DistilBERT reports NEGATIVE/POSITIVE (uppercase) and FinBERT's index
        # order differs; the registry tuple is the single source of truth so
        # API score keys stay stable across models.
        self.labels = list(self._config.labels)

    def predict(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts. The full flow, spelled out:

        1. Tokenize: text -> subword IDs. padding=True pads the batch to the
           longest member so it forms one rectangular tensor; truncation
           enforces the 512-token model limit.
        2. Forward pass: one batched call. Batching is THE key GPU win —
           classifying 500 texts in one tensor is dramatically faster than
           500 single-text calls, because the per-call overhead is paid once.
        3. Softmax: the model outputs logits (unnormalized scores). Softmax
           maps them to probabilities that sum to 1, which is what humans
           (and our confidence bars) actually want to read.
        """
        import torch

        assert self.is_loaded, "call load() before predict()"
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_TOKENS,
        ).to(self.device)

        # no_grad(): we're not training, so skip building the autograd graph
        # — less memory, more speed.
        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu()
        results = []
        for row in probs:
            scores = {label: round(float(p), 4) for label, p in zip(self.labels, row)}
            results.append({"label": max(scores, key=scores.get), "scores": scores})
        return results

    def explain(self, text: str) -> dict:
        """Token-level attribution via Layer Integrated Gradients (captum).

        What IG computes, in one paragraph: pick a "no information" baseline
        input (here: all padding tokens), then walk a straight line in
        embedding space from that baseline to the real input in n_steps
        increments, accumulating the model's gradient at each step. The
        integral assigns each input token a share of the change in the
        predicted class's logit. Tokens with large positive attribution
        pushed the model TOWARD its prediction; negative pushed away.
        We attach IG at the embedding layer (LayerIntegratedGradients)
        because raw token IDs are discrete — you can't differentiate
        through an integer lookup, but you can through its embedding.

        RoBERTa-only for now: LayerIntegratedGradients is wired to
        self._model.roberta.embeddings. DistilBERT uses distilbert.* and
        BERT uses bert.* — other registry models can be compared but not
        explained until per-architecture embedding hooks are added.
        """
        import torch
        from captum.attr import LayerIntegratedGradients

        assert self.is_loaded, "call load() before explain()"
        prediction = self.predict([text])[0]
        target = self.labels.index(prediction["label"])

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.MAX_TOKENS
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        def forward(ids, mask):
            return self._model(input_ids=ids, attention_mask=mask).logits

        lig = LayerIntegratedGradients(forward, self._model.roberta.embeddings)

        # Baseline = same length, but every content token replaced by <pad>,
        # keeping the <s>/</s> specials in place. "What would the model say
        # about a sentence with no words in it?"
        baseline = torch.full_like(input_ids, self._tokenizer.pad_token_id)
        baseline[0, 0] = input_ids[0, 0]
        baseline[0, -1] = input_ids[0, -1]

        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=50,  # more steps = better integral approximation, slower
        )
        # One attribution per (token, embedding_dim); collapse the embedding
        # axis and L2-normalize so the UI gets comparable magnitudes.
        scores = attributions.sum(dim=-1).squeeze(0)
        scores = scores / (torch.norm(scores) + 1e-9)

        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])
        special = {self._tokenizer.bos_token, self._tokenizer.eos_token, self._tokenizer.pad_token}
        token_attrs = [
            # "Ġ" is the BPE marker for "preceded by a space" — swap it back
            # for display so tokens rejoin into readable text.
            {"token": tok.replace("Ġ", " "), "attribution": round(float(a), 4)}
            for tok, a in zip(tokens, scores)
            if tok not in special
        ]
        return {**prediction, "tokens": token_attrs}


# Case-insensitive map from a detector's raw class name to the app's canonical
# ("human", "ai") pair. Detectors label the same two classes with different
# words — Human/AI, human/ai, real/fake — so we normalize here instead of
# trusting each checkpoint's casing or vocabulary.
_CANONICAL_DETECTOR_LABELS = {
    "human": "human",
    "real": "human",
    "ai": "ai",
    "fake": "ai",
}


def _canonicalize_detector_labels(id2label: dict[int, str], model_id: str) -> list[str]:
    """Map a softmax detector's raw id2label onto canonical labels in LOGIT
    ORDER (index 0 first).

    Read by integer index, not dict-iteration order: a checkpoint that puts
    "ai" at index 0 must still line its label up with logit 0. Unknown raw
    labels fail loudly — guessing which class means "ai" would silently invert
    every score for that model.
    """
    ordered = []
    for i in range(len(id2label)):
        raw = id2label[i]
        canonical = _CANONICAL_DETECTOR_LABELS.get(raw.strip().lower())
        if canonical is None:
            raise ValueError(
                f"Detector '{model_id}' exposes unmappable label(s) "
                f"{list(id2label.values())}; cannot canonicalize to "
                "('human', 'ai'). Refusing to guess."
            )
        ordered.append(canonical)
    return ordered


def _load_desklib_detector(source: str):
    """Instantiate desklib's custom single-logit detector, copied VERBATIM from
    its model card (models/desklib-ai-text-detector-v1.01/README.md).

    The checkpoint is a custom PreTrainedModel subclass — DeBERTa-v3-large base
    + mean pooling + a single-logit head — not a standard sequence classifier.
    AutoModelForSequenceClassification would bolt a fresh 2-logit head on top
    and drop the trained single-logit head, so it must be built explicitly.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            # Initialize the base transformer model.
            self.model = AutoModel.from_config(config)
            # Define a classifier head.
            self.classifier = nn.Linear(config.hidden_size, 1)
            # Model card calls self.init_weights(); transformers 5.x moved the
            # tied-weights bookkeeping (all_tied_weights_keys) into post_init(),
            # which then calls init_weights() itself. Calling init_weights()
            # directly on v5 raises AttributeError, so use post_init().
            self.post_init()

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Forward pass through the transformer
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            # Mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

            # Classifier
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())

            output = {"logits": logits}
            if loss is not None:
                output["loss"] = loss
            return output

    return DesklibAIDetectionModel.from_pretrained(source)


class DetectorModel:
    """Detector-family counterpart of SentimentModel: same interface the cache
    and routes rely on (load/predict/is_loaded/labels/device), but a
    task-appropriate load path and output head.

    Educational point: "binary classifier" hides two different architectures.
    A 2-logit softmax head and a 1-logit sigmoid head produce the same kind of
    answer, but conflating them mangles the probabilities — you can't softmax a
    single logit. So this wrapper branches on the registry's output_adapter:
    desklib emits ONE logit where sigmoid(logit) = P(ai); fakespot/oxidane emit
    two logits softmaxed over [human, ai]. Both funnel into the same
    {"human", "ai"} score dict so the API response shape is identical
    regardless of the underlying architecture.
    """

    # RoBERTa caps at 512; DeBERTa-v3 handles more, but inputs are already
    # bounded to 2000 chars upstream, so 512 tokens is ample and universal.
    MAX_TOKENS = 512

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model_name = config.name
        self._tokenizer = None
        self._model = None
        self.device: str | None = None
        # Canonical labels from the registry — ("human", "ai"). Available
        # before load(); the softmax path additionally derives a logit-ordered
        # label list from config.id2label at load time.
        self.labels: list[str] = list(config.labels)
        self._softmax_labels: list[str] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load tokenizer + weights via the per-architecture path the registry
        selected. Same local-dir → Hub-name fallback as SentimentModel."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        source = resolve_model_source(self._config)
        self._tokenizer = AutoTokenizer.from_pretrained(source)

        if self._config.output_adapter == "single_logit_sigmoid":
            # desklib: custom class — Auto* would mis-load the checkpoint.
            self._model = _load_desklib_detector(source)
        else:
            # fakespot/oxidane: standard sequence classifier. Derive canonical
            # labels from the checkpoint's own id2label so scores line up with
            # the logits even when a detector orders its classes ai-first.
            self._model = AutoModelForSequenceClassification.from_pretrained(source)
            self._softmax_labels = _canonicalize_detector_labels(
                self._model.config.id2label, self._config.id
            )
        self._model.to(self.device)
        # eval() disables dropout — deterministic inference, not training.
        self._model.eval()

    def predict(self, texts: list[str]) -> list[dict]:
        import torch

        assert self.is_loaded, "call load() before predict()"
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_TOKENS,
        ).to(self.device)

        with torch.no_grad():
            if self._config.output_adapter == "single_logit_sigmoid":
                # desklib's custom forward returns a plain dict, not a
                # transformers ModelOutput — read ["logits"], not .logits.
                logits = self._model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                )["logits"]
            else:
                logits = self._model(**enc).logits

        if self._config.output_adapter == "single_logit_sigmoid":
            # One logit per text: sigmoid(logit) = P(ai). Emit both sides so the
            # response shape matches the softmax detectors exactly.
            p_ai = torch.sigmoid(logits.squeeze(-1)).cpu()
            return [
                {
                    "label": "ai" if float(p) >= 0.5 else "human",
                    "scores": {"human": round(1 - float(p), 4), "ai": round(float(p), 4)},
                }
                for p in p_ai
            ]

        # softmax adapter: probabilities over the canonical, logit-ordered labels.
        probs = torch.softmax(logits, dim=-1).cpu()
        results = []
        for row in probs:
            scores = {label: round(float(p), 4) for label, p in zip(self._softmax_labels, row)}
            results.append({"label": max(scores, key=scores.get), "scores": scores})
        return results


class BaseTextModel(Protocol):
    """What the cache and routes actually need from a model. SentimentModel
    satisfies this today; DetectorModel (Task 19) will too. The cache must
    not assume every model is a SentimentModel — detector checkpoints have
    different architectures and output heads."""

    labels: list[str]
    device: str | None
    is_loaded: bool

    def load(self) -> None: ...
    def predict(self, texts: list[str]) -> list[dict]: ...


def build_model(cfg: ModelConfig) -> BaseTextModel:
    if cfg.task == ModelTask.SENTIMENT:
        return SentimentModel(cfg)
    if cfg.task == ModelTask.AI_TEXT_DETECTION:
        return DetectorModel(cfg)
    raise ValueError(f"Unsupported model task: {cfg.task}")


async def get_or_load_model(app, model_id: str) -> BaseTextModel:
    if model_id in app.state.model_cache:
        return app.state.model_cache[model_id]
    if model_id not in app.state.model_locks:
        app.state.model_locks[model_id] = asyncio.Lock()
    async with app.state.model_locks[model_id]:
        # Double-checked: another coroutine may have finished loading while we
        # awaited the lock, so re-check the cache before paying to load again.
        if model_id not in app.state.model_cache:
            cfg = get_model_config(model_id)
            m = build_model(cfg)
            # ~500MB of weights load synchronously; run it off the event loop
            # so other requests (health, in-flight analyze) stay responsive.
            await asyncio.to_thread(m.load)
            app.state.model_cache[model_id] = m
        return app.state.model_cache[model_id]

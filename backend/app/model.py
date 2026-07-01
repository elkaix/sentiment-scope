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

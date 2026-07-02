import { Suspense, use, useState } from "react";
import { getModelInfo } from "../api";
import type { ModelInfo } from "../api";

/**
 * The educational heart of the app: a plain-language walkthrough of what
 * happens between "user types a sentence" and "the UI shows 87% positive".
 */
export default function HowItWorks() {
  // React 19 data fetching: a stable promise created once, read with use()
  // inside a Suspense boundary. A backend failure resolves to null and the
  // article simply renders without the live spec footer.
  const [infoPromise] = useState(() => getModelInfo().catch(() => null));

  return (
    <article className="space-y-8 text-[15px] leading-relaxed text-slate-600">
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">1. Tokenization</h2>
        <p>
          Neural networks can't read text; they read numbers. A byte-pair-encoding (BPE)
          tokenizer splits your sentence into subword pieces ("incredible" might become
          "incred" + "ible") and maps each piece to an integer ID from a ~50k-entry
          vocabulary. Rare words split into more pieces; common words stay whole. This is
          why the explanation view highlights sub-word chunks rather than whole words.
        </p>
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">2. The transformer encoder</h2>
        <p>
          Those IDs pass through RoBERTa: 12 layers of self-attention. Each layer lets
          every token "look at" every other token and update its representation based on
          context, so the "bank" in "river bank" and "bank account" ends up with different
          vectors. After 12 rounds of this, the model has a contextual summary of the whole
          sentence.
        </p>
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">3. Classification head + softmax</h2>
        <p>
          A small linear layer maps that summary to three raw scores (logits), one per
          class. Softmax exponentiates and normalizes them into probabilities that sum
          to 1. Those are the confidence bars you see on the Analyze tab. High entropy
          (three similar bars) means the model is genuinely unsure.
        </p>
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">4. Explainability: Integrated Gradients</h2>
        <p>
          To answer "which words made it say that?", we use Integrated Gradients: start
          from an empty baseline sentence (all padding tokens), interpolate step-by-step
          toward the real input in embedding space, and accumulate the gradients of the
          predicted class along the way. Each token gets a share of the credit. Green
          tokens pushed the model toward its answer, red pushed away.
        </p>
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">Honest limitations</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>The model was trained on tweets; long or formal text is out-of-domain.</li>
          <li>Inputs are truncated to 512 tokens; anything beyond is invisible to the model.</li>
          <li>English only; sarcasm and irony remain hard.</li>
          <li>IG is an approximation (50 integration steps), not a ground-truth explanation.</li>
        </ul>
      </section>
      <section className="space-y-2">
        <h2 className="text-base font-semibold text-slate-900">Why models disagree</h2>
        <p>
          The Compare tab runs one sentence through models trained on different data:
          tweets, movie reviews, financial news. Disagreement is the interesting part.
          A binary model has no neutral class to fall back on, so it must pick a side,
          and a finance model reads "revenue outlook improved" very differently from a
          general-purpose one. Confidence numbers are not comparable across models with
          different label sets: treat each readout as that model's opinion, not as
          ground truth.
        </p>
      </section>
      <Suspense fallback={null}>
        <ModelSpecFooter promise={infoPromise} />
      </Suspense>
    </article>
  );
}

function ModelSpecFooter({ promise }: { promise: Promise<ModelInfo | null> }) {
  const info = use(promise);
  if (!info) return null;

  return (
    <footer className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <h2 className="mb-3 font-mono text-xs font-semibold text-slate-500">live model spec</h2>
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 font-mono text-xs">
        <dt className="text-slate-400">model</dt>
        <dd className="break-all text-slate-700">{info.name}</dd>
        <dt className="text-slate-400">labels</dt>
        <dd className="text-slate-700">{info.labels.join(" / ")}</dd>
        <dt className="text-slate-400">max tokens</dt>
        <dd className="tabular-nums text-slate-700">{info.max_tokens}</dd>
        <dt className="text-slate-400">device</dt>
        <dd className="text-slate-700">{info.device}</dd>
      </dl>
      <p className="mt-3 text-xs leading-relaxed text-slate-500">{info.description}</p>
    </footer>
  );
}

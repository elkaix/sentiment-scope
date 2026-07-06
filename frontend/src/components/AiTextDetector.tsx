import { useActionState, useState } from "react";
import { compareAiDetectors } from "../api";
import type { AiDetectItem } from "../api";
import ConfidenceBars from "./ConfidenceBars";

/**
 * AI text detector tab. The teaching point is disagreement: one input is run
 * through EVERY detector at once (compareAiDetectors with no model_ids), so
 * users see that "is this AI?" has no single answer. Two guardrails are load
 * bearing here, not decoration:
 *   - the API's own uncertainty warning is rendered verbatim and prominently,
 *     so a probability can never read as proof of authorship;
 *   - when detectors disagree we say so out loud and frame it as uncertainty.
 */

// A generic, formal, hedge-heavy paragraph: the register detectors flag this
// style readily, so the tab demonstrates likely disagreement without making
// the user hunt for AI-ish text.
const DEFAULT_TEXT =
  "The committee recommends a phased operational rollout, with quarterly controls, " +
  "cross-functional reporting, and measurable adoption targets for each business unit.";

interface DetectState {
  rows: AiDetectItem[];
  disagreement: boolean;
  warning: string;
  reviewedText: string;
  error: string | null;
}

const EMPTY: DetectState = {
  rows: [],
  disagreement: false,
  warning: "",
  reviewedText: "",
  error: null,
};

export default function AiTextDetector() {
  const [text, setText] = useState(DEFAULT_TEXT);

  // Same form-action lifecycle as the Compare tab: submit -> pending -> result
  // or error, owned by useActionState (no hand-rolled loading/error state).
  const [state, formAction, isPending] = useActionState<DetectState>(async () => {
    try {
      const res = await compareAiDetectors(text);
      return {
        rows: res.results,
        disagreement: res.disagreement,
        warning: res.warning,
        reviewedText: text,
        error: null,
      };
    } catch (e) {
      return { ...EMPTY, error: e instanceof Error ? e.message : "Detection failed" };
    }
  }, EMPTY);

  return (
    <div className="space-y-6">
      {/* Always-on educational copy. This is the app's own framing, distinct
          from the API warning callout that appears with results below. */}
      <div className="space-y-1 text-sm leading-relaxed text-slate-500">
        <p>AI detectors are probabilistic. Treat this as a model signal, not proof of authorship.</p>
        <p>
          <span className="font-mono text-slate-600">P(ai)</span> is each detector's estimated
          probability that the text is AI-generated: a calibrated-ish score, never a verdict.
        </p>
      </div>

      <form action={formAction} className="space-y-4">
        <textarea
          className="w-full rounded-lg border border-slate-300 bg-white p-3 focus:border-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
          rows={5}
          maxLength={2000}
          aria-label="Text to check for AI authorship"
          placeholder="Paste text to run through every AI detector"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          type="submit"
          className="min-h-11 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:opacity-50"
          disabled={isPending || !text.trim()}
        >
          {isPending ? "Detecting…" : "Detect AI text"}
        </button>
      </form>

      {state.error && (
        <p role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700">
          {state.error}
        </p>
      )}

      {state.rows.length > 0 && (
        <div className="space-y-4">
          <DetectionHighlight text={state.reviewedText} evidence={state.rows.find(isAiEvidence)} />

          {/* Prominent uncertainty callout. Renders the backend's warning
              string verbatim so the disclaimer can never drift from the model
              that actually ran, a Global Constraint rather than fine print. */}
          <div
            role="note"
            aria-label="Detector uncertainty warning"
            className="rounded-lg border border-amber-300 bg-amber-50 p-4"
          >
            <p className="mb-1 font-mono text-xs font-semibold uppercase tracking-wide text-amber-800">
              Uncertainty
            </p>
            <p className="text-sm leading-relaxed text-amber-900">{state.warning}</p>
          </div>

          {state.disagreement && (
            <div
              role="status"
              className="rounded-lg border border-slate-300 bg-slate-50 p-3 text-sm text-slate-700"
            >
              <span className="font-semibold text-slate-900">Detectors disagree.</span> They reached
              different verdicts on this text. Read that as a strong signal of uncertainty, not a
              tie to break.
            </div>
          )}

          <div className="space-y-4">
            {state.rows.map((r) => (
              <DetectorReadout key={r.model_id} item={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function isAiEvidence(item: AiDetectItem) {
  return item.label.toLowerCase() === "ai";
}

function DetectionHighlight({ text, evidence }: { text: string; evidence: AiDetectItem | undefined }) {
  if (!evidence) return null;

  return (
    <section
      role="region"
      aria-label="AI-likely highlighted text"
      className="space-y-3 rounded-lg border border-amber-300 bg-amber-50 p-4"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-semibold text-amber-950">AI-likely text</p>
        <p className="font-mono text-xs tabular-nums text-amber-900">
          {(evidence.confidence * 100).toFixed(1)}% confidence
        </p>
      </div>
      <mark className="block whitespace-pre-wrap rounded-md bg-amber-100 px-3 py-2 text-base leading-7 text-amber-950">
        {text}
      </mark>
      <p className="text-xs leading-relaxed text-amber-900">
        Highlight applies to the submitted passage as a whole. This detector does not return
        word-level authorship evidence.
      </p>
    </section>
  );
}

function DetectorReadout({ item }: { item: AiDetectItem }) {
  return (
    <section
      role="group"
      aria-label={item.model_id}
      className="space-y-3 rounded-xl border border-slate-200 bg-white p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-mono text-[13px] font-semibold text-slate-900">{item.name}</h3>
          <p className="text-xs text-slate-500">{item.domain}</p>
        </div>
        {/* Verdict badge stays ink, not chroma: colour is reserved for
            sentiment across this app, and a human/ai call is not sentiment. */}
        <span className="shrink-0 rounded-full bg-slate-900 px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
          {item.label}
        </span>
      </div>
      {/* Full {human, ai} distribution, not just the winner: "ai 51%" and
          "ai 99%" are very different answers. */}
      <ConfidenceBars scores={item.scores} />
      <dl className="flex gap-8 border-t border-slate-100 pt-3 font-mono text-xs">
        <div className="space-y-0.5">
          <dt className="text-slate-400">confidence</dt>
          <dd className="tabular-nums text-slate-700">{(item.confidence * 100).toFixed(1)}%</dd>
        </div>
        <div className="space-y-0.5">
          <dt className="text-slate-400">latency</dt>
          <dd className="tabular-nums text-slate-700">{item.latency_ms.toFixed(0)} ms</dd>
        </div>
      </dl>
      {item.note && <p className="text-xs leading-relaxed text-slate-500">{item.note}</p>}
    </section>
  );
}

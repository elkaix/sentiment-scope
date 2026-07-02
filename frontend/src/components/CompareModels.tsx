import { Suspense, use, useActionState, useState } from "react";
import { compareModels, getModels } from "../api";
import type { CompareItem, ModelSummary } from "../api";
import ConfidenceBars from "./ConfidenceBars";

/**
 * Mirrors the backend's DEFAULT_COMPARE_MODELS: only these two are loaded at
 * startup. The other registry entries (finbert, xlm-twitter) are loaded
 * lazily by the backend on first use, so the UI keeps them off by default
 * and warns about the one-time load cost.
 */
const DEFAULT_COMPARE = ["twitter-roberta", "distilbert-sst2"];

const LABEL_BADGE: Record<string, string> = {
  negative: "bg-red-100 text-red-700",
  neutral: "bg-slate-100 text-slate-700",
  positive: "bg-emerald-100 text-emerald-700",
};

interface CompareState {
  rows: CompareItem[];
  error: string | null;
}

export default function CompareModels() {
  // React 19 data fetching: create the registry promise once (useState
  // initializer runs a single time, so its identity is stable) and let a
  // child read it with use() under Suspense - no useEffect/setState dance.
  const [registryPromise] = useState(() =>
    getModels("sentiment")
      .then((r) => r.models)
      .catch(() => [] as ModelSummary[]),
  );

  return (
    <Suspense fallback={<p className="text-sm text-slate-400">Loading model registry…</p>}>
      <CompareForm registryPromise={registryPromise} />
    </Suspense>
  );
}

function CompareForm({ registryPromise }: { registryPromise: Promise<ModelSummary[]> }) {
  const registry = use(registryPromise);
  const [text, setText] = useState("Our quarterly revenue outlook improved");
  const [selected, setSelected] = useState<string[]>(DEFAULT_COMPARE);

  // The comparison is a classic form action: submit -> pending -> result or
  // error. useActionState owns that lifecycle, so there is no hand-rolled
  // loading/error useState pair here.
  const [result, formAction, isPending] = useActionState<CompareState>(
    async () => {
      try {
        return { rows: (await compareModels(text, selected)).results, error: null };
      } catch (e) {
        return { rows: [], error: e instanceof Error ? e.message : "Compare failed" };
      }
    },
    { rows: [], error: null },
  );

  const toggle = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  if (registry.length === 0) {
    return <p className="text-sm text-slate-500">Model registry unavailable. Is the backend running?</p>;
  }

  return (
    <div className="space-y-6">
      <form action={formAction} className="space-y-5">
        <textarea
          className="w-full rounded-lg border border-slate-300 bg-white p-3 focus:border-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
          rows={3}
          aria-label="Text to compare"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <fieldset className="space-y-1">
          <legend className="mb-2 text-sm font-medium text-slate-700">Models to compare</legend>
          {registry.map((m) => (
            <label
              key={m.id}
              className="flex cursor-pointer items-baseline gap-2.5 rounded-md px-2 py-1.5 text-sm hover:bg-slate-50"
            >
              <input
                type="checkbox"
                className="relative top-0.5 size-4 accent-slate-900"
                checked={selected.includes(m.id)}
                onChange={() => toggle(m.id)}
              />
              <span className="font-mono text-[13px] font-medium text-slate-900">{m.id}</span>
              <span className="text-slate-500">{m.domain}</span>
              {!m.loaded && !m.default && (
                <span className="text-xs text-amber-700">loads on first use, may take a moment</span>
              )}
            </label>
          ))}
        </fieldset>
        <button
          type="submit"
          className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:opacity-50"
          disabled={isPending || !text.trim() || selected.length === 0}
        >
          {isPending ? "Comparing…" : "Compare models"}
        </button>
      </form>

      {result.error && (
        <p role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700">
          {result.error}
        </p>
      )}

      {result.rows.length > 0 && (
        // Side-by-side readouts make disagreement the point of the page:
        // same sentence, different training data, different verdicts.
        <div className="grid gap-4 sm:grid-cols-2">
          {result.rows.map((r) => (
            <ModelReadout key={r.model_id} item={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function ModelReadout({ item }: { item: CompareItem }) {
  return (
    <section
      role="group"
      aria-label={item.model_id}
      className="space-y-3 rounded-xl border border-slate-200 bg-white p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate font-mono text-[13px] font-semibold text-slate-900">{item.model_id}</h3>
          <p className="text-xs text-slate-500">{item.domain}</p>
        </div>
        <span
          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize ${LABEL_BADGE[item.label] ?? "bg-slate-100 text-slate-700"}`}
        >
          {item.label}
        </span>
      </div>
      {/* Dynamic scores: a 2-class model renders exactly its own keys, so a
          missing "neutral" bar is honest, not a bug. */}
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

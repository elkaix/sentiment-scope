import { useState } from "react";
import { analyze, explainText } from "../api";
import type { AnalyzeResult, ExplainResult } from "../api";
import ConfidenceBars from "./ConfidenceBars";
import TokenHeatmap from "./TokenHeatmap";

const LABEL_BADGE: Record<string, string> = {
  negative: "bg-red-100 text-red-700",
  neutral: "bg-slate-100 text-slate-700",
  positive: "bg-emerald-100 text-emerald-700",
};

export default function AnalyzeForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Both buttons share one submit path; `withExplain` decides which endpoint.
  // Explain is opt-in because Integrated Gradients costs ~50 forward passes.
  const run = async (withExplain: boolean) => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      if (withExplain) {
        const res = await explainText(text);
        setResult(res);
        setExplanation(res);
      } else {
        setResult(await analyze(text));
      }
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <textarea
        className="w-full rounded-lg border border-slate-300 bg-white p-3 focus:border-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
        rows={4}
        maxLength={2000}
        placeholder="Type or paste text to analyze… e.g. 'The battery life on this phone is incredible'"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex gap-3">
        <button
          className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:opacity-50"
          disabled={loading || !text.trim()}
          onClick={() => run(false)}
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
        <button
          className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:opacity-50"
          disabled={loading || !text.trim()}
          onClick={() => run(true)}
          title="Slower: runs Integrated Gradients to show which words drove the prediction"
        >
          Analyze + Explain
        </button>
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <div className="space-y-4 rounded-lg border border-slate-200 p-4">
          <span
            className={`inline-block rounded-full px-3 py-1 text-sm font-semibold capitalize ${LABEL_BADGE[result.label] ?? ""}`}
          >
            {result.label}
          </span>
          <ConfidenceBars scores={result.scores} />
          {explanation && (
            <div className="space-y-1 border-t border-slate-200 pt-4">
              <p className="text-sm font-medium text-slate-600">
                Which words drove this prediction (Integrated Gradients):
              </p>
              <TokenHeatmap tokens={explanation.tokens} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

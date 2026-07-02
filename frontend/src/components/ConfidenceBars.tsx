import type { DynamicScores, Scores } from "../api";

/**
 * Horizontal bars for class probabilities. Renders Object.entries(scores)
 * so it works for binary (DistilBERT) and 3-class models. Showing the full softmax distribution (not just the winner) is deliberate: "positive 51%"
 * and "positive 98%" are very different answers, and hiding that nuance is
 * how ML demos mislead people.
 */

const BAR_COLOR: Record<string, string> = {
  negative: "bg-red-500",
  neutral: "bg-slate-400",
  positive: "bg-emerald-500",
};

const fallbackColor = "bg-indigo-400";

export default function ConfidenceBars({ scores }: { scores: Scores | DynamicScores }) {
  return (
    <div className="space-y-2">
      {Object.entries(scores).map(([label, value]) => (
        <div key={label} className="flex items-center gap-3">
          <span className="w-20 text-sm capitalize text-slate-600">{label}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-slate-200">
            <div
              className={`h-full ${BAR_COLOR[label] ?? fallbackColor} transition-all`}
              style={{ width: `${value * 100}%` }}
            />
          </div>
          <span className="w-14 text-right text-sm tabular-nums text-slate-600">
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

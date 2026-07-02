import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { BatchResult } from "../api";

const LABEL_COLOR: Record<string, string> = {
  negative: "#ef4444",
  neutral: "#94a3b8",
  positive: "#10b981",
};

/**
 * Two views of the same batch: label counts (how many rows landed in each
 * class) and mean softmax scores (how confident the model was on average).
 * Both matter — 100 barely-positive rows and 100 emphatic ones have the
 * same counts but very different mean scores.
 */
export default function AggregateCharts({ aggregates }: { aggregates: BatchResult["aggregates"] }) {
  const countData = Object.entries(aggregates.counts).map(([label, count]) => ({ label, count }));
  const meanData = Object.entries(aggregates.mean_scores).map(([label, mean]) => ({ label, mean }));

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-600">Sentiment counts</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={countData}>
            <XAxis dataKey="label" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count">
              {countData.map((d) => (
                <Cell key={d.label} fill={LABEL_COLOR[d.label]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-600">Mean confidence per class</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={meanData}>
            <XAxis dataKey="label" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="mean">
              {meanData.map((d) => (
                <Cell key={d.label} fill={LABEL_COLOR[d.label]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

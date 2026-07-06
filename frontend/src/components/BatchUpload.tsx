import { Suspense, lazy, useRef, useState } from "react";
import { analyzeCsv } from "../api";
import type { BatchResult } from "../api";

const AggregateCharts = lazy(() => import("./AggregateCharts"));

const LABEL_TEXT: Record<string, string> = {
  negative: "text-red-600",
  neutral: "text-slate-500",
  positive: "text-emerald-600",
};

export default function BatchUpload() {
  const [result, setResult] = useState<BatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      // The file goes straight to the backend, which owns CSV parsing and
      // validation, one source of truth for what a valid upload is.
      setResult(await analyzeCsv(file));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border-2 border-dashed border-slate-300 p-8 text-center">
        <p className="mb-3 text-slate-600">
          Upload a CSV with a <code className="rounded bg-slate-100 px-1">text</code> column
          (max 500 rows)
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="mx-auto block text-sm"
          onChange={(e) => onFile(e.target.files?.[0])}
          disabled={loading}
        />
        {loading && <p className="mt-3 text-slate-600">Analyzing batch…</p>}
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <>
          <Suspense fallback={<p className="text-sm text-slate-500">Preparing aggregate charts...</p>}>
            <AggregateCharts aggregates={result.aggregates} />
          </Suspense>
          <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-50">
                <tr>
                  <th className="p-2">Text</th>
                  <th className="p-2">Label</th>
                  <th className="p-2 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    <td className="max-w-md truncate p-2" title={r.text}>{r.text}</td>
                    <td className={`p-2 font-medium capitalize ${LABEL_TEXT[r.label] ?? ""}`}>{r.label}</td>
                    <td className="p-2 text-right tabular-nums">
                      {(Math.max(...Object.values(r.scores)) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

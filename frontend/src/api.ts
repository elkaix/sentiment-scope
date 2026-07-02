/**
 * Typed API client — the single place the frontend talks to the backend.
 * Components import these functions and never touch fetch() directly, so
 * error handling and response typing live in exactly one file.
 *
 * All paths are relative (/api/...): the Vite dev server proxies them to
 * FastAPI locally, and nginx does the same inside Docker. The UI never
 * needs to know where the backend lives.
 */

/** Strict 3-class scores for default-model analyze/batch endpoints. */
export interface Scores {
  negative: number;
  neutral: number;
  positive: number;
}

export interface AnalyzeResult {
  label: string;
  scores: Scores;
}

/** Dynamic scores for compare — keys vary by model (binary vs 3-class). */
export type DynamicScores = Record<string, number>;

export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface ExplainResult extends AnalyzeResult {
  tokens: TokenAttribution[];
}

export interface BatchItem extends AnalyzeResult {
  text: string;
}

export interface BatchResult {
  results: BatchItem[];
  aggregates: { counts: Record<string, number>; mean_scores: Scores };
}

export interface ModelInfo {
  name: string;
  labels: string[];
  max_tokens: number;
  device: string;
  description: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  task: string;
  labels: string[];
  domain: string;
  note: string;
  default: boolean;
  loaded: boolean;
}

export interface CompareItem {
  model_id: string;
  name: string;
  domain: string;
  label: string;
  scores: DynamicScores;
  confidence: number;
  latency_ms: number;
  note: string;
}

export type AiDetectionScores = Record<string, number>;

export interface AiDetectItem {
  model_id: string;
  name: string;
  domain: string;
  label: string;
  scores: AiDetectionScores;
  confidence: number;
  latency_ms: number;
  note: string;
}

export interface AiDetectCompareResponse {
  results: AiDetectItem[];
  disagreement: boolean;
  warning: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    // FastAPI puts human-readable errors in { detail } — surface that
    // instead of a bare status code whenever it's available.
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const analyze = (text: string) => request<AnalyzeResult>("/api/analyze", postJson({ text }));

export const explainText = (text: string) => request<ExplainResult>("/api/explain", postJson({ text }));

export const analyzeCsv = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  // Note: no Content-Type header — the browser sets multipart boundaries itself.
  return request<BatchResult>("/api/analyze/csv", { method: "POST", body: form });
};

export const getModelInfo = () => request<ModelInfo>("/api/model");

export const getModels = (task?: "sentiment" | "ai_text_detection") =>
  request<{ models: ModelSummary[] }>(
    task ? `/api/models?task=${task}` : "/api/models",
  );

export const compareModels = (text: string, model_ids?: string[]) =>
  request<{ results: CompareItem[] }>("/api/compare", postJson({ text, model_ids }));

export const compareAiDetectors = (text: string, model_ids?: string[]) =>
  request<AiDetectCompareResponse>("/api/ai-detect/compare", postJson({ text, model_ids }));

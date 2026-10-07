// Metrics API client. Mirrors the fetch wrapper pattern in features/scene/api.ts.

export interface ModelSplitMetrics {
  model: string;
  dataset: string;
  split: string;
  protocol: string;
  crop: string | null;
  gt_range_m: number[] | null;
  num_frames: number;
  frames_evaluated?: number;
  frames_invalid?: number;
  /** aggregate metric values; NaN has been washed to null upstream */
  aggregate: Record<string, number | null>;
  per_frame: PerFramePoint[];
}

export interface PerFramePoint {
  index?: number;
  n?: number;
  [key: string]: unknown;
}

export interface MetricsSummary {
  available?: boolean;
  summary?: unknown;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export function fetchSplitMetrics(ds: string, split: string): Promise<ModelSplitMetrics[]> {
  return getJSON<ModelSplitMetrics[]>(
    `/api/metrics/${encodeURIComponent(ds)}/${encodeURIComponent(split)}`,
  );
}

export function fetchMetricsSummary(): Promise<MetricsSummary> {
  return getJSON<MetricsSummary>("/api/metrics/summary");
}

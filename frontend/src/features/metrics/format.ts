// Metric metadata: display order, labels, best-value direction, formatting.

export type MetricDirection = "min" | "max" | "one";

export interface MetricMeta {
  key: string;
  label: string;
  direction: MetricDirection;
}

/** Preferred column order; unknown keys are appended alphabetically. */
export const METRIC_ORDER: MetricMeta[] = [
  { key: "abs_rel", label: "abs_rel", direction: "min" },
  { key: "abs_mean_m", label: "abs_mean (m)", direction: "min" },
  { key: "rmse_m", label: "RMSE (m)", direction: "min" },
  { key: "rmse_log", label: "RMSE log", direction: "min" },
  { key: "sq_rel", label: "sq_rel", direction: "min" },
  { key: "delta1", label: "δ1", direction: "max" },
  { key: "delta2", label: "δ2", direction: "max" },
  { key: "delta3", label: "δ3", direction: "max" },
  { key: "scale_median", label: "尺度中位数", direction: "one" },
  { key: "scale_log_std", label: "尺度 log σ", direction: "min" },
  { key: "abs_mean", label: "abs_mean", direction: "min" },
  { key: "d1", label: "d1", direction: "max" },
  { key: "d1_s", label: "d1_s", direction: "max" },
  { key: "d1_us", label: "d1_us", direction: "max" },
  { key: "d1_seq_a", label: "d1_seq_a", direction: "max" },
  { key: "abs_rel_s", label: "abs_rel_s", direction: "min" },
  { key: "abs_mean_s", label: "abs_mean_s", direction: "min" },
  { key: "abs_rel_a", label: "abs_rel_a", direction: "min" },
  { key: "abs_mean_a", label: "abs_mean_a", direction: "min" },
  { key: "abs_rel_us", label: "abs_rel_us", direction: "min" },
  { key: "abs_mean_us", label: "abs_mean_us", direction: "min" },
  { key: "abs_rel_da", label: "abs_rel_da", direction: "min" },
  { key: "abs_mean_da", label: "abs_mean_da", direction: "min" },
  { key: "abs_rel_seq_a", label: "abs_rel_seq_a", direction: "min" },
  { key: "abs_mean_seq_a", label: "abs_mean_seq_a", direction: "min" },
  { key: "scale_error", label: "scale_error", direction: "min" },
  { key: "median_ratio", label: "median_ratio", direction: "one" },
];

export function metricMeta(key: string): MetricMeta {
  return METRIC_ORDER.find((m) => m.key === key) ?? { key, label: key, direction: "min" };
}

/** Union of aggregate keys across models, in preferred order. */
export function metricKeys(models: { aggregate: Record<string, number | null> }[]): string[] {
  const keys = new Set<string>();
  for (const m of models) for (const k of Object.keys(m.aggregate)) keys.add(k);
  const known = METRIC_ORDER.map((m) => m.key).filter((k) => keys.has(k));
  const extra = [...keys].filter((k) => !METRIC_ORDER.some((m) => m.key === k)).sort();
  return [...known, ...extra];
}

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function fmt(v: unknown, digits = 3): string {
  return isNum(v) ? v.toFixed(digits) : "—";
}

/** score for best-value comparison; lower is better. null = not eligible. */
export function bestScore(v: number | null, dir: MetricDirection): number | null {
  if (v === null) return null;
  if (dir === "min") return v;
  if (dir === "max") return -v;
  return Math.abs(v - 1); // median_ratio: closest to 1 wins
}

/** per-column best value (raw metric value) for highlight; null when no model qualifies. */
export function bestValues(
  models: { aggregate: Record<string, number | null> }[],
  keys: string[],
): Map<string, number> {
  const out = new Map<string, number>();
  for (const key of keys) {
    const { direction } = metricMeta(key);
    let best: number | null = null;
    let bestSc = Infinity;
    for (const m of models) {
      const v = m.aggregate[key];
      if (!isNum(v)) continue;
      const sc = bestScore(v, direction);
      if (sc !== null && sc < bestSc) {
        bestSc = sc;
        best = v;
      }
    }
    if (best !== null) out.set(key, best);
  }
  return out;
}

/** downsample an array to at most max points by striding. */
export function downsample<T>(arr: T[], max: number): T[] {
  if (arr.length <= max) return arr;
  const stride = Math.ceil(arr.length / max);
  const out: T[] = [];
  for (let i = 0; i < arr.length; i += stride) out.push(arr[i]);
  return out;
}

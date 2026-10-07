import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Skeleton } from "../../components/ui/skeleton";
import { Select } from "../../components/ui/select";
import { fetchMetricsSummary, type MetricsSummary } from "./api";
import { fmt, isNum, metricMeta } from "./format";

const AXIS_TICK = { fill: "var(--fg-muted)", fontSize: 11 };

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: MetricsSummary };

interface Row {
  key: string;
  model: string;
  value: number;
}

type Dict = Record<string, unknown>;

function object(value: unknown): Dict | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Dict)
    : null;
}

function collectRows(summary: unknown, metric: string): Row[] {
  const root = object(summary);
  if (!root) return [];
  const rows: Row[] = [];
  const datasets = object(root.datasets);
  if (datasets) {
    for (const [dataset, datasetValue] of Object.entries(datasets)) {
      const splits = object(object(datasetValue)?.splits);
      if (!splits) continue;
      for (const [split, splitValue] of Object.entries(splits)) {
        const models = object(object(splitValue)?.models);
        if (!models) continue;
        for (const [model, modelValue] of Object.entries(models)) {
          const doc = object(modelValue);
          const aggregate = object(doc?.raw_frame_mean) ?? object(doc?.aggregate);
          const value = aggregate?.[metric];
          if (isNum(value)) rows.push({ key: `${dataset}/${split}`, model, value });
        }
      }
    }
    return rows;
  }
  // Legacy flat summary: "dataset/split" -> model -> aggregate.
  for (const [key, modelsValue] of Object.entries(root)) {
    const models = object(modelsValue);
    if (!models) continue;
    for (const [model, aggregateValue] of Object.entries(models)) {
      const value = object(aggregateValue)?.[metric];
      if (isNum(value)) rows.push({ key, model, value });
    }
  }
  return rows;
}

/** cross-split overview: best model per split for a chosen metric (from /api/metrics/summary) */
export function SummaryOverview() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [metric, setMetric] = useState("abs_rel");

  useEffect(() => {
    let alive = true;
    fetchMetricsSummary()
      .then((d) => alive && setState({ kind: "ready", data: d }))
      .catch((e) => alive && setState({ kind: "error", message: String(e) }));
    return () => {
      alive = false;
    };
  }, []);

  const { rows, available } = useMemo(() => {
    if (state.kind !== "ready") return { rows: [] as Row[], available: false };
    const s = state.data;
    if (s.available === false || !s.summary) return { rows: [], available: false };
    const dir = metricMeta(metric).direction;
    const candidates = collectRows(s.summary, metric);
    const rs: Row[] = [];
    const bySplit = new Map<string, Row[]>();
    for (const row of candidates) bySplit.set(row.key, [...(bySplit.get(row.key) ?? []), row]);
    for (const [key, models] of bySplit) {
      let best: Row | null = null;
      let bestSc = Infinity;
      for (const row of models) {
        const { model, value: v } = row;
        if (!isNum(v)) continue;
        const sc = dir === "max" ? -v : dir === "one" ? Math.abs(v - 1) : v;
        if (sc < bestSc) {
          bestSc = sc;
          best = { key, model, value: v };
        }
      }
      if (best) rs.push(best);
    }
    return { rows: rs, available: true };
  }, [state, metric]);

  if (state.kind === "loading") return <Skeleton className="h-72" />;
  if (state.kind === "error")
    return <div className="card text-[13px] text-danger">/api/metrics/summary 加载失败:{state.message}</div>;
  if (!available || rows.length === 0)
    return <div className="card text-[13px] text-fg-muted">暂无指标数据（/api/metrics/summary 不可用）。</div>;

  const chartData = [...rows].sort((a, b) => a.value - b.value).map((r) => ({
    split: r.key,
    value: r.value,
  }));

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-[12px] text-fg-muted">指标</span>
        <Select.Root value={metric} onValueChange={setMetric}>
          <Select.Trigger className="w-44 font-mono">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {["abs_rel", "abs_mean_m", "rmse_m", "delta1", "delta2", "delta3", "sq_rel"].map(
              (k) => (
                <Select.Item key={k} value={k}>
                  {metricMeta(k).label}
                </Select.Item>
              ),
            )}
          </Select.Content>
        </Select.Root>
        <span className="text-[12px] text-fg-muted">各 split 的最佳模型（{rows.length} 个 split）</span>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <div className="card max-h-[420px] overflow-auto p-0">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 bg-card">
              <tr className="h-9 text-left text-fg-muted">
                <th className="px-3 font-medium">dataset / split</th>
                <th className="px-3 font-medium">最佳模型</th>
                <th className="px-3 text-right font-medium">{metricMeta(metric).label}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} className="h-9 border-t border-border">
                  <td className="px-3 font-mono">{r.key}</td>
                  <td className="max-w-56 truncate px-3 font-mono">{r.model}</td>
                  <td className="num px-3 text-right">{fmt(r.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <div style={{ height: Math.max(240, chartData.length * 30 + 60) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[...chartData].reverse()}
                layout="vertical"
                margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical horizontal={false} />
                <XAxis
                  type="number"
                  tick={AXIS_TICK}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  tickFormatter={(v: number) => v.toPrecision(2)}
                />
                <YAxis
                  type="category"
                  dataKey="split"
                  width={150}
                  tick={{ ...AXIS_TICK, fontFamily: "var(--font-mono)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <RTooltip
                  cursor={{ fill: "var(--card-2)" }}
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "var(--fg)", fontFamily: "var(--font-mono)" }}
                  itemStyle={{ color: "var(--fg-muted)" }}
                  formatter={(v) => [fmt(v), metricMeta(metric).label]}
                />
                <Bar dataKey="value" fill="var(--success)" isAnimationActive={false}>
                  <LabelList
                    dataKey="value"
                    position="right"
                    fill="var(--fg-muted)"
                    fontSize={10}
                    formatter={(v: unknown) => (isNum(v) ? v.toPrecision(3) : "")}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

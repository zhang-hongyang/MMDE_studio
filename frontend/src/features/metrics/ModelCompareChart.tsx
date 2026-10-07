import { useMemo, useState } from "react";
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
import { Select } from "../../components/ui/select";
import type { ModelSplitMetrics } from "./api";
import { fmt, isNum, metricKeys, metricMeta } from "./format";

const AXIS_TICK = { fill: "var(--fg-muted)", fontSize: 11 };

interface Props {
  models: ModelSplitMetrics[];
}

/** Bar chart comparing one metric across models of a split. */
export function ModelCompareChart({ models }: Props) {
  const keys = useMemo(() => metricKeys(models), [models]);
  const numericKeys = keys.filter((k) => models.some((m) => isNum(m.aggregate[k])));
  const [metric, setMetric] = useState<string>("");

  const key = metric && numericKeys.includes(metric) ? metric : (numericKeys[0] ?? "abs_rel");

  const data = useMemo(() => {
    const rows = models
      .map((m) => ({ model: m.model, value: m.aggregate[key] }))
      .filter((r) => isNum(r.value))
      .sort((a, b) => (a.value as number) - (b.value as number));
    // horizontal layout renders the first category at the bottom; flip so best is on top
    return rows.reverse();
  }, [models, key]);

  const horizontal = data.length > 20;
  const truncated = data.length > 100;
  const shown = truncated ? data.slice(0, 100) : data;
  const chartHeight = horizontal ? Math.max(280, shown.length * 22 + 60) : 320;

  if (numericKeys.length === 0) return null;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-[12px] text-fg-muted">指标</span>
        <Select.Root value={key} onValueChange={setMetric}>
          <Select.Trigger className="w-44 font-mono">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {numericKeys.map((k) => (
              <Select.Item key={k} value={k}>
                {metricMeta(k).label}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
        <span className="num text-[12px] text-fg-muted">
          {models.length} 个模型{truncated ? `，仅显示前 100` : ""}
        </span>
      </div>
      <div className={horizontal ? "max-h-[480px] overflow-auto" : undefined}>
        <div style={{ height: chartHeight }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={shown}
              layout={horizontal ? "vertical" : "horizontal"}
              margin={{ top: 8, right: horizontal ? 56 : 8, bottom: 4, left: 8 }}
            >
              <CartesianGrid
                stroke="var(--border)"
                strokeDasharray="3 3"
                horizontal={!horizontal}
                vertical={horizontal}
              />
              {horizontal ? (
                <>
                  <XAxis
                    type="number"
                    tick={AXIS_TICK}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    tickFormatter={(v: number) => v.toPrecision(2)}
                  />
                  <YAxis
                    type="category"
                    dataKey="model"
                    width={132}
                    tick={{ ...AXIS_TICK, fontFamily: "var(--font-mono)" }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                  />
                </>
              ) : (
                <>
                  <XAxis
                    dataKey="model"
                    tick={{ ...AXIS_TICK, fontFamily: "var(--font-mono)" }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    angle={-30}
                    textAnchor="end"
                    height={70}
                  />
                  <YAxis
                    tick={AXIS_TICK}
                    tickLine={false}
                    axisLine={{ stroke: "var(--border)" }}
                    tickFormatter={(v: number) => v.toPrecision(2)}
                  />
                </>
              )}
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
                formatter={(v) => [fmt(v), metricMeta(key).label]}
              />
              <Bar dataKey="value" fill="var(--primary)" isAnimationActive={false}>
                <LabelList
                  dataKey="value"
                  position={horizontal ? "right" : "top"}
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
  );
}

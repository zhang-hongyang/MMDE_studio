import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Select } from "../../components/ui/select";
import type { ModelSplitMetrics, PerFramePoint } from "./api";
import { downsample, isNum, metricKeys, metricMeta } from "./format";
import { cn } from "../../lib/utils";

const AXIS_TICK = { fill: "var(--fg-muted)", fontSize: 11 };
const MAX_SERIES_POINTS = 1000;
const MAX_MODELS = 4;

/** distinct stroke (color + dash) per series so lines never differ by color alone */
const STROKES = [
  { stroke: "#3b82f6", dash: undefined },
  { stroke: "#22c55e", dash: "7 3" },
  { stroke: "#f59e0b", dash: "2 3" },
  { stroke: "#a78bfa", dash: "8 3 2 3" },
] as const;

interface Props {
  models: ModelSplitMetrics[];
}

interface Series {
  model: string;
  stroke: string;
  dash: string | undefined;
  data: { frame: number; value: number }[];
}

export function PerFrameChart({ models }: Props) {
  const keys = useMemo(() => metricKeys(models), [models]);
  const numericKeys = keys.filter((k) => models.some((m) => m.per_frame.some((p) => isNum(p[k]))));
  const [metric, setMetric] = useState<string>("");
  const key = metric && numericKeys.includes(metric) ? metric : (numericKeys[0] ?? "abs_rel");

  const [selected, setSelected] = useState<string[]>([]);
  const selectedSafe =
    selected.filter((m) => models.some((x) => x.model === m)).slice(0, MAX_MODELS);
  const effective =
    selectedSafe.length > 0
      ? selectedSafe
      : models
          .slice()
          .sort((a, b) => a.model.localeCompare(b.model))
          .slice(0, Math.min(2, MAX_MODELS))
          .map((m) => m.model);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const series = useMemo<Series[]>(() => {
    return effective
      .map((name, i) => {
        const entry = models.find((m) => m.model === name);
        if (!entry) return null;
        const pts = entry.per_frame
          .map((p: PerFramePoint, idx: number) => ({
            frame: typeof p.index === "number" ? p.index : idx,
            value: p[key],
          }))
          .filter((p) => isNum(p.value))
          .map((p) => ({ frame: p.frame, value: p.value as number }));
        const s = STROKES[i % STROKES.length];
        const series: Series = {
          model: name,
          stroke: s.stroke as string,
          dash: s.dash,
          data: downsample(pts, MAX_SERIES_POINTS),
        };
        return series;
      })
      .filter((s): s is Series => s !== null && s.data.length > 0);
  }, [effective, models, key]);

  if (numericKeys.length === 0) return null;

  const toggleSelect = (name: string) =>
    setSelected((prev) => {
      const cur = prev.filter((m) => models.some((x) => x.model === m));
      if (cur.includes(name)) return cur.filter((m) => m !== name);
      if (cur.length >= MAX_MODELS) return cur;
      return [...cur, name];
    });

  const toggleHidden = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const visible = series.filter((s) => !hidden.has(s.model));
  const frameMax = series.reduce((mx, s) => Math.max(mx, s.data[s.data.length - 1]?.frame ?? 0), 0);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
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
        <span className="text-[12px] text-fg-muted">模型（{effective.length}/{MAX_MODELS}）</span>
        {models.map((m) => {
          const on = effective.includes(m.model);
          return (
            <button
              key={m.model}
              onClick={() => toggleSelect(m.model)}
              className={cn(
                "max-w-44 truncate rounded-[6px] border px-2 py-0.5 font-mono text-[12px]",
                on
                  ? "border-primary/50 bg-primary/10 text-fg"
                  : "border-border text-fg-muted hover:bg-card-2",
              )}
              title={m.model}
            >
              {m.model}
            </button>
          );
        })}
      </div>
      <div style={{ height: 340 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="frame"
              domain={[0, Math.max(1, frameMax)]}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{ value: "帧号", position: "insideBottomRight", offset: -2, fill: "var(--fg-muted)", fontSize: 11 }}
            />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tickFormatter={(v: number) => v.toPrecision(2)}
            />
            <RTooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                fontSize: 12,
              }}
              labelStyle={{ color: "var(--fg-muted)" }}
              formatter={(v) => [isNum(v) ? v.toFixed(4) : "—", ""]}
            />
            {visible.map((s) => (
              <Line
                key={s.model}
                data={s.data}
                type="monotone"
                dataKey="value"
                name={s.model}
                stroke={s.stroke}
                strokeDasharray={s.dash}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {/* clickable legend: toggle series without removing the selection */}
      <div className="mt-1 flex flex-wrap items-center gap-3">
        {series.map((s) => {
          const off = hidden.has(s.model);
          return (
            <button
              key={s.model}
              onClick={() => toggleHidden(s.model)}
              className={cn(
                "inline-flex max-w-56 items-center gap-1.5 truncate text-[12px]",
                off ? "text-fg-muted/50 line-through" : "text-fg",
              )}
              title={`点击${off ? "显示" : "隐藏"} ${s.model}`}
            >
              <svg width="22" height="8" aria-hidden>
                <line
                  x1="0"
                  y1="4"
                  x2="22"
                  y2="4"
                  stroke={s.stroke}
                  strokeWidth="2"
                  strokeDasharray={s.dash}
                />
              </svg>
              <span className="truncate font-mono">{s.model}</span>
            </button>
          );
        })}
        {visible.length === 0 && (
          <span className="text-[12px] text-warning">所有序列已隐藏，点击图例恢复</span>
        )}
      </div>
    </div>
  );
}

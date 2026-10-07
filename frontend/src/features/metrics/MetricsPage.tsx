import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Film, Layers } from "lucide-react";
import { useRegistryStore } from "../../stores/registry";
import { Badge } from "../../components/ui/badge";
import { Select } from "../../components/ui/select";
import { Skeleton } from "../../components/ui/skeleton";
import { Tabs } from "../../components/ui/tabs";
import { fetchSplitMetrics, type ModelSplitMetrics } from "./api";
import { fmt } from "./format";
import { MetricsTable } from "./MetricsTable";
import { ModelCompareChart } from "./ModelCompareChart";
import { PerFrameChart } from "./PerFrameChart";
import { SummaryOverview } from "./SummaryOverview";
import { SplitSelectors } from "../../lib/SplitSelectors";

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: ModelSplitMetrics[] };

export function MetricsPage() {
  const registry = useRegistryStore((s) => s.registry);
  const [searchParams, setSearchParams] = useSearchParams();
  const [state, setState] = useState<State>({ kind: "loading" });

  const dsParam = searchParams.get("dataset");
  const splitParam = searchParams.get("split");
  const dataset =
    (dsParam && registry?.datasets.some((d) => d.name === dsParam) && dsParam) ||
    registry?.datasets[0]?.name ||
    "";
  const split =
    (splitParam &&
      registry?.datasets.find((d) => d.name === dataset)?.splits.some((s) => s.name === splitParam) &&
      splitParam) ||
    registry?.datasets.find((d) => d.name === dataset)?.splits[0]?.name ||
    "";
  const splitInfo = registry?.datasets
    .find((d) => d.name === dataset)?.splits.find((s) => s.name === split);

  const setDS = (name: string, sp?: string) => {
    const d = registry?.datasets.find((x) => x.name === name);
    const s = sp ?? d?.splits[0]?.name ?? "";
    setSearchParams({ dataset: name, split: s });
  };

  useEffect(() => {
    if (!dataset || !split) return;
    let alive = true;
    setState({ kind: "loading" });
    fetchSplitMetrics(dataset, split)
      .then((d) => alive && setState({ kind: "ready", data: d }))
      .catch((e) => alive && setState({ kind: "error", message: String(e) }));
    return () => {
      alive = false;
    };
  }, [dataset, split]);

  const meta = useMemo(() => {
    if (state.kind !== "ready") return null;
    const protocols = [...new Set(state.data.map((m) => m.protocol))].sort();
    const ranges = [...new Set(state.data.map((m) => (m.gt_range_m ?? []).join("–")))]
      .filter((s) => s)
      .sort();
    const frames = [...new Set(state.data.map((m) => m.num_frames))].sort((a, b) => a - b);
    return { protocols, ranges, frames };
  }, [state]);

  if (!registry) {
    return (
      <div className="min-h-0 flex-1 space-y-2 overflow-auto p-4">
        <Skeleton className="h-9 w-96" />
        <Skeleton className="h-72" />
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Select.Root value={dataset} onValueChange={(v) => setDS(v)}>
          <Select.Trigger className="w-44 font-mono" aria-label="数据集">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {registry.datasets.map((d) => (
              <Select.Item key={d.name} value={d.name}>
                {d.title || d.name}（{d.name}）
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
        <SplitSelectors
          dataset={registry.datasets.find((d) => d.name === dataset)}
          split={split}
          onSplitChange={(v) => setDS(dataset, v)}
        />
        {meta && (
          <div className="flex items-center gap-1.5 text-[12px] text-fg-muted">
            protocol
            {meta.protocols.length > 0 ? (
              meta.protocols.map((p) => (
                <Badge key={p} variant="outline" className="font-mono">
                  {p}
                </Badge>
              ))
            ) : (
              <span>—</span>
            )}
            <span className="mx-1">·</span>
            GT 范围
            <span className="num font-mono">{meta.ranges.join(" / ") || "—"}</span>
            <span className="mx-1">·</span>
            <Film className="h-3 w-3" />
            <span className="num">
              {meta.frames.map((f) => f.toLocaleString()).join(" / ")} 帧
            </span>
          </div>
        )}
      </div>

      {state.kind === "loading" ? (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-72" />
        </div>
      ) : state.kind === "error" ? (
        <div className="card max-w-lg text-[13px] text-danger">
          /api/metrics/{dataset}/{split} 加载失败:{state.message}
        </div>
      ) : state.data.length === 0 ? (
        <div className="card text-[13px] text-fg-muted">
          {splitInfo?.evaluation_available === false
            ? "该官方 test split 不公开 GT，仅提供全量推理与可视化，不生成本地指标。"
            : "该 split 暂无指标文件。可先在 Tasks 页创建 Evaluation 任务生成指标。"}
        </div>
      ) : (
        <>
          <div className="mb-1 flex items-center gap-1.5 text-[13px] font-medium text-fg-muted">
            <Layers className="h-3.5 w-3.5" />
            模型 × 指标矩阵
            <span className="text-fg-muted/70">
              （{state.data.length} 个模型，绿色为每列最佳值，点击表头排序）
            </span>
          </div>
          <div className="mb-3">
            <MetricsTable models={state.data} />
          </div>
          <Tabs.Root defaultValue="compare">
            <Tabs.List>
              <Tabs.Trigger value="compare">模型对比</Tabs.Trigger>
              <Tabs.Trigger value="perframe">每帧误差曲线</Tabs.Trigger>
              <Tabs.Trigger value="summary">跨 split 总览</Tabs.Trigger>
            </Tabs.List>
            <div className="card mt-2">
              <Tabs.Content value="compare">
                <ModelCompareChart models={state.data} />
              </Tabs.Content>
              <Tabs.Content value="perframe">
                <PerFrameChart models={state.data} />
              </Tabs.Content>
              <Tabs.Content value="summary">
                <SummaryOverview />
              </Tabs.Content>
            </div>
          </Tabs.Root>
          <p className="mt-2 text-[12px] text-fg-muted">
            数据协议 {meta?.protocols.join(", ") || "—"}；数值为 aggregate 均值，
            每帧曲线数据点 {fmt(state.data[0]?.per_frame.length ?? null, 0)} 帧/模型。
          </p>
        </>
      )}
    </div>
  );
}

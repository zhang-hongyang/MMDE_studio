import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Car, Plane, Satellite, Box, Database, Film, Cpu, Boxes, SlidersHorizontal, Layers } from "lucide-react";
import { useRegistryStore } from "../stores/registry";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Skeleton } from "../components/ui/skeleton";
import { Tooltip } from "../components/ui/tooltip";
import { Separator } from "../components/ui/separator";
import { splitLabel } from "../lib/SplitSelectors";

const PLATFORM_META: Record<string, { label: string; icon: React.ElementType }> = {
  vehicle: { label: "vehicle", icon: Car },
  uav: { label: "uav", icon: Plane },
  satellite: { label: "satellite", icon: Satellite },
};

// ---- /api/metrics/summary -------------------------------------------------

interface ModelMetrics {
  abs_rel?: number | null;
  abs_mean_m?: number | null;
  delta1?: number | null;
  [key: string]: number | null | undefined;
}

interface MetricsSummary {
  available?: boolean;
  summary?: unknown;
}

type MetricsState =
  | { kind: "loading" }
  | { kind: "hidden" }
  | { kind: "error"; message: string }
  | { kind: "ready"; rows: MetricRow[] };

interface MetricRow {
  key: string; // "dataset/split"
  model: string;
  absRel: number;
  d1: number | null;
  absMean: number | null;
}

function fmt(v: number | null | undefined, digits = 3): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

/** pick the model with the smallest non-null abs_rel for one dataset/split */
function bestModel(models: Record<string, ModelMetrics>): MetricRow | null {
  let best: { model: string; absRel: number } | null = null;
  for (const [model, m] of Object.entries(models)) {
    const v = m.abs_rel;
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (!best || v < best.absRel) best = { model, absRel: v };
  }
  if (!best) return null;
  const m = models[best.model];
  return {
    key: "",
    model: best.model,
    absRel: best.absRel,
    d1: typeof m.delta1 === "number" ? m.delta1 : null,
    absMean: typeof m.abs_mean_m === "number" ? m.abs_mean_m : null,
  };
}

function summaryModels(summary: unknown, dataset: string, split: string): Record<string, ModelMetrics> {
  if (!summary || typeof summary !== "object") return {};
  const root = summary as Record<string, unknown>;
  const datasets = root.datasets as Record<string, unknown> | undefined;
  const datasetDoc = datasets?.[dataset] as Record<string, unknown> | undefined;
  const splits = datasetDoc?.splits as Record<string, unknown> | undefined;
  const splitDoc = splits?.[split] as Record<string, unknown> | undefined;
  const models = splitDoc?.models as Record<string, unknown> | undefined;
  if (models) {
    return Object.fromEntries(
      Object.entries(models).map(([name, value]) => {
        const doc = value as Record<string, unknown>;
        return [name, (doc.raw_frame_mean ?? doc.aggregate ?? {}) as ModelMetrics];
      }),
    );
  }
  return (root[`${dataset}/${split}`] as Record<string, ModelMetrics> | undefined) ?? {};
}

function MetricsSection() {
  const registry = useRegistryStore((s) => s.registry);
  const [state, setState] = useState<MetricsState>({ kind: "loading" });

  useEffect(() => {
    let alive = true;
    fetch("/api/metrics/summary")
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json() as Promise<MetricsSummary>;
      })
      .then((data) => {
        if (!alive) return;
        if (data.available === false || !data.summary) {
          setState({ kind: "hidden" });
          return;
        }
        const rows: MetricRow[] = [];
        // registry order; skip splits missing from the summary
        for (const d of registry?.datasets ?? []) {
          for (const sp of d.splits) {
            const key = `${d.name}/${sp.name}`;
            const row = bestModel(summaryModels(data.summary, d.name, sp.name));
            if (row) rows.push({ ...row, key });
          }
        }
        setState({ kind: "ready", rows });
      })
      .catch((e) => {
        if (alive) setState({ kind: "error", message: String(e) });
      });
    return () => {
      alive = false;
    };
  }, [registry]);

  if (state.kind === "hidden") return null;

  return (
    <section className="mt-2">
      <h2 className="mb-2 flex items-center gap-1.5 text-[13px] font-medium text-fg-muted">
        <Layers className="h-3.5 w-3.5" />
        指标速览
        <span className="text-fg-muted/70">（各子集最佳模型的 abs_rel / AbsMean / δ1）</span>
      </h2>
      {state.kind === "loading" ? (
        <Skeleton className="h-36" />
      ) : state.kind === "error" ? (
        <div className="card text-[13px] text-fg-muted">
          指标不可用：/api/metrics/summary 加载失败（{state.message}）
        </div>
      ) : state.rows.length === 0 ? (
        <div className="card text-[13px] text-fg-muted">暂无指标数据。</div>
      ) : (
        <div className="card max-h-80 overflow-auto p-0">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 bg-card">
              <tr className="h-9 text-left text-fg-muted">
                <th className="px-3 font-medium">dataset / split</th>
                <th className="px-3 font-medium">最佳模型</th>
                <th className="px-3 text-right font-medium">abs_rel</th>
                <th className="px-3 text-right font-medium">AbsMean (m)</th>
                <th className="px-3 text-right font-medium">δ1</th>
              </tr>
            </thead>
            <tbody>
              {state.rows.map((r) => (
                <tr key={r.key} className="h-9 border-t border-border">
                  <td className="px-3 font-mono">{r.key}</td>
                  <td className="max-w-56 truncate px-3 font-mono">{r.model}</td>
                  <td className="num px-3 text-right">{fmt(r.absRel)}</td>
                  <td className="num px-3 text-right">{fmt(r.absMean)}</td>
                  <td className="num px-3 text-right">{fmt(r.d1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

// ---- page -----------------------------------------------------------------

export function OverviewPage() {
  const { registry, loading, error, load } = useRegistryStore();
  const navigate = useNavigate();

  if (!registry && !loading && !error) void load();

  if (error) {
    return (
      <div className="p-4">
        <div className="card max-w-lg text-[13px] text-danger">
          /api/registry 加载失败：{error}（请确认后端已启动于 :8010）
        </div>
      </div>
    );
  }
  if (!registry) {
    return (
      <div className="grid grid-cols-1 gap-2 p-4 md:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-40" />
        ))}
      </div>
    );
  }

  const groups = new Map<string, typeof registry.datasets>();
  for (const d of registry.datasets) {
    const g = groups.get(d.platform) ?? [];
    g.push(d);
    groups.set(d.platform, g);
  }

  return (
    <div className="min-h-0 overflow-auto p-4">
      <h1 className="mb-3 text-[18px] font-semibold">数据集总览</h1>
      {[...groups.entries()].map(([platform, datasets]) => {
        const pm = PLATFORM_META[platform] ?? { label: platform, icon: Box };
        const Icon = pm.icon;
        return (
          <section key={platform} className="mb-4">
            <h2 className="mb-2 flex items-center gap-1.5 text-[13px] font-medium text-fg-muted">
              <Icon className="h-3.5 w-3.5" />
              {pm.label}
              <span className="num">（{datasets.length}）</span>
            </h2>
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
              {datasets.map((d) => (
                <div key={d.name} className="card min-w-0">
                  <div className="mb-2 flex min-w-0 items-center gap-2">
                    <span className="truncate text-[14px] font-medium">{d.title || d.name}</span>
                    <Badge variant="outline" className="shrink-0 font-mono">
                      {pm.label}
                    </Badge>
                    <Badge variant="outline" className="ml-auto shrink-0 font-mono">
                      {d.name}
                    </Badge>
                  </div>
                  <table className="w-full text-[12px]">
                    <thead>
                      <tr className="h-9 text-left text-fg-muted">
                        <th className="font-medium">类型 / 子集</th>
                        <th className="font-medium">帧数</th>
                        <th className="font-medium">模型数</th>
                        <th className="font-medium">场景模型</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {d.splits.map((sp) => {
                        const sceneOK = sp.scene_models.length > 0;
                        const ctrlOK = sp.control_sources.length > 0;
                        const unavailable = Object.entries(sp.model_availability ?? {})
                          .filter(([, value]) => value.available === false);
                        const unavailableTitle = unavailable
                          .map(([name, value]) => `${name}: ${value.reason ?? "unavailable"}`)
                          .join("\n");
                        return (
                          <tr key={sp.name} className="h-9 border-t border-border">
                            <td className="max-w-32 truncate font-mono">
                              <span className="block text-[11px] text-fg-muted">{sp.test_type}</span>
                              {splitLabel(sp)}
                              {!sp.evaluation_available && (
                                <span className="ml-1 text-warning">仅推理</span>
                              )}
                            </td>
                            <td className="num">
                              <span className="inline-flex items-center gap-1">
                                <Film className="h-3 w-3 text-fg-muted" />
                                {sp.n_frames.toLocaleString()}
                              </span>
                            </td>
                            <td className="num">
                              <span className="inline-flex items-center gap-1">
                                <Cpu className="h-3 w-3 text-fg-muted" />
                                {sp.models.length}
                                {unavailable.length > 0 && (
                                  <span className="text-warning" title={unavailableTitle}>
                                    （{unavailable.length} 不可用）
                                  </span>
                                )}
                              </span>
                            </td>
                            <td className="num">
                              <span className="inline-flex items-center gap-1">
                                <Boxes className="h-3 w-3 text-fg-muted" />
                                {sp.scene_models.length || "—"}
                              </span>
                            </td>
                            <td>
                              <div className="flex flex-wrap items-center justify-end gap-1">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="px-1.5"
                                  aria-label={`在 Explorer 中打开 ${d.name}/${sp.name}`}
                                  onClick={() => navigate(`/explore/${d.name}/${sp.name}`)}
                                >
                                  <Database className="h-3 w-3" />
                                  Explore
                                </Button>
                                {sceneOK ? (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="px-1.5"
                                    aria-label={`在 Scene 中打开 ${d.name}/${sp.name}`}
                                    onClick={() => navigate(`/scene/${d.name}/${sp.name}`)}
                                  >
                                    <Boxes className="h-3 w-3" />
                                    Scene
                                  </Button>
                                ) : (
                                  <Tooltip.Root>
                                    <Tooltip.Trigger asChild>
                                      <span tabIndex={0} aria-label={`Scene 不可用：${d.name}/${sp.name} 无场景模型`}>
                                        <Button variant="ghost" size="sm" className="px-1.5" disabled>
                                          <Boxes className="h-3 w-3" />
                                          Scene
                                        </Button>
                                      </span>
                                    </Tooltip.Trigger>
                                    <Tooltip.Content>该 split 无场景模型</Tooltip.Content>
                                  </Tooltip.Root>
                                )}
                                {ctrlOK ? (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="px-1.5"
                                    aria-label={`在 Controls 中打开 ${d.name}/${sp.name}`}
                                    onClick={() => navigate(`/controls/${d.name}/${sp.name}`)}
                                  >
                                    <SlidersHorizontal className="h-3 w-3" />
                                    Controls
                                  </Button>
                                ) : (
                                  <Tooltip.Root>
                                    <Tooltip.Trigger asChild>
                                      <span tabIndex={0} aria-label={`Controls 不可用：${d.name}/${sp.name} 无控制点源`}>
                                        <Button variant="ghost" size="sm" className="px-1.5" disabled>
                                          <SlidersHorizontal className="h-3 w-3" />
                                          Controls
                                        </Button>
                                      </span>
                                    </Tooltip.Trigger>
                                    <Tooltip.Content>该 split 无控制点源</Tooltip.Content>
                                  </Tooltip.Root>
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </section>
        );
      })}

      <Separator className="my-4" />
      <MetricsSection />
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import { Inbox, PlusCircle, RefreshCw } from "lucide-react";
import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import { Skeleton } from "../../components/ui/skeleton";
import {
  fetchTaskCatalog,
  fetchTasks,
  type Task,
  type TaskSpec,
  type TaskStatus,
} from "./api";
import { LogView } from "./LogView";
import { TaskWizard } from "./TaskWizard";
import { cn } from "../../lib/utils";

const STATUS_META: Record<
  TaskStatus,
  { variant: "default" | "success" | "danger" | "warning"; label: string; pulse?: boolean }
> = {
  queued: { variant: "default", label: "queued" },
  running: { variant: "default", label: "running", pulse: true },
  succeeded: { variant: "success", label: "succeeded" },
  failed: { variant: "danger", label: "failed" },
  cancelled: { variant: "warning", label: "cancelled" },
};

function fmtTime(ts: number | null | undefined): string {
  return typeof ts === "number"
    ? new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false })
    : "—";
}

/** compact one-line summary of non-null params */
function paramsSummary(params: Record<string, unknown>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== null && v !== undefined && v !== false && v !== "")
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : String(v)}`);
  return parts.join(" ");
}

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; tasks: Task[] };

export function TasksPage() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [catalog, setCatalog] = useState<TaskSpec[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback((silent = false) => {
    if (!silent) setRefreshing(true);
    fetchTasks(undefined, 200)
      .then((tasks) => setState({ kind: "ready", tasks }))
      .catch((e) => {
        if (!silent) setState((s) => (s.kind === "loading" ? { kind: "error", message: String(e) } : s));
      })
      .finally(() => setRefreshing(false));
  }, []);

  useEffect(() => {
    refresh();
    fetchTaskCatalog()
      .then(setCatalog)
      .catch((e) => setCatalogError(String(e)));
  }, [refresh]);

  // poll every 3s while the page is visible
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refresh(true);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const tasks = state.kind === "ready" ? state.tasks : [];
  const selected = tasks.find((t) => t.id === selectedId) ?? null;

  const onCreated = (task: Task) => {
    refresh(true);
    setSelectedId(task.id);
  };

  return (
    <div className="flex h-full min-h-0 flex-col p-4">
      <div className="mb-2 flex shrink-0 items-center gap-2">
        <Button onClick={() => setWizardOpen(true)}>
          <PlusCircle className="h-3.5 w-3.5" />
          新建任务
        </Button>
        <Button variant="outline" onClick={() => refresh()} disabled={refreshing}>
          <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
          刷新
        </Button>
        {state.kind === "ready" && (
          <span className="num text-[12px] text-fg-muted">{tasks.length} 个任务（3s 自动轮询）</span>
        )}
        {catalogError && (
          <span className="text-[12px] text-warning">catalog 加载失败:{catalogError}</span>
        )}
      </div>

      {/* task list */}
      <div className="card min-h-0 flex-1 overflow-auto p-0">
        {state.kind === "loading" ? (
          <div className="space-y-2 p-3">
            <Skeleton className="h-8" />
            <Skeleton className="h-8" />
            <Skeleton className="h-8" />
          </div>
        ) : state.kind === "error" ? (
          <div className="p-3 text-[13px] text-danger">
            /api/tasks 加载失败:{state.message}（请确认后端支持任务 API）
          </div>
        ) : tasks.length === 0 ? (
          <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-fg-muted">
            <Inbox className="h-8 w-8" />
            <span className="text-[13px]">暂无任务</span>
            <Button variant="outline" size="sm" onClick={() => setWizardOpen(true)}>
              <PlusCircle className="h-3.5 w-3.5" />
              创建第一个任务
            </Button>
          </div>
        ) : (
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 z-10 bg-card">
              <tr className="h-9 text-left text-fg-muted">
                <th className="px-3 font-medium">ID</th>
                <th className="px-3 font-medium">类型</th>
                <th className="px-3 font-medium">参数</th>
                <th className="px-3 font-medium">状态</th>
                <th className="px-3 font-medium">创建</th>
                <th className="px-3 font-medium">开始</th>
                <th className="px-3 font-medium">结束</th>
                <th className="px-3 text-right font-medium">exit</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => {
                const meta = STATUS_META[t.status] ?? STATUS_META.queued;
                return (
                  <tr
                    key={t.id}
                    onClick={() => setSelectedId(t.id === selectedId ? null : t.id)}
                    className={cn(
                      "h-9 cursor-pointer border-t border-border hover:bg-card-2",
                      t.id === selectedId && "bg-primary/10",
                    )}
                  >
                    <td className="px-3 font-mono">{t.id.slice(0, 8)}</td>
                    <td className="px-3">
                      <Badge variant="outline" className="font-mono">
                        {t.type}
                      </Badge>
                    </td>
                    <td className="max-w-80 truncate px-3 font-mono text-fg-muted" title={paramsSummary(t.params)}>
                      {paramsSummary(t.params) || "—"}
                    </td>
                    <td className="px-3">
                      <Badge variant={meta.variant} className="font-mono">
                        {meta.pulse && (
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
                        )}
                        {meta.label}
                      </Badge>
                    </td>
                    <td className="num whitespace-nowrap px-3 text-fg-muted">{fmtTime(t.created_at)}</td>
                    <td className="num whitespace-nowrap px-3 text-fg-muted">{fmtTime(t.started_at)}</td>
                    <td className="num whitespace-nowrap px-3 text-fg-muted">{fmtTime(t.finished_at)}</td>
                    <td className="num px-3 text-right">{t.exit_code ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* log view */}
      <div className="mt-2 h-[38%] shrink-0 overflow-hidden rounded-lg border border-border">
        <LogView task={selected} onTaskChanged={() => refresh(true)} />
      </div>

      <TaskWizard
        open={wizardOpen}
        onOpenChange={setWizardOpen}
        catalog={catalog}
        onCreated={onCreated}
      />
    </div>
  );
}

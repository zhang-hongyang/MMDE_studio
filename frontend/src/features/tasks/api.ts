// Tasks API client. Mirrors the fetch wrapper pattern in features/scene/api.ts.

export type TaskStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export const TERMINAL_STATUSES: TaskStatus[] = ["succeeded", "failed", "cancelled"];

export interface TaskParamSpec {
  name: string;
  label: string;
  type: "str" | "int" | "float" | "bool" | "list";
  required: boolean;
  default: unknown;
  /** static choices. Key ABSENT = free text; explicit null = dynamic choices (model selector). */
  choices?: string[] | null;
  help?: string;
}

export interface TaskSpec {
  type: string;
  title: string;
  description: string;
  params: TaskParamSpec[];
}

export interface Task {
  id: string;
  type: string;
  params: Record<string, unknown>;
  status: TaskStatus;
  created_at: number | null;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  pid?: number | null;
  /** present only on GET /api/tasks/{id} (recent 50 lines) */
  logs?: LogLine[];
}

export interface LogLine {
  seq: number;
  line: string;
  ts?: number | null;
}

export interface SubmitBody {
  type: string;
  params: Record<string, unknown>;
}

async function reqJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const body = (await r.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export function fetchTaskCatalog(): Promise<TaskSpec[]> {
  return reqJSON<TaskSpec[]>("/api/tasks/catalog");
}

export function fetchTasks(status?: TaskStatus, limit = 200): Promise<Task[]> {
  const qs = new URLSearchParams();
  if (status) qs.set("status", status);
  qs.set("limit", String(limit));
  return reqJSON<Task[]>(`/api/tasks?${qs.toString()}`);
}

export function submitTask(body: SubmitBody): Promise<Task> {
  return reqJSON<Task>("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchTask(id: string): Promise<Task> {
  return reqJSON<Task>(`/api/tasks/${encodeURIComponent(id)}`);
}

export function fetchTaskLog(id: string, after = 0): Promise<{ lines: LogLine[] }> {
  return reqJSON<{ lines: LogLine[] }>(
    `/api/tasks/${encodeURIComponent(id)}/log?after=${after}`,
  );
}

export function cancelTask(id: string): Promise<Task> {
  return reqJSON<Task>(`/api/tasks/${encodeURIComponent(id)}/cancel`, { method: "POST" });
}

export function taskWSURL(id: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/tasks/${encodeURIComponent(id)}`;
}

/** websocket message shapes from /ws/tasks/{id} */
export type WSMessage =
  | { type: "status"; task_id?: string; status?: TaskStatus }
  | { type: "log"; seq: number; line: string; ts?: number | null }
  | { type: "final"; status?: TaskStatus; exit_code?: number | null }
  | { type: "error"; detail?: string };

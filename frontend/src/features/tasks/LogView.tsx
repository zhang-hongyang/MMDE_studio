import { useCallback, useEffect, useRef, useState } from "react";
import { Ban, TerminalSquare } from "lucide-react";
import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import {
  cancelTask,
  fetchTaskLog,
  taskWSURL,
  TERMINAL_STATUSES,
  type LogLine,
  type Task,
  type TaskStatus,
  type WSMessage,
} from "./api";
import { cn } from "../../lib/utils";

function fmtTime(ts: number | null | undefined): string {
  return typeof ts === "number" ? new Date(ts * 1000).toLocaleTimeString("zh-CN") : "—";
}

const STATUS_META: Record<TaskStatus, { variant: "default" | "outline" | "success" | "danger" | "warning"; pulse?: boolean }> = {
  queued: { variant: "outline" },
  running: { variant: "default", pulse: true },
  succeeded: { variant: "success" },
  failed: { variant: "danger" },
  cancelled: { variant: "warning" },
};

interface Props {
  task: Task | null;
  onTaskChanged: () => void;
}

/** live log terminal for one task: WS first, 1s polling fallback. */
export function LogView({ task, onTaskChanged }: Props) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [liveStatus, setLiveStatus] = useState<TaskStatus | null>(null);
  const [transport, setTransport] = useState<"ws" | "poll" | null>(null);
  const [done, setDone] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  const boxRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const linesRef = useRef<LogLine[]>([]);
  const lastSeqRef = useRef(0);
  const statusRef = useRef<TaskStatus | null>(null);

  const status: TaskStatus = liveStatus ?? task?.status ?? "queued";

  const appendLines = useCallback((incoming: LogLine[]) => {
    const fresh = incoming.filter((l) => l.seq > lastSeqRef.current);
    if (fresh.length === 0) return;
    lastSeqRef.current = fresh[fresh.length - 1].seq;
    linesRef.current = [...linesRef.current, ...fresh];
    setLines(linesRef.current);
  }, []);

  // reset whenever the selected task changes
  useEffect(() => {
    linesRef.current = [];
    lastSeqRef.current = 0;
    stickRef.current = true;
    setLines([]);
    setLiveStatus(null);
    setTransport(null);
    setDone(false);
  }, [task?.id]);

  // track latest known status for the poll loop
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // WS connection (with polling fallback)
  useEffect(() => {
    if (!task) return;
    const id = task.id;
    let alive = true;
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let gotFinal = false;
    let emptyTicks = 0;

    const poll = () => {
      setTransport("poll");
      const tick = () => {
        fetchTaskLog(id, lastSeqRef.current)
          .then(({ lines: ls }) => {
            if (!alive) return;
            if (ls.length > 0) {
              emptyTicks = 0;
              appendLines(ls);
            } else {
              emptyTicks += 1;
            }
            if (
              statusRef.current &&
              TERMINAL_STATUSES.includes(statusRef.current) &&
              emptyTicks >= 2
            ) {
              window.clearInterval(timer);
              setDone(true);
            }
          })
          .catch(() => {
            /* backend may be restarting; keep polling */
          });
      };
      tick();
      timer = window.setInterval(tick, 1000);
    };

    try {
      ws = new WebSocket(taskWSURL(id));
    } catch {
      poll();
      return () => {
        alive = false;
        window.clearInterval(timer);
      };
    }

    ws.onopen = () => {
      if (alive) setTransport("ws");
    };
    ws.onmessage = (ev: MessageEvent<string>) => {
      let msg: WSMessage;
      try {
        msg = JSON.parse(ev.data) as WSMessage;
      } catch {
        return;
      }
      if (msg.type === "status" && msg.status) {
        if (alive) setLiveStatus(msg.status);
      } else if (msg.type === "log") {
        appendLines([{ seq: msg.seq, line: msg.line, ts: msg.ts }]);
      } else if (msg.type === "final") {
        gotFinal = true;
        if (alive) {
          if (msg.status) setLiveStatus(msg.status);
          setDone(true);
        }
        onTaskChanged();
      } else if (msg.type === "error") {
        ws?.close();
      }
    };
    ws.onerror = () => ws?.close();
    ws.onclose = () => {
      if (!alive) return;
      if (!gotFinal) poll(); // unexpected close / late joiner race -> fallback
    };

    return () => {
      alive = false;
      window.clearInterval(timer);
      ws?.close();
    };
    // appendLines is stable; onTaskChanged identity changes with every parent
    // render, so it is intentionally excluded.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.id, appendLines]);

  // auto-scroll to bottom unless the user scrolled up
  useEffect(() => {
    const el = boxRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  if (!task) {
    return (
      <div className="card flex h-full items-center justify-center text-[13px] text-fg-muted">
        在上方列表中选择一个任务以查看日志。
      </div>
    );
  }

  const meta = STATUS_META[status] ?? STATUS_META.queued;
  const canCancel = status === "running" || status === "queued";

  const onCancel = () => {
    setCancelling(true);
    cancelTask(task.id)
      .catch(() => undefined)
      .finally(() => {
        setCancelling(false);
        onTaskChanged();
      });
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-border bg-card px-3">
        <TerminalSquare className="h-4 w-4 text-fg-muted" />
        <span className="font-mono text-[12px]">{task.id}</span>
        <Badge variant={meta.variant} className="font-mono">
          {meta.pulse && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />}
          {status}
        </Badge>
        <span className="num text-[12px] text-fg-muted">{lines.length} 行</span>
        <span className="text-[12px] text-fg-muted">
          {transport === "ws" ? "WS 实时" : transport === "poll" ? "轮询(1s)" : "连接中…"}
        </span>
        {done && task.exit_code !== null && (
          <span className="num text-[12px] text-fg-muted">exit {task.exit_code}</span>
        )}
        <div className="ml-auto">
          {canCancel && (
            <Button variant="danger" size="sm" onClick={onCancel} disabled={cancelling}>
              <Ban className="h-3 w-3" />
              取消任务
            </Button>
          )}
        </div>
      </div>
      <div
        ref={boxRef}
        onScroll={() => {
          const el = boxRef.current;
          if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
        className={cn(
          "min-h-0 flex-1 overflow-auto bg-card-2 px-3 py-2 font-mono text-[12px] leading-5",
          "whitespace-pre-wrap break-all text-fg",
        )}
      >
        {lines.length === 0 ? (
          <span className="text-fg-muted">
            {status === "queued" ? "排队中，等待执行…" : "等待日志输出…"}
          </span>
        ) : (
          lines.map((l) => (
            <div key={l.seq} className="flex gap-2">
              <span className="num shrink-0 text-fg-muted/60">{fmtTime(l.ts)}</span>
              <span>{l.line}</span>
            </div>
          ))
        )}
        {done && status !== "running" && lines.length > 0 && (
          <div className="mt-1 text-fg-muted">
            — 任务结束（{status}
            {task.exit_code !== null ? `，exit ${task.exit_code}` : ""}）—
          </div>
        )}
      </div>
    </div>
  );
}

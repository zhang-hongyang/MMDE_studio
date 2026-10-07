import { useEffect, useRef } from "react";
import { api } from "../../lib/api";
import { useExploreStore } from "../../stores/explore";
import { ScrollArea } from "../../components/ui/scroll-area";
import { Button } from "../../components/ui/button";
import { Minimap } from "./Minimap";
import { cn } from "../../lib/utils";

const WINDOW_BEFORE = 30;
const WINDOW_AFTER = 170;

export function FramePanel() {
  const s = useExploreStore();
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [s.idx]);

  const start = Math.max(0, s.idx - WINDOW_BEFORE);
  const end = Math.min(s.frames.length, s.idx + WINDOW_AFTER);
  const rows = s.frames.slice(start, end);

  const rectText = s.rect
    ? `[${s.rect.u0 | 0},${s.rect.v0 | 0}]-[${s.rect.u1 | 0},${s.rect.v1 | 0}]`
    : "全帧";

  const frameB = s.mode === "frame" && s.idxB !== null ? s.frames[s.idxB] : null;

  return (
    <aside className="flex w-[280px] shrink-0 flex-col gap-2 overflow-hidden border-r border-border bg-card p-2">
      <Minimap />
      <div className="flex items-center justify-between text-[12px] text-fg-muted">
        <span>
          当前区域：<span className="num text-fg">{rectText}</span>
        </span>
        {s.rect && (
          <Button variant="ghost" size="sm" onClick={() => s.setRect(null)}>
            重置
          </Button>
        )}
      </div>
      <p className="text-[12px] leading-4 text-fg-muted">
        在上方帧A图上拖框选局部区域，双击恢复全帧。
      </p>

      {frameB && (
        <figure className="m-0">
          <img
            src={api.rgbURL(s.dataset, s.split, frameB.idx ?? s.idxB, 320)}
            alt={`帧B #${s.idxB} 缩略图`}
            className="w-full rounded-[6px] border border-border"
          />
          <figcaption className="num mt-0.5 text-[12px] text-fg-muted">
            帧B #{s.idxB}（{frameB.seq_id}）
          </figcaption>
        </figure>
      )}

      <Separator />
      <ScrollArea.Root className="min-h-0 flex-1">
        <div className="flex flex-col gap-px pr-1">
          {rows.map((f, i) => {
            const pos = start + i;
            const active = pos === s.idx;
            return (
              <button
                key={f.idx ?? pos}
                ref={active ? activeRef : undefined}
                onClick={() => s.setIdx(pos)}
                className={cn(
                  "flex h-9 shrink-0 items-center gap-2 rounded-[6px] border-l-2 px-2 text-left text-[12px]",
                  "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  active
                    ? "border-l-primary bg-primary/15 text-fg"
                    : "border-l-transparent text-fg-muted hover:bg-card-2 hover:text-fg",
                )}
              >
                <span className="num w-10 shrink-0 font-mono">#{pos}</span>
                <span className="truncate font-mono">{f.seq_id}</span>
                <span className="num ml-auto shrink-0 font-mono">{f.frame_id}</span>
              </button>
            );
          })}
        </div>
      </ScrollArea.Root>
    </aside>
  );
}

function Separator() {
  return <div className="h-px w-full bg-border" />;
}

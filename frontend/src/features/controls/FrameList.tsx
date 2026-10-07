import { useEffect, useRef } from "react";
import { useControlsStore } from "../../stores/controls";
import { ScrollArea } from "../../components/ui/scroll-area";
import { cn } from "../../lib/utils";

const WINDOW_BEFORE = 40;
const WINDOW_AFTER = 160;

/** Simplified frame list for the Controls page (windowed around current). */
export function FrameList() {
  const s = useControlsStore();
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [s.idx]);

  const start = Math.max(0, s.idx - WINDOW_BEFORE);
  const end = Math.min(s.frames.length, s.idx + WINDOW_AFTER);
  const rows = s.frames.slice(start, end);

  return (
    <aside className="flex w-[240px] shrink-0 flex-col gap-2 overflow-hidden border-l border-border bg-card p-2">
      <div className="flex items-center justify-between text-[12px] text-fg-muted">
        <span>
          帧列表 <span className="num text-fg">{s.frames.length}</span> 帧
        </span>
      </div>
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

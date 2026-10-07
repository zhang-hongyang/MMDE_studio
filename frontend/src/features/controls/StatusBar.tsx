import { errPredValues, median, useControlsStore } from "../../stores/controls";

/** Bottom status bar: current frame, per-source point counts, median model err. */
export function StatusBar() {
  const frames = useControlsStore((s) => s.frames);
  const idx = useControlsStore((s) => s.idx);
  const controls = useControlsStore((s) => s.controls);
  const enabled = useControlsStore((s) => s.enabled);
  const model = useControlsStore((s) => s.model);
  const loading = useControlsStore((s) => s.loading);

  const frame = frames[idx];

  const medErr = (() => {
    if (!model) return null;
    const all: number[] = [];
    for (const src of enabled) {
      const c = controls[src];
      if (c) all.push(...errPredValues(c));
    }
    return median(all);
  })();

  return (
    <footer className="flex h-9 shrink-0 items-center gap-4 overflow-x-auto border-t border-border bg-card px-3 text-[12px] text-fg-muted whitespace-nowrap">
      <span>
        帧 <span className="num text-fg">#{idx}</span>
        {frame && (
          <span className="num">
            {" "}
            （{frame.seq_id} / {frame.frame_id}）
          </span>
        )}
      </span>
      {enabled.length === 0 && <span>未选择控制源</span>}
      {enabled.map((src) => {
        const c = controls[src];
        return (
          <span key={src} className="inline-flex items-center gap-1">
            <span className="max-w-40 truncate font-mono">{src}</span>
            <span className="num text-fg">{c ? c.n : "—"}</span> 点
          </span>
        );
      })}
      <span className="ml-auto" />
      {model && (
        <span>
          中位 err<span className="text-fg-muted/70">(vs {model})</span>：
          <span className="num text-fg">{medErr !== null ? medErr.toFixed(3) : "—"}</span>
        </span>
      )}
      {loading && <span>加载中…</span>}
    </footer>
  );
}

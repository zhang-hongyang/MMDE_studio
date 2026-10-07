import { useExploreStore } from "../../stores/explore";

export function StatusBar() {
  const meta = useExploreStore((s) => s.meta);
  const idx = useExploreStore((s) => s.idx);
  const idxB = useExploreStore((s) => s.idxB);
  const mode = useExploreStore((s) => s.mode);
  const delta = useExploreStore((s) => s.delta);
  const rect = useExploreStore((s) => s.rect);
  const ctrl = useExploreStore((s) => s.ctrl);
  const showCtrl = useExploreStore((s) => s.showCtrl);
  const stats = useExploreStore((s) => s.stats);
  const loading = useExploreStore((s) => s.loading);
  const error = useExploreStore((s) => s.error);

  const parts: string[] = [];
  if (meta) {
    parts.push(`帧 ${idx}（${meta.seq_id} #${meta.frame_id}）`);
    parts.push(`A ${(stats?.nA ?? 0).toLocaleString()} 点 / B ${(stats?.nB ?? 0).toLocaleString()} 点`);
    if (mode === "frame") {
      if (stats?.frameStats) {
        const st = stats.frameStats;
        parts.push(
          `重叠带内 ${st.band}/${st.sampled}（${((100 * st.band) / st.sampled).toFixed(0)}%）` +
            ` · med|Δd| ${st.med.toFixed(3)} m · P95 ${st.p95.toFixed(2)} m`,
        );
      } else if (idxB === null) {
        parts.push(idx + delta >= 0 ? "帧B超出当前序列或无位姿" : "帧B不可用");
      }
    }
    if (showCtrl && ctrl) {
      parts.push(`控制 ${ctrl.n.toLocaleString()} 点`);
    }
    parts.push(`区域：${rect ? `[${rect.u0 | 0},${rect.v0 | 0}]-[${rect.u1 | 0},${rect.v1 | 0}]` : "全帧"}`);
  } else if (error) {
    parts.push(`加载失败：${error}`);
  } else {
    parts.push("加载中…");
  }
  if (loading && meta) parts.push("加载中…");

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 overflow-hidden border-t border-border bg-card px-3 text-[12px] text-fg-muted whitespace-nowrap">
      <span className="truncate">{parts.join("　")}</span>
    </footer>
  );
}

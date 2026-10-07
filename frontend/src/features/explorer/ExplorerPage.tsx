import { useEffect, useRef } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useExploreStore, type CompareMode } from "../../stores/explore";
import { ExplorerToolbar } from "./Toolbar";
import { FramePanel } from "./FramePanel";
import { StatusBar } from "./StatusBar";
import { Viewport, type ViewportHandle } from "./Viewport";
import { ColorLegend } from "./ColorLegend";
import { useExplorerGeometry, useOverlayColors } from "./useExplorerGeometry";
import { Skeleton } from "../../components/ui/skeleton";

export function ExplorerPage() {
  const { dataset = "", split = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const init = useExploreStore((s) => s.init);

  // initialize store from route + search params (once per dataset/split)
  useEffect(() => {
    if (!dataset || !split) return;
    void init(dataset, split, {
      idx: parseInt(searchParams.get("idx") ?? "", 10) || undefined,
      camera: searchParams.get("cam") ?? undefined,
      modelA: searchParams.get("modelA") ?? undefined,
      modelB: searchParams.get("modelB") ?? undefined,
      mode: (searchParams.get("mode") as CompareMode) ?? undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, split]);

  const idx = useExploreStore((s) => s.idx);
  const mode = useExploreStore((s) => s.mode);
  const modelA = useExploreStore((s) => s.modelA);
  const modelB = useExploreStore((s) => s.modelB);
  const camera = useExploreStore((s) => s.camera);
  const meta = useExploreStore((s) => s.meta);
  const idxB = useExploreStore((s) => s.idxB);
  const psize = useExploreStore((s) => s.psize);
  const brightness = useExploreStore((s) => s.brightness);
  const loading = useExploreStore((s) => s.loading);

  // write state back to URL (shareable / restorable)
  useEffect(() => {
    const p = new URLSearchParams();
    if (idx > 0) p.set("idx", String(idx));
    if (camera) p.set("cam", camera);
    if (modelA) p.set("modelA", modelA);
    if (modelB && mode !== "frame") p.set("modelB", modelB);
    if (mode !== "model") p.set("mode", mode);
    setSearchParams(p, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, mode, modelA, modelB, camera, dataset, split]);

  const colors = useOverlayColors();
  const geo = useExplorerGeometry(colors);
  const viewportARef = useRef<ViewportHandle>(null);
  const viewportBRef = useRef<ViewportHandle>(null);

  const overlays = (built: typeof geo.cloudA) => {
    const list: { built: NonNullable<typeof geo.cloudA>; size: number }[] = [];
    if (built) list.push({ built, size: psize });
    if (geo.gt) list.push({ built: geo.gt, size: psize * 1.3 });
    if (geo.ctrl) list.push({ built: geo.ctrl, size: psize * 2.2 });
    return list;
  };

  const labelA = mode === "frame" ? `帧 #${idx}` : `A · ${modelA}`;
  const labelB =
    mode === "frame"
      ? idxB !== null
        ? `帧 #${idxB}（已配准到A）`
        : "帧B 不可用"
      : mode === "overlay"
        ? `B · ${modelB}`
        : `B · ${modelB}`;

  const showSkeleton = !meta;
  const error = useExploreStore((s) => s.error);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ExplorerToolbar />
      <div className="flex min-h-0 flex-1">
        <FramePanel />
        <main className="relative flex min-w-0 flex-1 flex-col">
          {showSkeleton ? (
            error ? (
              <div className="flex flex-1 items-center justify-center p-4">
                <div className="card max-w-lg text-[13px] text-danger">
                  帧数据加载失败：{error}（请确认后端已启动于 :8000）
                </div>
              </div>
            ) : (
              <div className="flex flex-1 flex-col gap-2 p-2">
                <Skeleton className="flex-1" />
              </div>
            )
          ) : mode === "overlay" ? (
            <div className="relative flex min-h-0 flex-1 flex-col">
              <Viewport
                ref={viewportARef}
                viewportId="A"
                label={`${labelA} vs ${labelB}`}
                clouds={overlays(geo.cloudA).concat(
                  geo.cloudB ? [{ built: geo.cloudB, size: psize }] : [],
                )}
                brightness={brightness}
                className="flex-1"
              />
              <ColorLegend className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-[6px] border border-border bg-card/80 px-2 py-1" />
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex min-h-0 flex-1">
                <Viewport
                  ref={viewportARef}
                  viewportId="A"
                  label={labelA}
                  clouds={overlays(geo.cloudA)}
                  brightness={brightness}
                  className="flex-1"
                />
                <div className="w-px shrink-0 bg-border" />
                <Viewport
                  ref={viewportBRef}
                  viewportId="B"
                  label={labelB}
                  clouds={overlays(geo.cloudB)}
                  brightness={brightness}
                  className="flex-1"
                />
              </div>
              <ColorLegend className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-[6px] border border-border bg-card/80 px-2 py-1" />
            </div>
          )}
          {loading && !showSkeleton && (
            <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center">
              <span className="mt-2 rounded-[6px] border border-border bg-card/85 px-2 py-0.5 text-[12px] text-fg-muted">
                加载中…
              </span>
            </div>
          )}
        </main>
      </div>
      <StatusBar />
    </div>
  );
}

import { useEffect } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useControlsStore, type CtrlColorMode, type CtrlColormap } from "../../stores/controls";
import { ControlsToolbar } from "./Toolbar";
import { ImageView } from "./ImageView";
import { FrameList } from "./FrameList";
import { StatusBar } from "./StatusBar";
import { ColorLegend } from "./components/ColorLegend";
import { Skeleton } from "../../components/ui/skeleton";

export function ControlsPage() {
  const { dataset = "", split = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const init = useControlsStore((s) => s.init);

  // initialize store from route + search params (once per dataset/split)
  useEffect(() => {
    if (!dataset || !split) return;
    void init(dataset, split, {
      idx: parseInt(searchParams.get("idx") ?? "", 10) || undefined,
      enabled: searchParams.get("src")?.split(",").filter(Boolean),
      model: searchParams.get("model") ?? undefined,
      colorMode: (searchParams.get("color") as CtrlColorMode) ?? undefined,
      colormap: (searchParams.get("cmap") as CtrlColormap) ?? undefined,
      psize: parseInt(searchParams.get("ps") ?? "", 10) || undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, split]);

  const idx = useControlsStore((s) => s.idx);
  const enabled = useControlsStore((s) => s.enabled);
  const colorMode = useControlsStore((s) => s.colorMode);
  const colormap = useControlsStore((s) => s.colormap);
  const model = useControlsStore((s) => s.model);
  const psize = useControlsStore((s) => s.psize);
  const frames = useControlsStore((s) => s.frames);
  const playing = useControlsStore((s) => s.playing);
  const playInterval = useControlsStore((s) => s.playInterval);
  const error = useControlsStore((s) => s.error);
  const loading = useControlsStore((s) => s.loading);

  // write state back to URL (shareable / restorable)
  useEffect(() => {
    const p = new URLSearchParams();
    if (idx > 0) p.set("idx", String(idx));
    if (enabled.length > 0) p.set("src", enabled.join(","));
    if (colorMode !== "depth") p.set("color", colorMode);
    if (colormap !== "turbo") p.set("cmap", colormap);
    if (model) p.set("model", model);
    if (psize !== 3) p.set("ps", String(psize));
    setSearchParams(p, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, enabled, colorMode, colormap, model, psize, dataset, split]);

  // playback: advance frame every interval, stop at the last frame
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const s = useControlsStore.getState();
      if (s.idx >= s.frames.length - 1) s.set({ playing: false });
      else s.setIdx(s.idx + 1);
    }, playInterval);
    return () => window.clearInterval(id);
  }, [playing, playInterval]);

  const showSkeleton = frames.length === 0 && !error;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ControlsToolbar />
      {error ? (
        <div className="flex flex-1 items-center justify-center p-4">
          <div className="card max-w-lg text-[13px] text-danger">
            数据加载失败：{error}（请确认后端已启动于 :8010）
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <main className="relative flex min-w-0 flex-1 flex-col">
            {showSkeleton ? (
              <div className="flex flex-1 flex-col gap-2 p-2">
                <Skeleton className="flex-1" />
              </div>
            ) : (
              <>
                <ImageView />
                <ColorLegend className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-[6px] border border-border bg-card/85 px-2 py-1" />
                {loading && (
                  <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center">
                    <span className="mt-2 rounded-[6px] border border-border bg-card/85 px-2 py-0.5 text-[12px] text-fg-muted">
                      加载中…
                    </span>
                  </div>
                )}
              </>
            )}
          </main>
          <FrameList />
        </div>
      )}
      <StatusBar />
    </div>
  );
}

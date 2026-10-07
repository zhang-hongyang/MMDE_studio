// Scene page (M2): fused-scene point cloud viewer + fly-through vehicle
// replay. State lives in stores.ts; playback in replay/; see those files for
// the pc_viewer.html ports. Route: /scene/:dataset/:split.

import { useCallback, useEffect, useRef } from "react";
import { Navigate, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Telescope } from "lucide-react";
import { useRegistryStore } from "../../stores/registry";
import { api } from "../../lib/api";
import { useSceneStore, DEFAULT_BUDGET } from "./stores";
import { SceneToolbar } from "./components/SceneToolbar";
import { SceneViewport, type ViewportApi } from "./components/SceneViewport";
import { TrajectoryMap, SegmentToggles } from "./components/TrajectoryMap";
import { ReplayBar } from "./components/ReplayBar";
import { useReplay } from "./replay/useReplay";
import { Button } from "../../components/ui/button";
import { Skeleton } from "../../components/ui/skeleton";

/** first split that has fused scenes, else the first split */
function preferredSplit(ds: { splits: { name: string; scene_models: string[] }[] }): string {
  const withScenes = ds.splits.find((s) => s.scene_models.length > 0);
  return withScenes?.name ?? ds.splits[0]?.name ?? "";
}

export function ScenePage() {
  const { dataset = "", split = "" } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const registry = useRegistryStore((s) => s.registry);

  const dsInfo = registry?.datasets.find((d) => d.name === dataset);
  const splitInfo = dsInfo?.splits.find((sp) => sp.name === split);
  const models = splitInfo?.scene_models ?? [];
  const modelsKey = models.join(",");

  const model = useSceneStore((s) => s.model);
  const budget = useSceneStore((s) => s.budget);
  const index = useSceneStore((s) => s.index);
  const error = useSceneStore((s) => s.error);
  const status = useSceneStore((s) => s.status);
  const flyOn = useSceneStore((s) => s.fly.on);
  const flyFrame = useSceneStore((s) => s.fly.frame);

  // initialize the store from route + search params (once per dataset/split)
  const initedRef = useRef("");
  useEffect(() => {
    if (!dataset || !split || !registry) return;
    if (models.length === 0) return; // split without scenes: redirect below
    const key = `${dataset}/${split}/${modelsKey}`;
    if (initedRef.current === key) return;
    initedRef.current = key;
    const qBudget = parseFloat(searchParams.get("budget") ?? "");
    void useSceneStore.getState().init(dataset, split, models, {
      model: searchParams.get("model") ?? undefined,
      budget: Number.isFinite(qBudget) && qBudget > 0 ? qBudget : undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, split, modelsKey, registry]);

  // write display state back to the URL (shareable / restorable)
  useEffect(() => {
    if (!model) return;
    const p = new URLSearchParams();
    if (model) p.set("model", model);
    if (budget !== DEFAULT_BUDGET) p.set("budget", String(budget));
    setSearchParams(p, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, budget]);

  const viewportApi = useRef<ViewportApi | null>(null);
  const onViewportReady = useCallback((api: ViewportApi) => {
    viewportApi.current = api;
  }, []);
  useReplay(viewportApi);

  // split without fused scenes: route to the dataset's first scene split
  const needsRedirect =
    !!registry && !!dsInfo && !!splitInfo && splitInfo.scene_models.length === 0;
  const targetSplit = needsRedirect ? preferredSplit(dsInfo) : "";
  if (needsRedirect && targetSplit && targetSplit !== split) {
    return <Navigate to={`/scene/${dataset}/${targetSplit}`} replace />;
  }

  const showEmpty = !!registry && !!splitInfo && models.length === 0;
  const showInvalid = !!registry && !!dataset && !splitInfo;
  const showSkeleton = !registry || (!!splitInfo && models.length > 0 && !index && !error && !showEmpty);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <SceneToolbar />
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-[300px] shrink-0 flex-col gap-2 overflow-y-auto border-r border-border p-3">
          <TrajectoryMap apiRef={viewportApi} />
          <SegmentToggles />
          {flyOn && index && (
            <div className="card p-1.5">
              <img
                src={api.rgbURL(dataset, split, flyFrame, 320)}
                alt={`车载视角当前帧 ${flyFrame}`}
                className="w-full rounded-[4px]"
              />
              <div className="num px-0.5 pt-1 text-[12px] text-fg-muted">
                帧 {flyFrame} / {index.n_frames - 1}
              </div>
            </div>
          )}
          <p className="text-[12px] leading-5 text-fg-muted">
            3D 区左键旋转、右键平移、滚轮缩放，双击复位视角；点击轨迹图可平移视角。车载视角回放中 Space
            暂停 / Esc 退出，拖动进度条跳转。
          </p>
        </aside>
        <main
          className="relative min-w-0 flex-1"
          onDoubleClick={() => viewportApi.current?.refit()}
        >
          {showInvalid ? (
            <div className="flex h-full items-center justify-center p-4">
              <div className="card max-w-lg text-[13px]">
                数据集或 split 不存在：<span className="font-mono">{dataset}/{split}</span>
              </div>
            </div>
          ) : showEmpty ? (
            <div className="flex h-full items-center justify-center p-4">
              <div className="card flex max-w-lg flex-col items-start gap-3 text-[13px]">
                <p>
                  <span className="font-mono">{dataset}/{split}</span> 暂无融合场景模型。先在{" "}
                  <code className="font-mono">Explorer</code> 里确认预测，再运行{" "}
                  <code className="font-mono">scripts/fuse_depth_scene.py</code> 融合连续帧场景。
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate(`/explore/${dataset}/${split}`)}
                >
                  <Telescope className="h-3.5 w-3.5" />
                  去 Explorer
                </Button>
              </div>
            </div>
          ) : showSkeleton ? (
            <div className="flex h-full flex-col gap-2 p-2">
              <Skeleton className="flex-1" />
            </div>
          ) : error && !index ? (
            <div className="flex h-full items-center justify-center p-4">
              <div className="card max-w-lg text-[13px] text-danger">{error}</div>
            </div>
          ) : (
            <SceneViewport onReady={onViewportReady} />
          )}
          {flyOn && !showSkeleton && !showEmpty && <ReplayBar />}
        </main>
      </div>
      <div className="flex h-7 shrink-0 items-center border-t border-border bg-card px-3 text-[12px] text-fg-muted">
        <span className="truncate">{status || "就绪"}</span>
      </div>
    </div>
  );
}

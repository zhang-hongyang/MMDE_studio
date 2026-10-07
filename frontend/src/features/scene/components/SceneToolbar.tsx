// Scene toolbar (two wrapped rows, no horizontal scrolling):
//   Row 1: dataset (only those with fused scenes) · scene model · replay ctl
//   Row 2: point size · brightness · point budget · fly-through toggle

import { useNavigate } from "react-router-dom";
import { Car, Pause, Play, X } from "lucide-react";
import { useRegistryStore } from "../../../stores/registry";
import { useSceneStore } from "../stores";
import { Select } from "../../../components/ui/select";
import { ModelSelect } from "../../../lib/ModelSelect";
import { Slider } from "../../../components/ui/slider";
import { Button } from "../../../components/ui/button";
import { Tooltip } from "../../../components/ui/tooltip";
import { cn } from "../../../lib/utils";
import { SplitSelectors } from "../../../lib/SplitSelectors";

function Field({
  label,
  children,
  title,
}: {
  label: string;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <label className="inline-flex shrink-0 items-center gap-1.5 text-[12px] text-fg-muted" title={title}>
      <span className="whitespace-nowrap">{label}</span>
      {children}
    </label>
  );
}

/** first split that has fused scenes, else the first split */
function preferredSplit(ds: { splits: { name: string; scene_models?: string[] }[] | undefined }): string {
  const withScenes = ds.splits?.find((s) => (s.scene_models?.length ?? 0) > 0);
  return withScenes?.name ?? ds.splits?.[0]?.name ?? "";
}

/** vertical divider with a fixed height — the Separator component's h-full
 *  would fight the auto-height wrapped toolbar (circular percentage) */
function VDiv() {
  return <span aria-hidden className="mx-1 h-6 w-px shrink-0 self-center bg-border" />;
}

export function SceneToolbar() {
  const navigate = useNavigate();
  const registry = useRegistryStore((s) => s.registry);
  const s = useSceneStore();

  // only datasets that have at least one split with fused scene models
  const sceneDatasets = (registry?.datasets ?? []).filter((d) =>
    d.splits.some((sp) => sp.scene_models.length > 0),
  );
  const dsInfo = registry?.datasets.find((d) => d.name === s.dataset);
  const splitInfo = dsInfo?.splits.find((sp) => sp.name === s.split);
  const models = splitInfo?.scene_models ?? [];
  const canFly = !!s.index?.cam_quat?.length && s.fly.plen > 0;

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-border bg-card px-3 py-1.5">
      {/* Row 1: navigation + replay controls */}
      <div className="flex flex-wrap items-center gap-2">
        <Field label="数据集">
          <Select.Root
            value={s.dataset}
            onValueChange={(ds) => {
              const d = registry?.datasets.find((x) => x.name === ds);
              if (d) navigate(`/scene/${ds}/${preferredSplit(d)}`);
            }}
          >
            <Select.Trigger className="w-[150px]" aria-label="数据集">
              <Select.Value placeholder="数据集" />
            </Select.Trigger>
            <Select.Content>
              {sceneDatasets.map((d) => (
                <Select.Item key={d.name} value={d.name}>
                  {d.title || d.name}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
        </Field>
        <SplitSelectors
          dataset={dsInfo}
          split={s.split}
          sceneOnly
          onSplitChange={(sp) => navigate(`/scene/${s.dataset}/${sp}`)}
        />
        <Field label="场景模型">
          <ModelSelect
            models={models}
            value={s.model}
            ariaLabel="场景模型"
            className="w-[150px]"
            onValueChange={(v) => void s.setModel(v)}
          />
        </Field>

        {s.fly.on && (
          <>
            <VDiv />
            {/* compact replay controls; the full scrub bar stays the floating ReplayBar */}
            <div className="flex items-center gap-1">
              <Button
                variant="secondary"
                size="sm"
                aria-label={s.fly.playing ? "暂停回放（Space）" : "继续回放（Space）"}
                onClick={() => s.setFlyPlaying(!s.fly.playing)}
              >
                {s.fly.playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              </Button>
              <Button variant="ghost" size="sm" aria-label="退出回放（Esc）" onClick={() => s.exitFly()}>
                <X className="h-3.5 w-3.5" />
                退出
              </Button>
            </div>
          </>
        )}
      </div>

      <VDiv />

      {/* Row 2: display controls */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Field label="点大小" title="点尺寸乘子（1 = 按体素默认）">
          <Slider.Root
            className="w-24"
            min={0.2}
            max={4}
            step={0.05}
            value={[s.psize]}
            onValueChange={(v) => s.setPsize(v[0])}
            aria-label="点大小"
          >
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumb />
          </Slider.Root>
        </Field>
        <Field label="亮度" title="点云曝光乘子（整体线性缩放）">
          <Slider.Root
            className="w-24"
            min={0.4}
            max={2}
            step={0.05}
            value={[s.brightness]}
            onValueChange={(v) => s.setBrightness(v[0])}
            aria-label="亮度"
          >
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumb />
          </Slider.Root>
        </Field>
        <Field label="点预算" title="就近细节块的点预算（软钳制）">
          <Slider.Root
            className="w-28"
            min={0.5}
            max={10}
            step={0.5}
            value={[s.budget / 1e6]}
            onValueChange={(v) => s.setBudget(v[0] * 1e6)}
            aria-label="点预算（百万点）"
          >
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumb />
          </Slider.Root>
          <span className="num w-10">{(s.budget / 1e6).toFixed(1)}M</span>
        </Field>

        <VDiv />

        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Button
              variant={s.fly.on ? "default" : "outline"}
              size="sm"
              aria-label="车载视角回放（沿轨迹重放，Space 暂停 / Esc 退出）"
              aria-pressed={s.fly.on}
              disabled={!canFly}
              className={cn(!canFly && "pointer-events-auto")}
              onClick={() => (s.fly.on ? s.exitFly() : s.enterFly())}
            >
              <Car className="h-3.5 w-3.5" />
              车载视角
            </Button>
          </Tooltip.Trigger>
          <Tooltip.Content>
            {canFly
              ? "沿轨迹重放车载相机视角（Space 暂停 / Esc 退出）"
              : "场景索引缺位姿数据或没有选中片段，无法回放"}
          </Tooltip.Content>
        </Tooltip.Root>
      </div>
    </div>
  );
}

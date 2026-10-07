// Fly-through replay control bar (port of pc_viewer.html flyCtl): play/pause,
// speed, view mode, second-granularity scrub, time readout, exit.

import { Pause, Play, X } from "lucide-react";
import { useSceneStore } from "../stores";
import { Button } from "../../../components/ui/button";
import { Select } from "../../../components/ui/select";
import { Slider } from "../../../components/ui/slider";

export function ReplayBar() {
  const fly = useSceneStore((s) => s.fly);
  const scrubMax = Math.max(0, Math.ceil(fly.plen) - 1);
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
      <div className="pointer-events-auto flex max-w-full items-center gap-2 rounded-[8px] border border-border bg-card/90 px-3 py-2 backdrop-blur">
        <Button
          variant="secondary"
          size="sm"
          aria-label={fly.playing ? "暂停回放（Space）" : "继续回放（Space）"}
          onClick={() => useSceneStore.getState().setFlyPlaying(!fly.playing)}
        >
          {fly.playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </Button>
        <Select.Root
          value={String(fly.speed)}
          onValueChange={(v) => useSceneStore.getState().setFlySpeed(parseFloat(v))}
        >
          <Select.Trigger className="w-[64px]" aria-label="回放倍速">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {[0.5, 1, 2, 4].map((v) => (
              <Select.Item key={v} value={String(v)}>
                {v}×
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
        <Select.Root
          value={fly.view}
          onValueChange={(v) => useSceneStore.getState().setFlyView(v as "car" | "chase")}
        >
          <Select.Trigger className="w-[104px]" aria-label="回放视角">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            <Select.Item value="car">车载相机</Select.Item>
            <Select.Item value="chase">车后跟随</Select.Item>
          </Select.Content>
        </Select.Root>
        <Slider.Root
          className="w-40 sm:w-56"
          min={0}
          max={scrubMax}
          step={1}
          value={[Math.min(fly.p, scrubMax)]}
          onValueChange={(v) => useSceneStore.getState().setFlyP(v[0])}
          aria-label="回放进度（秒）"
        >
          <Slider.Track>
            <Slider.Range />
          </Slider.Track>
          <Slider.Thumb />
        </Slider.Root>
        <span className="num whitespace-nowrap text-[12px] text-fg-muted">
          帧 {fly.frame} · {Math.min(fly.p, Math.ceil(fly.plen))}/{fly.plen.toFixed(0)}s
        </span>
        <Button
          variant="ghost"
          size="sm"
          aria-label="退出回放（Esc）"
          onClick={() => useSceneStore.getState().exitFly()}
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

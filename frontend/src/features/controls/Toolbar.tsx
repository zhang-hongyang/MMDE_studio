import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";
import { useRegistryStore } from "../../stores/registry";
import {
  useControlsStore,
  COLOR_MODE_LABEL,
  type CtrlColorMode,
  type CtrlColormap,
} from "../../stores/controls";
import { Select } from "../../components/ui/select";
import { ModelSelect } from "../../lib/ModelSelect";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Slider } from "../../components/ui/slider";
import { Separator } from "../../components/ui/separator";
import { Tooltip } from "../../components/ui/tooltip";
import { cn } from "../../lib/utils";
import { SplitSelectors } from "../../lib/SplitSelectors";

function Field({ label, children, title }: { label: string; children: React.ReactNode; title?: string }) {
  return (
    <label className="inline-flex shrink-0 items-center gap-1.5 text-[12px] text-fg-muted" title={title}>
      <span className="whitespace-nowrap">{label}</span>
      {children}
    </label>
  );
}

const COLOR_MODES: CtrlColorMode[] = ["depth", "weight", "errpred", "errgt"];

export function ControlsToolbar() {
  const navigate = useNavigate();
  const registry = useRegistryStore((s) => s.registry);
  const s = useControlsStore();

  const dsInfo = registry?.datasets.find((d) => d.name === s.dataset);

  const [idxText, setIdxText] = useState(String(s.idx));
  useEffect(() => setIdxText(String(s.idx)), [s.idx]);
  const commitIdx = () => {
    const v = parseInt(idxText, 10);
    if (Number.isFinite(v)) s.setIdx(v);
    else setIdxText(String(s.idx));
  };

  const togglePlay = () => s.set({ playing: !s.playing });

  return (
    <div className="flex h-14 shrink-0 items-center gap-2 overflow-x-auto border-b border-border bg-card px-3 whitespace-nowrap">
      {/* dataset / split — from /api/registry */}
      <Field label="数据集">
        <Select.Root
          value={s.dataset}
          onValueChange={(ds) => {
            const d = registry?.datasets.find((x) => x.name === ds);
            if (d) navigate(`/controls/${ds}/${d.splits[0]?.name ?? ""}`);
          }}
        >
          <Select.Trigger className="w-[140px]" aria-label="数据集">
            <Select.Value placeholder="数据集" />
          </Select.Trigger>
          <Select.Content>
            {(registry?.datasets ?? []).map((d) => (
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
        onSplitChange={(sp) => navigate(`/controls/${s.dataset}/${sp}`)}
      />

      <Separator orientation="vertical" className="mx-1 h-6" />

      {/* control source multi-select chips */}
      <Field label="控制源" title="叠加显示的控制点源（可多选）">
        <span className="inline-flex items-center gap-1">
          {s.controlSources.length === 0 && (
            <span className="text-fg-muted/70">（无控制点源）</span>
          )}
          {s.controlSources.map((src) => {
            const on = s.enabled.includes(src);
            return (
              <button
                key={src}
                type="button"
                aria-pressed={on}
                aria-label={`控制源 ${src}`}
                onClick={() => s.toggleSource(src)}
                className={cn(
                  "h-[28px] rounded-[6px] border px-2 font-mono text-[12px]",
                  "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  on
                    ? "border-primary bg-primary/15 text-fg"
                    : "border-border bg-card-2 text-fg-muted hover:bg-border/50",
                )}
              >
                {src}
              </button>
            );
          })}
        </span>
      </Field>

      <Separator orientation="vertical" className="mx-1 h-6" />

      {/* frame navigation */}
      <div className="inline-flex shrink-0 items-center gap-1">
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Button
              variant="outline"
              size="sm"
              aria-label="上一帧"
              disabled={s.idx <= 0}
              onClick={() => s.setIdx(s.idx - 1)}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
          </Tooltip.Trigger>
          <Tooltip.Content>上一帧</Tooltip.Content>
        </Tooltip.Root>
        <Input
          className="w-[64px] text-center"
          value={idxText}
          aria-label="帧序号"
          onChange={(e) => setIdxText(e.target.value)}
          onBlur={commitIdx}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitIdx();
          }}
        />
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Button
              variant="outline"
              size="sm"
              aria-label="下一帧"
              disabled={s.idx >= s.frames.length - 1}
              onClick={() => s.setIdx(s.idx + 1)}
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </Tooltip.Trigger>
          <Tooltip.Content>下一帧</Tooltip.Content>
        </Tooltip.Root>
        <span className="num text-[12px] text-fg-muted">/ {Math.max(0, s.frames.length - 1)}</span>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Button
              variant={s.playing ? "secondary" : "outline"}
              size="sm"
              aria-label={s.playing ? "暂停播放" : "播放"}
              disabled={s.frames.length === 0}
              onClick={togglePlay}
            >
              {s.playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </Button>
          </Tooltip.Trigger>
          <Tooltip.Content>{s.playing ? "暂停" : "逐帧播放"}</Tooltip.Content>
        </Tooltip.Root>
        <Field label="间隔" title="播放帧间隔（毫秒）">
          <Select.Root
            value={String(s.playInterval)}
            onValueChange={(v) => s.set({ playInterval: parseInt(v, 10) || 400 })}
          >
            <Select.Trigger className="w-[84px]" aria-label="播放间隔">
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              {[200, 400, 600, 800, 1000].map((ms) => (
                <Select.Item key={ms} value={String(ms)}>
                  {ms} ms
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
        </Field>
      </div>

      <Separator orientation="vertical" className="mx-1 h-6" />

      {/* model */}
      <Field label="模型" title="选择后控制点返回 d_pred_aligned，用于 err=模型 着色">
        <ModelSelect
          models={s.models}
          value={s.model}
          ariaLabel="模型"
          className="w-[140px]"
          onValueChange={(v) => {
            s.set({ model: v });
            void s.reload();
          }}
        />
      </Field>

      <Separator orientation="vertical" className="mx-1 h-6" />

      {/* display */}
      <Field label="着色">
        <Select.Root
          value={s.colorMode}
          onValueChange={(v) => s.set({ colorMode: v as CtrlColorMode })}
        >
          <Select.Trigger className="w-[110px]" aria-label="着色模式">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {COLOR_MODES.map((m) => (
              <Select.Item key={m} value={m}>
                {COLOR_MODE_LABEL[m]}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
      </Field>
      <Field label="色标">
        <Select.Root
          value={s.colormap}
          onValueChange={(v) => s.set({ colormap: v as CtrlColormap })}
        >
          <Select.Trigger className="w-[90px]" aria-label="色标">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            <Select.Item value="turbo">turbo</Select.Item>
            <Select.Item value="plasma">plasma</Select.Item>
          </Select.Content>
        </Select.Root>
      </Field>
      <Field label="点大小">
        <Slider.Root
          className="w-24"
          min={1}
          max={8}
          step={1}
          value={[s.psize]}
          onValueChange={(v) => s.set({ psize: v[0] })}
          aria-label="点大小"
        >
          <Slider.Track>
            <Slider.Range />
          </Slider.Track>
          <Slider.Thumb />
        </Slider.Root>
      </Field>
      <span className="num text-[12px] text-fg-muted">{s.psize}px</span>
    </div>
  );
}

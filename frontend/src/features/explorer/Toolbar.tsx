import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight, HelpCircle } from "lucide-react";
import { useRegistryStore } from "../../stores/registry";
import { useExploreStore, type CompareMode, type Coloring } from "../../stores/explore";
import { ModelSelect } from "../../lib/ModelSelect";
import { Select } from "../../components/ui/select";
import { Tabs } from "../../components/ui/tabs";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Switch } from "../../components/ui/switch";
import { Slider } from "../../components/ui/slider";
import { Dialog } from "../../components/ui/dialog";
import { Tooltip } from "../../components/ui/tooltip";
import { SplitSelectors } from "../../lib/SplitSelectors";

function Field({ label, children, title }: { label: string; children: React.ReactNode; title?: string }) {
  return (
    <label className="inline-flex shrink-0 items-center gap-1.5 text-[12px] text-fg-muted" title={title}>
      <span className="whitespace-nowrap">{label}</span>
      {children}
    </label>
  );
}

const MODE_LABEL: Record<CompareMode, string> = {
  model: "模型对比",
  frame: "帧间对比",
  overlay: "叠加",
};

const COLORING_OPTIONS: { value: Coloring; label: string }[] = [
  { value: "rgb", label: "RGB" },
  { value: "diff", label: "|A−B| 差异" },
  { value: "fdiff", label: "Δd 帧间差异" },
  { value: "fcolor", label: "帧色（A橙/B蓝）" },
];

function coloringOptionsFor(mode: CompareMode): Coloring[] {
  return mode === "frame" ? ["fdiff", "rgb", "fcolor"] : ["rgb", "diff"];
}

export function ExplorerToolbar() {
  const navigate = useNavigate();
  const registry = useRegistryStore((s) => s.registry);
  const s = useExploreStore();

  const dsInfo = registry?.datasets.find((d) => d.name === s.dataset);
  const splitInfo = dsInfo?.splits.find((sp) => sp.name === s.split);
  const cameras = splitInfo?.cameras ?? [];
  const showCamera = cameras.length > 1;

  const [idxText, setIdxText] = useState(String(s.idx));
  useEffect(() => setIdxText(String(s.idx)), [s.idx]);

  const commitIdx = () => {
    const v = parseInt(idxText, 10);
    if (Number.isFinite(v)) s.setIdx(v);
    else setIdxText(String(s.idx));
  };

  const onMode = (mode: CompareMode) => {
    if (mode === s.mode) return;
    const coloring: Coloring = mode === "frame" ? "fdiff" : "rgb";
    s.set({ mode, coloring, onlyOverlap: false });
    void s.reload().then(() => s.requestHomeView());
  };

  const coloringOpts = coloringOptionsFor(s.mode);
  const effectiveColoring = coloringOpts.includes(s.coloring) ? s.coloring : coloringOpts[0];

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-border bg-card px-3 py-1.5">
      {/* dataset / split / camera — all from /api/registry */}
      <div className="flex items-center gap-2">
      <Field label="数据集">
        <Select.Root
          value={s.dataset}
          onValueChange={(ds) => {
            const d = registry?.datasets.find((x) => x.name === ds);
            if (d) navigate(`/explore/${ds}/${d.splits[0]?.name ?? ""}`);
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
        onSplitChange={(sp) => navigate(`/explore/${s.dataset}/${sp}`)}
      />
      {showCamera && (
        <Field label="相机" title="多相机数据集：按流过滤帧序列">
          <Select.Root
            value={s.camera}
            onValueChange={(camera) => s.setCamera(camera)}
          >
            <Select.Trigger className="w-[110px]" aria-label="相机">
              <Select.Value placeholder="相机" />
            </Select.Trigger>
            <Select.Content>
              {cameras.map((c) => (
                <Select.Item key={c} value={c}>
                  {c}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
        </Field>
      )}
      </div>

      <span aria-hidden className="mx-1 h-6 w-px shrink-0 self-center bg-border" />

      {/* mode + frame navigation */}
      <div className="flex items-center gap-2">
      <Tabs.Root value={s.mode} onValueChange={(v) => onMode(v as CompareMode)}>
        <Tabs.List>
          {(Object.keys(MODE_LABEL) as CompareMode[]).map((m) => (
            <Tabs.Trigger key={m} value={m}>
              {MODE_LABEL[m]}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
      </Tabs.Root>

      <span aria-hidden className="mx-1 h-6 w-px shrink-0 self-center bg-border" />

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
      </div>

      {/* frame compare extras */}
      {s.mode === "frame" && (
        <>
          <Field label="Δ" title="帧B与帧A的帧号差">
            <Input
              className="w-[56px]"
              type="number"
              min={1}
              max={60}
              value={s.delta}
              onChange={(e) => {
                const v = Math.max(1, Math.min(60, parseInt(e.target.value, 10) || 1));
                s.set({ delta: v });
              }}
              onBlur={() => void s.reload().then(() => s.requestHomeView())}
              onKeyDown={(e) => {
                if (e.key === "Enter") void s.reload().then(() => s.requestHomeView());
              }}
            />
          </Field>
          <Field label="只看重叠" title="只显示两帧深度差在阈值内的重叠点">
            <Switch
              checked={s.onlyOverlap}
              onCheckedChange={(v) => s.set({ onlyOverlap: v })}
              aria-label="只看重叠"
            />
          </Field>
        </>
      )}
      </div>

      <span aria-hidden className="mx-1 h-6 w-px shrink-0 self-center bg-border" />

      {/* model selects */}
      <div className="flex items-center gap-2">
      <Field label="模型A">
        <ModelSelect
          models={s.models}
          value={s.modelA}
          ariaLabel="模型A"
          className="w-[120px]"
          onValueChange={(v) => {
            s.set({ modelA: v });
            void s.reload().then(() => s.requestHomeView());
          }}
        />
      </Field>
      {s.mode !== "frame" && (
        <Field label="模型B">
          <ModelSelect
            models={s.models}
            value={s.modelB}
            ariaLabel="模型B"
            className="w-[120px]"
            onValueChange={(v) => {
              s.set({ modelB: v });
              void s.reload().then(() => s.requestHomeView());
            }}
          />
        </Field>
      )}
      </div>

      <span aria-hidden className="mx-1 h-6 w-px shrink-0 self-center bg-border" />

      {/* display / view options */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <Field label="着色">
        <Select.Root
          value={effectiveColoring}
          onValueChange={(v) => s.set({ coloring: v as Coloring })}
        >
          <Select.Trigger className="w-[130px]" aria-label="着色模式">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {COLORING_OPTIONS.filter((o) => coloringOpts.includes(o.value)).map((o) => (
              <Select.Item key={o.value} value={o.value}>
                {o.label}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
      </Field>
      {effectiveColoring === "diff" && (
        <Field label="色标">
          <Select.Root value={s.colormap} onValueChange={(v) => s.set({ colormap: v as typeof s.colormap })}>
            <Select.Trigger className="w-[90px]" aria-label="差异色标">
              <Select.Value />
            </Select.Trigger>
            <Select.Content>
              <Select.Item value="turbo">turbo</Select.Item>
              <Select.Item value="jet">jet</Select.Item>
              <Select.Item value="plasma">plasma</Select.Item>
            </Select.Content>
          </Select.Root>
        </Field>
      )}
      <Field label="点大小">
        <Slider.Root
          className="w-24"
          min={0.01}
          max={0.3}
          step={0.005}
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
      <Field label="亮度" title="点云曝光乘子（整体线性缩放，差异着色等相对颜色不受影响）">
        <Slider.Root
          className="w-24"
          min={0.4}
          max={2}
          step={0.05}
          value={[s.brightness]}
          onValueChange={(v) => s.set({ brightness: v[0] })}
          aria-label="亮度"
        >
          <Slider.Track>
            <Slider.Range />
          </Slider.Track>
          <Slider.Thumb />
        </Slider.Root>
      </Field>
      <Field label="GT">
        <Switch checked={s.showGT} onCheckedChange={(v) => s.set({ showGT: v })} aria-label="显示GT" />
      </Field>
      <Field label="控制点">
        <Switch
          checked={s.showCtrl}
          onCheckedChange={(v) => {
            s.set({ showCtrl: v });
            void s.reload();
          }}
          aria-label="显示控制点"
        />
      </Field>
      {s.showCtrl && (
        <>
          <Field label="源">
            <Select.Root value={s.ctrlSrc} onValueChange={(v) => {
              s.set({ ctrlSrc: v });
              void s.reload();
            }}>
              <Select.Trigger className="w-[120px]" aria-label="控制源">
                <Select.Value placeholder="源" />
              </Select.Trigger>
              <Select.Content>
                {s.controlSources.map((c) => (
                  <Select.Item key={c} value={c}>
                    {c}
                  </Select.Item>
                ))}
              </Select.Content>
            </Select.Root>
          </Field>
          <Field label="点着色">
            <Select.Root value={s.ctrlColor} onValueChange={(v) => s.set({ ctrlColor: v as typeof s.ctrlColor })}>
              <Select.Trigger className="w-[100px]" aria-label="控制点着色">
                <Select.Value />
              </Select.Trigger>
              <Select.Content>
                <Select.Item value="errpred">err=模型</Select.Item>
                <Select.Item value="errgt">err=真值</Select.Item>
                <Select.Item value="depth">深度</Select.Item>
                <Select.Item value="weight">权重</Select.Item>
              </Select.Content>
            </Select.Root>
          </Field>
        </>
      )}
      <Field label="同步视角" title="把 A 视口相机复制到 B 视口">
        <Switch checked={s.syncView} onCheckedChange={(v) => s.set({ syncView: v })} aria-label="同步视角" />
      </Field>
      </div>

      <div className="ml-auto">
      <Dialog.Root>
        <Dialog.Trigger asChild>
          <Button variant="ghost" size="sm" aria-label="帮助" className="ml-auto">
            <HelpCircle className="h-4 w-4" />
          </Button>
        </Dialog.Trigger>
        <Dialog.Content>
          <Dialog.Title>操作说明</Dialog.Title>
          <Dialog.Description asChild>
            <div className="space-y-1.5 text-[13px] leading-5">
              <p>· 3D 视口：左键旋转、右键平移、滚轮缩放、双击复位视角。</p>
              <p>· 缩略图：拖拽框选局部区域过滤 3D 点，双击恢复全帧。</p>
              <p>· 帧间对比用 meta.T 把帧 B 配准到帧 A 坐标系做时序差分。</p>
              <p>· 着色 |A−B| 与 Δd 按 P95 归一化；差异色标见视口下方图例。</p>
            </div>
          </Dialog.Description>
        </Dialog.Content>
      </Dialog.Root>
      </div>
    </div>
  );
}

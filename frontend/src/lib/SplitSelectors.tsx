import { Select } from "../components/ui/select";
import type { DatasetInfo, SplitInfo } from "./api";

export type TestType = "test_single" | "test_sequence";

const TYPE_LABEL: Record<TestType, string> = {
  test_single: "test_single · 单帧测试",
  test_sequence: "test_sequence · 连续序列",
};

export function splitLabel(split: SplitInfo): string {
  if (split.name === "eigen_test") return "Eigen test · 可评估";
  if (split.name === "official_test_anonymous") return "Official test · 仅推理";
  if (split.name === "val_official") return "Official val · 可评估";
  if (split.name === "official_test") return "Official test · 本地 LiDAR 评估";
  const match = split.name.match(/sequence_seed\d+_(\d+)$/);
  return match ? `连续序列 ${Number(match[1])}` : split.name;
}

interface Props {
  dataset?: DatasetInfo;
  split: string;
  onSplitChange: (split: string) => void;
  sceneOnly?: boolean;
}

export function SplitSelectors({ dataset, split, onSplitChange, sceneOnly = false }: Props) {
  const all = (dataset?.splits ?? []).filter(
    (item) => !sceneOnly || item.scene_models.length > 0,
  );
  const current = all.find((item) => item.name === split) ?? all[0];
  const type = (current?.test_type ?? "test_single") as TestType;
  const types = (["test_single", "test_sequence"] as TestType[]).filter((value) =>
    all.some((item) => item.test_type === value),
  );
  const subsets = all.filter((item) => item.test_type === type);

  return (
    <>
      <label className="inline-flex shrink-0 items-center gap-1.5 text-[12px] text-fg-muted">
        <span className="whitespace-nowrap">类型</span>
        <Select.Root
          value={type}
          onValueChange={(value) => {
            const next = all.find((item) => item.test_type === value);
            if (next) onSplitChange(next.name);
          }}
        >
          <Select.Trigger className="w-[190px] font-mono" aria-label="测试类型">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {types.map((value) => (
              <Select.Item key={value} value={value}>
                {TYPE_LABEL[value]}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
      </label>
      <label className="inline-flex shrink-0 items-center gap-1.5 text-[12px] text-fg-muted">
        <span className="whitespace-nowrap">{type === "test_sequence" ? "序列" : "子集"}</span>
        <Select.Root value={current?.name ?? ""} onValueChange={onSplitChange}>
          <Select.Trigger className="w-[190px] font-mono" aria-label="测试子集">
            <Select.Value />
          </Select.Trigger>
          <Select.Content>
            {subsets.map((item) => (
              <Select.Item key={item.name} value={item.name}>
                {splitLabel(item)} · {item.n_frames} 帧
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
      </label>
    </>
  );
}

import { useEffect, useMemo, useState } from "react";
import { PlusCircle } from "lucide-react";
import { useRegistryStore } from "../../stores/registry";
import { Button } from "../../components/ui/button";
import { Dialog } from "../../components/ui/dialog";
import { Input } from "../../components/ui/input";
import { Select } from "../../components/ui/select";
import { Switch } from "../../components/ui/switch";
import { submitTask, type Task, type TaskParamSpec, type TaskSpec } from "./api";
import { cn } from "../../lib/utils";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  catalog: TaskSpec[];
  onCreated: (task: Task) => void;
}

type FieldValue = string | boolean;

/** the backend omits `choices` for free-text params and sends choices:null
 * only for the dynamic model selector (see spec_to_json in app/tasks/catalog.py) */
function isDynamicModel(p: TaskParamSpec): boolean {
  return p.choices === null && p.type === "str" && p.name === "model";
}

function FieldLabel({ p, inline }: { p: TaskParamSpec; inline?: boolean }) {
  return (
    <span
      className={cn(
        "flex items-center gap-1 text-[12px] text-fg-muted",
        inline ? "mb-0" : "mb-1",
      )}
    >
      {p.label}
      {p.required && <span className="text-danger">*</span>}
      {p.help && (
        <span className="truncate text-fg-muted/70" title={p.help}>
          （{p.help}）
        </span>
      )}
    </span>
  );
}

function defaultFields(spec: TaskSpec | null): Record<string, FieldValue> {
  const out: Record<string, FieldValue> = {};
  for (const p of spec?.params ?? []) {
    if (p.type === "bool") out[p.name] = p.default === true;
    else if (p.default !== null && p.default !== undefined) out[p.name] = String(p.default);
    else out[p.name] = "";
  }
  return out;
}

export function TaskWizard({ open, onOpenChange, catalog, onCreated }: Props) {
  const registry = useRegistryStore((s) => s.registry);
  const [type, setType] = useState("");
  const [fields, setFields] = useState<Record<string, FieldValue>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const spec = catalog.find((s) => s.type === type) ?? null;

  // reset form when opening / switching type
  useEffect(() => {
    if (open) {
      setType((t) => (catalog.some((s) => s.type === t) ? t : (catalog[0]?.type ?? "")));
      setErrors({});
      setSubmitError(null);
    }
  }, [open, catalog]);
  useEffect(() => {
    setFields(defaultFields(spec));
    setErrors({});
  }, [spec, open]);

  // dynamic model choices from the registry (dataset/split selected in the form)
  const dynModels = useMemo(() => {
    if (!spec) return [];
    const ds = String(fields.dataset ?? "");
    const sp = String(fields.split ?? "");
    const models = registry?.datasets.find((d) => d.name === ds)?.splits.find((s) => s.name === sp)
      ?.models;
    return models ?? [];
  }, [spec, fields.dataset, fields.split, registry]);

  const set = (name: string, v: FieldValue) => setFields((f) => ({ ...f, [name]: v }));

  const validate = (): boolean => {
    const errs: Record<string, string> = {};
    for (const p of spec?.params ?? []) {
      const raw = fields[p.name];
      const empty = p.type === "bool" ? false : typeof raw !== "string" || raw.trim() === "";
      if (p.required && empty) {
        errs[p.name] = "必填";
        continue;
      }
      if (empty) continue;
      if (p.type === "int" && !/^-?\d+$/.test(String(raw).trim()))
        errs[p.name] = "应为整数";
      if (p.type === "float" && !/^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(String(raw).trim()))
        errs[p.name] = "应为数字";
      if (p.type === "list" && String(raw).split(",").every((s) => s.trim() === ""))
        errs[p.name] = "至少一个值";
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const onSubmit = () => {
    if (!spec || !validate()) return;
    const params: Record<string, unknown> = {};
    for (const p of spec.params) {
      const raw = fields[p.name];
      if (p.type === "bool") {
        if (raw === true) params[p.name] = true;
      } else if (typeof raw === "string" && raw.trim() !== "") {
        const v = raw.trim();
        if (p.type === "int") params[p.name] = parseInt(v, 10);
        else if (p.type === "float") params[p.name] = parseFloat(v);
        else if (p.type === "list")
          params[p.name] = v.split(",").map((s) => s.trim()).filter(Boolean);
        else params[p.name] = v;
      }
    }
    setSubmitting(true);
    setSubmitError(null);
    submitTask({ type: spec.type, params })
      .then((task) => {
        onOpenChange(false);
        onCreated(task);
      })
      .catch((e) => setSubmitError(String(e)))
      .finally(() => setSubmitting(false));
  };

  const renderField = (p: TaskParamSpec) => {
    const err = errors[p.name];

    if (Array.isArray(p.choices)) {
      const cur = String(fields[p.name] ?? "");
      return (
        <div key={p.name}>
          <FieldLabel p={p} />
          <Select.Root
            value={p.choices.includes(cur) ? cur : String(p.default ?? "")}
            onValueChange={(v) => set(p.name, v)}
          >
            <Select.Trigger className="w-full font-mono" aria-invalid={!!err}>
              <Select.Value placeholder={p.required ? "选择…" : "（默认）"} />
            </Select.Trigger>
            <Select.Content>
              {p.choices.map((c) => (
                <Select.Item key={c} value={c}>
                  {c}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          {err && <span className="text-[12px] text-danger">{err}</span>}
        </div>
      );
    }

    if (isDynamicModel(p) && dynModels.length > 0) {
      const cur = String(fields[p.name] ?? "");
      return (
        <div key={p.name}>
          <FieldLabel p={p} />
          <Select.Root value={dynModels.includes(cur) ? cur : ""} onValueChange={(v) => set(p.name, v)}>
            <Select.Trigger className="w-full font-mono" aria-invalid={!!err}>
              <Select.Value placeholder={dynModels.length ? "选择模型…" : "无可用模型"} />
            </Select.Trigger>
            <Select.Content>
              {dynModels.map((c) => (
                <Select.Item key={c} value={c}>
                  {c}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          {err && <span className="text-[12px] text-danger">{err}</span>}
        </div>
      );
    }

    if (p.type === "bool") {
      return (
        <div key={p.name} className="flex h-[32px] items-center gap-2">
          <Switch checked={fields[p.name] === true} onCheckedChange={(v) => set(p.name, v)} />
          <FieldLabel p={p} inline />
        </div>
      );
    }

    return (
      <div key={p.name}>
        <FieldLabel p={p} />
        <Input
          className="w-full"
          aria-invalid={!!err}
          placeholder={p.type === "list" ? "逗号分隔，如 a, b, c" : p.name}
          value={String(fields[p.name] ?? "")}
          onChange={(e) => set(p.name, e.target.value)}
        />
        {err && <span className="text-[12px] text-danger">{err}</span>}
      </div>
    );
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content className="max-h-[85vh] overflow-auto">
        <Dialog.Title>新建任务</Dialog.Title>
        {catalog.length === 0 ? (
          <Dialog.Description>后端未提供任何任务类型（/api/tasks/catalog 为空）。</Dialog.Description>
        ) : (
          <>
            <div className="grid gap-3">
              <div>
                <span className="mb-1 block text-[12px] text-fg-muted">任务类型</span>
                <Select.Root value={type} onValueChange={setType}>
                  <Select.Trigger className="w-full">
                    <Select.Value />
                  </Select.Trigger>
                  <Select.Content>
                    {catalog.map((s) => (
                      <Select.Item key={s.type} value={s.type}>
                        {s.title}
                        <span className="ml-1 font-mono text-fg-muted">（{s.type}）</span>
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Root>
                {spec && (
                  <p className="mt-1 text-[12px] leading-4 text-fg-muted">{spec.description}</p>
                )}
              </div>
              {(spec?.params ?? []).map(renderField)}
            </div>
            {submitError && (
              <p className="rounded-[6px] border border-danger/40 bg-danger/10 px-2 py-1 text-[12px] text-danger">
                提交失败:{submitError}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <Dialog.Close asChild>
                <Button variant="outline" size="md">
                  取消
                </Button>
              </Dialog.Close>
              <Button onClick={onSubmit} disabled={submitting || !spec}>
                <PlusCircle className="h-3.5 w-3.5" />
                {submitting ? "提交中…" : "提交任务"}
              </Button>
            </div>
          </>
        )}
      </Dialog.Content>
    </Dialog.Root>
  );
}

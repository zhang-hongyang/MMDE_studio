import { create } from "zustand";
import { api, type FrameRef } from "../lib/api";
import { decodeCtrl, type DecodedCtrl } from "../lib/decode";
import { loadModelTaxonomy, visibleModels } from "../lib/modelTaxonomy";

export type CtrlColorMode = "depth" | "weight" | "errpred" | "errgt";
export type CtrlColormap = "plasma" | "turbo";

export const COLOR_MODE_LABEL: Record<CtrlColorMode, string> = {
  depth: "深度",
  weight: "权重",
  errpred: "err=模型",
  errgt: "err=真值",
};

interface ControlsState {
  dataset: string;
  split: string;
  frames: FrameRef[];
  models: string[];
  controlSources: string[];

  /** position in frames[] */
  idx: number;
  /** enabled control sources (multi-select chips) */
  enabled: string[];
  /** selected model, "" = none (errpred needs a model) */
  model: string;
  colorMode: CtrlColorMode;
  colormap: CtrlColormap;
  /** point radius in image pixels */
  psize: number;
  playInterval: number;
  playing: boolean;

  loading: boolean;
  error: string | null;
  /** per-source decoded control points for the current frame */
  controls: Record<string, DecodedCtrl | null>;

  set: (p: Partial<ControlsState>) => void;
  init: (
    dataset: string,
    split: string,
    opts?: Partial<Pick<ControlsState, "idx" | "enabled" | "model" | "colorMode" | "colormap" | "psize">>,
  ) => Promise<void>;
  setIdx: (idx: number) => void;
  toggleSource: (src: string) => void;
  reload: () => Promise<void>;
}

let loadToken = 0;

/** finite |log(d / pred)| values of a decoded control set */
export function errPredValues(c: DecodedCtrl): number[] {
  const out: number[] = [];
  for (let i = 0; i < c.n; i++) {
    const p = c.pred[i];
    if (Number.isFinite(p) && p > 0 && c.d[i] > 0) out.push(Math.abs(Math.log(c.d[i] / p)));
  }
  return out;
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export const useControlsStore = create<ControlsState>((set, get) => ({
  dataset: "",
  split: "",
  frames: [],
  models: [],
  controlSources: [],

  idx: 0,
  enabled: [],
  model: "",
  colorMode: "depth",
  colormap: "turbo",
  psize: 3,
  playInterval: 400,
  playing: false,

  loading: false,
  error: null,
  controls: {},

  set: (p) => set(p),

  init: async (dataset, split, opts) => {
    const token = ++loadToken;
    set({
      dataset,
      split,
      frames: [],
      models: [],
      controlSources: [],
      idx: 0,
      enabled: [],
      controls: {},
      error: null,
      playing: false,
    });
    try {
      const [frames, models, controlSources] = await Promise.all([
        api.frames(dataset, split),
        api.models(dataset, split).catch(() => [] as string[]),
        api.controlSources(dataset, split).catch(() => [] as string[]),
      ]);
      if (token !== loadToken) return;
      const enabled =
        opts?.enabled?.filter((s) => controlSources.includes(s)) ??
        [...controlSources];
      // fall back out of taxonomy-excluded models (e.g. old URLs ?model=dav2_rel)
      const taxonomy = await loadModelTaxonomy();
      const visible = taxonomy ? visibleModels(models, taxonomy) : models;
      const model =
        opts?.model && visible.includes(opts.model) ? opts.model : (visible[0] ?? "");
      const idx = Math.min(Math.max(0, opts?.idx ?? 0), Math.max(0, frames.length - 1));
      set({
        frames,
        models,
        controlSources,
        enabled,
        model,
        idx,
        colorMode: opts?.colorMode ?? "depth",
        colormap: opts?.colormap ?? "turbo",
        psize: opts?.psize ?? 3,
      });
      await get().reload();
    } catch (e) {
      if (token === loadToken) set({ error: String(e), loading: false });
    }
  },

  setIdx: (idx) => {
    const { frames } = get();
    const clamped = Math.max(0, Math.min(frames.length - 1, idx));
    if (clamped === get().idx) return;
    set({ idx: clamped });
    void get().reload();
  },

  toggleSource: (src) => {
    const { enabled } = get();
    set({
      enabled: enabled.includes(src)
        ? enabled.filter((s) => s !== src)
        : [...enabled, src],
    });
  },

  reload: async () => {
    const s = get();
    const { dataset, split, frames, controlSources } = s;
    if (!dataset || !split || frames.length === 0) return;
    const row = frames[s.idx]?.idx ?? s.idx;
    const token = ++loadToken;
    set({ loading: true, error: null });
    try {
      const entries = await Promise.all(
        controlSources.map(
          async (src) =>
            [
              src,
              await api
                .controls(dataset, split, row, src, s.model)
                .then(decodeCtrl)
                .catch(() => null),
            ] as const,
        ),
      );
      if (token !== loadToken) return;
      set({ controls: Object.fromEntries(entries), loading: false });
    } catch (e) {
      if (token === loadToken) set({ loading: false, error: String(e) });
    }
  },
}));

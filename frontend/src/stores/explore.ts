import { create } from "zustand";
import { api, type FrameMeta, type FrameRef } from "../lib/api";
import { decodeCtrl, decodeGT, decodePoints, type DecodedCtrl, type DecodedGT, type DecodedPoints } from "../lib/decode";
import { relPose, type Mat4 } from "../lib/pose";
import { loadModelTaxonomy, visibleModels } from "../lib/modelTaxonomy";

export type CompareMode = "model" | "frame" | "overlay";
export type Coloring = "rgb" | "diff" | "fdiff" | "fcolor";
export type CtrlColorMode = "errpred" | "errgt" | "depth" | "weight";
export type ColormapName = "plasma" | "turbo" | "jet";

export interface Rect {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

export interface FrameDiffStats {
  sampled: number;
  band: number;
  med: number;
  p95: number;
}

export interface ViewStats {
  nA: number;
  nB: number;
  frameStats: FrameDiffStats | null;
  /** bump whenever geometries rebuild (for status text) */
  version: number;
}

interface ExploreState {
  dataset: string;
  split: string;
  camera: string; // "" = all streams
  frames: FrameRef[];
  framesAll: FrameRef[]; // full unfiltered list (re-filter on camera switch)
  models: string[];
  controlSources: string[];

  idx: number; // position in frames[]
  mode: CompareMode;
  modelA: string;
  modelB: string;
  delta: number;
  onlyOverlap: boolean;

  coloring: Coloring;
  colormap: ColormapName;
  psize: number;
  /** point-cloud exposure multiplier (0.4–2.0) */
  brightness: number;
  showGT: boolean;
  showCtrl: boolean;
  ctrlSrc: string;
  ctrlColor: CtrlColorMode;
  syncView: boolean;

  rect: Rect | null;

  // loaded frame data
  loading: boolean;
  error: string | null;
  meta: FrameMeta | null;
  metaB: FrameMeta | null;
  cloudA: DecodedPoints | null;
  cloudB: DecodedPoints | null;
  gt: DecodedGT | null;
  ctrl: DecodedCtrl | null;
  Trel: Mat4 | null;
  idxB: number | null; // position in frames[]

  stats: ViewStats | null;
  /** bump to request camera refit (recentre, keep orbit direction) */
  fitVersion: number;
  /** bump to restore the original camera view (frame/model/mode/camera switch) */
  homeVersion: number;

  set: (p: Partial<ExploreState>) => void;
  init: (dataset: string, split: string, opts?: Partial<Pick<ExploreState, "idx" | "camera" | "modelA" | "modelB" | "mode">>) => Promise<void>;
  setIdx: (idx: number, fit?: boolean) => void;
  setCamera: (camera: string) => void;
  setRect: (rect: Rect | null) => void;
  reload: () => Promise<void>;
  setStats: (s: Omit<ViewStats, "version">) => void;
  requestFit: () => void;
  requestHomeView: () => void;
}

let loadToken = 0;

export const useExploreStore = create<ExploreState>((set, get) => ({
  dataset: "",
  split: "",
  camera: "",
  frames: [],
  framesAll: [],
  models: [],
  controlSources: [],

  idx: 0,
  mode: "model",
  modelA: "",
  modelB: "",
  delta: 1,
  onlyOverlap: false,

  coloring: "rgb",
  colormap: "turbo",
  psize: 0.06,
  brightness: 1,
  showGT: false,
  showCtrl: false,
  ctrlSrc: "",
  ctrlColor: "errpred",
  syncView: false,

  rect: null,

  loading: false,
  error: null,
  meta: null,
  metaB: null,
  cloudA: null,
  cloudB: null,
  gt: null,
  ctrl: null,
  Trel: null,
  idxB: null,

  stats: null,
  fitVersion: 0,
  homeVersion: 0,

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
      meta: null,
      metaB: null,
      cloudA: null,
      cloudB: null,
      gt: null,
      ctrl: null,
      Trel: null,
      idxB: null,
      rect: null,
      stats: null,
      error: null,
      ...(opts?.mode ? { mode: opts.mode } : {}),
    });
    try {
      const [frames, models, controlSources] = await Promise.all([
        api.frames(dataset, split),
        api.models(dataset, split),
        api.controlSources(dataset, split).catch(() => [] as string[]),
      ]);
      if (token !== loadToken) return;
      const cameras = [...new Set(frames.map((f) => f.camera).filter(Boolean))] as string[];
      const camera =
        opts?.camera && cameras.includes(opts.camera)
          ? opts.camera
          : cameras.length > 1
            ? cameras.includes("narrow_undist")
              ? "narrow_undist"
              : cameras[0]
            : "";
      const filtered = camera ? frames.filter((f) => f.camera === camera) : frames;
      // fall back out of taxonomy-excluded models (e.g. old URLs ?modelA=dav2_rel)
      const taxonomy = await loadModelTaxonomy();
      const visible = taxonomy ? visibleModels(models, taxonomy) : models;
      const modelA =
        opts?.modelA && visible.includes(opts.modelA)
          ? opts.modelA
          : visible[0] ?? "";
      const modelB =
        opts?.modelB && visible.includes(opts.modelB)
          ? opts.modelB
          : visible.length > 1
            ? visible[1]
            : visible[0] ?? "";
      const idx = Math.min(opts?.idx ?? 0, Math.max(0, filtered.length - 1));
      set({
        frames: filtered,
        framesAll: frames,
        models,
        controlSources,
        camera,
        modelA,
        modelB,
        ctrlSrc: controlSources[0] ?? "",
        idx,
      });
      await get().reload();
    } catch (e) {
      if (token === loadToken) set({ error: String(e), loading: false });
    }
  },

  setIdx: (idx, fit = true) => {
    const { frames } = get();
    const clamped = Math.max(0, Math.min(frames.length - 1, idx));
    if (clamped === get().idx) return;
    set({ idx: clamped, rect: null });
    void get().reload();
    if (fit) get().requestHomeView();
  },

  setCamera: (camera) => {
    const { framesAll, camera: cur } = get();
    if (camera === cur || framesAll.length === 0) return;
    const filtered = camera ? framesAll.filter((f) => f.camera === camera) : framesAll;
    set({ camera, frames: filtered, idx: 0, rect: null, idxB: null, Trel: null });
    void get().reload().then(() => get().requestHomeView());
  },

  setRect: (rect) => {
    set({ rect });
    get().requestFit();
  },

  reload: async () => {
    const s = get();
    const { dataset, split, frames } = s;
    if (!dataset || !split || frames.length === 0) return;
    const row = frames[s.idx]?.idx ?? s.idx;
    const token = ++loadToken;
    set({ loading: true, error: null });
    try {
      let meta: FrameMeta | null = null;
      let metaB: FrameMeta | null = null;
      let cloudA: DecodedPoints | null = null;
      let cloudB: DecodedPoints | null = null;
      let gt: DecodedGT | null = null;
      let ctrl: DecodedCtrl | null = null;
      let Trel: Mat4 | null = null;
      let idxB: number | null = null;

      if (s.mode === "frame") {
        const m = s.modelA;
        const iB = s.idx + s.delta;
        const fA = frames[s.idx];
        const fB = frames[iB];
        const okB = !!(fA && fB && fB.seq_id === fA.seq_id);
        const rowB = okB ? (fB.idx ?? iB) : -1;
        const [metaR, AR, gtR, ctrlR, metaBR, BR] = await Promise.all([
          api.meta(dataset, split, row),
          api.points(dataset, split, row, m).then(decodePoints).catch(() => null),
          api.gt(dataset, split, row).then(decodeGT).catch(() => null),
          s.showCtrl && s.ctrlSrc
            ? api.controls(dataset, split, row, s.ctrlSrc, m).then(decodeCtrl).catch(() => null)
            : Promise.resolve(null),
          okB ? api.meta(dataset, split, rowB).catch(() => null) : Promise.resolve(null),
          okB ? api.points(dataset, split, rowB, m).then(decodePoints).catch(() => null) : Promise.resolve(null),
        ]);
        if (token !== loadToken) return;
        meta = metaR;
        cloudA = AR;
        gt = gtR;
        ctrl = ctrlR;
        metaB = okB ? metaBR : null;
        idxB = okB && metaBR && BR ? iB : null;
        const TA = metaR?.T;
        const TB = metaBR?.T;
        cloudB = idxB !== null && TA != null && TB != null ? BR : null;
        if (cloudB && TA != null && TB != null) {
          Trel = relPose(TA, TB);
        } else {
          cloudB = null;
          idxB = null;
        }
      } else {
        const [metaR, AR, BR, gtR, ctrlR] = await Promise.all([
          api.meta(dataset, split, row),
          s.modelA
            ? api.points(dataset, split, row, s.modelA).then(decodePoints).catch(() => null)
            : Promise.resolve(null),
          s.modelB
            ? api.points(dataset, split, row, s.modelB).then(decodePoints).catch(() => null)
            : Promise.resolve(null),
          api.gt(dataset, split, row).then(decodeGT).catch(() => null),
          s.showCtrl && s.ctrlSrc
            ? api.controls(dataset, split, row, s.ctrlSrc, s.modelA).then(decodeCtrl).catch(() => null)
            : Promise.resolve(null),
        ]);
        if (token !== loadToken) return;
        meta = metaR;
        cloudA = AR;
        cloudB = BR;
        gt = gtR;
        ctrl = ctrlR;
      }
      set({ meta, metaB, cloudA, cloudB, gt, ctrl, Trel, idxB, loading: false });
    } catch (e) {
      if (token === loadToken) set({ loading: false, error: String(e) });
    }
  },

  setStats: (s) => set((prev) => ({ stats: { ...s, version: (prev.stats?.version ?? 0) + 1 } })),

  requestFit: () => set((prev) => ({ fitVersion: prev.fitVersion + 1 })),

  requestHomeView: () => set((prev) => ({ homeVersion: prev.homeVersion + 1 })),
}));

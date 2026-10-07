// Scene page store: index/overview/chunk streaming (ported from pc_viewer.html
// scene mode), segment selection, fly-through playback state.
//
// Non-reactive controller state (invalidation tokens, one-swap-at-a-time
// chunk loading, proximity-eval throttle) lives module-level below, like the
// old file's globals.

import { create } from "zustand";
import { fetchSceneCloud, fetchSceneIndex } from "./api";
import { normalizeSceneIndex, type SceneCloud, type SceneIndex } from "./decodeScene";
import { buildFlySegs, flyPlaylistLen, flyRuntime } from "./replay/engine";

export type FlyView = "car" | "chase";

/** soft point-pixel cap for the point shader (软钳制实现细节，固定默认值) */
export const PT_MAX = 10;
/** default point budget for streamed fine chunks */
export const DEFAULT_BUDGET = 3e6;

export interface FlyState {
  on: boolean;
  playing: boolean;
  speed: number;
  view: FlyView;
  /** floor(seconds) — drives the scrub slider */
  p: number;
  /** total playlist length in seconds */
  plen: number;
  /** nearest global frame (thumbnail + label) */
  frame: number;
  /** interpolated world xy of the replay camera (minimap ring) */
  x: number;
  y: number;
  /** bumps every engine tick that moved the pose (minimap redraw) */
  tick: number;
}

interface SceneState {
  dataset: string;
  split: string;
  /** scene models available for the current split */
  models: string[];
  model: string;
  /** point budget for streamed fine chunks */
  budget: number;
  /** point-size multiplier on the voxel-derived base size (1 = default) */
  psize: number;
  /** point-cloud brightness multiplier (shader uniform, 1 = default) */
  brightness: number;

  index: SceneIndex | null;
  /** per-group selection (which sequences are loaded / replayed) */
  sel: boolean[];
  /** selected groups' overview clouds, keyed by group index */
  overviews: Record<number, SceneCloud>;
  /** streamed fine chunks, key "s{g}:{i}"; null = reserved/loading */
  chunks: Record<string, SceneCloud | null>;

  loading: boolean;
  error: string | null;
  status: string;
  /** bump to request a camera refit */
  fitVersion: number;
  /** orbit target mirror (minimap dot + chunk proximity evals) */
  target: { x: number; y: number; z: number };
  /** bumps when the orbit target moves */
  targetVersion: number;
  dragging: boolean;

  fly: FlyState;

  init: (
    dataset: string,
    split: string,
    models: string[],
    opts?: { model?: string; budget?: number },
  ) => Promise<void>;
  setModel: (model: string) => Promise<void>;
  setBudget: (budget: number) => void;
  setPsize: (psize: number) => void;
  setBrightness: (brightness: number) => void;
  toggleGroup: (k: number) => Promise<void>;
  setAllGroups: (on: boolean) => Promise<void>;
  refreshChunks: () => Promise<void>;
  requestFit: () => void;
  setTarget: (x: number, y: number, z: number) => void;
  setDragging: (dragging: boolean) => void;

  enterFly: () => void;
  exitFly: () => void;
  setFlyPlaying: (playing: boolean) => void;
  setFlySpeed: (speed: number) => void;
  setFlyView: (view: FlyView) => void;
  setFlyP: (p: number) => void;
  /** engine tick: publish interpolated pose for the minimap ring */
  flyMoved: (x: number, y: number, frame: number, p: number) => void;
}

// ---- module-level controller state (non-reactive) ----
let loadTok = 0;
let chunkBusy = false;
let chunkPending = false;
let evalAt = 0;
let evalTarget = { x: 1e9, y: 1e9, z: 1e9 };

/** proximity-eval gate from pc_viewer.html's render loop: re-pick chunks only
 * when the target moved >50 m and 800 ms have passed. Returns true if a
 * refresh is due; on acceptance it stamps the current target/time. */
function evalDue(target: { x: number; y: number; z: number }): boolean {
  const now = performance.now();
  const d2 =
    (target.x - evalTarget.x) ** 2 + (target.y - evalTarget.y) ** 2 + (target.z - evalTarget.z) ** 2;
  if (now - evalAt > 800 && d2 > 2500) {
    evalAt = now;
    evalTarget = { ...target };
    return true;
  }
  return false;
}

function sceneStatus(s: Pick<SceneState, "index" | "sel" | "chunks">): string {
  const idx = s.index;
  if (!idx) return "";
  let n = 0;
  let k = 0;
  let nChunks = 0;
  idx.groups.forEach((g, gi) => {
    if (!s.sel[gi]) return;
    n += g.overview_n;
    nChunks += g.chunks.length;
  });
  for (const rec of Object.values(s.chunks)) if (rec) n += rec.n;
  for (const rec of Object.values(s.chunks)) if (rec) k++;
  const tag =
    idx.groups.length > 1 ? `选中 ${s.sel.filter(Boolean).length}/${idx.groups.length} 段 · ` : "";
  return `场景 ${idx.model}　${tag}${k}/${nChunks} 块 · ${n.toLocaleString()} 点`;
}

export const useSceneStore = create<SceneState>((set, get) => {
  /** drop all clouds and load the selected groups' overviews + nearby chunks */
  async function applySelection(): Promise<void> {
    const tok = ++loadTok;
    const { index, sel } = get();
    if (!index) return;
    set({ overviews: {}, chunks: {}, status: "加载场景粗览…" });
    for (let k = 0; k < index.groups.length; k++) {
      if (!sel[k]) continue;
      try {
        const ov = await fetchSceneCloud(
          get().dataset,
          get().split,
          index.model,
          index.groups[k].overview,
          index.groups[k],
          index.created,
        );
        if (tok !== loadTok) return;
        set((s) => ({ overviews: { ...s.overviews, [k]: ov } }));
      } catch (e) {
        if (tok === loadTok) set({ error: `S${k + 1} 粗览加载失败：${e}` });
        return;
      }
    }
    if (tok !== loadTok) return;
    evalAt = 0;
    evalTarget = { ...get().target };
    set({ status: `场景 ${index.model}　按预算加载就近细节块…` });
    await refreshChunks(true);
    if (tok === loadTok) set({ status: sceneStatus(get()) });
  }

  async function loadIndex(): Promise<void> {
    const tok = ++loadTok;
    const { dataset, split, model } = get();
    if (!model) {
      set({ index: null, sel: [], overviews: {}, chunks: {}, loading: false, status: "" });
      return;
    }
    set({ loading: true, error: null, status: "加载场景索引…", index: null });
    let idx: SceneIndex;
    try {
      idx = normalizeSceneIndex(await fetchSceneIndex(dataset, split, model));
    } catch (e) {
      if (tok === loadTok)
        set({ loading: false, error: `场景索引加载失败：${e}`, status: "" });
      return;
    }
    if (tok !== loadTok) return;
    // default selection: first sequence only (pc_viewer.html behavior)
    const sel = idx.groups.map((_, k) => k === 0);
    flyRuntime.p = 0;
    set({
      index: idx,
      sel,
      loading: false,
      status: "",
      fitVersion: get().fitVersion + 1, // fit camera to the new bbox
    });
    syncFlyPlan();
    await applySelection();
  }

  /** rebuild the fly playlist after index/selection changes; rewind if playing */
  function syncFlyPlan(): void {
    const { index, sel, fly } = get();
    const plen = index ? flyPlaylistLen(buildFlySegs(index, sel)) : 0;
    const p = flyRuntime.p >= plen ? Math.max(0, plen - 1e-6) : flyRuntime.p;
    flyRuntime.p = p;
    set({
      fly: {
        ...fly,
        plen,
        p: Math.floor(p),
        on: fly.on && plen > 0,
        playing: fly.on && plen > 0 ? fly.playing : false,
        frame: index && plen > 0 ? index.groups[sel.findIndex(Boolean)]?.start ?? 0 : 0,
      },
    });
  }

  /** Budgeted proximity streaming: selected groups' overviews stay always; fine
   * chunks load near the view target up to the detail budget and are dropped
   * when the target moves away — keeps GPU vertex count bounded.
   * Callers gate non-forced refreshes with evalDue(); explicit changes
   * (selection/detail/budget/drag-end) pass force=true. */
  async function refreshChunks(force = false): Promise<void> {
    const { index, sel, chunks, budget, target, dragging } = get();
    if (!index || get().loading) return;
    if (!force && (dragging || !evalDue(target))) return;
    if (chunkBusy) {
      chunkPending = true;
      return;
    }
    chunkBusy = true;
    const tok = loadTok;
    const t = target;
    const pool = new Map<string, { file: string; n: number; g: number; center: [number, number, number] }>();
    index.groups.forEach((g, gi) => {
      if (!sel[gi]) return;
      g.chunks.forEach((c, i) => pool.set(`s${gi}:${i}`, { file: c.file, n: c.n, g: gi, center: c.center }));
    });
    const lim = budget;
    const want = new Set<string>();
    let sum = 0;
    const byDist = [...pool.keys()]
      .map((k) => {
        const c = pool.get(k)!;
        return [k, (c.center[0] - t.x) ** 2 + (c.center[1] - t.y) ** 2] as const;
      })
      .sort((a, b) => a[1] - b[1]);
    for (const [k] of byDist) {
      if (sum >= lim) break;
      want.add(k);
      sum += pool.get(k)!.n;
    }
    const next: Record<string, SceneCloud | null> = {};
    for (const k of Object.keys(chunks)) {
      if (want.has(k)) next[k] = chunks[k]; // keep loaded / reserved
    }
    for (const k of want) {
      if (!(k in next)) next[k] = null; // reserve: no duplicate loads
    }
    set({ chunks: next });
    const missing = [...want].filter((k) => next[k] === null);
    let cursor = 0;
    const worker = async (): Promise<void> => {
      while (cursor < missing.length) {
        if (tok !== loadTok) return;
        const k = missing[cursor++];
        const rec = pool.get(k)!;
        const g = index.groups[rec.g];
        try {
          const cloud = await fetchSceneCloud(
            get().dataset,
            get().split,
            index.model,
            rec.file,
            g,
            index.created,
          );
          if (tok !== loadTok) return;
          // a newer refresh revoked this reservation -> drop; adding it would
          // leave an orphan cloud nobody disposes
          if (!(k in get().chunks)) return;
          set((s) => ({
            chunks: { ...s.chunks, [k]: cloud },
            status: sceneStatus({ ...s, chunks: { ...s.chunks, [k]: cloud } }),
          }));
        } catch {
          if (tok !== loadTok) return;
          set((s) => {
            const rest = { ...s.chunks };
            delete rest[k];
            return { chunks: rest };
          });
          return;
        }
      }
    };
    await Promise.all([worker(), worker()]);
    chunkBusy = false;
    if (chunkPending) {
      // re-run reads the *current* index/selection; don't gate on the old token
      chunkPending = false;
      void refreshChunks(true);
    } else {
      chunkPending = false;
      if (tok === loadTok) set({ status: sceneStatus(get()) });
    }
  }

  return {
    dataset: "",
    split: "",
    models: [],
    model: "",
    budget: DEFAULT_BUDGET,
    psize: 1,
    brightness: 1,
    index: null,
    sel: [],
    overviews: {},
    chunks: {},
    loading: false,
    error: null,
    status: "",
    fitVersion: 0,
    target: { x: 0, y: 0, z: 0 },
    targetVersion: 0,
    dragging: false,
    fly: {
      on: false,
      playing: false,
      speed: 1,
      view: "car",
      p: 0,
      plen: 0,
      frame: 0,
      x: 0,
      y: 0,
      tick: 0,
    },

    async init(dataset, split, models, opts) {
      const model =
        opts?.model && models.includes(opts.model) ? opts.model : (models[0] ?? "");
      set({
        dataset,
        split,
        models,
        model,
        budget: opts?.budget ?? get().budget,
        error: null,
      });
      await loadIndex();
    },

    async setModel(model) {
      if (model === get().model) return;
      set({ model });
      await loadIndex();
    },

    setBudget(budget) {
      set({ budget });
      void refreshChunks(true);
    },
    setPsize(psize) {
      set({ psize });
    },
    setBrightness(brightness) {
      set({ brightness });
    },

    async toggleGroup(k) {
      const sel = [...get().sel];
      sel[k] = !sel[k];
      flyRuntime.p = 0; // timeline rebuilt: rewind playback
      set({ sel });
      syncFlyPlan();
      await applySelection();
    },
    async setAllGroups(on) {
      const sel = get().sel.map(() => on);
      flyRuntime.p = 0;
      set({ sel });
      syncFlyPlan();
      await applySelection();
    },

    refreshChunks: () => refreshChunks(true),

    requestFit() {
      set((s) => ({ fitVersion: s.fitVersion + 1 }));
    },
    setTarget(x, y, z) {
      set((s) => ({ target: { x, y, z }, targetVersion: s.targetVersion + 1 }));
    },
    setDragging(dragging) {
      set({ dragging });
    },

    enterFly() {
      const { index, fly } = get();
      if (!index || !index.cam_quat?.length || !fly.plen) return;
      flyRuntime.p = 0;
      const start = buildFlySegs(index, get().sel)[0]?.start ?? 0;
      set({ fly: { ...fly, on: true, playing: true, p: 0, frame: start } });
    },
    exitFly() {
      const { fly } = get();
      if (!fly.on) return;
      set({ fly: { ...fly, on: false, playing: false } });
    },
    setFlyPlaying(playing) {
      set((s) => ({ fly: { ...s.fly, playing } }));
    },
    setFlySpeed(speed) {
      set((s) => ({ fly: { ...s.fly, speed } }));
    },
    setFlyView(view) {
      set((s) => ({ fly: { ...s.fly, view } }));
    },
    setFlyP(p) {
      flyRuntime.p = p;
      set((s) => ({ fly: { ...s.fly, p } }));
    },
    flyMoved(x, y, frame, p) {
      set((s) => ({
        fly: { ...s.fly, x, y, frame, p: Math.floor(p), tick: s.fly.tick + 1 },
      }));
    },
  };
});

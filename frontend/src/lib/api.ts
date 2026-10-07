// Backend API client. All binary endpoints return ArrayBuffer;
// decoding lives in src/lib/decode.ts.

export interface SplitInfo {
  name: string;
  test_type: "test_single" | "test_sequence";
  n_frames: number;
  evaluation_available: boolean;
  cameras: string[];
  control_sources: string[];
  models: string[];
  model_availability?: Record<string, {
    available: boolean;
    mode?: string;
    reason?: string;
    representation?: string;
  }>;
  scene_models: string[];
}

export interface DatasetInfo {
  name: string;
  platform: string;
  title: string;
  splits: SplitInfo[];
}

export interface Registry {
  datasets: DatasetInfo[];
}

export interface FrameRef {
  idx: number;
  seq_id: string;
  frame_id: string;
  camera?: string;
}

export interface FrameMeta {
  K: number[][];
  w: number;
  h: number;
  seq_id: string;
  frame_id: string;
  /** T_world_camera (camera -> world), 4x4 nested */
  T?: number[][] | null;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

async function getArrayBuffer(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.arrayBuffer();
}

export function calibQS(calib: string): string {
  return calib ? `?calib=${encodeURIComponent(calib)}` : "";
}

export const api = {
  registry: () => getJSON<Registry>("/api/registry"),

  frames: (dataset: string, split: string, camera?: string) =>
    getJSON<FrameRef[]>(
      `/api/frames?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}` +
        (camera ? `&camera=${encodeURIComponent(camera)}` : ""),
    ),

  models: (ds: string, split: string) => getJSON<string[]>(`/api/models/${ds}/${split}`),

  meta: (ds: string, split: string, idx: number) =>
    getJSON<FrameMeta>(`/api/meta/${ds}/${split}/${idx}`),

  points: (ds: string, split: string, idx: number, model: string, calib = "") =>
    getArrayBuffer(`/api/points/${ds}/${split}/${idx}/${model}${calibQS(calib)}`),

  gt: (ds: string, split: string, idx: number, calib = "") =>
    getArrayBuffer(`/api/gt/${ds}/${split}/${idx}${calibQS(calib)}`),

  controlSources: (ds: string, split: string) =>
    getJSON<string[]>(`/api/control_sources?dataset=${encodeURIComponent(ds)}&split=${encodeURIComponent(split)}`),

  controls: (ds: string, split: string, idx: number, source: string, model: string, calib = "") =>
    getArrayBuffer(
      `/api/controls/${ds}/${split}/${idx}/${source}?model=${encodeURIComponent(model)}` +
        (calib ? `&calib=${encodeURIComponent(calib)}` : ""),
    ),

  rgbURL: (ds: string, split: string, idx: number, w?: number) =>
    `/api/rgb/${ds}/${split}/${idx}${w ? `?w=${w}` : ""}`,
};

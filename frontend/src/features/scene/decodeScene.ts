// Fused-scene index/blob decoding (ported from pc_viewer.html scene mode).
//
// Blob layout (little-endian, verified against live data):
//   u32 n | u8 rgb[3n] | zero pad to 2-byte alignment ((3n) & 1)
//         | u16 qx[n] | u16 qy[n] | u16 qz[n]
// World coordinate = origin + q * quant_step (quant_step is a scalar).
// NOTE: the task brief said "pad to 4 bytes"; the actual writer pads to even
// ((3n) & 1) — a 114026-byte overview_s0.bin with n=12669 decodes exactly with
// the even-pad formula and reproduces the group's bbox.

export interface BBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface SceneChunkInfo {
  file: string;
  n: number;
  center: [number, number, number];
}

export interface SceneGroup {
  seq_id: string;
  start: number;
  end: number;
  n_points: number;
  depth_scale?: number;
  bbox: BBox;
  origin: [number, number, number];
  quant_step: number;
  /** per-sequence overview blob name, e.g. "overview_s0.bin" */
  overview: string;
  overview_n: number;
  chunks: SceneChunkInfo[];
}

export interface SceneSeq {
  seq_id: string;
  start: number;
  end: number;
}

export interface SceneIndex {
  dataset?: string;
  split?: string;
  model: string;
  created?: string;
  format: number;
  n_frames: number;
  stride?: number;
  min_depth?: number;
  max_depth?: number;
  voxel: number;
  depth_scale?: number;
  n_points: number;
  bbox: BBox;
  groups: SceneGroup[];
  /** world-frame camera positions, [x,y,z] (z up) */
  cam_pos: [number, number, number][];
  /** world-frame camera orientations, OpenCV cam->world [w,x,y,z] */
  cam_quat: [number, number, number, number][];
  /** per-frame capture time in seconds (relative) */
  cam_t: number[];
  seqs: SceneSeq[];
}

/** decoded point cloud payload for one blob */
export interface SceneCloud {
  n: number;
  pos: Float32Array; // 3n, world coordinates
  col: Uint8Array; // 3n, rgb
}

interface Quantized {
  origin: [number, number, number];
  quant_step: number;
}

export function decodeSceneCloud(ab: ArrayBuffer, g: Quantized): SceneCloud {
  const n = new DataView(ab).getUint32(0, true);
  const col = new Uint8Array(ab, 4, 3 * n);
  const off = 4 + 3 * n + ((3 * n) & 1); // mirror the writer's alignment pad
  const qx = new Uint16Array(ab, off, n);
  const qy = new Uint16Array(ab, off + 2 * n, n);
  const qz = new Uint16Array(ab, off + 4 * n, n);
  const o = g.origin;
  const s = g.quant_step;
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = o[0] + qx[i] * s;
    pos[3 * i + 1] = o[1] + qy[i] * s;
    pos[3 * i + 2] = o[2] + qz[i] * s;
  }
  return { n, pos, col };
}

/** Pre-per-sequence ("format 1") indexes fused every drive into one cloud;
 * expose them as a single unselectable group so old scenes still open. */
export function normalizeSceneIndex(idx: SceneIndex): SceneIndex {
  if (idx.format === 2) return idx;
  const g = idx as unknown as SceneIndex & SceneGroup;
  return {
    ...idx,
    format: 2,
    groups: [
      {
        seq_id: "全部(旧融合)",
        start: 0,
        end: idx.n_frames,
        n_points: idx.n_points,
        bbox: idx.bbox,
        origin: g.origin,
        quant_step: g.quant_step,
        overview: "overview.bin",
        overview_n: g.overview_n,
        chunks: g.chunks,
      },
    ],
    seqs: idx.seqs ?? [],
    cam_pos: idx.cam_pos ?? [],
    cam_quat: idx.cam_quat ?? [],
    cam_t: idx.cam_t ?? [],
  };
}

// Point-cloud geometry builders — ports of pc_viewer.html buildCloud / buildFrameB /
// buildGT / buildCtrl. Image coordinates: v axis points down, so 3D y is image-down
// (camera.up = (0,-1,0) keeps the front view upright, matching the old viewer).
import * as THREE from "three";
import type { FrameMeta } from "../../lib/api";
import type { DecodedCtrl, DecodedGT, DecodedPoints } from "../../lib/decode";
import { colormap, rdbu, type ColormapName, type RGB } from "../../lib/colormap";
import type { Mat4 } from "../../lib/pose";
import type { CtrlColorMode, Rect } from "../../stores/explore";

/** soft point budget per viewport; stride doubles until under budget */
export const POINT_BUDGET = 600_000;

export interface BuiltCloud {
  geometry: THREE.BufferGeometry;
  count: number;
  box: THREE.Box3 | null;
}

const Z_MIN = 0.3;
const Z_MAX = 120;

const GRAY_MISS: RGB = [0.45, 0.45, 0.47];
const CTRL_MISS: RGB = [0.35, 0.35, 0.38];
const CTRL_ERR_NORM = 0.5;

const tmpColor: RGB = [0, 0, 0];

function validZ(z: number): boolean {
  return z > Z_MIN && z < Z_MAX && Number.isFinite(z);
}

function makeCloud(pos: Float32Array, col: Float32Array, count: number): BuiltCloud {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(0, count * 3), 3));
  g.setAttribute("color", new THREE.BufferAttribute(col.subarray(0, count * 3), 3));
  g.computeBoundingBox();
  return { geometry: g, count, box: g.boundingBox };
}

/** extra grid subsample factor so total points stay under budget */
export function budgetStride(n: number, budget = POINT_BUDGET): number {
  let extra = 1;
  while (n / (extra * extra) > budget) extra *= 2;
  return extra;
}

export interface CloudOptions {
  meta: FrameMeta;
  rect?: Rect | null;
  /** "rgb" | "diff" | "fcolor"(fixed color below) */
  colorMode: "rgb" | "diff" | "fixed";
  fixedColor?: RGB;
  /** dim rgb (used for frame-A cloud in fdiff overlay) */
  dim?: boolean;
  other?: DecodedPoints | null;
  colormap?: ColormapName;
  budget?: number;
}

export function buildCloud(d: DecodedPoints, opts: CloudOptions): BuiltCloud {
  const { meta } = opts;
  const fx = meta.K[0][0];
  const fy = meta.K[1][1];
  const cx = meta.K[0][2];
  const cy = meta.K[1][2];
  const depth = d.dRaw;
  const extra = budgetStride(d.n, opts.budget);
  const rect = opts.rect ?? null;

  // collect selected grid cells
  const idxs: number[] = [];
  for (let r = 0; r < d.hp; r += extra) {
    const v = r * d.step;
    if (rect && (v < rect.v0 || v >= rect.v1)) continue;
    for (let c = 0; c < d.wp; c += extra) {
      if (rect && (c * d.step < rect.u0 || c * d.step >= rect.u1)) continue;
      idxs.push(r * d.wp + c);
    }
  }

  // |A-B| diff normalization from p95
  let diff: Float32Array | null = null;
  let diffNorm = 1;
  if (opts.colorMode === "diff" && opts.other) {
    const o = opts.other;
    diff = new Float32Array(idxs.length);
    for (let k = 0; k < idxs.length; k++) diff[k] = Math.abs(d.dRaw[idxs[k]] - o.dRaw[idxs[k]]);
    const srt = Array.from(diff).sort((a, b) => a - b);
    diffNorm = Math.max(1e-6, srt[Math.floor(srt.length * 0.95)] || 1);
  }

  const pos = new Float32Array(idxs.length * 3);
  const col = new Float32Array(idxs.length * 3);
  let j = 0;
  const cm = opts.colormap ?? "turbo";
  for (let k = 0; k < idxs.length; k++) {
    const gi = idxs[k];
    const z = depth[gi];
    if (!validZ(z)) continue;
    const c = gi % d.wp;
    const r = (gi / d.wp) | 0;
    pos[3 * j] = (c * d.step - cx) * z / fx;
    pos[3 * j + 1] = (r * d.step - cy) * z / fy;
    pos[3 * j + 2] = z;
    if (opts.colorMode === "fixed" && opts.fixedColor) {
      col[3 * j] = opts.fixedColor[0];
      col[3 * j + 1] = opts.fixedColor[1];
      col[3 * j + 2] = opts.fixedColor[2];
    } else if (diff) {
      const cc = colormap(cm, diff[k] / diffNorm, tmpColor);
      col[3 * j] = cc[0];
      col[3 * j + 1] = cc[1];
      col[3 * j + 2] = cc[2];
    } else {
      const s = opts.dim ? 0.35 : 1.0;
      col[3 * j] = (d.rgb[3 * gi] / 255) * s;
      col[3 * j + 1] = (d.rgb[3 * gi + 1] / 255) * s;
      col[3 * j + 2] = (d.rgb[3 * gi + 2] / 255) * s;
    }
    j++;
  }
  return makeCloud(pos, col, j);
}

export interface FrameDiffStats {
  sampled: number;
  band: number;
  med: number;
  p95: number;
}

export interface FrameBOptions {
  metaA: FrameMeta;
  metaB: FrameMeta;
  Trel: Mat4;
  cloudA: DecodedPoints | null;
  rect?: Rect | null;
  onlyOverlap: boolean;
  /** "fdiff" (diverging Δd) | "fixed" (B fixed color) | "rgb" */
  colorMode: "fdiff" | "fixed" | "rgb";
  fixedColor?: RGB;
  colormap?: ColormapName;
  budget?: number;
}

/** Bilinear sample of frame A's dense depth grid at image pixel (u, v). */
function sampleGrid(depth: Float32Array, d: DecodedPoints, u: number, v: number): number {
  const gu = u / d.step;
  const gv = v / d.step;
  if (!(gu >= 0 && gv >= 0 && gu <= d.wp - 1 && gv <= d.hp - 1)) return NaN;
  const c0 = Math.floor(gu);
  const r0 = Math.floor(gv);
  const c1 = Math.min(c0 + 1, d.wp - 1);
  const r1 = Math.min(r0 + 1, d.hp - 1);
  const fu = gu - c0;
  const fv = gv - r0;
  const w = d.wp;
  const d00 = depth[r0 * w + c0];
  const d01 = depth[r0 * w + c1];
  const d10 = depth[r1 * w + c0];
  const d11 = depth[r1 * w + c1];
  if (!validZ(d00) || !validZ(d01) || !validZ(d10) || !validZ(d11)) return NaN;
  return (1 - fu) * (1 - fv) * d00 + fu * (1 - fv) * d01 + (1 - fu) * fv * d10 + fu * fv * d11;
}

const overlapBand = (dd: number, za: number) => Math.abs(dd) <= Math.max(0.15, 0.05 * za);

/** Build frame B's cloud transformed into frame A's camera system. */
export function buildFrameB(d: DecodedPoints, opts: FrameBOptions): { cloud: BuiltCloud; stats: FrameDiffStats | null } {
  const { metaA, metaB, Trel } = opts;
  const Ka = metaA.K;
  const Kb = metaB.K;
  const fxb = Kb[0][0];
  const fyb = Kb[1][1];
  const cxb = Kb[0][2];
  const cyb = Kb[1][2];
  const fxa = Ka[0][0];
  const fya = Ka[1][1];
  const cxa = Ka[0][2];
  const cya = Ka[1][2];
  const WA = metaA.w;
  const HA = metaA.h;
  const depthB = d.dRaw;
  const depthA = opts.cloudA ? opts.cloudA.dRaw : null;
  const rect = opts.rect ?? null;
  const n = d.n;
  const extra = budgetStride(n, opts.budget);

  const dd = new Float32Array(n);
  const zaS = new Float32Array(n);
  const vis = new Uint8Array(n); // 0 skip | 1 no-overlap | 2 sampled against A
  const pt = new Float32Array(n * 3);
  let cntSampled = 0;

  for (let r = 0; r < d.hp; r++) {
    for (let c = 0; c < d.wp; c++) {
      const gi = r * d.wp + c;
      const z = depthB[gi];
      if (!validZ(z)) continue;
      const xb = (c * d.step - cxb) * z / fxb;
      const yb = (r * d.step - cyb) * z / fyb;
      const xa = Trel[0] * xb + Trel[1] * yb + Trel[2] * z + Trel[3];
      const ya = Trel[4] * xb + Trel[5] * yb + Trel[6] * z + Trel[7];
      const za = Trel[8] * xb + Trel[9] * yb + Trel[10] * z + Trel[11];
      if (!(za > 0.3 && za < 300 && Number.isFinite(za))) continue;
      const u = (fxa * xa) / za + cxa;
      const v = (fya * ya) / za + cya;
      if (u < 0 || v < 0 || u >= WA || v >= HA) {
        vis[gi] = 1;
        continue;
      }
      if (rect && (u < rect.u0 || u >= rect.u1 || v < rect.v0 || v >= rect.v1)) continue;
      pt[3 * gi] = xa;
      pt[3 * gi + 1] = ya;
      pt[3 * gi + 2] = za;
      const zs = depthA && opts.cloudA ? sampleGrid(depthA, opts.cloudA, u, v) : NaN;
      if (Number.isNaN(zs)) {
        vis[gi] = 1;
        continue;
      }
      vis[gi] = 2;
      dd[gi] = za - zs;
      zaS[gi] = zs;
      cntSampled++;
    }
  }

  let stats: FrameDiffStats | null = null;
  let norm = 1;
  if (cntSampled >= 10) {
    const absdd = new Float32Array(cntSampled);
    let p = 0;
    let band = 0;
    for (let gi = 0; gi < n; gi++) {
      if (vis[gi] !== 2) continue;
      absdd[p++] = Math.abs(dd[gi]);
      if (overlapBand(dd[gi], zaS[gi])) band++;
    }
    absdd.sort();
    stats = {
      sampled: cntSampled,
      band,
      med: absdd[(cntSampled / 2) | 0],
      p95: absdd[Math.min(cntSampled - 1, (cntSampled * 0.95) | 0)],
    };
    norm = Math.max(0.1, stats.p95);
  }

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  let j = 0;
  for (let r = 0; r < d.hp; r += extra) {
    for (let c = 0; c < d.wp; c += extra) {
      const gi = r * d.wp + c;
      if (vis[gi] === 0) continue;
      if (opts.onlyOverlap && !(vis[gi] === 2 && overlapBand(dd[gi], zaS[gi]))) continue;
      pos[3 * j] = pt[3 * gi];
      pos[3 * j + 1] = pt[3 * gi + 1];
      pos[3 * j + 2] = pt[3 * gi + 2];
      let cc: RGB | null = null;
      if (opts.colorMode === "fdiff") {
        cc =
          vis[gi] === 2
            ? rdbu(0.5 + 0.5 * Math.max(-1, Math.min(1, dd[gi] / norm)), tmpColor)
            : GRAY_MISS;
      } else if (opts.colorMode === "fixed") {
        cc = opts.fixedColor ?? null;
      }
      if (cc) {
        col[3 * j] = cc[0];
        col[3 * j + 1] = cc[1];
        col[3 * j + 2] = cc[2];
      } else {
        col[3 * j] = d.rgb[3 * gi] / 255;
        col[3 * j + 1] = d.rgb[3 * gi + 1] / 255;
        col[3 * j + 2] = d.rgb[3 * gi + 2] / 255;
      }
      j++;
    }
  }
  return { cloud: makeCloud(pos, col, j), stats };
}

export function buildGT(meta: FrameMeta, gt: DecodedGT, rect?: Rect | null): BuiltCloud {
  const fx = meta.K[0][0];
  const fy = meta.K[1][1];
  const cx = meta.K[0][2];
  const cy = meta.K[1][2];
  const sel: number[] = [];
  for (let i = 0; i < gt.n; i++) {
    if (!rect || (gt.u[i] >= rect.u0 && gt.u[i] < rect.u1 && gt.v[i] >= rect.v0 && gt.v[i] < rect.v1))
      sel.push(i);
  }
  const pos = new Float32Array(sel.length * 3);
  const col = new Float32Array(sel.length * 3);
  sel.forEach((gi, k) => {
    const z = gt.d[gi];
    pos[3 * k] = ((gt.u[gi] - cx) * z) / fx;
    pos[3 * k + 1] = ((gt.v[gi] - cy) * z) / fy;
    pos[3 * k + 2] = z;
    col[3 * k] = col[3 * k + 1] = col[3 * k + 2] = 1;
  });
  return makeCloud(pos, col, sel.length);
}

export function buildCtrl(
  meta: FrameMeta,
  c: DecodedCtrl,
  mode: CtrlColorMode,
  rect?: Rect | null,
): BuiltCloud {
  const fx = meta.K[0][0];
  const fy = meta.K[1][1];
  const cx = meta.K[0][2];
  const cy = meta.K[1][2];
  const sel: number[] = [];
  for (let i = 0; i < c.n; i++) {
    if (!rect || (c.u[i] >= rect.u0 && c.u[i] < rect.u1 && c.v[i] >= rect.v0 && c.v[i] < rect.v1))
      sel.push(i);
  }
  const pos = new Float32Array(sel.length * 3);
  const col = new Float32Array(sel.length * 3);
  sel.forEach((gi, k) => {
    const z = c.d[gi];
    pos[3 * k] = ((c.u[gi] - cx) * z) / fx;
    pos[3 * k + 1] = ((c.v[gi] - cy) * z) / fy;
    pos[3 * k + 2] = z;
    let cc: RGB;
    if (mode === "depth") {
      cc = colormap("plasma", Math.min(1, Math.max(0, (z - 2) / 78)), tmpColor);
    } else if (mode === "weight") {
      cc = colormap("plasma", Math.min(1, Math.max(0, c.w[gi])), tmpColor);
    } else {
      const err =
        mode === "errgt"
          ? c.errgt[gi]
          : Number.isFinite(c.pred[gi]) && c.pred[gi] > 0
            ? Math.abs(Math.log(z / c.pred[gi]))
            : NaN;
      cc = Number.isFinite(err) ? rdbu(Math.min(1, err / CTRL_ERR_NORM), tmpColor) : CTRL_MISS;
    }
    col[3 * k] = cc[0];
    col[3 * k + 1] = cc[1];
    col[3 * k + 2] = cc[2];
  });
  return makeCloud(pos, col, sel.length);
}

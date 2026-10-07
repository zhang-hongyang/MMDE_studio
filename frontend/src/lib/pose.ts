// Rigid pose math: T_world_camera (camera -> world) helpers.
// Ported from pc_viewer.html.

export type Mat4 = Float64Array; // row-major 16

export function mat4FromNested(a: number[][]): Mat4 {
  const m = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) m[i * 4 + j] = a[i][j];
  return m;
}

export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const c = new Float64Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
      c[i * 4 + j] = s;
    }
  return c;
}

/** inverse of rigid [R t; 0 1] (row-major): [R^T, -R^T t] */
export function invRigid(m: Mat4): Mat4 {
  const Rt = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
  const t = [m[3], m[7], m[11]];
  const nt = [
    -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]),
    -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]),
    -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]),
  ];
  const out = new Float64Array(16);
  out[0] = Rt[0]; out[1] = Rt[1]; out[2] = Rt[2]; out[3] = nt[0];
  out[4] = Rt[3]; out[5] = Rt[4]; out[6] = Rt[5]; out[7] = nt[1];
  out[8] = Rt[6]; out[9] = Rt[7]; out[10] = Rt[8]; out[11] = nt[2];
  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

/** B-camera -> A-camera transform */
export function relPose(TwcA: number[][], TwcB: number[][]): Mat4 {
  return mat4Mul(invRigid(mat4FromNested(TwcA)), mat4FromNested(TwcB));
}

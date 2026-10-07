// Binary payload decoders (little-endian, see backend API contract).

export interface DecodedPoints {
  hp: number;
  wp: number;
  step: number;
  n: number;
  /** raw model depth, hp*wp raster order */
  dRaw: Float32Array;
  /** GT-scale-aligned depth */
  dAl: Float32Array;
  /** rgb, 3*hp*wp */
  rgb: Uint8Array;
}

export interface DecodedGT {
  n: number;
  u: Float32Array;
  v: Float32Array;
  d: Float32Array;
}

export interface DecodedCtrl {
  n: number;
  u: Float32Array;
  v: Float32Array;
  d: Float32Array;
  w: Float32Array;
  /** model-A aligned depth at control pixel (NaN possible) */
  pred: Float32Array;
  /** |log(d/gt)| at nearest GT within 3px (NaN possible) */
  errgt: Float32Array;
}

/** u32 hp,wp,step | f32 depth_raw | f32 depth_aligned | u8 rgb[3n] */
export function decodePoints(ab: ArrayBuffer): DecodedPoints {
  const dv = new DataView(ab);
  const hp = dv.getUint32(0, true);
  const wp = dv.getUint32(4, true);
  const step = dv.getUint32(8, true);
  const n = hp * wp;
  return {
    hp,
    wp,
    step,
    n,
    dRaw: new Float32Array(ab, 12, n),
    dAl: new Float32Array(ab, 12 + 4 * n, n),
    rgb: new Uint8Array(ab, 12 + 8 * n, 3 * n),
  };
}

/** u32 N | f32 u | f32 v | f32 d */
export function decodeGT(ab: ArrayBuffer): DecodedGT {
  const n = new DataView(ab).getUint32(0, true);
  return {
    n,
    u: new Float32Array(ab, 4, n),
    v: new Float32Array(ab, 4 + 4 * n, n),
    d: new Float32Array(ab, 4 + 8 * n, n),
  };
}

/** u32 n | f32 u | f32 v | f32 d_ctrl | f32 w | f32 d_pred_aligned | f32 err_gt */
export function decodeCtrl(ab: ArrayBuffer): DecodedCtrl {
  const n = new DataView(ab).getUint32(0, true);
  const off = (i: number) => 4 + 4 * n * i;
  return {
    n,
    u: new Float32Array(ab, off(0), n),
    v: new Float32Array(ab, off(1), n),
    d: new Float32Array(ab, off(2), n),
    w: new Float32Array(ab, off(3), n),
    pred: new Float32Array(ab, off(4), n),
    errgt: new Float32Array(ab, off(5), n),
  };
}

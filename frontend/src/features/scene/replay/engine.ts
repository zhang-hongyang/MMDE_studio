// Fly-through replay math, ported from pc_viewer.html "fly mode".
//
// index cam_quat is OpenCV cam->world (z fwd, y down); three's camera looks
// down -Z with +Y up, hence the 180-deg-about-X fix quaternion.
// Playback position = SECONDS over the selected drives' real timeline (from
// cam_t; capture is only 1-2 Hz so never assume a frame rate). The playlist
// concatenates selected segments: t=0 starts drive S1, ends where S2 begins.

import * as THREE from "three";
import type { SceneIndex } from "../decodeScene";

export const FLY_FIX = new THREE.Quaternion(1, 0, 0, 0);

export interface FlySeg {
  start: number; // first global frame
  end: number; // one past the last global frame
  tl: number; // segment length in seconds (from cam_t)
  tl0: number; // segment start on the concatenated timeline
}

export function buildFlySegs(index: SceneIndex | null, sel: boolean[]): FlySeg[] {
  if (!index) return [];
  const ts = index.cam_t;
  const segs: FlySeg[] = [];
  let cum = 0;
  index.groups.forEach((g, k) => {
    if (!sel[k]) return;
    const tl = Math.max(1e-3, ts[g.end - 1] - ts[g.start]);
    segs.push({ start: g.start, end: g.end, tl, tl0: cum });
    cum += tl;
  });
  return segs;
}

export function flyPlaylistLen(segs: FlySeg[]): number {
  return segs.length ? segs[segs.length - 1].tl0 + segs[segs.length - 1].tl : 0;
}

/** playback seconds -> fractional global frame */
export function flyP2Frame(segs: FlySeg[], t: number): number {
  for (const s of segs) {
    if (t <= s.tl0 + s.tl) {
      const u = Math.min(1, Math.max(0, (t - s.tl0) / s.tl));
      return s.start + u * (s.end - s.start - 1);
    }
  }
  const last = segs[segs.length - 1];
  return last.end - 1;
}

export interface FlyPose {
  pos: THREE.Vector3;
  q: THREE.Quaternion;
  i: number; // floor frame
  gf: number; // fractional global frame
}

export function flyPose(index: SceneIndex, segs: FlySeg[], tsec: number): FlyPose | null {
  const plen = flyPlaylistLen(segs);
  if (!plen) return null;
  const gf = flyP2Frame(segs, Math.min(Math.max(tsec, 0), plen));
  const i0 = Math.floor(gf);
  const a = gf - i0;
  const P = index.cam_pos;
  const Q = index.cam_quat;
  const pos = new THREE.Vector3().fromArray(P[i0]);
  const q = new THREE.Quaternion(Q[i0][1], Q[i0][2], Q[i0][3], Q[i0][0]);
  // adjacent-frame lerp only inside one drive (never across segment seams)
  if (a > 0 && index.seqs.some((s) => i0 >= s.start && i0 + 1 < s.end)) {
    pos.lerp(new THREE.Vector3().fromArray(P[i0 + 1]), a);
    q.slerp(new THREE.Quaternion(Q[i0 + 1][1], Q[i0 + 1][2], Q[i0 + 1][3], Q[i0 + 1][0]), a);
  }
  return { pos, q, i: i0, gf };
}

/** Mutable playback clock. Lives outside React/zustand: the rAF loop writes
 * fractional seconds at 60 Hz while the store only mirrors floor(p) for the
 * scrub slider (same split as pc_viewer.html's state.fly.p vs flyScrub). */
export const flyRuntime = {
  p: 0,
};

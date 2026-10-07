// Fly-through playback loop (port of pc_viewer.html applyFly): a rAF loop
// owns the fractional playback clock (engine.flyRuntime.p), advances it in
// wall-clock seconds × speed, interpolates the odometry pose, applies it to
// the viewport camera, prefetches RGB thumbnails ahead of the playhead, and
// publishes throttled UI state (scrub position, nearest frame, minimap ring).

import { useEffect, useRef } from "react";
import { api } from "../../../lib/api";
import { useSceneStore } from "../stores";
import { buildFlySegs, flyPose, flyPlaylistLen, flyRuntime, type FlySeg } from "./engine";
import type { ViewportApi } from "../components/SceneViewport";

const PREFETCH_AHEAD = 4;
const PREFETCH_KEEP = 24;
const IMAGE_THROTTLE_MS = 40;

export function useReplay(viewport: React.RefObject<ViewportApi | null>): void {
  const flyOn = useSceneStore((s) => s.fly.on);
  const index = useSceneStore((s) => s.index);
  const sel = useSceneStore((s) => s.sel);

  // playlist rebuilt whenever the index or the segment selection changes
  const segsRef = useRef<FlySeg[]>([]);
  useEffect(() => {
    segsRef.current = buildFlySegs(index, sel);
    flyRuntime.p = Math.min(flyRuntime.p, Math.max(0, flyPlaylistLen(segsRef.current) - 1e-6));
  }, [index, sel]);

  // Space = pause/resume, Esc = exit (pc_viewer.html keydown handler)
  useEffect(() => {
    if (!flyOn) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        const st = useSceneStore.getState();
        st.setFlyPlaying(!st.fly.playing);
      } else if (e.code === "Escape") {
        useSceneStore.getState().exitFly();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flyOn]);

  // the playback loop itself
  useEffect(() => {
    if (!flyOn || !index) return;
    const vp = viewport.current;
    if (!vp) return;
    vp.saveOrbit();
    flyRuntime.p = useSceneStore.getState().fly.p; // resume where the scrubber is

    // warm the first frames so the opening image isn't a cold fetch
    const pre: HTMLImageElement[] = [];
    {
      const st = useSceneStore.getState();
      const j0 = segsRef.current[0]?.start ?? 0;
      for (let j = j0; j <= j0 + PREFETCH_AHEAD && j < index.n_frames; j++) {
        const im = new Image();
        im.src = api.rgbURL(st.dataset, st.split, j, 320);
        pre.push(im);
      }
    }

    let raf = 0;
    let last = performance.now();
    let lastImgAt = 0;
    let lastImgFrame = -1;
    let lastEvalAt = performance.now();
    let lastEvalX = Infinity;
    let lastEvalY = Infinity;
    let lastPublishKey = "";

    const loop = (now: number): void => {
      raf = requestAnimationFrame(loop);
      const st = useSceneStore.getState();
      const segs = segsRef.current;
      const plen = flyPlaylistLen(segs);
      if (!plen) return;

      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (st.fly.playing) {
        // p is real seconds: 1x speed = wall-clock playback (dt clamp only
        // absorbs tab-away hitches, as in pc_viewer.html)
        flyRuntime.p += Math.min(0.25, dt) * st.fly.speed;
        if (flyRuntime.p >= plen) {
          flyRuntime.p = plen;
          if (st.fly.playing) st.setFlyPlaying(false);
        }
      }
      const pose = flyPose(index, segs, flyRuntime.p);
      if (!pose) return;
      vp.applyFlyPose(pose, st.fly.view);

      // sidebar plays the RGB matching the *current interpolated pose*: nearest
      // frame (not floor, which trails the lerped camera by up to a full
      // frame), and the next frames are prefetched so the <img> swap hits the
      // browser cache instead of waiting on a cold server decode.
      const fr = Math.min(Math.max(Math.round(pose.gf), 0), index.n_frames - 1);
      if (fr !== lastImgFrame && now - lastImgAt > IMAGE_THROTTLE_MS) {
        lastImgFrame = fr;
        lastImgAt = now;
        for (let k = 1; k <= PREFETCH_AHEAD; k++) {
          const j = fr + k;
          if (j >= index.n_frames) break;
          const im = new Image();
          im.src = api.rgbURL(st.dataset, st.split, j, 320);
          pre.push(im);
        }
        if (pre.length > PREFETCH_KEEP) pre.splice(0, pre.length - PREFETCH_KEEP);
      }

      // publish UI-mirroring state only when the pose actually moved
      const key = `${flyRuntime.p.toFixed(3)}:${fr}:${st.fly.view}`;
      if (key !== lastPublishKey) {
        lastPublishKey = key;
        st.flyMoved(pose.pos.x, pose.pos.y, fr, flyRuntime.p);
      }

      // chunk streaming follows the vehicle: same 800 ms / 50 m gate as the
      // old render loop, evaluated against the follow target
      const t = vp.getTarget();
      const d2 = (t.x - lastEvalX) ** 2 + (t.y - lastEvalY) ** 2;
      if (now - lastEvalAt > 800 && d2 > 2500) {
        lastEvalAt = now;
        lastEvalX = t.x;
        lastEvalY = t.y;
        st.setTarget(t.x, t.y, t.z);
        void st.refreshChunks();
      }
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      vp.restoreOrbit();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flyOn, index, viewport]);
}

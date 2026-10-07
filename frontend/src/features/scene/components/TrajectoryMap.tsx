// Top-down trajectory minimap (port of pc_viewer.html drawMinimapScene):
// per-group trajectories colored by selection, S1/S2… segment toggles, white
// ring at the replaying camera, white dot at the orbit target. Click pans the
// view to the clicked spot (nearest trajectory sample keeps road elevation);
// double-click refits the camera.

import { useCallback, useEffect, useRef } from "react";
import { useSceneStore } from "../stores";
import type { ViewportApi } from "./SceneViewport";

const GROUP_COLORS = ["#e69f00", "#56b4e9", "#009e73", "#cc79a7", "#0072b2", "#d55e00"];
const OFF_COLOR = "#454b54";

export function groupColor(k: number): string {
  return GROUP_COLORS[k % GROUP_COLORS.length];
}

interface TrajectoryMapProps {
  apiRef: React.RefObject<ViewportApi | null>;
}

export function TrajectoryMap({ apiRef }: TrajectoryMapProps) {
  const index = useSceneStore((s) => s.index);
  const sel = useSceneStore((s) => s.sel);
  const flyX = useSceneStore((s) => s.fly.x);
  const flyY = useSceneStore((s) => s.fly.y);
  const flyOn = useSceneStore((s) => s.fly.on);
  const targetVersion = useSceneStore((s) => s.targetVersion);

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mmRef = useRef<{ s: number; ox: number; oy: number } | null>(null);
  const downRef = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const idx = useSceneStore.getState().index;
    if (!idx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = wrap.clientWidth;
    const H = wrap.clientHeight;
    if (W < 10 || H < 10) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const st = useSceneStore.getState();
    const ex = idx.bbox.max[0] - idx.bbox.min[0];
    const ey = idx.bbox.max[1] - idx.bbox.min[1];
    const pad = 14;
    const s = Math.min((W - 2 * pad) / ex, (H - 2 * pad) / ey);
    const ox = (W - s * ex) / 2;
    const oy = (H - s * ey) / 2;
    mmRef.current = { s, ox, oy };
    const XY = (p: [number, number, number]): [number, number] => [
      ox + (p[0] - idx.bbox.min[0]) * s,
      oy + (p[1] - idx.bbox.min[1]) * s,
    ];

    idx.groups.forEach((g, k) => {
      const on = st.sel[k];
      ctx.strokeStyle = on ? groupColor(k) : OFF_COLOR;
      ctx.lineWidth = on ? 2 : 1;
      ctx.beginPath();
      for (let i = g.start; i < g.end; i++) {
        const [x, y] = XY(idx.cam_pos[i]);
        if (i === g.start) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "top";
    idx.groups.forEach((g, k) => {
      ctx.fillStyle = st.sel[k] ? groupColor(k) : OFF_COLOR;
      ctx.fillText(`${st.sel[k] ? "☑" : "☐"} S${k + 1} ${g.seq_id}`, 6, 4 + k * 12);
    });

    if (st.fly.on) {
      // white ring at the replaying camera
      const [x, y] = XY([st.fly.x, st.fly.y, 0]);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 7);
      ctx.stroke();
    }
    const t = apiRef.current?.getTarget();
    if (t) {
      const [tx, ty] = XY([t.x, t.y, 0]);
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(tx, ty, 3.5, 0, 7);
      ctx.fill();
    }
  }, [apiRef]);

  // redraw on state changes + container resize
  useEffect(() => {
    draw();
  }, [draw, index, sel, flyOn, flyX, flyY, targetVersion]);
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [draw]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    downRef.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = downRef.current;
    if (d && Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true;
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    // only a plain click pans (no drag-select in scene mode)
    const wasClick = downRef.current && !downRef.current.moved;
    downRef.current = null;
    if (!wasClick) return;
    const idx = useSceneStore.getState().index;
    const mm = mmRef.current;
    if (!idx || !mm) return;
    if (e.button !== 0) return;
    const st = useSceneStore.getState();
    if (st.fly.on) st.exitFly(); // pc_viewer.html: minimap click exits fly
    const r = canvasRef.current!.getBoundingClientRect();
    const wx = idx.bbox.min[0] + (e.clientX - r.left - mm.ox) / mm.s;
    const wy = idx.bbox.min[1] + (e.clientY - r.top - mm.oy) / mm.s;
    // nearest trajectory sample keeps road elevation
    let bz = idx.cam_pos[0][2];
    let bd = Infinity;
    for (const p of idx.cam_pos) {
      const d = (p[0] - wx) * (p[0] - wx) + (p[1] - wy) * (p[1] - wy);
      if (d < bd) {
        bd = d;
        bz = p[2];
      }
    }
    apiRef.current?.panTo(wx, wy, bz);
    draw();
  };

  if (!index) return null;
  return (
    <div ref={wrapRef} className="relative h-[220px] w-full overflow-hidden rounded-[6px] border border-border bg-black">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
        aria-label="轨迹俯视图"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={() => apiRef.current?.refit()}
      />
    </div>
  );
}

/** S1/S2… segment toggles: switch which sequences' chunks load + replay. */
export function SegmentToggles() {
  const index = useSceneStore((s) => s.index);
  const sel = useSceneStore((s) => s.sel);
  if (!index || index.groups.length < 2) return null;
  const all = sel.every(Boolean);
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
      <button
        type="button"
        className="rounded-[6px] border border-border px-1.5 py-0.5 text-fg-muted hover:bg-card-2"
        aria-pressed={all}
        onClick={() => void useSceneStore.getState().setAllGroups(!all)}
      >
        全部
      </button>
      {index.groups.map((g, k) => (
        <button
          key={g.seq_id}
          type="button"
          title={`${g.seq_id} · 帧 ${g.start}-${g.end - 1} · ${(g.n_points / 1e6).toFixed(1)}M 点`}
          aria-pressed={sel[k]}
          onClick={() => void useSceneStore.getState().toggleGroup(k)}
          className={
            "rounded-[6px] border px-1.5 py-0.5 transition-colors duration-150 " +
            (sel[k] ? "border-transparent text-white" : "border-border text-fg-muted hover:bg-card-2")
          }
          style={sel[k] ? { backgroundColor: groupColor(k) } : undefined}
        >
          S{k + 1}
        </button>
      ))}
    </div>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { useExploreStore, type Rect } from "../../stores/explore";

interface DragState {
  u: number;
  v: number;
}

/**
 * Frame-A thumbnail with ROI rectangle selection.
 * Drag to filter 3D points to the region; double-click to reset.
 */
export function Minimap({ width = 640 }: { width?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const meta = useExploreStore((s) => s.meta);
  const rect = useExploreStore((s) => s.rect);
  const setRect = useExploreStore((s) => s.setRect);
  const dataset = useExploreStore((s) => s.dataset);
  const split = useExploreStore((s) => s.split);
  const row = useExploreStore((s) => s.frames[s.idx]?.idx ?? s.idx);

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [dragRect, setDragRect] = useState<Rect | null>(null);

  useEffect(() => {
    let alive = true;
    const im = new Image();
    im.src = api.rgbURL(dataset, split, row, width);
    im.decode()
      .then(() => {
        if (alive) setImg(im);
      })
      .catch(() => {
        /* keep previous image on failure */
      });
    return () => {
      alive = false;
    };
  }, [dataset, split, row, width]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !meta) return;
    const H = Math.round((canvas.width * meta.h) / meta.w);
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, H);
    const r = dragRect ?? rect;
    if (r) {
      const s = canvas.width / meta.w;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(r.u0 * s, r.v0 * s, (r.u1 - r.u0) * s, (r.v1 - r.v0) * s);
    }
  }, [meta, img, rect, dragRect]);

  useEffect(() => {
    draw();
  }, [draw]);

  const toImg = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>): DragState => {
      const c = canvasRef.current!;
      const b = c.getBoundingClientRect();
      return {
        u: ((e.clientX - b.left) * (meta?.w ?? b.width)) / b.width,
        v: ((e.clientY - b.top) * (meta?.h ?? b.height)) / b.height,
      };
    },
    [meta],
  );

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    dragRef.current = toImg(e);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const p = toImg(e);
    const d = dragRef.current;
    setDragRect({
      u0: Math.min(d.u, p.u),
      v0: Math.min(d.v, p.v),
      u1: Math.max(d.u, p.u),
      v1: Math.max(d.v, p.v),
    });
  };

  const onPointerUp = () => {
    if (!dragRef.current) return;
    dragRef.current = null;
    const r = dragRect;
    setDragRect(null);
    if (r && r.u1 - r.u0 >= 4 && r.v1 - r.v0 >= 4) setRect(r);
  };

  const onDoubleClick = () => setRect(null);

  if (!meta) return null;
  return (
    <canvas
      ref={canvasRef}
      className="w-full cursor-crosshair rounded-[6px] border border-border bg-black touch-none"
      width={width}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={onDoubleClick}
    />
  );
}

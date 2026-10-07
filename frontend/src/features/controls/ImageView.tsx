import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { colormap, type RGB } from "../../lib/colormap";
import { useControlsStore, COLOR_MODE_LABEL } from "../../stores/controls";
import { Skeleton } from "../../components/ui/skeleton";

interface Hover {
  src: string;
  i: number;
  /** cursor position within the container, CSS px (for the tooltip) */
  x: number;
  y: number;
}

function p95Cap(vals: number[]): number {
  if (vals.length === 0) return 1;
  const s = [...vals].sort((a, b) => a - b);
  return Math.max(s[Math.floor(0.95 * (s.length - 1))], 1e-3);
}

/**
 * Full-resolution RGB frame with sparse control points overlaid.
 * Canvas backing store is the image's native size; CSS scales it to fit.
 */
export function ImageView() {
  const dataset = useControlsStore((s) => s.dataset);
  const split = useControlsStore((s) => s.split);
  const frames = useControlsStore((s) => s.frames);
  const idx = useControlsStore((s) => s.idx);
  const controls = useControlsStore((s) => s.controls);
  const enabled = useControlsStore((s) => s.enabled);
  const colorMode = useControlsStore((s) => s.colorMode);
  const colormapName = useControlsStore((s) => s.colormap);
  const psize = useControlsStore((s) => s.psize);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  const row = frames[idx]?.idx ?? idx;

  useEffect(() => {
    let alive = true;
    const im = new Image();
    im.src = api.rgbURL(dataset, split, row);
    im.decode()
      .then(() => {
        if (alive) setImg(im);
      })
      .catch(() => {
        if (alive) setImg(null);
      });
    return () => {
      alive = false;
    };
  }, [dataset, split, row]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (img) {
      if (canvas.width !== img.naturalWidth) canvas.width = img.naturalWidth;
      if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // per-frame normalization range for the active coloring mode
    const vals: number[] = [];
    for (const src of enabled) {
      const c = controls[src];
      if (!c) continue;
      for (let i = 0; i < c.n; i++) {
        const x =
          colorMode === "depth"
            ? c.d[i]
            : colorMode === "weight"
              ? c.w[i]
              : colorMode === "errpred"
                ? c.pred[i] > 0 && c.d[i] > 0
                  ? Math.abs(Math.log(c.d[i] / c.pred[i]))
                  : NaN
                : c.errgt[i];
        if (Number.isFinite(x)) vals.push(x);
      }
    }
    if (vals.length === 0) return;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of vals) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (colorMode === "errpred" || colorMode === "errgt") {
      lo = 0;
      hi = p95Cap(vals);
    }
    const span = hi - lo || 1;

    const tmp: RGB = [0, 0, 0];
    for (const src of enabled) {
      const c = controls[src];
      if (!c) continue;
      for (let i = 0; i < c.n; i++) {
        const x =
          colorMode === "depth"
            ? c.d[i]
            : colorMode === "weight"
              ? c.w[i]
              : colorMode === "errpred"
                ? c.pred[i] > 0 && c.d[i] > 0
                  ? Math.abs(Math.log(c.d[i] / c.pred[i]))
                  : NaN
                : c.errgt[i];
        if (!Number.isFinite(x)) continue; // NaN points are not drawn
        const [r, g, b] = colormap(colormapName, (x - lo) / span, tmp);
        ctx.fillStyle = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
        ctx.beginPath();
        ctx.arc(c.u[i], c.v[i], psize, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (hover) {
      const c = controls[hover.src];
      if (c) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(c.u[hover.i], c.v[hover.i], psize + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [img, controls, enabled, colorMode, colormapName, psize, hover]);

  useEffect(() => {
    draw();
  }, [draw]);

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !img) return;
    const b = canvas.getBoundingClientRect();
    const u = ((e.clientX - b.left) * img.naturalWidth) / b.width;
    const v = ((e.clientY - b.top) * img.naturalHeight) / b.height;
    const scale = b.width / img.naturalWidth; // CSS px per image px
    const thresh = Math.max(10, psize * 2) / scale;
    let best: { src: string; i: number; dist: number } | null = null;
    for (const src of enabled) {
      const c = controls[src];
      if (!c) continue;
      for (let i = 0; i < c.n; i++) {
        const du = c.u[i] - u;
        const dv = c.v[i] - v;
        const dist = du * du + dv * dv;
        if (dist <= thresh * thresh && (!best || dist < best.dist)) {
          best = { src, i, dist };
        }
      }
    }
    const wb = wrap.getBoundingClientRect();
    setHover(
      best
        ? { src: best.src, i: best.i, x: e.clientX - wb.left, y: e.clientY - wb.top }
        : null,
    );
  };

  const hovered = hover ? controls[hover.src] : null;

  return (
    <div ref={wrapRef} className="relative flex min-h-0 flex-1 items-center justify-center p-2">
      {!img ? (
        <Skeleton className="h-full w-full max-w-3xl" />
      ) : (
        <canvas
          ref={canvasRef}
          width={img.naturalWidth}
          height={img.naturalHeight}
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
          className="max-h-full max-w-full rounded-[6px] border border-border bg-black"
          aria-label="RGB 影像与控制点叠加视图"
        />
      )}
      {hover && hovered && (
        <div
          className="pointer-events-none absolute z-10 rounded-[6px] border border-border bg-card/90 px-2 py-1 text-[12px] shadow-md"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          <div className="font-mono text-fg-muted">{hover.src}</div>
          <div className="num">
            u={hovered.u[hover.i].toFixed(1)} v={hovered.v[hover.i].toFixed(1)}
          </div>
          <div className="num">
            d={hovered.d[hover.i].toFixed(2)} w={hovered.w[hover.i].toFixed(3)}
          </div>
          <div className="num text-fg-muted">
            {COLOR_MODE_LABEL.errpred}：
            {hovered.pred[hover.i] > 0
              ? Math.abs(Math.log(hovered.d[hover.i] / hovered.pred[hover.i])).toFixed(3)
              : "—"}
            {" · "}
            {COLOR_MODE_LABEL.errgt}：
            {Number.isFinite(hovered.errgt[hover.i])
              ? hovered.errgt[hover.i].toFixed(3)
              : "—"}
          </div>
        </div>
      )}
    </div>
  );
}

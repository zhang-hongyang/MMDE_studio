import { useMemo } from "react";
import { colormap, type RGB } from "../../../lib/colormap";
import { useControlsStore, type CtrlColormap } from "../../../stores/controls";
import { cn } from "../../../lib/utils";

function gradientCSS(name: CtrlColormap, n = 16): string {
  const tmp: RGB = [0, 0, 0];
  const parts: string[] = [];
  for (let i = 0; i <= n; i++) {
    const [r, g, b] = colormap(name, i / n, tmp);
    parts.push(`rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`);
  }
  return `linear-gradient(to right, ${parts.join(",")})`;
}

/** Color scale legend for the controls 2D view. Range is computed from the
 * currently enabled sources / coloring mode. */
export function ColorLegend({ className }: { className?: string }) {
  const colorMode = useControlsStore((s) => s.colorMode);
  const colormapName = useControlsStore((s) => s.colormap);
  const controls = useControlsStore((s) => s.controls);
  const enabled = useControlsStore((s) => s.enabled);

  const { min, max, left, right } = useMemo(() => {
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
    if (vals.length === 0) return { min: 0, max: 0, left: "—", right: "—" };
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of vals) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (colorMode === "errpred" || colorMode === "errgt") {
      // robust cap: P95 so a few outliers don't squash the scale
      const s = [...vals].sort((a, b) => a - b);
      hi = Math.max(s[Math.floor(0.95 * (s.length - 1))], 1e-3);
      lo = 0;
    }
    const fmt = (v: number) =>
      v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    return { min: lo, max: hi, left: fmt(lo), right: fmt(hi) };
  }, [controls, enabled, colorMode]);

  const bg = useMemo(() => gradientCSS(colormapName), [colormapName]);

  return (
    <div className={cn("flex items-center gap-1.5 text-[12px] text-fg-muted", className)}>
      <span className="num">{left}</span>
      <div className="h-2 w-28 rounded-[4px] border border-border" style={{ background: bg }} />
      <span className="num">{right}</span>
      <span className="text-fg-muted/70">
        {colorMode === "depth" ? "深度" : colorMode === "weight" ? "权重" : "|log err|"}
      </span>
      {min === max && <span className="num">（无有效点）</span>}
    </div>
  );
}

import { colormap, rdbu } from "../../lib/colormap";
import { useExploreStore } from "../../stores/explore";
import { cn } from "../../lib/utils";

function gradientCSS(stops: (t: number) => string, n = 16): string {
  const parts: string[] = [];
  for (let i = 0; i <= n; i++) parts.push(stops(i / n));
  return `linear-gradient(to right, ${parts.join(",")})`;
}

/** color scale legend for |A-B| / Δd coloring modes */
export function ColorLegend({ className }: { className?: string }) {
  const coloring = useExploreStore((s) => s.coloring);
  const colormap = useExploreStore((s) => s.colormap);
  const mode = useExploreStore((s) => s.mode);

  const active =
    coloring === "diff" || (mode === "frame" && coloring === "fdiff");
  if (!active) return null;

  let bg: string;
  let left: string;
  let mid: string | null = null;
  let right: string;
  if (coloring === "diff") {
    const hex = (t: number) => {
      const [r, g, b] = colormapNameToFn(colormap)(t);
      return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
    };
    bg = gradientCSS(hex);
    left = "0";
    right = "P95";
  } else {
    const hex = (t: number) => {
      const [r, g, b] = rdbu(t);
      return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
    };
    bg = gradientCSS(hex);
    left = "-P95";
    mid = "0";
    right = "+P95";
  }
  return (
    <div className={cn("flex items-center gap-1.5 text-[12px] text-fg-muted", className)}>
      <span className="num">{left}</span>
      <div className="h-2 w-24 rounded-[4px] border border-border" style={{ background: bg }} />
      {mid && <span className="num">{mid}</span>}
      <span className="num">{right}</span>
    </div>
  );
}

function colormapNameToFn(name: "plasma" | "turbo" | "jet") {
  return (t: number) => colormap(name, t);
}

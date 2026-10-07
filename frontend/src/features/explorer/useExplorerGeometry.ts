import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { useExploreStore } from "../../stores/explore";
import { useThemeStore } from "../../stores/theme";
import { buildCloud, buildCtrl, buildFrameB, buildGT, type BuiltCloud, type FrameDiffStats } from "./geometry";
import type { RGB } from "../../lib/colormap";

/** overlay / fixed colors resolved from design tokens (accent = A, primary = B) */
export function useOverlayColors(): { a: RGB; b: RGB } {
  const theme = useThemeStore((s) => s.theme);
  const [colors, setColors] = useState<{ a: RGB; b: RGB }>({
    a: [0.851, 0.467, 0.024], // dark --accent #D97706
    b: [0.231, 0.373, 0.965], // dark --primary #3B82F6
  });
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    const parse = (name: string): RGB => {
      const c = new THREE.Color(cs.getPropertyValue(name).trim());
      return [c.r, c.g, c.b];
    };
    setColors({ a: parse("--accent"), b: parse("--primary") });
  }, [theme]);
  return colors;
}

export interface ExplorerGeometry {
  cloudA: BuiltCloud | null;
  cloudB: BuiltCloud | null;
  gt: BuiltCloud | null;
  ctrl: BuiltCloud | null;
  frameStats: FrameDiffStats | null;
}

/**
 * Builds all BufferGeometries for the current explorer state.
 * Heavy work is memoized on data + display params; geometries are disposed
 * on change/unmount. Stats are reported to the store for the status bar.
 */
export function useExplorerGeometry(colors: { a: RGB; b: RGB }): ExplorerGeometry {
  const meta = useExploreStore((s) => s.meta);
  const metaB = useExploreStore((s) => s.metaB);
  const cloudA = useExploreStore((s) => s.cloudA);
  const cloudB = useExploreStore((s) => s.cloudB);
  const gt = useExploreStore((s) => s.gt);
  const ctrl = useExploreStore((s) => s.ctrl);
  const Trel = useExploreStore((s) => s.Trel);
  const idxB = useExploreStore((s) => s.idxB);
  const mode = useExploreStore((s) => s.mode);
  const coloring = useExploreStore((s) => s.coloring);
  const colormap = useExploreStore((s) => s.colormap);
  const rect = useExploreStore((s) => s.rect);
  const onlyOverlap = useExploreStore((s) => s.onlyOverlap);
  const showGT = useExploreStore((s) => s.showGT);
  const showCtrl = useExploreStore((s) => s.showCtrl);
  const ctrlColor = useExploreStore((s) => s.ctrlColor);
  const setStats = useExploreStore((s) => s.setStats);

  const built = useMemo<ExplorerGeometry>(() => {
    const out: ExplorerGeometry = { cloudA: null, cloudB: null, gt: null, ctrl: null, frameStats: null };
    if (!meta) return out;

    if (mode === "frame") {
      if (cloudA) {
        out.cloudA = buildCloud(cloudA, {
          meta,
          rect,
          colorMode: "rgb",
        });
      }
      if (cloudB && metaB && Trel && idxB !== null) {
        const colorMode = coloring === "fdiff" ? "fdiff" : coloring === "fcolor" ? "fixed" : "rgb";
        const r = buildFrameB(cloudB, {
          metaA: meta,
          metaB,
          Trel,
          cloudA,
          rect,
          onlyOverlap,
          colorMode,
          fixedColor: colors.b,
          colormap,
        });
        out.cloudB = r.cloud;
        out.frameStats = r.stats;
      }
    } else if (mode === "overlay") {
      // single-viewport overlay: RGB → fixed token colors (A accent / B primary)
      if (cloudA) {
        out.cloudA = buildCloud(cloudA, {
          meta,
          rect,
          colorMode: coloring === "diff" ? "diff" : "fixed",
          fixedColor: colors.a,
          other: cloudB,
          colormap,
        });
      }
      if (cloudB) {
        out.cloudB = buildCloud(cloudB, {
          meta,
          rect,
          colorMode: coloring === "diff" ? "diff" : "fixed",
          fixedColor: colors.b,
          other: cloudA,
          colormap,
        });
      }
    } else {
      // model side-by-side: RGB or |A-B| diff in both viewports
      if (cloudA) {
        out.cloudA = buildCloud(cloudA, {
          meta,
          rect,
          colorMode: coloring === "diff" ? "diff" : "rgb",
          other: cloudB,
          colormap,
        });
      }
      if (cloudB) {
        out.cloudB = buildCloud(cloudB, {
          meta,
          rect,
          colorMode: coloring === "diff" ? "diff" : "rgb",
          other: cloudA,
          colormap,
        });
      }
    }

    if (showGT && gt && gt.n > 0) out.gt = buildGT(meta, gt, rect);
    if (showCtrl && ctrl && ctrl.n > 0) out.ctrl = buildCtrl(meta, ctrl, ctrlColor, rect);
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, metaB, cloudA, cloudB, gt, ctrl, Trel, idxB, mode, coloring, colormap, rect, onlyOverlap, showGT, showCtrl, ctrlColor, colors]);

  // dispose stale geometries
  useEffect(() => {
    return () => {
      built.cloudA?.geometry.dispose();
      built.cloudB?.geometry.dispose();
      built.gt?.geometry.dispose();
      built.ctrl?.geometry.dispose();
    };
  }, [built]);

  // report stats for the status bar
  useEffect(() => {
    setStats({
      nA: built.cloudA?.count ?? 0,
      nB: built.cloudB?.count ?? 0,
      frameStats: built.frameStats,
    });
  }, [built, setStats]);

  return built;
}

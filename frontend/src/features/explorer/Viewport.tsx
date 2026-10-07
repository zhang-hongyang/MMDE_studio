import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  type MutableRefObject,
} from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useExploreStore } from "../../stores/explore";
import { viewBus } from "./viewBus";
import type { BuiltCloud } from "./geometry";
import { cn } from "../../lib/utils";

export interface ViewportHandle {
  refit: () => void;
}

interface ViewportProps {
  viewportId: "A" | "B";
  label: string;
  clouds: { built: BuiltCloud; size: number }[];
  /** exposure multiplier applied to every point material (探索 store 亮度) */
  brightness: number;
  onActivate?: () => void;
  className?: string;
}

/** shared initial/home view: the ORIGINAL pinhole camera pose — sit just
 * behind the camera centre (z = -3) looking straight down the optical axis
 * (+z), so the view matches the RGB thumbnail (x right, image-down y with
 * camera.up = (0,-1,0)). */
const INIT_TARGET = new THREE.Vector3(0, 0, 20);
const INIT_POSITION = new THREE.Vector3(0, 0, -3);

function CameraRig({
  viewportId,
  controlsRef,
  doFitRef,
}: {
  viewportId: "A" | "B";
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
  doFitRef: MutableRefObject<() => void>;
}) {
  const fitVersion = useExploreStore((s) => s.fitVersion);
  const homeVersion = useExploreStore((s) => s.homeVersion);
  const syncView = useExploreStore((s) => s.syncView);
  const { camera, scene } = useThree();

  const doFit = useCallback(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    const box = new THREE.Box3();
    scene.traverse((o) => {
      if ((o as THREE.Points).isPoints) box.expandByObject(o);
    });
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const dist = box.getSize(new THREE.Vector3()).length() * 0.6 + 1;
    const offset = camera.position.clone().sub(controls.target);
    if (offset.lengthSq() < 1e-8) offset.set(1, 0.6, 2);
    offset.setLength(dist);
    controls.target.copy(center);
    camera.position.copy(center).add(offset);
    controls.update();
  }, [camera, scene, controlsRef]);

  useEffect(() => {
    doFitRef.current = doFit;
  }, [doFit, doFitRef]);

  // one-time initial view — set imperatively so re-renders never reset panning
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.target.copy(INIT_TARGET);
    camera.position.copy(INIT_POSITION);
    controls.update();
  }, [camera, controlsRef]);

  // data switch (frame/model/mode/camera): restore the original camera view
  useEffect(() => {
    if (homeVersion > 0) {
      const controls = controlsRef.current;
      if (!controls) return;
      controls.target.copy(INIT_TARGET);
      camera.position.copy(INIT_POSITION);
      controls.update();
    }
  }, [homeVersion, camera, controlsRef]);

  // refit on request (rect changes, double-click): recentre on the cloud but
  // keep the user's current orbit direction
  useEffect(() => {
    if (fitVersion > 0) doFit();
  }, [fitVersion, doFit]);

  // publish A's camera; mirror into B when syncView is on
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    if (viewportId === "A") {
      const publish = () => {
        viewBus.publish({
          position: [camera.position.x, camera.position.y, camera.position.z],
          target: [controls.target.x, controls.target.y, controls.target.z],
        });
      };
      controls.addEventListener("change", publish);
      return () => {
        controls.removeEventListener("change", publish);
      };
    }
    if (!syncView) return;
    const apply = (v: { position: [number, number, number]; target: [number, number, number] }) => {
      camera.position.set(...v.position);
      controls.target.set(...v.target);
      controls.update();
    };
    const unsub = viewBus.subscribe(apply);
    const latest = viewBus.latest();
    if (latest) apply(latest);
    return unsub;
  }, [viewportId, syncView, camera, controlsRef]);

  return null;
}

export const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(
  { viewportId, label, clouds, brightness, onActivate, className },
  ref,
) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const doFitRef = useRef<() => void>(() => {});
  const tint = useMemo(() => new THREE.Color(brightness, brightness, brightness), [brightness]);

  useImperativeHandle(ref, () => ({ refit: () => doFitRef.current() }), []);

  return (
    <div
      className={cn("relative min-h-0 min-w-0 overflow-hidden", className)}
      style={{ background: "var(--card-2)" }}
      onDoubleClick={() => doFitRef.current()}
      onPointerDown={onActivate}
    >
      {/* flat: NoToneMapping — vertex colors must pass through raw, like the
          old r128 viewer; default ACESFilmic washes highlights out white */}
      <Canvas
        dpr={[1, 1.5]}
        frameloop="demand"
        flat
        gl={{ antialias: false, alpha: true }}
        onCreated={({ gl }) => {
          // r169 default SRGB output would re-encode (brighten) the raw
          // vertex colors; linear output = pixel values straight to screen,
          // matching the RGB thumbnail (R3F's `linear` prop only sets the
          // removed `outputEncoding` and is a no-op on three >= r152).
          gl.outputColorSpace = THREE.LinearSRGBColorSpace;
        }}
        camera={{ fov: 60, near: 0.05, far: 2000, position: INIT_POSITION, up: [0, -1, 0] }}
      >
        <CameraRig viewportId={viewportId} controlsRef={controlsRef} doFitRef={doFitRef} />
        {/* no inertia: stop the moment the mouse button releases.
            reverseHorizontalOrbit: with camera.up=(0,-1,0) (y-down cloud
            coords) OrbitControls' azimuth is mirrored vs the old pc_viewer —
            this flag restores the old viewer's drag feel (verified against
            pc_viewer.html). */}
        <OrbitControls
          ref={controlsRef}
          minDistance={0.5}
          maxDistance={500}
          enableDamping={false}
          reverseHorizontalOrbit
        />
        <axesHelper args={[2]} />
        {clouds.map((c, i) => (
          <points key={i} geometry={c.built.geometry}>
            <pointsMaterial size={c.size} vertexColors sizeAttenuation color={tint} />
          </points>
        ))}
      </Canvas>
      <div className="pointer-events-none absolute left-2 top-2 rounded-[6px] bg-black/55 px-2 py-0.5 text-[12px] font-semibold text-white">
        {label}
      </div>
    </div>
  );
});

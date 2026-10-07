// Scene viewport: fused point cloud (overviews + streamed chunks) on an
// orbit-controlled canvas. Ports pc_viewer.html's scene-mode camera setup
// (z-up odometry frame, kilometer far plane, custom point-size shader with a
// soft pixel cap) and exposes an imperative API for the replay engine and
// the trajectory minimap.

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useSceneStore, PT_MAX, type FlyView } from "../stores";
import { FLY_FIX, type FlyPose } from "../replay/engine";
import type { SceneCloud } from "../decodeScene";

export interface ViewportApi {
  /** remember the orbit camera so fly-through can restore it on exit */
  saveOrbit: () => void;
  restoreOrbit: () => void;
  /** apply an interpolated replay pose (car or chase view); also moves the
   * orbit target so chunk streaming and the minimap follow the vehicle */
  applyFlyPose: (pose: FlyPose, view: FlyView) => void;
  /** pan the orbit target, keeping the current offset (pure fly/pan) */
  panTo: (x: number, y: number, z: number) => void;
  /** fit the camera to the scene bbox */
  refit: () => void;
  getTarget: () => THREE.Vector3;
}

interface SceneViewportProps {
  onReady: (api: ViewportApi) => void;
}

/** ShaderMaterial instead of PointsMaterial: sizeAttenuation sprites explode to
 * hundreds of px for near points -> massive overdraw is what made dragging
 * lag. SOFT cap: sizes up to uMax are physically exact; beyond that they keep
 * growing at 0.4 exponent (port of pc_viewer.html makeSceneMat). */
function makeSceneMaterial(uSize: number, uMax: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: uSize },
      uMax: { value: uMax },
      uPx: { value: 300 },
      uBrightness: { value: 1 },
    },
    vertexShader: `
      attribute vec3 color;
      uniform float uSize; uniform float uMax; uniform float uPx;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float s = max(1.0, uSize * uPx / max(1.0, -mv.z));
        gl_PointSize = s <= uMax ? s : uMax * pow(s / uMax, 0.4);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      uniform float uBrightness;
      void main() { gl_FragColor = vec4(vColor * uBrightness, 1.0); }`,
  });
}

function CloudPoints({ cloud, mat }: { cloud: SceneCloud; mat: THREE.ShaderMaterial }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(cloud.pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(cloud.col, 3, true)); // normalized u8
    g.computeBoundingSphere();
    return g;
  }, [cloud]);
  useEffect(() => () => geo.dispose(), [geo]);
  return <points geometry={geo} material={mat} />;
}

const tmpFwd = new THREE.Vector3();
const tmpUp = new THREE.Vector3();
const tmpLook = new THREE.Vector3();

/** Everything that needs the real three camera/scene/controls lives inside the
 * Canvas and publishes an imperative API for the replay engine + minimap. */
function SceneRig({ onReady }: { onReady: (api: ViewportApi) => void }) {
  const index = useSceneStore((s) => s.index);
  const overviews = useSceneStore((s) => s.overviews);
  const chunks = useSceneStore((s) => s.chunks);
  const fitVersion = useSceneStore((s) => s.fitVersion);
  const flyOn = useSceneStore((s) => s.fly.on);
  const psize = useSceneStore((s) => s.psize);
  const brightness = useSceneStore((s) => s.brightness);

  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const orbitBackup = useRef<{ pos: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const gl = useThree((s) => s.gl);
  const size = useThree((s) => s.size);

  const baseSize = index ? Math.max(0.04, index.voxel * 0.55) : 0;
  const mat = useMemo(
    () => (index ? makeSceneMaterial(baseSize, PT_MAX) : null),
    [index, baseSize],
  );
  useEffect(() => () => mat?.dispose(), [mat]);
  // 点大小 slider: multiplier on the voxel-derived base size; 亮度: exposure
  // (frameloop="demand" when not replaying — uniform mutation alone doesn't
  // schedule a frame, so invalidate explicitly)
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    if (!mat) return;
    mat.uniforms.uSize.value = baseSize * psize;
    invalidate();
  }, [mat, baseSize, psize, invalidate]);
  useEffect(() => {
    if (!mat) return;
    mat.uniforms.uBrightness.value = brightness;
    invalidate();
  }, [mat, brightness, invalidate]);
  // px-space point size needs the focal length in device pixels, per resize
  useEffect(() => {
    if (mat) mat.uniforms.uPx.value = gl.domElement.height / (2 * Math.tan((60 * Math.PI) / 360));
  }, [mat, size, gl]);

  const doFit = () => {
    const controls = controlsRef.current;
    if (!controls || !index) return;
    const box = new THREE.Box3(new THREE.Vector3(...index.bbox.min), new THREE.Vector3(...index.bbox.max));
    const center = box.getCenter(new THREE.Vector3());
    const dist = box.getSize(new THREE.Vector3()).length() * 0.6 + 1;
    const offset = camera.position.clone().sub(controls.target);
    if (offset.lengthSq() < 1e-8) offset.set(1, -1, 0.5);
    offset.setLength(dist);
    controls.target.copy(center);
    camera.position.copy(center).add(offset);
    camera.up.set(0, 0, 1); // odometry world frame has z up
    controls.update();
    useSceneStore.getState().setTarget(center.x, center.y, center.z);
  };
  const doFitRef = useRef(doFit);
  doFitRef.current = doFit;

  // publish the imperative API (re-created if the camera object changes)
  useEffect(() => {
    const api: ViewportApi = {
      saveOrbit: () => {
        const controls = controlsRef.current;
        if (!controls) return;
        orbitBackup.current = { pos: camera.position.clone(), target: controls.target.clone() };
      },
      restoreOrbit: () => {
        const controls = controlsRef.current;
        const b = orbitBackup.current;
        if (!controls || !b) return;
        camera.position.copy(b.pos);
        controls.target.copy(b.target);
        camera.up.set(0, 0, 1);
        controls.update();
        useSceneStore.getState().setTarget(b.target.x, b.target.y, b.target.z);
      },
      applyFlyPose: (pose, view) => {
        const controls = controlsRef.current;
        if (!controls) return;
        const fwd = tmpFwd.set(0, 0, 1).applyQuaternion(pose.q);
        const upw = tmpUp.set(0, -1, 0).applyQuaternion(pose.q);
        camera.up.set(0, 0, 1);
        if (view === "chase") {
          camera.position.copy(pose.pos).addScaledVector(fwd, -9).addScaledVector(upw, 3.5);
          camera.lookAt(tmpLook.copy(pose.pos).addScaledVector(fwd, 30));
        } else {
          camera.position.copy(pose.pos);
          camera.quaternion.copy(pose.q).multiply(FLY_FIX);
        }
        camera.updateMatrixWorld();
        // minimap dot + chunk streaming follow
        controls.target.copy(pose.pos).addScaledVector(fwd, 30);
      },
      panTo: (x, y, z) => {
        const controls = controlsRef.current;
        if (!controls) return;
        const offset = camera.position.clone().sub(controls.target);
        controls.target.set(x, y, z);
        camera.position.copy(controls.target).add(offset);
        controls.update();
        useSceneStore.getState().setTarget(x, y, z);
      },
      refit: () => doFitRef.current(),
      getTarget: () => controlsRef.current?.target ?? new THREE.Vector3(),
    };
    onReady(api);
  }, [camera, onReady]);

  // refit whenever a new scene index arrives (fitVersion bumps in the store);
  // afterwards re-pick chunks around the new target (the old viewer did
  // fitView() synchronously before its chunk pass — we fit in an effect)
  useEffect(() => {
    if (fitVersion > 0) {
      doFitRef.current();
      void useSceneStore.getState().refreshChunks();
    }
  }, [fitVersion]);

  return (
    <>
      <axesHelper args={[2]} />
      <OrbitControls
        ref={controlsRef}
        enabled={!flyOn}
        minDistance={0.5}
        maxDistance={6000}
        // the scene spans kilometers — wider zoom range than Explorer
        onStart={() => useSceneStore.getState().setDragging(true)}
        onEnd={() => {
          const st = useSceneStore.getState();
          st.setDragging(false);
          void st.refreshChunks();
        }}
        onChange={() => {
          const t = controlsRef.current?.target;
          if (t) useSceneStore.getState().setTarget(t.x, t.y, t.z);
        }}
      />
      {mat &&
        Object.entries(overviews).map(([k, c]) => (
          <CloudPoints key={`ov${k}`} cloud={c} mat={mat} />
        ))}
      {mat &&
        Object.entries(chunks).map(([k, c]) => c && <CloudPoints key={k} cloud={c} mat={mat} />)}
    </>
  );
}

export function SceneViewport({ onReady }: SceneViewportProps) {
  const flyOn = useSceneStore((s) => s.fly.on);
  return (
    <Canvas
      dpr={[1, 1.5]}
      frameloop={flyOn ? "always" : "demand"}
      gl={{ antialias: false, alpha: true }}
      camera={{ fov: 60, near: 0.05, far: 20000, up: [0, 0, 1], position: [0, -40, 20] }}
    >
      <SceneRig onReady={onReady} />
    </Canvas>
  );
}

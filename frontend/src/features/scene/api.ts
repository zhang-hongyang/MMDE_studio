import { decodeSceneCloud, type SceneCloud, type SceneGroup, type SceneIndex } from "./decodeScene";

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export function sceneBase(ds: string, split: string, model: string): string {
  return `/api/scene/${ds}/${split}/${model}`;
}

export async function fetchSceneIndex(ds: string, split: string, model: string): Promise<SceneIndex> {
  return getJSON<SceneIndex>(`${sceneBase(ds, split, model)}/index.json`);
}

/** ?v= mirrors pc_viewer.html: bust the browser cache across re-fusions
 * (blob names stay the same when a scene is re-fused). */
export async function fetchSceneCloud(
  ds: string,
  split: string,
  model: string,
  file: string,
  group: SceneGroup,
  created?: string,
): Promise<SceneCloud> {
  const v = created ? `?v=${encodeURIComponent(created)}` : "";
  const r = await fetch(`${sceneBase(ds, split, model)}/blob/${file}${v}`);
  if (!r.ok) throw new Error(`${r.status} ${file}`);
  return decodeSceneCloud(await r.arrayBuffer(), group);
}

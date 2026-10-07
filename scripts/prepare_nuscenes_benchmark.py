#!/usr/bin/env python3
"""Build evaluable nuScenes official-val and three continuous MMDE splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nuscenes_io import iter_json_array, project_lidar, transform  # noqa:E402


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--runtime-root", type=Path, required=True)
    ap.add_argument("--val-scenes-json", type=Path, required=True)
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--sequence-count", type=int, default=3)
    ap.add_argument("--control-fraction", type=float, default=0.2)
    return ap.parse_args()


def split_points(uv: np.ndarray, depth: np.ndarray, token: str,
                 seed: int, fraction: float):
    # Multiple LiDAR returns can project into the same image pixel.  Split
    # after pixel-level de-duplication so a control pixel can never also be an
    # evaluation pixel.  Keep the nearest return, matching sparse rasterizing.
    pixel = np.rint(uv).astype(np.int64)
    nearest: dict[tuple[int, int], int] = {}
    for index, (xy, value) in enumerate(zip(pixel, depth)):
        key = (int(xy[0]), int(xy[1]))
        old = nearest.get(key)
        if old is None or value < depth[old]:
            nearest[key] = index
    unique = np.asarray(list(nearest.values()), dtype=np.int64)
    uv, depth = uv[unique], depth[unique]
    token_seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) ^ seed
    order = np.random.default_rng(token_seed).permutation(len(depth))
    count = min(len(depth) - 10, max(5, int(round(len(depth) * fraction))))
    if count < 5:
        raise ValueError(f"not enough projected LiDAR points for {token}")
    control, evaluation = order[:count], order[count:]
    return ((uv[control], depth[control]), (uv[evaluation], depth[evaluation]))


def save_sparse(path: Path, uv: np.ndarray, depth: np.ndarray,
                controls: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uv": uv.astype(np.float32), "depth": depth.astype(np.float32)}
    if controls:
        payload["weight"] = np.ones(len(depth), dtype=np.float32)
    np.savez_compressed(path, **payload)


def hardlink(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)


def main() -> int:
    args = arguments()
    if not 0 < args.control_fraction < 1:
        raise SystemExit("--control-fraction must be between zero and one")
    dataroot = args.dataroot.resolve()
    metadata = dataroot / "v1.0-trainval"
    val_names = json.loads(args.val_scenes_json.read_text(encoding="utf-8"))
    scenes = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    scene_by_name = {scene["name"]: scene for scene in scenes}
    missing = sorted(set(val_names) - set(scene_by_name))
    if missing:
        raise SystemExit(f"official val scenes missing from release: {missing}")
    selected_scenes = [scene_by_name[name] for name in val_names]
    rng = random.Random(args.seed)
    continuous_names = sorted(rng.sample(val_names, args.sequence_count))

    samples = json.loads((metadata / "sample.json").read_text(encoding="utf-8"))
    sample_by_token = {sample["token"]: sample for sample in samples}
    selected_tokens, position = [], {}
    for scene in selected_scenes:
        token, index = scene["first_sample_token"], 0
        while token:
            position[token] = (scene["name"], index)
            selected_tokens.append(token)
            token = sample_by_token[token]["next"]
            index += 1

    sensors = json.loads((metadata / "sensor.json").read_text(encoding="utf-8"))
    channels = {item["token"]: item["channel"] for item in sensors}
    calibrations_list = json.loads(
        (metadata / "calibrated_sensor.json").read_text(encoding="utf-8"))
    calibrations = {item["token"]: item for item in calibrations_list}
    calibration_channels = {
        item["token"]: channels[item["sensor_token"]]
        for item in calibrations_list}
    wanted = set(selected_tokens)
    sample_data = {token: {} for token in selected_tokens}
    for record in iter_json_array(metadata / "sample_data.json"):
        if record["sample_token"] not in wanted or not record["is_key_frame"]:
            continue
        channel = calibration_channels[record["calibrated_sensor_token"]]
        if channel in (args.camera, "LIDAR_TOP"):
            sample_data[record["sample_token"]][channel] = record
    incomplete = [token for token, items in sample_data.items()
                  if args.camera not in items or "LIDAR_TOP" not in items]
    if incomplete:
        raise SystemExit(f"missing camera/LiDAR records: {incomplete[:10]}")

    wanted_poses = {item["ego_pose_token"] for records in sample_data.values()
                    for item in records.values()}
    ego_poses = {record["token"]: record
                 for record in iter_json_array(metadata / "ego_pose.json")
                 if record["token"] in wanted_poses}
    if set(ego_poses) != wanted_poses:
        raise SystemExit("selected records reference missing ego poses")

    output = args.output_root.resolve()
    runtime = args.runtime_root
    split_name = "val_official"
    split_root = output / "mmde_test" / "nuscenes" / split_name
    runtime_split = runtime / "mmde_test" / "nuscenes" / split_name
    frames, by_scene = [], {name: [] for name in continuous_names}
    previous_runtime = {}
    for index, token in enumerate(selected_tokens):
        scene_name, scene_frame = position[token]
        camera, lidar = sample_data[token][args.camera], sample_data[token]["LIDAR_TOP"]
        suffix = Path(camera["filename"]).suffix.lower() or ".jpg"
        image_name = f"{index:06d}{suffix}"
        image_out = split_root / "images" / image_name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataroot / camera["filename"], image_out)
        uv, depth = project_lidar(dataroot, lidar, camera,
                                  calibrations, ego_poses)
        (control_uv, control_depth), (eval_uv, eval_depth) = split_points(
            uv, depth, token, args.seed, args.control_fraction)
        save_sparse(split_root / "gt_full" / f"{index:06d}.npz", uv, depth)
        save_sparse(split_root / "gt" / f"{index:06d}.npz", eval_uv, eval_depth)
        save_sparse(split_root / "sparse_controls" / "lidar" /
                    f"{index:06d}.npz", control_uv, control_depth, True)
        camera_cal = calibrations[camera["calibrated_sensor_token"]]
        camera_ego = ego_poses[camera["ego_pose_token"]]
        world_from_camera = transform(
            camera_ego["rotation"], camera_ego["translation"]) @ transform(
                camera_cal["rotation"], camera_cal["translation"])
        runtime_image = runtime_split / "images" / image_name
        frame = {
            "seq_id": scene_name, "frame_id": scene_frame,
            "timestamp": sample_by_token[token]["timestamp"] / 1_000_000.0,
            "camera": args.camera, "image_path": str(runtime_image),
            "prev_image_path": previous_runtime.get(scene_name),
            "K": camera_cal["camera_intrinsic"],
            "T_world_camera": world_from_camera.tolist(),
            "depth_gt_path": str(runtime_split / "gt" / f"{index:06d}.npz"),
            "depth_gt_full_path": str(runtime_split / "gt_full" /
                                      f"{index:06d}.npz"),
            "sparse_depth_path": str(runtime_split / "sparse_controls" /
                                     "lidar" / f"{index:06d}.npz"),
            "source_sample_token": token,
            "evaluation_protocol": "heldout_lidar_80pct",
        }
        frames.append(frame)
        if scene_name in by_scene:
            by_scene[scene_name].append((index, frame))
        previous_runtime[scene_name] = str(runtime_image)
        if (index + 1) % 100 == 0 or index + 1 == len(selected_tokens):
            print(f"[{index + 1}/{len(selected_tokens)}] nuScenes frames", flush=True)

    split_root.mkdir(parents=True, exist_ok=True)
    with (split_root / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False) + "\n")

    continuous = []
    for ordinal, scene_name in enumerate(continuous_names, start=1):
        name = f"sequence_seed{args.seed}_{ordinal:02d}"
        target = output / "mmde_test" / "nuscenes" / name
        target_runtime = runtime / "mmde_test" / "nuscenes" / name
        sequence_frames = []
        for new_index, (old_index, original) in enumerate(by_scene[scene_name]):
            suffix = Path(original["image_path"]).suffix
            for folder, extension in (("images", suffix), ("gt", ".npz"),
                                      ("gt_full", ".npz")):
                hardlink(split_root / folder / f"{old_index:06d}{extension}",
                         target / folder / f"{new_index:06d}{extension}")
            hardlink(split_root / "sparse_controls" / "lidar" /
                     f"{old_index:06d}.npz",
                     target / "sparse_controls" / "lidar" /
                     f"{new_index:06d}.npz")
            frame = dict(original)
            frame["image_path"] = str(target_runtime / "images" /
                                      f"{new_index:06d}{suffix}")
            frame["prev_image_path"] = (sequence_frames[-1]["image_path"]
                                        if sequence_frames else None)
            frame["depth_gt_path"] = str(target_runtime / "gt" /
                                         f"{new_index:06d}.npz")
            frame["depth_gt_full_path"] = str(target_runtime / "gt_full" /
                                              f"{new_index:06d}.npz")
            frame["sparse_depth_path"] = str(
                target_runtime / "sparse_controls" / "lidar" /
                f"{new_index:06d}.npz")
            sequence_frames.append(frame)
        with (target / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for frame in sequence_frames:
                stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
        continuous.append({"split": name, "scene": scene_name,
                           "frames": len(sequence_frames)})

    registry = {"summary_path": str(runtime / "mmde_result" / "summary.json"),
                "datasets": {"nuscenes": {
                    "platform": "vehicle", "title": "nuScenes benchmark",
                    "test_root": str(runtime / "mmde_test" / "nuscenes"),
                    "pred_root": str(runtime / "mmde_result" / "preds" / "nuscenes"),
                    "metrics_root": str(runtime / "mmde_result" / "metrics" / "nuscenes"),
                    "scene_root": str(runtime / "mmde_scene" / "nuscenes")}}}
    (output / "datasets_nuscenes.yaml").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": 1, "dataset": "nuScenes-v1.0-trainval",
        "official_split": "val", "official_split_frames": len(frames),
        "camera": args.camera, "random_seed": args.seed,
        "control_fraction": args.control_fraction,
        "control_and_evaluation_points_are_disjoint": True,
        "continuous_splits": continuous,
        "val_scenes_sha256": hashlib.sha256(
            args.val_scenes_json.read_bytes()).hexdigest(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest_nuscenes.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

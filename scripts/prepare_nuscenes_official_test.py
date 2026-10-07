#!/usr/bin/env python3
"""Build the full evaluable nuScenes v1.0-test CAM_FRONT split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nuscenes_benchmark import save_sparse, split_points  # noqa:E402
from nuscenes_io import iter_json_array, project_lidar, transform  # noqa:E402


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataroot", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--runtime-root", type=Path, required=True)
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--control-fraction", type=float, default=0.2)
    return ap.parse_args()


def copy_once(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)


def main() -> int:
    args = arguments()
    if not 0 < args.control_fraction < 1:
        raise SystemExit("--control-fraction must be between zero and one")
    dataroot = args.dataroot.resolve()
    metadata = dataroot / "v1.0-test"
    scenes = sorted(
        json.loads((metadata / "scene.json").read_text(encoding="utf-8")),
        key=lambda item: item["name"],
    )
    samples = json.loads((metadata / "sample.json").read_text(encoding="utf-8"))
    sample_by_token = {sample["token"]: sample for sample in samples}
    selected_tokens: list[str] = []
    position: dict[str, tuple[str, int]] = {}
    for scene in scenes:
        token, index = scene["first_sample_token"], 0
        while token:
            position[token] = (scene["name"], index)
            selected_tokens.append(token)
            token = sample_by_token[token]["next"]
            index += 1
    if len(selected_tokens) != 6008:
        raise SystemExit(f"expected 6008 test samples, found {len(selected_tokens)}")

    sensors = json.loads((metadata / "sensor.json").read_text(encoding="utf-8"))
    channels = {item["token"]: item["channel"] for item in sensors}
    calibration_list = json.loads(
        (metadata / "calibrated_sensor.json").read_text(encoding="utf-8"))
    calibrations = {item["token"]: item for item in calibration_list}
    calibration_channels = {
        item["token"]: channels[item["sensor_token"]]
        for item in calibration_list
    }
    wanted = set(selected_tokens)
    sample_data = {token: {} for token in selected_tokens}
    for record in iter_json_array(metadata / "sample_data.json"):
        if record["sample_token"] not in wanted or not record["is_key_frame"]:
            continue
        channel = calibration_channels[record["calibrated_sensor_token"]]
        if channel in (args.camera, "LIDAR_TOP"):
            sample_data[record["sample_token"]][channel] = record
    incomplete = [token for token, records in sample_data.items()
                  if args.camera not in records or "LIDAR_TOP" not in records]
    if incomplete:
        raise SystemExit(f"missing camera/LiDAR records: {incomplete[:10]}")

    wanted_poses = {record["ego_pose_token"]
                    for records in sample_data.values()
                    for record in records.values()}
    ego_poses = {record["token"]: record
                 for record in iter_json_array(metadata / "ego_pose.json")
                 if record["token"] in wanted_poses}
    if set(ego_poses) != wanted_poses:
        raise SystemExit("selected records reference missing ego poses")

    output = args.output_root.resolve()
    runtime = args.runtime_root
    split = "official_test"
    target = output / "mmde_test" / "nuscenes" / split
    runtime_target = runtime / "mmde_test" / "nuscenes" / split
    frames = []
    previous_runtime: dict[str, str] = {}
    for index, token in enumerate(selected_tokens):
        scene_name, scene_frame = position[token]
        camera = sample_data[token][args.camera]
        lidar = sample_data[token]["LIDAR_TOP"]
        suffix = Path(camera["filename"]).suffix.lower() or ".jpg"
        image_name = f"{index:06d}{suffix}"
        copy_once(dataroot / camera["filename"], target / "images" / image_name)
        uv, depth = project_lidar(
            dataroot, lidar, camera, calibrations, ego_poses)
        (control_uv, control_depth), (eval_uv, eval_depth) = split_points(
            uv, depth, token, args.seed, args.control_fraction)
        save_sparse(target / "gt_full" / f"{index:06d}.npz", uv, depth)
        save_sparse(target / "gt" / f"{index:06d}.npz", eval_uv, eval_depth)
        save_sparse(target / "sparse_controls" / "lidar" /
                    f"{index:06d}.npz", control_uv, control_depth, True)

        camera_cal = calibrations[camera["calibrated_sensor_token"]]
        camera_ego = ego_poses[camera["ego_pose_token"]]
        world_from_camera = transform(
            camera_ego["rotation"], camera_ego["translation"]) @ transform(
                camera_cal["rotation"], camera_cal["translation"])
        runtime_image = runtime_target / "images" / image_name
        frames.append({
            "seq_id": scene_name,
            "frame_id": scene_frame,
            "timestamp": sample_by_token[token]["timestamp"] / 1_000_000.0,
            "camera": args.camera,
            "image_path": str(runtime_image),
            "prev_image_path": previous_runtime.get(scene_name),
            "K": camera_cal["camera_intrinsic"],
            "T_world_camera": world_from_camera.tolist(),
            "depth_gt_path": str(runtime_target / "gt" / f"{index:06d}.npz"),
            "depth_gt_full_path": str(runtime_target / "gt_full" /
                                      f"{index:06d}.npz"),
            "sparse_depth_path": str(runtime_target / "sparse_controls" /
                                     "lidar" / f"{index:06d}.npz"),
            "source_sample_token": token,
            "evaluation_available": True,
            "evaluation_protocol": "heldout_lidar_80pct",
        })
        previous_runtime[scene_name] = str(runtime_image)
        if (index + 1) % 100 == 0 or index + 1 == len(selected_tokens):
            print(f"[{index + 1}/{len(selected_tokens)}] nuScenes test frames",
                  flush=True)

    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f"frames.jsonl.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
    os.replace(temporary, target / "frames.jsonl")
    manifest = {
        "version": 1,
        "dataset": "nuScenes-v1.0-test",
        "split": split,
        "frames": len(frames),
        "scenes": len(scenes),
        "camera": args.camera,
        "random_seed": args.seed,
        "control_fraction": args.control_fraction,
        "control_and_evaluation_points_are_disjoint": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest_nuscenes_official_test.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

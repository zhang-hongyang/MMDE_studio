#!/usr/bin/env python3
"""Build KITTI Eigen-test and three random continuous MMDE splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

EARTH_RADIUS_M = 6378137.0


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--calibration-root", type=Path,
                    help="date-level KITTI calibration files; defaults to data-root")
    ap.add_argument("--oxts-root", type=Path,
                    help="KITTI tree containing OXTS text; defaults to data-root")
    ap.add_argument("--velodyne-root", type=Path,
                    help="KITTI tree containing Velodyne bins; defaults to data-root")
    ap.add_argument("--official-gt-depths", type=Path,
                    help="pre-generated Eigen gt_depths.npz for the official split")
    ap.add_argument("--test-files", type=Path, required=True)
    ap.add_argument("--speed2metric-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--runtime-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--sequence-count", type=int, default=3)
    ap.add_argument("--sequence-length", type=int, default=50)
    ap.add_argument("--control-fraction", type=float, default=0.2)
    return ap.parse_args()


def calibration(path: Path) -> dict[str, np.ndarray]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            result[key] = np.asarray([float(x) for x in value.split()])
        except ValueError:
            pass
    return result


def rigid(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = translation.reshape(3)
    return matrix


def rotations(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class PoseReader:
    def __init__(self, data_root: Path, calibration_root: Path, oxts_root: Path,
                 folder: str, camera: int):
        self.data_root, self.folder, self.camera = data_root, folder, camera
        self.date = folder.split("/")[0]
        date_root = calibration_root / self.date
        cam = calibration(date_root / "calib_cam_to_cam.txt")
        velo = calibration(date_root / "calib_velo_to_cam.txt")
        imu = calibration(date_root / "calib_imu_to_velo.txt")
        projection = cam[f"P_rect_0{camera}"].reshape(3, 4)
        self.K = projection[:, :3]
        rect = np.eye(4)
        rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
        cam0_from_velo = rect @ rigid(velo["R"], velo["T"])
        velo_from_imu = rigid(imu["R"], imu["T"])
        camn_from_cam0 = np.eye(4)
        camn_from_cam0[0, 3] = projection[0, 3] / projection[0, 0]
        self.camera_from_imu = camn_from_cam0 @ cam0_from_velo @ velo_from_imu
        self.oxts_root = oxts_root / folder / "oxts" / "data"
        first = sorted(self.oxts_root.glob("*.txt"))[0]
        values = [float(x) for x in first.read_text().split()]
        self.scale = math.cos(values[0] * math.pi / 180.0)
        self.origin = np.linalg.inv(self._world_from_imu(values))

    def _world_from_imu(self, values: list[float]) -> np.ndarray:
        lat, lon, alt, roll, pitch, yaw = values[:6]
        tx = self.scale * lon * math.pi * EARTH_RADIUS_M / 180.0
        ty = self.scale * EARTH_RADIUS_M * math.log(
            math.tan((90.0 + lat) * math.pi / 360.0))
        return rigid(rotations(roll, pitch, yaw), np.array([tx, ty, alt]))

    def pose(self, index: int) -> np.ndarray:
        path = self.oxts_root / f"{index:010d}.txt"
        values = [float(x) for x in path.read_text().split()]
        world_from_imu = self.origin @ self._world_from_imu(values)
        return world_from_imu @ np.linalg.inv(self.camera_from_imu)


def sparse_from_depth(depth: np.ndarray):
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    return np.column_stack((x, y)).astype(np.float32), depth[y, x].astype(np.float32)


def split_points(uv, depth, identity: str, seed: int, fraction: float):
    value = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) ^ seed
    order = np.random.default_rng(value).permutation(len(depth))
    count = min(len(depth) - 10, max(5, int(round(len(depth) * fraction))))
    if count < 5:
        raise ValueError(f"not enough LiDAR points: {identity}")
    control, evaluation = order[:count], order[count:]
    return ((uv[control], depth[control]), (uv[evaluation], depth[evaluation]))


def save_sparse(path: Path, uv, depth, controls=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uv": uv.astype(np.float32), "depth": depth.astype(np.float32)}
    if controls:
        payload["weight"] = np.ones(len(depth), dtype=np.float32)
    np.savez_compressed(path, **payload)


def available_windows(data_root: Path, oxts_root: Path,
                      velodyne_root: Path, length: int):
    candidates = []
    for drive in sorted(data_root.glob("20*/20*_sync")):
        image_dir = drive / "image_02" / "data"
        velo_dir = velodyne_root / drive.relative_to(data_root) / \
            "velodyne_points" / "data"
        oxts_dir = oxts_root / drive.relative_to(data_root) / "oxts" / "data"
        if not (image_dir.is_dir() and velo_dir.is_dir() and oxts_dir.is_dir()):
            continue
        indices = sorted(int(path.stem) for path in image_dir.glob("*.png")
                         if (velo_dir / f"{path.stem}.bin").is_file()
                         and (oxts_dir / f"{path.stem}.txt").is_file())
        runs, current = [], []
        for index in indices:
            if current and index != current[-1] + 1:
                if len(current) >= length:
                    runs.append(current)
                current = []
            current.append(index)
        if len(current) >= length:
            runs.append(current)
        if runs:
            candidates.append((drive.relative_to(data_root).as_posix(), runs))
    return candidates


def copy_context(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def build_frame(*, data_root, calibration_root, velodyne_root, output, runtime,
                split, out_index, folder,
                frame_index, camera, generate_depth_map, pose_reader,
                seed, control_fraction, independent_pair, eval_crop,
                depth_override=None):
    source_image = (data_root / folder / f"image_0{camera}" / "data" /
                    f"{frame_index:010d}.png")
    target_root = output / "mmde_test" / "kitti" / split
    runtime_root = runtime / "mmde_test" / "kitti" / split
    image_target = target_root / "images" / f"{out_index:06d}.png"
    copy_context(source_image, image_target)
    velo = (velodyne_root / folder / "velodyne_points" / "data" /
            f"{frame_index:010d}.bin")
    dense = (np.asarray(depth_override, dtype=np.float32)
             if depth_override is not None else
             generate_depth_map(calibration_root / folder.split("/")[0],
                                velo, camera, True))
    uv, depth = sparse_from_depth(dense)
    identity = f"{folder}:{frame_index}:{camera}"
    (ctrl_uv, ctrl_d), (eval_uv, eval_d) = split_points(
        uv, depth, identity, seed, control_fraction)
    save_sparse(target_root / "gt_full" / f"{out_index:06d}.npz", uv, depth)
    save_sparse(target_root / "gt" / f"{out_index:06d}.npz", eval_uv, eval_d)
    save_sparse(target_root / "sparse_controls" / "lidar" /
                f"{out_index:06d}.npz", ctrl_uv, ctrl_d, True)
    previous_index = frame_index - 1
    previous_source = (data_root / folder / f"image_0{camera}" / "data" /
                       f"{previous_index:010d}.png")
    previous_runtime = None
    previous_pose = None
    next_runtime = None
    next_pose = None
    if previous_source.is_file():
        previous_target = target_root / "context" / f"{out_index:06d}_prev.png"
        copy_context(previous_source, previous_target)
        previous_runtime = str(runtime_root / "context" /
                               f"{out_index:06d}_prev.png")
        previous_pose = pose_reader.pose(previous_index).tolist()
    elif independent_pair:
        # KITTI raw frame 0 has no predecessor.  Preserve a real two-view
        # metric baseline by pairing it with frame 1 from the same drive.
        next_index = frame_index + 1
        next_source = (data_root / folder / f"image_0{camera}" / "data" /
                       f"{next_index:010d}.png")
        if next_source.is_file():
            next_target = target_root / "context" / f"{out_index:06d}_next.png"
            copy_context(next_source, next_target)
            next_runtime = str(runtime_root / "context" /
                               f"{out_index:06d}_next.png")
            next_pose = pose_reader.pose(next_index).tolist()
    pair_candidates = []
    if independent_pair:
        choices = []
        target_translation = pose_reader.pose(frame_index)[:3, 3]
        for offset in (-1, 1, -3, 3, -5, 5, -10, 10):
            pair_index = frame_index + offset
            if pair_index < 0:
                continue
            pair_source = (data_root / folder / f"image_0{camera}" / "data" /
                           f"{pair_index:010d}.png")
            if not pair_source.is_file():
                continue
            try:
                pair_pose = pose_reader.pose(pair_index)
            except FileNotFoundError:
                continue
            baseline = float(np.linalg.norm(pair_pose[:3, 3] -
                                            target_translation))
            choices.append((abs(offset), -baseline, offset, pair_source,
                            pair_pose, baseline))
        if choices:
            by_distance = sorted(choices)
            selected = [by_distance[0]]
            sufficient = [item for item in by_distance if item[-1] >= 0.2]
            if sufficient:
                selected.append(sufficient[0])
            selected.append(min(choices, key=lambda item: item[1]))
            seen = set()
            for _, _, offset, pair_source, pair_pose, baseline in selected:
                if offset in seen:
                    continue
                seen.add(offset)
                pair_target = (target_root / "context" /
                               f"{out_index:06d}_pair_{offset:+d}.png")
                copy_context(pair_source, pair_target)
                pair_candidates.append({
                    "image_path": str(runtime_root / "context" /
                                      pair_target.name),
                    "T_world_camera": pair_pose.tolist(),
                    "frame_offset": offset,
                    "baseline_m": baseline,
                })
    return {
        "seq_id": folder, "frame_id": frame_index, "camera": f"image_0{camera}",
        "image_path": str(runtime_root / "images" / f"{out_index:06d}.png"),
        "prev_image_path": previous_runtime,
        "next_image_path": next_runtime,
        "K": pose_reader.K.tolist(),
        "T_world_camera": pose_reader.pose(frame_index).tolist(),
        "prev_T_world_camera": previous_pose,
        "next_T_world_camera": next_pose,
        "pair_candidates": pair_candidates,
        "depth_gt_path": str(runtime_root / "gt" / f"{out_index:06d}.npz"),
        "depth_gt_full_path": str(runtime_root / "gt_full" /
                                  f"{out_index:06d}.npz"),
        "sparse_depth_path": str(runtime_root / "sparse_controls" / "lidar" /
                                 f"{out_index:06d}.npz"),
        "independent_pair": independent_pair,
        "eval_crop": eval_crop,
        "evaluation_protocol": "heldout_lidar_80pct",
    }


def write_frames(path: Path, frames: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False) + "\n")


def main() -> int:
    args = arguments()
    sys.path.insert(0, str(args.speed2metric_root.resolve()))
    from manydepth2.kitti_utils import generate_depth_map
    data_root, output, runtime = (args.data_root.resolve(),
                                  args.output_root.resolve(), args.runtime_root)
    calibration_root = ((args.calibration_root or args.data_root).resolve())
    oxts_root = ((args.oxts_root or args.data_root).resolve())
    velodyne_root = ((args.velodyne_root or args.data_root).resolve())
    lines = [line.split() for line in args.test_files.read_text().splitlines()
             if line.strip()]
    official_depths = None
    if args.official_gt_depths:
        official_depths = np.load(args.official_gt_depths, allow_pickle=True)["data"]
        if len(official_depths) != len(lines):
            raise SystemExit("official GT count does not match test-files")
    readers = {}
    official = []
    for index, (folder, frame_text, side) in enumerate(lines):
        camera = 2 if side in ("l", "2") else 3
        key = (folder, camera)
        readers.setdefault(key, PoseReader(data_root, calibration_root, oxts_root,
                                           folder, camera))
        official.append(build_frame(
            data_root=data_root, calibration_root=calibration_root,
            velodyne_root=velodyne_root,
            output=output, runtime=runtime,
            split="eigen_test", out_index=index, folder=folder,
            frame_index=int(frame_text), camera=camera,
            generate_depth_map=generate_depth_map, pose_reader=readers[key],
            seed=args.seed, control_fraction=args.control_fraction,
            independent_pair=True, eval_crop="eigen",
            depth_override=(official_depths[index]
                            if official_depths is not None else None)))
        if (index + 1) % 100 == 0 or index + 1 == len(lines):
            print(f"[{index + 1}/{len(lines)}] KITTI Eigen frames", flush=True)
    write_frames(output / "mmde_test" / "kitti" / "eigen_test" /
                 "frames.jsonl", official)

    rng = random.Random(args.seed)
    candidates = available_windows(data_root, oxts_root, velodyne_root,
                                    args.sequence_length)
    if len(candidates) < args.sequence_count:
        raise SystemExit(f"only {len(candidates)} continuous KITTI drives available")
    chosen = rng.sample(candidates, args.sequence_count)
    continuous = []
    for ordinal, (folder, runs) in enumerate(chosen, start=1):
        run = rng.choice(runs)
        start = rng.randint(0, len(run) - args.sequence_length)
        indices = run[start:start + args.sequence_length]
        split = f"sequence_seed{args.seed}_{ordinal:02d}"
        reader = PoseReader(data_root, calibration_root, oxts_root, folder, 2)
        frames = [build_frame(
            data_root=data_root, calibration_root=calibration_root,
            velodyne_root=velodyne_root,
            output=output, runtime=runtime, split=split,
            out_index=i, folder=folder, frame_index=frame_index, camera=2,
            generate_depth_map=generate_depth_map, pose_reader=reader,
            seed=args.seed, control_fraction=args.control_fraction,
            independent_pair=False, eval_crop=None)
                  for i, frame_index in enumerate(indices)]
        write_frames(output / "mmde_test" / "kitti" / split /
                     "frames.jsonl", frames)
        continuous.append({"split": split, "drive": folder,
                           "first_frame": indices[0], "last_frame": indices[-1],
                           "frames": len(frames)})

    registry = {"summary_path": str(runtime / "mmde_result" / "summary.json"),
                "datasets": {"kitti": {
                    "platform": "vehicle", "title": "KITTI benchmark",
                    "test_root": str(runtime / "mmde_test" / "kitti"),
                    "pred_root": str(runtime / "mmde_result" / "preds" / "kitti"),
                    "metrics_root": str(runtime / "mmde_result" / "metrics" / "kitti"),
                    "scene_root": str(runtime / "mmde_scene" / "kitti")}}}
    (output / "datasets_kitti.yaml").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": 1, "dataset": "KITTI Raw", "official_split": "eigen_test",
        "official_split_frames": len(official), "random_seed": args.seed,
        "control_fraction": args.control_fraction,
        "control_and_evaluation_points_are_disjoint": True,
        "continuous_splits": continuous,
        "test_files_sha256": hashlib.sha256(args.test_files.read_bytes()).hexdigest(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest_kitti.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

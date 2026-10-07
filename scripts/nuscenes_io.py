"""Dependency-free nuScenes JSON, pose, and LiDAR projection helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np


def iter_json_array(path: Path,
                    chunk_size: int = 16 * 1024 * 1024) -> Iterator[dict]:
    """Stream records from a top-level JSON array without loading it whole."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if position >= len(buffer) - 1 and not eof:
                chunk = handle.read(chunk_size)
                eof = not chunk
                buffer = buffer[position:] + chunk
                position = 0
            while position < len(buffer) and (
                    buffer[position].isspace() or
                    (started and buffer[position] == ",")):
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"unterminated JSON array: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected JSON array: {path}")
                started = True
                position += 1
                continue
            while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                eof = not chunk
                buffer = buffer[position:] + chunk
                position = 0
                continue
            yield value
            position = end


def transform(rotation: list[float], translation: list[float]) -> np.ndarray:
    """Return a homogeneous transform from nuScenes wxyz + xyz fields."""
    w, x, y, z = np.asarray(rotation, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def project_lidar(dataroot: Path, lidar_record: dict, camera_record: dict,
                  calibrations: dict[str, dict],
                  ego_poses: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Project LIDAR_TOP into a camera and retain nearest return per pixel."""
    points = np.fromfile(dataroot / lidar_record["filename"], dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"unexpected LiDAR shape: {lidar_record['filename']}")
    xyz = points.reshape(-1, 5)[:, :3].astype(np.float64)
    homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)

    lidar_cal = calibrations[lidar_record["calibrated_sensor_token"]]
    camera_cal = calibrations[camera_record["calibrated_sensor_token"]]
    lidar_ego = ego_poses[lidar_record["ego_pose_token"]]
    camera_ego = ego_poses[camera_record["ego_pose_token"]]
    global_from_lidar = transform(
        lidar_ego["rotation"], lidar_ego["translation"]) @ transform(
            lidar_cal["rotation"], lidar_cal["translation"])
    global_from_camera = transform(
        camera_ego["rotation"], camera_ego["translation"]) @ transform(
            camera_cal["rotation"], camera_cal["translation"])
    camera_from_global = np.linalg.inv(global_from_camera)
    camera_points = (
        camera_from_global @ global_from_lidar @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    intrinsic = np.asarray(camera_cal["camera_intrinsic"], dtype=np.float64)
    projected = (intrinsic @ camera_points.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    width, height = int(camera_record["width"]), int(camera_record["height"])
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(depth) &
             (depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) &
             (uv[:, 1] >= 0) & (uv[:, 1] < height))
    uv, depth = uv[valid], depth[valid]
    if not len(depth):
        raise ValueError(
            f"LiDAR projection is empty for {camera_record['filename']}")

    pixel = np.floor(uv).astype(np.int64)
    key = pixel[:, 1] * width + pixel[:, 0]
    order = np.argsort(depth)
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return uv[keep].astype(np.float32), depth[keep].astype(np.float32)

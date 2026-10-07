#!/usr/bin/env python3
"""Fuse one MMDE continuous split/model into the Scene viewer format."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct

import cv2
import numpy as np
import yaml


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--voxel", type=float, default=0.12)
    parser.add_argument("--overview-points", type=int, default=120_000)
    parser.add_argument("--chunk-points", type=int, default=250_000)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            q = np.array([(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                          (rotation[0, 1] + rotation[1, 0]) / s,
                          (rotation[0, 2] + rotation[2, 0]) / s])
        elif axis == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            q = np.array([(rotation[0, 2] - rotation[2, 0]) / s,
                          (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                          (rotation[1, 2] + rotation[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            q = np.array([(rotation[1, 0] - rotation[0, 1]) / s,
                          (rotation[0, 2] + rotation[2, 0]) / s,
                          (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return [round(float(value), 7) for value in q]


def voxel_filter(points: np.ndarray, colors: np.ndarray, voxel: float):
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return points[indices], colors[indices]


def quantizer(points: np.ndarray) -> tuple[np.ndarray, float]:
    origin = points.min(axis=0).astype(np.float64)
    extent = points.max(axis=0).astype(np.float64) - origin
    step = max(float(extent.max()) / 65535.0, 1e-6)
    return origin, step


def write_blob(path: Path, points: np.ndarray, colors: np.ndarray,
               origin: np.ndarray, step: float) -> None:
    quantized = np.rint((points.astype(np.float64) - origin) / step)
    quantized = np.clip(quantized, 0, 65535).astype("<u2")
    rgb = np.ascontiguousarray(colors, dtype=np.uint8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<I", len(points)))
        stream.write(rgb.tobytes())
        if (3 * len(points)) & 1:
            stream.write(b"\0")
        for axis in range(3):
            stream.write(np.ascontiguousarray(quantized[:, axis]).tobytes())


def fuse_frame(frame: dict, prediction: Path, stride: int,
               min_depth: float, max_depth: float):
    image = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read image: {frame['image_path']}")
    depth = np.load(prediction).astype(np.float32, copy=False)
    height, width = image.shape[:2]
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    y, x = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    k = np.asarray(frame["K"], dtype=np.float64)
    zv = z[valid].astype(np.float64)
    xv = (x[valid] - k[0, 2]) * zv / k[0, 0]
    yv = (y[valid] - k[1, 2]) * zv / k[1, 1]
    camera = np.column_stack((xv, yv, zv))
    transform = np.asarray(frame["T_world_camera"], dtype=np.float64)
    world = camera @ transform[:3, :3].T + transform[:3, 3]
    rgb = image[::stride, ::stride][valid][:, ::-1]
    return world.astype(np.float32), np.ascontiguousarray(rgb, dtype=np.uint8)


def spatial_order(points: np.ndarray) -> np.ndarray:
    xy = points[:, :2].astype(np.float64)
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    _, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    return np.argsort(centered @ direction, kind="stable")


def main() -> int:
    args = arguments()
    if not args.split.startswith("sequence_"):
        raise SystemExit("scene fusion is only defined for test_sequence splits")
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    try:
        entry = config["datasets"][args.dataset]
    except KeyError:
        raise SystemExit(f"unknown dataset: {args.dataset}") from None
    test_root = Path(entry["test_root"])
    pred_root = Path(entry["pred_root"])
    scene_root = Path(entry["scene_root"])
    frames_path = test_root / args.split / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()
              if line.strip()]
    if not frames:
        raise SystemExit("empty frames manifest")

    target = scene_root / args.split / args.model
    if target.exists() and not args.force:
        raise SystemExit(f"scene already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)

    positions, quaternions, times = [], [], []
    t0 = None
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    try:
        for index, frame in enumerate(frames):
            transform = np.asarray(frame["T_world_camera"], dtype=np.float64)
            positions.append([round(float(value), 6) for value in transform[:3, 3]])
            quaternions.append(quaternion_wxyz(transform[:3, :3]))
            timestamp = float(frame.get("timestamp", index * 0.1))
            t0 = timestamp if t0 is None else t0
            times.append(round(timestamp - t0, 4))
            prediction = pred_root / args.split / args.model / f"{index:06d}.npy"
            if not prediction.is_file():
                raise FileNotFoundError(prediction)
            points, colors = fuse_frame(
                frame, prediction, args.stride, args.min_depth, args.max_depth)
            grouped.setdefault(str(frame.get("seq_id", "sequence")), []).append(
                (index, points, colors))

        groups, sequences = [], []
        all_min, all_max = [], []
        point_total = 0
        for group_index, (seq_id, records) in enumerate(grouped.items()):
            point_parts = [record[1] for record in records if len(record[1])]
            color_parts = [record[2] for record in records if len(record[1])]
            if not point_parts:
                continue
            points = np.concatenate(point_parts)
            colors = np.concatenate(color_parts)
            points, colors = voxel_filter(points, colors, args.voxel)
            if not len(points):
                continue
            order = spatial_order(points)
            points, colors = points[order], colors[order]
            origin, step = quantizer(points)
            minimum = points.min(axis=0).astype(float)
            maximum = points.max(axis=0).astype(float)
            all_min.append(minimum)
            all_max.append(maximum)
            point_total += len(points)

            overview_count = min(len(points), args.overview_points)
            overview_indices = np.linspace(
                0, len(points) - 1, overview_count, dtype=np.int64)
            overview_name = f"overview_s{group_index}.bin"
            write_blob(temporary / overview_name, points[overview_indices],
                       colors[overview_indices], origin, step)
            chunks = []
            for chunk_index, start in enumerate(range(0, len(points), args.chunk_points)):
                stop = min(len(points), start + args.chunk_points)
                name = f"chunk_s{group_index}_{chunk_index:03d}.bin"
                write_blob(temporary / name, points[start:stop], colors[start:stop],
                           origin, step)
                center = points[start:stop].mean(axis=0)
                chunks.append({"file": name, "n": stop - start,
                               "center": [round(float(value), 5) for value in center]})
            start_frame, end_frame = records[0][0], records[-1][0] + 1
            groups.append({
                "seq_id": seq_id,
                "start": start_frame,
                "end": end_frame,
                "n_points": len(points),
                "bbox": {"min": minimum.tolist(), "max": maximum.tolist()},
                "origin": origin.tolist(),
                "quant_step": step,
                "overview": overview_name,
                "overview_n": overview_count,
                "chunks": chunks,
            })
            sequences.append({"seq_id": seq_id, "start": start_frame, "end": end_frame})

        if not groups:
            raise ValueError("no valid positive depth points; scene is unavailable")
        bbox_min = np.min(np.stack(all_min), axis=0)
        bbox_max = np.max(np.stack(all_max), axis=0)
        index = {
            "format": 2,
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "created": datetime.now(timezone.utc).isoformat(),
            "n_frames": len(frames),
            "stride": args.stride,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "voxel": args.voxel,
            "depth_scale": 1.0,
            "n_points": point_total,
            "bbox": {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
            "groups": groups,
            "cam_pos": positions,
            "cam_quat": quaternions,
            "cam_t": times,
            "seqs": sequences,
        }
        (temporary / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"dataset": args.dataset, "split": args.split,
                      "model": args.model, "frames": len(frames),
                      "points": point_total, "target": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

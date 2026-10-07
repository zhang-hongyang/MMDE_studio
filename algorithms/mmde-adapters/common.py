"""Shared registry and array utilities for heavyweight MMDE adapters."""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--datasets-config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="first source index for deterministic sharding")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="source-index stride for deterministic sharding")
    ap.add_argument("--group-offset", type=int, default=0,
                    help="first sequence-group index for temporal sharding")
    ap.add_argument("--group-stride", type=int, default=1,
                    help="sequence-group stride for temporal sharding")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return ap


def load_context(args: argparse.Namespace) -> tuple[list[dict], Path]:
    cfg_path = Path(args.datasets_config)
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        entry = doc["datasets"][args.dataset]
    except KeyError:
        raise SystemExit(f"dataset {args.dataset!r} not in {cfg_path}")
    frame_file = Path(entry["test_root"]) / args.split / "frames.jsonl"
    if not frame_file.is_file():
        raise SystemExit(f"missing frame manifest: {frame_file}")
    frames = []
    with frame_file.open(encoding="utf-8") as stream:
        for source_idx, line in enumerate(stream):
            line = line.strip()
            if line:
                frame = json.loads(line)
                frame["_source_idx"] = source_idx
                frames.append(frame)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise SystemExit("--max-frames must be positive")
        frames = frames[:args.max_frames]
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    if not 0 <= args.frame_offset < args.frame_stride:
        raise SystemExit("--frame-offset must be in [0, frame-stride)")
    if args.group_stride <= 0:
        raise SystemExit("--group-stride must be positive")
    if not 0 <= args.group_offset < args.group_stride:
        raise SystemExit("--group-offset must be in [0, group-stride)")
    frames = frames[args.frame_offset::args.frame_stride]
    if not frames:
        raise SystemExit(f"no frames in {frame_file}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return frames, out


def grouped_frames(frames: list[dict]) -> list[list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for frame in frames:
        key = str(frame.get("seq_id", "__single_sequence__"))
        groups.setdefault(key, []).append(frame)
    return list(groups.values())


def prediction_path(out: Path, frame: dict) -> Path:
    return out / f"{int(frame['_source_idx']):06d}.npy"


def valid_prediction(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.size > 0 and np.isfinite(arr).all()
    except (OSError, ValueError):
        return False


def save_prediction(out: Path, frame: dict, value: np.ndarray) -> Path:
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"prediction must be HxW, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("prediction contains NaN/Inf")
    path = prediction_path(out, frame)
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as stream:
        np.save(stream, arr, allow_pickle=False)
    os.replace(tmp, path)
    print(f"wrote {path} shape={arr.shape} "
          f"range=[{arr.min():.6g}, {arr.max():.6g}]", flush=True)
    return path


def native_shape(frame: dict) -> tuple[int, int]:
    image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame["image_path"])
    return image.shape[:2]


def resize_native(value: np.ndarray, frame: dict) -> np.ndarray:
    h, w = native_shape(frame)
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    return arr


def finite_depth(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def inverse_proxy(metric_depth: np.ndarray) -> np.ndarray:
    depth = finite_depth(metric_depth)
    proxy = np.zeros_like(depth, dtype=np.float32)
    valid = depth > 1e-6
    proxy[valid] = 1.0 / depth[valid]
    if np.any(valid):
        lo = float(proxy[valid].min())
        hi = float(proxy[valid].max())
        proxy[valid] = (proxy[valid] - lo) / (hi - lo + 1e-8)
    return proxy


def rasterize_sparse_depth(frame: dict, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize MMDE sparse GT into an HxW metric-depth seed map."""
    path_value = frame.get("sparse_depth_path") or frame.get("depth_gt_path")
    if not path_value:
        raise ValueError("frame has neither sparse_depth_path nor depth_gt_path")
    path = Path(path_value)
    h, w = shape
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if {"uv", "depth"} <= set(data.files):
                uv = np.asarray(data["uv"])
                depth = np.asarray(data["depth"], dtype=np.float32)
                x = np.rint(uv[:, 0]).astype(np.int64)
                y = np.rint(uv[:, 1]).astype(np.int64)
                keep = ((x >= 0) & (x < w) & (y >= 0) & (y < h) &
                        np.isfinite(depth) & (depth > 0))
                out = np.zeros((h, w), dtype=np.float32)
                # If several points land on one pixel, retain the nearest.
                for px, py, z in zip(x[keep], y[keep], depth[keep]):
                    old = out[py, px]
                    if old == 0 or z < old:
                        out[py, px] = z
                return out
            for key in ("depth", "sparse_depth", "arr_0"):
                if key in data.files:
                    arr = np.asarray(data[key], dtype=np.float32)
                    break
            else:
                raise ValueError(f"unsupported sparse npz keys: {data.files}")
    elif path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        arr = raw.astype(np.float32)
        if raw.dtype == np.uint16:
            arr /= float(frame.get("sparse_depth_scale", 1000.0))
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    return finite_depth(arr)

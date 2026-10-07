#!/usr/bin/env python3
"""Build KITTI official anonymous depth-completion test for inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--runtime-root", type=Path, required=True)
    return ap.parse_args()


def copy_once(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)


def main() -> int:
    args = arguments()
    source = args.selection_root.resolve() / "test_depth_completion_anonymous"
    images = sorted((source / "image").glob("*.png"))
    if len(images) != 1000:
        raise SystemExit(f"expected 1000 KITTI test images, found {len(images)}")
    output = args.output_root.resolve()
    runtime = args.runtime_root
    split = "official_test_anonymous"
    target = output / "mmde_test" / "kitti" / split
    runtime_target = runtime / "mmde_test" / "kitti" / split
    frames = []
    for index, image in enumerate(images):
        intrinsic = source / "intrinsics" / f"{image.stem}.txt"
        sparse_png = source / "velodyne_raw" / image.name
        if not intrinsic.is_file() or not sparse_png.is_file():
            raise FileNotFoundError(f"missing official input for {image.name}")
        K = np.loadtxt(intrinsic, dtype=np.float64).reshape(3, 3)
        sparse_raw = cv2.imread(str(sparse_png), cv2.IMREAD_UNCHANGED)
        if sparse_raw is None or sparse_raw.dtype != np.uint16:
            raise ValueError(f"invalid KITTI sparse depth: {sparse_png}")
        sparse = sparse_raw.astype(np.float32) / 256.0
        y, x = np.nonzero(sparse > 0)
        depth = sparse[y, x]
        if len(depth) < 5:
            raise ValueError(f"too few KITTI sparse points: {sparse_png}")
        out_name = f"{index:06d}.png"
        copy_once(image, target / "images" / out_name)
        control_path = target / "sparse_controls" / "lidar" / f"{index:06d}.npz"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = control_path.with_name(control_path.name + f".tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                uv=np.column_stack((x, y)).astype(np.float32),
                depth=depth.astype(np.float32),
                weight=np.ones(len(depth), dtype=np.float32),
            )
        os.replace(temporary, control_path)
        frames.append({
            "seq_id": f"anonymous-{index:04d}",
            "frame_id": image.stem,
            "camera": "image",
            "image_path": str(runtime_target / "images" / out_name),
            "prev_image_path": None,
            "K": K.tolist(),
            "T_world_camera": None,
            "sparse_depth_path": str(runtime_target / "sparse_controls" /
                                     "lidar" / f"{index:06d}.npz"),
            "source_frame_name": image.name,
            "independent_pair": True,
            "temporal_context_available": False,
            "evaluation_available": False,
            "evaluation_protocol": "official_anonymous_no_public_gt",
        })
        if (index + 1) % 100 == 0 or index + 1 == len(images):
            print(f"[{index + 1}/{len(images)}] KITTI anonymous test frames",
                  flush=True)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f"frames.jsonl.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
    os.replace(temporary, target / "frames.jsonl")
    availability = {
        "evaluation_available": False,
        "reason": "KITTI official test ground truth is private",
        "models": {
            "unidepth_v2": {"mode": "metric_monocular"},
            "moge3": {"mode": "metric_monocular"},
            "mtd_dav2": {"mode": "official_sparse_lidar_input"},
            "manydepth2_vel": {"mode": "monocular_branch_no_temporal_context"},
            "ptc_dav2": {
                "available": False,
                "reason": "anonymous selection has no temporal correspondence or metric pose",
            },
        },
    }
    (target / "availability.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": 1,
        "dataset": "KITTI-depth-selection",
        "split": split,
        "frames": len(frames),
        "evaluation_available": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest_kitti_official_test.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

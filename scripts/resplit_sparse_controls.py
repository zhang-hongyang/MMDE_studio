#!/usr/bin/env python3
"""Rebuild pixel-disjoint sparse controls and held-out LiDAR evaluation GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


def split_points(uv: np.ndarray, depth: np.ndarray, identity: str,
                 seed: int, fraction: float):
    pixel = np.rint(uv).astype(np.int64)
    nearest: dict[tuple[int, int], int] = {}
    for index, (xy, value) in enumerate(zip(pixel, depth)):
        key = (int(xy[0]), int(xy[1]))
        old = nearest.get(key)
        if old is None or value < depth[old]:
            nearest[key] = index
    unique = np.asarray(list(nearest.values()), dtype=np.int64)
    uv, depth = uv[unique], depth[unique]
    value = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) ^ seed
    order = np.random.default_rng(value).permutation(len(depth))
    count = min(len(depth) - 10, max(5, int(round(len(depth) * fraction))))
    if count < 5:
        raise ValueError(f"not enough unique projected points for {identity}")
    control, evaluation = order[:count], order[count:]
    return ((uv[control], depth[control]),
            (uv[evaluation], depth[evaluation]))


def atomic_sparse(path: Path, uv: np.ndarray, depth: np.ndarray,
                  controls: bool = False) -> None:
    payload = {"uv": uv.astype(np.float32),
               "depth": depth.astype(np.float32)}
    if controls:
        payload["weight"] = np.ones(len(depth), dtype=np.float32)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-config", type=Path, required=True)
    ap.add_argument("--dataset", default="nuscenes")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--control-fraction", type=float, default=0.2)
    args = ap.parse_args()
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    root = Path(config["datasets"][args.dataset]["test_root"])
    report = {}
    for split_root in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = split_root / "frames.jsonl"
        if not manifest.is_file():
            continue
        frames = [json.loads(line) for line in manifest.read_text().splitlines()
                  if line.strip()]
        minimum_control = minimum_eval = 1 << 60
        for index, frame in enumerate(frames):
            with np.load(frame["depth_gt_full_path"]) as full:
                uv = np.asarray(full["uv"], dtype=np.float32)
                depth = np.asarray(full["depth"], dtype=np.float32)
            identity = str(frame.get("source_sample_token") or
                           f"{frame['seq_id']}:{frame['frame_id']}")
            (control_uv, control_depth), (eval_uv, eval_depth) = split_points(
                uv, depth, identity, args.seed, args.control_fraction)
            atomic_sparse(Path(frame["sparse_depth_path"]), control_uv,
                          control_depth, True)
            atomic_sparse(Path(frame["depth_gt_path"]), eval_uv, eval_depth)
            minimum_control = min(minimum_control, len(control_depth))
            minimum_eval = min(minimum_eval, len(eval_depth))
        report[split_root.name] = {
            "frames": len(frames), "minimum_control_points": minimum_control,
            "minimum_evaluation_points": minimum_eval,
        }
        print(json.dumps({split_root.name: report[split_root.name]}), flush=True)
    print(json.dumps({"dataset": args.dataset, "splits": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

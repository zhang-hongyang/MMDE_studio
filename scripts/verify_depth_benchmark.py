#!/usr/bin/env python3
"""Completion gate for MMDE benchmark data, predictions, and metrics."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import yaml


def verify_prediction(path: Path) -> str | None:
    """Fully scan one prediction while allowing file-level I/O parallelism."""
    if not path.is_file():
        return "missing"
    try:
        value = np.load(path, mmap_mode="r")
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"invalid prediction {value.shape}")
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-config", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--check-predictions", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel full-file prediction checks")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    report = {"datasets": {}, "errors": []}
    prediction_jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for dataset, entry in config["datasets"].items():
            test_root = Path(entry["test_root"])
            pred_root = Path(entry["pred_root"])
            metrics_root = Path(entry["metrics_root"])
            dataset_report = {}
            for split_root in sorted(test_root.iterdir()):
                manifest = split_root / "frames.jsonl"
                if not manifest.is_file():
                    continue
                frames = [json.loads(line) for line in manifest.read_text().splitlines()
                          if line.strip()]
                availability = {
                    bool(frame.get("evaluation_available", True)) for frame in frames
                }
                if len(availability) > 1:
                    report["errors"].append(
                        f"{dataset}/{split_root.name}: mixed evaluation availability")
                evaluable = availability != {False}
                checked_points = 0
                for index, frame in enumerate(frames):
                    required = ["image_path"]
                    if evaluable:
                        required += ["depth_gt_path", "depth_gt_full_path",
                                     "sparse_depth_path"]
                    elif frame.get("sparse_depth_path"):
                        required.append("sparse_depth_path")
                    for field in required:
                        if not Path(frame.get(field, "")).is_file():
                            report["errors"].append(
                                f"{dataset}/{split_root.name}/{index}: missing {field}")
                    if evaluable:
                        try:
                            with np.load(frame["depth_gt_path"]) as gt, \
                                 np.load(frame["sparse_depth_path"]) as control:
                                gt_pixel = {tuple(x) for x in
                                            np.rint(gt["uv"]).astype(int)}
                                control_pixel = {tuple(x) for x in
                                                 np.rint(control["uv"]).astype(int)}
                                if gt_pixel & control_pixel:
                                    report["errors"].append(
                                        f"{dataset}/{split_root.name}/{index}: "
                                        "GT/control overlap")
                                if len(gt["depth"]) < 10 or len(control["depth"]) < 5:
                                    report["errors"].append(
                                        f"{dataset}/{split_root.name}/{index}: "
                                        "too few points")
                                checked_points += len(gt["depth"]) + len(control["depth"])
                        except (OSError, KeyError, ValueError) as exc:
                            report["errors"].append(
                                f"{dataset}/{split_root.name}/{index}: {exc}")
                    if args.check_predictions:
                        for model in args.models:
                            path = (pred_root / split_root.name / model /
                                    f"{index:06d}.npy")
                            future = executor.submit(verify_prediction, path)
                            label = f"{dataset}/{split_root.name}/{model}/{index}"
                            prediction_jobs.append((label, future))
                if args.check_predictions and evaluable:
                    for model in args.models:
                        if not (metrics_root / split_root.name /
                                f"{model}.json").is_file():
                            report["errors"].append(
                                f"{dataset}/{split_root.name}/{model}: metrics missing")
                dataset_report[split_root.name] = {
                    "frames": len(frames),
                    "evaluation_available": evaluable,
                    "lidar_points_checked": checked_points,
                }
            report["datasets"][dataset] = dataset_report
        for label, future in prediction_jobs:
            error = future.result()
            if error is not None:
                report["errors"].append(f"{label}: {error}")
    report["ok"] = not report["errors"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

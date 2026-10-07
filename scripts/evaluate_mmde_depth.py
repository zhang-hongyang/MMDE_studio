#!/usr/bin/env python3
"""Evaluate MMDE depth predictions on sparse, held-out metric ground truth."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-config", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--minimum-depth", type=float, default=0.1)
    ap.add_argument("--maximum-depth", type=float, default=80.0)
    ap.add_argument("--depth-bins", type=float, nargs="+",
                    default=[0.1, 10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--gt-field", choices=("heldout", "full"),
                    default="heldout")
    return ap.parse_args()


def load_frames(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_sparse(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (np.asarray(data["uv"], dtype=np.float32),
                np.asarray(data["depth"], dtype=np.float32))


def sample_prediction(prediction: np.ndarray, uv: np.ndarray,
                      image_size: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_size
    pred_h, pred_w = prediction.shape
    # Match resize half-pixel coordinates, but clamp at image boundaries so
    # valid LiDAR points on the final row/column never mix with NaN padding.
    x = (uv[:, 0] + 0.5) * (pred_w / image_w) - 0.5
    y = (uv[:, 1] + 0.5) * (pred_h / image_h) - 0.5
    x = np.clip(x, 0.0, pred_w - 1.0)
    y = np.clip(y, 0.0, pred_h - 1.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, pred_w - 1), np.minimum(y0 + 1, pred_h - 1)
    wx, wy = x - x0, y - y0
    value = ((1 - wx) * (1 - wy) * prediction[y0, x0] +
             wx * (1 - wy) * prediction[y0, x1] +
             (1 - wx) * wy * prediction[y1, x0] +
             wx * wy * prediction[y1, x1])
    return np.asarray(value, dtype=np.float32)


def eigen_crop(uv: np.ndarray, height: int, width: int) -> np.ndarray:
    return ((uv[:, 1] >= 0.40810811 * height) &
            (uv[:, 1] < 0.99189189 * height) &
            (uv[:, 0] >= 0.03594771 * width) &
            (uv[:, 0] < 0.96405229 * width))


def frame_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ratio = np.maximum(gt / pred, pred / gt)
    error = np.abs(gt - pred)
    return {
        "abs_rel": float(np.mean(error / gt)),
        "abs_mean_m": float(np.mean(error)),
        "sq_rel": float(np.mean((gt - pred) ** 2 / gt)),
        "rmse_m": float(np.sqrt(np.mean((gt - pred) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def evaluate_model(model: str, frames: list[dict], pred_dir: Path,
                   args) -> dict:
    raw_rows, aligned_rows = [], []
    raw_abs_sum = raw_rel_sum = squared_sum = 0.0
    pixel_count = 0
    missing = invalid = 0
    scales = []
    bins = {
        f"{low:g}-{high:g}": {"minimum_m": low, "maximum_m": high,
                              "pixels": 0, "abs_error_sum_m": 0.0,
                              "abs_rel_sum": 0.0}
        for low, high in zip(args.depth_bins[:-1], args.depth_bins[1:])
    }
    gt_key = "depth_gt_path" if args.gt_field == "heldout" else "depth_gt_full_path"
    for index, frame in enumerate(frames):
        pred_path = pred_dir / f"{index:06d}.npy"
        gt_value = frame.get(gt_key)
        if not pred_path.is_file() or not gt_value:
            missing += 1
            continue
        pred = np.asarray(np.load(pred_path), dtype=np.float32).squeeze()
        if pred.ndim != 2:
            invalid += 1
            continue
        uv, gt = load_sparse(Path(gt_value))
        # Reading only the image header is materially faster than decoding
        # every RGB frame again during sparse LiDAR evaluation.
        with Image.open(frame["image_path"]) as image:
            width, height = image.size
        keep = np.isfinite(gt) & (gt > args.minimum_depth) & (gt < args.maximum_depth)
        if frame.get("eval_crop") == "eigen":
            keep &= eigen_crop(uv, height, width)
        sampled = sample_prediction(pred, uv, (height, width))
        keep &= np.isfinite(sampled) & (sampled > args.minimum_depth) & \
                (sampled < args.maximum_depth)
        gt_valid, pred_valid = gt[keep], sampled[keep]
        if len(gt_valid) < 10:
            invalid += 1
            continue
        raw_rows.append(frame_metrics(gt_valid, pred_valid))
        scale = float(np.median(pred_valid / gt_valid))
        scales.append(scale)
        aligned = pred_valid / max(scale, 1e-8)
        aligned_rows.append(frame_metrics(gt_valid, aligned))
        error = np.abs(gt_valid - pred_valid)
        raw_abs_sum += float(error.sum())
        raw_rel_sum += float((error / gt_valid).sum())
        squared_sum += float(((gt_valid - pred_valid) ** 2).sum())
        pixel_count += len(gt_valid)
        for item in bins.values():
            chosen = ((gt_valid >= item["minimum_m"]) &
                      (gt_valid < item["maximum_m"]))
            if np.any(chosen):
                item["pixels"] += int(chosen.sum())
                item["abs_error_sum_m"] += float(error[chosen].sum())
                item["abs_rel_sum"] += float((error[chosen] / gt_valid[chosen]).sum())
    bin_metrics = {}
    for name, item in bins.items():
        count = item["pixels"]
        bin_metrics[name] = {
            "minimum_m": item["minimum_m"],
            "maximum_m": item["maximum_m"],
            "pixel_count": count,
            "abs_mean_m": item["abs_error_sum_m"] / count if count else None,
            "abs_rel": item["abs_rel_sum"] / count if count else None,
        }
    log_scales = np.log(np.maximum(scales, 1e-8)) if scales else np.array([])
    return {
        "model": model,
        "frames_total": len(frames),
        "frames_evaluated": len(raw_rows),
        "frames_missing_prediction_or_gt": missing,
        "frames_invalid": invalid,
        "pixel_count": pixel_count,
        "raw_frame_mean": mean_metrics(raw_rows),
        "median_aligned_frame_mean": mean_metrics(aligned_rows),
        "raw_pixel_pooled": {
            "abs_rel": raw_rel_sum / pixel_count if pixel_count else None,
            "abs_mean_m": raw_abs_sum / pixel_count if pixel_count else None,
            "rmse_m": math.sqrt(squared_sum / pixel_count) if pixel_count else None,
        },
        "depth_bins_pixel_pooled": bin_metrics,
        "scale_pred_over_gt": {
            "median": float(np.median(scales)) if scales else None,
            "log_std": float(np.std(log_scales)) if scales else None,
            "log_mad": (float(np.median(np.abs(log_scales - np.median(log_scales))))
                        if scales else None),
        },
    }


def main() -> int:
    args = arguments()
    doc = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    entry = doc["datasets"][args.dataset]
    test_root, pred_root = Path(entry["test_root"]), Path(entry["pred_root"])
    metrics_root = Path(entry["metrics_root"])
    frames = load_frames(test_root / args.split / "frames.jsonl")
    split_pred = pred_root / args.split
    models = args.models or sorted(path.name for path in split_pred.iterdir()
                                   if path.is_dir())
    output = metrics_root / args.split
    summary = {"dataset": args.dataset, "split": args.split,
               "ground_truth": args.gt_field,
               "minimum_depth_m": args.minimum_depth,
               "maximum_depth_m": args.maximum_depth, "models": {}}
    for model in models:
        metrics = evaluate_model(model, frames, split_pred / model, args)
        summary["models"][model] = metrics
        atomic_json(output / f"{model}.json", metrics)
        raw = metrics["raw_frame_mean"]
        print(f"{model}: frames={metrics['frames_evaluated']}/{len(frames)} "
              f"AbsRel={raw.get('abs_rel')} AbsMean={raw.get('abs_mean_m')}",
              flush=True)
    atomic_json(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

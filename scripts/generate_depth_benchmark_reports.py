#!/usr/bin/env python3
"""Export all evaluable MMDE metric JSON files as stable CSV reports."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


OVERVIEW_FIELDS = (
    "dataset",
    "split",
    "model",
    "frames_total",
    "frames_evaluated",
    "frames_invalid",
    "frames_missing_prediction_or_gt",
    "abs_rel",
    "abs_mean_m",
    "rmse_m",
    "delta1",
    "median_aligned_abs_rel",
    "median_aligned_abs_mean_m",
    "scale_pred_over_gt_median",
    "scale_log_std",
)

BIN_FIELDS = (
    "dataset",
    "split",
    "model",
    "depth_bin_m",
    "minimum_m",
    "maximum_m",
    "pixel_count",
    "abs_rel",
    "abs_mean_m",
)


def _read_metrics(config_path: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for dataset, entry in sorted(config["datasets"].items()):
        root = Path(entry["metrics_root"])
        if not root.is_dir():
            continue
        for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for metric_path in sorted(split_dir.glob("*.json")):
                if metric_path.name == "summary.json":
                    continue
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                if metric.get("model"):
                    yield dataset, split_dir.name, metric


def _atomic_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    overview: list[dict[str, Any]] = []
    depth_bins: list[dict[str, Any]] = []
    for dataset, split, metric in _read_metrics(args.datasets_config):
        raw = metric.get("raw_frame_mean", {})
        aligned = metric.get("median_aligned_frame_mean", {})
        scale = metric.get("scale_pred_over_gt", {})
        common = {"dataset": dataset, "split": split, "model": metric["model"]}
        overview.append(
            {
                **common,
                "frames_total": metric.get("frames_total"),
                "frames_evaluated": metric.get("frames_evaluated"),
                "frames_invalid": metric.get("frames_invalid"),
                "frames_missing_prediction_or_gt": metric.get(
                    "frames_missing_prediction_or_gt"
                ),
                "abs_rel": raw.get("abs_rel"),
                "abs_mean_m": raw.get("abs_mean_m"),
                "rmse_m": raw.get("rmse_m"),
                "delta1": raw.get("delta1"),
                "median_aligned_abs_rel": aligned.get("abs_rel"),
                "median_aligned_abs_mean_m": aligned.get("abs_mean_m"),
                "scale_pred_over_gt_median": scale.get("median"),
                "scale_log_std": scale.get("log_std"),
            }
        )
        for label, values in sorted(metric.get("depth_bins_pixel_pooled", {}).items()):
            depth_bins.append(
                {
                    **common,
                    "depth_bin_m": label,
                    "minimum_m": values.get("minimum_m"),
                    "maximum_m": values.get("maximum_m"),
                    "pixel_count": values.get("pixel_count"),
                    "abs_rel": values.get("abs_rel"),
                    "abs_mean_m": values.get("abs_mean_m"),
                }
            )

    _atomic_csv(args.output_dir / "metrics_overview.csv", OVERVIEW_FIELDS, overview)
    _atomic_csv(args.output_dir / "depth_bins.csv", BIN_FIELDS, depth_bins)
    print(
        json.dumps(
            {
                "metrics_overview_rows": len(overview),
                "depth_bin_rows": len(depth_bins),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregated metrics endpoints (read-only over precomputed JSON files)."""
from __future__ import annotations

import json
import logging
import math

from fastapi import APIRouter

from ..registry import get_registry
from .common import get_dataset, get_split

log = logging.getLogger(__name__)

router = APIRouter()


def _benchmark_metric(payload: dict, ds: str, split: str) -> dict:
    """Translate benchmark-v2 metrics to the stable Metrics-page contract."""
    raw = payload.get("raw_frame_mean")
    if not isinstance(raw, dict):
        return payload
    bins = payload.get("depth_bins_pixel_pooled") or {}
    minimums = [v.get("minimum_m") for v in bins.values()
                if isinstance(v, dict) and isinstance(v.get("minimum_m"), (int, float))]
    maximums = [v.get("maximum_m") for v in bins.values()
                if isinstance(v, dict) and isinstance(v.get("maximum_m"), (int, float))]
    aggregate = dict(raw)
    aggregate.update({
        "scale_median": (payload.get("scale_pred_over_gt") or {}).get("median"),
        "scale_log_std": (payload.get("scale_pred_over_gt") or {}).get("log_std"),
    })
    return {
        "model": payload.get("model"),
        "dataset": ds,
        "split": split,
        "protocol": "raw_metric_depth_heldout_lidar",
        "crop": "eigen" if ds == "kitti" and split == "eigen_test" else None,
        "gt_range_m": ([min(minimums), max(maximums)]
                       if minimums and maximums else None),
        "num_frames": payload.get("frames_total", 0),
        "frames_evaluated": payload.get("frames_evaluated", 0),
        "frames_invalid": payload.get("frames_invalid", 0),
        "aggregate": aggregate,
        "per_frame": payload.get("per_frame") or [],
        "depth_bins": bins,
    }


def _json_safe(obj):
    """Replace NaN/Inf floats with None: the on-disk metrics legitimately
    carry NaN (e.g. seq-aligned errors on single-frame splits), but JSON
    responses must be compliant."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@router.get("/api/metrics/summary")
def metrics_summary():
    path = get_registry().summary_path
    if path is None or not path.is_file():
        return {"available": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"available": True, "summary": _json_safe(json.load(f))}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("summary.json unreadable: %s", exc)
        return {"available": False}


@router.get("/api/metrics/{ds}/{split}")
def metrics_list(ds: str, split: str):
    entry = get_dataset(ds)
    get_split(entry, split)
    if entry.metrics_root is None:
        return []
    d = entry.metrics_root / split
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):  # top level only; subdirs (v1_backup) skipped
        if p.name == "summary.json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not payload.get("model"):
                continue
            out.append(_json_safe(_benchmark_metric(payload, ds, split)))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("skipping unreadable metrics file %s: %s", p, exc)
    return out

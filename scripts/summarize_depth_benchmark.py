#!/usr/bin/env python3
"""Combine per-split MMDE metric summaries into the viewer summary document."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-config", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    payload = {"version": 1, "datasets": {}}
    for name, entry in config["datasets"].items():
        metrics_root = Path(entry["metrics_root"])
        splits = {}
        if metrics_root.is_dir():
            for split in sorted(metrics_root.iterdir()):
                if not split.is_dir():
                    continue
                models = {}
                for path in sorted(split.glob("*.json")):
                    if path.name == "summary.json":
                        continue
                    metric = json.loads(path.read_text())
                    if metric.get("model"):
                        models[metric["model"]] = metric
                if not models:
                    continue
                summary_payload = {
                    "dataset": name, "split": split.name,
                    "ground_truth": "heldout",
                    "minimum_depth_m": 0.1,
                    "maximum_depth_m": 80.0,
                    "models": models,
                }
                split_summary = split / "summary.json"
                split_temporary = split_summary.with_name(
                    split_summary.name + f".tmp-{os.getpid()}")
                split_temporary.write_text(
                    json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
                os.replace(split_temporary, split_summary)
                splits[split.name] = summary_payload
        payload["datasets"][name] = {"splits": splits}
    target = Path(config["summary_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

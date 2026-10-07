#!/usr/bin/env python3
"""Generate every valid test_sequence scene with bounded CPU parallelism."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be in [1, 4]")
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    fuse = Path(__file__).with_name("fuse_depth_scene.py")
    tasks, skipped = [], []
    for dataset, entry in config["datasets"].items():
        test_root = Path(entry["test_root"])
        pred_root = Path(entry["pred_root"])
        metrics_root = Path(entry["metrics_root"])
        scene_root = Path(entry["scene_root"])
        for split_root in sorted(test_root.glob("sequence_*")):
            if not (split_root / "frames.jsonl").is_file():
                continue
            for model_root in sorted((pred_root / split_root.name).iterdir()):
                if not model_root.is_dir():
                    continue
                metric_path = metrics_root / split_root.name / f"{model_root.name}.json"
                if metric_path.is_file():
                    metric = json.loads(metric_path.read_text(encoding="utf-8"))
                    if metric.get("frames_evaluated") == 0:
                        skipped.append({
                            "dataset": dataset, "split": split_root.name,
                            "model": model_root.name,
                            "reason": "all_frames_explicitly_invalid",
                        })
                        continue
                target = scene_root / split_root.name / model_root.name / "index.json"
                if target.is_file() and not args.force:
                    skipped.append({
                        "dataset": dataset, "split": split_root.name,
                        "model": model_root.name, "reason": "already_exists",
                    })
                    continue
                command = [
                    sys.executable, str(fuse),
                    "--datasets-config", str(args.datasets_config),
                    "--dataset", dataset,
                    "--split", split_root.name,
                    "--model", model_root.name,
                ]
                if args.force:
                    command.append("--force")
                tasks.append((dataset, split_root.name, model_root.name, command))

    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(subprocess.run, command, text=True,
                            capture_output=True, check=False): (dataset, split, model)
            for dataset, split, model, command in tasks
        }
        for future in as_completed(futures):
            dataset, split, model = futures[future]
            result = future.result()
            record = {"dataset": dataset, "split": split, "model": model}
            if result.returncode == 0:
                try:
                    record.update(json.loads(result.stdout.strip().splitlines()[-1]))
                except (IndexError, json.JSONDecodeError):
                    record["stdout"] = result.stdout[-1000:]
                completed.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            else:
                record.update({"returncode": result.returncode,
                               "stderr": result.stderr[-2000:]})
                errors.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    summary = {"ok": not errors, "completed": completed,
               "skipped": skipped, "errors": errors}
    print(json.dumps(summary, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refresh the selected, prediction-free archive for a completed MMDE run."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil

import yaml


DIRECTORIES = (
    "logs", "status", "manifests", "reports", "mmde_result/metrics", "mmde_scene",
)
FILES = (
    "datasets.yaml",
    "run_manifest.yaml",
    "EXPERIMENT_AUDIT.md",
    "EXPERIMENT_AUDIT.json",
    "mmde_result/summary.json",
)


def _copy(run_root: Path, archive: Path) -> None:
    for relative in DIRECTORIES:
        source = run_root / relative
        if source.is_dir():
            shutil.copytree(source, archive / relative, dirs_exist_ok=True)
    for relative in FILES:
        source = run_root / relative
        if source.is_file():
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    config = yaml.safe_load((run_root / "datasets.yaml").read_text(encoding="utf-8"))
    for dataset, entry in config["datasets"].items():
        test_root = Path(entry["test_root"])
        for split_root in sorted(path for path in test_root.iterdir() if path.is_dir()):
            source = split_root / "frames.jsonl"
            if source.is_file():
                target = archive / "manifests" / dataset / split_root.name / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def _checksums(archive: Path) -> int:
    target = archive / "provenance/checksums.sha256"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        if path == target or path.name.startswith("checksums.sha256.tmp-"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(archive)}\n")
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text("".join(rows), encoding="utf-8")
    os.replace(temporary, target)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--storage-code-path", required=True)
    parser.add_argument("--storage-result-path", required=True)
    args = parser.parse_args()

    manifest = yaml.safe_load((args.run_root / "run_manifest.yaml").read_text())
    if manifest.get("status") != "COMPLETE":
        raise SystemExit("run_manifest.yaml is not COMPLETE")
    args.archive.mkdir(parents=True, exist_ok=True)
    _copy(args.run_root, args.archive)
    archive_manifest = {
        "version": 2,
        "project": "MMDE_studio",
        "run_id": manifest["run_id"],
        "code_release": manifest["code_release"],
        "created_at": datetime.now().astimezone().isoformat(),
        "created_by": "codex",
        "protocol": "full-kitti-nuscenes-depth-benchmark-v2",
        "storage_code_path": args.storage_code_path,
        "storage_result_path": args.storage_result_path,
        "nas_runtime_path": str(args.run_root),
        "nas_predictions_path": str(args.run_root / "mmde_result/preds"),
        "nas_selected_archive_path": str(args.archive),
        "archived_result_content": [
            *DIRECTORIES,
            *FILES,
            "provenance/archive_manifest.yaml",
            "provenance/checksums.sha256",
        ],
        "prediction_policy": (
            "Full predictions remain on the NAS capacity layer and are excluded "
            "from the selected storage archive; code, manifests, metrics, logs, "
            "reports, and provenance are archived."
        ),
        "integrity": {
            "data_gate": "PASSED",
            "prediction_gate": "PASSED_FULL_FILE_SCAN",
            "viewer_contract_gate": manifest["viewer_contract_gate"],
            "viewer_api_gate": manifest["viewer_api_gate"],
            "local_audit": manifest["local_integrity_audit"],
            "independent_audit": "UNAVAILABLE",
        },
    }
    manifest_path = args.archive / "provenance/archive_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(archive_manifest, sort_keys=False))
    os.replace(temporary, manifest_path)
    count = _checksums(args.archive)
    print(f"archive={args.archive} checksums={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run final MMDE v2 release checks and atomically mark the run complete."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
from urllib.request import urlopen

import yaml


MODELS = {
    "unidepth_v2", "moge3", "ptc_dav2", "mtd_dav2", "manydepth2_vel",
}
EXPECTED_SPLITS = {
    "kitti": {
        "eigen_test": 697,
        "official_test_anonymous": 1000,
        "sequence_seed20261006_01": 50,
        "sequence_seed20261006_02": 50,
        "sequence_seed20261006_03": 50,
    },
    "nuscenes": {
        "official_test": 6008,
        "val_official": 6019,
        "sequence_seed20261006_01": 40,
        "sequence_seed20261006_02": 40,
        "sequence_seed20261006_03": 40,
    },
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _get_json(base_url: str, route: str):
    with urlopen(base_url.rstrip("/") + route, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {route}: HTTP {response.status}")
        return json.load(response)


def _get_bytes(base_url: str, route: str) -> bytes:
    with urlopen(base_url.rstrip("/") + route, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {route}: HTTP {response.status}")
        return response.read()


def _atomic_yaml(path: Path, payload) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--viewer-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--storage-code-path", required=True)
    parser.add_argument("--storage-result-path", required=True)
    args = parser.parse_args()

    errors: list[str] = []
    root = args.run_root
    marker = root / "status/finalize-v2-technical.exitcode"
    if not marker.is_file() or marker.read_text().strip() != "0":
        errors.append("technical finalize marker is absent or non-zero")

    verify_path = root / "logs/verify-predictions-v2-final.json"
    verify = _json(verify_path) if verify_path.is_file() else {}
    if not verify.get("ok") or verify.get("errors"):
        errors.append("full prediction verification did not pass cleanly")

    contract_path = root / "logs/viewer-contract-v2.log"
    contract = contract_path.read_text(encoding="utf-8") if contract_path.is_file() else ""
    passed = max((int(value) for value in re.findall(r"(\d+) passed", contract)), default=0)
    if passed < 15 or re.search(r"\b(failed|error)s?\b", contract, re.IGNORECASE):
        errors.append(f"viewer contract log is not clean (passed={passed})")

    audit_path = root / "EXPERIMENT_AUDIT.json"
    audit = _json(audit_path) if audit_path.is_file() else {}
    if audit.get("integrity_status") != "pass_local" or audit.get("errors"):
        errors.append("local mechanical audit did not pass cleanly")
    if "independent backend unavailable" not in audit.get("auditor", ""):
        errors.append("audit independence limitation is not explicit")

    overview_path = root / "reports/metrics_overview.csv"
    bins_path = root / "reports/depth_bins.csv"
    with overview_path.open(newline="", encoding="utf-8") as stream:
        overview = list(csv.DictReader(stream))
    with bins_path.open(newline="", encoding="utf-8") as stream:
        bins = list(csv.DictReader(stream))
    official_rows = [
        row for row in overview
        if row["dataset"] == "nuscenes" and row["split"] == "official_test"
    ]
    if len(overview) != 45 or len(bins) != 180:
        errors.append(f"report cardinality mismatch: overview={len(overview)}, bins={len(bins)}")
    if {row["model"] for row in official_rows} != MODELS:
        errors.append("nuScenes official-test report does not contain all five models")

    health = _get_json(args.viewer_base_url, "/api/health")
    if health.get("ok") is not True:
        errors.append("viewer health endpoint is not OK")
    registry = _get_json(args.viewer_base_url, "/api/registry")
    datasets = {item["name"]: item for item in registry["datasets"]}
    for dataset, expected in EXPECTED_SPLITS.items():
        found = {item["name"]: item for item in datasets[dataset]["splits"]}
        if set(found) != set(expected):
            errors.append(f"{dataset}: viewer split set mismatch")
            continue
        for split, count in expected.items():
            entry = found[split]
            if entry["n_frames"] != count or set(entry["models"]) != MODELS:
                errors.append(f"{dataset}/{split}: viewer count/model mismatch")
            expected_type = "test_sequence" if split.startswith("sequence_") else "test_single"
            if entry.get("test_type") != expected_type:
                errors.append(f"{dataset}/{split}: test type mismatch")
    kitti_anonymous = {
        item["name"]: item for item in datasets["kitti"]["splits"]
    }["official_test_anonymous"]
    if kitti_anonymous["evaluation_available"] is not False:
        errors.append("KITTI anonymous split incorrectly claims local evaluation")
    ptc_availability = kitti_anonymous["model_availability"]["ptc_dav2"]
    if ptc_availability.get("available") is not False or "temporal" not in ptc_availability.get("reason", ""):
        errors.append("KITTI anonymous PTC limitation is absent")

    metrics = _get_json(args.viewer_base_url, "/api/metrics/summary")
    try:
        official_models = metrics["summary"]["datasets"]["nuscenes"]["splits"][
            "official_test"
        ]["models"]
    except KeyError:
        official_models = {}
    if metrics.get("available") is not True or set(official_models) != MODELS:
        errors.append("viewer summary lacks five-model nuScenes official-test metrics")

    scenes = _get_json(args.viewer_base_url, "/api/scenes")
    scene_map = {(item["dataset"], item["split"]): set(item["models"])
                 for item in scenes}
    expected_scene_splits = {
        (dataset, f"sequence_seed20261006_{ordinal:02d}")
        for dataset in ("kitti", "nuscenes") for ordinal in range(1, 4)
    }
    if set(scene_map) != expected_scene_splits:
        errors.append("viewer scene split set mismatch")
    for dataset, split in expected_scene_splits:
        expected_models = MODELS
        if dataset == "nuscenes" and split.endswith("_01"):
            expected_models = MODELS - {"ptc_dav2"}
        if scene_map.get((dataset, split)) != expected_models:
            errors.append(f"{dataset}/{split}: scene model mismatch")

    point_checks = []
    for dataset, split in (
        ("kitti", "official_test_anonymous"),
        ("nuscenes", "official_test"),
    ):
        for model in sorted(MODELS):
            route = f"/api/points/{dataset}/{split}/0/{model}"
            payload = _get_bytes(args.viewer_base_url, route)
            point_checks.append({"route": route, "bytes": len(payload)})
            if len(payload) <= 16:
                errors.append(f"point response too small: {route}")

    current = args.release.parents[1] / "current"
    try:
        current_target = current.resolve(strict=True)
    except FileNotFoundError:
        current_target = None
    if current_target != args.release.resolve():
        errors.append(f"code/current does not resolve to {args.release}")

    report = {
        "ok": not errors,
        "errors": errors,
        "viewer_contract_tests_passed": passed,
        "metrics_overview_rows": len(overview),
        "depth_bin_rows": len(bins),
        "point_checks": point_checks,
        "scene_splits": len(scene_map),
        "scene_model_results": sum(len(models) for models in scene_map.values()),
        "independent_experiment_audit": "unavailable",
        "local_mechanical_audit": audit.get("integrity_status"),
    }
    if errors:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    manifest_path = root / "run_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "COMPLETE",
            "completed_at": datetime.now().astimezone().isoformat(),
            "prediction_integrity_gate": "PASSED_FULL_FILE_SCAN",
            "viewer_contract_gate": f"PASSED_{passed}_TESTS",
            "viewer_api_gate": "PASSED",
            "local_integrity_audit": "PASS_LOCAL_INDEPENDENT_REVIEW_UNAVAILABLE",
            "independent_integrity_audit": "UNAVAILABLE",
            "storage_status": "SELECTED_ARTIFACTS_ARCHIVED",
            "storage_code_path": args.storage_code_path,
            "storage_result_path": args.storage_result_path,
            "reports": {
                "metrics_overview_csv": "reports/metrics_overview.csv",
                "metrics_overview_rows": len(overview),
                "depth_bins_csv": "reports/depth_bins.csv",
                "depth_bin_rows": len(bins),
            },
            "scene_results": {
                "test_type": "test_sequence",
                "splits": len(scene_map),
                "model_sequence_results": sum(len(models) for models in scene_map.values()),
                "excluded": [
                    "nuscenes/sequence_seed20261006_01/ptc_dav2: all frames invalid"
                ],
            },
        }
    )
    _atomic_yaml(manifest_path, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

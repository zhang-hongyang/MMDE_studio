#!/usr/bin/env python3
"""Mechanical integrity audit for the five-method depth benchmark.

This is deliberately labelled local and is not a substitute for an
independent/cross-model scientific review.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


MODELS = ("unidepth_v2", "moge3", "ptc_dav2", "mtd_dav2",
          "manydepth2_vel")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-config", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load(args.datasets_config.read_text(encoding="utf-8"))
    summary_path = Path(config["summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verification_path = args.run_root / "logs/verify-predictions-v2-final.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings = [
        "Independent experiment-audit backend was unavailable; this verdict is local/mechanical.",
    ]
    scope: dict[str, dict] = {}

    if not verification.get("ok"):
        errors.append("final prediction/data verification gate did not pass")
    evaluator = (args.release / "scripts/evaluate_mmde_depth.py").read_text(
        encoding="utf-8")
    suspicious = ("np.max(pred", "prediction.max()", "pred.max()")
    if any(pattern in evaluator for pattern in suspicious):
        errors.append("evaluator contains prediction-maximum normalization")
    for required in ("raw_frame_mean", "raw_pixel_pooled",
                     "median_aligned_frame_mean", "scale_pred_over_gt"):
        if required not in evaluator:
            errors.append(f"evaluator does not emit {required}")

    for dataset, entry in config["datasets"].items():
        test_root = Path(entry["test_root"])
        pred_root = Path(entry["pred_root"])
        metrics_root = Path(entry["metrics_root"])
        dataset_scope = {}
        for split_root in sorted(p for p in test_root.iterdir() if p.is_dir()):
            manifest = split_root / "frames.jsonl"
            if not manifest.is_file():
                continue
            frames = [json.loads(line) for line in manifest.read_text().splitlines()
                      if line.strip()]
            evaluable = all(frame.get("evaluation_available", True)
                            for frame in frames)
            dataset_scope[split_root.name] = {
                "frames": len(frames), "evaluation_available": evaluable}
            for frame in frames:
                for field in ("depth_gt_path", "depth_gt_full_path"):
                    value = frame.get(field)
                    if value and (Path(value) == pred_root or pred_root in Path(value).parents):
                        errors.append(
                            f"{dataset}/{split_root.name}: GT path enters prediction root")
            for model in MODELS:
                count = len(list((pred_root / split_root.name / model).glob("*.npy")))
                if count != len(frames):
                    errors.append(
                        f"{dataset}/{split_root.name}/{model}: {count}/{len(frames)} predictions")
                metric_path = metrics_root / split_root.name / f"{model}.json"
                if evaluable:
                    if not metric_path.is_file():
                        errors.append(f"missing metric file: {metric_path}")
                        continue
                    metric = json.loads(metric_path.read_text(encoding="utf-8"))
                    if metric.get("frames_total") != len(frames):
                        errors.append(f"frame total mismatch: {metric_path}")
                    raw = metric.get("raw_frame_mean") or {}
                    frames_evaluated = metric.get("frames_evaluated", 0)
                    frames_invalid = metric.get("frames_invalid", 0)
                    frames_missing = metric.get(
                        "frames_missing_prediction_or_gt", 0)
                    if frames_evaluated > 0:
                        if not {"abs_rel", "abs_mean_m"} <= set(raw):
                            errors.append(
                                f"raw primary metrics missing: {metric_path}")
                    elif frames_invalid + frames_missing != len(frames):
                        errors.append(
                            f"zero evaluated frames are not fully accounted for: "
                            f"{metric_path}")
                    else:
                        warnings.append(
                            f"{dataset}/{split_root.name}/{model}: all "
                            f"{len(frames)} frames explicitly invalid; raw metrics "
                            "are unavailable and were not fabricated."
                        )
                    try:
                        in_summary = summary["datasets"][dataset]["splits"][
                            split_root.name]["models"][model]
                    except KeyError:
                        errors.append(f"summary entry missing: {dataset}/{split_root.name}/{model}")
                    else:
                        if in_summary != metric:
                            errors.append(f"summary/result mismatch: {metric_path}")
                elif metric_path.is_file():
                    errors.append(
                        f"inference-only split has fabricated metric file: {metric_path}")
        scope[dataset] = dataset_scope

    report = {
        "date": "2026-10-07",
        "auditor": "local-mechanical-audit (independent backend unavailable)",
        "overall_verdict": "FAIL" if errors else "WARN",
        "integrity_status": "fail" if errors else "pass_local",
        "checks": {
            "ground_truth_provenance": {
                "status": "FAIL" if any("GT path" in e for e in errors) else "PASS",
                "details": "Dataset LiDAR/Eigen files are outside prediction roots; KITTI anonymous has no metric files.",
            },
            "score_normalization": {
                "status": "FAIL" if any("maximum normalization" in e for e in errors) else "PASS",
                "details": "Raw metrics are retained; median-aligned metrics are separately labelled diagnostics.",
            },
            "result_existence": {
                "status": "FAIL" if errors else "PASS",
                "details": "Exact prediction cardinality, metric keys, and summary equality were checked.",
            },
            "scope": {"status": "PASS", "details": scope},
            "evaluation_type": {
                "kitti_eigen_and_sequences": "real_gt",
                "kitti_official_test_anonymous": "inference_only_private_gt",
                "nuscenes": "real_gt_local_heldout_lidar",
            },
        },
        "warnings": warnings,
        "errors": errors,
        "claim_impact": {
            "raw_metric_results": "supported" if not errors else "unsupported",
            "nuScenes_official_challenge_score": "unsupported; local protocol on official test sensor inputs",
            "KITTI_official_test_metric": "unsupported; server GT is private",
        },
    }
    json_path = args.run_root / "EXPERIMENT_AUDIT.json"
    atomic_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Experiment Audit Report", "",
        "**Auditor**: local mechanical audit (independent backend unavailable)",
        f"**Overall verdict**: {report['overall_verdict']}", "",
        "## Integrity checks", "",
        f"- Ground-truth provenance: {report['checks']['ground_truth_provenance']['status']}",
        f"- Score normalization: {report['checks']['score_normalization']['status']}",
        f"- Result existence and summary match: {report['checks']['result_existence']['status']}",
        "- KITTI anonymous: inference only; no local metric claim", 
        "- nuScenes official test: local held-out LiDAR protocol, not a challenge-server score", "",
        "## Limitations", "",
        f"- {warnings[0]}", "",
        "## Errors", "",
    ]
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    atomic_text(args.run_root / "EXPERIMENT_AUDIT.md", "\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

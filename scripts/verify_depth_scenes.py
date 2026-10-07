#!/usr/bin/env python3
"""Validate every generated Scene index and quantized point-cloud blob."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    indexes = blobs = fused_points = payload_bytes = 0
    for path in sorted(args.scene_root.glob("*/*/*/index.json")):
        index = json.loads(path.read_text(encoding="utf-8"))
        indexes += 1
        n_frames = index.get("n_frames")
        if n_frames not in (40, 50):
            errors.append(f"unexpected frame count: {path}: {n_frames}")
        for field in ("cam_pos", "cam_quat", "cam_t"):
            if len(index.get(field, [])) != n_frames:
                errors.append(f"{field} length mismatch: {path}")
        bbox_values = [value for side in index.get("bbox", {}).values()
                       for value in side]
        if len(bbox_values) != 6 or not all(math.isfinite(value)
                                            for value in bbox_values):
            errors.append(f"invalid bbox: {path}")
        counted = 0
        for group in index.get("groups", []):
            references = [(group["overview"], group["overview_n"])]
            references += [(chunk["file"], chunk["n"])
                           for chunk in group.get("chunks", [])]
            for name, expected_points in references:
                blob_path = path.parent / name
                try:
                    data = blob_path.read_bytes()
                except OSError:
                    errors.append(f"missing blob: {blob_path}")
                    continue
                actual_points = struct.unpack_from("<I", data)[0]
                expected_bytes = (4 + 3 * expected_points +
                                  ((3 * expected_points) & 1) +
                                  6 * expected_points)
                if actual_points != expected_points or len(data) != expected_bytes:
                    errors.append(
                        f"blob mismatch: {blob_path}: points "
                        f"{actual_points}/{expected_points}, bytes "
                        f"{len(data)}/{expected_bytes}")
                blobs += 1
                payload_bytes += len(data)
            counted += group.get("n_points", 0)
        if counted != index.get("n_points"):
            errors.append(f"index point count mismatch: {path}")
        fused_points += index.get("n_points", 0)
    report = {
        "ok": not errors,
        "errors": errors,
        "indexes": indexes,
        "blobs": blobs,
        "fused_points": fused_points,
        "payload_bytes": payload_bytes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

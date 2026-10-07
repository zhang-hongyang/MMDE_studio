#!/usr/bin/env python3
"""PTC-Depth sequence refinement using DAV2 inverse-depth observations.

Sequence starts are back-filled from the first two frames in reverse order so
every sequence frame has a scientifically valid two-view PTC estimate.
"""
from __future__ import annotations

import cv2
import numpy as np
import os
import sys

from common import (base_parser, finite_depth, grouped_frames, load_context,
                    prediction_path, save_prediction, valid_prediction)
from dav2 import build_dav2


def translation(frame: dict) -> np.ndarray | None:
    value = frame.get("T_world_camera")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    return matrix[:3, 3] if matrix.shape == (4, 4) else None


def image_and_inverse(frame: dict, dav2):
    image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame["image_path"])
    inverse = finite_depth(dav2.infer_image(image, input_size=518))
    return image, inverse.astype(np.float32)


def make_pipeline(frame: dict, image: np.ndarray):
    from ptc_depth import PTCDepth
    h, w = image.shape[:2]
    K = np.asarray(frame.get("K"), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("PTC-Depth requires a 3x3 K in every frame")
    return PTCDepth(H=h, W=w, fx=float(K[0, 0]), fy=float(K[1, 1]),
                    cx=float(K[0, 2]), cy=float(K[1, 2]))


def metric_baseline(left: dict, right: dict) -> float:
    a, b = translation(left), translation(right)
    return float(np.linalg.norm(b - a)) if a is not None and b is not None else 0.0


def adjacent_candidates(frame: dict) -> list[dict]:
    """Return same-drive temporal contexts in the builder's retry order."""
    candidates = []
    for item in frame.get("pair_candidates", []):
        if item.get("image_path") and item.get("T_world_camera") is not None:
            candidates.append({**frame, **item})
    if candidates:
        return candidates
    # Backward compatibility for already-built manifests.
    for prefix in ("prev", "next"):
        path = frame.get(f"{prefix}_image_path")
        pose = frame.get(f"{prefix}_T_world_camera")
        if path and pose is not None:
            candidates.append({**frame, "image_path": path,
                               "T_world_camera": pose})
    return candidates


def main() -> int:
    cv2.setNumThreads(int(os.environ.get("MMDE_OPENCV_THREADS", "1")))
    args = base_parser(__doc__).parse_args()
    frames, out = load_context(args)
    dav2 = build_dav2(args.device)

    groups = grouped_frames(frames)
    for group in groups[args.group_offset::args.group_stride]:
        if group and group[0].get("independent_pair"):
            for frame in group:
                target = prediction_path(out, frame)
                if valid_prediction(target):
                    continue
                # Eigen frames are independent and not ordered by drive time.
                # Try real same-drive contexts from smaller to larger baseline;
                # low-motion adjacent frames can make PTC return no depth.
                candidates = adjacent_candidates(frame)
                if not candidates:
                    raise ValueError(
                        "independent PTC frame requires an adjacent image and pose")
                image, inverse = image_and_inverse(frame, dav2)
                raw = None
                for adjacent in candidates:
                    adjacent_image, adjacent_inverse = image_and_inverse(adjacent, dav2)
                    pipeline = make_pipeline(adjacent, adjacent_image)
                    pipeline(adjacent_image, adjacent_inverse, 0.0)
                    result = pipeline(image, inverse,
                                      metric_baseline(adjacent, frame))
                    candidate = np.asarray(result["depth"], dtype=np.float32)
                    if np.isfinite(candidate).any():
                        raw = candidate
                        break
                if raw is None:
                    # Zero is an explicit invalid-depth sentinel.  Keeping one
                    # output per requested frame lets the viewer expose this
                    # genuine no-parallax failure while the evaluator counts
                    # the frame as invalid instead of silently dropping it.
                    print(f"PTC produced no depth for source frame "
                          f"{frame['_source_idx']}; writing invalid sentinel",
                          file=sys.stderr, flush=True)
                    raw = np.zeros(image.shape[:2], dtype=np.float32)
                save_prediction(out, frame, finite_depth(raw))
            continue
        pipeline = None
        previous_frame = None
        first_payload = None
        for index, frame in enumerate(group):
            image, inverse = image_and_inverse(frame, dav2)
            if pipeline is None:
                pipeline = make_pipeline(frame, image)
                first_payload = (image, inverse)
            baseline = (metric_baseline(previous_frame, frame)
                        if previous_frame is not None else 0.0)
            result = pipeline(image, inverse, baseline)
            raw = np.asarray(result["depth"], dtype=np.float32)
            if np.isfinite(raw).any() and not valid_prediction(prediction_path(out, frame)):
                save_prediction(out, frame, finite_depth(raw))
            elif (index > 0 and
                  not valid_prediction(prediction_path(out, frame))):
                print(f"PTC produced no depth for source frame "
                      f"{frame['_source_idx']}; writing invalid sentinel",
                      file=sys.stderr, flush=True)
                save_prediction(out, frame,
                                np.zeros(image.shape[:2], dtype=np.float32))
            previous_frame = frame

        first = group[0]
        first_path = prediction_path(out, first)
        if not valid_prediction(first_path) and len(group) >= 2:
            # Prime a fresh tracker on frame 1, then infer frame 0 with the
            # same physical baseline. This avoids a fabricated mono fallback.
            second = group[1]
            second_image, second_inverse = image_and_inverse(second, dav2)
            reverse = make_pipeline(second, second_image)
            reverse(second_image, second_inverse, 0.0)
            first_image, first_inverse = first_payload
            result = reverse(first_image, first_inverse,
                             metric_baseline(second, first))
            raw = np.asarray(result["depth"], dtype=np.float32)
            if not np.isfinite(raw).any():
                print(f"PTC reverse warm-up produced no depth for source "
                      f"frame {first['_source_idx']}; writing invalid sentinel",
                      file=sys.stderr, flush=True)
                raw = np.zeros(first_image.shape[:2], dtype=np.float32)
            save_prediction(out, first, finite_depth(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

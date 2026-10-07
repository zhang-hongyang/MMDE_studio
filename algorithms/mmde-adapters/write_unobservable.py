#!/usr/bin/env python3
"""Write explicit invalid sentinels when a method is physically unobservable."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from common import base_parser, load_context, prediction_path, save_prediction


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    frames, out = load_context(args)
    for frame in frames:
        path = prediction_path(out, frame)
        if path.is_file():
            continue
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame["image_path"])
        save_prediction(out, frame, np.zeros(image.shape[:2], dtype=np.float32))
    (out / "availability.json").write_text(
        json.dumps({
            "available": False,
            "reason": args.reason,
            "frames": len(frames),
            "representation": "zero_depth_invalid_sentinel",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

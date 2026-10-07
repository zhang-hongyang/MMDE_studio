#!/usr/bin/env python3
"""UniDepthV2 metric depth adapter for MMDE frames.jsonl datasets."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (base_parser, finite_depth, load_context, prediction_path,
                    resize_native, save_prediction, valid_prediction)

DEFAULT_MODEL = ("/home/ZhangHongyang/ResearchHub-scratch/models/"
                 "SpaceDepth-Completion/foundation-20260924/"
                 "unidepth-v2-vitl14")


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--pretrained", default=os.environ.get(
        "UNIDEPTH_V2_MODEL", DEFAULT_MODEL))
    ap.add_argument("--resolution-level", type=int, default=9)
    args = ap.parse_args()
    frames, out = load_context(args)
    pending = [f for f in frames if not valid_prediction(prediction_path(out, f))]
    if not pending:
        print(f"all {len(frames)} UniDepthV2 predictions already valid", flush=True)
        return 0

    from unidepth.models import UniDepthV2

    model_path = Path(args.pretrained).expanduser()
    pretrained = str(model_path.resolve()) if model_path.exists() else args.pretrained
    device = torch.device(args.device)
    model = UniDepthV2.from_pretrained(pretrained).to(device).eval()
    if hasattr(model, "resolution_level"):
        model.resolution_level = args.resolution_level

    for frame in pending:
        bgr = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).to(device).permute(2, 0, 1)
        with torch.inference_mode():
            result = model.infer(image)
        depth = result["depth"].detach().float().cpu().numpy().squeeze()
        depth = resize_native(finite_depth(depth), frame)
        save_prediction(out, frame, depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

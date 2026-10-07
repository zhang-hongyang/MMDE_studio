#!/usr/bin/env python3
"""MoGe-3 metric depth adapter for MMDE frames.jsonl datasets."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch

from common import (base_parser, finite_depth, load_context, prediction_path,
                    save_prediction, valid_prediction)

DEFAULT_MODEL = ("/home/ZhangHongyang/ResearchHub-scratch/models/"
                 "MMDE_studio/moge3/model.pt")


def main() -> int:
    ap = base_parser(__doc__)
    local_default = DEFAULT_MODEL if os.path.isfile(DEFAULT_MODEL) else \
        "Ruicheng/moge-3-vitl"
    ap.add_argument("--pretrained", default=os.environ.get(
        "MOGE3_MODEL", local_default))
    ap.add_argument("--resolution-level", type=int, default=7)
    ap.add_argument("--refine-steps", type=int, default=3)
    args = ap.parse_args()
    frames, out = load_context(args)
    pending = [f for f in frames if not valid_prediction(prediction_path(out, f))]
    if not pending:
        print(f"all {len(frames)} MoGe-3 predictions already valid", flush=True)
        return 0

    from moge.model.v3 import MoGeModel
    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    for frame in pending:
        bgr = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
        image = image.permute(2, 0, 1) / 255.0
        fov_x = None
        if frame.get("K"):
            fx = float(frame["K"][0][0])
            fov_x = math.degrees(2.0 * math.atan(rgb.shape[1] / (2.0 * fx)))
        with torch.inference_mode():
            result = model.infer(
                image, fov_x=fov_x, resolution_level=args.resolution_level,
                refine_steps=args.refine_steps, use_fp16=args.device == "cuda")
        depth = finite_depth(result["depth"].detach().float().cpu().numpy())
        save_prediction(out, frame, depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

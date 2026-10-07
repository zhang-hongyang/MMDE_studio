#!/usr/bin/env python3
"""MapAnything feed-forward metric depth adapter for MMDE sequences."""
from __future__ import annotations

import os

import numpy as np
import torch

from common import (base_parser, finite_depth, grouped_frames, load_context,
                    prediction_path, resize_native, save_prediction,
                    valid_prediction)

DEFAULT_MODEL = ("/home/ZhangHongyang/ResearchHub-scratch/models/"
                 "MMDE_studio/map-anything")


def main() -> int:
    ap = base_parser(__doc__)
    local_default = DEFAULT_MODEL if os.path.isdir(DEFAULT_MODEL) else \
        "facebook/map-anything"
    ap.add_argument("--pretrained", default=os.environ.get(
        "MAPANYTHING_MODEL", local_default))
    args = ap.parse_args()
    frames, out = load_context(args)

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = torch.device(args.device)
    model = MapAnything.from_pretrained(args.pretrained).to(device).eval()
    for group in grouped_frames(frames):
        if all(valid_prediction(prediction_path(out, f)) for f in group):
            print(f"skip complete sequence {group[0].get('seq_id')}", flush=True)
            continue
        views = load_images([str(f["image_path"]) for f in group])
        with torch.inference_mode():
            predictions = model.infer(
                views, memory_efficient_inference=True, minibatch_size=1,
                use_amp=args.device == "cuda", amp_dtype="bf16",
                apply_mask=True, mask_edges=True)
        if len(predictions) != len(group):
            raise RuntimeError(
                f"MapAnything returned {len(predictions)} views for {len(group)}")
        for frame, pred in zip(group, predictions):
            # Upstream returns B,H,W,1 for each view.
            depth = pred["depth_z"][0].squeeze(-1)
            depth = depth.detach().float().cpu().numpy()
            depth = resize_native(finite_depth(depth), frame)
            save_prediction(out, frame, depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

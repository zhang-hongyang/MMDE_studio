#!/usr/bin/env python3
"""LingBot-Map SDPA streaming-depth adapter for MMDE sequences."""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np
import torch

from common import (base_parser, finite_depth, grouped_frames, load_context,
                    prediction_path, resize_native, save_prediction,
                    valid_prediction)


DEFAULT_CHECKPOINT = ("/home/ZhangHongyang/ResearchHub-scratch/models/"
                      "MMDE_studio/lingbot-map/lingbot-map.pt")


def build_model(checkpoint: Path, device: torch.device):
    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=1024, kv_cache_sliding_window=64,
        kv_cache_scale_frames=2, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=True,
        camera_num_iterations=4)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"checkpoint keys: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    return model.to(device).eval()


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--checkpoint", default=os.environ.get(
        "LINGBOT_MAP_CHECKPOINT", DEFAULT_CHECKPOINT))
    args = ap.parse_args()
    frames, out = load_context(args)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"missing LingBot-Map checkpoint: {checkpoint}")

    from lingbot_map.utils.load_fn import load_and_preprocess_images
    device = torch.device(args.device)
    model = build_model(checkpoint, device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    for group in grouped_frames(frames):
        if all(valid_prediction(prediction_path(out, f)) for f in group):
            print(f"skip complete sequence {group[0].get('seq_id')}", flush=True)
            continue
        paths = [str(f["image_path"]) for f in group]
        images = load_and_preprocess_images(
            paths, mode="crop", image_size=518, patch_size=14).to(device)
        scale_frames = min(2, len(group))
        amp = (torch.amp.autocast("cuda", dtype=dtype)
               if args.device == "cuda" else nullcontext())
        with torch.inference_mode(), amp:
            pred = model.inference_streaming(
                images, num_scale_frames=scale_frames,
                keyframe_interval=1, output_device=torch.device("cpu"))
        depth = pred["depth"].detach().float().cpu().numpy()
        # Current upstream layout is B,N,H,W,1.  Also tolerate the older
        # B,N,1,H,W convention so a source update fails cleanly by shape.
        if depth.ndim == 5 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0, ...]
        if depth.ndim != 3 or depth.shape[0] != len(group):
            raise RuntimeError(f"LingBot depth shape {depth.shape}, "
                               f"expected {len(group)} frames")
        for frame, item in zip(group, depth):
            save_prediction(out, frame,
                            resize_native(finite_depth(item), frame))
        model.clean_kv_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

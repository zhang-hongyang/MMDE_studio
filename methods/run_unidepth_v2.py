#!/usr/bin/env python3
"""Run UniDepthV2 metric monocular depth through the MMDE registry."""
import argparse

from _launcher import finish, launch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--model-name", default="unidepth_v2")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    finish(launch(interpreter_key="unidepth_v2",
                  worker="infer_unidepth_v2.py",
                  dataset=args.dataset, split=args.split,
                  model_name=args.model_name, max_frames=args.max_frames,
                  device=args.device,
                  extra=["--frame-offset", str(args.frame_offset),
                         "--frame-stride", str(args.frame_stride)]))


if __name__ == "__main__":
    main()

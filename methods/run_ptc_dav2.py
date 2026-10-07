#!/usr/bin/env python3
"""Run PTC-Depth with Depth Anything V2 relative-depth inputs."""
import argparse

from _launcher import finish, launch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--group-offset", type=int, default=0)
    ap.add_argument("--group-stride", type=int, default=1)
    ap.add_argument("--model-name", default="ptc_dav2")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    finish(launch(interpreter_key="temporal", worker="infer_ptc_dav2.py",
                  dataset=args.dataset, split=args.split,
                  model_name=args.model_name, max_frames=args.max_frames,
                  device=args.device,
                  extra=["--group-offset", str(args.group_offset),
                         "--group-stride", str(args.group_stride)]))


if __name__ == "__main__":
    main()

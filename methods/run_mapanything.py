#!/usr/bin/env python3
"""Run MapAnything feed-forward metric reconstruction/depth in MMDE."""
import argparse

from _launcher import finish, launch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--model-name", default="mapanything")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    finish(launch(interpreter_key="mapanything",
                  worker="infer_mapanything.py", dataset=args.dataset,
                  split=args.split, model_name=args.model_name,
                  max_frames=args.max_frames, device=args.device))


if __name__ == "__main__":
    main()

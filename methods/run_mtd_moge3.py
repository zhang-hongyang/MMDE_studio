#!/usr/bin/env python3
"""Run MoGe-3 followed by MTD sparse-seed metric calibration.

This is an MMDE composition, not an upstream MTD-native backbone: MoGe-3
metric depth is converted to an inverse-depth proxy before MTD fitting.
"""
import argparse

from _launcher import finish, launch, output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--model-name", default="mtd_moge3")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    # Reusable canonical MoGe-3 predictions.  The worker skips finite existing
    # outputs, so rerunning the composition does not repeat expensive inference.
    code = launch(interpreter_key="moge3", worker="infer_moge3.py",
                  dataset=args.dataset, split=args.split, model_name="moge3",
                  max_frames=args.max_frames, device=args.device)
    if code:
        finish(code)
    backbone_dir = output_dir(args.dataset, args.split, "moge3")
    finish(launch(interpreter_key="temporal", worker="infer_mtd.py",
                  dataset=args.dataset, split=args.split,
                  model_name=args.model_name, max_frames=args.max_frames,
                  device=args.device,
                  extra=["--backbone", "moge3", "--backbone-dir",
                         str(backbone_dir)]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MMDE method: Marigold V2 (Log-stage2) monocular depth on a dataset split.

mmde/methods contract (executed by the MMDE-Studio task center, argv list):

    run_marigold_v2.py --dataset DATASET --split SPLIT [--max-frames N]
                       [--model-name NAME] [--device cuda|cpu]

This wrapper stays light (backend deps only: PyYAML + numpy). The heavy
inference runs as a subprocess in the algorithm's own interpreter:

    interpreter  algorithms/marigold-v2/scripts/infer.py
                 --image_dir <staged frames> --output_dir <tmp>
                 --modality depth        (native resolution)

Interpreter resolution order:
    1. $MARIGOLD_V2_PYTHON
    2. methods/interpreters.yaml -> "marigold_v2"   (machine-local config)
    3. sys.executable (fallback: backend env with marigoldv2 installed)

Output: one float32 depth map per frame in the MMDE preds tree
    {pred_root}/{split}/{model_name}/{idx:06d}.npy
where {idx} is the frame's line number in frames.jsonl. Predictions come
back at the staged image's own resolution (native-res mode), which matches
the dataset images because the wrapper stages the frames.jsonl images
themselves. Depth is relative/affine-ambiguous, as with every method here;
the per-frame affine alignment to sparse GT happens at serve time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

METHODS_DIR = Path(__file__).resolve().parent
STUDIO_ROOT = METHODS_DIR.parent
ALGO_ROOT = STUDIO_ROOT / "algorithms" / "marigold-v2"
INFER_SCRIPT = ALGO_ROOT / "scripts" / "infer.py"
INTERPRETERS = METHODS_DIR / "interpreters.yaml"


def resolve_interpreter() -> str:
    env = os.environ.get("MARIGOLD_V2_PYTHON")
    if env:
        return env
    if INTERPRETERS.is_file():
        data = yaml.safe_load(INTERPRETERS.read_text(encoding="utf-8")) or {}
        hit = data.get("marigold_v2")
        if hit:
            return str(hit)
    return sys.executable


def dataset_entry(dataset: str) -> dict:
    cfg = Path(os.environ.get("MMDE_STUDIO_DATASETS")
               or STUDIO_ROOT / "backend" / "config" / "datasets.yaml")
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    try:
        return doc["datasets"][dataset]
    except KeyError:
        raise SystemExit(f"dataset {dataset!r} not in {cfg}")


def load_frames(test_root: Path, split: str) -> list[dict]:
    frames = []
    with open(test_root / split / "frames.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="registry dataset name")
    ap.add_argument("--split", required=True, help="split name")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="smoke test: process only the first N frames")
    ap.add_argument("--model-name", default="marigold_v2",
                    help="prediction subdir under {pred_root}/{split}/")
    ap.add_argument("--device", default=None, choices=("cuda", "cpu"),
                    help="cpu forces CUDA_VISIBLE_DEVICES=''")
    args = ap.parse_args()

    entry = dataset_entry(args.dataset)
    test_root = Path(entry["test_root"])
    pred_root = Path(entry["pred_root"])
    frames = load_frames(test_root, args.split)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise SystemExit(f"no frames in {test_root}/{args.split}")

    interp = resolve_interpreter()
    out_dir = pred_root / args.split / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix=f"mmde_marigold_v2_{args.dataset}_"))
    stage_imgs = stage / "images"
    stage_imgs.mkdir(parents=True, exist_ok=True)
    for idx, fr in enumerate(frames):
        target = stage_imgs / f"{idx:06d}.jpg"
        if not target.exists():
            os.symlink(fr["image_path"], target)

    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [interp, str(INFER_SCRIPT), "--image_dir", str(stage_imgs),
           "--output_dir", str(stage / "out"), "--modality", "depth"]
    print(f"[marigold_v2] {len(frames)} frames -> {out_dir}", flush=True)
    print(f"[marigold_v2] {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=ALGO_ROOT, env=env)
    except OSError as exc:
        raise SystemExit(f"failed to launch interpreter {interp!r}: {exc}; "
                         f"set MARIGOLD_V2_PYTHON or methods/interpreters.yaml")
    if proc.returncode != 0:
        raise SystemExit(f"infer.py exited with {proc.returncode}")

    preds = {int(p.stem): p for p in
             (stage / "out").rglob("predictions_npy/*.npy")}
    missing = [idx for idx in range(len(frames)) if idx not in preds]
    if missing:
        raise SystemExit(f"missing predictions for frame idx: {missing}")
    for idx in range(len(frames)):
        arr = np.load(preds[idx]).astype(np.float32)
        np.save(out_dir / f"{idx:06d}.npy", arr)
        print(f"[marigold_v2] wrote {out_dir / f'{idx:06d}.npy'} "
              f"{arr.shape} range [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]",
              flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[marigold_v2] done: {len(frames)} frames -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

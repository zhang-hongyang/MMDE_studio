#!/usr/bin/env python3
"""MTD sparse-seed calibration with DAV2 or cached MoGe-3 predictions."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (base_parser, inverse_proxy, load_context, native_shape,
                    prediction_path, rasterize_sparse_depth, save_prediction,
                    valid_prediction)
from dav2 import build_dav2

DEFAULT_MTD_ROOT = Path(
    "/home/ZhangHongyang/ResearchHub-workspaces/MMDE_studio/code/current/"
    "algorithms/mtd")


class CachedMoGeBackbone:
    """Expose cached metric predictions through MTD's infer_image contract."""

    def __init__(self, prediction_dir: Path):
        self.prediction_dir = prediction_dir
        self.frame: dict | None = None

    def select(self, frame: dict) -> None:
        self.frame = frame

    def infer_image(self, _rgb: np.ndarray, input_size: int = 518) -> np.ndarray:
        del input_size
        if self.frame is None:
            raise RuntimeError("no frame selected")
        path = prediction_path(self.prediction_dir, self.frame)
        if not path.is_file():
            raise FileNotFoundError(f"missing MoGe-3 prediction: {path}")
        return inverse_proxy(np.load(path))


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--backbone", required=True, choices=("dav2", "moge3"))
    ap.add_argument("--backbone-dir")
    args = ap.parse_args()
    frames, out = load_context(args)

    mtd_root = Path(os.environ.get("MMDE_MTD_ROOT", DEFAULT_MTD_ROOT))
    sys.path.insert(0, str(mtd_root / "src"))
    from core.bilateral import RecursiveBilateralFilter
    from run_demo import run_pipeline

    cfg_path = mtd_root / "src" / "configs" / "kitti_dc.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_alg = cfg["algorithm"]
    cfg_alg["min_depth"] = 0.1
    cfg_alg["max_depth"] = 100.0
    # The exact full-release MTD implementation is not public yet.  These are
    # the upstream v0.1 settings, retained explicitly for reproducibility.
    cfg_alg["use_dadp"] = False

    if args.backbone == "dav2":
        backbone = build_dav2(args.device)
    else:
        if not args.backbone_dir:
            raise SystemExit("--backbone-dir is required for moge3")
        backbone = CachedMoGeBackbone(Path(args.backbone_dir))

    bf = cfg_alg.get("bilateral_filter", {})
    rbf = RecursiveBilateralFilter(
        iterations=bf.get("iterations", 1),
        spatial_sigma=bf.get("spatial_sigma", 0.1),
        color_sigma=bf.get("color_sigma", 0.01),
        kernel_size=bf.get("kernel_size", 7),
        min_depth=cfg_alg["min_depth"],
    ).to(args.device)

    for frame in frames:
        if valid_prediction(prediction_path(out, frame)):
            continue
        rgb = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(frame["image_path"])
        if isinstance(backbone, CachedMoGeBackbone):
            backbone.select(frame)
        sparse = rasterize_sparse_depth(frame, native_shape(frame))
        if np.count_nonzero(sparse) < 5:
            raise ValueError("MTD requires at least five sparse metric seeds")
        result = run_pipeline(rgb, sparse, cfg_alg, backbone, rbf,
                              args.device, input_size=518)
        save_prediction(out, frame, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

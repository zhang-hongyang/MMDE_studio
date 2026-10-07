"""Fast contract tests for the method-adapter data boundary."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (inverse_proxy, load_context, rasterize_sparse_depth,
                    save_prediction, valid_prediction)  # noqa: E402


class AdapterCommonTests(unittest.TestCase):
    def test_registry_indices_sparse_raster_and_atomic_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_value:
            tmp = Path(tmp_value)
            image = tmp / "frame.jpg"
            cv2.imwrite(str(image), np.zeros((4, 6, 3), dtype=np.uint8))
            sparse = tmp / "sparse.npz"
            np.savez(sparse,
                     uv=np.array([[1.2, 2.1], [1.4, 2.2], [5.0, 3.0]],
                                 dtype=np.float32),
                     depth=np.array([5.0, 3.0, 7.0], dtype=np.float32))
            test_root = tmp / "test"
            split = test_root / "tiny"
            split.mkdir(parents=True)
            frame = {"seq_id": "s", "image_path": str(image),
                     "depth_gt_path": str(sparse)}
            (split / "frames.jsonl").write_text(
                json.dumps(frame) + "\n", encoding="utf-8")
            cfg = tmp / "datasets.yaml"
            cfg.write_text(yaml.safe_dump({"datasets": {"d": {
                "test_root": str(test_root), "pred_root": str(tmp / "pred")
            }}}), encoding="utf-8")

            class Args:
                datasets_config = str(cfg)
                dataset = "d"
                split = "tiny"
                output_dir = str(tmp / "out")
                max_frames = None
                frame_offset = 0
                frame_stride = 1
                group_offset = 0
                group_stride = 1

            frames, out = load_context(Args)
            self.assertEqual(frames[0]["_source_idx"], 0)
            raster = rasterize_sparse_depth(frames[0], (4, 6))
            self.assertEqual(raster[2, 1], 3.0)  # nearest duplicate wins
            self.assertEqual(raster[3, 5], 7.0)
            path = save_prediction(out, frames[0], np.ones((4, 6)))
            self.assertTrue(valid_prediction(path))

    def test_uint16_sparse_scale_is_manifest_controlled(self):
        with tempfile.TemporaryDirectory() as tmp_value:
            path = Path(tmp_value) / "sparse.png"
            cv2.imwrite(str(path), np.array([[0, 256]], dtype=np.uint16))
            frame = {"sparse_depth_path": str(path),
                     "sparse_depth_scale": 256.0}
            value = rasterize_sparse_depth(frame, (1, 2))
            self.assertEqual(value[0, 1], 1.0)

    def test_inverse_proxy_is_finite_and_near_is_larger(self):
        proxy = inverse_proxy(np.array([[1.0, 2.0, np.inf, 0.0]],
                                       dtype=np.float32))
        self.assertTrue(np.isfinite(proxy).all())
        self.assertGreater(proxy[0, 0], proxy[0, 1])
        self.assertEqual(proxy[0, 2], 0.0)


if __name__ == "__main__":
    unittest.main()

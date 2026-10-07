"""Demo dataset: a single RGB image paired with a sparse depth map.

Sparse depth can be supplied as:

* A ``.npy`` file — float32 metres directly, or uint16 millimetres (auto-detected
  by dtype).  This is the recommended format for the bundled demo.
* A 16-bit PNG (uint16) — decoded according to ``sparse_format``:
  ``"mm_uint16"``    depth in mm → divide by 1000,
  ``"kitti_uint16"`` KITTI-style 1/256 metres.

``sparse_path`` is required; missing or non-existent paths raise an error.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import PIL.Image as pil


def load_demo_sample(
    image_path: str,
    sparse_path: str,
    sparse_format: str = "mm_uint16",
    gt_path: str = "",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load one demo sample.

    Args:
        image_path:        Path to an RGB image (any common format).
        sparse_path:       Path to sparse depth file (``.npy`` or PNG). Required.
        sparse_format:     Decoding hint for PNG files:
                           ``"mm_uint16"``    – uint16, depth in mm (/1000),
                           ``"kitti_uint16"`` – uint16, 1/256 metres.
                           Ignored when the file is a ``.npy``.
        gt_path:           Optional dense GT depth PNG for evaluation.

    Returns:
        rgb:          (H, W, 3) uint8.
        sparse_depth: (H, W) float32 metres (0 = invalid).
        gt_depth:     (H, W) float32 metres, or None.
    """
    # --- RGB ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # --- Sparse depth ---
    if not sparse_path:
        raise ValueError(
            "sparse_path is required. Set dataset.sparse_path in the config "
            "or pass --sparse to run_demo.py "
            "(e.g. assets/demo/example/sparse_depth.npy)."
        )
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(f"Sparse depth file not found: {sparse_path}")
    sparse_depth = _load_sparse(sparse_path, sparse_format, H, W)

    # --- GT (optional) ---
    gt_depth = None
    if gt_path and os.path.exists(gt_path):
        gt_depth = _load_gt_png(gt_path, H, W)

    return rgb, sparse_depth.astype(np.float32), gt_depth


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_sparse(path: str, fmt: str, H: int, W: int) -> np.ndarray:
    """Load sparse depth from either a .npy or a PNG file."""
    if path.lower().endswith('.npy'):
        raw = np.load(path)
        # uint16 → assume millimetres; float32 → assume metres already
        if raw.dtype == np.uint16:
            dep = raw.astype(np.float32) / 1000.0
        else:
            dep = raw.astype(np.float32)
        if dep.shape != (H, W):
            dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
        return dep

    # PNG path
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Sparse depth not found: {path}")
    if raw.ndim == 3:
        # Multi-channel PNG is unexpected for a depth map.
        # If all channels are equal (greyscale stored as BGR) take channel 0;
        # otherwise raise a clear error so the user knows to fix the file.
        if np.allclose(raw[..., 0], raw[..., 1]) and np.allclose(raw[..., 0], raw[..., 2]):
            raw = raw[..., 0]
        else:
            raise ValueError(
                f"Sparse depth PNG has {raw.shape[2]} channels: {path}\n"
                "Expected a single-channel (greyscale) 16-bit depth PNG.\n"
                "Use a .npy file or regenerate the depth PNG."
            )
    if fmt == "mm_uint16":
        dep = raw.astype(np.float32) / 1000.0
    elif fmt == "kitti_uint16":
        dep = raw.astype(np.float32) / 256.0
    else:
        raise ValueError(f"Unknown sparse_format: {fmt!r}")
    if dep.shape != (H, W):
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
    return dep


def _load_gt_png(path: str, H: int, W: int) -> Optional[np.ndarray]:
    try:
        dep = np.array(pil.open(path)).astype(np.float32)
        if dep.max() > 1000:
            dep /= 1000.0
        return cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return None


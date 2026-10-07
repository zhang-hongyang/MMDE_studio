"""Superpixel / over-segmentation backends for the MTD pipeline."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_superpixels(rgb: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Run the superpixel method selected in the JSON ``segmentation`` block.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        cfg: ``algorithm.segmentation`` dict from the config file.

    Returns:
        (H, W) int32 label map; each connected superpixel shares one ID.

    Supported ``method`` values:
        - ``felzenszwalb``  (default) — scikit-image graph-based segmentation
        - ``fast_slic``     — [fast-slic](https://github.com/Algy/fast-slic) SLIC (CPU, very fast)
        - ``opencv_lsc``    — OpenCV ``ximgproc`` Linear Spectral Clustering (needs opencv-contrib)
    """
    method = cfg.get('method', 'felzenszwalb').lower()
    if method == 'felzenszwalb':
        return _felzenszwalb(rgb, cfg)
    if method == 'fast_slic':
        return _fast_slic(rgb, cfg)
    if method == 'opencv_lsc':
        return _opencv_lsc(rgb, cfg)
    raise ValueError(
        f"Unknown segmentation.method: {method!r}. "
        "Choose 'felzenszwalb', 'fast_slic', or 'opencv_lsc'."
    )


def _subcfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Method-specific sub-dict; fall back to top-level keys for backward compatibility."""
    sub = cfg.get(name)
    if isinstance(sub, dict):
        merged = dict(cfg)
        merged.update(sub)
        return merged
    return cfg


def _felzenszwalb(rgb: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    from skimage import segmentation

    c = _subcfg(cfg, 'felzenszwalb')
    return segmentation.felzenszwalb(
        rgb,
        scale    = c.get('scale', 100),
        sigma    = c.get('sigma', 0.5),
        min_size = c.get('min_size', 50),
    ).astype(np.int32)


def _fast_slic(rgb: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    c = _subcfg(cfg, 'fast_slic')
    num_components  = int(c.get('num_components', 1600))
    compactness     = float(c.get('compactness', 10))
    min_size_factor = float(c.get('min_size_factor', 0))
    use_avx2        = bool(c.get('use_avx2', True))

    try:
        if use_avx2:
            from fast_slic.avx2 import SlicAvx2
            slic = SlicAvx2(
                num_components=num_components,
                compactness=compactness,
                min_size_factor=min_size_factor,
            )
        else:
            from fast_slic import Slic
            slic = Slic(
                num_components=num_components,
                compactness=compactness,
                min_size_factor=min_size_factor,
            )
    except ImportError as e:
        raise ImportError(
            "fast_slic is required for segmentation.method='fast_slic'. "
            "Install with:  pip install fast_slic"
        ) from e

    labels = slic.iterate(rgb)
    return np.asarray(labels, dtype=np.int32)


def _opencv_lsc(rgb: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    import cv2

    if not hasattr(cv2, 'ximgproc'):
        raise ImportError(
            "opencv-contrib-python is required for segmentation.method='opencv_lsc'. "
            "Install with:  pip install opencv-contrib-python"
        )

    c = _subcfg(cfg, 'opencv_lsc')
    region_size = int(c.get('region_size', 20))
    ratio       = float(c.get('ratio', 0.075))
    iterations  = int(c.get('iterations', 1))

    # OpenCV LSC expects BGR uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sp  = cv2.ximgproc.createSuperpixelLSC(bgr, region_size=region_size, ratio=ratio)
    sp.iterate(iterations)
    return sp.getLabels().astype(np.int32)

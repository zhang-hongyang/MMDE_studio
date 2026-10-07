"""Preprocessing utilities: sparse depth outlier removal and morphological kernels."""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Morphological kernels used for neighbourhood-based outlier detection
# ---------------------------------------------------------------------------

DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)


def outlier_removal(lidar: np.ndarray, threshold: float = 1.0):
    """Remove isolated LiDAR outliers using multi-scale diamond neighbourhood averaging.

    A point is flagged as an outlier if it is above the local average by more
    than *threshold* metres at *all three* neighbourhood scales (7, 9, 13).

    Args:
        lidar: 2-D float32 sparse depth map (metres, 0 = invalid).
        threshold: Depth difference threshold in metres.

    Returns:
        lidar_cleared: Sparse depth with outliers set to zero.
        potential_outliers: Float map where non-zero indicates outlier location.
    """
    sparse_lidar = lidar.astype(np.float32)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float32)

    def _local_avg(kernel):
        total = cv2.filter2D(sparse_lidar, -1, kernel.astype(np.float32))
        count = cv2.filter2D(valid_pixels, -1, kernel.astype(np.float32))
        return total / (count + 1e-5)

    avg7  = _local_avg(DIAMOND_KERNEL_7)
    avg9  = _local_avg(DIAMOND_KERNEL_9)
    avg13 = _local_avg(DIAMOND_KERNEL_13)

    out7  = ((sparse_lidar - avg7)  > threshold).astype(np.float32)
    out9  = ((sparse_lidar - avg9)  > threshold).astype(np.float32)
    out13 = ((sparse_lidar - avg13) > threshold).astype(np.float32)

    potential_outliers = out7 + out9 + out13
    lidar_cleared = (sparse_lidar * (1.0 - potential_outliers)).astype(np.float32)
    lidar_cleared[lidar_cleared < 0] = 0.0
    return lidar_cleared, potential_outliers

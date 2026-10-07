"""KITTI Depth Completion dataset loader.

Supports the **official KITTI DC validation split** (val_selection_cropped).

Directory layout expected under ``data_path``::

    val_selection_cropped/
    ├── image/
    │   └── 2011_09_26_drive_0002_sync_image_0000000005_image_02.png
    ├── velodyne_raw/
    │   └── 2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png
    └── groundtruth_depth/
        └── 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png

Both the sparse velodyne_raw and groundtruth_depth PNGs use the standard
KITTI 16-bit encoding: divide by 256 to get depth in metres.
"""

import os
from collections import Counter
from typing import Optional, Tuple

import cv2
import numpy as np
import PIL.Image as pil
import torch
from torch.utils.data import Dataset

from core.preprocess import outlier_removal


# ---------------------------------------------------------------------------
# KITTI calibration & projection helpers
# ---------------------------------------------------------------------------

def _read_calib_file(path: str) -> dict:
    """Parse a KITTI calibration text file into a dict of numpy arrays."""
    float_chars = set("0123456789.e+- ")
    data: dict = {}
    with open(path, 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split())))
                except ValueError:
                    pass
    return data


def _sub2ind(matrix_size, row_sub, col_sub):
    m, n = matrix_size
    return row_sub * (n - 1) + col_sub - 1


def generate_depth_map(calib_dir: str, velo_filename: str, cam: int = 2) -> np.ndarray:
    """Project raw velodyne points onto the camera image plane.

    Args:
        calib_dir:     Path to the KITTI calibration folder for the drive.
        velo_filename: Path to the ``.bin`` velodyne scan.
        cam:           Camera index (default 2 = left colour camera).

    Returns:
        Float32 depth map in metres, shape (H, W).
    """
    cam2cam  = _read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = _read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    V2C = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    V2C = np.vstack((V2C, [0, 0, 0, 1.0]))

    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    R_rect = np.eye(4)
    R_rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect  = cam2cam[f'P_rect_0{cam}'].reshape(3, 4)
    P_v2im  = P_rect @ R_rect @ V2C

    points = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0
    points = points[points[:, 0] >= 0, :]           # keep points in front

    pts_im = (P_v2im @ points.T).T                  # (N, 3) homogeneous
    pts_im[:, :2] /= pts_im[:, 2:3]

    pts_im[:, 0] = np.round(pts_im[:, 0]) - 1
    pts_im[:, 1] = np.round(pts_im[:, 1]) - 1
    in_bounds = (
        (pts_im[:, 0] >= 0) & (pts_im[:, 1] >= 0) &
        (pts_im[:, 0] < im_shape[1]) & (pts_im[:, 1] < im_shape[0])
    )
    pts_im = pts_im[in_bounds]

    depth = np.zeros(im_shape[:2], dtype=np.float32)
    depth[pts_im[:, 1].astype(int), pts_im[:, 0].astype(int)] = pts_im[:, 2]

    inds = _sub2ind(depth.shape, pts_im[:, 1], pts_im[:, 0])
    dupes = [item for item, cnt in Counter(inds).items() if cnt > 1]
    for dd in dupes:
        idx  = np.where(inds == dd)[0]
        x_l  = int(pts_im[idx[0], 0])
        y_l  = int(pts_im[idx[0], 1])
        depth[y_l, x_l] = pts_im[idx, 2].min()

    depth[depth < 0] = 0
    return depth


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class KITTIDCDataset(Dataset):
    """KITTI Depth Completion val_selection_cropped dataset.

    Args:
        data_path:         Root of ``val_selection_cropped``.
        remove_outliers:   Whether to apply outlier removal on sparse depth.
        do_flip:           Horizontal flip augmentation.
        height / width:    If given, resize outputs to this resolution.
    """

    def __init__(
        self,
        data_path: str,
        remove_outliers: bool = True,
        do_flip:         bool = False,
        height:          Optional[int] = None,
        width:           Optional[int] = None,
    ):
        self.data_path       = os.path.expanduser(data_path)
        self.remove_outliers = remove_outliers
        self.do_flip         = do_flip
        self.height          = height
        self.width           = width

        image_dir = os.path.join(self.data_path, "image")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"KITTI image dir not found: {image_dir}")
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Return (rgb, sparse_depth, gt_depth, filename).

        All arrays are (H, W) or (H, W, 3) float32 in metres.
        gt_depth is zero-filled if not available.
        """
        fname = self.filenames[idx]
        rgb   = self._load_image(fname)
        sp    = self._load_sparse(fname)
        gt    = self._load_gt(fname)

        if self.remove_outliers and sp is not None:
            sp, _ = outlier_removal(sp)
            sp    = sp.clip(0)

        if sp is None:
            H, W = rgb.shape[:2]
            sp   = np.zeros((H, W), dtype=np.float32)
        if gt is None:
            H, W = rgb.shape[:2]
            gt   = np.zeros((H, W), dtype=np.float32)

        return rgb, sp, gt, fname

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_image(self, fname: str) -> np.ndarray:
        path = os.path.join(self.data_path, "image", fname)
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.do_flip:
            img = np.fliplr(img).copy()
        return img.astype(np.uint8)

    def _load_sparse(self, fname: str) -> Optional[np.ndarray]:
        """Load velodyne_raw 16-bit PNG → float32 metres."""
        velo_name = fname.replace('_image_', '_velodyne_raw_', 1)
        path = os.path.join(self.data_path, "velodyne_raw", velo_name)
        if not os.path.exists(path):
            return None
        dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if dep is None:
            return None
        dep = dep.astype(np.float32) / 256.0
        if self.do_flip:
            dep = np.fliplr(dep).copy()
        return dep

    def _load_gt(self, fname: str) -> Optional[np.ndarray]:
        """Load groundtruth_depth 16-bit PNG → float32 metres."""
        gt_name = fname.replace('_image_', '_groundtruth_depth_', 1)
        path = os.path.join(self.data_path, "groundtruth_depth", gt_name)
        if not os.path.exists(path):
            return None
        try:
            dep = np.array(pil.open(path)).astype(np.float32) / 256.0
        except Exception:
            return None
        if self.do_flip:
            dep = np.fliplr(dep).copy()
        return dep

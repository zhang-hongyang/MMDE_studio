"""Depth-completion evaluation metrics (MAE only)."""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


def calculate_depth_error(
    estimated_depth,
    gt_depth,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> dict:
    """Compute MAE against ground-truth depth.

    Only pixels where *gt_depth* > 0 are evaluated.

    Returns:
        ``{'mae': <float>}`` — mean absolute error in metres.
    """
    if isinstance(estimated_depth, torch.Tensor):
        estimated_depth = estimated_depth.cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()

    valid = gt_depth > 0
    pred  = np.clip(estimated_depth[valid].astype(np.float64), min_depth, max_depth)
    tgt   = np.clip(gt_depth[valid].astype(np.float64),        min_depth, max_depth)

    mae = mean_absolute_error(tgt, pred)
    return {'mae': mae}


METRIC_NAMES = ['mae']


def print_metrics(errors: list, prefix: str = ""):
    """Print mean MAE over a list of per-image error dicts."""
    mean_mae = np.mean([e['mae'] for e in errors])
    label = f"{prefix}: " if prefix else ""
    print(f"{label}MAE = {mean_mae:.6f}")
    return {'mae': mean_mae}

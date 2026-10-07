"""Segment-based disparity fitting and graph propagation.

Core idea
---------
A monocular depth model predicts relative (affine-invariant) disparity.  For
each Felzenszwalb segment that contains at least *min_count* sparse LiDAR
points we fit an affine mapping

    sparse_disp = a_i * model_disp + b_i

using ordinary least-squares.  Segments that receive no sparse measurements
can optionally be filled in via **graph propagation**: we build a k-NN graph
over segment centroids and propagate the per-segment scale factor T =
median(sparse_disp) / median(model_disp) from valid to invalid segments.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Disparity conversion helpers
# ---------------------------------------------------------------------------

def convert_sparse_depth_to_disp(sparse_depth: np.ndarray, min_depth: float) -> np.ndarray:
    """Convert a sparse metric depth map to sparse disparity (numpy).

    disp = min_depth / depth  for valid (depth > 0) pixels, 0 elsewhere.
    """
    return np.where(sparse_depth > 1e-6, min_depth / (sparse_depth + 1e-6), 0.0).astype(np.float32)


def convert_sparse_depth_to_disp_tensor(
    sparse_depth: np.ndarray, min_depth: float
) -> torch.Tensor:
    """Convert a sparse metric depth map to sparse disparity (PyTorch tensor)."""
    disp = np.where(sparse_depth > 1e-6, min_depth / (sparse_depth + 1e-6), 0.0)
    return torch.tensor(disp, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-segment affine fitting (GPU-parallel)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fit_disp_to_sparse_disp_relation_parallel(
    model_disp: torch.Tensor,
    sparse_disp: torch.Tensor,
    segments: torch.Tensor,
    *,
    min_count: int = 3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a per-segment affine mapping  y = a*x + b  in a single GPU pass.

    Args:
        model_disp: Normalised model disparity map  (H, W).
        sparse_disp: Sparse LiDAR disparity map     (H, W), 0 = invalid.
        segments:   Integer segment label map        (H, W).
        min_count:  Minimum valid pixels to fit a segment.
        eps:        Numerical stability floor.

    Returns:
        segment_ids: 1-D tensor of fitted segment IDs.
        a:           Slope  for each fitted segment.
        b:           Intercept for each fitted segment.
        global_valid_mask: Boolean mask of length max_seg_id+1.
    """
    assert model_disp.shape == sparse_disp.shape == segments.shape
    device = model_disp.device

    flat_md  = model_disp.reshape(-1).float()
    flat_sd  = sparse_disp.reshape(-1).float()
    flat_seg = segments.reshape(-1).long()

    valid_pix = flat_sd > 0
    flat_md  = flat_md[valid_pix]
    flat_sd  = flat_sd[valid_pix]
    flat_seg = flat_seg[valid_pix]

    segment_ids, inv = torch.unique(flat_seg, return_inverse=True)
    K = segment_ids.numel()
    if K == 0:
        n_glob = int(segments.max().item()) + 1
        empty = torch.empty(0, device=device)
        return (segment_ids, empty, empty,
                torch.zeros(n_glob, dtype=torch.bool, device=device))

    ones       = torch.ones_like(flat_md)
    seg_n      = torch.zeros(K, device=device).scatter_add_(0, inv, ones)
    seg_sum_x  = torch.zeros(K, device=device).scatter_add_(0, inv, flat_md)
    seg_sum_y  = torch.zeros(K, device=device).scatter_add_(0, inv, flat_sd)
    seg_sum_xx = torch.zeros(K, device=device).scatter_add_(0, inv, flat_md * flat_md)
    seg_sum_xy = torch.zeros(K, device=device).scatter_add_(0, inv, flat_md * flat_sd)

    denom  = seg_n * seg_sum_xx - seg_sum_x * seg_sum_x
    valid  = (seg_n >= min_count) & (denom.abs() > eps)

    a = torch.zeros(K, device=device)
    b = torch.zeros(K, device=device)
    a[valid] = (seg_n[valid] * seg_sum_xy[valid] - seg_sum_x[valid] * seg_sum_y[valid]) / denom[valid]
    b[valid] = (seg_sum_y[valid] * seg_sum_xx[valid] - seg_sum_x[valid] * seg_sum_xy[valid]) / denom[valid]

    n_glob = int(segments.max().item()) + 1
    global_valid_mask = torch.zeros(n_glob, dtype=torch.bool, device=device)
    global_valid_mask[segment_ids[valid]] = True

    return segment_ids[valid], a[valid], b[valid], global_valid_mask


@torch.no_grad()
def apply_disp_affine_parallel(
    model_disp: torch.Tensor,
    segments: torch.Tensor,
    segment_ids: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    n_segments_global: Optional[int] = None,
) -> torch.Tensor:
    """Apply per-segment affine mapping y = a*x + b in O(H*W) via look-up table.

    Args:
        model_disp:         (H, W) normalised model disparity.
        segments:           (H, W) integer segment labels.
        segment_ids / a / b: output of :func:`fit_disp_to_sparse_disp_relation_parallel`.
        n_segments_global:  If None, inferred from segments.

    Returns:
        estimated_disp: (H, W) float32, 0 for segments with no valid fit.
    """
    device = model_disp.device
    if n_segments_global is None:
        n_segments_global = int(segments.max().item()) + 1

    a_lut = torch.zeros(n_segments_global, device=device, dtype=torch.float32)
    b_lut = torch.zeros(n_segments_global, device=device, dtype=torch.float32)
    a_lut[segment_ids] = a
    b_lut[segment_ids] = b

    seg_idx = segments.long()
    return a_lut[seg_idx] * model_disp + b_lut[seg_idx]


# ---------------------------------------------------------------------------
# Graph propagation for segments without sparse coverage
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_segment_median_ratio(
    model_disp: torch.Tensor,
    sparse_disp: torch.Tensor,
    segments: torch.Tensor,
    global_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-segment scale T = median(sparse_disp) / median(model_disp).

    Used as a simpler alternative to affine fitting for the propagation step.

    Returns:
        T:               Scale tensor of length n_segments.
        valid_segments:  Boolean mask of length n_segments (True = has sparse pts).
    """
    device = model_disp.device
    n_seg  = int(segments.max().item()) + 1

    flat_md  = model_disp.reshape(-1)
    flat_sd  = sparse_disp.reshape(-1)
    flat_seg = segments.reshape(-1)
    valid_px = flat_sd > 0

    # Model disparity median per segment
    sort_seg, sort_idx = torch.sort(flat_seg)
    sort_md = flat_md[sort_idx]
    seg_changes = torch.cat([
        torch.tensor([True], device=device),
        sort_seg[1:] != sort_seg[:-1],
        torch.tensor([True], device=device),
    ])
    bounds = torch.where(seg_changes)[0]
    seg_med_md = torch.zeros(n_seg, device=device)
    for i in range(len(bounds) - 1):
        s, e = bounds[i].item(), bounds[i + 1].item()
        sid  = sort_seg[s].item()
        seg_med_md[sid] = torch.median(sort_md[s:e])

    # Sparse disparity median per segment (valid pixels only)
    vseg = flat_seg[valid_px]
    vsd  = flat_sd[valid_px]
    sort_vseg, sort_vidx = torch.sort(vseg)
    sort_vsd = vsd[sort_vidx]
    seg_changes_v = torch.cat([
        torch.tensor([True], device=device),
        sort_vseg[1:] != sort_vseg[:-1],
        torch.tensor([True], device=device),
    ])
    bounds_v = torch.where(seg_changes_v)[0]
    seg_med_sd = torch.full((n_seg,), float('nan'), device=device)
    for i in range(len(bounds_v) - 1):
        s, e = bounds_v[i].item(), bounds_v[i + 1].item()
        sid  = sort_vseg[s].item()
        seg_med_sd[sid] = torch.median(sort_vsd[s:e])

    T = torch.where(
        global_valid_mask,
        seg_med_sd / (seg_med_md + 1e-8),
        torch.ones(n_seg, device=device),
    )
    return T, global_valid_mask


@torch.no_grad()
def build_segment_graph(
    segments: torch.Tensor,
    valid_segments: torch.Tensor,
    k_neighbors: int = 4,
    tau: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a k-NN graph over segment centroids for T-value propagation.

    Args:
        segments:       (H, W) integer segment labels.
        valid_segments: Boolean mask of length n_segments.
        k_neighbors:    Number of nearest neighbours per segment.
        tau:            Bandwidth for Gaussian edge weights (auto if None).

    Returns:
        topk_idx:   (n_segments, k) neighbour indices.
        norm_weights: (n_segments, k) normalised Gaussian weights.
    """
    device     = segments.device
    n_seg      = valid_segments.shape[0]
    H, W       = segments.shape

    y_coord, x_coord = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    flat_seg = segments.reshape(-1)
    seg_sum_x = torch.zeros(n_seg, device=device).scatter_add_(0, flat_seg, x_coord.reshape(-1))
    seg_sum_y = torch.zeros(n_seg, device=device).scatter_add_(0, flat_seg, y_coord.reshape(-1))
    seg_cnt   = torch.bincount(flat_seg, minlength=n_seg).float()
    centers   = torch.stack([
        seg_sum_x / (seg_cnt + 1e-8),
        seg_sum_y / (seg_cnt + 1e-8),
    ], dim=1)  # (n_seg, 2)

    dist_mat = torch.cdist(centers, centers)  # (n_seg, n_seg)
    dist_mat.masked_fill_(~valid_segments.unsqueeze(0), float('inf'))
    dist_mat.fill_diagonal_(float('inf'))

    k_actual = min(k_neighbors, n_seg - 1)
    topk_dist, topk_idx = torch.topk(dist_mat, k=k_actual, dim=1, largest=False)

    if tau is None:
        tau = torch.median(topk_dist, dim=1).values.unsqueeze(1).clamp(min=1e-3)
    weights = torch.exp(-topk_dist / tau)
    weights = torch.nan_to_num(weights, nan=0.0)

    norm_w = weights / (weights.sum(dim=1, keepdim=True) + 1e-15)
    return topk_idx, norm_w



@torch.no_grad()
def propagate_segment_values_iterative(
    vals: torch.Tensor,
    valid_segments: torch.Tensor,
    graph_idx: torch.Tensor,
    graph_weights: torch.Tensor,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    max_iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iteratively fill invalid segment values by diffusing from valid neighbours.

    At each iteration, segments that have at least one valid neighbour in the
    graph receive a weighted-average of those neighbours' values.  The process
    repeats until all reachable segments are filled or *max_iters* is reached.

    Args:
        vals:           Per-segment value tensor  (n_segments,).
        valid_segments: Boolean mask              (n_segments,); True = already valid.
        graph_idx:      (n_segments, k) neighbour indices.
        graph_weights:  (n_segments, k) normalised Gaussian weights.
        vmin / vmax:    Optional clamp range for the final values.
        max_iters:      Maximum number of diffusion steps.

    Returns:
        updated_vals:  Filled value tensor (n_segments,).
        updated_valid: Updated validity mask (n_segments,).
    """
    cur_vals  = vals.clone()
    cur_valid = valid_segments.clone()

    for iter1 in range(max_iters):
        
        source = cur_vals.clone()
        source[~cur_valid] = 0.0

        nbr_vals  = source[graph_idx]                        # (N, k)
        nbr_valid = cur_valid[graph_idx].float()             # (N, k)

        eff_w   = graph_weights * nbr_valid                  # (N, k)
        support = eff_w.sum(dim=1, keepdim=True)             # (N, 1)
        norm_w  = eff_w / support.clamp_min(1e-8)
        new_v   = (nbr_vals * norm_w).sum(dim=1)             # (N,)

        update_mask = (~cur_valid) & (support.squeeze(1) > 1e-8)
        if not update_mask.any():
            break

        # print(f"Iteration {iter1} of {max_iters}")

        cur_vals[update_mask]  = new_v[update_mask]
        cur_valid[update_mask] = True

    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else -1e9
        hi = vmax if vmax is not None else  1e9
        cur_vals = torch.clamp(cur_vals, lo, hi)

    return cur_vals, cur_valid


@torch.no_grad()
def propagate_T_values(
    T: torch.Tensor,
    valid_segments: torch.Tensor,
    graph_idx: torch.Tensor,
    graph_weights: torch.Tensor,
    clamp_range: Tuple[float, float] = (0.05, 20.0),
) -> torch.Tensor:
    """Fill invalid segment T-values via one step of weighted-neighbour averaging.

    Args:
        T:              Per-segment scale tensor (n_segments,).
        valid_segments: Boolean mask (n_segments,).
        graph_idx:      (n_segments, k) neighbour indices.
        graph_weights:  (n_segments, k) normalised weights.
        clamp_range:    (min, max) clamp for the output T.

    Returns:
        Updated T tensor with invalid segments filled from neighbours.
    """
    source_T = T.clone()
    source_T[~valid_segments] = 0.0

    neighbor_T = source_T[graph_idx]               # (n_seg, k)
    new_T      = (neighbor_T * graph_weights).sum(dim=1)
    updated_T  = torch.where(valid_segments, T, new_T)
    return torch.clamp(updated_T, *clamp_range)


# ---------------------------------------------------------------------------
# Numpy backend — per-segment affine fitting (slower, sklearn OLS)
# ---------------------------------------------------------------------------

def fit_disp_to_sparse_disp_relation_numpy(
    model_disp: np.ndarray,
    sparse_disp: np.ndarray,
    segments: np.ndarray,
    min_count: int = 3,
) -> Dict[int, Tuple[float, float]]:
    """Per-segment OLS affine fit using sklearn (numpy / CPU).

    Replicates the original benchmark implementation exactly.

    Args:
        model_disp:  (H, W) normalised model disparity (0–1).
        sparse_disp: (H, W) sparse LiDAR disparity (0 = invalid).
        segments:    (H, W) integer segment labels.
        min_count:   Minimum valid pixels required to fit a segment.

    Returns:
        Dict mapping segment_id → (slope a, intercept b).
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for the numpy segment backend. "
                          "Install it with: pip install scikit-learn")

    params: Dict[int, Tuple[float, float]] = {}
    for seg_id in np.unique(segments):
        mask  = segments == seg_id
        md    = model_disp[mask]
        sd    = sparse_disp[mask]
        valid = sd > 0
        md, sd = md[valid], sd[valid]
        if len(md) < min_count:
            continue
        reg = LinearRegression().fit(md.reshape(-1, 1), sd)
        params[seg_id] = (float(reg.coef_[0]), float(reg.intercept_))
    return params


def apply_disp_affine_numpy(
    model_disp: np.ndarray,
    segments: np.ndarray,
    params: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """Apply per-segment affine mapping from :func:`fit_disp_to_sparse_disp_relation_numpy`.

    Segments not present in *params* are left at 0 (unfitted).

    Returns:
        estimated_disp: (H, W) float32, 0 for unfitted segments.
    """
    result = np.zeros_like(model_disp, dtype=np.float32)
    for seg_id, (a, b) in params.items():
        mask = segments == seg_id
        result[mask] = a * model_disp[mask] + b
    return result

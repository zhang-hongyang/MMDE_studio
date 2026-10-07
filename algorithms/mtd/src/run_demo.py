"""Single-image demo inference script.

Usage
-----
    python run_demo.py --config configs/demo.json

    # Or override image / sparse depth directly:
    python run_demo.py --config configs/demo.json \
                       --image  path/to/image.jpg \
                       --sparse path/to/sparse.png

The script runs the full depth-completion pipeline on one image and saves the
metric depth map (16-bit PNG, millimetres) to the output directory.

Feature switches (in the JSON config under "algorithm"):

    use_segment_propagation  (default true)
        Propagate scale factors to segments with no LiDAR coverage via a
        k-NN segment graph.

    use_dadp  (default false)
        Apply Discontinuity-Aware Dynamic Programming (DADP) after the
        bilateral filter for sharper depth boundaries.

    bilateral_filter.use_anchor  (default true)
        When smoothing a/b maps, keep LiDAR-fitted segment pixels fixed as
        RBF anchors.  Set false to diffuse from source_mask only (no anchor).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils.paths import resolve_checkpoint, resolve_from_root, setup_dav2_import

setup_dav2_import()

from core.bilateral import RecursiveBilateralFilter
from core.dadp import run_dadp
from core.metrics import calculate_depth_error
from core.segment import (
    apply_disp_affine_numpy,
    apply_disp_affine_parallel,
    build_segment_graph,
    compute_segment_median_ratio,
    convert_sparse_depth_to_disp,
    convert_sparse_depth_to_disp_tensor,
    fit_disp_to_sparse_disp_relation_numpy,
    fit_disp_to_sparse_disp_relation_parallel,
    propagate_segment_values_iterative,
    propagate_T_values,
)
from datasets.demo import load_demo_sample
from depth_anything_v2.dpt import DepthAnythingV2

from core.superpixels import compute_superpixels


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536, 1536]},
}


# ---------------------------------------------------------------------------
# Inference pipeline (shared with run_kitti.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_pipeline(
    rgb:          np.ndarray,
    sparse_depth: np.ndarray,
    cfg_alg:      dict,
    model,
    rbf:          RecursiveBilateralFilter,
    device:       str,
    input_size:   int = 518,
) -> np.ndarray:
    """Full depth-completion pipeline (shared logic with run_kitti.py).

    When *use_segment_propagation* is True (default) the pipeline uses
    per-segment affine parameters (a, b) instead of a single T scale.
    They are propagated iteratively through a segment graph and then
    smoothed by the bilateral filter operating on the a/b maps directly.
    The adjusted disparity is  d_adj = a_filled * d_rel + b_filled.

    When *use_segment_propagation* is False a global median-ratio fallback
    is used (cheap, less accurate).
    """
    MIN_D        = cfg_alg['min_depth']
    MAX_D        = cfg_alg['max_depth']
    seg_c        = cfg_alg['segmentation']
    seg_backend  = cfg_alg.get('segment_backend', 'torch')
    use_prop     = cfg_alg.get('use_segment_propagation', True)
    use_dadp     = cfg_alg.get('use_dadp', False)
    dadp_variant = cfg_alg.get('dadp_variant', 'torch_taylor')
    min_cnt      = cfg_alg.get('segment_fitting', {}).get('min_count', 5)

    # ------------------------------------------------------------------
    # 1. Monocular relative-depth
    # ------------------------------------------------------------------
    depth_rel = model.infer_image(rgb, input_size=input_size)
    depth_rel = (depth_rel - depth_rel.min()) / (depth_rel.max() - depth_rel.min() + 1e-8)

    H, W = rgb.shape[:2]
    if depth_rel.shape != (H, W):
        depth_rel = cv2.resize(depth_rel, (W, H), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # 2. Segmentation
    # ------------------------------------------------------------------
    seg_np = compute_superpixels(rgb, seg_c)

    # ------------------------------------------------------------------
    # 3. Per-segment affine fit  [switch: segment_backend]
    #    After this block we always have (as tensors):
    #      depth_t, sparse_t, seg_t, valid_mask,
    #      seg_ids (fitted IDs), a / b (fitted slopes / intercepts),
    #      est_depth (used by the fallback path)
    # ------------------------------------------------------------------
    if seg_backend == 'numpy':
        sparse_disp_np = convert_sparse_depth_to_disp(sparse_depth, MIN_D)
        params         = fit_disp_to_sparse_disp_relation_numpy(
            depth_rel, sparse_disp_np, seg_np, min_count=min_cnt
        )
        est_disp_np  = np.clip(apply_disp_affine_numpy(depth_rel, seg_np, params), 0, 1)
        est_depth_np = np.where(est_disp_np > 0, MIN_D / (est_disp_np + 1e-6), 0.0).clip(0, MAX_D)

        est_depth  = torch.from_numpy(est_depth_np).float().to(device)
        depth_t    = torch.from_numpy(depth_rel).float().to(device)
        sparse_t   = torch.from_numpy(sparse_disp_np).float().to(device)
        seg_t      = torch.from_numpy(seg_np).long().to(device)
        valid_mask = torch.zeros(int(seg_np.max()) + 1, dtype=torch.bool, device=device)
        for sid in params:
            valid_mask[sid] = True
        # Convert params dict to tensors for a/b propagation
        if params:
            _sids  = sorted(params.keys())
            seg_ids = torch.tensor(_sids, dtype=torch.long, device=device)
            a = torch.tensor([params[s][0] for s in _sids], dtype=torch.float32, device=device)
            b = torch.tensor([params[s][1] for s in _sids], dtype=torch.float32, device=device)
        else:
            seg_ids = torch.empty(0, dtype=torch.long, device=device)
            a = torch.empty(0, dtype=torch.float32, device=device)
            b = torch.empty(0, dtype=torch.float32, device=device)
    else:
        depth_t   = torch.from_numpy(depth_rel).float().to(device)
        sparse_t  = convert_sparse_depth_to_disp_tensor(sparse_depth, MIN_D).to(device)
        seg_t     = torch.from_numpy(seg_np).long().to(device)
        seg_ids, a, b, valid_mask = fit_disp_to_sparse_disp_relation_parallel(
            depth_t, sparse_t, seg_t, min_count=min_cnt
        )
        est_disp  = apply_disp_affine_parallel(depth_t, seg_t, seg_ids, a, b).clamp(0, 1)
        est_depth = torch.where(
            est_disp > 0,
            torch.tensor(MIN_D, device=device) / (est_disp + 1e-6),
            torch.zeros_like(est_disp),
        ).clamp(0, MAX_D)

    # Pre-compute RGB tensor (reused in both propagation and depth-level RBF)
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 4. [Switch] a/b propagation  [use_segment_propagation]
    #    When enabled: propagate per-segment affine (a, b) through a
    #    centroid-based kNN segment graph, then smooth via the bilateral
    #    filter operating on the a/b maps.  The adjusted disparity is
    #      d_adj = a_filled * depth_rel + b_filled
    #    This replaces both the old T-propagation and the depth-level RBF.
    # ------------------------------------------------------------------
    if use_prop:
        prop_k = cfg_alg.get('propagation', {}).get('k_neighbors', 4)
        ab_iters = cfg_alg.get('propagation', {}).get('ab_iters', 8)
        n_seg    = int(seg_t.max().item()) + 1

        # Initialise a_vals with median of fitted slopes (sensible default)
        a_default = torch.median(a).item() if a.numel() > 0 else 1.0
        a_vals = torch.full((n_seg,), a_default, device=device, dtype=torch.float32)
        b_vals = torch.zeros(n_seg, device=device, dtype=torch.float32)
        if seg_ids.numel() > 0:
            a_vals[seg_ids] = a
            b_vals[seg_ids] = b

        # Build centroid-based segment graph
        graph_idx, graph_w = build_segment_graph(seg_t, valid_mask, k_neighbors=prop_k)

        # Iteratively fill uncovered segments from neighbours
        a_vals, a_vmask = propagate_segment_values_iterative(
            a_vals, valid_mask, graph_idx, graph_w,
            vmin=-10.0, vmax=10.0, max_iters=ab_iters,
        )
        b_vals, b_vmask = propagate_segment_values_iterative(
            b_vals, valid_mask, graph_idx, graph_w,
            vmin=-10.0, vmax=10.0, max_iters=ab_iters,
        )

        # Pixel-level a/b maps
        a_map = a_vals[seg_t]
        b_map = b_vals[seg_t]

        # Build masks: source = segments that received a propagated value;
        # anchor = segments that originally had sparse LiDAR coverage
        seg_source_mask = a_vmask & b_vmask
        source_mask_px  = seg_source_mask[seg_t].unsqueeze(0).unsqueeze(0)
        anchor_mask_px  = valid_mask[seg_t].unsqueeze(0).unsqueeze(0)

        a_seed = torch.where(seg_source_mask[seg_t], a_map, torch.zeros_like(a_map))
        b_seed = torch.where(seg_source_mask[seg_t], b_map, torch.zeros_like(b_map))

        use_anchor = cfg_alg.get('bilateral_filter', {}).get('use_anchor', True)
        rbf_anchor_kw = (
            dict(anchor_mask=anchor_mask_px, keep_anchor_fixed=True)
            if use_anchor else dict(anchor_mask=None, keep_anchor_fixed=False)
        )

        # RBF smoothes the a/b maps (not the depth map)
        a_filled = rbf(
            a_seed.unsqueeze(0).unsqueeze(0), rgb_t,
            source_mask=source_mask_px, **rbf_anchor_kw,
        ).squeeze()
        b_filled = rbf(
            b_seed.unsqueeze(0).unsqueeze(0), rgb_t,
            source_mask=source_mask_px, **rbf_anchor_kw,
        ).squeeze()

        adjusted_disp = a_filled * depth_t + b_filled
        merged = torch.where(
            adjusted_disp > 0,
            torch.tensor(MIN_D, device=device) / (adjusted_disp + 1e-6),
            torch.zeros_like(adjusted_disp),
        ).clamp(MIN_D, MAX_D)

        # No depth-level RBF in the a/b path; merged is already smooth
        rbf_out = merged

    else:
        # ------------------------------------------------------------------
        # Fallback: global median-ratio scale
        # ------------------------------------------------------------------
        valid_sd = sparse_t[sparse_t > 0]
        valid_md = depth_t.reshape(-1)[sparse_t.reshape(-1) > 0]
        Tdis_med = (torch.median(valid_sd / (valid_md + 1e-8)).item()
                    if valid_sd.numel() > 0 else 1.0)
        fallback = (torch.tensor(MIN_D, device=device) / (Tdis_med * depth_t + 1e-10)).clamp(0, MAX_D)
        merged   = torch.where(est_depth > MIN_D, est_depth, fallback)

        # Depth-level bilateral filter (original behaviour)
        rbf_out = rbf(merged.unsqueeze(0).unsqueeze(0), rgb_t).squeeze()
        rbf_out = torch.where(rbf_out < 0.1, merged, rbf_out)

    # ------------------------------------------------------------------
    # 5. [Switch] DADP  [use_dadp / dadp_variant]
    # ------------------------------------------------------------------
    sp_t = torch.from_numpy(sparse_depth).float().to(device)

    if use_dadp:
        dadp_c      = cfg_alg.get('dadp', {})
        seed_energy = dadp_c.get('seed_energy', -10.0)
        iters       = dadp_c.get('iterations', 1)

        result = run_dadp(
            variant     = dadp_variant,
            guidance    = rbf_out.unsqueeze(0).unsqueeze(0),
            depth       = rbf_out.unsqueeze(0).unsqueeze(0),
            sparse_mask = sp_t.unsqueeze(0).unsqueeze(0),
            seed_energy = seed_energy,
            iterations  = iters,
        ).squeeze().cpu().float().numpy()
    else:
        result = rbf_out.cpu().float().numpy()

    return result.clip(MIN_D, MAX_D)


def save_depth(out_dir: str, result: np.ndarray):
    """Save metric depth as 16-bit PNG (millimetres)."""
    os.makedirs(out_dir, exist_ok=True)
    depth_mm = (result * 1000).astype(np.uint16)
    path = os.path.join(out_dir, 'depth_mm.png')
    cv2.imwrite(path, depth_mm)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo depth completion")
    parser.add_argument('--config', default='configs/demo.json',
                        help='Path to JSON config (relative to src/ or absolute)')
    parser.add_argument('--image',  default='', help='Override image path from config')
    parser.add_argument('--sparse', default='', help='Override sparse depth path from config')
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_HERE, args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    cfg_model = cfg['model']
    cfg_data  = cfg['dataset']
    cfg_alg   = cfg['algorithm']
    cfg_out   = cfg.get('output', {})

    # CLI overrides
    if args.image:
        cfg_data['image_path']  = args.image
    if args.sparse:
        cfg_data['sparse_path'] = args.sparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    encoder = cfg_model['encoder']
    model   = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt = resolve_checkpoint(cfg_model['checkpoint'])
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download weights with:  bash scripts/download_checkpoints.sh vitl"
        )
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model   = model.to(device).eval()

    # ------------------------------------------------------------------
    # Bilateral filter
    # ------------------------------------------------------------------
    bf = cfg_alg.get('bilateral_filter', {})
    rbf = RecursiveBilateralFilter(
        iterations    = bf.get('iterations',    1),
        spatial_sigma = bf.get('spatial_sigma', 0.1),
        color_sigma   = bf.get('color_sigma',   0.01),
        kernel_size   = bf.get('kernel_size',   7),
        min_depth     = cfg_alg['min_depth'],
    ).to(device)

    # ------------------------------------------------------------------
    # Load sample
    # ------------------------------------------------------------------
    rgb, sparse_depth, gt_depth = load_demo_sample(
        image_path   = resolve_from_root(cfg_data['image_path']),
        sparse_path  = resolve_from_root(cfg_data.get('sparse_path', '')) if cfg_data.get('sparse_path') else '',
        sparse_format= cfg_data.get('sparse_format', 'mm_uint16'),
        gt_path      = resolve_from_root(cfg_data.get('gt_path', '')) if cfg_data.get('gt_path') else '',
    )

    print(f"Image: {rgb.shape}")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    result = run_pipeline(
        rgb, sparse_depth, cfg_alg, model, rbf, device,
        input_size=cfg_model.get('input_size', 518),
    )

    # ------------------------------------------------------------------
    # Evaluate (optional)
    # ------------------------------------------------------------------
    if gt_depth is not None and gt_depth.max() > 0:
        err = calculate_depth_error(
            result, gt_depth,
            min_depth = cfg_alg['min_depth'],
            max_depth = cfg_alg['max_depth'],
        )
        print(f"\n=== Evaluation ===  MAE: {err['mae']:.6f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = resolve_from_root(cfg_out.get('output_dir', 'output_demo'))
    save_depth(out_dir, result)


if __name__ == '__main__':
    main()

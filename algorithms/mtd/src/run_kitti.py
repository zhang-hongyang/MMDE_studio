"""KITTI Depth Completion evaluation script.

Usage
-----
    python run_kitti.py --config configs/kitti_dc.json

Feature switches (all in the JSON config under "algorithm"):

    segment_backend          "torch" (default) or "numpy"
        "torch"  — parallel GPU OLS fit (fast).
        "numpy"  — sklearn LinearRegression per segment (matches original
                   benchmark code; marginally slower but numerically identical).

    use_segment_propagation  true (default) / false
        Build a k-NN segment graph and propagate scale median ratio to segments that
        have no sparse LiDAR coverage.  When false, falls back to a global
        median-ratio estimate (cheaper, less accurate for distant segments).

    use_params_propagation   false (default) / true
        When true (and use_segment_propagation is also true), propagate the
        per-segment affine parameters (a, b) instead of a single median scale.
        The a/b maps are then smoothed by the bilateral filter, and the
        final disparity is  d_adj = a_filled * d_rel + b_filled.
        This path skips the depth-level bilateral filter step.
        When false (default) the original T-propagation + depth-level RBF
        pipeline is used, preserving full backward compatibility.

    use_dadp                 false (default) / true
        Apply Discontinuity-Aware Dynamic Programming after the bilateral
        filter.  Choose the variant with dadp_variant.

    dadp_variant             "torch_taylor" (default) / "torch_bspline" / "numpy_dadp"
        torch_taylor  — Taylor-expansion candidate interpolation, stable at 1 iteration.
        torch_bspline — Hermite/B-spline interpolation, best with 3–5 iterations.
        numpy_dadp     — CPU numpy single-pass (slowest, weakest).
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
from core.superpixels import compute_superpixels
from core.dadp import run_dadp
from core.metrics import calculate_depth_error, print_metrics
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
from datasets.kitti_dc import KITTIDCDataset
from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536, 1536]},
}


def load_model(cfg_model: dict, device: str):
    encoder = cfg_model['encoder']
    model   = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    ckpt = resolve_checkpoint(cfg_model['checkpoint'])
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download weights with:  bash scripts/download_checkpoints.sh vitl"
        )
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Per-image inference pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_one(
    rgb:          np.ndarray,
    sparse_depth: np.ndarray,
    cfg_alg:      dict,
    model,
    rbf:          RecursiveBilateralFilter,
    device:       str,
    input_size:   int = 518,
) -> np.ndarray:
    """Run the full depth-completion pipeline on a single frame.

    Reads all behaviour switches from *cfg_alg*:
        segment_backend          "torch" | "numpy"
        use_segment_propagation  true | false
        use_params_propagation   true | false  (only active when use_segment_propagation=true)
        use_dadp                 true | false
        dadp_variant             "torch_taylor" | "torch_bspline" | "numpy_dadp"
    """
    MIN_D          = cfg_alg['min_depth']
    MAX_D          = cfg_alg['max_depth']
    seg_c          = cfg_alg['segmentation']
    seg_backend    = cfg_alg.get('segment_backend', 'torch')
    use_prop       = cfg_alg.get('use_segment_propagation', True)
    use_ab_prop    = cfg_alg.get('use_params_propagation', False)
    use_dadp       = cfg_alg.get('use_dadp', False)
    dadp_variant   = cfg_alg.get('dadp_variant', 'torch_taylor')
    min_cnt        = cfg_alg.get('segment_fitting', {}).get('min_count', 3)

    # ------------------------------------------------------------------
    # 1. Monocular relative-depth inference
    # ------------------------------------------------------------------
    depth_rel = model.infer_image(rgb, input_size=input_size)
    depth_rel = (depth_rel - depth_rel.min()) / (depth_rel.max() - depth_rel.min() + 1e-8)

    # ------------------------------------------------------------------
    # 2. Superpixel segmentation  [switch: segmentation.method]
    # ------------------------------------------------------------------
    segment_np = compute_superpixels(rgb, seg_c)

    # ------------------------------------------------------------------
    # 3. Per-segment affine fitting  [switch: segment_backend]
    #    After this block we always have (as tensors):
    #      depth_t, sparse_t, seg_t, valid_mask,
    #      seg_ids (fitted segment IDs), a / b (slopes / intercepts),
    #      est_depth (used by the T-propagation fallback path)
    # ------------------------------------------------------------------
    if seg_backend == 'numpy':
        sparse_disp_np = convert_sparse_depth_to_disp(sparse_depth, MIN_D)
        params         = fit_disp_to_sparse_disp_relation_numpy(
            depth_rel, sparse_disp_np, segment_np, min_count=min_cnt
        )
        est_disp_np    = apply_disp_affine_numpy(depth_rel, segment_np, params)
        est_disp_np    = np.clip(est_disp_np, 0, 1)
        est_depth_np   = np.where(est_disp_np > 0, MIN_D / (est_disp_np + 1e-6), 0.0)
        est_depth_np   = np.clip(est_depth_np, 0, MAX_D)

        est_depth = torch.from_numpy(est_depth_np).float().to(device)
        depth_t   = torch.from_numpy(depth_rel).float().to(device)
        sparse_t  = torch.from_numpy(convert_sparse_depth_to_disp(sparse_depth, MIN_D)).float().to(device)
        seg_t     = torch.from_numpy(segment_np).long().to(device)
        valid_mask = torch.zeros(int(segment_np.max()) + 1, dtype=torch.bool, device=device)
        for sid in params:
            valid_mask[sid] = True
        # Convert params dict to tensors for a/b propagation
        if params:
            _sids   = sorted(params.keys())
            seg_ids = torch.tensor(_sids, dtype=torch.long, device=device)
            a = torch.tensor([params[s][0] for s in _sids], dtype=torch.float32, device=device)
            b = torch.tensor([params[s][1] for s in _sids], dtype=torch.float32, device=device)
        else:
            seg_ids = torch.empty(0, dtype=torch.long, device=device)
            a = torch.empty(0, dtype=torch.float32, device=device)
            b = torch.empty(0, dtype=torch.float32, device=device)

    else:  # 'torch'
        depth_t  = torch.from_numpy(depth_rel).float().to(device)
        sparse_t = convert_sparse_depth_to_disp_tensor(sparse_depth, MIN_D).to(device)
        seg_t    = torch.from_numpy(segment_np).long().to(device)

        seg_ids, a, b, valid_mask = fit_disp_to_sparse_disp_relation_parallel(
            depth_t, sparse_t, seg_t, min_count=min_cnt
        )
        est_disp  = apply_disp_affine_parallel(depth_t, seg_t, seg_ids, a, b).clamp(0, 1)
        est_depth = torch.where(
            est_disp > 0,
            torch.tensor(MIN_D, device=device) / (est_disp + 1e-6),
            torch.zeros_like(est_disp),
        ).clamp(0, MAX_D)

    # Pre-compute RGB tensor (reused across branches)
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 4. [Switch] Propagation  [use_segment_propagation / use_params_propagation]
    # ------------------------------------------------------------------
    if use_prop and use_ab_prop:
        # ---- a/b propagation path (new) --------------------------------
        prop_k   = cfg_alg.get('propagation', {}).get('k_neighbors', 4)
        ab_iters = cfg_alg.get('propagation', {}).get('ab_iters', 8)
        n_seg    = int(seg_t.max().item()) + 1

        a_default = torch.median(a).item() if a.numel() > 0 else 1.0
        a_vals = torch.full((n_seg,), a_default, device=device, dtype=torch.float32)
        b_vals = torch.zeros(n_seg, device=device, dtype=torch.float32)
        if seg_ids.numel() > 0:
            a_vals[seg_ids] = a
            b_vals[seg_ids] = b

        # Build centroid-based segment graph
        graph_idx, graph_w = build_segment_graph(seg_t, valid_mask, k_neighbors=prop_k)

        # Iteratively propagate a and b through the graph
        a_vals, a_vmask = propagate_segment_values_iterative(
            a_vals, valid_mask, graph_idx, graph_w,
            vmin=-10.0, vmax=10.0, max_iters=ab_iters,
        )
        b_vals, b_vmask = propagate_segment_values_iterative(
            b_vals, valid_mask, graph_idx, graph_w,
            vmin=-10.0, vmax=10.0, max_iters=ab_iters,
        )

        a_map = a_vals[seg_t]
        b_map = b_vals[seg_t]

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

        # RBF smooths the a/b maps
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

        # No depth-level RBF in the a/b path
        rbf_out = merged

    elif use_prop:
        # ---- original median-ratio-propagation path (default, backward compatible) ----
        prop_k = cfg_alg.get('propagation', {}).get('k_neighbors', 4)
        T_vals, valid_segs = compute_segment_median_ratio(
            depth_t, sparse_t, seg_t, valid_mask
        )
        graph_idx, graph_w = build_segment_graph(seg_t, valid_segs, k_neighbors=prop_k)
        T_vals    = propagate_T_values(T_vals, valid_segs, graph_idx, graph_w)
        T_map     = T_vals[seg_t]
        adj_disp  = depth_t * T_map
        adj_depth = torch.where(
            adj_disp > 0,
            torch.tensor(MIN_D, device=device) / (adj_disp + 1e-6),
            torch.zeros_like(adj_disp),
        ).clamp(MIN_D, MAX_D)
        merged = torch.where(est_depth > MIN_D, est_depth, adj_depth)

        # Depth-level bilateral filter (original behaviour)
        rbf_out = rbf(merged.unsqueeze(0).unsqueeze(0), rgb_t).squeeze()
        rbf_out = torch.where(rbf_out < 0.1, merged, rbf_out)

    else:
        # ---- global median-ratio fallback (matches original benchmark code) ----
        valid_sd = sparse_t[sparse_t > 0]
        valid_md = depth_t.reshape(-1)[sparse_t.reshape(-1) > 0]
        Tdis_med = (torch.median(valid_sd / (valid_md + 1e-8)).item()
                    if valid_sd.numel() > 0 else 1.0)
        fallback = (torch.tensor(MIN_D, device=device) / (Tdis_med * depth_t + 1e-10)).clamp(0, MAX_D)
        merged   = torch.where(est_depth > MIN_D, est_depth, fallback)

        rbf_out = rbf(merged.unsqueeze(0).unsqueeze(0), rgb_t).squeeze()
        rbf_out = torch.where(rbf_out < 0.1, merged, rbf_out)


    # ------------------------------------------------------------------
    # 5. Combine with sparse seeds  (seeds are authoritative)
    # ------------------------------------------------------------------
    sp_t = torch.from_numpy(sparse_depth).float().to(device)
    T2   = torch.where(sp_t > 0, sp_t, rbf_out)

    # ------------------------------------------------------------------
    # 6. [Switch] DADP refinement  [use_dadp / dadp_variant]
    # ------------------------------------------------------------------
    if use_dadp:
        dadp_cfg    = cfg_alg.get('dadp', {})
        seed_energy = dadp_cfg.get('seed_energy', -10.0)
        iters       = dadp_cfg.get('iterations', 1)

        result = run_dadp(
            variant     = dadp_variant,
            guidance    = rbf_out.unsqueeze(0).unsqueeze(0),
            depth       = T2.unsqueeze(0).unsqueeze(0),
            sparse_mask = sp_t.unsqueeze(0).unsqueeze(0),
            seed_energy = seed_energy,
            iterations  = iters,
        ).squeeze().cpu().float().numpy()
    else:
        result = T2.cpu().float().numpy()

    return np.clip(result, MIN_D, MAX_D)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KITTI DC evaluation")
    parser.add_argument('--config', default='configs/kitti_dc.json',
                        help='Path to JSON config file')
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_HERE, args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    cfg_model  = cfg['model']
    cfg_data   = cfg['dataset']
    cfg_alg    = cfg['algorithm']
    cfg_out    = cfg.get('output', {})

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Build model and filter
    # ------------------------------------------------------------------
    model = load_model(cfg_model, device)

    bf_cfg = cfg_alg.get('bilateral_filter', {})
    rbf    = RecursiveBilateralFilter(
        iterations    = bf_cfg.get('iterations',    1),
        spatial_sigma = bf_cfg.get('spatial_sigma', 0.05),
        color_sigma   = bf_cfg.get('color_sigma',   0.001),
        kernel_size   = bf_cfg.get('kernel_size',   7),
        min_depth     = cfg_alg['min_depth'],
    ).to(device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = KITTIDCDataset(
        data_path       = resolve_from_root(cfg_data['data_path']),
        remove_outliers = cfg_data.get('remove_outliers', True),
        do_flip         = cfg_data.get('do_flip', False),
    )
    print(f"Found {len(dataset)} images")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    errors       = []
    print_every  = cfg_out.get('print_every', 100)

    for k in range(len(dataset)):
        rgb, sparse_depth, gt_depth, fname = dataset[k]

        if gt_depth.shape == (60, 60):
            # Corrupted GT — skip
            continue

        print(f"[{k+1:4d}/{len(dataset)}] {fname}")

        result = process_one(
            rgb, sparse_depth, cfg_alg, model, rbf, device,
            input_size=cfg_model.get('input_size', 518),
        )

        if gt_depth is not None and gt_depth.max() > 0:
            err = calculate_depth_error(
                result, gt_depth,
                min_depth = cfg_alg['min_depth'],
                max_depth = cfg_alg['max_depth'],
            )
            errors.append(err)

        if (k + 1) % print_every == 0 and errors:
            print_metrics(errors, prefix=f"@{k+1}")

    # ------------------------------------------------------------------
    # Final results
    # ------------------------------------------------------------------
    if errors:
        print("\n===== Final Results =====")
        print_metrics(errors, prefix="mean")
    else:
        print("No valid GT frames found — skipping metric summary.")


if __name__ == '__main__':
    main()

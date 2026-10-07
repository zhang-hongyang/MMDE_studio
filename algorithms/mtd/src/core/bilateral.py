"""Recursive Bilateral Filter for sparse-to-dense depth completion.

The filter propagates depth values from sparse LiDAR points using joint
bilateral weights that respect colour edges in the RGB guidance image.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveBilateralFilter(nn.Module):
    """GPU-accelerated recursive bilateral filter.

    At each iteration the depth at every pixel is updated to the
    colour-guided, spatially-weighted average of its k×k neighbourhood,
    using only pixels that already hold a valid depth value (> min_depth).
    Sparse seed values from LiDAR are preserved after filtering via a
    where() guard.

    Args:
        iterations:     Number of diffusion iterations.
        spatial_sigma:  Relative spatial bandwidth  (fraction of kernel area).
        color_sigma:    Colour similarity bandwidth.
        kernel_size:    Side length of the square filter kernel (odd recommended).
        min_depth:      Threshold below which a pixel is treated as invalid.
    """

    def __init__(
        self,
        iterations: int = 1,
        spatial_sigma: float = 0.1,
        color_sigma: float = 0.01,
        kernel_size: int = 7,
        min_depth: float = 1e-6,
    ):
        super().__init__()
        self.iterations    = iterations
        self.spatial_sigma = spatial_sigma
        self.color_sigma   = color_sigma
        self.kernel_size   = kernel_size
        self.min_depth     = min_depth

        k = kernel_size
        center = k // 2
        offsets = torch.tensor(
            [(i - center, j - center) for i in range(k) for j in range(k)],
            dtype=torch.float32,
        )
        self.register_buffer('kernel_offsets', offsets)

    # ------------------------------------------------------------------
    def forward(
        self,
        depth: torch.Tensor,
        image: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        growth_thresh: float = 0.0,
        keep_anchor_fixed: bool = False,
    ) -> torch.Tensor:
        """Filter *depth* guided by *image*.

        Args:
            depth: (B, 1, H, W) sparse / semi-dense depth (or any scalar map).
            image: (B, 3, H, W) uint8-range RGB  [0, 255].
            source_mask: Optional (B, 1, H, W) bool tensor.  When provided,
                only pixels where source_mask is True act as diffusion sources
                (overrides the default ``filtered > min_depth`` heuristic).
            anchor_mask: Optional (B, 1, H, W) bool tensor.  Anchor pixels are
                always included as sources.  When *keep_anchor_fixed* is True
                their original *depth* values are preserved throughout.
            growth_thresh: Minimum weight-sum required before a pixel is
                admitted into the valid set for the next iteration.
            keep_anchor_fixed: If True, anchor pixels are reset to their
                original *depth* values after every diffusion step.

        Returns:
            Filtered map (B, 1, H, W).
        """
        B, _, H, W = depth.shape
        device     = depth.device
        k          = self.kernel_size
        n_px       = k * k

        # Convert RGB to luminance in [0, 1] for colour weighting
        gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2])
        gray = gray.unsqueeze(1) / 255.0
        gray = F.interpolate(gray, size=(H, W), mode='bilinear', align_corners=False)

        # Spatial weights — computed once, reused across iterations
        dx = self.kernel_offsets[:, 0].view(1, -1, 1, 1)
        dy = self.kernel_offsets[:, 1].view(1, -1, 1, 1)
        spatial_w = torch.exp(-(dx ** 2 + dy ** 2) / (k ** 2 * self.spatial_sigma))

        # Pre-compute colour patches (image does not change)
        gray_patches = F.unfold(gray, kernel_size=k, padding=k // 2).view(B, 1, n_px, H, W)
        color_dist   = (gray_patches - gray.unsqueeze(2)) ** 2 / self.color_sigma
        color_w      = torch.exp(-color_dist)

        filtered = depth.clone()

        if source_mask is None and anchor_mask is None:
            # ---- original behaviour: threshold-based valid mask ----
            for _ in range(self.iterations):
                valid_mask    = (filtered > self.min_depth).float()
                depth_patches = F.unfold(filtered,   kernel_size=k, padding=k // 2).view(B, 1, n_px, H, W)
                mask_patches  = F.unfold(valid_mask, kernel_size=k, padding=k // 2).view(B, 1, n_px, H, W)

                weights    = spatial_w.view(1, 1, n_px, 1, 1) * color_w * mask_patches
                weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-30)

                new_depth  = (weights * depth_patches).sum(dim=2, keepdim=True) / weight_sum
                filtered   = new_depth.squeeze(2)
        else:
            # ---- mask-guided diffusion (for propagating a/b maps etc.) ----
            current_mask = (source_mask.bool() if source_mask is not None
                            else (filtered > self.min_depth))
            if anchor_mask is not None:
                current_mask = current_mask | anchor_mask.bool()

            for _ in range(self.iterations):
                src_mask_f    = current_mask.float()
                depth_patches = F.unfold(filtered,   kernel_size=k, padding=k // 2).view(B, 1, n_px, H, W)
                mask_patches  = F.unfold(src_mask_f, kernel_size=k, padding=k // 2).view(B, 1, n_px, H, W)

                weights    = spatial_w.view(1, 1, n_px, 1, 1) * color_w * mask_patches
                weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-30)

                new_depth  = (weights * depth_patches).sum(dim=2, keepdim=True) / weight_sum
                new_depth  = new_depth.squeeze(2)

                newly_supported = (~current_mask) & (weight_sum.squeeze(2) > growth_thresh)

                if anchor_mask is not None and keep_anchor_fixed:
                    filtered = torch.where(current_mask, filtered, new_depth)
                    filtered = torch.where(anchor_mask.bool(), depth, filtered)
                else:
                    filtered = new_depth

                current_mask = current_mask | newly_supported

        return filtered

"""Discontinuity-Aware Dynamic Programming (DADP).

Two variants are provided:

    torch_dadp  — Taylor-expansion candidate interpolation, GPU (default)
    numpy_dadp  — Single-pass CPU numpy implementation
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared convolution kernels
# ---------------------------------------------------------------------------

_KER_XX = torch.tensor([[0, 0, 0],
                         [1, -2, 1],
                         [0, 0, 0]], dtype=torch.float32)

_KER_YY = torch.tensor([[0,  1, 0],
                         [0, -2, 0],
                         [0,  1, 0]], dtype=torch.float32)

_GX = torch.tensor([[0,    0,   0],
                     [-0.5, 0, 0.5],
                     [0,    0,   0]], dtype=torch.float32)

_GY = torch.tensor([[0, -0.5, 0],
                     [0,  0,  0],
                     [0,  0.5, 0]], dtype=torch.float32)

_LAP_ALPHA = torch.tensor([[0, -1,  0],
                             [-1, 4, -1],
                             [0, -1,  0]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    k = kernel.to(x.device, x.dtype).expand(C, 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=C)


def _build_offsets(device, dtype, tau: float = 0.3):
    dx = -tau * torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1],
                               device=device, dtype=dtype).view(1, 9, 1)
    dy = -tau * torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1],
                               device=device, dtype=dtype).view(1, 9, 1)
    return dx, dy


# ---------------------------------------------------------------------------
# torch_dadp  (GPU)
# ---------------------------------------------------------------------------

def DADP_optim(
    guidance: torch.Tensor,
    depth: torch.Tensor,
    sparse_mask: torch.Tensor,
    *,
    seed_energy: float = -10.0,
    iterations: int = 1,
) -> torch.Tensor:
    """DADP with second-order Taylor-expansion candidate interpolation (GPU).

    Args:
        guidance:    (B, 1, H, W) smooth depth used to detect discontinuities.
        depth:       (B, 1, H, W) depth to optimise.
        sparse_mask: (B, 1, H, W) LiDAR depth; > 0 marks anchor pixels.
        seed_energy: Large negative value seeded at LiDAR pixels.
        iterations:  Number of DP passes.

    Returns:
        Optimised depth (B, 1, H, W).
    """
    B, C, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    dx, dy = _build_offsets(device, dtype)

    Energy = None
    for i in range(1, iterations + 1):
        delta_E = _conv(guidance, _KER_XX).abs() + _conv(guidance, _KER_YY).abs()

        if Energy is None:
            Energy = delta_E.clone()
            Energy = torch.where(
                sparse_mask > 0,
                torch.as_tensor(seed_energy, device=device, dtype=dtype),
                Energy,
            )

        dE_unf = F.unfold(delta_E, kernel_size=3, padding=1)
        E_unf  = F.unfold(Energy,  kernel_size=3, padding=1)

        E_corr = E_unf + dE_unf
        E_corr[:, 4, :] = E_unf[:, 4, :] + seed_energy / 2.0
        E_unf = E_corr

        TGu  = _conv(depth, _GX)
        TGv  = _conv(depth, _GY)
        f_xy = _conv(depth, _LAP_ALPHA / 4.0)

        n_unf   = F.unfold(depth, kernel_size=3, padding=1)
        TGu_unf = F.unfold(TGu,   kernel_size=3, padding=1)
        TGv_unf = F.unfold(TGv,   kernel_size=3, padding=1)
        fxy_unf = F.unfold(f_xy,  kernel_size=3, padding=1)

        candidates = (n_unf
                      + dx * TGu_unf
                      + dy * TGv_unf
                      + (dx ** 2 + dy ** 2) * fxy_unf)

        idx      = E_unf.argmin(dim=1)
        picked   = torch.gather(candidates, 1, idx.unsqueeze(1))
        picked_E = torch.gather(E_unf,      1, idx.unsqueeze(1))

        w      = 1.0 / (i + 1)
        depth  = picked.view(B, 1, H, W) * w + depth * (1.0 - w)
        Energy = picked_E.view(B, 1, H, W)

    return depth


# ---------------------------------------------------------------------------
# numpy_dadp  (CPU)
# ---------------------------------------------------------------------------

def DADP_optim_numpy(
    guidance: np.ndarray,
    depth: np.ndarray,
    sparse_mask: np.ndarray,
) -> np.ndarray:
    """DADP single-pass implementation on CPU (numpy).

    Args:
        guidance:    (H, W) smooth depth used to compute Laplacian energy.
        depth:       (H, W) initial depth to update.
        sparse_mask: (H, W) LiDAR depth; > 0 = valid anchor.

    Returns:
        Updated depth map (H, W).
    """
    h, w  = guidance.shape
    lap_h = np.array([[-1, 2, -1]])
    lap_v = np.array([[-1], [2], [-1]])
    lap_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    g  = guidance.astype(np.float32)
    Zx = np.abs(cv2.filter2D(g, -1, lap_h))
    Zy = np.abs(cv2.filter2D(g, -1, lap_v))
    Z  = Zx + Zy

    Za = np.where(sparse_mask > 0, -10.0, Z)

    ih = np.inf * np.ones((h, 1))
    iv = np.inf * np.ones((1, w))
    E  = np.array([
        np.hstack((ih, Z[:, :-1]))  + np.hstack((ih, Za[:, :-1])),
        np.hstack((Z[:, 1:], ih))   + np.hstack((Za[:, 1:], ih)),
        np.vstack((iv, Z[:-1, :]))  + np.vstack((iv, Za[:-1, :])),
        np.vstack((Z[1:, :], iv))   + np.vstack((Za[1:, :], iv)),
        Za - 5.0,
    ])  # (5, H, W)

    best = np.argmin(E, axis=0)

    i  = 1
    d  = depth.astype(np.float32)
    Gx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    Gy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    TGu  = cv2.filter2D(d, -1, Gx)
    TGv  = cv2.filter2D(d, -1, Gy)
    Zlap = cv2.filter2D(d, -1, lap_k)

    V1 = np.ones((h, 1))
    H1 = np.ones((1, w))
    al = 1.0 - 1.0 / (i + 1)
    be = 1.0 / (i + 1)

    cands = np.array([
        al * d + be * (np.hstack((0*V1, d[:, :-1])) + np.hstack((0*V1, TGu[:, :-1])) + 0.5*np.hstack((0*V1, Zlap[:, :-1]))),
        al * d + be * (np.hstack((d[:, 1:], 0*V1))  - np.hstack((TGu[:, 1:], 0*V1))  + 0.5*np.hstack((Zlap[:, 1:], 0*V1))),
        al * d + be * (np.vstack((0*H1, d[:-1, :]))  + np.vstack((0*H1, TGv[:-1, :])) + 0.5*np.vstack((0*H1, Zlap[:-1, :]))),
        al * d + be * (np.vstack((d[1:, :], 0*H1))   - np.vstack((TGv[1:, :], 0*H1))  + 0.5*np.vstack((Zlap[1:, :], 0*H1))),
        d,
    ])  # (5, H, W)

    flat = best.reshape(-1)
    return cands.reshape(5, -1)[flat, np.arange(h * w)].reshape(h, w)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def run_dadp(
    variant: str,
    guidance: torch.Tensor,
    depth: torch.Tensor,
    sparse_mask: torch.Tensor,
    seed_energy: float = -10.0,
    iterations: int = 1,
) -> torch.Tensor:
    """Dispatch to the requested DADP variant.

    ``variant`` must be ``"torch_dadp"`` (GPU) or ``"numpy_dadp"`` (CPU).
    For ``"numpy_dadp"`` the tensors are automatically moved to numpy and
    back so the caller always receives a torch.Tensor.
    """
    if variant == 'torch_dadp':
        return DADP_optim(guidance, depth, sparse_mask,
                          seed_energy=seed_energy, iterations=iterations)
    elif variant == 'numpy_dadp':
        result_np = DADP_optim_numpy(
            guidance    = guidance.squeeze().cpu().float().numpy(),
            depth       = depth.squeeze().cpu().float().numpy(),
            sparse_mask = sparse_mask.squeeze().cpu().float().numpy(),
        )
        return torch.from_numpy(result_np).to(guidance.device).view_as(depth)
    else:
        raise ValueError(f"Unknown dadp_variant {variant!r}. "
                         f"Choose 'torch_dadp' or 'numpy_dadp'.")

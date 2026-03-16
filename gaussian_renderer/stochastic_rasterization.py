#
# Stochastic Gaussian Rendering - Pure PyTorch Implementation
# Based on: https://arxiv.org/html/2503.24366v1
#
# This module provides a drop-in replacement for the CUDA-based
# GaussianRasterizer using stochastic Gaussian rendering in pure PyTorch
# (no C extensions required).
#
# Algorithm:
#   For each pixel p, instead of deterministically compositing ALL Gaussians
#   in depth order (standard approach), we stochastically sample K Gaussians
#   proportional to their contribution weights alpha_i * T_i and estimate the
#   pixel colour as:
#
#     C_hat(p) = Z * mean(c_{j_1}, ..., c_{j_K}) + T_N * bg
#
#   where:
#     Z       = 1 - T_N = accumulated foreground alpha
#     T_N     = final transmittance after all Gaussians
#     j_k     ~ Categorical( alpha_i * T_i / Z )  (K i.i.d. samples)
#     T_i     = prod_{j < i} (1 - alpha_j)         (front-to-back transmittance)
#
#   E[C_hat(p)] = C(p)  (unbiased estimator of the deterministic render)
#
#   Forward + backward are handled fully by PyTorch autograd.
#   The multinomial sampling step uses a straight-through estimator:
#   gradients flow through the gathered sample colours and through Z / T_N,
#   but NOT through the sampling distribution itself.
#
# Usage:
#   from gaussian_renderer.stochastic_rasterization import (
#       GaussianRasterizationSettings,
#       StochasticGaussianRasterizer as GaussianRasterizer,
#   )
#

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sh_utils import eval_sh, eval_shfs_4d
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    build_scaling_rotation_4d,
)


# ---------------------------------------------------------------------------
# Public settings type – identical layout to diff_gaussian_rasterization so
# existing code that constructs GaussianRasterizationSettings continues to work
# without modification.
# ---------------------------------------------------------------------------

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    sh_degree_t: int
    campos: torch.Tensor
    timestamp: float
    time_duration: float
    rot_4d: bool
    gaussian_dim: int
    force_sh_3d: bool
    prefiltered: bool
    debug: bool


# ---------------------------------------------------------------------------
# Internal helpers – covariance and projection maths
# ---------------------------------------------------------------------------

def _build_cov3D(scaling: torch.Tensor, scale_modifier: float,
                 rotation: torch.Tensor) -> torch.Tensor:
    """Build [N, 6] upper-triangle 3D covariance from scaling [N,3] and
    rotation quaternion [N,4]."""
    L = build_scaling_rotation(scale_modifier * scaling, rotation)
    cov = L.transpose(1, 2) @ L  # [N, 3, 3]
    out = torch.zeros(scaling.shape[0], 6, device=scaling.device,
                      dtype=scaling.dtype)
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = cov[:, 0, 1]
    out[:, 2] = cov[:, 0, 2]
    out[:, 3] = cov[:, 1, 1]
    out[:, 4] = cov[:, 1, 2]
    out[:, 5] = cov[:, 2, 2]
    return out


def _build_cov3D_4d_conditional(
    scaling_xyzt: torch.Tensor,
    scale_modifier: float,
    rotation_l: torch.Tensor,
    rotation_r: torch.Tensor,
    ts: torch.Tensor,
    timestamp: float,
    prefilter_var: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 3-D conditional covariance of a 4-D Gaussian at *timestamp*.

    Returns
    -------
    cov3D   : [N, 6] upper-triangle covariance
    delta   : [N, 3] mean offset  mu_3D(t) - mu_3D
    margin  : [N, 1] temporal Gaussian weight  exp(-0.5 dt^2 / sigma_t)
    """
    dt = timestamp - ts  # [N, 1]

    L = build_scaling_rotation_4d(scale_modifier * scaling_xyzt,
                                  rotation_l, rotation_r)   # [N, 4, 4]
    full_cov = L @ L.transpose(1, 2)                         # [N, 4, 4]

    cov11 = full_cov[:, :3, :3]      # [N, 3, 3]
    cov12 = full_cov[:, :3, 3:4]    # [N, 3, 1]
    cov_t = full_cov[:, 3:4, 3:4]   # [N, 1, 1]  (scalar per Gaussian)

    sigma_t = cov_t[:, 0, 0:1]      # [N, 1]
    if prefilter_var > 0.0:
        sigma_t = sigma_t + prefilter_var

    marginal_t = torch.exp(-0.5 * dt * dt / sigma_t.clamp(min=1e-8))

    # Conditional covariance: Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21
    cov3D_cond = cov11 - cov12 @ cov12.transpose(1, 2) / cov_t.clamp(min=1e-8)

    delta = (cov12.squeeze(-1) / cov_t[:, 0, :]) * dt  # [N, 3]

    c = torch.zeros(scaling_xyzt.shape[0], 6, device=scaling_xyzt.device,
                    dtype=scaling_xyzt.dtype)
    c[:, 0] = cov3D_cond[:, 0, 0]
    c[:, 1] = cov3D_cond[:, 0, 1]
    c[:, 2] = cov3D_cond[:, 0, 2]
    c[:, 3] = cov3D_cond[:, 1, 1]
    c[:, 4] = cov3D_cond[:, 1, 2]
    c[:, 5] = cov3D_cond[:, 2, 2]

    return c, delta, marginal_t


def _project_gaussians(
    means3D: torch.Tensor,
    cov3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    H: int, W: int,
    tanfovx: float, tanfovy: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3-D Gaussians to 2-D screen space via EWA splatting.

    The matrix conventions follow the CUDA renderer exactly:
      * viewmatrix / projmatrix are stored *transposed* in the PyTorch tensor
        (world_view_transform = getWorld2View2(R,T).T), so the point transform
        in homogeneous coordinates is  p_out = p_in @ M  (row-vector × matrix).
      * The rotation sub-block W = viewmatrix[:3, :3]  is used for covariance
        rotation (identical to the CUDA 'W' matrix).

    Returns
    -------
    means2D : [N, 2]   screen-space pixel coordinates
    cov2D   : [N, 2, 2] 2-D covariance matrices
    depths  : [N]      camera-space depth
    valid   : [N]      frustum + radius validity mask
    radii   : [N]      int  screen-space 3-sigma radii (pixels)
    """
    N = means3D.shape[0]
    device = means3D.device
    dtype = means3D.dtype

    focal_x = W / (2.0 * tanfovx)
    focal_y = H / (2.0 * tanfovy)

    # ---- point transform ------------------------------------------------
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    p_hom = torch.cat([means3D, ones], dim=1)            # [N, 4]
    p_cam = p_hom @ viewmatrix                            # [N, 4]
    depths = p_cam[:, 2]                                  # [N]

    # ---- projection to NDC then screen coords --------------------------
    p_clip = p_hom @ projmatrix                          # [N, 4]
    inv_w = 1.0 / (p_clip[:, 3] + 1e-7)
    p_ndc = p_clip[:, :3] * inv_w.unsqueeze(1)           # [N, 3]

    # ndc2Pix(v, S) = ((v + 1) * S - 1) * 0.5
    means2D = torch.zeros(N, 2, device=device, dtype=dtype)
    means2D[:, 0] = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
    means2D[:, 1] = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5

    # ---- EWA 2-D covariance projection ---------------------------------
    # Clamp view-space tangents to avoid distortion far from image centre
    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy
    tx = torch.clamp(p_cam[:, 0] / (depths + 1e-8), -limx, limx) * depths
    ty = torch.clamp(p_cam[:, 1] / (depths + 1e-8), -limy, limy) * depths
    tz = depths

    # Jacobian J [N, 3, 3] of the perspective projection at each point
    J = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -focal_y * ty / (tz * tz)
    # Row 2 stays zero

    # W matrix = viewmatrix[:3, :3]  (rotation part, see module docstring)
    W_rot = viewmatrix[:3, :3].unsqueeze(0).expand(N, 3, 3)  # [N, 3, 3]
    T = W_rot @ J                                             # [N, 3, 3]

    # Reconstruct symmetric 3-D covariance [N, 3, 3] from 6-vector
    Vrk = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    Vrk[:, 0, 0] = cov3D[:, 0]
    Vrk[:, 0, 1] = cov3D[:, 1]; Vrk[:, 1, 0] = cov3D[:, 1]
    Vrk[:, 0, 2] = cov3D[:, 2]; Vrk[:, 2, 0] = cov3D[:, 2]
    Vrk[:, 1, 1] = cov3D[:, 3]
    Vrk[:, 1, 2] = cov3D[:, 4]; Vrk[:, 2, 1] = cov3D[:, 4]
    Vrk[:, 2, 2] = cov3D[:, 5]

    # cov2D_full = T^T @ Vrk @ T, take upper-left 2×2
    cov_full = T.transpose(1, 2) @ Vrk @ T              # [N, 3, 3]
    cov2D = cov_full[:, :2, :2].clone()
    # Low-pass filter: every Gaussian is at least 1 pixel wide/high
    cov2D[:, 0, 0] = cov2D[:, 0, 0] + 0.3
    cov2D[:, 1, 1] = cov2D[:, 1, 1] + 0.3

    # ---- radii (3 sigma of largest eigenvalue) --------------------------
    det = (cov2D[:, 0, 0] * cov2D[:, 1, 1]
           - cov2D[:, 0, 1].pow(2)).clamp(min=1e-8)
    mid = 0.5 * (cov2D[:, 0, 0] + cov2D[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    radii = torch.ceil(3.0 * torch.sqrt(lambda1)).int()  # [N]

    # ---- validity mask -------------------------------------------------
    valid = depths > 0.2
    valid = valid & (radii > 0)
    valid = valid & (det > 0)
    # Gaussian must touch the image
    r_f = radii.float()
    valid = (valid
             & (means2D[:, 0] + r_f >= 0) & (means2D[:, 0] - r_f < W)
             & (means2D[:, 1] + r_f >= 0) & (means2D[:, 1] - r_f < H))

    return means2D, cov2D, depths, valid, radii


def _evaluate_sh_colors(
    means3D: torch.Tensor,
    shs: torch.Tensor,
    campos: torch.Tensor,
    sh_degree: int,
    sh_degree_t: int,
    ts: Optional[torch.Tensor],
    timestamp: float,
    time_duration: float,
    gaussian_dim: int,
    force_sh_3d: bool,
) -> torch.Tensor:
    """Evaluate spherical harmonics and return clamped RGB colours [N, 3]."""
    shs_view = shs.transpose(1, 2).view(-1, 3, shs.shape[-1])  # [N, 3, C_sh]

    dir_pp = (means3D - campos.unsqueeze(0)).detach()
    dir_pp_norm = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp(min=1e-8)

    if gaussian_dim == 3 or force_sh_3d:
        rgb = eval_sh(sh_degree, shs_view, dir_pp_norm)
    else:
        dir_t = (ts - timestamp).detach()
        rgb = eval_shfs_4d(sh_degree, sh_degree_t, shs_view,
                           dir_pp_norm, dir_t, time_duration)

    return torch.clamp_min(rgb + 0.5, 0.0)  # [N, 3]


# ---------------------------------------------------------------------------
# Core stochastic rasterisation
# ---------------------------------------------------------------------------

def _stochastic_rasterize(
    means2D: torch.Tensor,    # [N, 2]  screen coords (used for grad flow)
    means2D_proj: torch.Tensor,  # [N, 2]  actual projected coords
    cov2D: torch.Tensor,       # [N, 2, 2]
    depths: torch.Tensor,      # [N]
    colors: torch.Tensor,      # [N, 3]
    opacities: torch.Tensor,   # [N, 1] or [N]
    flow_2d: torch.Tensor,     # [N, 2]
    radii: torch.Tensor,       # [N] int
    H: int, W: int,
    bg_color: torch.Tensor,    # [3]
    num_samples: int,
    tile_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Tile-based stochastic Gaussian rasterisation.

    When ``num_samples > 0`` the renderer is stochastic: for every pixel it
    draws ``num_samples`` Gaussians from the distribution

        p_i = alpha_i * T_i / Z

    and estimates the colour as  Z * mean(c_sampled) + T_N * bg.

    When ``num_samples <= 0`` the full deterministic compositing is used
    (equivalent to the CUDA renderer with ``num_samples = infinity``).

    The key extra arg ``means2D`` (the zero-initialised ``screenspace_points``
    from ``render()``) is added to ``means2D_proj`` so that autograd writes
    the 2-D projection gradients into ``means2D.grad`` for densification.

    Returns
    -------
    out_color   : [3, H, W]
    radii_out   : [N]        (int, 0 for invisible Gaussians)
    out_depth   : [1, H, W]
    out_alpha   : [1, H, W]
    out_flow    : [2, H, W]
    covs_com    : [N, 6]     (3-D covariance; empty tensor placeholder)
    """
    N = means2D_proj.shape[0]
    device = means2D_proj.device
    dtype = colors.dtype

    # ---- connect projection to screenspace_points for grad flow --------
    # means2D starts at 0; adding means2D_proj keeps the same numerical
    # value but makes autograd send ∂loss/∂means2D_proj → means2D.grad
    effective_means2D = means2D[:, :2] + means2D_proj

    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)   # [N]

    # Depth sort (ascending = front first)
    sort_idx = torch.argsort(depths)
    means2D_s    = effective_means2D[sort_idx]   # [N, 2]
    cov2D_s      = cov2D[sort_idx]               # [N, 2, 2]
    colors_s     = colors[sort_idx]              # [N, 3]
    opac_s       = opacities[sort_idx]           # [N]
    flow_s       = flow_2d[sort_idx]             # [N, 2]
    depths_s     = depths[sort_idx]              # [N]
    radii_s      = radii[sort_idx]               # [N]

    # Conic form (inverse covariance) for speed: [[a,b],[b,c]] = Sigma^{-1}
    det = (cov2D_s[:, 0, 0] * cov2D_s[:, 1, 1]
           - cov2D_s[:, 0, 1].pow(2)).clamp(min=1e-8)   # [N]
    inv_det = 1.0 / det
    conic_a = cov2D_s[:, 1, 1] * inv_det   # [N]
    conic_b = -cov2D_s[:, 0, 1] * inv_det  # [N]
    conic_c = cov2D_s[:, 0, 0] * inv_det   # [N]

    # Output buffers
    out_color = torch.zeros(3, H, W, device=device, dtype=dtype)
    out_depth = torch.zeros(1, H, W, device=device, dtype=dtype)
    out_flow  = torch.zeros(2, H, W, device=device, dtype=dtype)
    out_T     = torch.ones( 1, H, W, device=device, dtype=dtype)

    # Radii output (0 = invisible)
    radii_out = torch.zeros(N, dtype=torch.int32, device=device)

    n_ty = (H + tile_size - 1) // tile_size
    n_tx = (W + tile_size - 1) // tile_size

    for ty in range(n_ty):
        for tx in range(n_tx):
            y0 = ty * tile_size
            y1 = min(y0 + tile_size, H)
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, W)
            tH = y1 - y0
            tW = x1 - x0

            # ---- find Gaussians overlapping this tile ------------------
            r_f = radii_s.float()
            tile_mask = (
                (means2D_s[:, 0] + r_f >= x0)
                & (means2D_s[:, 0] - r_f < x1)
                & (means2D_s[:, 1] + r_f >= y0)
                & (means2D_s[:, 1] - r_f < y1)
            )

            if not tile_mask.any():
                # No Gaussians: fill with background
                out_color[:, y0:y1, x0:x1] = (
                    bg_color.view(3, 1, 1).expand(3, tH, tW))
                continue

            # Extract tile Gaussians (already depth-sorted)
            t_means   = means2D_s[tile_mask]   # [M, 2]
            t_colors  = colors_s[tile_mask]    # [M, 3]
            t_opac    = opac_s[tile_mask]      # [M]
            t_ca      = conic_a[tile_mask]     # [M]
            t_cb      = conic_b[tile_mask]     # [M]
            t_cc      = conic_c[tile_mask]     # [M]
            t_depths  = depths_s[tile_mask]    # [M]
            t_flow    = flow_s[tile_mask]      # [M, 2]
            t_radii   = radii_s[tile_mask]     # [M]
            M = t_means.shape[0]

            # ---- per-pixel Gaussian contributions ----------------------
            y_px = torch.arange(y0, y1, device=device, dtype=torch.float32)
            x_px = torch.arange(x0, x1, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y_px, x_px, indexing='ij')  # [tH, tW]

            # Displacement from each Gaussian mean to each pixel [tH, tW, M]
            dx = xx.unsqueeze(2) - t_means[:, 0].reshape(1, 1, M)
            dy = yy.unsqueeze(2) - t_means[:, 1].reshape(1, 1, M)

            # Mahalanobis power: -0.5*(a*dx^2 + 2b*dx*dy + c*dy^2)
            power = -0.5 * (
                t_ca.view(1, 1, M) * dx * dx
                + 2.0 * t_cb.view(1, 1, M) * dx * dy
                + t_cc.view(1, 1, M) * dy * dy
            )  # [tH, tW, M]

            # alpha = opacity * exp(power)  clamped to [0, 0.99]
            alpha = (t_opac.view(1, 1, M)
                     * torch.exp(power.clamp(max=0.0))).clamp(max=0.99)
            # Discard very small contributions (same threshold as CUDA)
            alpha = alpha * (alpha >= (1.0 / 255.0)).float()  # [tH, tW, M]

            # ---- transmittance (front-to-back) -------------------------
            # T_after[..., i] = prod_{j <= i} (1 - alpha_j)
            one_minus_alpha = (1.0 - alpha).clamp(min=1e-8)
            T_after  = torch.cumprod(one_minus_alpha, dim=-1)      # [tH, tW, M]
            T_before = torch.cat(
                [torch.ones(tH, tW, 1, device=device, dtype=dtype),
                 T_after[..., :-1]], dim=-1)                        # [tH, tW, M]
            T_final  = T_after[..., -1:]                            # [tH, tW, 1]

            # Weight for each Gaussian: w_i = alpha_i * T_before_i
            weights = alpha * T_before                              # [tH, tW, M]
            Z = weights.sum(dim=-1, keepdim=True)                   # [tH, tW, 1]

            # ---- colour estimate --------------------------------------
            if num_samples > 0 and M > 0:
                # --- stochastic path ---
                # Sample K indices from the categorical distribution p_i = w_i / Z
                Z_safe = Z.clamp(min=1e-8)
                probs = (weights / Z_safe).clamp(min=0.0)          # [tH, tW, M]
                # Renormalize for numerical safety
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                K = min(num_samples, M)
                # multinomial requires 2-D input [batch, classes]
                probs_2d = probs.reshape(-1, M)                    # [tH*tW, M]
                idx = torch.multinomial(probs_2d, K, replacement=True)  # [tH*tW, K]

                # Gather sampled colours: [tH*tW, K, 3]
                sampled = t_colors[idx]                             # [tH*tW, K, 3]
                mean_c  = sampled.mean(dim=1).reshape(tH, tW, 3)   # [tH, tW, 3]

                # C_hat = Z * mean(c_sampled) + T_N * bg
                bg = bg_color.view(1, 1, 3).to(dtype)
                tile_color = Z * mean_c + T_final * bg              # [tH, tW, 3]

            else:
                # --- deterministic path (full compositing) ---
                # C = sum_i c_i * alpha_i * T_before_i + T_N * bg
                bg = bg_color.view(1, 1, 3).to(dtype)
                tile_color = (
                    (t_colors.view(1, 1, M, 3) * weights.unsqueeze(-1)).sum(2)
                    + T_final * bg
                )  # [tH, tW, 3]

            # ---- auxiliary outputs (always deterministic) -------------
            tile_depth = (t_depths.view(1, 1, M) * weights).sum(-1, keepdim=True)
            tile_flow  = (t_flow.view(1, 1, M, 2) * weights.unsqueeze(-1)).sum(2)

            # ---- write outputs ----------------------------------------
            out_color[:, y0:y1, x0:x1] = tile_color.permute(2, 0, 1)
            out_depth[0, y0:y1, x0:x1] = tile_depth[..., 0]
            out_flow[:, y0:y1, x0:x1]  = tile_flow.permute(2, 0, 1)
            out_T[0, y0:y1, x0:x1]     = T_final[..., 0]

            # Mark Gaussians in this tile as visible
            orig_indices = sort_idx[tile_mask]
            radii_out[orig_indices] = torch.max(
                radii_out[orig_indices], t_radii)

    out_alpha = 1.0 - out_T
    covs_com  = torch.zeros(N, 6, device=device, dtype=dtype)  # placeholder

    return out_color, radii_out, out_depth, out_alpha, out_flow, covs_com


# ---------------------------------------------------------------------------
# Public API – drop-in replacement for GaussianRasterizer
# ---------------------------------------------------------------------------

class StochasticGaussianRasterizer(nn.Module):
    """Pure-PyTorch stochastic Gaussian renderer.

    Drop-in replacement for the CUDA-based ``GaussianRasterizer``.  Accepts
    exactly the same constructor argument (``raster_settings``) and the same
    ``forward()`` signature.  Additionally accepts ``num_samples`` to control
    the stochastic approximation.

    Parameters
    ----------
    raster_settings : GaussianRasterizationSettings
    num_samples : int
        Number of Gaussian samples per pixel.
        * ``num_samples > 0``  → stochastic rendering (paper algorithm)
        * ``num_samples <= 0`` → deterministic (full alpha compositing)
        Default: 4
    tile_size : int
        Tile side length in pixels.  16 works well for most scenes.
    """

    def __init__(self, raster_settings: GaussianRasterizationSettings,
                 num_samples: int = 4, tile_size: int = 16):
        super().__init__()
        self.raster_settings = raster_settings
        self.num_samples = num_samples
        self.tile_size = tile_size

    # ------------------------------------------------------------------
    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask for points inside the view frustum."""
        rs = self.raster_settings
        with torch.no_grad():
            ones = torch.ones(positions.shape[0], 1, device=positions.device,
                              dtype=positions.dtype)
            p_hom = torch.cat([positions, ones], dim=1)  # [N, 4]
            p_clip = p_hom @ rs.projmatrix               # [N, 4]
            p_cam  = p_hom @ rs.viewmatrix               # [N, 4]
            depth  = p_cam[:, 2]
            return depth > 0.2

    # ------------------------------------------------------------------
    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        flow_2d: Optional[torch.Tensor] = None,
        ts: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        scales_t: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        rotations_r: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
        prefilter_var: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render the scene stochastically.

        Accepts the same arguments as ``GaussianRasterizer.forward()``.

        Returns
        -------
        rendered_image : [3, H, W]
        radii          : [N]   int  (0 = invisible)
        depth          : [1, H, W]
        alpha          : [1, H, W]  accumulated foreground alpha
        flow           : [2, H, W]
        covs_com       : [N, 6]     (placeholder – always zeros)
        """
        rs = self.raster_settings
        H  = rs.image_height
        W  = rs.image_width
        N  = means3D.shape[0]

        if (shs is None and colors_precomp is None) or \
           (shs is not None and colors_precomp is not None):
            raise ValueError(
                "Provide exactly one of 'shs' or 'colors_precomp'.")

        if (scales is None or rotations is None) and cov3D_precomp is None:
            raise ValueError(
                "Provide exactly one of scale/rotation pair or "
                "cov3D_precomp.")

        # ------------------------------------------------------------------
        # 1.  Covariance
        # ------------------------------------------------------------------
        if cov3D_precomp is not None:
            # Caller already computed the 3-D covariance
            cov3D = cov3D_precomp if cov3D_precomp.numel() > 0 else None
            # For empty cov3D_precomp passed as torch.Tensor([]):
            if cov3D is None or cov3D.shape[0] == 0:
                cov3D = _build_cov3D(scales, rs.scale_modifier, rotations)
        else:
            if rs.rot_4d and ts is not None:
                # 4-D conditional covariance
                from scene.gaussian_model import GaussianModel  # avoid circular
                scaling_xyzt = torch.cat([scales, scales_t], dim=1)
                cov3D, delta, marginal_t = _build_cov3D_4d_conditional(
                    scaling_xyzt, rs.scale_modifier,
                    rotations, rotations_r,
                    ts, rs.timestamp, prefilter_var)
                # Apply temporal weighting to opacities and mean offset
                opacities = opacities * marginal_t
                means3D   = means3D + delta
            elif rs.gaussian_dim == 4 and ts is not None and scales_t is not None:
                # 4-D without full rotation: simple marginal temporal
                dt     = ts - rs.timestamp            # [N, 1]
                sigma  = scales_t * rs.scale_modifier
                if prefilter_var > 0.0:
                    sigma = sigma + prefilter_var
                marginal_t = torch.exp(
                    -0.5 * dt * dt / sigma.clamp(min=1e-8))
                opacities = opacities * marginal_t
                cov3D = _build_cov3D(scales, rs.scale_modifier, rotations)
            else:
                cov3D = _build_cov3D(scales, rs.scale_modifier, rotations)

        # ------------------------------------------------------------------
        # 2.  Projection
        # ------------------------------------------------------------------
        means2D_proj, cov2D, depths, valid, radii = _project_gaussians(
            means3D, cov3D,
            rs.viewmatrix, rs.projmatrix,
            H, W, rs.tanfovx, rs.tanfovy,
        )

        # ------------------------------------------------------------------
        # 3.  Colours
        # ------------------------------------------------------------------
        if colors_precomp is not None:
            colors = colors_precomp
        else:
            # Evaluate spherical harmonics
            ts_for_sh = ts if (rs.gaussian_dim == 4
                               and not rs.force_sh_3d) else None
            colors = _evaluate_sh_colors(
                means3D, shs, rs.campos,
                rs.sh_degree, rs.sh_degree_t,
                ts_for_sh, rs.timestamp, rs.time_duration,
                rs.gaussian_dim, rs.force_sh_3d,
            )

        # ------------------------------------------------------------------
        # 4.  Filter invisible Gaussians
        # ------------------------------------------------------------------
        if not valid.all():
            means3D_v     = means3D[valid]
            means2D_v     = means2D[valid]
            means2D_proj_v = means2D_proj[valid]
            cov2D_v       = cov2D[valid]
            depths_v      = depths[valid]
            colors_v      = colors[valid]
            opac_v        = opacities[valid] if opacities.shape[0] == N \
                            else opacities
            if opac_v.shape[0] == N:
                opac_v = opac_v[valid]
            radii_v       = radii[valid]

            flow_2d_use = flow_2d if flow_2d is not None else \
                torch.zeros(N, 2, device=means3D.device, dtype=means3D.dtype)
            flow_v = flow_2d_use[valid]

            # Allocate full-size radii tensor (0 = invisible)
            radii_full = torch.zeros(N, dtype=torch.int32,
                                     device=means3D.device)
        else:
            means2D_v      = means2D
            means2D_proj_v = means2D_proj
            cov2D_v        = cov2D
            depths_v       = depths
            colors_v       = colors
            opac_v         = opacities
            radii_v        = radii
            flow_2d_use = flow_2d if flow_2d is not None else \
                torch.zeros(N, 2, device=means3D.device, dtype=means3D.dtype)
            flow_v         = flow_2d_use
            radii_full     = None  # will be returned directly

        # ------------------------------------------------------------------
        # 5.  Stochastic rasterisation
        # ------------------------------------------------------------------
        if means2D_v.shape[0] == 0:
            # No visible Gaussians
            bg = rs.bg
            out_color = bg.view(3, 1, 1).expand(3, H, W).clone()
            out_depth = torch.zeros(1, H, W, device=means3D.device,
                                    dtype=means3D.dtype)
            out_alpha = torch.zeros(1, H, W, device=means3D.device,
                                    dtype=means3D.dtype)
            out_flow  = torch.zeros(2, H, W, device=means3D.device,
                                    dtype=means3D.dtype)
            covs_com  = torch.zeros(N, 6, device=means3D.device,
                                    dtype=means3D.dtype)
            radii_ret = torch.zeros(N, dtype=torch.int32,
                                    device=means3D.device)
            return out_color, radii_ret, out_depth, out_alpha, out_flow, covs_com

        out_color, radii_vis, out_depth, out_alpha, out_flow, covs_vis = \
            _stochastic_rasterize(
                means2D_v, means2D_proj_v,
                cov2D_v, depths_v,
                colors_v, opac_v, flow_v, radii_v,
                H, W, rs.bg,
                self.num_samples, self.tile_size,
            )

        # ------------------------------------------------------------------
        # 6.  Reconstruct full-size radii
        # ------------------------------------------------------------------
        if radii_full is not None:
            radii_full[valid] = radii_vis
            covs_com = torch.zeros(N, 6, device=means3D.device,
                                   dtype=means3D.dtype)
        else:
            radii_full = radii_vis
            covs_com   = covs_vis

        return out_color, radii_full, out_depth, out_alpha, out_flow, covs_com

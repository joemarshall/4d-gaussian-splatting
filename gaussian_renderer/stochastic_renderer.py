"""
Stochastic Transparency Renderer for 4D Gaussian Splatting.

Implements the stochastic transparency algorithm described in:
  "Stochastic Transparency for Gaussian Splatting"
  https://arxiv.org/abs/2503.24366

This renderer uses only PyTorch built-ins (no C++/CUDA extensions required)
and serves as a drop-in alternative to the CUDA-based rasterizer.

Algorithm overview
------------------
For each pixel p and S independent samples:
  1. Compute the alpha contribution alpha_i(p) for every Gaussian i that
     projects onto p.
  2. Sort Gaussians front-to-back by camera-space depth.
  3. Derive alpha-compositing weights  w_i = alpha_i * T_i
     where T_i = prod_{j<i}(1 - alpha_j)  (transmittance before Gaussian i).
  4. Forward pass  – sample one Gaussian per (sample, pixel) from
     Categorical(w_0, ..., w_{M-1}, bg_weight) using the Gumbel-max trick.
     Average the sampled colours over S samples.
     E[sampled colour] = standard alpha compositing result (unbiased).
  5. Backward pass – straight-through estimator: gradients are propagated
     through the closed-form expected-value formula (= standard alpha
     compositing), which equals E[stochastic output].
"""

import math
from typing import Dict, Optional, Tuple

import torch

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_scaling_rotation_4d
from utils.sh_utils import eval_sh, eval_shfs_4d

# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------


def _quat_to_rotmat(r: torch.Tensor) -> torch.Tensor:
    """Convert [N, 4] quaternions [w, x, y, z] to [N, 3, 3] rotation matrices."""
    q = r / r.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = r.shape[0]
    R = torch.zeros(N, 3, 3, device=r.device, dtype=r.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _compute_cov3d(
    scales: torch.Tensor, rotations: torch.Tensor, scale_mod: float = 1.0
) -> torch.Tensor:
    """Compute [N, 3, 3] 3-D covariance matrices from scales and quaternions."""
    R = _quat_to_rotmat(rotations)
    S = torch.diag_embed(scale_mod * scales)
    L = R @ S
    return L @ L.transpose(1, 2)


def _unpack_cov3d(packed: torch.Tensor) -> torch.Tensor:
    """Unpack [N, 6] upper-triangle representation to [N, 3, 3] matrices."""
    N = packed.shape[0]
    c = torch.zeros(N, 3, 3, device=packed.device, dtype=packed.dtype)
    c[:, 0, 0] = packed[:, 0]
    c[:, 0, 1] = packed[:, 1]
    c[:, 0, 2] = packed[:, 2]
    c[:, 1, 0] = packed[:, 1]
    c[:, 1, 1] = packed[:, 3]
    c[:, 1, 2] = packed[:, 4]
    c[:, 2, 0] = packed[:, 2]
    c[:, 2, 1] = packed[:, 4]
    c[:, 2, 2] = packed[:, 5]
    return c


def _project_gaussians(
    means3D: torch.Tensor,
    cov3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    W: int,
    H: int,
    tanfovx: float,
    tanfovy: float,
    znear: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3-D Gaussians into 2-D screen space.

    The viewmatrix convention matches the CUDA renderer: it stores the
    transposed world-to-view matrix (W_view^T), so camera-space coordinates
    are obtained via  p_cam = p_hom @ viewmatrix  (row-vector convention).

    Returns
    -------
    means2D : [N, 2]    pixel-space centres
    cov2D   : [N, 2, 2] screen-space covariance (with anti-aliasing reg.)
    depths  : [N]       camera-space z values
    valid   : [N]       bool mask — True for Gaussians in front of the camera
    """
    N = means3D.shape[0]
    device = means3D.device
    dtype = means3D.dtype

    # Camera-space position (homogeneous multiply, row-vector convention)
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    p_hom = torch.cat([means3D, ones], dim=1)   # [N, 4]
    p_cam = p_hom @ viewmatrix                  # [N, 4]
    x_cam, y_cam, depths = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    valid = depths > znear
    z = depths.clamp(min=znear)

    # Focal lengths in pixels
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    # Screen-space centres (pixel coordinates, top-left origin)
    x_s = x_cam / z * fx + W * 0.5
    y_s = y_cam / z * fy + H * 0.5
    means2D = torch.stack([x_s, y_s], dim=1)    # [N, 2]

    # Perspective-projection Jacobian  J  [N, 2, 3]
    #   J = [fx/z,   0,    -fx*x/z²]
    #       [0,    fy/z,   -fy*y/z²]
    zeros = torch.zeros_like(z)
    J = torch.stack(
        [fx / z, zeros, -fx * x_cam / (z * z),
         zeros, fy / z, -fy * y_cam / (z * z)],
        dim=1,
    ).view(N, 2, 3)

    # World-to-camera rotation  W_rot  [3, 3]
    # viewmatrix = Rt^T = [R | 0 ; t^T | 1], so viewmatrix.T = Rt = [R^T | t ; 0 | 1]
    # The upper-left 3×3 of viewmatrix.T  is  R^T  (world-to-camera rotation).
    W_rot = viewmatrix[:3, :3].T                             # [3, 3]
    W_exp = W_rot.unsqueeze(0).expand(N, -1, -1)             # [N, 3, 3]

    T = J @ W_exp                                            # [N, 2, 3]
    cov2D = T @ cov3D @ T.transpose(1, 2)                   # [N, 2, 2]

    # Low-pass filter (same anti-aliasing constant as the CUDA renderer)
    cov2D[:, 0, 0] = cov2D[:, 0, 0] + 0.3
    cov2D[:, 1, 1] = cov2D[:, 1, 1] + 0.3

    return means2D, cov2D, depths, valid


def _cov2d_inv_and_radius(
    cov2D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute [N, 2, 2] inverse covariances and [N] pixel radii.

    radius = 3 * sqrt(max eigenvalue of cov2D).
    """
    det = (
        cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]
    ).clamp(min=1e-10)
    inv = 1.0 / det
    cov2D_inv = torch.stack(
        [
            cov2D[:, 1, 1] * inv,
            -cov2D[:, 0, 1] * inv,
            -cov2D[:, 1, 0] * inv,
            cov2D[:, 0, 0] * inv,
        ],
        dim=1,
    ).view(-1, 2, 2)

    tr = cov2D[:, 0, 0] + cov2D[:, 1, 1]
    disc = ((tr * 0.5) ** 2 - det).clamp(min=0.0)
    lam_max = tr * 0.5 + disc.sqrt()
    radii = 3.0 * lam_max.clamp(min=0.1).sqrt()

    return cov2D_inv, radii


# ---------------------------------------------------------------------------
# Core stochastic transparency pass
# ---------------------------------------------------------------------------


def stochastic_transparency_forward(
    alpha_mp: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    bg_color: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stochastic Transparency rendering for one image tile.

    Forward pass
    ~~~~~~~~~~~~
    For each of S independent samples per pixel:
      * For every Gaussian i (front-to-back), independently sample
        u_i ~ Uniform(0, 1).  Gaussian i is *accepted* if u_i < alpha_i(p).
      * Select the first accepted Gaussian for this pixel in this sample.
      * Return its colour.
    Implemented efficiently via the Gumbel-max trick on alpha-compositing
    weights, which produces exact samples from the same distribution.

    Expected value:  E[sampled colour] = Σ_i  w_i * c_i + bg_weight * c_bg
    where  w_i = alpha_i * T_i  and  T_i = prod_{j<i}(1-alpha_j).
    This equals the standard alpha-compositing result (unbiased estimator).

    Backward pass
    ~~~~~~~~~~~~~
    Straight-through estimator: gradients are propagated through the
    closed-form expected-value (alpha-compositing) formula, not through the
    discrete stochastic sampling step.

    Parameters
    ----------
    alpha_mp   : [M, P]   alpha values; Gaussians sorted front-to-back; P pixels
    colors     : [M, 3]   per-Gaussian RGB colours
    depths     : [M]      camera-space depths (sorted front-to-back)
    bg_color   : [3]      background colour
    num_samples: S        stochastic samples per pixel

    Returns
    -------
    color_3P : [3, P]
    alpha_1P : [1, P]
    depth_1P : [1, P]
    """
    M, P = alpha_mp.shape
    device = alpha_mp.device
    dtype = alpha_mp.dtype
    S = num_samples

    if M == 0:
        return (
            bg_color[:, None].expand(3, P).contiguous(),
            torch.zeros(1, P, device=device, dtype=dtype),
            torch.zeros(1, P, device=device, dtype=dtype),
        )

    # ------------------------------------------------------------------
    # Backward path: closed-form expected value (= standard alpha compositing)
    # ------------------------------------------------------------------
    # Exclusive transmittance in log-space for numerical stability:
    #   T[0] = 1,  T[i] = prod_{j<i}(1 - alpha[j])
    log_om = torch.log((1.0 - alpha_mp).clamp(min=1e-10))   # [M, P]
    log_T_incl = torch.cumsum(log_om, dim=0)                 # inclusive log-cumprod
    # Shift right by one to get exclusive log-cumprod
    log_T = torch.cat(
        [torch.zeros(1, P, device=device, dtype=dtype), log_T_incl[:-1]], dim=0
    )                                                        # [M, P]
    T = torch.exp(log_T)                                     # [M, P]

    weights = alpha_mp * T                                   # [M, P]
    bg_weight = torch.exp(log_T_incl[-1:])                  # [1, P]  (= prod of all (1-a))

    # Expected colour  [P, 3]
    exp_color = (
        (weights.unsqueeze(-1) * colors.unsqueeze(1)).sum(0)
        + bg_weight.T * bg_color.unsqueeze(0)
    )
    # Expected alpha   [1, P]
    exp_alpha = 1.0 - bg_weight
    # Expected depth   [1, P]
    exp_depth = (weights * depths.unsqueeze(1)).sum(0, keepdim=True)

    # ------------------------------------------------------------------
    # Forward path: stochastic sampling via Gumbel-max trick
    # ------------------------------------------------------------------
    # Augment with a "background" option (index M) whose weight is bg_weight.
    all_weights = torch.cat([weights, bg_weight], dim=0)               # [M+1, P]
    all_colors = torch.cat([colors, bg_color.unsqueeze(0)], dim=0)    # [M+1, 3]
    all_depths = torch.cat([depths, depths.new_zeros(1)], dim=0)       # [M+1]

    with torch.no_grad():
        eps = 1e-20
        gumbel = -torch.log(
            -torch.log(
                torch.rand(S, M + 1, P, device=device, dtype=dtype).clamp(min=eps)
            ).clamp(min=eps)
        )                                                    # [S, M+1, P]
        log_w = torch.log(all_weights.unsqueeze(0).clamp(min=eps))     # [1, M+1, P]
        sel = (log_w + gumbel).argmax(dim=1)                # [S, P]  selected Gaussian index

        sel_flat = sel.reshape(-1)                          # [S*P]
        stoch_color = all_colors[sel_flat].reshape(S, P, 3).mean(0)          # [P, 3]
        stoch_depth = all_depths[sel_flat].reshape(S, P).mean(0, keepdim=True)  # [1, P]

    # ------------------------------------------------------------------
    # Straight-through estimator:
    #   output (forward)  = stochastic value
    #   gradient (backward) flows through the expected-value formula
    # ------------------------------------------------------------------
    color_3P = stoch_color.T + (exp_color.T - exp_color.T.detach())    # [3, P]
    depth_1P = stoch_depth + (exp_depth - exp_depth.detach())          # [1, P]
    alpha_1P = exp_alpha                                                # [1, P]

    return color_3P, alpha_1P, depth_1P


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------


class StochasticTransparencyRenderer:
    """
    Pure-PyTorch Gaussian renderer using Stochastic Transparency.

    Implements the forward and backward pass from:
        "Stochastic Transparency for Gaussian Splatting"
        https://arxiv.org/abs/2503.24366

    No C++/CUDA extensions are required.  Acts as a drop-in replacement for
    the original CUDA-based renderer; returns the same output dictionary.

    Parameters
    ----------
    num_samples : int
        Monte-Carlo samples per pixel S.  Higher values reduce noise at the
        cost of more computation.  Default: 8.
    tile_size : int
        Tile edge length (pixels) for memory-efficient processing.  Default: 16.
    """

    def __init__(self, num_samples: int = 8, tile_size: int = 16) -> None:
        self.num_samples = num_samples
        self.tile_size = tile_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_colors(
        self,
        pc: GaussianModel,
        viewpoint_camera,
        means3D: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate per-Gaussian RGB colours via spherical harmonics."""
        cam = viewpoint_camera.camera_center.to(pc.get_xyz.device)
        shs = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
        dir_pp = (means3D - cam.unsqueeze(0)).detach()
        dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp(min=1e-8)

        if pc.gaussian_dim == 3 or pc.force_sh_3d:
            rgb = eval_sh(pc.active_sh_degree, shs, dir_pp)
        else:
            dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
            rgb = eval_shfs_4d(
                pc.active_sh_degree,
                pc.active_sh_degree_t,
                shs,
                dir_pp,
                dir_t,
                pc.time_duration[1] - pc.time_duration[0],
            )
        return torch.clamp_min(rgb + 0.5, 0.0)

    def _prepare(
        self,
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        scaling_modifier: float,
        override_color: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Build per-Gaussian (means3D, colors, opacity, cov3D, active_mask).

        active_mask is a [N_total] bool tensor (True = keep) or None (keep all).
        """
        means3D = pc.get_xyz         # [N_total, 3]
        opacity = pc.get_opacity     # [N_total, 1]

        if pc.gaussian_dim == 4:
            if pipe.compute_cov3D_python:
                cov_packed, delta = pc.get_current_covariance_and_mean_offset(
                    scaling_modifier, viewpoint_camera.timestamp
                )
                means3D = means3D + delta
                cov3D = _unpack_cov3d(cov_packed)
                marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
                opacity = opacity * marginal_t
            else:
                s4 = torch.cat([pc.get_scaling, pc.get_scaling_t], dim=1)
                L4 = build_scaling_rotation_4d(
                    scaling_modifier * s4, pc.get_rotation, pc.get_rotation_r
                )
                Cov4 = L4 @ L4.transpose(1, 2)               # [N, 4, 4]
                Cxx = Cov4[:, :3, :3]
                Cxt = Cov4[:, :3, 3:4]
                Ctt = Cov4[:, 3:4, 3:4].clamp(min=1e-10)
                dt = viewpoint_camera.timestamp - pc.get_t    # [N, 1]
                delta = (Cxt / Ctt).squeeze(-1) * dt
                means3D = means3D + delta
                cov3D = Cxx - (Cxt @ Cxt.transpose(1, 2)) / Ctt
                sigma_t = pc.get_scaling_t
                marginal_t = torch.exp(
                    -0.5 * dt ** 2 / (sigma_t ** 2 + 1e-10)
                )
                opacity = opacity * marginal_t
            active_mask = opacity[:, 0] > 0.05
        else:
            if pipe.compute_cov3D_python:
                cov3D = _unpack_cov3d(pc.get_covariance(scaling_modifier))
            else:
                cov3D = _compute_cov3d(pc.get_scaling, pc.get_rotation, scaling_modifier)
            active_mask = None

        if override_color is not None:
            colors = override_color
        else:
            colors = self._get_colors(pc, viewpoint_camera, means3D)

        return means3D, colors, opacity, cov3D, active_mask

    # ------------------------------------------------------------------

    def _render_tiles(
        self,
        means2D: torch.Tensor,
        cov2D_inv: torch.Tensor,
        depths: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        radii: torch.Tensor,
        W: int,
        H: int,
        bg_color: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tile-based stochastic rendering.

        Gaussians must be pre-sorted front-to-back.

        Returns rendered_color [3, H, W], alpha [1, H, W], depth [1, H, W].
        The output is assembled by concatenating tile tensors, so the entire
        computation graph is differentiable end-to-end.
        """
        T = self.tile_size
        S = self.num_samples
        device = means2D.device
        dtype = means2D.dtype

        tiles_x = (W + T - 1) // T
        tiles_y = (H + T - 1) // T

        row_tensors = []
        for ty in range(tiles_y):
            py_min = ty * T
            py_max = min(py_min + T, H)
            tile_h = py_max - py_min

            col_tensors = []
            for tx in range(tiles_x):
                px_min = tx * T
                px_max = min(px_min + T, W)
                tile_w = px_max - px_min

                # Tile centre and half-diagonal (for Gaussian culling)
                cx = (px_min + px_max - 1) * 0.5
                cy = (py_min + py_max - 1) * 0.5
                hd = math.sqrt(tile_w ** 2 + tile_h ** 2) * 0.5

                dist2 = (means2D[:, 0] - cx) ** 2 + (means2D[:, 1] - cy) ** 2
                tile_mask = dist2 <= (radii + hd) ** 2

                if not tile_mask.any():
                    # Pure background tile
                    tile_rgba = torch.cat(
                        [
                            bg_color[:, None, None].expand(3, tile_h, tile_w),
                            torch.zeros(2, tile_h, tile_w, device=device, dtype=dtype),
                        ],
                        dim=0,
                    )                                          # [5, tile_h, tile_w]
                    col_tensors.append(tile_rgba)
                    continue

                t_idx = tile_mask.nonzero(as_tuple=True)[0]
                t_means = means2D[t_idx]       # [M, 2]
                t_cinv = cov2D_inv[t_idx]      # [M, 2, 2]
                t_dep = depths[t_idx]          # [M]
                t_col = colors[t_idx]          # [M, 3]
                t_opa = opacities[t_idx]       # [M, 1]

                # Pixel grid (sample at pixel centres)
                px = (
                    torch.arange(px_min, px_max, device=device, dtype=dtype) + 0.5
                )
                py = (
                    torch.arange(py_min, py_max, device=device, dtype=dtype) + 0.5
                )
                gx, gy = torch.meshgrid(px, py, indexing="xy")
                pix_x = gx.reshape(-1)    # [P]
                pix_y = gy.reshape(-1)    # [P]

                # Per-(Gaussian, pixel) alpha values
                dx = pix_x.unsqueeze(0) - t_means[:, 0].unsqueeze(1)   # [M, P]
                dy = pix_y.unsqueeze(0) - t_means[:, 1].unsqueeze(1)   # [M, P]
                c00 = t_cinv[:, 0, 0].unsqueeze(1)
                c01 = t_cinv[:, 0, 1].unsqueeze(1)
                c11 = t_cinv[:, 1, 1].unsqueeze(1)
                md2 = dx * dx * c00 + 2.0 * dx * dy * c01 + dy * dy * c11  # [M, P]
                density = torch.exp(-0.5 * md2)                             # [M, P]
                alpha_mp = (t_opa * density).clamp(max=0.99)               # [M, P]

                tile_col, tile_alp, tile_dep = stochastic_transparency_forward(
                    alpha_mp, t_col, t_dep, bg_color, S
                )
                # tile_col [3,P], tile_alp [1,P], tile_dep [1,P]
                tile_rgba = torch.cat(
                    [tile_col, tile_alp, tile_dep], dim=0
                ).view(5, tile_h, tile_w)
                col_tensors.append(tile_rgba)

            # [5, tile_h, W]
            row_tensors.append(torch.cat(col_tensors, dim=2))

        # [5, H, W]
        full = torch.cat(row_tensors, dim=1)
        return full[:3], full[3:4], full[4:5]

    # ------------------------------------------------------------------

    def render(
        self,
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        override_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render the scene using stochastic transparency.

        Drop-in replacement for ``gaussian_renderer.render()``.

        Parameters
        ----------
        viewpoint_camera :
            Camera to render from.
        pc :
            GaussianModel holding the scene Gaussians.
        pipe :
            PipelineParams.  Relevant attributes:
              ``compute_cov3D_python`` — compute 3-D covariance in Python.
              ``renderer``            — must equal ``"stochastic"`` to reach here.
              ``num_samples``         — overrides constructor value if present.
        bg_color :
            Background colour tensor [3] on GPU.
        scaling_modifier :
            Gaussian scale multiplier.
        override_color :
            Pre-computed per-Gaussian colours [N, 3]; bypasses SH evaluation.

        Returns
        -------
        dict with keys:
            ``render``            [3, H, W]  rendered RGB image
            ``viewspace_points``  [N, 3]     screen-space means (for densification)
            ``visibility_filter`` [N]        bool mask of visible Gaussians
            ``radii``             [N]        screen-space radii in pixels
            ``depth``             [1, H, W]  depth map
            ``alpha``             [1, H, W]  accumulated alpha map
            ``flow``              [2, H, W]  zeros (API compatibility)
        """
        device = pc.get_xyz.device
        N_total = pc.get_xyz.shape[0]
        H = int(viewpoint_camera.image_height)
        W = int(viewpoint_camera.image_width)

        # screenspace_points is kept for API / training-loop compatibility.
        # The "add-zero" trick connects it into the autograd graph so that
        # screen-space position gradients accumulate in screenspace_points.grad,
        # which is what the densification logic reads.
        screenspace_points = (
            torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True)
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        # ---- Prepare Gaussian parameters ----------------------------------------
        means3D, colors, opacity, cov3D, active_mask = self._prepare(
            viewpoint_camera, pc, pipe, scaling_modifier, override_color
        )

        # ---- Temporal pre-filter (4-D only) ------------------------------------
        if active_mask is not None:
            active_idx_1 = active_mask.nonzero(as_tuple=True)[0]
            means3D = means3D[active_mask]
            colors = colors[active_mask]
            opacity = opacity[active_mask]
            cov3D = cov3D[active_mask]
        else:
            active_idx_1 = torch.arange(N_total, device=device)

        # ---- Project to 2-D -----------------------------------------------------
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        viewmat = viewpoint_camera.world_view_transform.to(device)

        means2D_proj, cov2D, depths, valid = _project_gaussians(
            means3D, cov3D, viewmat, W, H, tanfovx, tanfovy
        )

        # ---- Filter behind-camera Gaussians -------------------------------------
        if not valid.all():
            active_idx = active_idx_1[valid]
            means3D = means3D[valid]
            means2D_proj = means2D_proj[valid]
            cov2D = cov2D[valid]
            depths = depths[valid]
            opacity = opacity[valid]
            colors = colors[valid]
        else:
            active_idx = active_idx_1

        N = means3D.shape[0]

        # ---- Empty scene fallback -----------------------------------------------
        if N == 0:
            bg_img = bg_color[:, None, None].expand(3, H, W).contiguous()
            return {
                "render": bg_img,
                "viewspace_points": screenspace_points,
                "visibility_filter": torch.zeros(
                    N_total, dtype=torch.bool, device=device
                ),
                "radii": torch.zeros(N_total, device=device),
                "depth": torch.zeros(1, H, W, device=device),
                "alpha": torch.zeros(1, H, W, device=device),
                "flow": torch.zeros(2, H, W, device=device),
            }

        # ---- Inject screenspace_points into the computation graph ---------------
        # Adding screenspace_points[active_idx, :2] (== 0 in the forward pass)
        # ensures that after loss.backward(), screenspace_points.grad[active_idx, :2]
        # holds the gradients of the loss w.r.t. screen-space positions, which the
        # densification logic reads via viewspace_point_tensor.grad[filter, :2].
        means2D = means2D_proj + screenspace_points[active_idx, :2]

        # ---- 2-D covariance inverse and pixel radii -----------------------------
        cov2D_inv, radii2D = _cov2d_inv_and_radius(cov2D)

        # ---- Sort by depth (front to back) --------------------------------------
        order = depths.argsort()
        means2D_s = means2D[order]
        cov2D_inv_s = cov2D_inv[order]
        depths_s = depths[order]
        colors_s = colors[order]
        opacity_s = opacity[order]
        radii2D_s = radii2D[order]

        # ---- Tile-based stochastic rendering ------------------------------------
        rendered, alpha_map, depth_map = self._render_tiles(
            means2D_s, cov2D_inv_s, depths_s, colors_s, opacity_s, radii2D_s,
            W, H, bg_color,
        )

        # ---- Visibility and radii (full N_total tensors) -----------------------
        in_image = (
            (means2D[:, 0] >= 0)
            & (means2D[:, 0] < W)
            & (means2D[:, 1] >= 0)
            & (means2D[:, 1] < H)
            & (radii2D > 0)
        )
        radii_all = torch.zeros(N_total, device=device)
        visibility_all = torch.zeros(N_total, dtype=torch.bool, device=device)
        radii_all[active_idx] = radii2D
        visibility_all[active_idx] = in_image

        return {
            "render": rendered,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_all,
            "radii": radii_all,
            "depth": depth_map,
            "alpha": alpha_map,
            "flow": torch.zeros(2, H, W, device=device),
        }

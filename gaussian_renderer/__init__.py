#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
import math
from .diff_gaussian_rasterization_fastgs import (
    GaussianRasterizationSettings as GaussianRasterizationSettingsFastGS,
    GaussianRasterizer as GaussianRasterizerFastGS,
    calculateGaussianVisibilityContribution as calculateGaussianVisibilityContributionFastGS,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d

try:
    from .diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings as GaussianRasterizationSettingsFastGS,
        GaussianRasterizer as GaussianRasterizerFastGS,
    )
    _FASTGS_AVAILABLE = True
except Exception:
    _FASTGS_AVAILABLE = False


def render_fastgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                  mult: float = 0.5, scaling_modifier: float = 1.0,
                  override_color=None, get_flag=False, metric_map=None):
    """Render using the FastGS engine (AccuTile + multi-view metric accumulation).

    For 4D Gaussians the marginal opacity at the given timestamp is folded into
    the per-Gaussian opacity before rendering so that the fast 3-D rasterizer
    produces a meaningful image.  The returned ``accum_metric_counts`` tensor
    can be used directly in :func:`~utils.fast_utils.compute_gaussian_score_fastgs`.

    Args:
        viewpoint_camera: camera object with FoVx/FoVy, world_view_transform, etc.
        pc: GaussianModel instance.
        pipe: pipeline parameters (compute_cov3D_python, convert_SHs_python, debug).
        bg_color: background colour tensor on CUDA.
        mult: AccuTile tile-size multiplier (beta in FastGS paper, default 0.5).
        scaling_modifier: uniform scale modifier.
        override_color: optional pre-computed per-Gaussian colour tensor.
        get_flag: if True, accumulate ``accum_metric_counts`` from ``metric_map``.
        metric_map: 1-D int32 tensor of shape (H*W,) with high-error pixel flags.

    Returns:
        dict with keys ``render``, ``viewspace_points``, ``visibility_filter``,
        ``radii``, and ``accum_metric_counts``.
    """
    if not _FASTGS_AVAILABLE:
        raise RuntimeError(
            "diff_gaussian_rasterization_fastgs is not available. "
            "Please ensure the FastGS CUDA extension compiled successfully."
        )

    # screenspace_points has 4 columns: (x, y, abs_x, abs_y) – the extra two
    # columns capture absolute-value gradients used by FastGS densification.
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(
            int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
            dtype=torch.int32,
            device="cuda",
        )

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    prefilter_var = -1.0
    prefilter_var = -1.0
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r
            if pc.prefilter_var > 0.0:
                prefilter_var = pc.prefilter_var



    raster_settings = GaussianRasterizationSettingsFastGS(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult=mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map,
        time_duration = pc.time_duration[1] if pc.gaussian_dim == 4 else None,
        rot_4d = pc.rot_4d if pc.gaussian_dim == 4 else None,
        gaussian_dim = pc.gaussian_dim,
        force_sh_3d = pc.force_sh_3d,
        degree_t = pc.active_sh_degree_t if pc.gaussian_dim == 4 else None,
        timestamp = viewpoint_camera.timestamp if pc.gaussian_dim == 4 else None,
        prefilter_var = prefilter_var
    )



    rasterizer = GaussianRasterizerFastGS(raster_settings=raster_settings)


    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(
                    pc.active_sh_degree, pc.active_sh_degree_t, shs_view,
                    dir_pp_normalized, dir_t,
                    pc.time_duration[1] - pc.time_duration[0],
                )
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc = pc.get_sh_features_dc
            shs = pc.get_sh_features_rest
    else:
        colors_precomp = override_color

    rendered_image, rendered_depth, radii, accum_metric_counts,out_means3D = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "accum_metric_counts": accum_metric_counts,
        "out_means3D": out_means3D,
    }


    def calculate_gaussian_visibilities(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor):
        """Calculate per-Gaussian visibility contributions.
        
        Args:
            viewpoint_camera: camera object with FoVx/FoVy, world_view_transform, etc.
            pc: GaussianModel instance.
            pipe: pipeline parameters.
            bg_color: background colour tensor on CUDA.
        
        Returns:
            Visibility contribution tensor.
        """


        means3D = pc.get_xyz
        opacities = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        raster_settings = GaussianRasterizationSettingsFastGS(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            mult=0.5,
            prefiltered=False,
            debug=pipe.debug,
            prefilter_var = prefilter_var
        )


        return calculateGaussianVisibilityContributionFastGS(
                means3D,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                raster_settings,
                opacity_cutoff=0.01,
            )



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, compute_contrib = False):
    # Use the FastGS render engine for all rendering
    return render_fastgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier, override_color=override_color)

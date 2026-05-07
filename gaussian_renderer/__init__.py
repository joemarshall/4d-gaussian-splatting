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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d

USE_FASTGS = False


if USE_FASTGS:
    from .diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings as GaussianRasterizationSettingsFastGS
    from .diff_gaussian_rasterization_fastgs import GaussianRasterizer as GaussianRasterizerFastGS
else:
    from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def _make_rasterizer(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, metric_map = None):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if USE_FASTGS:
        if metric_map is None:
            metric_map = torch.zeros(
                int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
                dtype=torch.int32,
                device=means3d.device,
            )


        raster_settings = GaussianRasterizationSettingsFastGS(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            sh_degree_t=pc.active_sh_degree_t,
            campos=viewpoint_camera.camera_center,
            timestamp=viewpoint_camera.timestamp,
            time_duration=pc.time_duration[1]-pc.time_duration[0],
            rot_4d=pc.rot_4d,
            gaussian_dim=pc.gaussian_dim,
            force_sh_3d=pc.force_sh_3d,
            prefiltered=False,
            debug=pipe.debug,
            metric_map = metric_map,
            mult = 0.5,
            get_flag = False,
            prefilter_var = -1.0


        )
        rasterizer = GaussianRasterizerFastGS(raster_settings=raster_settings)


    else:
        if metric_map is None:
            metric_map = torch.empty([], device="cuda")

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            sh_degree_t=pc.active_sh_degree_t,
            campos=viewpoint_camera.camera_center,
            timestamp=viewpoint_camera.timestamp,
            time_duration=pc.time_duration[1]-pc.time_duration[0],
            rot_4d=pc.rot_4d,
            gaussian_dim=pc.gaussian_dim,
            force_sh_3d=pc.force_sh_3d,
            prefiltered=False,
            debug=pipe.debug,
            metric_map = metric_map,

        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, tensor_gradient_2d_buffer: torch.Tensor,scaling_modifier = 1.0, override_color = None, metric_map = None,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tensor_gradient_2d_buffer.grad=None
    screenspace_points = tensor_gradient_2d_buffer

    # Set up rasterization configuration
    rasterizer = _make_rasterizer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, metric_map)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    prefilter_var = -1.0
    marginal_t = None
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

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    mask = None
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if USE_FASTGS:
        dc = shs [:,0:1]
        sh = shs[:,1:]
        print("Rasterizing with FastGS!")
        print("DC",dc.shape)
        print("SH",sh.shape)
        rendered_image, depth, radii, metric_count, out_means3D = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = sh,
            colors_precomp = colors_precomp,
#            flow_2d = flow_2d,
            opacities = opacity,
            ts = ts,
            scales = scales,
            scales_t = scales_t,
            rotations = rotations,
            rotations_r = rotations_r,
            cov3D_precomp = cov3D_precomp,
            )

    else:
        rendered_image, radii, depth, alpha, flow, covs_com,metric_count = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            flow_2d = flow_2d,
            opacities = opacity,
            ts = ts,
            scales = scales,
            scales_t = scales_t,
            rotations = rotations,
            rotations_r = rotations_r,
            cov3D_precomp = cov3D_precomp,
            prefilter_var = prefilter_var)
    
    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None], align_corners=False)[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    
    if mask is not None:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii
    # with torch.no_grad():
    #     vis_range = pc._visible_range
    #     if vis_range is not None and vis_range.shape[0] == means3D.shape[0]:
    #         time_offset = torch.abs(ts.squeeze() - viewpoint_camera.timestamp)
    #         visibles_time =  time_offset < vis_range
#            print(torch.sum(visibles_time), "visible gaussians out of", radii_all.shape[0])
    if USE_FASTGS:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii_all > 0,
                "radii": radii_all,
                "depth": depth
                }
    else:
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii_all > 0,
                "radii": radii_all,
                "depth": depth,
                "alpha": alpha,
                "flow": flow,
                "metric_counts": metric_count}

def calculate_gaussian_contribution(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, error_map: torch.Tensor = None):
    """
    Calculate the contribution of each Gaussian to the final render, for use in densification criteria. 
    
    Background tensor (bg_color) must be on GPU!

    n.b. this renders backwards to get contribution to render for each gaussian
         - this is only used for pruning so no differentiation is provided
    
    """
    if error_map is None:
        error_map = torch.empty([], device="cuda")
    # Set up rasterization configuration
    rasterizer = _make_rasterizer(viewpoint_camera, pc, pipe, bg_color, metric_map=error_map)

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    prefilter_var = -1.0
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


    if USE_FASTGS:
        print("Gaussian contribution calculation not implemented for FastGS yet.")
        render_outputs = None
    else:    
        num_pts,gaussian_contribution,weighted_contribution = rasterizer.calculate_gaussian_contributions(
            means3D=pc.get_xyz,
            ts = ts,
            scales = scales,
            scales_t = scales_t,
            rotations = rotations,
            rotations_r = rotations_r,
            cov3D_precomp = cov3D_precomp,
            opacities = opacity,
            prefilter_var = prefilter_var,
        )

    render_outputs = {"num_pts": num_pts, "visual_contribution": gaussian_contribution, "error_contribution": weighted_contribution}

    return render_outputs